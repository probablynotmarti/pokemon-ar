"""Turn-based battle engine (singles, player vs CPU).

Pure Python: every random outcome goes through the injected `rng`, so a
seeded or rigged rng makes a battle fully reproducible. The engine never
renders anything; each turn returns a list of `Event`s (game-style text +
data) for the UI/AR layer to animate in order.

Turn order (GAME_DESIGN.md §3): switches → items → moves → end of turn
(burn, Leech Seed). Moves go by priority bracket; inside a bracket who
goes first is a coin flip.
"""

import random
from dataclasses import dataclass, field

from .damage import CRIT_CHANCE, STAGE_MULT, final_damage, move_power
from .data import (ITEMS, LEVEL, MOVES, PARTY_SIZE, POKEDEX, STARTING_BAG,
                   Move, calc_stats, effectiveness)

STRUGGLE = Move("Struggle", "Typeless", "physical", 50, None, 1,
                effect="struggle_recoil")
STAT_NAMES = {"atk": "Attack", "def": "Defense", "spa": "Sp. Atk",
              "spd": "Sp. Def", "spe": "Speed"}
STATUS_TAGS = {"slp": "SLP", "par": "PAR", "frz": "FRZ", "brn": "BRN"}
# Types that can't get a status (Gen 6+).
STATUS_IMMUNE = {"par": "Electric", "frz": "Ice", "brn": "Fire"}


# ── Battle state ──────────────────────────────────────────────────────
@dataclass
class Mon:
    species: object
    slot: int                       # 1–6 on the physical mat
    stats: dict
    hp: int
    pp: dict
    status: str | None = None       # "slp" | "par" | "frz" | "brn"
    sleep_turns: int = 0
    stages: dict = field(default_factory=lambda: dict.fromkeys(STAT_NAMES, 0))
    flinched: bool = False
    seeded: bool = False
    down: bool = False              # faint already announced

    @classmethod
    def create(cls, key, slot):
        sp = POKEDEX[key]
        stats = calc_stats(sp)
        return cls(sp, slot, stats, stats["hp"],
                   {m: MOVES[m].pp for m in sp.moves})

    @property
    def name(self):
        return self.species.name

    @property
    def max_hp(self):
        return self.stats["hp"]

    @property
    def fainted(self):
        return self.hp <= 0

    def stat(self, key, ignore=None):
        """Stat with stages; ignore="neg"/"pos" drops those stages (crits)."""
        stage = self.stages[key]
        if (ignore == "neg" and stage < 0) or (ignore == "pos" and stage > 0):
            stage = 0
        return int(self.stats[key] * STAGE_MULT[stage])


@dataclass
class Side:
    label: str                      # "player" | "cpu"
    party: list                     # Mons in slot order
    bag: dict
    active: int = 0                 # index into party

    @property
    def mon(self):
        return self.party[self.active]

    @property
    def can_battle(self):
        return any(not m.fainted for m in self.party)

    def bench(self):
        return [i for i, m in enumerate(self.party)
                if i != self.active and not m.fainted]


def build_side(label, slots, bag=None):
    """slots: {slot_number: species_key} as read off the mat (1 = top-left)."""
    if not slots:
        raise ValueError(f"{label}: no cards on the mat")
    for slot, key in slots.items():
        if not 1 <= slot <= PARTY_SIZE:
            raise ValueError(f"{label}: slot {slot} is off the 3x2 grid")
        if key not in POKEDEX:
            raise ValueError(f"{label}: unknown card {key!r}")
    party = [Mon.create(slots[s], s) for s in sorted(slots)]
    return Side(label, party, dict(STARTING_BAG if bag is None else bag))


def parse_mat(text):
    """"1=politoed,2=registeel" → {1: "politoed", 2: "registeel"}.
    Stand-in for the grid camera until card detection exists."""
    slots = {}
    for part in text.split(","):
        slot, key = part.split("=")
        slots[int(slot)] = key.strip().lower()
    return slots


# ── Actions (what a menu pick produces) ───────────────────────────────
@dataclass(frozen=True)
class Fight:
    move: str


@dataclass(frozen=True)
class UseItem:
    item: str
    target: int                     # party index


@dataclass(frozen=True)
class Switch:
    target: int                     # party index


@dataclass(frozen=True)
class Run:
    pass


@dataclass
class Event:
    kind: str       # switch_out, switch_in, item, move, miss, fail, damage,
                    # effective, crit, heal, stat, status, cant_move, cure,
                    # recoil, residual, faint, flinch, forfeit, win
    text: str
    side: str | None = None
    data: dict = field(default_factory=dict)


# ── Engine ────────────────────────────────────────────────────────────
class Battle:
    def __init__(self, player, cpu, rng=None, cpu_brain=None):
        self.player, self.cpu = player, cpu
        self.rng = rng or random.Random()
        if cpu_brain is None:
            from .cpu import choose_action as cpu_brain
        self.cpu_brain = cpu_brain      # fn(battle, side, rng) -> action
        self.winner = None              # "player" | "cpu"
        self.turn = 0
        self._moved = set()

    # ── queries ──
    def other(self, side):
        return self.cpu if side is self.player else self.player

    def needs_switch(self, side):
        return self.winner is None and side.mon.fainted and side.can_battle

    def legal_actions(self, side):
        if self.winner:
            return []
        if self.needs_switch(side):
            return [Switch(i) for i in side.bench()]
        mon = side.mon
        acts = [Fight(m) for m in mon.species.moves if mon.pp[m] > 0]
        if not acts:
            acts = [Fight("Struggle")]
        for item, count in side.bag.items():
            if count > 0:
                acts += [UseItem(item, i) for i, m in enumerate(side.party)
                         if self._item_has_effect(ITEMS[item], m)]
        acts += [Switch(i) for i in side.bench()]
        if side is self.player:
            acts.append(Run())
        return acts

    def estimate_damage(self, user, target, move_name):
        """Average no-crit damage (used by the CPU and the info card)."""
        move = MOVES[move_name]
        if move.category == "status":
            return 0
        if move.effect == "halve_target_hp":
            return max(1, target.hp // 2) if self._eff(move, target) else 0
        lo = self._damage(user, target, move, crit=False, roll=85)
        hi = self._damage(user, target, move, crit=False, roll=100)
        return (lo + hi) / 2

    # ── turn entry points ──
    def play_turn(self, player_action, cpu_action=None):
        if self.winner:
            raise RuntimeError("battle is over")
        if self.needs_switch(self.player) or self.needs_switch(self.cpu):
            raise RuntimeError("a fainted Pokémon must be replaced first")
        if player_action not in self.legal_actions(self.player):
            raise ValueError(f"illegal action {player_action}")
        if cpu_action is None:
            cpu_action = self.cpu_brain(self, self.cpu, self.rng)

        self.turn += 1
        self._moved = set()
        ev = []
        if isinstance(player_action, Run):
            ev.append(Event("forfeit", "You forfeited the battle.", "player"))
            return self._end(self.cpu, ev)

        acts = {"player": player_action, "cpu": cpu_action}
        sides = [self.player, self.cpu]
        self.rng.shuffle(sides)         # order inside switch/item phases
        for s in sides:
            if isinstance(acts[s.label], Switch):
                self._switch(s, acts[s.label].target, ev)
        for s in sides:
            if isinstance(acts[s.label], UseItem):
                self._use_item(s, acts[s.label], ev)

        movers = [s for s in (self.player, self.cpu)
                  if isinstance(acts[s.label], Fight)]
        movers.sort(key=lambda s: (-self._move(acts[s.label].move).priority,
                                   self.rng.random()))
        for s in movers:
            if self.winner:
                break
            if not s.mon.fainted:
                self._use_move(s, acts[s.label].move, acts, ev)
                self._moved.add(s.label)

        for s in (self.player, self.cpu):
            s.mon.flinched = False
        if not self.winner:
            self._end_of_turn(ev)
        if self.needs_switch(self.cpu):     # CPU replaces in slot order
            self._switch(self.cpu, self.cpu.bench()[0], ev)
        return ev

    def replace_fainted(self, side, target):
        """The free, non-cancellable switch after a faint."""
        if not self.needs_switch(side):
            raise RuntimeError("no fainted Pokémon to replace")
        if target not in side.bench():
            raise ValueError(f"can't send out party index {target}")
        ev = []
        self._switch(side, target, ev)
        return ev

    # ── phases ──
    def _switch(self, side, target, ev):
        old = side.mon
        old.stages = dict.fromkeys(STAT_NAMES, 0)
        old.flinched = old.seeded = False
        if not old.fainted:
            txt = (f"{old.name}, come back!" if side is self.player
                   else f"The CPU withdrew {old.name}!")
            ev.append(Event("switch_out", txt, side.label, {"slot": old.slot}))
        side.active = target
        new = side.mon
        txt = (f"Go! {new.name}!" if side is self.player
               else f"The CPU sent out {new.name}!")
        ev.append(Event("switch_in", txt, side.label, {"slot": new.slot}))

    def _item_has_effect(self, item, mon):
        if item.target == "fainted":
            return mon.fainted
        if mon.fainted:
            return False
        if item.effect == "heal":
            return mon.hp < mon.max_hp
        if item.effect == "cure_status":
            return mon.status is not None
        return False

    def _use_item(self, side, act, ev):
        item, mon = ITEMS[act.item], side.party[act.target]
        side.bag[act.item] -= 1
        who = "You" if side is self.player else "The CPU"
        ev.append(Event("item", f"{who} used a {item.name}!", side.label,
                        {"item": item.name, "slot": mon.slot}))
        if item.effect == "heal":
            self._heal(side, mon, item.amount, ev)
        elif item.effect == "cure_status":
            mon.status, mon.sleep_turns = None, 0
            ev.append(Event("cure", f"{self._mon_who(side, mon)} was cured!",
                            side.label, {"slot": mon.slot}))
        elif item.effect == "revive_half":
            mon.hp = mon.max_hp // 2
            mon.status, mon.sleep_turns, mon.down = None, 0, False
            ev.append(Event("heal", f"{self._mon_who(side, mon)} was revived!",
                            side.label, {"slot": mon.slot, "hp": mon.hp}))

    def _use_move(self, side, move_name, acts, ev):
        user, foe_side = side.mon, self.other(side)
        target = foe_side.mon
        move = self._move(move_name)
        if not self._can_act(side, user, ev):
            return
        if move is not STRUGGLE:
            user.pp[move.name] -= 1
        ev.append(Event("move", f"{self._who(side)} used {move.name}!",
                        side.label, {"move": move.name, "type": move.type}))
        self._execute(side, user, foe_side, target, move, acts, ev)
        if move.effect == "self_ko" and not user.fainted:
            user.hp = 0                 # Explosion: faints hit or miss
            self._faint_check(side, user, ev)
            self._check_winner(ev, attacker=side, self_ko=True)

    def _execute(self, side, user, foe_side, target, move, acts, ev):
        if move.effect == "fails_unless_target_attacks":
            foe_act = acts[foe_side.label]
            if (not isinstance(foe_act, Fight) or foe_side.label in self._moved
                    or self._move(foe_act.move).category == "status"):
                ev.append(Event("fail", "But it failed!", side.label))
                return
        if move.accuracy is not None and \
                self.rng.randint(1, 100) > move.accuracy:
            ev.append(Event("miss", f"{self._who(foe_side)} avoided the attack!",
                            side.label))
            return
        if move.category == "status":
            self._status_move(side, user, foe_side, target, move, ev)
        else:
            self._damaging_move(side, user, foe_side, target, move, ev)

    def _can_act(self, side, user, ev):
        who = self._who(side)
        if user.flinched:
            ev.append(Event("flinch", f"{who} flinched and couldn't move!",
                            side.label))
            return False
        if user.status == "frz":
            if self.rng.random() < 0.2:
                user.status = None
                ev.append(Event("cure", f"{who} thawed out!", side.label))
            else:
                ev.append(Event("cant_move", f"{who} is frozen solid!",
                                side.label, {"status": "frz"}))
                return False
        if user.status == "slp":
            if user.sleep_turns > 0:
                user.sleep_turns -= 1
                ev.append(Event("cant_move", f"{who} is fast asleep.",
                                side.label, {"status": "slp"}))
                return False
            user.status = None
            ev.append(Event("cure", f"{who} woke up!", side.label))
        if user.status == "par" and self.rng.random() < 0.25:
            ev.append(Event("cant_move", f"{who} is paralyzed! It can't move!",
                            side.label, {"status": "par"}))
            return False
        return True

    def _status_move(self, side, user, foe_side, target, move, ev):
        fx = move.effect
        if fx == "heal_1/2":
            if user.hp == user.max_hp:
                return self._fail(side, ev)
            self._heal(side, user, user.max_hp // 2, ev)
        elif fx == "sleep":
            if target.status:
                return self._fail(side, ev)
            self._inflict(foe_side, target, "slp", ev)
        elif fx == "leech_seed":
            if target.seeded or "Grass" in target.species.types:
                return self._fail(side, ev)
            target.seeded = True
            ev.append(Event("status", f"{self._who(foe_side)} was seeded!",
                            foe_side.label, {"slot": target.slot,
                                             "status": "seeded"}))
        elif fx == "eat_bag_berry":
            if side.bag.get("Oran Berry", 0) <= 0:
                return self._fail(side, ev, "But it failed! (No berries left)")
            side.bag["Oran Berry"] -= 1
            ev.append(Event("item", f"{self._who(side)} stuffed an Oran "
                                    "Berry in its cheeks!", side.label,
                            {"item": "Oran Berry", "slot": user.slot}))
            self._heal(side, user, ITEMS["Oran Berry"].amount, ev)
        for stat, delta in move.boosts:
            self._stage(side, user, stat, delta, ev)
        for stat, delta in move.target_boosts:
            self._stage(foe_side, target, stat, delta, ev)

    def _damaging_move(self, side, user, foe_side, target, move, ev):
        eff = self._eff(move, target)
        if eff == 0:
            ev.append(Event("fail", f"It doesn't affect {self._who(foe_side)}...",
                            side.label))
            return
        if move.effect == "steal_boosts":
            stolen = {k: v for k, v in target.stages.items() if v > 0}
            if stolen:
                for k, v in stolen.items():
                    target.stages[k] = 0
                    user.stages[k] = min(6, user.stages[k] + v)
                ev.append(Event("stat", f"{self._who(side)} stole the target's "
                                        "boosted stats!", side.label,
                                {"slot": user.slot, "stolen": stolen}))
        crit = False
        if move.effect == "halve_target_hp":
            dmg = max(1, target.hp // 2)
        else:
            crit = self.rng.random() < CRIT_CHANCE
            dmg = max(1, self._damage(user, target, move, crit=crit,
                                      roll=self.rng.randint(85, 100)))
        dmg = min(dmg, target.hp)
        target.hp -= dmg
        ev.append(Event("damage", "", foe_side.label,
                        {"slot": target.slot, "amount": dmg, "hp": target.hp,
                         "max_hp": target.max_hp, "effectiveness": eff,
                         "crit": crit}))
        if crit:
            ev.append(Event("crit", "A critical hit!", side.label))
        if move.effect != "halve_target_hp" and eff != 1:
            ev.append(Event("effective",
                            "It's super effective!" if eff > 1
                            else "It's not very effective...",
                            side.label, {"effectiveness": eff}))

        recoil = (max(1, user.max_hp // 4) if move.effect == "struggle_recoil"
                  else max(1, int(dmg * move.recoil)) if move.recoil else 0)
        if recoil:
            user.hp = max(0, user.hp - recoil)
            ev.append(Event("recoil", f"{self._who(side)} was damaged by "
                                      "the recoil!", side.label,
                            {"slot": user.slot, "amount": recoil,
                             "hp": user.hp}))
        for stat, delta in move.boosts:
            if not user.fainted:
                self._stage(side, user, stat, delta, ev)

        if not target.fainted and self.rng.random() * 100 < move.effect_chance:
            fx = move.effect
            if fx == "flinch" and foe_side.label not in self._moved:
                target.flinched = True
            elif fx in ("paralyze", "freeze", "burn"):
                status = {"paralyze": "par", "freeze": "frz", "burn": "brn"}[fx]
                if not target.status and \
                        STATUS_IMMUNE[status] not in target.species.types:
                    self._inflict(foe_side, target, status, ev)
            for stat, delta in move.target_boosts:
                self._stage(foe_side, target, stat, delta, ev)

        self._faint_check(foe_side, target, ev)
        self._faint_check(side, user, ev)
        if move.effect != "self_ko":        # _use_move decides that one
            self._check_winner(ev, attacker=side)

    def _end_of_turn(self, ev):
        for side in (self.player, self.cpu):
            mon, foe = side.mon, self.other(side).mon
            if mon.fainted:
                continue
            if mon.status == "brn":
                dmg = min(mon.hp, max(1, mon.max_hp // 16))
                mon.hp -= dmg
                ev.append(Event("residual", f"{self._who(side)} was hurt by "
                                            "its burn!", side.label,
                                {"slot": mon.slot, "amount": dmg,
                                 "hp": mon.hp, "cause": "brn"}))
            if mon.seeded and not mon.fainted:
                dmg = min(mon.hp, max(1, mon.max_hp // 8))
                mon.hp -= dmg
                ev.append(Event("residual", f"{self._who(side)}'s health is "
                                            "sapped by Leech Seed!", side.label,
                                {"slot": mon.slot, "amount": dmg,
                                 "hp": mon.hp, "cause": "seeded"}))
                if not foe.fainted:
                    self._heal(self.other(side), foe, dmg, ev)
            self._faint_check(side, mon, ev)
        self._check_winner(ev)

    # ── helpers ──
    def _move(self, name):
        return STRUGGLE if name == "Struggle" else MOVES[name]

    def _eff(self, move, target):
        if move.type == "Typeless":
            return 1
        return effectiveness(move.type, target.species.types)

    def _damage(self, user, target, move, *, crit, roll):
        atk_key, def_key = (("atk", "def") if move.category == "physical"
                            else ("spa", "spd"))
        atk = user.stat(atk_key, ignore="neg" if crit else None)
        dfn = target.stat(def_key, ignore="pos" if crit else None)
        power = (move.power if move is STRUGGLE
                 else move_power(move, user.species, target.species,
                                 user_hp_frac=(user.hp, user.max_hp),
                                 target_status=target.status))
        stab = 1.5 if move.type in user.species.types else 1
        return final_damage(LEVEL, power, atk, dfn, crit=crit, roll=roll,
                            stab=stab, eff=self._eff(move, target),
                            burned=(user.status == "brn"
                                    and move.category == "physical"))

    def _who(self, side):
        return self._mon_who(side, side.mon)

    def _mon_who(self, side, mon):
        return mon.name if side is self.player else f"The opposing {mon.name}"

    def _fail(self, side, ev, text="But it failed!"):
        ev.append(Event("fail", text, side.label))

    def _heal(self, side, mon, amount, ev):
        gained = min(amount, mon.max_hp - mon.hp)
        mon.hp += gained
        ev.append(Event("heal", f"{self._mon_who(side, mon)} regained health!",
                        side.label,
                        {"slot": mon.slot, "amount": gained, "hp": mon.hp}))

    def _stage(self, side, mon, stat, delta, ev):
        old = mon.stages[stat]
        new = max(-6, min(6, old + delta))
        who = self._mon_who(side, mon)
        if new == old:
            txt = (f"{who}'s {STAT_NAMES[stat]} won't go any "
                   f"{'higher' if delta > 0 else 'lower'}!")
        else:
            mon.stages[stat] = new
            verb = {2: "rose sharply", 1: "rose", -1: "fell",
                    -2: "harshly fell"}[max(-2, min(2, new - old))]
            txt = f"{who}'s {STAT_NAMES[stat]} {verb}!"
        ev.append(Event("stat", txt, side.label,
                        {"slot": mon.slot, "stat": stat, "stage": new}))

    def _inflict(self, side, mon, status, ev):
        mon.status = status
        if status == "slp":
            mon.sleep_turns = self.rng.randint(1, 3)
        verb = {"slp": "fell asleep", "par": "is paralyzed! It may be "
                "unable to move", "frz": "was frozen solid",
                "brn": "was burned"}[status]
        ev.append(Event("status", f"{self._mon_who(side, mon)} {verb}!",
                        side.label, {"slot": mon.slot, "status": status}))

    def _faint_check(self, side, mon, ev):
        if mon.fainted and not mon.down:
            mon.down = True
            mon.status, mon.sleep_turns, mon.seeded = None, 0, False
            ev.append(Event("faint", f"{self._mon_who(side, mon)} fainted!",
                            side.label, {"slot": mon.slot}))

    def _check_winner(self, ev, attacker=None, self_ko=False):
        """Both sides out at once: recoil → attacker wins; Explosion →
        attacker loses; end-of-turn damage → counts as your loss."""
        if self.winner:
            return
        p_out, c_out = not self.player.can_battle, not self.cpu.can_battle
        if p_out and c_out:
            if attacker is None:
                winner = self.cpu
            else:
                winner = self.other(attacker) if self_ko else attacker
            return self._end(winner, ev)
        if p_out or c_out:
            return self._end(self.cpu if p_out else self.player, ev)

    def _end(self, winner, ev):
        self.winner = winner.label
        ev.append(Event("win", "You won the battle!" if winner is self.player
                        else "You lost the battle...", winner.label))
        return ev
