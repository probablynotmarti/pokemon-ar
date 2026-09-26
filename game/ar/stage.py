"""Who stands where on the board, and how they move.

Driven by battle Events (the same list the menu/HP bars animate), so
figurines lunge, flinch and faint in sync with the text. Pure maths over
time — no drawing here: placements(now) returns what the renderer should
draw this frame.
"""

import math
from dataclasses import dataclass, field

from . import move_fx
from .figurines import FLOATS
from .fx import FxSystem
from .render import Placement, yaw_towards
from game.data import MOVES

DUR = {"enter": 0.7, "leave": 0.45, "lunge": 0.7, "hop": 0.5, "hit": 0.6,
       "faint": 0.9, "heal": 0.6, "cheer": 0.8}
STATUS_TINT = {"par": ((40, 215, 245), 0.30), "slp": ((170, 140, 120), 0.30),
               "frz": ((250, 225, 170), 0.50), "brn": ((40, 90, 235), 0.28)}
FLOAT_LIFT = 6.0               # floaters (Solrock) hover this much higher
BATTLE_SCALE = 1.5             # in battle only two are out: show them big


def ease_out_back(p):
    c = 1.70158
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


@dataclass
class Actor:
    key: str
    side: str
    slot: int
    anims: list = field(default_factory=list)   # [(kind, t0)]
    gone: bool = False                          # finished leaving/fainting
    phase: float = 0.0                          # idle bob offset

    def start(self, kind, now):
        self.anims = [a for a in self.anims if a[0] != kind] + [(kind, now)]

    def active(self, now):
        """{kind: progress 0..1} for animations still running."""
        out, keep = {}, []
        for kind, t0 in self.anims:
            p = (now - t0) / DUR[kind]
            if p < 1:
                out[kind] = max(0.0, p)
                keep.append((kind, t0))
            elif kind in ("leave", "faint"):
                self.gone = True
        self.anims = keep
        return out


class Stage:
    """mode "setup": a figurine on every recognized card.
    mode "battle": only the two active Pokémon, animated by events."""

    def __init__(self):
        self.mode = "setup"
        self.actors = {}                         # (side, slot) → Actor
        self.active = {}                         # side → slot (battle)
        self.fx = FxSystem()                     # move effects / particles
        self._spots = {}                         # (side, slot) → (x, y) mm
        self._last_move = {}                     # side → move name

    # ── setup ──
    SETUP_GRACE_S = 1.0          # a card read that flickers doesn't blink

    def sync_setup(self, cards, now):
        """cards: {(side, slot): key} recognized right now. A card has to be
        gone for SETUP_GRACE_S before its Pokémon leaves, so a momentary
        misread (hand passing over, glare) doesn't make it blink out/in."""
        self.mode = "setup"
        self._missing = getattr(self, "_missing", {})
        for zone, key in cards.items():
            self._missing.pop(zone, None)
            a = self.actors.get(zone)
            if a is None or a.key != key or a.gone:
                a = Actor(key, *zone, phase=zone[1] * 1.7 + (zone[0] == "cpu") * 0.9)
                a.start("enter", now)
                self.actors[zone] = a
        for zone in list(self.actors):
            if zone not in cards:
                since = self._missing.setdefault(zone, now)
                if now - since >= self.SETUP_GRACE_S:
                    del self.actors[zone]
                    del self._missing[zone]

    # ── battle ──
    def start_battle(self, parties):
        """parties: {side: {slot: key}}. Nobody is out until send-out."""
        self.mode = "battle"
        self.parties = parties
        self.actors, self.active = {}, {}
        self.fx.clear()
        self._last_move = {}

    def send_out(self, side, slot, now):
        a = Actor(self.parties[side][slot], side, slot,
                  phase=1.3 if side == "cpu" else 0.0)
        a.start("enter", now)
        self.actors[(side, slot)] = a
        self.active[side] = slot
        spot = self._spots.get((side, slot))
        if spot is not None:
            move_fx.send_out_fx(self.fx, spot, now)

    def _actor(self, side):
        slot = self.active.get(side)
        return self.actors.get((side, slot)) if slot is not None else None

    def _spot(self, side):
        return self._spots.get((side, self.active.get(side)))

    def on_event(self, e, now):
        self._effects(e, now)
        a = self._actor(e.side) if e.side else None
        k = e.kind
        if k == "switch_in":
            self.send_out(e.side, e.data["slot"], now)
        elif a is None:
            return
        elif k == "switch_out":
            a.start("leave", now)
        elif k == "move":
            move = MOVES.get(e.data.get("move"))
            status = move is not None and move.category == "status"
            a.start("hop" if status else "lunge", now)
        elif k in ("damage", "recoil", "residual"):
            a.start("hit", now)
        elif k == "heal":
            a.start("heal", now)
        elif k == "faint":
            a.start("faint", now)
        elif k == "win":
            a.start("cheer", now)

    def _effects(self, e, now):
        """Battle effects for an event (needs card positions: known once a
        frame has been drawn)."""
        k, side = e.kind, e.side
        if side not in ("player", "cpu"):
            return
        foe = "cpu" if side == "player" else "player"
        me, them = self._spot(side), self._spot(foe)
        fx = self.fx
        if k == "move" and me and them:
            self._last_move[side] = e.data.get("move")
            move_fx.launch(fx, e.data.get("move"), me, them, now)
        elif k == "damage" and me:           # side = the one getting hit
            move_fx.impact(fx, self._last_move.get(foe), me, now)
        elif k == "effective" and e.data.get("effectiveness", 1) > 1:
            fx.shake(now, 12)
        elif k == "crit":
            fx.shake(now, 9)
        elif k in ("status", "cant_move") and me:
            move_fx.status_fx(fx, e.data.get("status"), me, now)
        elif k == "residual" and me:
            if e.data.get("cause") == "seeded" and them:
                move_fx.drain_fx(fx, me, them, now)
            else:
                move_fx.status_fx(fx, e.data.get("cause"), me, now)
        elif k == "heal" and me:
            move_fx.heal_fx(fx, me, now)
        elif k == "stat" and me:
            move_fx.stat_fx(fx, me, now, up="rose" in e.text)
        elif k == "faint" and me:
            move_fx.faint_fx(fx, me, now)

    # ── per frame ──
    def placements(self, now, where, status_of=lambda side, slot: None):
        """→ [(key, Placement, shadow_radius)] far → near.
        where(side, slot) → (x, y) board mm of that card."""
        live = {z: a for z, a in self.actors.items() if not a.gone}
        if self.mode == "battle":                 # remember every card spot
            for side, slots in self.parties.items():
                for slot in slots:
                    self._spots[(side, slot)] = where(side, slot)
        spots = {z: where(*z) for z in live}
        self._spots.update(spots)
        out = []
        for zone, a in live.items():
            x, y = spots[zone]
            # face the opponent: their active figurine, else the far half
            foe = "cpu" if a.side == "player" else "player"
            foe_zone = (foe, self.active.get(foe))
            if self.mode == "battle" and foe_zone in spots:
                tx, ty = spots[foe_zone]
            else:
                tx, ty = x * 0.5 + 127, (130.0 if a.side == "player" else 380.0)
            place = self._animate(a, now, x, y, tx, ty,
                                  status_of(a.side, a.slot))
            if place is not None:
                out.append((a.key, place, 22.0, a.side))
        out.sort(key=lambda item: item[1].y)          # far (small y) first
        return out

    def _animate(self, a, now, x, y, tx, ty, status):
        run = a.active(now)
        if a.gone:
            return None
        t = now + a.phase
        slow = 0.5 if status == "slp" else 1.0
        lift = 1.5 * math.sin(2 * math.pi * t * slow / 1.6) + 1.5
        if a.key in FLOATS:
            lift += FLOAT_LIFT + 2 * math.sin(2 * math.pi * t / 2.2)
        squash = 1 + 0.025 * math.sin(2 * math.pi * t * slow / 1.6)
        p = Placement(x, y, yaw_towards(x, y, tx, ty), squash=squash)
        if "enter" in run:
            q = run["enter"]
            p.scale = max(0.0, ease_out_back(q))
            p.yaw += 2 * math.pi * (1 - q) ** 2
            lift += 10 * math.sin(math.pi * q)
        if "leave" in run:
            q = run["leave"]
            p.scale = (1 - q) ** 2
            lift += 12 * q
        if "lunge" in run:
            q = run["lunge"]
            reach = min(0.45, 120 / max(1.0, math.hypot(tx - x, ty - y)))
            k = reach * math.sin(math.pi * q)
            p.x, p.y = x + (tx - x) * k, y + (ty - y) * k
            lift += 12 * math.sin(math.pi * q)
        if "hop" in run:
            lift += 16 * math.sin(math.pi * run["hop"])
        if "cheer" in run:
            lift += 14 * abs(math.sin(2 * math.pi * run["cheer"]))
        if "hit" in run:
            q = run["hit"]
            p.x += 6 * math.sin(q * 6 * math.pi) * (1 - q)
            # brighten (not white-out: a white flash vanishes against a
            # white table) in three quick pulses
            p.flash = 0.35 * max(0.0, math.sin(q * 3 * math.pi))
        if "faint" in run:
            q = run["faint"]
            p.scale = 1 - q ** 1.5
            p.squash *= 1 - 0.5 * q
            p.tint, p.tint_amount = (120, 120, 120), 0.6 * q
        if "heal" in run:
            p.tint, p.tint_amount = (120, 230, 120), 0.4 * math.sin(math.pi * run["heal"])
        if status in STATUS_TINT and p.tint_amount == 0:
            col, amt = STATUS_TINT[status]
            p.tint = col
            p.tint_amount = amt * (0.75 + 0.25 * math.sin(2 * math.pi * t))
        if self.mode == "battle":
            p.scale *= BATTLE_SCALE
            lift *= BATTLE_SCALE
        p.lift = max(0.0, lift)
        return p
