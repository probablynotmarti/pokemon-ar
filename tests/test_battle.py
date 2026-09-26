import random
import unittest

from game.battle import (Battle, Fight, Run, Switch, UseItem, build_side)
from game.cpu import choose_action


class Rigged(random.Random):
    """No crits/misses/secondary effects/para; coin flips favor the player."""
    def __init__(self, r=0.99):
        super().__init__(0)
        self.r = r

    def random(self):
        return self.r

    def randint(self, a, b):
        return a

    def shuffle(self, x):
        pass


def fixed(action):
    return lambda battle, side, rng: action


def make(player=None, cpu=None, cpu_action=Fight("Hyper Voice"), rng=None):
    p = build_side("player", player or {1: "politoed", 2: "registeel"})
    c = build_side("cpu", cpu or {1: "politoed", 2: "ducklett"})
    return Battle(p, c, rng or Rigged(), cpu_brain=fixed(cpu_action))


def kinds(events):
    return [e.kind for e in events]


class Setup(unittest.TestCase):
    def test_party_in_slot_order_lead_is_top_left(self):
        s = build_side("player", {5: "skwovet", 1: "registeel", 3: "ducklett"})
        self.assertEqual([m.slot for m in s.party], [1, 3, 5])
        self.assertEqual(s.mon.name, "Registeel")

    def test_bad_mats_rejected(self):
        with self.assertRaises(ValueError):
            build_side("player", {})
        with self.assertRaises(ValueError):
            build_side("player", {7: "skwovet"})
        with self.assertRaises(ValueError):
            build_side("player", {1: "pikachu"})

    def test_duplicates_are_separate_pokemon(self):
        s = build_side("player", {1: "ducklett", 2: "ducklett"})
        s.party[0].hp = 1
        self.assertEqual(s.party[1].hp, s.party[1].max_hp)


class TurnOrder(unittest.TestCase):
    def test_switch_happens_before_moves(self):
        b = make()
        ev = b.play_turn(Switch(1))
        self.assertEqual(b.player.mon.name, "Registeel")
        hit = next(e for e in ev if e.kind == "damage")
        self.assertEqual(hit.data["slot"], 2)       # Registeel took it
        self.assertLess(kinds(ev).index("switch_in"), kinds(ev).index("move"))

    def test_item_before_moves_and_consumed(self):
        b = make()
        b.player.mon.hp = 50
        ev = b.play_turn(UseItem("Potion", 0))
        self.assertLess(kinds(ev).index("item"), kinds(ev).index("move"))
        self.assertEqual(b.player.bag["Potion"], 2)

    def test_speed_is_a_coin_flip(self):
        firsts = set()
        for seed in range(40):
            b = make(rng=random.Random(seed))
            ev = b.play_turn(Fight("Hyper Voice"))
            firsts.add(next(e for e in ev if e.kind == "move").side)
        self.assertEqual(firsts, {"player", "cpu"})

    def test_priority_beats_coin_flip(self):
        for seed in range(20):
            b = make(player={1: "sudowoodo"}, rng=random.Random(seed))
            ev = b.play_turn(Fight("Sucker Punch"))
            self.assertEqual(next(e for e in ev if e.kind == "move").side,
                             "player")

    def test_sucker_punch_fails_vs_status_move(self):
        b = make(player={1: "sudowoodo"}, cpu_action=Fight("Hypnosis"))
        ev = b.play_turn(Fight("Sucker Punch"))
        self.assertIn("fail", kinds(ev))


class Moves(unittest.TestCase):
    def test_pp_goes_down_and_struggle_when_empty(self):
        b = make()
        b.play_turn(Fight("Hydro Pump"))
        self.assertEqual(b.player.mon.pp["Hydro Pump"], 4)
        for m in b.player.mon.pp:
            b.player.mon.pp[m] = 0
        fights = [a for a in b.legal_actions(b.player) if isinstance(a, Fight)]
        self.assertEqual(fights, [Fight("Struggle")])
        ev = b.play_turn(Fight("Struggle"))
        self.assertIn("recoil", kinds(ev))

    def test_double_edge_recoil_is_a_third(self):
        b = make(player={1: "sudowoodo"}, cpu_action=Fight("Hypnosis"))
        ev = b.play_turn(Fight("Double-Edge"))
        dealt = next(e for e in ev if e.kind == "damage" and e.side == "cpu")
        recoil = next(e for e in ev if e.kind == "recoil")
        self.assertEqual(recoil.data["amount"], dealt.data["amount"] // 3)

    def test_super_fang_halves(self):
        b = make(player={1: "skwovet"}, cpu={1: "registeel"},
                 cpu_action=Fight("Iron Defense"))
        b.play_turn(Fight("Super Fang"))
        self.assertEqual(b.cpu.mon.hp, 155 - 77)

    def test_stuff_cheeks_eats_bag_berry_then_fails(self):
        b = make(player={1: "skwovet"}, cpu_action=Fight("Hypnosis"),
                 rng=Rigged(r=0.99))
        b.player.bag["Oran Berry"] = 1
        b.player.mon.hp = 100
        b.play_turn(Fight("Stuff Cheeks"))
        self.assertEqual(b.player.bag["Oran Berry"], 0)
        self.assertEqual(b.player.mon.hp, 110)
        self.assertEqual(b.player.mon.stages["def"], 2)

        b2 = make(player={1: "skwovet"}, cpu_action=Fight("Hyper Voice"))
        b2.player.bag["Oran Berry"] = 0
        self.assertIn("fail", kinds(b2.play_turn(Fight("Stuff Cheeks"))))

    def test_sleep_blocks_moves_and_hypnosis_needs_clean_target(self):
        b = make(cpu_action=Fight("Hyper Voice"))
        ev = b.play_turn(Fight("Hypnosis"))     # Rigged: hits, 1 turn
        self.assertEqual(b.cpu.mon.status, "slp")
        self.assertIn("cant_move", kinds(ev))   # CPU slept through turn 1
        ev = b.play_turn(Fight("Hypnosis"))
        self.assertIn("fail", kinds(ev))        # still asleep → fails
        self.assertIn("damage", kinds(ev))      # CPU woke up and attacked

    def test_stages_reset_on_switch(self):
        b = make(player={1: "registeel", 2: "politoed"})
        b.play_turn(Fight("Iron Defense"))
        self.assertEqual(b.player.mon.stages["def"], 2)
        b.play_turn(Switch(1))
        self.assertEqual(b.player.party[0].stages["def"], 0)


class Fainting(unittest.TestCase):
    def test_forced_switch_is_required_and_free(self):
        b = make()
        b.player.mon.hp = 1
        b.play_turn(Fight("Hyper Voice"))
        self.assertTrue(b.needs_switch(b.player))
        self.assertEqual(b.legal_actions(b.player), [Switch(1)])  # no Back
        with self.assertRaises(RuntimeError):
            b.play_turn(Fight("Hydro Pump"))
        turn = b.turn
        b.replace_fainted(b.player, 1)
        self.assertEqual(b.turn, turn)          # didn't cost a turn
        self.assertEqual(b.player.mon.name, "Registeel")

    def test_cpu_replaces_in_slot_order(self):
        b = make(cpu={1: "skwovet", 3: "ducklett", 6: "registeel"},
                 cpu_action=Fight("Bite"))
        b.cpu.mon.hp = 1
        b.play_turn(Fight("Hydro Pump"))
        self.assertEqual(b.cpu.mon.slot, 3)

    def test_last_faint_wins(self):
        b = make(cpu={1: "ducklett"}, cpu_action=Fight("Roost"))
        b.cpu.mon.hp = 1
        ev = b.play_turn(Fight("Hyper Voice"))
        self.assertEqual(b.winner, "player")
        self.assertEqual(ev[-1].kind, "win")

    def test_run_forfeits(self):
        b = make()
        b.play_turn(Run())
        self.assertEqual(b.winner, "cpu")


class Items(unittest.TestCase):
    def test_no_wasted_items(self):
        b = make()
        legal = b.legal_actions(b.player)
        self.assertNotIn(UseItem("Potion", 0), legal)       # full HP
        self.assertNotIn(UseItem("Full Heal", 0), legal)    # no status
        self.assertNotIn(UseItem("Revive", 0), legal)       # not fainted

    def test_revive_benched_fainted(self):
        b = make()
        b.player.party[1].hp = 0
        self.assertIn(UseItem("Revive", 1), b.legal_actions(b.player))
        b.play_turn(UseItem("Revive", 1))
        self.assertEqual(b.player.party[1].hp, 155 // 2)
        self.assertEqual(b.player.bag["Revive"], 0)


class Fuzz(unittest.TestCase):
    def test_random_battles_always_finish(self):
        for seed in range(150):
            rng = random.Random(seed)
            b = Battle(build_side("player", {1: "ducklett", 2: "skwovet",
                                             5: "sudowoodo"}),
                       build_side("cpu", {1: "registeel", 2: "politoed"}),
                       rng)
            while not b.winner:
                if b.needs_switch(b.player):
                    b.replace_fainted(b.player, rng.choice(b.player.bench()))
                    continue
                b.play_turn(rng.choice(b.legal_actions(b.player)))
                self.assertLess(b.turn, 400)
            for s in (b.player, b.cpu):
                for m in s.party:
                    self.assertTrue(0 <= m.hp <= m.max_hp)
                self.assertTrue(all(v >= 0 for v in s.bag.values()))


if __name__ == "__main__":
    unittest.main()
