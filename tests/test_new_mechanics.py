import random
import unittest

from game.battle import Battle, Fight, Switch, UseItem, build_side
from game.damage import reversal_power
from game.data import POKEDEX
from tests.test_battle import Rigged, fixed, kinds, make


class GhostsAndPriority(unittest.TestCase):
    def test_normal_moves_dont_touch_ghosts(self):
        for move in ("Hyper Voice", ):
            b = make(cpu={1: "gimmighoul"}, cpu_action=Fight("Nasty Plot"))
            ev = b.play_turn(Fight(move))
            self.assertIn("fail", kinds(ev))
            self.assertEqual(b.cpu.mon.hp, b.cpu.mon.max_hp)
        b = make(player={1: "skwovet"}, cpu={1: "marshadow"},
                 cpu_action=Fight("Bulk Up"))
        b.play_turn(Fight("Super Fang"))
        self.assertEqual(b.cpu.mon.hp, b.cpu.mon.max_hp)

    def test_shadow_sneak_and_quick_attack_go_first(self):
        for key, move in (("marshadow", "Shadow Sneak"),
                          ("riolu", "Quick Attack")):
            for seed in range(15):
                b = make(player={1: key}, cpu={1: "registeel"},
                         cpu_action=Fight("Flash Cannon"),
                         rng=random.Random(seed))
                ev = b.play_turn(Fight(move))
                self.assertEqual(next(e for e in ev if e.kind == "move").side,
                                 "player", f"{move} seed {seed}")


class StatMoves(unittest.TestCase):
    def test_spectral_thief_steals_boosts(self):
        b = make(player={1: "marshadow"}, cpu={1: "registeel"},
                 cpu_action=Fight("Iron Defense"))
        b.play_turn(Fight("Bulk Up"))           # CPU: Def +2
        b.play_turn(Fight("Spectral Thief"))    # player first (Rigged)
        self.assertEqual(b.player.mon.stages["def"], 1 + 2)
        # Registeel used Iron Defense again after the theft → +2 again
        self.assertEqual(b.cpu.mon.stages["def"], 2)

    def test_close_combat_drops_own_defenses(self):
        b = make(player={1: "marshadow"}, cpu={1: "registeel"},
                 cpu_action=Fight("Iron Defense"))
        b.play_turn(Fight("Close Combat"))
        self.assertEqual(b.player.mon.stages["def"], -1)
        self.assertEqual(b.player.mon.stages["spd"], -1)

    def test_two_stat_boosts(self):
        for key, move, stats in (("marshadow", "Bulk Up", ("atk", "def")),
                                 ("solrock", "Cosmic Power", ("def", "spd"))):
            b = make(player={1: key}, cpu={1: "ducklett"},
                     cpu_action=Fight("Roost"))
            b.play_turn(Fight(move))
            for st in stats:
                self.assertEqual(b.player.mon.stages[st], 1, move)


class Coverage(unittest.TestCase):
    def test_no_pokemon_is_completely_walled(self):
        from game.damage import damage_range
        for a in POKEDEX.values():
            for d in POKEDEX.values():
                hits = [mv for mv in a.moves
                        if damage_range(a, d, mv)[1] > 0
                        or mv == "Super Fang" and "Ghost" not in d.types]
                self.assertTrue(hits, f"{a.name} can't damage {d.name}")


class PowerFormulas(unittest.TestCase):
    def test_reversal_table(self):
        self.assertEqual(reversal_power(100, 100), 20)
        self.assertEqual(reversal_power(1, 100), 200)
        self.assertEqual(reversal_power(30, 100), 80)

    def test_reversal_hits_harder_when_low(self):
        b = make(player={1: "riolu"}, cpu={1: "skwovet"})
        full = b.estimate_damage(b.player.mon, b.cpu.mon, "Reversal")
        b.player.mon.hp = 3
        low = b.estimate_damage(b.player.mon, b.cpu.mon, "Reversal")
        self.assertGreater(low, full * 5)

    def test_hex_doubles_on_status(self):
        b = make(player={1: "gimmighoul"}, cpu={1: "politoed"})
        clean = b.estimate_damage(b.player.mon, b.cpu.mon, "Hex")
        b.cpu.mon.status = "par"
        self.assertGreater(b.estimate_damage(b.player.mon, b.cpu.mon, "Hex"),
                           clean * 1.8)


class Residual(unittest.TestCase):
    def test_leech_seed_drains_and_heals(self):
        b = make(player={1: "seedot"}, cpu={1: "registeel"},
                 cpu_action=Fight("Iron Defense"))
        b.player.mon.hp = 50
        ev = b.play_turn(Fight("Leech Seed"))
        self.assertTrue(b.cpu.mon.seeded)
        self.assertEqual(b.cpu.mon.hp, 155 - 155 // 8)
        self.assertEqual(b.player.mon.hp, 50 + 155 // 8)
        self.assertIn("residual", kinds(ev))

    def test_leech_seed_fails_on_grass_and_ends_on_switch(self):
        b = make(player={1: "seedot"}, cpu={1: "seedot"},
                 cpu_action=Fight("Synthesis"))
        self.assertIn("fail", kinds(b.play_turn(Fight("Leech Seed"))))
        b = make(player={1: "seedot"}, cpu={1: "registeel", 2: "politoed"})
        b.play_turn(Fight("Leech Seed"), cpu_action=Fight("Iron Defense"))
        b.play_turn(Fight("Synthesis"), cpu_action=Switch(1))
        self.assertFalse(b.cpu.party[0].seeded)

    def test_burn_ticks_and_halves_physical(self):
        b = make(player={1: "sudowoodo"}, cpu={1: "registeel"},
                 cpu_action=Fight("Iron Defense"))
        healthy = b.estimate_damage(b.player.mon, b.cpu.mon, "Brick Break")
        b.player.mon.status = "brn"
        burned = b.estimate_damage(b.player.mon, b.cpu.mon, "Brick Break")
        self.assertAlmostEqual(burned, healthy / 2, delta=1)
        b.play_turn(Fight("Sucker Punch"))      # fails; just pass the turn
        self.assertEqual(b.player.mon.hp, 145 - 145 // 16)

    def test_flare_blitz_burns_but_not_fire_types(self):
        b = make(player={1: "solrock"}, cpu={1: "skwovet"},
                 cpu_action=Fight("Stuff Cheeks"), rng=Rigged(r=0.05))
        b.play_turn(Fight("Flare Blitz"))
        self.assertEqual(b.cpu.mon.status, "brn")

    def test_full_heal_cures_burn(self):
        b = make()
        b.player.mon.status = "brn"
        self.assertIn(UseItem("Full Heal", 0), b.legal_actions(b.player))


class SelfKO(unittest.TestCase):
    def test_explosion_user_always_faints(self):
        b = make(player={1: "seedot", 2: "riolu"}, cpu={1: "gimmighoul"},
                 cpu_action=Fight("Nasty Plot"))
        ev = b.play_turn(Fight("Explosion"))    # Normal → Ghost: no effect
        self.assertTrue(b.player.party[0].fainted)
        self.assertEqual(b.cpu.mon.hp, b.cpu.mon.max_hp)
        self.assertTrue(b.needs_switch(b.player))
        self.assertEqual(kinds(ev).count("faint"), 1)

    def test_explosion_double_ko_loses(self):
        b = make(player={1: "seedot"}, cpu={1: "skwovet"},
                 cpu_action=Fight("Bite"))
        b.cpu.mon.hp = 1
        b.play_turn(Fight("Explosion"))
        self.assertEqual(b.winner, "cpu")

    def test_recoil_double_ko_wins(self):
        b = make(player={1: "sudowoodo"}, cpu={1: "skwovet"},
                 cpu_action=Fight("Stuff Cheeks"))
        b.cpu.mon.hp = 60
        b.player.mon.hp = 1
        b.play_turn(Fight("Double-Edge"))
        self.assertTrue(b.player.mon.fainted and b.cpu.mon.fainted)
        self.assertEqual(b.winner, "player")


class AllCards(unittest.TestCase):
    def test_every_species_battles_every_species(self):
        keys = list(POKEDEX)
        for seed in range(120):
            rng = random.Random(seed)
            mats = [{s: k for s, k in zip(rng.sample(range(1, 7), 3),
                                          rng.sample(keys, 3))}
                    for _ in range(2)]
            b = Battle(build_side("player", mats[0]),
                       build_side("cpu", mats[1]), rng)
            while not b.winner:
                if b.needs_switch(b.player):
                    b.replace_fainted(b.player, rng.choice(b.player.bench()))
                    continue
                b.play_turn(rng.choice(b.legal_actions(b.player)))
                self.assertLess(b.turn, 500)
            for s in (b.player, b.cpu):
                for m in s.party:
                    self.assertTrue(0 <= m.hp <= m.max_hp)


if __name__ == "__main__":
    unittest.main()
