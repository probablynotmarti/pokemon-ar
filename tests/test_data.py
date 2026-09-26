import unittest

from game.data import (POKEDEX, MOVES, ITEMS, STARTING_BAG, TYPE_CHART,
                       calc_stats, effectiveness)
from game.damage import damage_range, heavy_slam_power


class DataIntegrity(unittest.TestCase):
    def test_every_pokemon_has_four_known_unique_moves(self):
        for s in POKEDEX.values():
            self.assertEqual(len(s.moves), 4, s.name)
            self.assertEqual(len(set(s.moves)), 4, s.name)
            for m in s.moves:
                self.assertIn(m, MOVES, f"{s.name}: {m}")

    def test_every_move_is_used(self):
        used = {m for s in POKEDEX.values() for m in s.moves}
        self.assertEqual(used, set(MOVES))

    def test_types_are_valid(self):
        for s in POKEDEX.values():
            for t in s.types:
                self.assertIn(t, TYPE_CHART)
        for m in MOVES.values():
            self.assertIn(m.type, TYPE_CHART)
        for row in TYPE_CHART.values():
            for t in row:
                self.assertIn(t, TYPE_CHART)

    def test_bag_items_exist(self):
        for name in STARTING_BAG:
            self.assertIn(name, ITEMS)


class GameFormulas(unittest.TestCase):
    def test_level50_stats_match_games(self):
        # Known in-game L50 values (31 IV / 0 EV / neutral).
        self.assertEqual(calc_stats(POKEDEX["registeel"]),
                         {"hp": 155, "atk": 95, "def": 170, "spa": 95,
                          "spd": 170, "spe": 70})
        self.assertEqual(calc_stats(POKEDEX["politoed"])["hp"], 165)

    def test_effectiveness(self):
        self.assertEqual(effectiveness("Fighting", ("Steel",)), 2)
        self.assertEqual(effectiveness("Ice", ("Water", "Flying")), 1)
        self.assertEqual(effectiveness("Rock", ("Water", "Flying")), 2)
        self.assertEqual(effectiveness("Normal", ("Rock",)), .5)
        self.assertEqual(effectiveness("Psychic", ("Dark",)), 0)

    def test_heavy_slam_weight_bands(self):
        self.assertEqual(heavy_slam_power(205, 5.5), 120)
        self.assertEqual(heavy_slam_power(205, 205), 40)
        self.assertEqual(heavy_slam_power(205, 60), 80)

    def test_damage_range_sane(self):
        lo, hi = damage_range(POKEDEX["politoed"], POKEDEX["sudowoodo"],
                              "Hydro Pump")
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0)
        self.assertEqual(damage_range(POKEDEX["politoed"],
                                      POKEDEX["sudowoodo"], "Hypnosis"),
                         (0, 0))


if __name__ == "__main__":
    unittest.main()
