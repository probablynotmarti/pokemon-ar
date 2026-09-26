import unittest

from game.battle import Battle, Fight, build_side
from game.menu import Menu
from tests.test_battle import Rigged, fixed


def setup(player=None, cpu=None, cpu_action=Fight("Hyper Voice")):
    b = Battle(build_side("player", player or {1: "politoed", 2: "registeel",
                                               5: "ducklett"}),
               build_side("cpu", cpu or {1: "politoed", 2: "skwovet"}),
               Rigged(), cpu_brain=fixed(cpu_action))
    return b, Menu(b)


class Clock:
    """Advance time past the cooldown between every input."""
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def pick(menu, clock, tile_id):
    menu.hovered = tile_id
    return menu.confirm(clock())


class Navigation(unittest.TestCase):
    def test_fight_a_turn_then_resume(self):
        b, m = setup()
        c = Clock()
        self.assertEqual([t.label for t in m.tiles()],
                         ["FIGHT", "BAG", "POKÉMON", "RUN"])
        pick(m, c, "fight")
        self.assertEqual(m.screen, "fight")
        self.assertEqual(len(m.tiles()), 4)
        ev = pick(m, c, "move:Ice Beam")
        self.assertTrue(ev)
        self.assertEqual(m.screen, "locked")
        self.assertEqual(m.tiles(), [])
        self.assertIsNone(pick(m, c, "fight"))     # locked while animating
        m.resume(c())
        self.assertEqual(m.screen, "main")
        self.assertEqual(b.turn, 1)

    def test_back_goes_up_one_level(self):
        b, m = setup()
        c = Clock()
        pick(m, c, "bag")
        self.assertTrue(m.back(c()))
        self.assertEqual(m.screen, "main")
        self.assertFalse(m.back(c()))              # nothing above main

    def test_bag_targets_are_filtered(self):
        b, m = setup()
        c = Clock()
        b.player.party[1].hp = 10                  # hurt Registeel (bench)
        pick(m, c, "bag")
        potion = m.tile("item:Potion")
        self.assertTrue(potion.enabled)
        self.assertFalse(m.tile("item:Revive").enabled)
        pick(m, c, "item:Potion")
        self.assertEqual(m.screen, "item_target")
        by_name = {t.label: t.enabled for t in m.tiles()}
        self.assertEqual(by_name, {"Politoed": False, "Registeel": True,
                                   "Ducklett": False})
        self.assertTrue(m.back(c()))
        self.assertEqual(m.screen, "bag")
        pick(m, c, "item:Potion")
        pick(m, c, "mon:1")
        self.assertEqual(b.player.party[1].hp, 30)

    def test_party_layout_mirrors_the_mat(self):
        b, m = setup()
        c = Clock()
        pick(m, c, "pokémon")
        rects = {t.label: t.rect for t in m.tiles()}
        # slot 5 = bottom row, middle column
        x, y, w, h = rects["Ducklett"]
        self.assertTrue(1 / 3 < x < 2 / 3 and y > 0.5)

    def test_switch_via_party_action(self):
        b, m = setup()
        c = Clock()
        pick(m, c, "pokémon")
        pick(m, c, "mon:0")                        # active one
        self.assertFalse(m.tile("switch").enabled)
        m.back(c())
        pick(m, c, "mon:1")
        pick(m, c, "switch")
        self.assertEqual(b.player.mon.name, "Registeel")

    def test_forced_switch_has_no_back(self):
        b, m = setup()
        c = Clock()
        b.player.mon.hp = 1
        pick(m, c, "fight")
        pick(m, c, "move:Hyper Voice")
        m.resume(c())
        self.assertEqual(m.screen, "forced_switch")
        self.assertFalse(m.back(c()))
        self.assertFalse(m.tile("mon:0").enabled)  # the fainted one
        pick(m, c, "mon:2")
        m.resume(c())
        self.assertEqual((m.screen, b.player.mon.name), ("main", "Ducklett"))

    def test_run_needs_confirmation(self):
        b, m = setup()
        c = Clock()
        pick(m, c, "run")
        pick(m, c, "no")
        self.assertEqual(m.screen, "main")
        pick(m, c, "run")
        pick(m, c, "yes")
        m.resume(c())
        self.assertEqual((m.screen, b.winner), ("game_over", "cpu"))


class InputFeel(unittest.TestCase):
    def test_cooldown_blocks_instant_double_pick(self):
        b, m = setup()
        m.hovered = "fight"
        m.confirm(10.0)
        m.hovered = "move:Ice Beam"
        self.assertIsNone(m.confirm(10.1))         # carried-over pinch
        self.assertEqual(m.screen, "fight")
        self.assertTrue(m.confirm(10.5))

    def test_hover_needs_dwell(self):
        b, m = setup()
        m.point((0.25, 0.25), 0.0)                 # FIGHT
        self.assertEqual(m.hovered, "fight")
        m.point((0.75, 0.25), 1.0)                 # flick to BAG…
        m.point((0.25, 0.25), 1.05)                # …and back: no change
        self.assertEqual(m.hovered, "fight")
        m.point((0.75, 0.25), 2.0)
        m.point((0.75, 0.25), 2.2)                 # held on BAG
        self.assertEqual(m.hovered, "bag")

    def test_confirm_with_lookback_cursor(self):
        b, m = setup()
        m.hovered = "bag"
        m.confirm(5.0, cursor=(0.2, 0.8))          # where the hand WAS
        self.assertEqual(m.screen, "party")

    def test_keyboard_moves(self):
        b, m = setup()
        m.hovered = "fight"
        m.move(1, 0)
        self.assertEqual(m.hovered, "bag")
        m.move(0, 1)
        self.assertEqual(m.hovered, "run")
        m.move(-1, 0)
        self.assertEqual(m.hovered, "pokémon")

    def test_info_lines(self):
        b, m = setup(cpu={1: "sudowoodo"})
        c = Clock()
        pick(m, c, "fight")
        m.hovered = "move:Hydro Pump"
        lines = m.info_lines()
        self.assertIn("Super effective", lines[2])


if __name__ == "__main__":
    unittest.main()
