"""Grid-camera tests: the real board photo + synthetic frames of cards on
the board. Needs OpenCV, so run with the camera venv:
    ~/.venvs/memecv/bin/python -m unittest tests.test_vision
(skipped under plain python3)."""
import pathlib
import unittest

try:
    import cv2
    import numpy as np
except ImportError:                        # plain python3: no OpenCV
    raise unittest.SkipTest("OpenCV not installed (use ~/.venvs/memecv)")

from game.vision.cards import CARDS_DIR  # noqa: E402

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"
if not list(CARDS_DIR.glob("*.jpg")) or not (FIXTURES / "board_empty.jpg").exists():
    raise unittest.SkipTest("needs your card reference photos (assets/cards) and "
                            "board photos (tests/fixtures) — personal files, "
                            "not in the repo; see README")

from game.vision.board import find_board, zone_of, zone_quad_mm
from game.vision.cards import EMPTY, UNKNOWN, CardIndex
from game.vision.reader import BoardReader, SlotVotes
from tests.scene import board_corner_px, photograph, table_image

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
BOARD = {"player": {1: "politoed", 2: "riolu", 3: "ducklett", 5: "seedot"},
         "cpu": {1: "sudowoodo", 2: "gimmighoul", 4: "solrock",
                 6: "marshadow"}}
_INDEX = None


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = CardIndex()
    return _INDEX


def read(frame, reader=None):
    r = reader or BoardReader(index())
    view = r.locate(frame)
    return view, r.identify(frame, view)


def keys(got):
    return {s: {n: k for n, (k, _) in slots.items()} for s, slots in got.items()}


def expected(board=BOARD):
    return {s: {n: board[s].get(n, EMPTY) for n in range(1, 7)}
            for s in ("player", "cpu")}


class Geometry(unittest.TestCase):
    def test_zones_in_reading_order_from_your_seat(self):
        # CPU half is the top of the image; slot 1 = top-left of each half
        self.assertEqual(zone_of(zone_quad_mm("cpu", 1).mean(0)), ("cpu", 1))
        self.assertEqual(zone_of((20, 20)), ("cpu", 1))
        self.assertEqual(zone_of((490, 240)), ("cpu", 6))
        self.assertEqual(zone_of((20, 270)), ("player", 1))   # by the fold
        self.assertEqual(zone_of((490, 500)), ("player", 6))
        self.assertIsNone(zone_of((-5, 100)))


class RealBoardPhoto(unittest.TestCase):
    """The user's own photo of the Monopoly-board back (no cards)."""

    def setUp(self):
        self.img = cv2.imread(str(FIXTURES / "board_empty.jpg"))

    def test_finds_the_board_corners(self):
        quad = find_board(self.img)
        self.assertIsNotNone(quad)
        want = np.float32([[0, 174], [883, 168], [781, 894], [116, 905]])
        self.assertLess(np.abs(quad - want).max(), 25, quad.round())

    def test_empty_board_reads_empty_despite_phone_shadow(self):
        view, got = read(self.img)
        self.assertTrue(view.found)
        self.assertEqual(keys(got), expected({"player": {}, "cpu": {}}))


REAL_PARTY = {"player": {1: "marshadow", 2: "solrock", 3: "seedot",
                         4: "riolu", 5: "gimmighoul", 6: "politoed"},
              "cpu": {3: "registeel", 4: "sudowoodo", 5: "ducklett",
                      6: "skwovet"}}


class RealLiveFrames(unittest.TestCase):
    """Frames from the user's iPhone and photos, 10 cards on the board.
    The CPU's cards lie upside down from the camera's point of view."""

    def load(self, name):
        return cv2.imread(str(FIXTURES / name))

    def test_live_frame_with_a_clipped_corner_reads_everything(self):
        # near-left board corner is off-frame: recovered from the edge lines
        view, got = read(self.load("live_corner_clipped.jpg"))
        self.assertFalse(view.partial)
        self.assertLess(view.quad[3][0], 0)          # corner left of frame
        self.assertEqual(keys(got), expected(REAL_PARTY))

    def test_live_frame_missing_the_near_edge_is_partial(self):
        # a whole edge is off-screen; card tops must not pass as an edge
        view, got = read(self.load("live_near_edge_missing.jpg"))
        self.assertTrue(view.partial)
        self.assertFalse(view.found)

    def test_photo_cut_off_and_unevenly_lit_is_partial_not_wrong(self):
        view, _ = read(self.load("board_cards_cutoff.jpg"))
        self.assertTrue(view.partial)

    def test_full_board_photo_is_not_partial(self):
        view, _ = read(self.load("board_empty.jpg"))
        self.assertFalse(view.partial)


class Reading(unittest.TestCase):
    def check(self, frame, board=BOARD, reader=None):
        view, got = read(frame, reader)
        self.assertTrue(view.found, "board not found")
        self.assertEqual(keys(got), expected(board))
        return view

    def test_1080p_views_with_glare_and_shadow(self):
        for seed in range(1, 7):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                self.check(photograph(table_image(BOARD, rng), rng,
                                      shadow=seed % 2 == 0)[0])

    def test_720p_steep_never_misreads(self):
        """Tiny, blurry far cards may come out '?' or be missed (the setup
        screen shows no label, so you'd see it) — but a card is never
        read as the wrong Pokémon, and no Pokémon appears from nothing."""
        right = total = 0
        for seed in range(9, 13):
            rng = np.random.default_rng(seed)
            frame, _ = photograph(table_image(BOARD, rng), rng, tilt=0.4,
                                  out=(1280, 720), blur=1.4, noise=8, jpeg=70)
            got = keys(read(frame)[1])
            for s in BOARD:
                for n in range(1, 7):
                    want = BOARD[s].get(n, EMPTY)
                    self.assertIn(got[s][n], (want, UNKNOWN, EMPTY),
                                  (seed, s, n))
                    if n in BOARD[s]:
                        total += 1
                        right += got[s][n] == want
        self.assertGreaterEqual(right / total, 0.7, f"{right}/{total}")

    def test_hand_over_a_board_corner_keeps_last_position(self):
        rng = np.random.default_rng(3)
        table = table_image(BOARD, rng)
        clean, M = photograph(table, np.random.default_rng(3))
        reader = BoardReader(index())
        reader.locate(clean)
        x, y = board_corner_px(M, 1)                  # top-right corner
        covered, _ = photograph(table, np.random.default_rng(3),
                                occlude=[(x, y, 90)])
        view = self.check(covered, reader=reader)
        self.assertTrue(view.stale)

    def test_board_really_moved_is_followed(self):
        rng = np.random.default_rng(7)
        table = table_image(BOARD, rng)
        first, _ = photograph(table, np.random.default_rng(1))
        moved, _ = photograph(table, np.random.default_rng(1), tilt=0.35)
        reader = BoardReader(index())
        q0 = reader.locate(first).quad.copy()
        views = [reader.locate(moved) for _ in range(reader.move_frames)]
        self.assertTrue(views[0].stale)                # 1 frame: ignored
        self.assertFalse(views[-1].stale)              # held: accepted
        self.assertGreater(np.abs(views[-1].quad - q0).max(), 20)
        self.check(moved, reader=reader)

    def test_upside_down_card(self):
        rng = np.random.default_rng(4)
        self.check(photograph(table_image(BOARD, rng,
                                          upside_down={("player", 2)}),
                              rng)[0])

    def test_every_card_every_zone(self):
        keys_ = ["registeel", "sudowoodo", "ducklett", "skwovet", "politoed",
                 "marshadow", "gimmighoul", "riolu", "seedot", "solrock"]
        rng = np.random.default_rng(5)
        board = {"player": dict(zip(range(1, 7), keys_[:6])),
                 "cpu": dict(zip(range(1, 5), keys_[6:]))}
        self.check(photograph(table_image(board, rng), rng)[0], board)

    def test_flip_swaps_halves(self):
        rng = np.random.default_rng(1)
        frame, _ = photograph(table_image(BOARD, rng), rng)
        _, got = read(frame, BoardReader(index(), flip=True))
        got = keys(got)
        for n, key in BOARD["player"].items():       # 180°: slot n → 7-n
            self.assertEqual(got["cpu"][7 - n], key)
        for n, key in BOARD["cpu"].items():
            self.assertEqual(got["player"][7 - n], key)

    def test_no_board_in_view(self):
        view, got = read(np.full((720, 1280, 3), 235, np.uint8))
        self.assertEqual((view.found, got), (False, {}))


class UnknownCards(unittest.TestCase):
    def test_card_without_reference_is_unknown_not_misread(self):
        import shutil
        import tempfile
        from tests.scene import CARDS
        rng = np.random.default_rng(6)
        frame, _ = photograph(table_image(BOARD, rng), rng)
        wrong = []
        for hidden in ("riolu", "seedot", "marshadow", "sudowoodo"):
            with tempfile.TemporaryDirectory() as tmp:
                for f in CARDS.glob("*.jpg"):
                    if f.stem != hidden:
                        shutil.copy(f, tmp)
                _, got = read(frame, BoardReader(CardIndex(tmp)))
                for s, slots in keys(got).items():
                    for n, key in slots.items():
                        if BOARD[s].get(n) == hidden and key != UNKNOWN:
                            wrong.append((hidden, s, n, key))
        self.assertEqual(wrong, [])


class Votes(unittest.TestCase):
    def test_majority_smooths_flicker(self):
        v = SlotVotes(window=5, min_votes=3)
        for key in ("riolu", "riolu", "unknown", "riolu", "empty"):
            v.add({"player": {2: (key, 0)}})
        self.assertEqual(v.party("player"), {2: "riolu"})

    def test_needs_enough_votes(self):
        v = SlotVotes(window=5, min_votes=3)
        v.add({"player": {1: ("seedot", 0)}})
        self.assertEqual(v.party("player"), {})


if __name__ == "__main__":
    unittest.main()
