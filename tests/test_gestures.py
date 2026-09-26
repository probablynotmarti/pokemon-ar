import unittest

from game.gestures import (GestureTracker, classify, finger_states,
                           pinch_ratio, to_cursor)
from tests.hands import fist, open_hand, peace, pinch

FPS = 30


def run(tracker, frames, t0=0.0):
    """frames: list of landmark lists (or None). Returns GestureFrames."""
    return [tracker.update(lm, t0 + i / FPS) for i, lm in enumerate(frames)]


def events(results):
    return [e for r in results for e in r.events]


class Classify(unittest.TestCase):
    def test_poses(self):
        self.assertEqual(classify(open_hand()), "open")
        self.assertEqual(classify(fist()), "fist")
        self.assertEqual(classify(peace()), "peace")
        self.assertEqual(classify(pinch()), "pinch")

    def test_distance_from_camera_doesnt_matter(self):
        for scale in (0.5, 1.0, 1.8):
            self.assertEqual(classify(open_hand(scale=scale)), "open")
            self.assertEqual(classify(pinch(scale=scale)), "pinch")
            self.assertEqual(classify(fist(scale=scale)), "fist")

    def test_finger_states(self):
        f = finger_states(peace())
        self.assertEqual((f["index"], f["middle"], f["ring"], f["pinky"]),
                         ("up", "up", "down", "down"))

    def test_pinch_hysteresis(self):
        lm = open_hand()
        # move the thumb so the gap sits between pinch_on and pinch_off
        tip = lm[8]
        size = pinch_ratio(pinch())  # tiny
        self.assertLess(size, 0.25)
        lm[4] = (tip[0] - 0.0175, tip[1])   # x is stretched by 16/9
        r = pinch_ratio(lm)
        self.assertTrue(0.25 < r < 0.40, r)
        self.assertEqual(classify(lm, was_pinching=False), "open")
        self.assertEqual(classify(lm, was_pinching=True), "pinch")


class Cursor(unittest.TestCase):
    def test_mirror_and_box(self):
        for got, want in zip(to_cursor((0.5, 0.5)), (0.5, 0.5)):
            self.assertAlmostEqual(got, want)
        x, _ = to_cursor((0.3, 0.5))        # hand on image-left…
        self.assertGreater(x, 0.5)          # …is the user's right
        self.assertEqual(to_cursor((0.0, 0.0)), (1, 0))  # clamped

    def test_filter_settles_on_target(self):
        tr = GestureTracker()
        res = run(tr, [open_hand(cx=0.5)] * 5 + [open_hand(cx=0.35)] * 30)
        self.assertAlmostEqual(res[-1].cursor[0],
                               to_cursor((0.35 - 0.0, 0))[0], delta=0.03)

    def test_no_hand_means_no_cursor(self):
        tr = GestureTracker()
        self.assertIsNone(run(tr, [None])[0].cursor)


class Events(unittest.TestCase):
    def test_pinch_fires_once_until_released(self):
        tr = GestureTracker()
        res = run(tr, [open_hand()] * 3 + [pinch()] * 20 + [open_hand()] * 3
                  + [pinch()] * 3)
        self.assertEqual(events(res), ["confirm", "confirm"])

    def test_fist_must_be_held(self):
        tr = GestureTracker()
        short = run(tr, [open_hand()] + [fist()] * 5 + [open_hand()])
        self.assertEqual(events(short), [])             # 0.17 s: too short
        long = run(tr, [fist()] * 40, t0=1.0)           # 1.3 s: one back
        self.assertEqual(events(long), ["back"])

    def test_peace_is_info(self):
        tr = GestureTracker()
        self.assertEqual(events(run(tr, [peace()] * 20)), ["info"])

    def test_confirm_uses_cursor_from_before_the_pinch(self):
        tr = GestureTracker()
        still = [open_hand(cx=0.5)] * 20
        # pinching drags the hand sideways; confirm should ignore the drift
        drift = [pinch(cx=0.5 - 0.02 * i) for i in range(1, 4)]
        res = run(tr, still + drift)
        hit = next(r for r in res if "confirm" in r.events)
        self.assertAlmostEqual(hit.confirm_cursor[0], 0.5, delta=0.02)

    def test_losing_the_hand_releases_pinch(self):
        tr = GestureTracker()
        res = run(tr, [pinch()] * 3 + [None] + [pinch()] * 3)
        self.assertEqual(events(res), ["confirm", "confirm"])


if __name__ == "__main__":
    unittest.main()
