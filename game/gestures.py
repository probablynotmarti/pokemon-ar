"""Hand landmarks → cursor + button events (GAME_DESIGN.md §6).

"Hand position is the cursor, hand shape is the button":
    open hand → move cursor · pinch → "confirm" (A) · fist held → "back" (B)
    peace held → "info"

Pure Python: takes MediaPipe's 21 normalized landmarks as (x, y) or
(x, y, z) tuples, so it's testable without a camera. All thresholds
live in GESTURE_CONFIG.
"""

import math
from collections import deque
from dataclasses import dataclass, field

GESTURE_CONFIG = {
    "aspect": 16 / 9,         # camera width/height, so distances are square
    "mirror": True,           # selfie view: move right → cursor right
    "box": (0.2, 0.2, 0.8, 0.8),  # region of the frame mapped to the menu
    "pinch_on": 0.25,         # thumb–index gap / hand size to start a pinch
    "pinch_off": 0.40,        # …and to end it (hysteresis)
    "finger_up": 1.15,        # tip-to-wrist / pip-to-wrist above → extended
    "finger_down": 1.0,       # …below → curled
    "hold_s": 0.4,            # fist / peace must be held this long
    "pinch_lookback_s": 0.10, # confirm uses where the cursor was this long ago
    "filter_min_cutoff": 1.2, # One-Euro: lower = steadier when still
    "filter_beta": 8.0,       # One-Euro: higher = snappier when moving
}

# MediaPipe hand landmark indices
WRIST, THUMB_TIP = 0, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
FINGERS = {            # name: (pip, tip)
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
INDEX_TIP = 8


def _dist(a, b, aspect):
    return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])


def hand_size(lm, aspect):
    return _dist(lm[WRIST], lm[MIDDLE_MCP], aspect) or 1e-6


def finger_states(lm, cfg=GESTURE_CONFIG):
    """{"index": "up" | "down" | "mid", ...}"""
    a, wrist, out = cfg["aspect"], lm[WRIST], {}
    for name, (pip, tip) in FINGERS.items():
        ratio = _dist(lm[tip], wrist, a) / (_dist(lm[pip], wrist, a) or 1e-6)
        out[name] = ("up" if ratio > cfg["finger_up"]
                     else "down" if ratio < cfg["finger_down"] else "mid")
    return out


def pinch_ratio(lm, cfg=GESTURE_CONFIG):
    a = cfg["aspect"]
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP], a) / hand_size(lm, a)


def classify(lm, was_pinching=False, cfg=GESTURE_CONFIG):
    """One frame → "fist" | "pinch" | "peace" | "open"."""
    f = finger_states(lm, cfg)
    if all(s == "down" for s in f.values()):
        return "fist"
    limit = cfg["pinch_off"] if was_pinching else cfg["pinch_on"]
    if pinch_ratio(lm, cfg) < limit:
        return "pinch"
    if (f["index"] == f["middle"] == "up"
            and f["ring"] == f["pinky"] == "down"):
        return "peace"
    return "open"


def palm_center(lm):
    pts = [lm[i] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)]
    return (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4)


def to_cursor(point, cfg=GESTURE_CONFIG):
    """Camera coords → menu coords in [0,1]², mirrored, box-stretched."""
    x, y = point
    if cfg["mirror"]:
        x = 1 - x
    x0, y0, x1, y1 = cfg["box"]
    return (min(1, max(0, (x - x0) / (x1 - x0))),
            min(1, max(0, (y - y0) / (y1 - y0))))


class OneEuro:
    """One-Euro filter (Casiez et al. 2012) for one scalar."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, t):
        if self.t is None or t <= self.t:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


@dataclass
class GestureFrame:
    cursor: tuple | None          # menu coords, None when no hand
    pose: str | None              # fist | pinch | peace | open | None
    events: list = field(default_factory=list)   # confirm | back | info
    confirm_cursor: tuple | None = None          # cursor to use for confirm


class GestureTracker:
    """Feed one frame at a time; get cursor + edge-triggered events."""

    def __init__(self, cfg=None):
        self.cfg = {**GESTURE_CONFIG, **(cfg or {})}
        self._reset()

    def _reset(self):
        c = self.cfg
        self.fx = OneEuro(c["filter_min_cutoff"], c["filter_beta"])
        self.fy = OneEuro(c["filter_min_cutoff"], c["filter_beta"])
        self.pinching = False
        self.hold_pose = None       # fist/peace currently being held
        self.hold_since = 0.0
        self.hold_fired = False
        self.history = deque()      # (t, cursor)

    def update(self, landmarks, t):
        if landmarks is None:       # hand left the frame: forget everything
            self._reset()
            return GestureFrame(None, None)
        c = self.cfg
        raw = to_cursor(palm_center(landmarks), c)
        cursor = (self.fx(raw[0], t), self.fy(raw[1], t))
        self.history.append((t, cursor))
        while self.history and t - self.history[0][0] > 1.0:
            self.history.popleft()

        pose = classify(landmarks, self.pinching, c)
        frame = GestureFrame(cursor, pose)

        now_pinching = pose == "pinch"
        if now_pinching and not self.pinching:
            frame.events.append("confirm")
            frame.confirm_cursor = self._cursor_at(t - c["pinch_lookback_s"])
        self.pinching = now_pinching

        if pose in ("fist", "peace"):
            if pose != self.hold_pose:
                self.hold_pose, self.hold_since = pose, t
                self.hold_fired = False
            elif not self.hold_fired and t - self.hold_since >= c["hold_s"]:
                frame.events.append("back" if pose == "fist" else "info")
                self.hold_fired = True
        else:
            self.hold_pose = None
        return frame

    def _cursor_at(self, t):
        best = self.history[0][1]
        for ts, cur in self.history:
            if ts <= t:
                best = cur
        return best
