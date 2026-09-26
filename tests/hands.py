"""Synthetic MediaPipe hand landmarks for tests (palm facing camera).

Coordinates are normalized image coords; the hand is roughly the size a
MacBook camera sees at arm's length.
"""

S = 0.06                 # scale: wrist → middle knuckle ≈ 0.06 * 1.6
X_OFF = {"index": -0.9, "middle": -0.3, "ring": 0.3, "pinky": 0.9}
BASE = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def hand(cx=0.5, cy=0.6, up=("index", "middle", "ring", "pinky"),
         pinch=False, scale=1.0):
    s = S * scale
    lm = [None] * 21
    lm[0] = (cx, cy)                                   # wrist
    for name, (mcp, pip, dip, tip) in BASE.items():
        x = cx + X_OFF[name] * s * 0.6
        lm[mcp] = (x, cy - 1.6 * s)
        if name in up:                                 # straight up
            lm[pip], lm[dip], lm[tip] = ((x, cy - 2.3 * s),
                                         (x, cy - 2.8 * s),
                                         (x, cy - 3.2 * s))
        else:                                          # curled into palm
            lm[pip], lm[dip], lm[tip] = ((x, cy - 2.0 * s),
                                         (x, cy - 1.5 * s),
                                         (x, cy - 1.1 * s))
    # thumb sticks out to the side
    lm[1], lm[2], lm[3] = ((cx - 0.8 * s, cy - 0.4 * s),
                           (cx - 1.4 * s, cy - 0.9 * s),
                           (cx - 1.8 * s, cy - 1.4 * s))
    lm[4] = (cx - 2.1 * s, cy - 1.9 * s)
    if pinch:                                          # thumb tip meets index tip
        lm[4] = (lm[8][0] - 0.1 * s, lm[8][1] + 0.1 * s)
    return lm


def open_hand(**kw):
    return hand(**kw)


def fist(**kw):
    return hand(up=(), **kw)


def peace(**kw):
    return hand(up=("index", "middle"), **kw)


def pinch(**kw):
    return hand(pinch=True, **kw)
