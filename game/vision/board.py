"""The play surface: the plain back of a Monopoly board (≈508 mm square,
brown cardboard with a fold cross), filmed from above by the phone.

Board frame (millimetres): origin at the board corner that's top-left in
the phone image, x right, y down. The far half (top of the image) is the
CPU's, the near half is yours, split at the horizontal fold. Each half is
cut into a 3x2 grid of zones; a card belongs to the zone its centre is in.
Slots are numbered in reading order as seen from your seat (1 = top-left,
nearest the CPU for the CPU half, nearest the fold for yours).

No printed markers: the board is found as the big uniform region in the
middle of the frame; cards are found on it by their bright silver border.
"""

import cv2
import numpy as np

BOARD_MM = 508.0                     # 20 in; exact size barely matters
HALF = {"cpu": 0.0, "player": BOARD_MM / 2}      # y where each half starts
COLS, ROWS = 3, 2
ZONE_MM = (BOARD_MM / COLS, BOARD_MM / 2 / ROWS)  # ≈169 x 127
CARD_MM = (63.0, 88.0)
TOP_PX_PER_MM = 2.0                  # resolution of the straightened board

BOARD_CONFIG = {
    "board_dist": 22.0,     # Lab distance still counted as "board colour"
    "min_board_frac": 0.08, # board must cover this much of the frame
    "board_L": (70, 235),   # cardboard brown: not too dark / bright (Lab L)…
    "board_min_b": 8.0,     # …and yellowish (Lab b − 128); the user's
                            # Monopoly back measured L≈157, b≈+21
    "card_bright": 20.0,    # card border: this much brighter than board…
    "card_chroma": 15.0,    # …or this different in colour
    "card_detail": 2.0,     # …and near fine detail (glare/shadow have none)
    "card_area": (0.55, 1.6),   # × a card's area
    "card_aspect": (0.58, 0.86),
    "zone_texture": 3.5,    # below this a zone is plain: skip the search
}                           # above this a zone "has something in it"


def board_corners_mm():
    s = BOARD_MM
    return np.float32([[0, 0], [s, 0], [s, s], [0, s]])


def zone_rect_mm(side, slot):
    c, r = (slot - 1) % COLS, (slot - 1) // COLS
    return (c * ZONE_MM[0], HALF[side] + r * ZONE_MM[1], *ZONE_MM)


def zone_quad_mm(side, slot):
    x, y, w, h = zone_rect_mm(side, slot)
    return np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def zone_of(center_mm):
    """(side, slot) for a point on the board, or None if off the board."""
    x, y = center_mm
    if not (0 <= x < BOARD_MM and 0 <= y < BOARD_MM):
        return None
    side = "cpu" if y < BOARD_MM / 2 else "player"
    c = min(COLS - 1, int(x // ZONE_MM[0]))
    r = min(ROWS - 1, int((y - HALF[side]) // ZONE_MM[1]))
    return side, r * COLS + c + 1


def _order(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()],
                       pts[d.argmax()]])


def find_board(frame, cfg=BOARD_CONFIG):
    """Image corners of the board (TL, TR, BR, BL), or None."""
    return detect_board(frame, cfg)[0]


def dominant_colour(lab):
    """Most common colour (Lab) in a region. Cardboard is one tight colour
    cluster; cards on top of it are spread over many colours, so this
    stays the board's colour even when the middle is full of cards."""
    px = lab.reshape(-1, 3)
    q = (px // [8, 4, 4]).astype(np.int32)
    keys = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[np.argmax(counts)]
    near = np.abs(q - [top >> 16, (top >> 8) & 255, top & 255]).max(1) <= 1
    return px[near].mean(0)


def detect_board(frame, cfg=BOARD_CONFIG):
    """→ (corners or None, partial). partial = the board runs off the edge
    of the frame, so its corners (and every zone) can't be trusted."""
    h, w = frame.shape[:2]
    scale = 800 / max(h, w)                      # work small: it's big & flat
    small = cv2.resize(frame, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 2),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    sh, sw = lab.shape[:2]
    centre = lab[int(sh * .2):int(sh * .8), int(sw * .2):int(sw * .8)]
    ref = dominant_colour(centre)
    lo, hi = cfg["board_L"]
    if not (lo <= ref[0] <= hi and ref[2] - 128 >= cfg["board_min_b"]):
        return None, False              # middle of the view isn't cardboard
    dist = np.sqrt((((lab - ref) * [0.5, 1, 1]) ** 2).sum(-1))
    mask = (dist < cfg["board_dist"]).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # drop thin slivers (e.g. the fuzzy rim of a shadow) that would stretch
    # the outline; the board itself is far thicker than this
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, False
    # the component covering the frame centre (or else the biggest)
    comp = labels[sh // 2, sw // 2]
    if comp == 0:
        comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == comp).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    if cv2.contourArea(hull) < cfg["min_board_frac"] * sh * sw:
        return None, False
    if cv2.contourArea(hull) > 0.97 * sh * sw:
        return None, False              # the whole frame: no board in view
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)   # every edge pixel
    quad, partial = _corners_from_edges(max(contours, key=len), sw, sh)
    if quad is not None:
        # real board edges enclose the whole visible board; a "fourth edge"
        # made of card tops (near edge off-screen) leaves board beyond it
        inside = np.zeros_like(blob)
        cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
        inside = cv2.dilate(inside, np.ones((9, 9), np.uint8))
        outside = np.count_nonzero(blob & ~inside) / max(1, np.count_nonzero(blob))
        if outside > 0.015:
            quad, partial = None, True
    if quad is None:
        return None, partial
    return quad / scale, partial


def _ransac_line(pts, rng, iters=300, tol=2.0):
    """Best-supported straight line through a point set → inlier mask."""
    best = None
    n = len(pts)
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        d = pts[j] - pts[i]
        norm = np.hypot(*d)
        if norm < 10:
            continue
        nrm = np.array([-d[1], d[0]]) / norm
        inl = np.abs((pts - pts[i]) @ nrm) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


def _fit(pts):
    vx, vy, x0, y0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01,
                                 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (p, d), (q, e) = l1, l2
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-6:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def _corners_from_edges(contour, w, h, border=3, min_frac=0.03):
    """Board corners = intersections of its 4 straight edges, found one by
    one (RANSAC) along the outline, ignoring outline that is just the
    frame's own edge. A corner a little off-frame is recovered; only a
    whole missing edge makes the view 'partial'.
    → (corners TL,TR,BR,BL or None, partial)"""
    pts = contour.reshape(-1, 2).astype(np.float32)
    on_frame = ((pts[:, 0] <= border) | (pts[:, 1] <= border) |
                (pts[:, 0] >= w - 1 - border) | (pts[:, 1] >= h - 1 - border))
    total = len(pts)
    pts = pts[~on_frame]
    rng = np.random.default_rng(0)                  # deterministic
    lines, groups = [], []
    rest = pts
    for _ in range(4):
        if len(rest) < 20:
            break
        inl = _ransac_line(rest, rng)
        if inl is None or inl.sum() < min_frac * total:
            break
        line = _fit(rest[inl])
        lines.append(line)
        groups.append(rest[inl].mean(0))
        p0, d = line                                 # remove this edge's points
        nrm = np.array([-d[1], d[0]])
        rest = rest[np.abs((rest - p0) @ nrm) > 5]
    if len(lines) < 4:
        return None, True
    c = np.mean(groups, 0)                           # order edges round the board
    order = np.argsort([np.arctan2(g[1] - c[1], g[0] - c[0]) for g in groups])
    lines = [lines[i] for i in order]
    corners = []
    for i in range(4):
        a, b = lines[i - 1], lines[i]
        if abs(float(a[1] @ b[1])) > 0.9:            # neighbours ~parallel
            return None, True
        corners.append(_intersect(a, b))
    corners = np.float32(corners)
    if np.abs(corners).max() > 3 * max(w, h) or \
            not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return None, True
    return _order(corners), False


def board_background(lab):
    """Local board colour: a median over ~170 mm windows at 1/16 scale.
    Follows glare, shadows and uneven light; a card is too small to pull
    the median, so it stands out against this."""
    h, w = lab.shape[:2]
    small = cv2.resize(lab, (max(8, w // 16), max(8, h // 16)),
                       interpolation=cv2.INTER_AREA)
    k = max(3, int(170 * TOP_PX_PER_MM / 16) | 1)
    bg = np.dstack([cv2.medianBlur(np.ascontiguousarray(small[..., i]), k)
                    for i in range(3)])
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def card_mask(top, cfg=BOARD_CONFIG):
    """Pixels of a straightened board image that belong to cards."""
    lab8 = cv2.cvtColor(cv2.GaussianBlur(top, (0, 0), 1.2), cv2.COLOR_BGR2LAB)
    d = lab8.astype(np.float32) - board_background(lab8).astype(np.float32)
    border = (d[..., 0] > cfg["card_bright"]) | \
        (np.hypot(d[..., 1], d[..., 2]) > cfg["card_chroma"])
    # only where there's fine detail nearby: card art and the crisp border
    # have it, glare and shadows (smooth) don't — stops glare bridging cards
    L = lab8[..., 0].astype(np.float32)
    detail = cv2.GaussianBlur(np.abs(L - cv2.GaussianBlur(L, (0, 0), 3)),
                              (0, 0), 4)
    near_detail = cv2.dilate((detail > cfg["card_detail"]).astype(np.uint8),
                             np.ones((9, 9), np.uint8)) > 0
    m = (border & near_detail).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # fill each outline (convex) so dark card art inside the border counts
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    for c in contours:
        cv2.fillConvexPoly(filled, cv2.convexHull(c), 255)
    return cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))


def portrait_corners(box):
    """minAreaRect points → TL, TR, BR, BL with the SHORT side on top, so
    the straightened crop is card-shaped (portrait) whatever the rotation."""
    box = _order(box)
    top = np.linalg.norm(box[1] - box[0])
    side = np.linalg.norm(box[3] - box[0])
    if top > side:                               # lying sideways: rotate
        box = np.roll(box, -1, axis=0)
    return box


def find_cards(top, px_per_mm=TOP_PX_PER_MM, cfg=BOARD_CONFIG):
    """Card quads (mm, TL/TR/BR/BL portrait) on a straightened board."""
    mask = card_mask(top, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    card_px = CARD_MM[0] * CARD_MM[1] * px_per_mm ** 2
    lo, hi = cfg["card_area"]
    alo, ahi = cfg["card_aspect"]
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not lo * card_px <= area <= hi * card_px:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if min(w, h) / max(w, h) < alo or min(w, h) / max(w, h) > ahi:
            continue
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        out.append(portrait_corners(box) / px_per_mm)
    return out


def zone_texture(zone_img):
    """How much fine detail a zone has, ignoring smooth glare/shadow."""
    L = cv2.cvtColor(zone_img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    smooth = cv2.GaussianBlur(L, (0, 0), 12)
    return float((L - smooth).std())


def find_card_in_zone(zone_img, px_per_mm, min_fill=0.5):
    """Inside one zone: the card is a card-shaped patch that's detailed all
    the way through (art, text). Works when its colour is close to the
    board's (orange cards). Shadows, folds, stains and grain don't qualify:
    a shadow is only detailed along its outline, a fold is thin, grain is
    too faint after pre-smoothing. → card quad (zone mm) or None."""
    L = cv2.cvtColor(zone_img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    L = cv2.GaussianBlur(L, (0, 0), 1.5)                 # drop cardboard grain
    detail = cv2.GaussianBlur(np.abs(L - cv2.GaussianBlur(L, (0, 0), 12)),
                              (0, 0), 3)
    raw = (detail > 4.0).astype(np.uint8)
    m = cv2.morphologyEx(raw * 255, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((labels == big).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    card_px = CARD_MM[0] * CARD_MM[1] * px_per_mm ** 2
    if not 0.45 * card_px <= cv2.contourArea(hull) <= 1.7 * card_px:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
    if not 0.55 <= min(w, h) / max(w, h) <= 0.9:
        return None
    inside = np.zeros_like(raw)
    cv2.fillConvexPoly(inside, hull, 1)
    if raw[inside > 0].mean() < min_fill:               # hollow = shadow
        return None
    return portrait_corners(cv2.boxPoints(((cx, cy), (w, h), ang))) / px_per_mm
