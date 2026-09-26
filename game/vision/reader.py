"""Grid camera: frame → where the board is → which card is in which slot.

Per zone: (1) a clean card shape (silver border) found by segmentation is
cut out and recognized; (2) otherwise, if the zone has any detail, search
the whole zone for a known card; (3) else cut out a card-shaped patch
that's detailed all through and recognize that; (4) a card shape we can't
recognize is "unknown", anything else (grain, folds, shadows) is empty.

    reader = BoardReader(CardIndex())
    view = reader.locate(frame)          # fast: find the board
    slots = reader.identify(frame, view) # slower: find + recognize cards
    votes.add(slots); votes.party("player")  -> {slot: key}

The phone is fixed above the table, so the board doesn't move: when a hand
covers a corner (or detection wobbles), the last good corners are kept.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import (BOARD_CONFIG, BOARD_MM, CARD_MM, TOP_PX_PER_MM,
                    board_corners_mm, detect_board, find_card_in_zone,
                    find_cards, zone_of, zone_quad_mm, zone_rect_mm,
                    zone_texture)
from .cards import EMPTY, UNKNOWN
from .geometry import REF_SIZE

SIDES = ("player", "cpu")


@dataclass
class BoardView:
    quad: np.ndarray | None = None        # board corners in the image
    H: np.ndarray | None = None           # board mm → image px
    cards: dict = field(default_factory=dict)   # (side, slot) → quad mm
    stale: bool = False                   # corners reused from earlier
    partial: bool = False                 # board runs off the frame edge

    @property
    def found(self):
        return self.H is not None

    def project(self, pts_mm):
        pts = np.asarray(pts_mm, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def zone_polygon(self, side, slot):
        return self.project(zone_quad_mm(side, slot))

    def card_polygon(self, side, slot):
        """Where the card in this slot is (last seen), else the zone's
        middle, card-sized."""
        q = self.cards.get((side, slot))
        if q is None:
            x, y, w, h = cv2.boundingRect(zone_quad_mm(side, slot))
            cx, cy = x + w / 2, y + h / 2
            hw, hh = CARD_MM[0] / 2, CARD_MM[1] / 2
            q = np.float32([[cx - hw, cy - hh], [cx + hw, cy - hh],
                            [cx + hw, cy + hh], [cx - hw, cy + hh]])
        return self.project(q)

    def outline(self):
        return self.project(board_corners_mm())


class BoardReader:
    def __init__(self, card_index=None, flip=False, move_tol=0.02,
                 move_frames=12, detect_every_s=0.0):
        self.cards = card_index
        self.flip = flip              # phone mounted the other way round
        self.move_tol = move_tol      # corner wobble allowed (× board size)
        self.move_frames = move_frames  # a real move must hold this long
        self.detect_every_s = detect_every_s  # live: board is static, so
        self.reset()                          # re-find it only this often

    def locate(self, frame, now=None):
        """Board position for this frame. With detect_every_s > 0 (live
        camera) the slow edge-fitting only runs that often; in between the
        last result is reused — the phone and board don't move."""
        if self.detect_every_s and now is not None and self._view_cache and \
                now - self._detect_t < self.detect_every_s:
            return self._view_cache
        view = self._locate(frame)
        if now is not None:
            self._detect_t, self._view_cache = now, view
        return view

    def _locate(self, frame):
        """The phone is fixed, so the board only 'moves' if a changed
        position holds for move_frames frames. A hand passing over a corner
        (skin is close to cardboard brown) never gets that far."""
        quad, partial = detect_board(frame)
        stale = True
        if self._last is None:
            if quad is not None:
                self._last, stale = quad, False
        elif quad is not None:
            size = float(np.linalg.norm(self._last[2] - self._last[0]))
            tol = self.move_tol * size
            if np.abs(quad - self._last).max() <= tol:
                self._last = 0.7 * self._last + 0.3 * quad      # smooth
                self._pending, self._pending_n, stale = None, 0, False
            else:
                if self._pending is not None and \
                        np.abs(quad - self._pending).max() <= tol:
                    self._pending_n += 1
                else:
                    self._pending, self._pending_n = quad, 1
                if self._pending_n >= self.move_frames:     # really moved
                    self._last, stale = self._pending, False
                    self._pending, self._pending_n = None, 0
                    self._positions = {}
        if self._last is None:
            return BoardView(partial=partial)
        corners = np.roll(self._last, 2, axis=0) if self.flip else self._last
        H = cv2.getPerspectiveTransform(board_corners_mm(), corners)
        return BoardView(self._last, H, dict(self._positions), stale,
                         partial)

    def reset(self):
        self._detect_t, self._view_cache = 0.0, None
        self._last = None
        self._pending, self._pending_n = None, 0
        self._positions = {}          # (side, slot) → card quad mm

    def top_down(self, frame, view):
        """Straightened board image, TOP_PX_PER_MM."""
        size = int(BOARD_MM * TOP_PX_PER_MM)
        S = np.diag([TOP_PX_PER_MM, TOP_PX_PER_MM, 1]).astype(np.float64)
        return cv2.warpPerspective(frame, S @ np.linalg.inv(view.H),
                                   (size, size))

    def card_image(self, frame, view, quad_mm):
        """Straightened card, cut from the full-resolution frame."""
        w, h = REF_SIZE
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        M = cv2.getPerspectiveTransform(view.project(quad_mm), dst)
        return cv2.warpPerspective(frame, M, (w, h))

    def zone_image(self, frame, view, side, slot, px_per_mm=3.0):
        x0, y0, w, h = zone_rect_mm(side, slot)
        S = np.array([[px_per_mm, 0, -x0 * px_per_mm],
                      [0, px_per_mm, -y0 * px_per_mm], [0, 0, 1]])
        size = (int(w * px_per_mm), int(h * px_per_mm))
        return cv2.warpPerspective(frame, S @ np.linalg.inv(view.H), size)

    def identify(self, frame, view):
        """{side: {slot: (key | "empty" | "unknown", inliers)}}"""
        if not view.found:
            return {}
        best = {}                                   # zone → (dist, quad mm)
        for quad in find_cards(self.top_down(frame, view)):
            centre = quad.mean(0)
            zone = zone_of(centre)
            if zone is None:
                continue
            d = float(np.linalg.norm(centre - zone_quad_mm(*zone).mean(0)))
            if zone not in best or d < best[zone][0]:
                best[zone] = (d, quad)              # one card per zone
        out = {s: {} for s in SIDES}
        positions = {}
        for side in SIDES:
            for slot in range(1, 7):
                zone = (side, slot)
                x0, y0, *_ = zone_rect_mm(side, slot)
                result = None
                if zone in best:                                    # (1)
                    quad = best[zone][1]
                    positions[zone] = quad      # kept even if unrecognized
                    key, n = self.cards.identify(
                        self.card_image(frame, view, quad))
                    if key not in (EMPTY, UNKNOWN):
                        result = (key, n)
                zimg = None
                if result is None:
                    zimg = self.zone_image(frame, view, side, slot)
                    if zone in best or zone_texture(zimg) >= \
                            BOARD_CONFIG["zone_texture"]:
                        key, n, corners = self.cards.search(zimg)   # (2)
                        if key:
                            quad = corners / 3.0 + [x0, y0]
                            # a card only counts for the zone its centre is
                            # in, not every zone it pokes into
                            if zone_of(quad.mean(0)) == zone:
                                positions[zone] = quad
                                result = (key, n)
                if result is None:
                    quad = find_card_in_zone(zimg, 3.0)             # (3)
                    if quad is not None:
                        quad = quad + [x0, y0]
                        positions.setdefault(zone, quad)
                        key, n = self.cards.identify(
                            self.card_image(frame, view, quad))
                        result = (key if key != EMPTY else UNKNOWN, n)
                if result is None:                                  # (4)
                    result = (UNKNOWN, 0) if zone in best else (EMPTY, 0)
                out[side][slot] = result
        self._positions = positions
        view.cards = dict(positions)
        return out


class SlotVotes:
    """Smooths per-frame readings: a slot's card is whatever it read most
    over the last `window` readings, once it has `min_votes` of them."""

    def __init__(self, window=5, min_votes=3):
        self.window, self.min_votes = window, min_votes
        self.hist = {(s, n): deque(maxlen=window) for s in SIDES
                     for n in range(1, 7)}

    def add(self, reading):
        for side, slots in reading.items():
            for slot, (key, _) in slots.items():
                self.hist[(side, slot)].append(key)

    def slot(self, side, slot):
        h = self.hist[(side, slot)]
        if not h:
            return None
        key, count = Counter(h).most_common(1)[0]
        return key if count >= self.min_votes else None

    def party(self, side):
        """{slot: species_key} for confidently recognized cards."""
        out = {}
        for n in range(1, 7):
            key = self.slot(side, n)
            if key not in (None, EMPTY, UNKNOWN):
                out[n] = key
        return out

    def unknown_slots(self, side):
        return [n for n in range(1, 7) if self.slot(side, n) == UNKNOWN]

    def seen(self, side):
        return any(self.hist[(side, n)] for n in range(1, 7))
