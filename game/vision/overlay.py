"""Draw board/slot overlays onto grid-camera frames, and crop the frame to
the board for the arena view."""

import cv2
import numpy as np

from .board import BOARD_MM, COLS, HALF, ZONE_MM
from .cards import EMPTY, UNKNOWN

SIDES = ("player", "cpu")
GREEN, ORANGE, GREY, WHITE, DARK = ((90, 220, 110), (40, 160, 255),
                                    (170, 170, 170), (255, 255, 255),
                                    (30, 30, 30))


def _label(img, text, org, color, scale):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, DARK,
                max(3, int(6 * scale)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                max(1, int(2.2 * scale)), cv2.LINE_AA)


def draw_zones(frame, view):
    """Thin zone grid + slot numbers + YOU / CPU halves (setup guide)."""
    if not view.found:
        return frame
    line = lambda a, b, col, t: cv2.line(                      # noqa: E731
        frame, tuple(view.project([a])[0].astype(int)),
        tuple(view.project([b])[0].astype(int)), col, t, cv2.LINE_AA)
    for i in range(COLS + 1):
        x = i * ZONE_MM[0]
        line((x, 0), (x, BOARD_MM), WHITE, 1)
    for j in range(5):
        y = j * ZONE_MM[1]
        line((0, y), (BOARD_MM, y), WHITE, 3 if j == 2 else 1)
    scale = max(0.6, frame.shape[1] / 1920 * 1.0)
    for side in SIDES:
        for slot in range(1, 7):
            x, y = view.zone_polygon(side, slot)[0]
            _label(frame, str(slot), (int(x) + 8, int(y) + int(30 * scale)),
                   WHITE, scale * 0.8)
        cx, cy = view.project([(BOARD_MM - 8, HALF[side] + 18)])[0]
        tag = "YOU" if side == "player" else "CPU"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        _label(frame, tag, (int(cx) - tw, int(cy)), WHITE, scale)
    cv2.polylines(frame, [view.outline().astype(np.int32)], True, WHITE, 2,
                  cv2.LINE_AA)
    return frame


def draw_slots(frame, view, styles):
    """styles: {(side, slot): (color_bgr, label, mode)} with mode in
    "active" | "normal" | "faint" | "fainted". Drawn on the card where it
    was last seen (or the middle of its zone)."""
    out = frame
    if not view.found:
        return out
    scale = max(0.6, frame.shape[1] / 1920 * 1.1)
    for (side, slot), (color, label, mode) in styles.items():
        if mode == "faint":
            continue                      # empty zones: the grid shows them
        poly = view.card_polygon(side, slot).astype(np.int32)
        if mode == "fainted":
            shade = out.copy()
            cv2.fillPoly(shade, [poly], DARK)
            cv2.addWeighted(shade, 0.55, out, 0.45, 0, out)
            cv2.line(out, tuple(poly[0]), tuple(poly[2]), GREY, 4, cv2.LINE_AA)
            cv2.line(out, tuple(poly[1]), tuple(poly[3]), GREY, 4, cv2.LINE_AA)
        thick = {"active": 8, "normal": 4, "fainted": 3}[mode]
        cv2.polylines(out, [poly], True, color, thick, cv2.LINE_AA)
        if label:
            slot_w = float(poly[:, 0].max() - poly[:, 0].min())
            (w1, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            s = min(scale, 0.9 * slot_w / w1)          # fit inside the card
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
            x = int(poly[:, 0].mean())
            y = int(poly[:, 1].max()) - int(14 * s)
            _label(out, label, (x - tw // 2, y), color, s)
    return out


def board_bbox(view, frame_shape, margin=0.04):
    """Pixel box around the board, or None."""
    if not view.found:
        return None
    pts = view.outline()
    h, w = frame_shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


class ArenaCrop:
    """Frames the board for the arena: the whole board plus headroom for
    the figurines standing on it, scaled to fit without distortion. Spare
    width is filled with a blurred, dimmed copy of the camera image. The
    framing follows the board smoothly."""

    def __init__(self, size, smooth=0.15, headroom=0.34):
        self.size, self.smooth, self.headroom = size, smooth, headroom
        self.box = None

    def __call__(self, frame, view):
        W, H = self.size
        fh, fw = frame.shape[:2]
        target = board_bbox(view, frame.shape) if view is not None else None
        if target is not None:                 # room above for figurines
            x0, y0, x1, y1 = target
            target = (x0, max(0.0, y0 - self.headroom * (y1 - y0)), x1, y1)
        else:
            target = self.box or (0, 0, fw, fh)
        self.box = target if self.box is None else tuple(
            b + self.smooth * (t - b) for b, t in zip(self.box, target))
        x0, y0, x1, y1 = (int(round(v)) for v in self.box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(fw, max(x0 + 2, x1)), min(fh, max(y0 + 2, y1))
        crop = frame[y0:y1, x0:x1]
        k = min(W / crop.shape[1], H / crop.shape[0])
        cw, ch = max(1, int(crop.shape[1] * k)), max(1, int(crop.shape[0] * k))
        fg = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
        # backdrop: the same frame, cover-scaled, blurred and dimmed —
        # blurred at 1/8 size, then scaled up (cheap, looks the same)
        sw, sh = W // 8, H // 8
        kb = max(sw / fw, sh / fh)
        small = cv2.resize(frame, (int(fw * kb) + 1, int(fh * kb) + 1),
                           interpolation=cv2.INTER_NEAREST)    # blurred anyway
        oy, ox = (small.shape[0] - sh) // 2, (small.shape[1] - sw) // 2
        small = cv2.GaussianBlur(small[oy:oy + sh, ox:ox + sw], (0, 0), 1.5)
        out = cv2.resize(cv2.convertScaleAbs(small, alpha=0.55), (W, H),
                         interpolation=cv2.INTER_LINEAR)
        px, py = (W - cw) // 2, (H - ch) // 2
        out[py:py + ch, px:px + cw] = fg
        return out


def setup_styles(votes):
    """Setup screen: green = recognized, orange ? = unknown, grey = empty."""
    from game.data import POKEDEX
    styles = {}
    for side in SIDES:
        for n in range(1, 7):
            key = votes.slot(side, n)
            if key in (None, EMPTY):
                styles[(side, n)] = (GREY, "", "faint")
            elif key == UNKNOWN:
                styles[(side, n)] = (ORANGE, "?", "normal")
            else:
                styles[(side, n)] = (GREEN, POKEDEX[key].name, "normal")
    return styles


def battle_styles(mons, hp, active, type_colors):
    """Battle: active card glows in its type colour, fainted are crossed out.
    mons/hp: {(side, slot): Mon / shown HP}; active: {side: slot}."""
    styles = {}
    for (side, slot), mon in mons.items():
        if hp[(side, slot)] <= 0:
            styles[(side, slot)] = (GREY, "", "fainted")
        elif active.get(side) == slot:
            r, g, b = type_colors[mon.species.types[0]]
            styles[(side, slot)] = ((b, g, r), mon.name, "active")
        else:
            styles[(side, slot)] = (WHITE, "", "normal")
    return styles
