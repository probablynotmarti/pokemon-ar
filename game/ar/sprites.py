"""Animated game sprites standing on the cards (billboards).

The sprites are the games' own 3D-model renders (assets/sprites/front|back,
see assets/ATTRIBUTION.md): the CPU's Pokémon shows its front (facing you),
yours shows its back — the 3DS/Switch battle camera. Each is a flat panel
standing upright on its card, turned to face the camera, drawn with the
same camera pose as everything else, so it sits in the scene like a
cut-out. Pixel art is kept crisp (nearest-neighbour).
"""

import pathlib

import cv2
import numpy as np
from PIL import Image, ImageSequence

SPRITES_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "sprites"
MM_PER_PX = 1.15              # sprite pixel → board mm (Politoed ≈ 85 mm tall)
_cache = {}


class Sprite:
    def __init__(self, path):
        im = Image.open(path)
        self.frames, self.durations = [], []
        for f in ImageSequence.Iterator(im):
            rgba = np.array(f.convert("RGBA"))
            self.frames.append(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            self.durations.append(max(20, f.info.get("duration", 40)) / 1000)
        # crop every frame to the union of visible pixels (feet on the floor)
        alpha = np.max([fr[..., 3] for fr in self.frames], 0)
        ys, xs = np.nonzero(alpha > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        self.frames = [fr[y0:y1, x0:x1] for fr in self.frames]
        self.cycle = float(sum(self.durations))
        self.starts = np.cumsum([0.0] + self.durations[:-1])

    @property
    def size_px(self):
        h, w = self.frames[0].shape[:2]
        return w, h

    def frame(self, t):
        k = int(np.searchsorted(self.starts, t % self.cycle, side="right")) - 1
        return self.frames[max(0, k)]


def get_sprite(key, view, sprites_dir=SPRITES_DIR):
    """view: "front" | "back". None if there's no such file."""
    ck = (key, view, str(sprites_dir))
    if ck not in _cache:
        path = pathlib.Path(sprites_dir) / view / f"{key}.gif"
        _cache[ck] = Sprite(path) if path.exists() else None
    return _cache[ck]


def camera_position(pose):
    K, rvec, tvec = pose
    R = cv2.Rodrigues(rvec)[0]
    return (-R.T @ np.asarray(tvec, float).reshape(3)).ravel()


def billboard_quad(place, width_mm, height_mm, pose):
    """3D corners (TL, TR, BR, BL) of an upright panel standing at the
    placement, turned (about the vertical) to face the camera."""
    cam = camera_position(pose)
    to_cam = np.array([cam[0] - place.x, cam[1] - place.y])
    n = np.hypot(*to_cam)
    fwd = to_cam / n if n > 1e-6 else np.array([0.0, 1.0])
    right = np.array([fwd[1], -fwd[0], 0.0])        # camera's right, flat
    base = np.array([place.x, place.y, -place.lift])
    up = np.array([0.0, 0.0, -1.0])                   # board z points down
    hw = width_mm / 2
    return np.float32([base - right * hw + up * height_mm,
                       base + right * hw + up * height_mm,
                       base + right * hw, base - right * hw])


def draw_sprite(img, sprite, place, pose, t, mm_per_px=MM_PER_PX):
    """Draw the sprite frame for time t at a Placement (BGR img, in place)."""
    if place.scale <= 0.01 or sprite is None:
        return
    K, rvec, tvec = pose
    frame = sprite.frame(t)
    h, w = frame.shape[:2]
    k = place.scale * mm_per_px
    quad3 = billboard_quad(place, w * k, h * k * place.squash, pose)
    uv, _ = cv2.projectPoints(quad3, rvec, tvec, K, None)
    uv = uv.reshape(-1, 2)
    x0, y0 = np.floor(uv.min(0)).astype(int) - 1
    x1, y1 = np.ceil(uv.max(0)).astype(int) + 1
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(uv - [cx0, cy0]))
    warped = cv2.warpPerspective(frame, M, (cx1 - cx0, cy1 - cy0),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255
    col = warped[..., :3].astype(np.float32)
    if place.tint is not None and place.tint_amount > 0:
        col = col * (1 - place.tint_amount) + np.float32(place.tint) * place.tint_amount
    if place.flash > 0:
        col = col * (1 - place.flash) + 255 * place.flash
    roi = img[cy0:cy1, cx0:cx1].astype(np.float32)
    img[cy0:cy1, cx0:cx1] = (roi * (1 - a) + col * a).astype(np.uint8)
