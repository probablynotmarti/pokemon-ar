"""Battle effects: image particles living in board space (mm), drawn with
the same camera as the Pokémon, so a Shadow Ball really flies across the
board from one card to the other.

Images: assets/fx/showdown/<name>.png (Pokémon Showdown's battle effects)
and assets/fx/kenney/<name>.png (CC0 particles) — see ATTRIBUTION.md.
Missing images are skipped silently, so the game runs without them.
"""

import math
import pathlib
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

FX_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "fx"
_images = {}


def fx_image(name):
    """BGRA image by name ("fireball", "kenney/flame_01"), cached; None if
    the file isn't there."""
    if name not in _images:
        path = None
        for p in (FX_DIR / "showdown" / f"{name}.png", FX_DIR / f"{name}.png",
                  FX_DIR / "kenney" / f"{name}.png"):
            if p.exists():
                path = p
                break
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path else None
        if img is not None and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        if img is not None and img.ndim == 2:            # kenney: grey = alpha
            img = cv2.merge([np.full_like(img, 255)] * 3 + [img])
        _images[name] = img
    return _images[name]


def lerp(a, b, p):
    return a + (b - a) * p


@dataclass
class Particle:
    img: str
    t0: float
    dur: float
    path: object                 # p → (x, y, h) board mm, h = height
    size: object = 30.0          # mm (float or p → mm)
    alpha: object = None         # p → 0..1 (default: quick fade in/out)
    angle: float = 0.0           # degrees on screen
    spin: float = 0.0            # degrees / second
    tint: tuple = None           # BGR multiply
    glow: bool = False           # additive (for wisps/balls)
    tag: str = ""

    def at(self, now):
        p = (now - self.t0) / self.dur
        if not 0 <= p <= 1:
            return None
        x, y, h = self.path(p)
        size = self.size(p) if callable(self.size) else self.size
        a = self.alpha(p) if self.alpha else min(1.0, p * 8, (1 - p) * 5)
        ang = self.angle + self.spin * (now - self.t0)
        return (x, y, h), size, a, ang


FX_SCALE = 1.7                   # all effects drawn this much bigger (mm)


@dataclass
class FxSystem:
    particles: list = field(default_factory=list)
    shakes: list = field(default_factory=list)       # (t0, dur, px)
    rng: random.Random = field(default_factory=lambda: random.Random(7))

    def add(self, **kw):
        self.particles.append(Particle(**kw))

    def shake(self, now, px=10, dur=0.35):
        self.shakes.append((now, dur, px))

    def clear(self):
        self.particles.clear()
        self.shakes.clear()

    def shake_offset(self, now):
        dx = dy = 0.0
        keep = []
        for t0, dur, px in self.shakes:
            p = (now - t0) / dur
            if p < 1:
                keep.append((t0, dur, px))
                if p >= 0:
                    amp = px * (1 - p)
                    dx += amp * math.sin(p * 40)
                    dy += amp * math.cos(p * 33) * 0.6
        self.shakes = keep
        return int(dx), int(dy)

    def draw(self, img, pose, now):
        K, rvec, tvec = pose
        R = cv2.Rodrigues(rvec)[0]
        t = np.asarray(tvec, float).reshape(3)
        live, keep = [], []
        for pt in self.particles:
            if now < pt.t0:
                keep.append(pt)
                continue
            s = pt.at(now)
            if s is None:
                continue                                  # finished
            keep.append(pt)
            (x, y, h), size, a, ang = s
            cam = R @ np.array([x, y, -h]) + t
            if cam[2] > 1:
                live.append((cam[2], pt, cam, size, a, ang))
        self.particles = keep
        for depth, pt, cam, size, a, ang in sorted(live, key=lambda l: -l[0]):
            src = fx_image(pt.img)
            if src is None or a <= 0.01:
                continue
            u = K[0, 0] * cam[0] / cam[2] + K[0, 2]
            v = K[1, 1] * cam[1] / cam[2] + K[1, 2]
            px = K[0, 0] * size * FX_SCALE / cam[2]        # size on screen
            _blit(img, src, u, v, px, ang, a, pt.tint, pt.glow)


def _blit(img, src, u, v, px, angle, alpha, tint, glow):
    h, w = src.shape[:2]
    k = px / max(h, w)
    if k * max(h, w) < 2:
        return
    half = int(math.ceil(0.75 * max(h, w) * k)) + 2       # rotated extent
    x0, y0, x1, y1 = int(u) - half, int(v) - half, int(u) + half, int(v) + half
    H, W = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, k)
    M[0, 2] += (u - cx0) - w / 2
    M[1, 2] += (v - cy0) - h / 2
    rw, rh = cx1 - cx0, cy1 - cy0
    roi = img[cy0:cy1, cx0:cx1]
    if glow:
        # additive glow: soft by nature, so big ones are drawn at 1/2 or
        # 1/4 resolution and scaled up; all 8-bit maths
        ds = 4 if rw * rh > 250_000 else 2 if rw * rh > 40_000 else 1
        pm = (src[..., :3].astype(np.float32) *
              (src[..., 3:4].astype(np.float32) / 255))
        gain = alpha * (np.float32(tint) / 255 if tint is not None else
                        np.ones(3, np.float32))
        pm = np.clip(pm * gain, 0, 255).astype(np.uint8)
        small = cv2.warpAffine(pm, M / ds, (rw // ds + 1, rh // ds + 1),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
        layer = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR) \
            if ds > 1 else small[:rh, :rw]
        img[cy0:cy1, cx0:cx1] = cv2.add(roi, layer)
        return
    warped = cv2.warpAffine(src, M, (rw, rh), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) * (alpha / 255)
    col = warped[..., :3].astype(np.float32)
    if tint is not None:
        col = col * (np.float32(tint) / 255)
    out = roi.astype(np.float32) * (1 - a) + col * a
    img[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)


def preload():
    """Decode every effect image up front (avoids a hitch on first use)."""
    for folder, prefix in ((FX_DIR / "showdown", ""), (FX_DIR / "kenney", "kenney/")):
        for p in sorted(folder.glob("*.png")) if folder.exists() else []:
            fx_image(prefix + p.stem)
