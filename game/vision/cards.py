"""Which card is in this (straightened) slot image?

1. SIFT keypoints vs each reference card, ratio test, RANSAC homography:
   the inlier count is the score. Handles glare, angle, upside-down cards.
2. Blank-paper check → "empty".
3. Fallback for small/blurry cards (far mat, 720p): compare tiny Lab
   colour thumbnails (brightness-normalized). Accepted only with a clear
   margin, and never against real SIFT evidence for a different card.
Anything else is "unknown" (no reference for it, or badly covered).
"""

import pathlib

import cv2
import numpy as np

from .geometry import REF_SIZE

CARDS_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "cards"
CARD_CONFIG = {
    "ratio": 0.75,          # Lowe ratio test
    "min_inliers": 14,      # below this it's not a confident match
    "margin": 1.4,          # best must beat runner-up by this factor
    "blank_std": 18.0,      # grayscale std below this → empty paper
    "thumb_max": 14.0,      # thumbnail distance must be under this…
    "thumb_margin": 1.35,   # …and beat the runner-up by this factor
    "sift_veto": 6,         # SIFT inliers that overrule a thumbnail guess
}
EMPTY, UNKNOWN = "empty", "unknown"


def _thumb(img):
    t = cv2.resize(img, (24, 33), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(cv2.GaussianBlur(t, (3, 3), 0),
                       cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab - lab.reshape(-1, 3).mean(0)      # ignore overall tint


def _prep(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.resize(gray, REF_SIZE)
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


class CardIndex:
    def __init__(self, cards_dir=CARDS_DIR, cfg=None, allow_empty=False):
        self.cfg = {**CARD_CONFIG, **(cfg or {})}
        self.sift = cv2.SIFT_create(nfeatures=800)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.refs = {}                       # key -> (keypoints, descriptors)
        self.thumbs = {}                     # key -> [upright, upside-down]
        for path in sorted(pathlib.Path(cards_dir).glob("*.jpg")):
            self.add(path.stem, cv2.imread(str(path)))
        if not self.refs and not allow_empty:
            raise FileNotFoundError(
                f"no reference cards in {cards_dir} — register yours with "
                "tools/grid_cam.py --register <pokemon> (see README)")

    def add(self, key, img):
        kp, des = self.sift.detectAndCompute(_prep(img), None)
        self.refs[key] = (kp, des)
        self.thumbs[key] = [_thumb(img), _thumb(cv2.rotate(img, cv2.ROTATE_180))]

    def thumb_distances(self, crop):
        t = _thumb(crop)
        return {k: min(float(np.abs(t - v).mean()) for v in pair)
                for k, pair in self.thumbs.items()}

    def scores(self, crop):
        """{key: inlier count} for one straightened card image."""
        return {k: n for k, (n, _) in self._match(_prep(crop)).items()}

    def _match(self, gray):
        """{key: (inliers, H ref→image)} for a prepared grayscale image."""
        kp, des = self.sift.detectAndCompute(gray, None)
        out = {k: (0, None) for k in self.refs}
        if des is None or len(kp) < 8:
            return out
        for key, (rkp, rdes) in self.refs.items():
            if rdes is None:
                continue
            pairs = self.matcher.knnMatch(des, rdes, k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2)
                    if m.distance < self.cfg["ratio"] * n.distance]
            if len(good) < 8:
                out[key] = (len(good) // 2, None)
                continue
            src = np.float32([rkp[m.trainIdx].pt for m in good])
            dst = np.float32([kp[m.queryIdx].pt for m in good])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
            out[key] = (int(mask.sum()) if mask is not None else 0, H)
        return out

    def search(self, region):
        """Find a card anywhere inside a larger image (e.g. a whole zone).
        → (key, inliers, card corners in region px) or (None, best, None)."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        res = self._match(gray)
        if not res:
            return None, 0, None
        ranked = sorted(res.items(), key=lambda kv: -kv[1][0])
        (best, (n, H)), runner = ranked[0], ranked[1][1][0] if len(ranked) > 1 else 0
        if H is None or n < self.cfg["min_inliers"] or n < self.cfg["margin"] * runner:
            return None, n, None
        w, h = REF_SIZE
        ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        return best, n, cv2.perspectiveTransform(ref_corners, H).reshape(4, 2)

    def identify(self, crop):
        """→ (key | "empty" | "unknown", inliers)."""
        c = self.cfg
        sc = self.scores(crop)
        if not sc:                                   # nothing registered yet
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return (EMPTY if float(gray.std()) < c["blank_std"] else UNKNOWN), 0
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        best, n = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        if n >= c["min_inliers"] and n >= c["margin"] * runner:
            return best, n
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < c["blank_std"]:
            return EMPTY, n
        dist = sorted((d, k) for k, d in self.thumb_distances(crop).items())
        if len(dist) < 2:
            return UNKNOWN, n
        (d1, k1), (d2, _) = dist[0], dist[1]
        vetoed = n >= c["sift_veto"] and best != k1
        if d1 < c["thumb_max"] and d2 >= c["thumb_margin"] * d1 and not vetoed:
            return k1, n
        return UNKNOWN, n
