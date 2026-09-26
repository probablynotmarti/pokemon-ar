"""Where the phone is, from the board alone.

The board is a known square lying flat, so its homography H (board mm →
image px) pins down the camera: with square pixels and the principal point
at the image centre, H's first two columns must be orthogonal and equally
long after removing the focal length — that gives f. Then solvePnP gives
the camera's rotation/translation. Board frame: x right, y toward you,
z INTO the table, so "up" out of the board is −z.
"""

import cv2
import numpy as np

from game.vision.board import board_corners_mm


def focal_from_homography(H, size):
    """Focal length (px) implied by a plane homography, or None."""
    w, h = size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], float)
    G = T @ H                                   # principal point → origin
    (a1, b1, c1), (a2, b2, c2) = G[:, 0], G[:, 1]
    cands = []
    if abs(c1 * c2) > 1e-12:                    # h1 ⟂ h2
        cands.append(-(a1 * a2 + b1 * b2) / (c1 * c2))
    if abs(c2 * c2 - c1 * c1) > 1e-12:          # |h1| = |h2|
        cands.append((a1 * a1 + b1 * b1 - a2 * a2 - b2 * b2) /
                     (c2 * c2 - c1 * c1))
    good = [np.sqrt(c) for c in cands if c > 0]
    good = [f for f in good if 0.4 * w < f < 4.0 * w]
    return float(np.median(good)) if good else None


def camera_matrix(f, size):
    w, h = size
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)


class PoseTracker:
    """Board homography → camera (K, rvec, tvec). The phone doesn't move,
    so the focal estimate is averaged over time and the pose is cached."""

    FALLBACK_HFOV_DEG = 69.0          # iPhone main camera, 16:9 video

    def __init__(self):
        self.f = None
        self._key = None
        self._pose = None

    def update(self, H, size, quad_px=None):
        key = (H.round(4).tobytes(), size)
        if key == self._key:
            return self._pose
        f = focal_from_homography(H, size)
        if f is not None:
            self.f = f if self.f is None else 0.8 * self.f + 0.2 * f
        fx = self.f or size[0] / (2 * np.tan(np.radians(self.FALLBACK_HFOV_DEG / 2)))
        K = camera_matrix(fx, size)
        obj = np.hstack([board_corners_mm(), np.zeros((4, 1), np.float32)])
        img = (quad_px if quad_px is not None else
               cv2.perspectiveTransform(board_corners_mm().reshape(-1, 1, 2),
                                        H).reshape(-1, 2))
        # a flat square has two exact poses (one mirrored, camera under the
        # table); keep the one with the camera above the board (z < 0)
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, np.float32(img), K, None, flags=cv2.SOLVEPNP_IPPE)
        best = None
        for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
            R = cv2.Rodrigues(rvec)[0]
            if (-R.T @ tvec).ravel()[2] >= 0:        # camera below the board
                continue
            if best is None or err < best[2]:
                best = (rvec, tvec, err)
        self._key = key
        self._pose = (K, best[0], best[1]) if best else None
        return self._pose
