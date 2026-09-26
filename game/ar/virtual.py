"""A virtual board for playing without the phone: same figurines, same
stage, seen from a virtual camera at your seat."""

import cv2
import numpy as np

from game.vision.board import BOARD_MM, board_corners_mm

from .pose import camera_matrix

TABLE_BGR, BOARD_BGR = (225, 228, 232), (120, 160, 196)


def virtual_scene(size=(1920, 1080), tilt_deg=40, dist=820, f=1700.0):
    """(background image, pose) with the board seen from your seat."""
    K = camera_matrix(f, size)
    t = np.radians(tilt_deg)
    centre = np.array([BOARD_MM / 2, BOARD_MM / 2, 0.0])
    cam = centre + np.array([0, dist * np.cos(t), -dist * np.sin(t)])
    zc = (centre - cam) / np.linalg.norm(centre - cam)
    xc = np.array([1.0, 0, 0])
    R = np.stack([xc, np.cross(zc, xc), zc])
    rvec, tvec = cv2.Rodrigues(R)[0], (-R @ cam).reshape(3, 1)
    img = np.full((size[1], size[0], 3), TABLE_BGR, np.uint8)
    obj = np.hstack([board_corners_mm(), np.zeros((4, 1), np.float32)])
    quad, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    quad = quad.reshape(-1, 2)
    cv2.fillConvexPoly(img, np.int32(quad), BOARD_BGR, cv2.LINE_AA)
    mid = np.float32([[0, BOARD_MM / 2, 0], [BOARD_MM, BOARD_MM / 2, 0]])
    fold, _ = cv2.projectPoints(mid, rvec, tvec, K, None)
    a, b = np.int32(fold.reshape(-1, 2))
    cv2.line(img, tuple(a), tuple(b), (105, 142, 178), 3, cv2.LINE_AA)
    H = cv2.getPerspectiveTransform(board_corners_mm(), np.float32(quad))
    return img, (K, rvec, tvec), H
