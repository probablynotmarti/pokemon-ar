"""Shared card geometry + perspective helpers."""

import cv2
import numpy as np

CARD_MM = (63.0, 88.0)          # standard Pokémon TCG card
REF_SIZE = (315, 440)           # rectified card image, 5 px/mm


def order_quad(pts):
    """4 points → TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()],
                     pts[d.argmax()]], np.float32)


def rectify(img, quad, size=REF_SIZE):
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(order_quad(quad), dst)
    return cv2.warpPerspective(img, M, (w, h))


def find_card_quads(img, min_area_frac=0.05):
    """Card-shaped quadrilaterals, best first (aspect ≈ 63:88)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area_frac * img.shape[0] * img.shape[1]
    target = CARD_MM[0] / CARD_MM[1]
    found = []
    for c in contours:
        if cv2.contourArea(c) < area_min:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_quad(approx)
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        aspect = min(w, h) / max(w, h)
        err = abs(aspect - target)
        if err < 0.08:
            found.append((err, -cv2.contourArea(approx), q))
    found.sort(key=lambda f: (round(f[0] / 0.02), f[1]))
    return [f[2] for f in found]
