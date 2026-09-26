"""Synthetic grid-camera frames: the brown Monopoly-board back with real
card photos on it, on a white sheet, seen by a tilted phone, with glare,
shadow, noise, blur and JPEG. Used by test_vision."""

import pathlib

import cv2
import numpy as np

from game.vision.board import BOARD_MM, zone_rect_mm

CARDS = pathlib.Path(__file__).resolve().parents[1] / "assets" / "cards"
PX = 2.5                               # table image resolution, px/mm
MARGIN = 90                            # sheet around the board, mm
BOARD_BGR = (118, 160, 196)            # cardboard brown (from the photo)


def _paste(canvas, img, quad_px):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad_px))
    size = canvas.shape[1::-1]
    warped = cv2.warpPerspective(img, M, size)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size)
    canvas[mask > 0] = warped[mask > 0]


def card_quad_mm(side, slot, rng, sloppy=1.0):
    """Card corners (mm) somewhere near the middle of its zone."""
    x, y, w, h = zone_rect_mm(side, slot)
    cx = x + w / 2 + rng.uniform(-25, 25) * sloppy
    cy = y + h / 2 + rng.uniform(-12, 12) * sloppy
    ang = np.deg2rad(rng.uniform(-8, 8) * sloppy)
    hw, hh = 63 / 2, 88 / 2
    pts = np.float32([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.float32([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return pts @ rot.T + [cx, cy]


def table_image(board, rng, upside_down=(), sloppy=1.0):
    """board: {"player": {slot: key}, "cpu": {...}} → table image (BGR)."""
    size = int((BOARD_MM + 2 * MARGIN) * PX)
    sheet = np.full((size, size, 3), 236, np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    sheet += (np.sin(xx / 23.0 + np.sin(yy / 41.0) * 3) * 7)[..., None]  # wrinkles
    table = np.clip(sheet, 0, 255).astype(np.uint8)
    b0, b1 = int(MARGIN * PX), int((MARGIN + BOARD_MM) * PX)
    brd = np.zeros((b1 - b0, b1 - b0, 3), np.float32)
    brd[:] = BOARD_BGR
    brd += rng.normal(0, 4, brd.shape)                        # cardboard grain
    mid = (b1 - b0) // 2
    brd[mid - 2:mid + 2, :] *= 0.93                            # fold cross
    brd[:, mid - 1:mid + 1] *= 0.95
    table[b0:b1, b0:b1] = np.clip(brd, 0, 255).astype(np.uint8)
    for side, slots in board.items():
        for slot, key in slots.items():
            card = cv2.imread(str(CARDS / f"{key}.jpg"))
            if (side, slot) in upside_down:
                card = cv2.rotate(card, cv2.ROTATE_180)
            quad = (card_quad_mm(side, slot, rng, sloppy) + MARGIN) * PX
            _paste(table, card, quad)
    return table


def photograph(table, rng, tilt=0.18, out=(1920, 1080), glare=True,
               shadow=False, noise=5.0, blur=1.0, jpeg=80, occlude=()):
    """Perspective 'phone photo' of the table. Returns (frame, table→frame M)."""
    th, tw = table.shape[:2]
    W, H = out
    inset = tilt * H * 0.5
    cx = W / 2 + rng.uniform(-30, 30)
    half = H * 0.52
    dst = np.float32([[cx - half + inset, 0], [cx + half - inset, 0],
                      [cx + half, H], [cx - half, H]])
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(table, M, (W, H), borderValue=(225, 225, 225))
    img = img.astype(np.float32)
    img *= np.linspace(0.85, 1.08, W)[None, :, None]           # uneven light
    yy, xx = np.mgrid[0:H, 0:W]
    if glare:
        gx, gy = rng.uniform(0.35, 0.65) * W, rng.uniform(0.3, 0.7) * H
        img += 70 * np.exp(-(((xx - gx) / 150) ** 2 +
                             ((yy - gy) / 80) ** 2))[..., None]
    if shadow:                                                 # phone/hand
        sx, sy = rng.uniform(0.55, 0.7) * W, rng.uniform(0.15, 0.3) * H
        img *= (1 - 0.55 * np.exp(-(((xx - sx) / 110) ** 2 +
                                    ((yy - sy) / 70) ** 2)))[..., None]
    for (x, y, r) in occlude:                                  # hand blobs
        cv2.circle(img, (int(x), int(y)), int(r), (120, 150, 200), -1)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), M


def board_corner_px(M, which=0):
    """Image position of a board corner (0=TL,1=TR,2=BR,3=BL)."""
    s = (MARGIN + np.float32([[0, 0], [BOARD_MM, 0], [BOARD_MM, BOARD_MM],
                              [0, BOARD_MM]])[which]) * PX
    return cv2.perspectiveTransform(np.float32([[s]]), M)[0, 0]
