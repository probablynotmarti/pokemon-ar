"""Software renderer for low-poly figurines, drawn straight onto the camera
frame (no OpenGL): transform → cull back faces → flat shade → sort far to
near → fill. A few hundred triangles per figurine keeps it to a few ms.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# light from above, slightly in front and to the left (board frame, where
# "up" is −z); ambient keeps the shadowed side readable
LIGHT = np.float32([-0.35, 0.45, -1.0])
LIGHT /= np.linalg.norm(LIGHT)
AMBIENT, DIFFUSE = 0.48, 0.62


@dataclass
class Placement:
    """Where a figurine stands on the board (mm) and how it's posed."""
    x: float
    y: float
    yaw: float = 0.0          # radians; 0 = facing +y (toward you)
    scale: float = 1.0
    lift: float = 0.0         # mm above the board
    flash: float = 0.0        # 0..1 blend toward white (hit)
    tint: tuple = None        # BGR to blend toward (status)
    tint_amount: float = 0.0
    squash: float = 1.0       # vertical stretch (bounce / faint)

    def to_board(self, verts):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x = verts[:, 0] * c - verts[:, 1] * s
        y = verts[:, 0] * s + verts[:, 1] * c
        z = verts[:, 2] * self.squash
        k = self.scale
        return np.stack([self.x + k * x, self.y + k * y,
                         -(k * z + self.lift)], 1).astype(np.float32)

    def rotate(self, normals):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        x = normals[:, 0] * c - normals[:, 1] * s
        y = normals[:, 0] * s + normals[:, 1] * c
        return np.stack([x, y, -normals[:, 2]], 1)


def yaw_towards(x, y, tx, ty):
    """Yaw that turns a figurine at (x, y) to face (tx, ty)."""
    return float(np.arctan2(-(tx - x), ty - y))


def _project(pts, K, rvec, tvec):
    R = cv2.Rodrigues(rvec)[0].astype(np.float32)
    cam = pts @ R.T + np.float32(tvec).ravel()
    z = np.maximum(cam[:, 2], 1e-3)
    uv = np.stack([K[0, 0] * cam[:, 0] / z + K[0, 2],
                   K[1, 1] * cam[:, 1] / z + K[1, 2]], 1)
    return cam, uv


def draw_shadow(img, place, radius, pose, strength=0.35):
    """Soft dark ellipse on the board under the figurine."""
    K, rvec, tvec = pose
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    k = place.scale * max(0.3, 1 - place.lift / 120)
    ring = np.stack([place.x + radius * k * np.cos(ang),
                     place.y + radius * k * 0.8 * np.sin(ang),
                     np.zeros_like(ang)], 1).astype(np.float32)
    _, uv = _project(ring, K, rvec, tvec)
    poly = np.round(uv).astype(np.int32)
    x0, y0 = np.maximum(poly.min(0) - 12, 0)
    x1, y1 = np.minimum(poly.max(0) + 12, [img.shape[1], img.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, poly - [x0, y0], 255, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32) / 255
    roi = img[y0:y1, x0:x1].astype(np.float32)
    roi *= (1 - strength * mask)[..., None]
    img[y0:y1, x0:x1] = roi.astype(np.uint8)


def draw_mesh(img, mesh, place, pose):
    """Draw a figurine onto img (BGR, in place). pose = (K, rvec, tvec)."""
    if place.scale <= 0.01:
        return
    K, rvec, tvec = pose
    pts = place.to_board(mesh.verts)
    cam, uv = _project(pts, K, rvec, tvec)
    tri = cam[mesh.faces]                                   # (M, 3, 3)
    n_cam = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    # visible if the face turns toward the camera (camera at the origin);
    # board z points down, which flips handedness, so the sign is reversed
    visible = (n_cam * tri[:, 0]).sum(1) > 0
    if not visible.any():
        return
    # shading in board space
    v = mesh.verts[mesh.faces[visible]]
    n_model = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n_board = place.rotate(n_model)
    n_board /= np.linalg.norm(n_board, axis=1, keepdims=True) + 1e-9
    shade = AMBIENT + DIFFUSE * np.clip(n_board @ LIGHT, 0, 1)
    col = mesh.colors[visible] * shade[:, None]
    if place.tint is not None and place.tint_amount > 0:
        col = col * (1 - place.tint_amount) + \
            np.float32(place.tint) * place.tint_amount
    if place.flash > 0:
        col = col * (1 - place.flash) + 255 * place.flash
    col = np.clip(col, 0, 255)
    depth = tri[visible][:, :, 2].mean(1)
    faces_uv = uv[mesh.faces[visible]]
    order = np.argsort(-depth)                               # far first
    pts16 = np.round(faces_uv * 16).astype(np.int32)
    for i in order:
        cv2.fillConvexPoly(img, pts16[i], col[i].tolist(), cv2.LINE_AA,
                           shift=4)
