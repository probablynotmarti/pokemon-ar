"""Tiny low-poly mesh toolkit: convex primitives with a colour each.

Model space (mm): x right, y forward (the way the Pokémon faces), z up,
feet at z = 0. Every primitive is convex, so its faces can be oriented
outward automatically — the renderer relies on that for back-face culling.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    verts: np.ndarray            # (N, 3) float32
    faces: np.ndarray            # (M, 3) int32, counter-clockwise from outside
    colors: np.ndarray           # (M, 3) float32, BGR 0–255

    @property
    def height(self):
        return float(self.verts[:, 2].max() - self.verts[:, 2].min())


def _outward(verts, faces, centre):
    v = verts[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    flip = (n * (v.mean(1) - centre)).sum(1) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, ::-1]
    return faces


def _part(verts, faces, color, centre):
    verts = np.asarray(verts, np.float32)
    faces = _outward(verts, np.asarray(faces, np.int32), np.asarray(centre))
    return Mesh(verts, faces, np.tile(np.float32(color), (len(faces), 1)))


def _icosphere(detail):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
         (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
         (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(detail):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f)


_SPHERES = {d: _icosphere(d) for d in (1, 2)}


def ellipsoid(centre, radii, color, detail=1):
    v, f = _SPHERES[detail]
    return _part(v * np.asarray(radii, float) + centre, f, color, centre)


def sphere(centre, r, color, detail=1):
    return ellipsoid(centre, (r, r, r), color, detail)


def _frame(axis):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return axis, u, np.cross(axis, u)


def cylinder(p0, p1, r0, r1, color, segments=10):
    """Tapered, capped cylinder from p0 (radius r0) to p1 (radius r1)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis, u, w = _frame(p1 - p0)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w
    verts = np.vstack([p0 + r0 * ring, p1 + r1 * ring, p0, p1])
    n, c0, c1 = segments, 2 * segments, 2 * segments + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i), (c0, j, i), (c1, n + i, n + j)]
    return _part(verts, faces, color, (p0 + p1) / 2)


def cone(base, tip, r, color, segments=10):
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    axis, u, w = _frame(tip - base)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = base + r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w)
    verts = np.vstack([ring, tip, base])
    t, b = segments, segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(i, j, t), (b, j, i)]
    return _part(verts, faces, color, base * 0.75 + tip * 0.25)


def box(centre, size, color):
    c, s = np.asarray(centre, float), np.asarray(size, float) / 2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)]) * s + c
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return _part(corners, faces, color, c)


def merge(parts):
    verts, faces, colors, off = [], [], [], 0
    for p in parts:
        verts.append(p.verts)
        faces.append(p.faces + off)
        colors.append(p.colors)
        off += len(p.verts)
    return Mesh(np.vstack(verts).astype(np.float32),
                np.vstack(faces).astype(np.int32),
                np.vstack(colors).astype(np.float32))


def mirror_x(parts):
    """Left+right copies of symmetric parts (arms, eyes…)."""
    out = []
    for p in parts:
        out.append(p)
        v = p.verts * np.float32([-1, 1, 1])
        out.append(Mesh(v, p.faces[:, ::-1].copy(), p.colors))
    return out
