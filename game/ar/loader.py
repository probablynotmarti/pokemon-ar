"""Figurine or real model? assets/models/<key>.glb (or .gltf/.obj/.ply/.stl)
replaces that Pokémon's built-in figurine automatically.

Real models are converted to the same flat-coloured low-poly mesh the
renderer draws: colours come from the model's textures/materials (one per
face), the model is stood on its feet, centred, and scaled to figurine
height. glTF is +Y up / facing +Z; that's mapped to our z-up / facing +y.
Needs `trimesh` (pip install trimesh) only if such a file exists.
"""

import pathlib

import numpy as np

from .figurines import BUILDERS
from .mesh import Mesh

MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "assets" / "models"
EXTS = (".glb", ".gltf", ".obj", ".ply", ".stl")
TARGET_HEIGHT_MM = 70.0
MAX_FACES = 6000              # more than this gets decimated (if possible)
_cache = {}


def model_path(key, models_dir=MODELS_DIR):
    for ext in EXTS:
        p = pathlib.Path(models_dir) / f"{key}{ext}"
        if p.exists():
            return p
    return None


def get_mesh(key, models_dir=MODELS_DIR):
    """The mesh to draw for a Pokémon (cached)."""
    ck = (key, str(models_dir))
    if ck not in _cache:
        path = model_path(key, models_dir)
        mesh = None
        if path is not None:
            try:
                mesh = load_model(path)
                print(f"model: using {path.name} for {key} "
                      f"({len(mesh.faces)} faces)")
            except Exception as e:                    # fall back, loudly
                print(f"model: couldn't load {path.name} ({e}); "
                      f"using the figurine for {key}")
        _cache[ck] = mesh if mesh is not None else BUILDERS[key]()
    return _cache[ck]


def load_model(path, target_height=TARGET_HEIGHT_MM):
    import trimesh
    scene = trimesh.load(str(path), force="scene")
    tm = scene.to_geometry() if hasattr(scene, "to_geometry") else \
        scene.dump(concatenate=True)
    if len(tm.faces) > MAX_FACES:
        try:
            tm = tm.simplify_quadric_decimation(face_count=MAX_FACES)
        except Exception:
            pass                                      # draw it anyway
    try:
        tm.fix_normals()                              # consistent outward faces
    except Exception:
        pass
    rgba = face_colours(tm)
    v = np.asarray(tm.vertices, np.float64)
    v = np.stack([-v[:, 0], v[:, 2], v[:, 1]], 1)     # Y-up/+Z fwd → z-up/+y fwd
    lo, hi = v.min(0), v.max(0)
    v -= [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    height = hi[2] - lo[2]
    v *= target_height / height if height > 0 else 1.0
    colors = np.asarray(rgba, np.float32)[:, [2, 1, 0]]   # RGBA → BGR
    return Mesh(v.astype(np.float32), np.asarray(tm.faces, np.int32), colors)


def face_colours(tm, fallback=(180, 180, 180, 255)):
    """One RGBA per face from whatever colour the model carries: texture
    (sampled per vertex), vertex/face colours, or a flat material colour."""
    vis = tm.visual
    n = len(tm.faces)
    try:
        if vis.kind == "texture" or hasattr(vis, "to_color"):
            rgba = vis.to_color().face_colors          # samples the texture
        else:
            rgba = vis.face_colors                     # vertex/face colours
        if rgba is not None and len(rgba) == n:
            return np.asarray(rgba)
    except Exception:
        pass
    mat = getattr(vis, "material", None)
    for attr in ("baseColorFactor", "main_color", "diffuse"):
        c = getattr(mat, attr, None) if mat is not None else None
        if c is not None:
            c = np.asarray(c, float).ravel()
            c = c * 255 if c.max() <= 1.0 else c
            return np.tile(np.r_[c[:3], 255], (n, 1))
    return np.tile(fallback, (n, 1))
