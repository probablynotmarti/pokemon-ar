"""Look at the figurines without playing.

    PY=~/.venvs/memecv/bin/python
    $PY tools/model_viewer.py                    # all 10 on a virtual board
    $PY tools/model_viewer.py --photo tests/fixtures/live_corner_clipped.jpg
                                                 # on the cards in a photo
Saves to snapshots/models_*.jpg and shows a window (any key closes).
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.ar.figurines import BUILDERS  # noqa: E402
from game.ar.loader import get_mesh  # noqa: E402
from game.ar.virtual import virtual_scene  # noqa: E402
from game.ar.pose import PoseTracker  # noqa: E402
from game.ar.render import Placement, draw_mesh, draw_shadow, yaw_towards  # noqa: E402
from game.vision.board import BOARD_MM, zone_quad_mm  # noqa: E402


def lineup():
    img, pose, _ = virtual_scene()
    keys = list(BUILDERS)
    spots = [("cpu", n) for n in range(1, 7)] + [("player", n) for n in (1, 2, 3, 5)]
    placed = []
    for key, zone in zip(keys, spots):
        cx, cy = zone_quad_mm(*zone).mean(0)
        placed.append((key, cx, cy, zone[0]))
    for key, cx, cy, side in sorted(placed, key=lambda p: p[2]):  # far first
        target_y = 400 if side == "cpu" else 100
        mesh = get_mesh(key)
        place = Placement(cx, cy, yaw_towards(cx, cy, BOARD_MM / 2, target_y))
        draw_shadow(img, place, 22, pose)
        draw_mesh(img, mesh, place, pose)
    return img


def on_photo(path):
    from game.vision.cards import CardIndex
    from game.vision.reader import BoardReader
    img = cv2.imread(str(path))
    reader = BoardReader(CardIndex())
    view = reader.locate(img)
    got = reader.identify(img, view)
    pose = PoseTracker().update(view.H, (img.shape[1], img.shape[0]))
    items = []
    for side, slots in got.items():
        for slot, (key, _) in slots.items():
            if key in BUILDERS:
                q = view.cards.get((side, slot))
                cx, cy = (q.mean(0) if q is not None
                          else zone_quad_mm(side, slot).mean(0))
                items.append((cy, key, cx, side))
    for cy, key, cx, side in sorted(items):
        foe_y = 380 if side == "cpu" else 130
        place = Placement(cx, cy, yaw_towards(cx, cy, BOARD_MM / 2, foe_y))
        draw_shadow(img, place, 22, pose)
        draw_mesh(img, get_mesh(key), place, pose)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()
    img = on_photo(args.photo) if args.photo else lineup()
    (ROOT / "snapshots").mkdir(exist_ok=True)
    name = f"models_{pathlib.Path(args.photo).stem}" if args.photo else "models_lineup"
    out = ROOT / "snapshots" / f"{name}.jpg"
    cv2.imwrite(str(out), img)
    print("saved", out.relative_to(ROOT))
    if not args.no_window:
        cv2.imshow("figurines", cv2.resize(img, (1280, int(1280 * img.shape[0] / img.shape[1]))))
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
