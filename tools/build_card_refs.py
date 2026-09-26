"""Cut clean reference images of each card out of phone photos.

    ~/.venvs/memecv/bin/python tools/build_card_refs.py
    ~/.venvs/memecv/bin/python tools/build_card_refs.py --only seedot=~/Downloads/IMG_9000.HEIC

Writes assets/cards/<key>.jpg (straightened, 315x440). A better source
is the grid camera itself: tools/grid_cam.py --register <key>.
"""
import argparse
import pathlib
import subprocess
import sys
import tempfile

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.vision.geometry import find_card_quads, rectify  # noqa: E402

DL = pathlib.Path.home() / "Downloads"
# Top card's corners (TL, TR, BR, BL) as fractions of the photo's width /
# height, read off by hand: stacked cards and shadows defeat auto-detection.
_PX = {  # measured on a 525x700 view
    "registeel": [(122, 215), (400, 222), (408, 605), (118, 612)],
    "sudowoodo": [(115, 215), (364, 213), (390, 586), (103, 592)],
    "ducklett": [(130, 183), (372, 181), (395, 543), (110, 550)],
    "skwovet": [(130, 203), (380, 201), (393, 566), (113, 572)],
    "politoed": [(125, 208), (375, 210), (378, 575), (100, 572)],
    "marshadow": [(100, 112), (415, 110), (400, 545), (120, 545)],
    "gimmighoul": [(105, 250), (400, 250), (412, 640), (100, 640)],
    "riolu": [(125, 178), (400, 180), (408, 560), (115, 563)],
    "seedot": [(70, 145), (398, 133), (398, 595), (78, 603)],
    "solrock": [(100, 185), (420, 183), (437, 652), (103, 655)],
}
CORNERS = {k: [(x / 525, y / 700) for x, y in v] for k, v in _PX.items()}
PHOTOS = {
    "registeel": "IMG_8857.HEIC", "sudowoodo": "IMG_8858.HEIC",
    "ducklett": "IMG_8859.HEIC", "skwovet": "IMG_8860.HEIC",
    "politoed": "IMG_8861.HEIC", "marshadow": "IMG_8881.HEIC",
    "gimmighoul": "IMG_8883.HEIC", "riolu": "IMG_8884.HEIC",
    "seedot": "IMG_8885.HEIC", "solrock": "IMG_8886.HEIC",
}


def load(path):
    path = pathlib.Path(path).expanduser()
    if path.suffix.lower() == ".heic":        # OpenCV can't read HEIC
        out = pathlib.Path(tempfile.gettempdir()) / (path.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", "-Z", "1600",
                        str(path), "--out", str(out)], check=True,
                       capture_output=True)
        path = out
    return cv2.imread(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[],
                    help="key=photo pairs (default: the built-in list)")
    args = ap.parse_args()
    jobs = dict(a.split("=", 1) for a in args.only) or \
        {k: DL / v for k, v in PHOTOS.items()}
    out_dir = ROOT / "assets" / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, photo in jobs.items():
        img = load(photo)
        if img is None:
            print(f"{key}: can't read {photo}")
            continue
        h, w = img.shape[:2]
        if key in CORNERS and not args.only:
            quad = [(x * w, y * h) for x, y in CORNERS[key]]
        else:
            quads = find_card_quads(img)
            if not quads:
                print(f"{key}: no card outline found in {photo}")
                continue
            quad = quads[0]
        cv2.imwrite(str(out_dir / f"{key}.jpg"), rectify(img, quad))
        print(f"{key}: ok")


if __name__ == "__main__":
    main()
