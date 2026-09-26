"""Grid camera viewer: check the phone sees the board and reads the cards.

    PY=~/.venvs/memecv/bin/python
    $PY tools/grid_cam.py                   # live overlay
    $PY tools/grid_cam.py --register seedot # put Seedot in YOUR slot 1,
                                            # press SPACE to save it as the
                                            # reference image for "seedot"
    $PY tools/grid_cam.py --flip            # phone mounted upside down
    $PY tools/grid_cam.py --image ~/Downloads/IMG_1234.HEIC
    $PY tools/grid_cam.py --image latest    # newest photo in ~/Downloads

Board = the back of a Monopoly board. Far half (top of the image) = CPU,
near half = you; each half is a 3x2 grid of zones (drawn on screen).
Green = recognized · orange ? = card not recognized.
Keys: SPACE = save (register mode) · f = flip halves · s = snapshot · q = quit
"""
import argparse
import pathlib
import sys
import time

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.data import POKEDEX  # noqa: E402
from game.vision.cameras import list_cameras, pick_grid_camera  # noqa: E402
from game.vision.cards import CardIndex  # noqa: E402
from game.vision.gridcam import GridCam  # noqa: E402
from game.vision.overlay import draw_slots, setup_styles  # noqa: E402
from game.vision.overlay import draw_zones  # noqa: E402
from game.vision.reader import BoardReader  # noqa: E402


def status_lines(view, votes):
    lines = []
    if view.partial:
        lines.append("! part of the board is outside the view: move the "
                     "phone back/up until all 4 edges show")
        if not view.found:
            return lines
    if not view.found:
        return ["board not found: is the whole board in view?"]
    for side, title in (("player", "YOU"), ("cpu", "CPU")):
        party = votes.party(side)
        names = ", ".join(f"{n}:{POKEDEX[k].name}" for n, k in party.items())
        unk = votes.unknown_slots(side)
        lines.append(f"{title}: {names or '-'}" +
                     (f"   ? in slot {unk}" if unk else ""))
    return lines


def show(frame, view, votes, extra=""):
    vis = draw_slots(draw_zones(frame.copy(), view), view, setup_styles(votes))
    for i, line in enumerate(status_lines(view, votes) + [extra]):
        cv2.putText(vis, line, (20, 50 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis, line, (20, 50 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2, cv2.LINE_AA)
    h, w = vis.shape[:2]
    return cv2.resize(vis, (1280, int(1280 * h / w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int)
    ap.add_argument("--register", metavar="KEY", choices=sorted(POKEDEX))
    ap.add_argument("--image")
    ap.add_argument("--no-window", action="store_true",
                    help="with --image: just print + save the result")
    ap.add_argument("--peek", action="store_true",
                    help="also save the annotated view to snapshots/live.jpg "
                         "every 2 s (so someone else can look at it)")
    ap.add_argument("--flip", action="store_true",
                    help="phone mounted the other way: CPU half at the bottom")
    args = ap.parse_args()
    cards = CardIndex(allow_empty=bool(args.register))
    reader = BoardReader(cards, flip=args.flip)

    if args.image:                                  # still photo
        from game.vision.reader import SlotVotes
        from tools.build_card_refs import load        # handles .HEIC
        path = pathlib.Path(args.image).expanduser()
        if args.image == "latest":                    # newest photo in Downloads
            photos = [p for p in (pathlib.Path.home() / "Downloads").iterdir()
                      if p.suffix.lower() in (".heic", ".jpg", ".jpeg", ".png")]
            if not photos:
                sys.exit("no photos in ~/Downloads")
            path = max(photos, key=lambda p: p.stat().st_mtime)
            print(f"reading {path.name}")
        if not path.exists():
            sys.exit(f"no such file: {path}  (use the real name, e.g. "
                     "IMG_8891.HEIC, or 'latest' for the newest photo)")
        args.image = str(path)
        frame = load(args.image)
        if frame is None:
            sys.exit(f"can't read {args.image}")
        if max(frame.shape[:2]) > 1920:               # like the live stream
            s = 1920 / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s,
                               interpolation=cv2.INTER_AREA)
        view = reader.locate(frame)
        votes = SlotVotes(window=1, min_votes=1)
        votes.add(reader.identify(frame, view))
        print("\n".join(status_lines(view, votes)))
        (ROOT / "snapshots").mkdir(exist_ok=True)
        out = ROOT / "snapshots" / f"read_{pathlib.Path(args.image).stem}.jpg"
        vis = show(draw_zones(frame.copy(), view), view, votes)
        cv2.imwrite(str(out), vis)
        print(f"saved {out.relative_to(ROOT)}")
        if not args.no_window:
            print("press any key in the window (or close it) to finish")
            cv2.imshow("grid", vis)
            while cv2.waitKey(100) == -1:             # key, or window closed
                if cv2.getWindowProperty("grid", cv2.WND_PROP_VISIBLE) < 1:
                    break
            cv2.destroyAllWindows()
        return

    cams = list_cameras(wait_for_phone_s=0 if args.cam is not None else 4)
    idx = args.cam if args.cam is not None else pick_grid_camera(cams)
    if idx is None:
        sys.exit("No iPhone camera found (macOS doesn't see it). Wake the "
                 "iPhone, keep it near the Mac with Wi-Fi + Bluetooth on (or "
                 "plug it in with a cable), check Settings → General → AirPlay "
                 "& Continuity → Continuity Camera, or pass --cam N. "
                 "Cameras: " + ", ".join(f"[{i}] {n}" for i, (n, _) in
                                         enumerate(cams)))
    name = cams[idx][0] if idx < len(cams) else str(idx)
    print(f"grid camera: [{idx}] {name}")
    grid = GridCam(idx, reader, name)
    extra = (f"REGISTER {args.register}: card in YOUR slot 1, press SPACE"
             if args.register else "")
    last_peek = 0.0
    if args.peek:
        (ROOT / "snapshots").mkdir(exist_ok=True)
    try:
        while True:
            frame, view, fps = grid.snapshot()
            if frame is None:
                time.sleep(0.03)
                continue
            vis = show(frame, view, grid.votes,
                       f"{extra}   {fps:.0f} fps, scan {grid.scan_s:.1f}s")
            cv2.imshow("grid", vis)
            if args.peek and time.monotonic() - last_peek > 2:
                last_peek = time.monotonic()
                cv2.imwrite(str(ROOT / "snapshots" / "live.jpg"), vis)
                cv2.imwrite(str(ROOT / "snapshots" / "live_raw.jpg"), frame)
                print("peek:", " | ".join(status_lines(view, grid.votes)),
                      flush=True)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                (ROOT / "snapshots").mkdir(exist_ok=True)
                path = ROOT / "snapshots" / f"grid_{int(time.time())}.jpg"
                cv2.imwrite(str(path), frame)
                print("saved", path)
            if k == ord("f"):
                reader.flip = not reader.flip
                reader.reset()
                grid.rescan()
                print("flipped:", reader.flip)
            if k == ord(" ") and args.register:
                quad = view.cards.get(("player", 1))
                if quad is None:
                    print("no card found in YOUR slot 1 yet — wait for its "
                          "outline to appear, then press SPACE")
                    continue
                crop = reader.card_image(frame, view, quad)
                path = ROOT / "assets" / "cards" / f"{args.register}.jpg"
                cv2.imwrite(str(path), crop)
                cards.add(args.register, crop)
                grid.rescan()
                print(f"saved {path.relative_to(ROOT)}")
    finally:
        grid.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
