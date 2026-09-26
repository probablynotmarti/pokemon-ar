"""Live battle window: phone camera reads the cards on the board, Mac
camera reads your hand gestures.

    PY=~/.venvs/memecv/bin/python        # has MediaPipe 0.10.21 + OpenCV
    $PY tools/hand_ui.py                  # iPhone grid cam + hand gestures
    $PY tools/hand_ui.py --mouse          # mouse instead of hand gestures
    $PY tools/hand_ui.py --you 1=riolu,2=seedot --cpu 1=gimmighoul,3=solrock
                                          # type the parties, skip the grid cam

Setup: the board is the back of a Monopoly board, filmed from above. Far
half = CPU, near half = you, each a 3x2 grid of zones (drawn on screen;
f flips the halves). Put the cards down, pinch START when every card is
green. The parties are locked until the battle ends (moving cards does
nothing); rematch goes back to setup.

Gestures: open hand = move cursor · pinch = pick / skip text ·
          fist (hold) = back · peace (hold) = info
Keys:     arrows = move · Enter = pick · Esc/Backspace = back · i = info ·
          r = rematch · c = next camera · f = flip halves (setup) ·
          s = screenshot · q = quit
Mouse:    move = cursor · left click = pick · right click = back

The arena shows the live board with a 3D figurine standing on each active
card (on the setup screen: on every recognized card). Figurines lunge,
flinch, faint and swap with the battle; drop assets/models/<pokemon>.glb in
to replace a figurine with a real model. Without the phone the same thing
plays on a virtual board.
"""

import argparse
import pathlib
import random
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from game.battle import STATUS_TAGS, Battle, build_side, parse_mat  # noqa: E402
from game.data import POKEDEX  # noqa: E402
from game.gestures import GestureTracker  # noqa: E402
from game.menu import Menu, Tile  # noqa: E402
from game.vision.cameras import (list_cameras, pick_grid_camera,  # noqa: E402
                                 pick_hand_camera)
from game.vision.overlay import (ArenaCrop, battle_styles,  # noqa: E402
                                 draw_slots, draw_zones, setup_styles)
from game.ar.loader import get_mesh  # noqa: E402
from game.ar.pose import PoseTracker  # noqa: E402
from game.ar.render import draw_mesh, draw_shadow  # noqa: E402
from game.ar.sprites import draw_sprite, get_sprite  # noqa: E402
from game.ar.stage import Stage  # noqa: E402
from game.ar.virtual import virtual_scene  # noqa: E402
from game.vision.board import zone_quad_mm  # noqa: E402
from game.vision.reader import BoardView  # noqa: E402

W, H = 1280, 720
MSG = (16, 516, 740, 190)            # message box
PANEL = (772, 516, 492, 190)         # menu panel (title strip + tiles)
TILES = (780, 548, 476, 152)         # where panel coords [0,1]² land
PREVIEW = (1000, 12, 268, 151)       # camera preview
ARENA = (W, 508)                     # live board view (top of the window)
DEFAULT_YOU = "1=politoed,2=riolu,3=ducklett,5=seedot"
DEFAULT_CPU = "1=sudowoodo,2=gimmighoul,4=solrock"

TYPE_COLORS = {
    "Normal": (168, 167, 122), "Fire": (238, 129, 48), "Water": (99, 144, 240),
    "Electric": (247, 208, 44), "Grass": (122, 199, 76), "Ice": (150, 217, 214),
    "Fighting": (194, 46, 40), "Poison": (163, 62, 161), "Ground": (226, 191, 101),
    "Flying": (169, 143, 243), "Psychic": (249, 85, 135), "Bug": (166, 185, 26),
    "Rock": (182, 161, 54), "Ghost": (115, 87, 151), "Dragon": (111, 53, 252),
    "Dark": (112, 87, 70), "Steel": (183, 183, 206), "Fairy": (214, 133, 173),
    "Typeless": (120, 120, 120),
}
TITLES = {
    "main": "What will {mon} do?", "fight": "Choose a move",
    "bag": "Bag", "item_target": "Use {item} on…", "party": "Pokémon",
    "party_action": "{pick}", "run_confirm": "Forfeit the battle?",
    "forced_switch": "Choose your next Pokémon", "locked": "",
    "game_over": "", "setup": "Ready?",
}
# how long each event holds the screen; "move" gives the attack effect
# time to play before the damage event lands (see game/ar/move_fx.py)
EVENT_SECONDS = {"move": 1.6, "damage": 0.8, "residual": 0.9, "recoil": 0.9,
                 "win": 2.5}
ARROWS = {                            # macOS, Windows, Linux codes
    (0, -1): (63232, 2490368, 65362), (0, 1): (63233, 2621440, 65364),
    (-1, 0): (63234, 2424832, 65361), (1, 0): (63235, 2555904, 65363),
}
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
              (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
              (0, 17)]


def load_font(size, bold=False):
    for path, idx in (("/System/Library/Fonts/Avenir Next.ttc", 2 if bold else 7),
                      ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0)):
        try:
            return ImageFont.truetype(path, size, index=idx)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {k: load_font(*v) for k, v in {
    "title": (22, True), "tile": (22, True), "sub": (15, False),
    "msg": (26, True), "hint": (15, False), "plate": (22, True),
    "small": (15, False), "big": (54, True)}.items()}


# ── camera ────────────────────────────────────────────────────────────
CAMERA_HELP = """
Couldn't open the camera. macOS only lets apps that have been granted
Camera access use it, and the permission belongs to the app that runs
this command (Terminal, iTerm, VS Code, the Claude app…).

  • Run it from macOS Terminal:  System Settings → Privacy & Security →
    Camera → turn on Terminal, then quit and reopen Terminal.
  • Wrong camera? See the list with --list-cams, pick one with --cam N,
    or press c in the game window to cycle through cameras.
  • Or try it without a camera:  add --mouse
"""


class HandCam:
    """Mac camera + MediaPipe Hands in a background thread, so the game
    loop never waits for the camera. latest() hands back the newest result
    with its capture time (the gesture tracker times holds with it)."""

    def __init__(self, index, name=""):
        import mediapipe as mp
        self.index, self.name = index, name
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(CAMERA_HELP)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.frame, self.landmarks, self.aspect = None, None, 16 / 9
        self.seq, self.stamp, self.failed = 0, 0.0, False
        self._switch_to = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        misses = 0
        while not self._stop.is_set():
            if self._switch_to is not None:
                index, name = self._switch_to
                self._switch_to = None
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    self.cap.release()
                    self.cap, self.index, self.name = cap, index, name
                else:
                    print(f"camera [{index}] {name} didn't open; staying put")
            ok, frame = self.cap.read()
            stamp = time.monotonic()
            if not ok:
                misses += 1
                if misses > 60:                  # ~3 s of nothing
                    self.failed = True
                    return
                time.sleep(0.05)
                continue
            misses = 0
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (640, int(640 * h / w)))
            res = self.hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            lms = None
            if res.multi_hand_landmarks:
                lms = [(p.x, p.y, p.z)
                       for p in res.multi_hand_landmarks[0].landmark]
            with self._lock:
                self.frame, self.landmarks, self.aspect = small, lms, w / h
                self.seq += 1
                self.stamp = stamp

    def latest(self):
        """(seq, capture time, landmarks); seq changes once per new frame."""
        with self._lock:
            return self.seq, self.stamp, self.landmarks

    def switch(self, index, name=""):
        """Swap to another camera (the 'c' key)."""
        self._switch_to = (index, name)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()
        self.hands.close()


# ── setup screen: one START tile, enabled once the board reads cleanly ──
class SetupMenu:
    screen = "setup"

    def __init__(self):
        self.hovered, self.info, self.ready = "start", False, False
        self.party_idx = self.item = None
        self.sub = ""

    def tiles(self):
        return [Tile("start", "START BATTLE", (0.03, 0.03, 0.94, 0.94),
                     self.ready, self.sub, "action")]

    def point(self, cursor, t):
        self.hovered = "start"

    def move(self, dx, dy):
        pass

    def back(self, t):
        return False

    def toggle_info(self):
        pass

    def info_lines(self):
        return []


def setup_status(grid):
    """(ready, message, START sub-label) from the grid camera's votes."""
    _, view, _ = grid.snapshot()
    v = grid.votes
    lines, ready = [], True
    if view is not None and view.partial:
        return False, ("Part of the board is outside the phone's view.\n"
                       "Move the phone up or back until all 4 edges show."), \
            "waiting for the whole board…"
    if view is None or not view.found:
        return False, "Can't find the board: is all of it in the phone's view?", \
            "waiting for the board…"
    if view.partial:
        return False, ("Part of the board is outside the phone's view.\n"
                       "Move the phone up or back until all 4 corners show."), \
            "waiting for the whole board…"
    for side, who in (("player", "You"), ("cpu", "CPU")):
        party = v.party(side)
        unknown = v.unknown_slots(side)
        names = ", ".join(POKEDEX[k].name for _, k in sorted(party.items()))
        lines.append(f"{who}: {names or 'no cards yet'}")
        if unknown:
            lines.append(f"   ? slot {', '.join(map(str, unknown))}: card not recognized")
            ready = False
        if not party:
            ready = False
    sub = "pinch to lock parties" if ready else "waiting for cards…"
    return ready, "\n".join(lines), sub


# ── what the screen currently shows (lags the engine during playback) ──
class View:
    def __init__(self, battle):
        self.sync(battle)

    def sync(self, battle):
        self.active = {s.label: s.mon.slot for s in (battle.player, battle.cpu)}
        self.mons, self.hp, self.target_hp, self.status = {}, {}, {}, {}
        for s in (battle.player, battle.cpu):
            for m in s.party:
                key = (s.label, m.slot)
                self.mons[key] = m
                self.hp[key] = self.target_hp[key] = float(m.hp)
                self.status[key] = m.status

    def apply(self, e):
        d = e.data
        if e.kind == "switch_in":
            self.active[e.side] = d["slot"]
        elif e.kind in ("damage", "heal", "recoil", "residual") and "hp" in d:
            self.target_hp[(e.side, d["slot"])] = float(d["hp"])
        elif e.kind == "status" and d.get("status") in STATUS_TAGS:
            self.status[(e.side, d["slot"])] = d["status"]
        elif e.kind == "faint":
            self.target_hp[(e.side, d["slot"])] = 0.0

    def tick(self, dt):
        for key, target in self.target_hp.items():
            cur, rate = self.hp[key], self.mons[key].max_hp / 0.6 * dt
            self.hp[key] = (min(target, cur + rate) if cur < target
                            else max(target, cur - rate))


# ── drawing ───────────────────────────────────────────────────────────
def rgb(c):
    return tuple(int(v) for v in c)


def rounded(draw, box, r, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def tile_px(rect):
    x, y, w, h = rect
    X, Y, TW, TH = TILES
    return (X + x * TW, Y + y * TH, X + (x + w) * TW, Y + (y + h) * TH)


class Scene:
    def __init__(self):
        sky = np.linspace((34, 42, 64), (70, 96, 120), 300)
        ground = np.linspace((64, 110, 88), (40, 70, 58), H - 300)
        grad = np.concatenate([sky, ground])[:, None, :]
        self.bg = np.repeat(grad, W, axis=1).astype(np.uint8)   # RGB

    def render(self, battle, menu, view, *, message, cursor, pose,
               hold, cam, mode, fps, arena=None):
        bg = self.bg.copy()
        if arena is not None:                  # live board from the phone
            bg[:ARENA[1]] = arena
        img = Image.fromarray(bg)
        d = ImageDraw.Draw(img, "RGBA")
        if view is not None:
            self._arena(d, view, live=arena is not None)
        lines = menu.info_lines() if menu.info else []
        self._message(d, message, menu, mode, lines)
        self._panel(d, battle, menu, cursor, pose, hold)
        out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self._preview(out, cam, pose, mode, fps)
        return out

    # arena: stand-in for the AR view
    def _arena(self, d, view, live=False):
        # live board: plates go in the margins beside it (the board sits
        # in the middle of the arena), so they don't hide the active cards
        spots = ((("cpu", (820, 250), (24, 40), False),
                  ("player", (330, 405), (956, 300), True)) if live else
                 (("cpu", (820, 250), (40, 40), False),
                  ("player", (330, 405), (820, 330), True)))
        for side, (px, py), plate, flip in spots:
            key = (side, view.active[side])
            mon = view.mons[key]
            color = rgb(TYPE_COLORS[mon.species.types[0]])
            if not live:                        # placeholder "models"
                d.ellipse((px - 130, py - 22, px + 130, py + 22),
                          fill=(20, 30, 25, 110))
            if view.hp[key] > 0.5 and not live:
                d.ellipse((px - 58, py - 150, px + 58, py - 34), fill=color,
                          outline=(255, 255, 255), width=3)
                d.text((px, py - 92), mon.name[:2].upper(), anchor="mm",
                       font=FONTS["big"], fill=(255, 255, 255))
            self._plate(d, plate, mon, view.hp[key], view.status[key],
                        show_numbers=flip, w=300 if live else 400)

    def _plate(self, d, xy, mon, hp, status, show_numbers, w=400):
        x, y = xy
        h = 100 if show_numbers else 80
        rounded(d, (x, y, x + w, y + h), 14, fill=(248, 248, 240, 235),
                outline=(40, 40, 40), width=3)
        d.text((x + 18, y + 12), mon.name, font=FONTS["plate"], fill=(20, 20, 20))
        d.text((x + w - 18, y + 14), "Lv50", anchor="ra", font=FONTS["small"],
               fill=(60, 60, 60))
        if status:
            sx = x + w - 110
            rounded(d, (sx, y + 14, sx + 50, y + 36), 6,
                    fill={"slp": (140, 140, 140), "par": (216, 180, 40),
                          "frz": (110, 190, 220), "brn": (220, 90, 50)}[status])
            d.text((sx + 25, y + 25), STATUS_TAGS[status], anchor="mm",
                   font=FONTS["small"], fill=(255, 255, 255))
        frac = max(0.0, hp / mon.max_hp)
        bar = (x + 70, y + 48, x + w - 18, y + 62)
        d.text((x + 40, y + 55), "HP", anchor="mm", font=FONTS["small"],
               fill=(200, 120, 20))
        rounded(d, bar, 6, fill=(70, 70, 70))
        col = ((80, 200, 90) if frac > 0.5 else (240, 200, 50) if frac > 0.2
               else (225, 70, 60))
        if frac > 0:
            rounded(d, (bar[0], bar[1], bar[0] + (bar[2] - bar[0]) * frac,
                        bar[3]), 6, fill=col)
        if show_numbers:
            d.text((x + w - 18, y + 70), f"{round(hp)}/{mon.max_hp}", anchor="ra",
                   font=FONTS["plate"], fill=(20, 20, 20))

    def _message(self, d, message, menu, mode, info_lines):
        x, y, w, h = MSG
        rounded(d, (x, y, x + w, y + h), 18, fill=(28, 36, 52, 235),
                outline=(255, 210, 90) if info_lines else (230, 230, 230),
                width=3)
        if info_lines:                  # ✌️ info card replaces the message
            d.text((x + 28, y + 24), info_lines[0], font=FONTS["msg"],
                   fill=(255, 230, 150))
            body = "\n".join(wrap(line, 62) for line in info_lines[1:])
            d.multiline_text((x + 28, y + 66), body, font=FONTS["sub"],
                             fill=(230, 230, 230), spacing=8)
        else:
            text = message if "\n" in message else wrap(message, 44)
            d.multiline_text((x + 28, y + 30), text,
                             font=FONTS["msg" if "\n" not in message else "title"],
                             fill=(255, 255, 255), spacing=10)
        hint = {"camera": "open hand: move  ·  pinch: pick  ·  hold fist: "
                          "back  ·  hold peace: info",
                "mouse": "mouse: move  ·  left click: pick  ·  right click: "
                         "back  ·  i: info"}[mode]
        if menu.screen == "game_over":
            hint = "pinch / Enter: rematch  ·  q: quit"
        elif menu.screen == "setup":
            hint = "move a card to fix it  ·  f: swap halves  ·  pinch START when all are green"
        d.text((x + 28, y + h - 34), hint, font=FONTS["hint"],
               fill=(170, 180, 200))

    def _panel(self, d, battle, menu, cursor, pose, hold):
        x, y, w, h = PANEL
        rounded(d, (x, y, x + w, y + h), 18, fill=(245, 245, 238, 240),
                outline=(40, 40, 40), width=3)
        pick = ""
        if menu.party_idx is not None:
            pick = battle.player.party[menu.party_idx].name
        title = TITLES[menu.screen].format(
            mon=battle.player.mon.name if battle else "", item=menu.item,
            pick=pick)
        d.text((x + 16, y + 8), title, font=FONTS["title"], fill=(30, 30, 30))
        for t in menu.tiles():
            box = tile_px(t.rect)
            hovered = t.id == menu.hovered
            if t.kind == "move":
                from game.data import MOVES
                name = t.payload.move
                typ = MOVES[name].type if name in MOVES else "Typeless"
                fill = rgb(TYPE_COLORS[typ])
            else:
                fill = (214, 222, 236)
            if not t.enabled:
                fill = (200, 200, 200)
            rounded(d, box, 12, fill=fill + (255,),
                    outline=(255, 90, 60) if hovered else (90, 90, 90),
                    width=5 if hovered else 2)
            dark = t.kind == "move" and t.enabled
            ink = (255, 255, 255) if dark else (30, 30, 30)
            if not t.enabled:
                ink = (140, 140, 140)
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            d.text((cx, cy - (9 if t.sub else 0)), t.label, anchor="mm",
                   font=FONTS["tile"], fill=ink)
            if t.sub:
                d.text((cx, cy + 16), t.sub, anchor="mm", font=FONTS["sub"],
                       fill=ink)
        if menu.screen == "game_over":
            won = battle.winner == "player"
            d.text((x + w / 2, y + h / 2), "YOU WON!" if won else "YOU LOST",
                   anchor="mm", font=FONTS["big"],
                   fill=(40, 150, 60) if won else (200, 60, 50))
        if cursor is not None and menu.tiles():
            X, Y, TW, TH = TILES
            px, py = X + cursor[0] * TW, Y + cursor[1] * TH
            r = 11
            d.ellipse((px - r, py - r, px + r, py + r),
                      fill=(255, 90, 60, 255 if pose == "pinch" else 120),
                      outline=(255, 255, 255), width=2)
            if hold:
                d.arc((px - 20, py - 20, px + 20, py + 20), -90,
                      -90 + 360 * hold, fill=(255, 90, 60), width=4)

    def _preview(self, out, cam, pose, mode, fps):
        x, y, w, h = PREVIEW
        if mode == "camera" and cam is not None and cam.frame is not None:
            small = cv2.flip(cv2.resize(cam.frame, (w, h)), 1)
            if cam.landmarks:
                pts = [(int((1 - p[0]) * w), int(p[1] * h)) for p in cam.landmarks]
                for a, b in HAND_LINKS:
                    cv2.line(small, pts[a], pts[b], (80, 230, 255), 2)
                for p in pts:
                    cv2.circle(small, p, 3, (255, 255, 255), -1)
            out[y:y + h, x:x + w] = small
            label = f"{pose or 'no hand'}  {fps:.0f} fps  [{cam.name[:18]}]"
        else:
            out[y:y + h, x:x + w] = (40, 40, 40)
            label = "mouse mode"
        cv2.rectangle(out, (x, y), (x + w, y + h), (230, 230, 230), 2)
        cv2.putText(out, label, (x + 8, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur])


# ── app loop ──────────────────────────────────────────────────────────
class App:
    def __init__(self, args, grid=None):
        self.args, self.grid = args, grid
        self.scene = Scene()
        self.arena_crop = ArenaCrop(ARENA)
        self.stage = Stage()
        self.pose = PoseTracker()
        self.use_sprites = not getattr(args, "figurines", False)
        from game.ar.fx import preload as preload_fx
        preload_fx()                          # effect images, up front
        if self.use_sprites:                  # decode the animations up front
            from game.data import POKEDEX
            for key in POKEDEX:
                for view in ("front", "back"):
                    get_sprite(key, view)
        if not grid:                          # no phone: a virtual board
            bg, vpose, vH = virtual_scene()
            quad = cv2.perspectiveTransform(
                np.float32([[[0, 0]], [[508, 0]], [[508, 508]], [[0, 508]]]),
                vH).reshape(-1, 2)
            self.virtual = (bg, vpose, BoardView(quad, vH))
        self.manual = (parse_mat(args.you or DEFAULT_YOU),
                       parse_mat(args.cpu or DEFAULT_CPU))
        if grid:
            self.start_setup()
        else:
            self.new_battle(*self.manual)

    def start_setup(self):
        """Read the board until the player pinches START."""
        self.phase = "setup"
        self.grid.rescan()
        self.battle = self.view = None
        self.menu = SetupMenu()
        self.queue, self.current, self.current_until = [], None, 0.0
        self.message = "Put your cards on the near half, the CPU's on the far half."

    def update_setup(self):
        ready, self.message, self.menu.sub = setup_status(self.grid)
        self.menu.ready = ready

    def new_battle(self, you, cpu):
        self.phase = "battle"
        a = self.args
        self.battle = Battle(build_side("player", you), build_side("cpu", cpu),
                             random.Random(a.seed))
        self.menu = Menu(self.battle)
        self.view = View(self.battle)
        b = self.battle
        self.stage.start_battle({"player": dict(you), "cpu": dict(cpu)})
        self.queue = [("intro", f"The CPU sent out {b.cpu.mon.name}!", "cpu"),
                      ("intro", f"Go! {b.player.mon.name}!", "player")]
        self.current, self.current_until = None, 0.0
        self.message = ""

    def rematch(self):
        if self.grid:
            self.start_setup()
        else:
            self.new_battle(*self.manual)

    def arena_image(self, now=None):
        """The arena (RGB): live board from the phone, or the virtual board,
        with slot overlays and the 3D figurines standing on the cards."""
        now = time.monotonic() if now is None else now
        if not self.grid:
            bg, pose, bview = self.virtual
            vis = bg.copy()
            self._figurines(vis, pose, bview, now)
            return self._shake(cv2.cvtColor(self.arena_crop(vis, bview),
                                            cv2.COLOR_BGR2RGB), now)
        frame, gview, _ = self.grid.snapshot()
        if frame is None:
            return None
        if self.phase == "setup":
            styles = setup_styles(self.grid.votes)
        else:
            styles = battle_styles(self.view.mons, self.view.hp,
                                   self.view.active, TYPE_COLORS)
        vis = frame.copy()
        if self.phase == "setup":
            draw_zones(vis, gview)
        vis = draw_slots(vis, gview, styles)
        if gview.found and not gview.partial:
            pose = self.pose.update(gview.H, (vis.shape[1], vis.shape[0]))
            if pose is not None:
                self._figurines(vis, pose, gview, now)
        return self._shake(cv2.cvtColor(self.arena_crop(vis, gview),
                                        cv2.COLOR_BGR2RGB), now)

    def _shake(self, arena, now):
        dx, dy = self.stage.fx.shake_offset(now)
        if dx or dy:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            arena = cv2.warpAffine(arena, M, (arena.shape[1], arena.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        return arena

    def _figurines(self, vis, pose, bview, now):
        if self.phase == "setup":
            cards = {(s, n): k for s in ("player", "cpu")
                     for n, k in self.grid.votes.party(s).items()}
            self.stage.sync_setup(cards, now)
            status = lambda s, n: None                         # noqa: E731
        else:
            status = lambda s, n: self.view.status.get((s, n))  # noqa: E731

        def where(side, slot):
            q = bview.cards.get((side, slot))
            return tuple((q if q is not None else zone_quad_mm(side, slot)).mean(0))

        for key, place, radius, side in self.stage.placements(now, where, status):
            draw_shadow(vis, place, radius, pose)
            # the games' camera: the CPU's Pokémon faces you, yours shows its back
            sprite = (get_sprite(key, "front" if side == "cpu" else "back")
                      if self.use_sprites else None)
            if sprite is not None:
                draw_sprite(vis, sprite, place, pose, now)
            else:
                draw_mesh(vis, get_mesh(key), place, pose)
        self.stage.fx.draw(vis, pose, now)

    def playing(self):
        return self.current is not None or bool(self.queue)

    def step_playback(self, now):
        if self.current is not None and now < self.current_until:
            return
        if self.queue:
            item = self.queue.pop(0)
            if isinstance(item, tuple):                 # intro line
                self.current, self.message = item, item[1]
                self.current_until = now + 1.2
                side = self.battle.player if item[2] == "player" else self.battle.cpu
                self.stage.send_out(item[2], side.mon.slot, now)
            else:
                self.view.apply(item)
                self.stage.on_event(item, now)
                self.current = item
                if item.text:
                    self.message = item.text
                self.current_until = now + EVENT_SECONDS.get(item.kind, 1.2)
        elif self.current is not None:
            self.current = None
            self.view.sync(self.battle)
            self.menu.resume(now)
            m = self.menu
            if m.screen == "main":
                self.message = f"What will {self.battle.player.mon.name} do?"
            elif m.screen == "forced_switch":
                self.message = "Choose your next Pokémon."

    def confirm(self, now, cursor=None):
        if self.phase == "setup":
            if self.menu.ready:
                parties = self.grid.lock()
                print("locked parties:", parties)
                self.new_battle(parties["player"], parties["cpu"])
            return
        if self.menu.screen == "game_over" and not self.playing():
            return self.rematch()
        if self.playing():                              # pinch skips text
            if self.current is not None:
                self.current_until = now
            return
        events = self.menu.confirm(now, cursor)
        if events:
            self.queue.extend(events)

    def back(self, now):
        if not self.playing():
            self.menu.back(now)

    def tick(self, now, dt):
        if self.phase == "setup":
            self.update_setup()
            return
        self.step_playback(now)
        self.view.tick(dt)
        if not self.playing() and not self.message:
            self.message = f"What will {self.battle.player.mon.name} do?"

    def key(self, k, now):
        if k in (ord("q"), ord("Q")):
            return False
        if k in (13, 10, ord(" ")):
            self.confirm(now)
        elif k in (27, 8, 127):
            self.back(now)
        elif k in (ord("i"), ord("I")):
            self.menu.toggle_info()
        elif k in (ord("r"), ord("R")):
            self.rematch()
        elif k in (ord("f"), ord("F")) and self.grid and self.phase == "setup":
            self.grid.reader.flip = not self.grid.reader.flip
            self.grid.reader.reset()
            self.grid.rescan()
        else:
            for (dx, dy), codes in ARROWS.items():
                if k in codes:
                    self.menu.move(dx, dy)
        return True


def open_grid(args, cams):
    """Phone camera + card reader, or None (typed parties / no iPhone)."""
    if args.you or args.cpu or args.no_grid:
        return None
    idx = args.grid_cam if args.grid_cam is not None else pick_grid_camera(cams)
    if idx is None:
        print("No iPhone camera found (Continuity Camera) — using typed "
              "parties. Unlock the iPhone near the Mac, or pass --grid-cam N.")
        return None
    from game.vision.cards import CardIndex
    from game.vision.gridcam import GridCam
    from game.vision.reader import BoardReader
    name = cams[idx][0] if idx < len(cams) else f"camera {idx}"
    print(f"grid camera: [{idx}] {name}")
    try:
        return GridCam(idx, BoardReader(CardIndex(), flip=args.flip), name)
    except FileNotFoundError as e:           # fresh install: no card photos yet
        print(f"{e}\n→ using typed parties for now")
        return None
    except RuntimeError as e:
        print(f"{e} — using typed parties instead")
        return None


def run(args):
    mode = "mouse" if args.mouse else "camera"
    cam, tracker = None, GestureTracker()
    wants_grid = not (args.you or args.cpu or args.no_grid or args.grid_cam is not None)
    cams = list_cameras(wait_for_phone_s=4 if wants_grid else 0)
    grid = open_grid(args, cams)
    app = App(args, grid)
    if mode == "camera":
        idx = pick_hand_camera(cams) if args.cam is None else args.cam
        name = cams[idx][0] if idx < len(cams) else f"camera {idx}"
        print(f"hand camera: [{idx}] {name}   (press c to switch camera)")
        cam = HandCam(idx, name)
    mouse = {"cursor": None, "click": [], }

    def on_mouse(ev, x, y, flags, _):
        X, Y, TW, TH = TILES
        cur = ((x - X) / TW, (y - Y) / TH)
        mouse["cursor"] = cur if -0.2 < cur[0] < 1.2 and -0.3 < cur[1] < 1.2 \
            else None
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse["click"].append("confirm")
        elif ev == cv2.EVENT_RBUTTONDOWN:
            mouse["click"].append("back")

    win = "Pokémon AR — battle"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.imshow(win, np.zeros((H, W, 3), np.uint8))
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)   # pop to front…
    cv2.waitKey(1)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 0)   # …then behave
    if mode == "mouse":
        cv2.setMouseCallback(win, on_mouse)

    last, fps = time.monotonic(), 0.0
    cursor, pose, hold, last_seq = None, None, 0.0, -1
    while True:
        now = time.monotonic()
        dt, last = now - last, now
        fps = 0.9 * fps + 0.1 / max(dt, 1e-3)
        events = []
        if mode == "camera":
            if cam.failed:
                print(CAMERA_HELP)
                break
            seq, stamp, landmarks = cam.latest()
            confirm_cursor = None
            if seq != last_seq:                   # one tracker step per frame
                last_seq = seq
                tracker.cfg["aspect"] = cam.aspect
                g = tracker.update(landmarks, stamp)
                cursor, pose, events = g.cursor, g.pose, g.events
                confirm_cursor = g.confirm_cursor
            hold = 0.0
            if tracker.hold_pose and not tracker.hold_fired:
                hold = min(1.0, (now - tracker.hold_since) /
                           tracker.cfg["hold_s"])
        else:
            cursor, pose = mouse["cursor"], None
            events, mouse["click"] = mouse["click"], []
            confirm_cursor = None

        if cursor is not None and not app.playing():
            app.menu.point(cursor, now)
        for e in events:
            if e == "confirm":
                app.confirm(now, confirm_cursor)
            elif e == "back":
                app.back(now)
            elif e == "info":
                app.menu.toggle_info()

        app.tick(now, dt)
        frame = app.scene.render(app.battle, app.menu, app.view,
                                 message=app.message, cursor=cursor,
                                 pose=pose, hold=hold, cam=cam, mode=mode,
                                 fps=fps, arena=app.arena_image(now))
        cv2.imshow(win, frame)
        k = cv2.waitKeyEx(1)
        if k in (ord("s"), ord("S")):                 # screenshot to share
            out = pathlib.Path(__file__).resolve().parents[1] / "snapshots"
            out.mkdir(exist_ok=True)
            path = out / f"game_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame)
            print("saved", path.name, flush=True)
            continue
        if k in (ord("c"), ord("C")) and cam is not None and cams:
            nxt = (cam.index + 1) % len(cams)
            print(f"switching to camera [{nxt}] {cams[nxt][0]}")
            cam.switch(nxt, cams[nxt][0])
            tracker, last_seq = GestureTracker(), -1
        elif k != -1 and not app.key(k, now):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    if cam:
        cam.close()
    if grid:
        grid.close()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--you", help=f"type your party, e.g. {DEFAULT_YOU}")
    ap.add_argument("--cpu", help=f"type the CPU party, e.g. {DEFAULT_CPU}")
    ap.add_argument("--grid-cam", type=int,
                    help="phone camera index (default: the iPhone)")
    ap.add_argument("--flip", action="store_true",
                    help="phone mounted the other way: CPU half at the bottom")
    ap.add_argument("--figurines", action="store_true",
                    help="show the built-in 3D figurines instead of the "
                         "animated game sprites")
    ap.add_argument("--no-grid", action="store_true",
                    help="don't use the phone; play the default parties")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cam", type=int, help="hand camera index (default: auto)")
    ap.add_argument("--mouse", action="store_true",
                    help="no camera; the mouse stands in for your hand")
    ap.add_argument("--list-cams", action="store_true",
                    help="print cameras in OpenCV index order and exit")
    args = ap.parse_args()
    if args.list_cams:
        cams = list_cameras(wait_for_phone_s=4)
        hand, grid = pick_hand_camera(cams), pick_grid_camera(cams)
        for i, (name, model) in enumerate(cams):
            tag = ("   ← hand cam" if i == hand else
                   "   ← grid cam" if i == grid else "")
            print(f"[{i}] {name:28} {model}{tag}")
        return
    try:
        run(args)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nBye!")


if __name__ == "__main__":
    main()
