"""Battle menu state machine (GAME_DESIGN.md §6 "Menu map").

Input-agnostic: the gesture tracker, keyboard, or mouse all drive it
through point() / move() / confirm() / back() / toggle_info(). Tiles live
in panel coordinates [0,1]²; the cursor snaps to the nearest tile.

When a pick completes the player's choice, the menu runs the battle turn
itself, returns the events, and locks until the UI calls resume() after
it has finished animating them.
"""

from dataclasses import dataclass

from .battle import STATUS_TAGS, Fight, Run, Switch, UseItem
from .data import GRID_COLS, GRID_ROWS, MOVES

MENU_CONFIG = {
    "cooldown_s": 0.3,     # ignore confirm/back right after a screen change
    "dwell_s": 0.12,       # cursor must sit on a new tile this long to hover
}

BACK_TO = {"fight": "main", "bag": "main", "party": "main",
           "run_confirm": "main", "item_target": "bag",
           "party_action": "party"}


@dataclass(frozen=True)
class Tile:
    id: str
    label: str
    rect: tuple            # (x, y, w, h) in panel coords
    enabled: bool = True
    sub: str = ""          # second line (type/PP, HP, count…)
    kind: str = ""         # move | item | mon | action
    payload: object = None

    @property
    def center(self):
        x, y, w, h = self.rect
        return (x + w / 2, y + h / 2)


def grid_rect(i, cols, rows, pad=0.02):
    c, r = i % cols, i // cols
    w, h = 1 / cols, 1 / rows
    return (c * w + pad, r * h + pad, w - 2 * pad, h - 2 * pad)


class Menu:
    def __init__(self, battle, cfg=None):
        self.battle = battle
        self.cfg = {**MENU_CONFIG, **(cfg or {})}
        self.info = False
        self.item = None            # item picked in BAG
        self.party_idx = None       # Pokémon picked in PARTY
        self.last_events = []
        self.resume(-1e9)

    # ── state ──
    def _go(self, screen, t):
        self.screen = screen
        self.hovered = None
        self._candidate, self._candidate_since = None, t
        self._changed_at = t

    def resume(self, t):
        """Call after the UI has finished animating a turn's events."""
        b = self.battle
        if b.winner:
            self._go("game_over", t)
        elif b.needs_switch(b.player):
            self._go("forced_switch", t)
        else:
            self._go("main", t)

    def _ready(self, t):
        return t - self._changed_at >= self.cfg["cooldown_s"]

    # ── tiles for the current screen ──
    def tiles(self):
        b, side = self.battle, self.battle.player
        legal = b.legal_actions(side) if not b.winner else []
        s = self.screen
        if s == "main":
            names = ("FIGHT", "BAG", "POKÉMON", "RUN")
            return [Tile(n.lower(), n, grid_rect(i, 2, 2), kind="action")
                    for i, n in enumerate(names)]
        if s == "fight":
            mon = side.mon
            if Fight("Struggle") in legal:
                return [Tile("move:Struggle", "Struggle", (0.02, 0.02, .96, .96),
                             kind="move", payload=Fight("Struggle"))]
            out = []
            for i, m in enumerate(mon.species.moves):
                mv = MOVES[m]
                out.append(Tile(f"move:{m}", m, grid_rect(i, 2, 2),
                                Fight(m) in legal,
                                f"{mv.type} · PP {mon.pp[m]}/{mv.pp}",
                                "move", Fight(m)))
            return out
        if s == "bag":
            out = []
            for i, (name, count) in enumerate(side.bag.items()):
                usable = any(isinstance(a, UseItem) and a.item == name
                             for a in legal)
                out.append(Tile(f"item:{name}", name, grid_rect(i, 3, 2),
                                usable, f"x{count}", "item", name))
            return out
        if s in ("item_target", "party", "forced_switch"):
            make = {"item_target": lambda i: UseItem(self.item, i),
                    "party": lambda i: Switch(i),
                    "forced_switch": lambda i: Switch(i)}[s]
            out = []
            for i, mon in enumerate(side.party):
                # PARTY: every tile is pickable (for INFO); SWITCH decides
                ok = s == "party" or make(i) in legal
                tag = STATUS_TAGS.get(mon.status, "")
                sub = "FAINTED" if mon.fainted else \
                    f"HP {mon.hp}/{mon.max_hp} {tag}".strip()
                if i == side.active and s == "party":
                    sub = f"{mon.hp}/{mon.max_hp} · active"
                out.append(Tile(f"mon:{i}", mon.name,
                                grid_rect(mon.slot - 1, GRID_COLS, GRID_ROWS),
                                ok, sub, "mon", i))
            return out
        if s == "party_action":
            can = Switch(self.party_idx) in legal
            return [Tile("switch", "SWITCH", grid_rect(0, 3, 1), can,
                         kind="action"),
                    Tile("summary", "INFO", grid_rect(1, 3, 1), kind="action"),
                    Tile("cancel", "CANCEL", grid_rect(2, 3, 1), kind="action")]
        if s == "run_confirm":
            return [Tile("yes", "YES, FORFEIT", grid_rect(0, 2, 1),
                         kind="action"),
                    Tile("no", "NO", grid_rect(1, 2, 1), kind="action")]
        return []                   # locked / game_over

    def tile(self, tile_id):
        return next((t for t in self.tiles() if t.id == tile_id), None)

    # ── input ──
    def nearest(self, cursor):
        tiles = self.tiles()
        if not tiles or cursor is None:
            return None
        return min(tiles, key=lambda t: (t.center[0] - cursor[0]) ** 2
                   + (t.center[1] - cursor[1]) ** 2)

    def point(self, cursor, t):
        """Cursor moved (panel coords). Hover changes after a short dwell."""
        near = self.nearest(cursor)
        if near is None:
            return
        if self.hovered is None:
            self.hovered = near.id
        if near.id == self.hovered:
            self._candidate = None
            return
        if near.id != self._candidate:
            self._candidate, self._candidate_since = near.id, t
        elif t - self._candidate_since >= self.cfg["dwell_s"]:
            self.hovered, self._candidate = near.id, None

    def move(self, dx, dy):
        """Keyboard: hop to the nearest tile in direction (dx, dy)."""
        tiles = self.tiles()
        if not tiles:
            return
        cur = self.tile(self.hovered) if self.hovered else None
        if cur is None:
            self.hovered = tiles[0].id
            return
        cx, cy = cur.center
        best, best_d = None, None
        for t in tiles:
            vx, vy = t.center[0] - cx, t.center[1] - cy
            if vx * dx + vy * dy <= 1e-6:       # not in that direction
                continue
            d = vx * vx + vy * vy + 3 * (vx * dy - vy * dx) ** 2
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best:
            self.hovered = best.id

    def toggle_info(self):
        self.info = not self.info

    def back(self, t):
        if not self._ready(t) or self.screen not in BACK_TO:
            return False
        self.info = False
        self._go(BACK_TO[self.screen], t)
        return True

    def confirm(self, t, cursor=None):
        """Pick the hovered tile (or the tile under `cursor`, used for the
        pinch look-back). Returns battle events if a turn was played."""
        if not self._ready(t):
            return None
        tile = self.nearest(cursor) if cursor is not None else \
            (self.tile(self.hovered) if self.hovered else None)
        if tile is None or not tile.enabled:
            return None
        self.hovered = tile.id
        s, b = self.screen, self.battle
        if s == "main":
            nxt = {"fight": "fight", "bag": "bag", "pokémon": "party",
                   "run": "run_confirm"}[tile.id]
            self._go(nxt, t)
        elif s == "fight":
            return self._play(tile.payload, t)
        elif s == "bag":
            self.item = tile.payload
            self._go("item_target", t)
        elif s == "item_target":
            return self._play(UseItem(self.item, tile.payload), t)
        elif s == "party":
            self.party_idx = tile.payload
            self._go("party_action", t)
        elif s == "party_action":
            if tile.id == "switch":
                return self._play(Switch(self.party_idx), t)
            if tile.id == "summary":
                self.info = True
            else:
                self._go("party", t)
        elif s == "run_confirm":
            if tile.id == "yes":
                return self._play(Run(), t)
            self._go("main", t)
        elif s == "forced_switch":
            ev = b.replace_fainted(b.player, tile.payload)
            self.last_events = ev
            self._go("locked", t)
            return ev
        return None

    def _play(self, action, t):
        ev = self.battle.play_turn(action)
        self.last_events = ev
        self.info = False
        self._go("locked", t)
        return ev

    # ── info card text ──
    def info_lines(self):
        """Detail card for the hovered tile (peace sign ✌️)."""
        b = self.battle
        tile = self.tile(self.hovered) if self.hovered else None
        if self.screen == "party_action":
            tile = Tile("mon", "", (0, 0, 0, 0), kind="mon",
                        payload=self.party_idx)
        if tile is None:
            return []
        if tile.kind == "move":
            name = tile.payload.move
            if name == "Struggle":
                return ["Struggle", "Typeless · 50 power · 1/4 recoil"]
            mv, user, foe = MOVES[name], b.player.mon, b.cpu.mon
            lines = [name, f"{mv.type} · {mv.category.title()} · "
                     f"Power {mv.power or '—'} · Acc {mv.accuracy or '—'}"]
            if mv.category != "status":
                eff = b._eff(mv, foe)
                word = ("No effect" if eff == 0 else "Super effective" if eff > 1
                        else "Not very effective" if eff < 1 else "Normal damage")
                pct = 100 * b.estimate_damage(user, foe, name) / foe.max_hp
                lines.append(f"vs {foe.name}: {word} (~{pct:.0f}% HP)")
            if mv.desc:
                lines.append(mv.desc)
            return lines
        if tile.kind == "mon":
            mon = b.player.party[tile.payload]
            return [f"{mon.name} · slot {mon.slot} · "
                    f"{'/'.join(mon.species.types)}",
                    f"HP {mon.hp}/{mon.max_hp}"
                    + (f" · {STATUS_TAGS[mon.status]}" if mon.status else ""),
                    "Moves: " + ", ".join(mon.species.moves)]
        if tile.kind == "item":
            from .data import ITEMS
            return [tile.payload, ITEMS[tile.payload].desc]
        return []
