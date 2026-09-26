# Pokémon AR Card Battle

Turn-based Pokémon battles played with real trading cards on a table, in
augmented reality. An iPhone looks down at the board and reads which card is in
which slot. Each Pokémon then appears standing on its card and fights with
animated attacks. You control the battle with hand gestures in front of your
Mac's camera.

- **Battle engine:** Switch/3DS-style singles against a CPU: 10 Pokémon, 40 moves,
  type chart, status conditions, items, switching.
- **Card reader:** an iPhone (Continuity Camera) films the back of a Monopoly
  board. The far half is the CPU's team and the near half is yours, 3×2 slots each.
  Cards are recognised with OpenCV. Once you start, the parties are locked in, so
  bumping a card changes nothing.
- **Hand controls:** your Mac's camera and MediaPipe Hands. Move your open hand to
  move the cursor, pinch to pick, hold a fist to go back, and hold a peace sign for
  move info.
- **AR:** the phone's position is worked out from the board alone. Animated
  Pokémon stand on their cards; move effects fly between them, and super-effective
  hits shake the view.

This is a fan project made for fun and not affiliated with Nintendo, Game Freak
or The Pokémon Company.

## Requirements

- An Apple Silicon Mac with a built-in camera.
- An iPhone used as a Continuity Camera. A cable is the most reliable connection.
- Python 3.12 (`brew install python@3.12`). MediaPipe has no builds for 3.13 or newer.
- The back of a Monopoly board (or any plain cardboard square of about 50 cm)
  and your Pokémon cards.

## Setup

```bash
git clone https://github.com/probablynotmarti/pokemon-ar.git
cd pokemon-ar
./setup.sh
```

`setup.sh` does four things:
1. Creates a `.venv`.
2. Installs the pinned dependencies (`requirements.txt`).
3. Downloads the Pokémon sprites and battle effects from their original sources
   (`tools/fetch_assets.py`).
4. Runs the tests.

**Register your cards once.** Recognition compares each card against a photo of your
own copy, so each card needs one. With the iPhone connected and pointed at the
board, put a card in **your** slot 1 (the near half, top-left) and press Space:

```bash
.venv/bin/python tools/grid_cam.py --register seedot     # marshadow, riolu, …
```

The Pokémon names are listed in `game/data.py`. Use `tools/grid_cam.py` on its own
to check that every card turns green.

## Play

Run this from **macOS Terminal**, which needs Camera permission:

```bash
./play.sh                  # iPhone reads the board, hand gestures
./play.sh --mouse          # mouse instead of gestures
./play.sh --you 1=riolu,2=seedot --cpu 1=solrock    # no iPhone: virtual board
./play.sh --figurines      # built-in low-poly 3D figurines instead of sprites
```

The **setup screen** shows the live board with a Pokémon on every recognised card.
Pinch **START BATTLE** once every card is green.

**Controls:**

| Gesture | Keyboard | Does |
|---|---|---|
| Open hand | Arrow keys | Move the cursor |
| Pinch | Enter | Pick / skip text |
| Hold fist | Esc | Back |
| Hold peace sign | `i` | Info card |

Other keys: `c` next camera · `f` swap board halves · `s` screenshot · `r` rematch · `q` quit.

## Layout

| Path | What |
|---|---|
| `game/battle.py`, `data.py`, `cpu.py` | Battle rules, Pokémon and moves, CPU brain |
| `game/gestures.py`, `menu.py` | Hand landmarks → cursor and buttons; menu state machine |
| `game/vision/` | Board and card detection, iPhone camera thread |
| `game/ar/` | Camera pose, sprites, figurines, move effects, animation stage |
| `tools/hand_ui.py` | The game |
| `tools/grid_cam.py` | Check the iPhone's view of the board; register cards |
| `GAME_DESIGN.md` | Full design and decisions log |

Tests: `.venv/bin/python -m unittest discover -s tests -t .` (about 116 tests). The
vision tests need your own card and board photos, so they skip on a fresh clone.

## Art and credits

The repo contains **code only**. Sprites and battle effects are downloaded by
`setup.sh`; see [`assets/ATTRIBUTION.md`](assets/ATTRIBUTION.md).

- Pokémon sprites: animated 3D-model renders from [Project Pokémon](https://projectpokemon.org/home/docs/spriteindex_148/)
  and [Pokémon Showdown](https://play.pokemonshowdown.com/sprites/). Pokémon © Nintendo / Creatures / GAME FREAK.
- Battle effects: [Pokémon Showdown](https://play.pokemonshowdown.com/fx/) and the
  [Kenney Particle Pack](https://kenney.nl/assets/particle-pack) (CC0, included).
