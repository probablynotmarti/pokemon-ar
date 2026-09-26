# Pokémon AR Battle: Game Design

Two cameras, one table:

| Camera | Device | Job |
|---|---|---|
| **Grid cam** | iPhone (shows up on the Mac as a webcam via Continuity Camera) | Looks down at the mat, reads which card sits in which slot, and gives the AR view the 3D models are drawn into |
| **Hand cam** | Mac built-in camera | MediaPipe Hands tracks the player's hand. Hand gestures drive the battle menus |

Stack: Python, OpenCV (grid and card recognition), MediaPipe Hands (gestures). All rules
live in pure Python under `game/` so they can be tested without a camera.

---

## 1. The board (back of a Monopoly board, 3 × 2 zones per side)

The play surface is the **plain brown back of a standard Monopoly board** (≈508 mm / 20 in
square, with a fold cross). The iPhone is fixed above it. **The far half, the top of the phone
image, is the CPU's. The near half is yours.** The horizontal fold divides them. Each half is
cut into a **3 × 2 grid of zones** (≈169 × 127 mm). A card belongs to whichever zone its centre
is in, so placement can be loose.

```
                 (far side: top of the phone image)
          ┌───────────┬───────────┬───────────┐
          │  1 LEAD   │     2     │     3     │   CPU HALF
          ├───────────┼───────────┼───────────┤   (you place its cards)
          │     4     │     5     │     6     │
          ╞═══════════╪═══════════╪═══════════╡ ← fold
          │  1 LEAD   │     2     │     3     │   YOUR HALF
          ├───────────┼───────────┼───────────┤
          │     4     │     5     │     6     │
          └───────────┴───────────┴───────────┘
                         YOUR SEAT
```

- Both halves are numbered in reading order **as seen from your seat**. **Slot 1 (top-left) is
  the lead.** If slot 1 is empty, the lowest filled slot leads.
- **The cards on the board are the whole party.** Empty zones are fine, and duplicate cards are
  separate Pokémon.
- Phone mounted the other way round? Press **f** on the setup screen (or use `--flip`) to swap
  the halves.
- **Party lock (decided):** the setup screen shows the live board with the zone grid and each
  card's name. **START** unlocks once the board is found, each side has at least one card, and
  no card shows "?". Pinch START to lock both parties. From then until the battle ends, **card
  reading stops**: sliding, bumping or removing a card changes nothing, and swaps only happen
  through the POKÉMON menu. Rematch returns to setup.

**How the phone reads it** (`game/vision/`, OpenCV; MediaPipe has no card detector):
- **Finding the board** (`board.py`): it's the big uniform-coloured region in the middle of the
  frame, and its outline gives the 4 corners, which straighten it into a top-down view. The
  phone is fixed, so the board only "moves" if a new position holds for about 12 frames. A hand
  over a corner (skin is close to cardboard brown) never gets that far.
- **Finding cards**: a card's silver border is brighter and less yellow than the board. Each pixel
  is compared with the *local* board colour, so glare and shadows don't count, and only pixels
  near fine detail count, so glare can't join cards together.
- **Per zone**, in order:
  1. A clean card shape is cut out and recognized (SIFT keypoints + RANSAC against a reference
     photo, with a colour-thumbnail fallback for small or blurry cards).
  2. Otherwise, a zone with any detail is searched as a whole for a known card.
  3. Otherwise, a card-shaped patch that's detailed all the way through (art, text) is cut out
     and tried. This catches orange cards that blend with the board.
  4. A card shape that can't be recognized is **"?"**. Anything else (cardboard grain, the fold,
     a shadow) is empty. A card with no reference is never read as a different card, and a
     test covers this.
- **Voting**: a zone is only trusted after the same result in 3 of the last 5 reads.
- **References** are in `assets/cards/<key>.jpg`, cut from your card photos. To recapture one
  through the phone itself, put the card in your slot 1 and run
  `tools/grid_cam.py --register <key>`, then press Space.

---

## 2. Roster and movesets

Stats use the Switch/3DS formula at **level 50** (31 IVs, 0 EVs, neutral nature). Each Pokémon
has 4 moves. Menu order is top-left, top-right, bottom-left, bottom-right. The
card's own attack is included where the video games have an equivalent.

### Registeel · Steel · HP 155 · card `s10D-047`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| **Heavy Slam** | Steel | Phys | 40–120 | 100 | 10 | Power from weight ratio (Registeel is 205 kg, so 120 vs everyone in this set). Card: 重磅衝撞 |
| Flash Cannon | Steel | Spec | 80 | 100 | 10 | 10% target Sp. Def −1 |
| Hammer Arm | Fighting | Phys | 100 | 90 | 10 | User Speed −1 |
| Iron Defense | Steel | Status | — | — | 15 | User Defense +2 |

### Sudowoodo · Rock · HP 145 · card `s10D-034`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| Rock Slide | Rock | Phys | 75 | 90 | 10 | 30% flinch |
| **Double-Edge** | Normal | Phys | 120 | 100 | 15 | User takes 1/3 recoil. Card: 捨身衝撞 (self-damage) |
| Brick Break | Fighting | Phys | 75 | 100 | 15 | — |
| Sucker Punch | Dark | Phys | 70 | 100 | 5 | **Priority +1** (always first). Fails if the target isn't attacking |

### Ducklett · Water/Flying · HP 137 · card `s10D-017`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| **Bubble Beam** | Water | Spec | 65 | 100 | 20 | 10% target Speed −1. Card: 潑水 |
| Air Slash | Flying | Spec | 75 | 95 | 15 | 30% flinch |
| Feather Dance | Flying | Status | — | 100 | 15 | Target Attack −2 |
| Roost | Flying | Status | — | — | 5 | Heal 50% max HP |

### Skwovet · Normal · HP 145 · card `s10D-057`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| **Bite** | Dark | Phys | 60 | 100 | 25 | 30% flinch. Card: 咬 |
| Body Slam | Normal | Phys | 85 | 100 | 15 | 30% paralysis |
| Super Fang | Normal | Phys | — | 90 | 10 | Halves the target's current HP |
| **Stuff Cheeks** | Normal | Status | — | — | 10 | **Eats an Oran Berry from your Bag**: heals 10 HP, Defense +2. Fails with no berries |

### Politoed · Water · HP 165 · card `s10D-013`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| **Hydro Pump** | Water | Spec | 110 | 80 | 5 | Card: 水炮濺射 |
| Ice Beam | Ice | Spec | 90 | 100 | 10 | 10% freeze |
| Hyper Voice | Normal | Spec | 90 | 100 | 10 | Stands in for the card's 君主樂舞 |
| Hypnosis | Psychic | Status | — | 60 | 20 | Sleep for 1–3 turns |

### Marshadow · Fighting/Ghost · HP 165 · card `m1L-069` (AR)
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| Spectral Thief | Ghost | Phys | 90 | 100 | 10 | Steals the target's stat boosts before hitting |
| Close Combat | Fighting | Phys | 120 | 100 | 5 | User Def −1 and Sp. Def −1 |
| **Shadow Sneak** | Ghost | Phys | 40 | 100 | 30 | **Priority +1**. Stands in for the card's シャドーソバット |
| Bulk Up | Fighting | Status | — | — | 20 | User Atk +1 and Def +1 |

### Gimmighoul · Ghost · HP 120 · card `m1L-019`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| Shadow Ball | Ghost | Spec | 80 | 100 | 15 | 20% target Sp. Def −1 |
| Hex | Ghost | Spec | 65 | 100 | 10 | **130 power** if the target has a status condition |
| Power Gem | Rock | Spec | 80 | 100 | 20 | Coverage: without it, Normal types would take 0 from Gimmighoul |
| Nasty Plot | Dark | Status | — | — | 20 | User Sp. Atk +2 |

### Riolu · Fighting · HP 115 · card `m1L-028`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| Force Palm | Fighting | Phys | 60 | 100 | 10 | 30% paralysis |
| **Quick Attack** | Normal | Phys | 40 | 100 | 30 | **Priority +1**. Card: かそくづき |
| Reversal | Fighting | Phys | 20–200 | 100 | 15 | The lower Riolu's HP, the stronger (200 at under 4% HP) |
| Crunch | Dark | Phys | 80 | 100 | 15 | 20% target Def −1. Coverage: Riolu's other attacks can't touch Ghosts |

### Seedot · Grass · HP 115 · card `m1L-007`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| **Seed Bomb** | Grass | Phys | 80 | 100 | 15 | Card: タネばくだん |
| **Synthesis** | Grass | Status | — | — | 5 | Heal 50% max HP. Card: ひとやすみ (heal) |
| Leech Seed | Grass | Status | — | 90 | 10 | End of every turn: target loses 1/8 max HP and your active Pokémon heals that much. Grass types are immune. Ends when the target switches out |
| Explosion | Normal | Phys | 250 | 100 | 5 | **Seedot faints**, whether or not it hits |

### Solrock · Rock/Psychic · HP 165 · card `m1L-027`
| Move | Type | Cat | Pow | Acc | PP | Effect |
|---|---|---|---|---|---|---|
| Rock Slide | Rock | Phys | 75 | 90 | 10 | 30% flinch |
| **Zen Headbutt** | Psychic | Phys | 80 | 90 | 15 | 20% flinch. Stands in for the card's コスモビーム |
| Flare Blitz | Fire | Phys | 120 | 100 | 15 | 1/3 recoil, 10% burn |
| Cosmic Power | Psychic | Status | — | — | 20 | User Def +1 and Sp. Def +1 |

Learnsets are close to the games but not exact. The rule used: every Pokémon must be able to
damage every other card in the roster, and a test enforces it.

Source of truth: `game/data.py`. To add a card, add one `Species` entry with 4 moves. The
tests check that every move exists.

---

## 3. Battle rules (same as the 3DS/Switch games unless noted)

**Turn flow**
1. Both sides pick an action at the same time: **Fight** (a move), **Bag** (an item),
   **Pokémon** (switch), or **Run** (forfeit).
2. Resolve in this order: **switches → items → moves**.
3. Moves go by **priority bracket** (Sucker Punch / Shadow Sneak / Quick Attack +1,
   everything else 0). Inside a bracket,
   **who goes first is a coin flip each turn**. That is the "speed is random" rule. Speed stats
   are stored for later but not used yet.
4. End of turn: burn (−1/16 max HP), then Leech Seed (−1/8, heals the other side's active
   Pokémon), then faint checks.
5. **Fainted → forced switch.** The POKÉMON menu opens and can't be cancelled. This switch is
   free (it doesn't cost a turn).
6. **Win** when every Pokémon on the other mat has fainted. RUN = forfeit.
   If both last Pokémon faint at the same moment: a **recoil** KO (Double-Edge, Flare Blitz)
   counts as a win for the attacker. **Explosion** counts as a loss for the Pokémon that used
   it. A double faint from end-of-turn damage counts as your loss.

**CPU brain** (`game/cpu.py`)
- At 25% HP or below, it heals half the time (Super Potion, then Potion, then Oran Berry).
- Otherwise it scores each move: expected % of your HP, with a bonus for a KO. Status moves
  are scored with simple rules, e.g. Roost only when it's below half HP, and Hypnosis only if
  you have no status yet.
- It plays the best move 75% of the time and a random move 25% of the time, so it isn't
  perfectly predictable.
- It never switches by choice. Fainted Pokémon are replaced in slot order.

**Damage** (`game/damage.py`, the Gen 5+ formula):
`((2·50/5+2) · Power · A/D / 50 + 2) × crit 1.5 (1/24 chance) × random 0.85–1.00 × STAB 1.5 × type`.
Physical moves use Atk/Def and special moves use SpA/SpD. The full 18-type chart is included.

**HP bar:** green above 50%, yellow 20–50%, red below 20%. It drains smoothly like the games,
and shows the Pokémon name, "Lv50", HP numbers (yours only), and a status tag (SLP / PAR / FRZ / BRN).

**Status conditions**
| Status | Effect |
|---|---|
| Sleep | Can't move for 1–3 turns |
| Paralysis | 25% chance to be fully paralyzed each turn (the speed drop is skipped while speed is random) |
| Freeze | Can't move. 20% chance to thaw each turn |
| Burn | Loses 1/16 max HP each turn. Its physical attacks do half damage. Fire types are immune |
| Flinch | Loses its move this turn. Only works if the flincher moved first |
| Leech Seed | Not a status (it stacks with one). Loses 1/8 max HP each turn to the other side. Ends when it switches out |

**Stat stages** −6…+6 (Atk, Def, SpA, SpD, Spe) reset when a Pokémon switches out, like
the games. **PP** is tracked per move. With 0 PP on all 4 moves, the Pokémon uses Struggle.

---

## 4. Bag (inventory)

Each player starts every battle with this (tunable in `STARTING_BAG`):

| Item | Qty | Effect | Can target |
|---|---|---|---|
| Potion | 3 | +20 HP | Non-fainted |
| Super Potion | 1 | +60 HP | Non-fainted |
| Oran Berry | 3 | +10 HP (Skwovet's Stuff Cheeks also eats these) | Non-fainted |
| Full Heal | 1 | Cures sleep, paralysis, freeze, and burn | Non-fainted |
| Revive | 1 | Fainted → 50% HP | Fainted only |

- **Using an item takes your turn**, like the games. Items resolve after switches and before
  moves.
- An item can target **any** Pokémon on your mat, not just the active one. Revive is the
  reason you'd pick a benched one.
- Items with 0 left are greyed out and can't be picked.
- The CPU has the same bag.

---

## 5. Swapping

- **Voluntary switch** (POKÉMON → pick → Switch): takes your turn and happens **before** any
  moves, so the incoming Pokémon takes the opponent's hit (this is how switching into a
  resisted hit works).
- You can't pick the active Pokémon or a fainted one. Those tiles are dimmed.
- **Forced switch** after a faint is free, and the menu can't be backed out of.
- The POKÉMON menu uses the **same 3 × 2 layout as your physical mat**, so the Pokémon in
  slot 5 of the table is in slot 5 on screen.
- AR: the outgoing model shrinks back into its card (the classic red recall beam), then the new
  one rises out of its card and walks into the arena.

---

## 6. Hand-gesture controls (Mac camera)

The idea: **hand position is the cursor, hand shape is the button.** It works like a
controller: open palm = move the cursor, pinch = A, fist = B, peace sign = info.

| Gesture | Shape (MediaPipe landmarks) | Acts as | What it does |
|---|---|---|---|
| ✋ **Open hand** (neutral) | Any relaxed hand | Cursor | Moving your hand moves the highlight between tiles |
| 🤏 **Pinch** | Thumb tip (4) touches index tip (8) | **A: confirm** | Picks the highlighted tile |
| ✊ **Fist** (hold 0.4 s) | All 4 fingertips curled past their middle joints | **B: back** | Goes up one menu. Does nothing in a forced switch |
| ✌️ **Peace** (hold 0.4 s) | Index and middle up, ring and pinky curled | **Info** | Toggles a detail card: move type, power, accuracy, PP, and "super effective?" against the current foe. On a Pokémon it shows HP, status, and moves |
| *(no hand)* | — | — | Cursor hidden, nothing fires. Safe to drop your hand |

**Why this set:** a pinch is the most reliable click MediaPipe gives you (it's the Vision Pro
click). Using the hand's position rather than a pointing finger for the cursor means the
cursor doesn't jump when you pinch.

### Cursor mechanics
- The cursor anchor is the **palm center**: the average of landmarks 0, 5, 9, 17. It is not a
  fingertip, so pinching and fist-making don't move it.
- The image is mirrored (moving right moves the cursor right). The **middle 60% of the camera
  frame** maps to the full menu, so small wrist movements are enough.
- A One-Euro filter smooths the cursor so it's steady when still and quick when moving.
- **Tile snapping:** the cursor highlights whichever tile it is nearest to. Nothing needs pixel
  precision. The highlight only moves after the cursor has been over the new tile for 120 ms,
  which stops flicker at borders.

### Anti-misfire rules
- **Edge-triggered:** a pinch fires once, when it starts. To pick again, you have to open your
  hand first. A pinch held from one menu into the next never counts.
- **Hysteresis:** a pinch starts when the thumb–index gap is under 0.25 × hand size and ends
  when it's over 0.40 × hand size. Hand size is the distance from the wrist (0) to the middle
  knuckle (9), so this works at any distance from the camera.
- **Freeze on pinch:** confirming uses the tile that was highlighted 100 ms *before* the pinch
  began, so any drift while closing your fingers doesn't count.
- **Menu cooldown:** gestures are ignored for 300 ms after any screen change.
- **Locked during animations:** no input is taken while moves play out.
- **Keyboard fallback** for development: arrow keys, Enter = A, Esc = B, I = info.

### Menu map

```
BATTLE MENU (2×2, bottom-right like the games)
┌───────────┬───────────┐
│  FIGHT    │   BAG     │
├───────────┼───────────┤
│ POKÉMON   │   RUN     │
└───────────┴───────────┘

FIGHT ─ pinch ─▶ MOVES (2×2: the 4 moves, type-colored, PP shown)
                   └ pinch a move ─▶ turn locked in ─▶ battle animation
BAG ─── pinch ─▶ ITEMS (grid of 5, count badges)
                   └ pinch item ─▶ PICK TARGET (3×2 mat layout, invalid dimmed)
                                     └ pinch ─▶ turn locked in
POKÉMON pinch ─▶ PARTY (3×2 mat layout)
                   └ pinch ─▶ [ SWITCH | INFO | CANCEL ]
                                     └ SWITCH ─▶ turn locked in
RUN ─── pinch ─▶ "Forfeit the battle?" [ YES | NO ]

✊ fist = back one level from anywhere except "turn locked in" and the forced switch.
```

State machine: `MAIN → {FIGHT, BAG → ITEM_TARGET, PARTY → PARTY_ACTION, RUN_CONFIRM} →
LOCKED → ANIMATING → (FORCED_SWITCH if fainted) → MAIN`.

---

## 7. Balance check (from `tools/balance_table.py`)

Each cell is the best move's max-roll damage as a % of the target's HP (no crit, Explosion
left out). Rows attack columns. `avg` = the attacker's average across the roster.

```

            Registee Sudowood Ducklett  Skwovet Politoed Marshado Gimmigho    Riolu   Seedot  Solrock   avg
Registeel        33%      84%      39%      78%      27%      47%      70%     110%      94%      73%   65%
Sudowoodo        32%      42%     127%      74%      41%      23%      71%      93%      80%      44%   62%
Ducklett         11%      46%      35%      41%      16%      37%      30%      95%     114%      41%   46%
Skwovet           8%      11%      45%      40%      27%      12%      40%      62%      54%      24%   32%
Politoed         31%     132%      46%     101%      23%      45%      75%     117%     154%     116%   84%
Marshadow        90%     120%      91%     215%      74%     106%     161%     167%     144%     101%  126%
Gimmighoul       20%      42%      71%      42%      26%      58%      96%      73%      88%      73%   58%
Riolu            28%      38%      34%      67%      24%      20%      61%      53%      45%      38%   40%
Seedot            6%      34%      35%      31%      43%      20%      30%      47%      20%      40%   30%
Solrock          47%      31%     122%      56%      40%      76%      57%     179%     153%      34%   79%
```

What the numbers show:
- **Marshadow is the strongest by a wide margin** (126% avg: it one-shots most of the roster).
  That fits a mythical Pokémon on the rarest card. If it feels unfair, the fixes are: allow only
  one Marshadow per mat, or play it at Lv 40.
- **Unevolved cards are weak**, like in the games (Skwovet, Seedot and Riolu average 30–40%).
  They have tricks instead: Super Fang (always takes half the target's HP), Explosion,
  Leech Seed, Reversal at low HP, and priority moves.
- Politoed, Solrock, Registeel and Sudowoodo form the solid middle.
- Other levers if a match feels lopsided: set level by evolution stage, or use the **HP printed
  on the card** instead of game HP.

---

## 8. Build order

1. ✅ `game/battle.py` + `game/cpu.py`: turn engine and CPU AI. Terminal version:
   `python3 tools/play_cli.py`.
2. ✅ `game/gestures.py`: landmarks → cursor + pinch/fist/peace, with the hysteresis, hold
   times, look-back and One-Euro smoothing from §6. `game/menu.py`: the menu state machine.
   Both are tested with synthetic hand landmarks (`tests/hands.py`).
3. ✅ Hand cam window, `tools/hand_ui.py`: MediaPipe 0.10.21 (1.x crashes on this Mac), camera
   preview with the hand skeleton, menu with cursor, HP bars that drain, text you skip with a
   pinch, rematch. It has mouse and keyboard fallbacks. The arena is a stand-in for the AR view.
4. ✅ Grid cam (iPhone via Continuity Camera) on the Monopoly-board back: board and card
   detection, card recognition, setup screen with party lock. The arena shows the live board:
   the active card glows in its type colour and fainted cards are crossed out. Tested on your
   real empty-board photo and on synthetic phone frames (glare, shadow, tilt, 720p, a hand over
   a corner, upside-down and unknown cards). Not yet run live with the phone.
5. ✅ AR figurines (`game/ar/`): the camera pose is solved from the board alone (focal
   length from the square's homography, then planar PnP; checked on your real frame:
   ≈45° FOV, camera ≈45 cm behind and 36 cm above the near edge). There are 10 low-poly
   figurines built from primitives, drawn by a small software renderer (flat shading and a
   contact shadow, 1–2 ms each). The stage animates them from battle events: send-out spin,
   idle bob, facing the opponent, attack lunge / status hop, hit flash and shake, faint,
   recall, and status tints. On the setup screen every recognized card gets its figurine; in
   battle only the two active ones show, at 1.5× size. Without the phone the same stage plays
   on a virtual board.
   **Real models:** drop `assets/models/<pokemon>.glb` (or .gltf/.obj/.ply/.stl) in and it
   replaces that figurine. It's stood up, centred, scaled to ~70 mm, and coloured from its
   textures. Models from games are Nintendo IP: fine locally, never in a public repo.
   Preview: `tools/model_viewer.py` (lineup) or `--photo <frame>` (on real cards).

### How to run

```bash
cd ~/Desktop/pokemon-ar
python3 -m unittest discover -s tests -t .                      # all tests
~/.venvs/memecv/bin/python -m unittest discover -s tests -t .  # + vision tests
~/.venvs/memecv/bin/python tools/grid_cam.py                    # check the phone's view of the board
~/.venvs/memecv/bin/python tools/model_viewer.py                # look at the figurines
~/.venvs/memecv/bin/python tools/hand_ui.py                     # full game
~/.venvs/memecv/bin/python tools/hand_ui.py --mouse             # no camera
~/.venvs/memecv/bin/python tools/hand_ui.py --you 1=marshadow,2=seedot --cpu 1=solrock,2=riolu   # no phone
```

Run it from macOS Terminal, since that is what has Camera permission. Cameras are picked by
name, using OpenCV's own order (sorted by device uniqueID): FaceTime for your hands, iPhone
for the grid. `--list-cams` shows the assignment, `--cam N` / `--grid-cam N` override it, and
**c** cycles the hand camera while playing.

6. ✅ Game sprites + move effects:
   - **Pokémon look like the games:** animated renders of the games' 3D models
     (`assets/sprites/front|back/*.gif`: 16 from Project Pokémon, 4 gaps filled from Showdown)
     stand on the cards as billboards that face the camera. The CPU's show their front and
     yours show their back, like the 3DS/Switch battle camera. The low-poly figurines remain
     as a fallback (`--figurines`, or used automatically when a sprite is missing).
   - **Move effects** (`game/ar/fx.py`, `move_fx.py`): particles in board space drawn with the
     same camera, using Showdown's battle-effect images plus Kenney's CC0 particles. Every one
     of the 40 moves has a *launch* (when the move is announced: projectiles, beams, falling
     rocks, sound rings, auras) and an *impact* (only when the damage event arrives, so misses
     show no hit). Status, heal, stat, faint and send-out moments have their own effects, and
     super-effective hits and crits shake the view. Cost: ~8–20 ms per frame during a move.
   - Where the art comes from and what may be published: `assets/ATTRIBUTION.md`.
     `.gitignore` keeps the Nintendo/Showdown art out of any repo.

## 9. Decisions log

| Date | Decision |
|---|---|
| 2026-09-26 | Opponent is a CPU. You place its cards on the far mat and the grid cam detects them |
| 2026-09-26 | Lead = top-left slot (slot 1) |
| 2026-09-26 | Party locked at battle start. Moving cards never switches; only the menu does |
| 2026-09-26 | Added m1L cards: Marshadow, Gimmighoul, Riolu, Seedot, Solrock (roster = 10) |
| 2026-09-26 | Play surface = back of a Monopoly board (no printed mats). Far half CPU, near half you, 3×2 zones each |
| 2026-09-26 | 3D: built-in low-poly figurines now; real .glb models can replace them one Pokémon at a time |
| 2026-09-26 | Pokémon shown as the games' animated model renders (Project Pokémon link from user; Showdown fills gaps). Effects: Showdown fx + Kenney CC0 |

## 10. Open questions

- **3D models:** figurines are built in code, so there's nothing to license. Any real models you
  drop into `assets/models/` are Nintendo IP: keep them out of a public repo (same approach as
  meme-cam's assets).
- Later: a second human player (sharing the Mac camera, or on a second device).
