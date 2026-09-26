"""Static game data: types, Pokémon, moves, items.

Pure data + tiny helpers, no camera or rendering imports, so the battle
engine and tests can use it without MediaPipe/OpenCV installed.

Stats follow the Switch/3DS games: base stats -> level-50 stats with
31 IVs, 0 EVs, neutral nature. Speed is stored but NOT used for turn
order yet (design rule: order within a priority bracket is a coin flip).
"""

from dataclasses import dataclass, field

LEVEL = 50

# ── Type chart (Gen 6+). Only non-1x entries are listed. ──────────────
TYPE_CHART = {
    "Normal":   {"Rock": .5, "Ghost": 0, "Steel": .5},
    "Fire":     {"Fire": .5, "Water": .5, "Grass": 2, "Ice": 2, "Bug": 2,
                 "Rock": .5, "Dragon": .5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": .5, "Grass": .5, "Ground": 2,
                 "Rock": 2, "Dragon": .5},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Ground": 0,
                 "Flying": 2, "Dragon": .5},
    "Grass":    {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5,
                 "Ground": 2, "Flying": .5, "Bug": .5, "Rock": 2,
                 "Dragon": .5, "Steel": .5},
    "Ice":      {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5,
                 "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": .5, "Flying": .5,
                 "Psychic": .5, "Bug": .5, "Rock": 2, "Ghost": 0,
                 "Dark": 2, "Steel": 2, "Fairy": .5},
    "Poison":   {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5,
                 "Ghost": .5, "Steel": 0, "Fairy": 2},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": .5, "Poison": 2,
                 "Flying": 0, "Bug": .5, "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": .5, "Grass": 2, "Fighting": 2, "Bug": 2,
                 "Rock": .5, "Steel": .5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": .5, "Dark": 0,
                 "Steel": .5},
    "Bug":      {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5,
                 "Flying": .5, "Psychic": 2, "Ghost": .5, "Dark": 2,
                 "Steel": .5, "Fairy": .5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5,
                 "Flying": 2, "Bug": 2, "Steel": .5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon":   {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark":     {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5,
                 "Fairy": .5},
    "Steel":    {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2,
                 "Rock": 2, "Steel": .5, "Fairy": 2},
    "Fairy":    {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2,
                 "Dark": 2, "Steel": .5},
}
TYPES = tuple(TYPE_CHART)


def effectiveness(move_type, defender_types):
    mult = 1.0
    for t in defender_types:
        mult *= TYPE_CHART[move_type].get(t, 1)
    return mult


# ── Moves ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Move:
    name: str
    type: str
    category: str          # "physical" | "special" | "status"
    power: int | None      # None for status / special-cased damage
    accuracy: int | None   # None = never misses
    pp: int
    priority: int = 0
    effect: str = ""       # engine hook id (see game/battle.py)
    effect_chance: int = 100   # % for secondary effects + target_boosts
    boosts: tuple = ()         # ((stat, delta), ...) on the USER, always
    target_boosts: tuple = ()  # ((stat, delta), ...) on the target
    recoil: float = 0          # fraction of damage dealt the user takes
    desc: str = ""


MOVES = {m.name: m for m in [
    # Registeel
    Move("Heavy Slam", "Steel", "physical", None, 100, 10,
         effect="weight_ratio_power",
         desc="Power 40–120: heavier user vs lighter target hits harder."),
    Move("Flash Cannon", "Steel", "special", 80, 100, 10,
         target_boosts=(("spd", -1),), effect_chance=10,
         desc="10% chance to lower target's Sp. Def."),
    Move("Hammer Arm", "Fighting", "physical", 100, 90, 10,
         boosts=(("spe", -1),),
         desc="Lowers user's Speed (cosmetic until speed matters)."),
    Move("Iron Defense", "Steel", "status", None, None, 15,
         boosts=(("def", 2),), desc="Sharply raises user's Defense."),
    # Sudowoodo
    Move("Rock Slide", "Rock", "physical", 75, 90, 10,
         effect="flinch", effect_chance=30,
         desc="30% flinch if the user moved first."),
    Move("Double-Edge", "Normal", "physical", 120, 100, 15,
         recoil=1 / 3, desc="User takes 1/3 of damage dealt."),
    Move("Brick Break", "Fighting", "physical", 75, 100, 15),
    Move("Sucker Punch", "Dark", "physical", 70, 100, 5, priority=1,
         effect="fails_unless_target_attacks",
         desc="Always goes first; fails if the target isn't attacking."),
    # Ducklett
    Move("Bubble Beam", "Water", "special", 65, 100, 20,
         target_boosts=(("spe", -1),), effect_chance=10),
    Move("Air Slash", "Flying", "special", 75, 95, 15,
         effect="flinch", effect_chance=30),
    Move("Feather Dance", "Flying", "status", None, 100, 15,
         target_boosts=(("atk", -2),), desc="Harshly lowers target's Attack."),
    Move("Roost", "Flying", "status", None, None, 5,
         effect="heal_1/2", desc="Restores half of max HP."),
    # Skwovet
    Move("Bite", "Dark", "physical", 60, 100, 25,
         effect="flinch", effect_chance=30),
    Move("Body Slam", "Normal", "physical", 85, 100, 15,
         effect="paralyze", effect_chance=30),
    Move("Super Fang", "Normal", "physical", None, 90, 10,
         effect="halve_target_hp",
         desc="Cuts target's current HP in half. Ghosts are immune."),
    Move("Stuff Cheeks", "Normal", "status", None, None, 10,
         effect="eat_bag_berry", boosts=(("def", 2),),
         desc="Eats an Oran Berry from the Bag: heals it + Defense +2. "
              "Fails if the Bag has no berries."),
    # Politoed
    Move("Hydro Pump", "Water", "special", 110, 80, 5),
    Move("Ice Beam", "Ice", "special", 90, 100, 10,
         effect="freeze", effect_chance=10),
    Move("Hyper Voice", "Normal", "special", 90, 100, 10),
    Move("Hypnosis", "Psychic", "status", None, 60, 20,
         effect="sleep", desc="Puts the target to sleep for 1–3 turns."),
    # Marshadow
    Move("Spectral Thief", "Ghost", "physical", 90, 100, 10,
         effect="steal_boosts",
         desc="Steals the target's stat boosts before hitting."),
    Move("Close Combat", "Fighting", "physical", 120, 100, 5,
         boosts=(("def", -1), ("spd", -1)),
         desc="Lowers user's Defense and Sp. Def."),
    Move("Shadow Sneak", "Ghost", "physical", 40, 100, 30, priority=1,
         desc="Always goes first."),
    Move("Bulk Up", "Fighting", "status", None, None, 20,
         boosts=(("atk", 1), ("def", 1)),
         desc="Raises user's Attack and Defense."),
    # Gimmighoul
    Move("Shadow Ball", "Ghost", "special", 80, 100, 15,
         target_boosts=(("spd", -1),), effect_chance=20),
    Move("Hex", "Ghost", "special", 65, 100, 10, effect="hex_power",
         desc="Double power (130) if the target has a status condition."),
    Move("Power Gem", "Rock", "special", 80, 100, 20,
         desc="Coverage so Normal types can't wall Gimmighoul."),
    Move("Nasty Plot", "Dark", "status", None, None, 20,
         boosts=(("spa", 2),), desc="Sharply raises user's Sp. Atk."),
    # Riolu
    Move("Force Palm", "Fighting", "physical", 60, 100, 10,
         effect="paralyze", effect_chance=30),
    Move("Quick Attack", "Normal", "physical", 40, 100, 30, priority=1,
         desc="Always goes first."),
    Move("Reversal", "Fighting", "physical", None, 100, 15,
         effect="reversal_power",
         desc="Power 20–200: the lower the user's HP, the stronger."),
    Move("Crunch", "Dark", "physical", 80, 100, 15,
         target_boosts=(("def", -1),), effect_chance=20,
         desc="20% chance to lower target's Defense. Hits Ghosts."),
    # Seedot
    Move("Seed Bomb", "Grass", "physical", 80, 100, 15),
    Move("Synthesis", "Grass", "status", None, None, 5,
         effect="heal_1/2", desc="Restores half of max HP."),
    Move("Leech Seed", "Grass", "status", None, 90, 10, effect="leech_seed",
         desc="Drains 1/8 of the target's max HP each turn to heal your "
              "active Pokémon. Grass types are immune; ends on switch."),
    Move("Explosion", "Normal", "physical", 250, 100, 5, effect="self_ko",
         desc="Huge damage, but the user faints (even if it misses)."),
    # Solrock
    Move("Zen Headbutt", "Psychic", "physical", 80, 90, 15,
         effect="flinch", effect_chance=20),
    Move("Flare Blitz", "Fire", "physical", 120, 100, 15,
         recoil=1 / 3, effect="burn", effect_chance=10,
         desc="1/3 recoil; 10% chance to burn."),
    Move("Cosmic Power", "Psychic", "status", None, None, 20,
         boosts=(("def", 1), ("spd", 1)),
         desc="Raises user's Defense and Sp. Def."),
]}


# ── Pokémon ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Species:
    key: str
    name: str
    card_name: str           # name as printed on the owned card
    dex: int
    types: tuple
    base: dict               # hp, atk, def, spa, spd, spe
    weight_kg: float
    moves: tuple             # exactly 4, in menu order (TL, TR, BL, BR)
    card_ids: tuple          # set code + number printed bottom-left
    card_hp: int             # HP printed on the TCG card (reference only)


def _base(hp, atk, df, spa, spd, spe):
    return {"hp": hp, "atk": atk, "def": df, "spa": spa, "spd": spd,
            "spe": spe}


POKEDEX = {s.key: s for s in [
    # ── s10D (Traditional Chinese) ──
    Species("registeel", "Registeel", "雷吉斯奇魯", 379, ("Steel",),
            _base(80, 75, 150, 75, 150, 50), 205.0,
            ("Heavy Slam", "Flash Cannon", "Hammer Arm", "Iron Defense"),
            ("s10D-047",), 130),
    Species("sudowoodo", "Sudowoodo", "樹才怪", 185, ("Rock",),
            _base(70, 100, 115, 30, 65, 30), 38.0,
            ("Rock Slide", "Double-Edge", "Brick Break", "Sucker Punch"),
            ("s10D-034",), 110),
    Species("ducklett", "Ducklett", "鴨寶寶", 580, ("Water", "Flying"),
            _base(62, 44, 50, 44, 50, 55), 5.5,
            ("Bubble Beam", "Air Slash", "Feather Dance", "Roost"),
            ("s10D-017",), 60),
    Species("skwovet", "Skwovet", "貪心栗鼠", 819, ("Normal",),
            _base(70, 55, 55, 35, 35, 25), 2.5,
            ("Bite", "Body Slam", "Super Fang", "Stuff Cheeks"),
            ("s10D-057",), 60),
    Species("politoed", "Politoed", "蚊香蛙皇", 186, ("Water",),
            _base(90, 75, 75, 90, 100, 70), 33.9,
            ("Hydro Pump", "Ice Beam", "Hyper Voice", "Hypnosis"),
            ("s10D-013",), 140),
    # ── m1L (Japanese) ──
    Species("marshadow", "Marshadow", "マーシャドー", 802,
            ("Fighting", "Ghost"), _base(90, 125, 80, 90, 90, 125), 22.2,
            ("Spectral Thief", "Close Combat", "Shadow Sneak", "Bulk Up"),
            ("m1L-069",), 90),
    Species("gimmighoul", "Gimmighoul", "コレクレー", 999, ("Ghost",),
            _base(45, 30, 70, 75, 70, 10), 5.0,
            ("Shadow Ball", "Hex", "Power Gem", "Nasty Plot"),
            ("m1L-019",), 70),
    Species("riolu", "Riolu", "リオル", 447, ("Fighting",),
            _base(40, 70, 40, 35, 40, 60), 20.2,
            ("Force Palm", "Quick Attack", "Reversal", "Crunch"),
            ("m1L-028",), 80),
    Species("seedot", "Seedot", "タネボー", 273, ("Grass",),
            _base(40, 40, 50, 30, 30, 30), 4.0,
            ("Seed Bomb", "Synthesis", "Leech Seed", "Explosion"),
            ("m1L-007",), 60),
    Species("solrock", "Solrock", "ソルロック", 338, ("Rock", "Psychic"),
            _base(90, 95, 85, 55, 65, 70), 154.0,
            ("Rock Slide", "Zen Headbutt", "Flare Blitz", "Cosmic Power"),
            ("m1L-027",), 110),
]}


def calc_stats(species, level=LEVEL, iv=31):
    """Level-N stats, 0 EVs, neutral nature (games' formula)."""
    out = {}
    for stat, b in species.base.items():
        core = (2 * b + iv) * level // 100
        out[stat] = core + level + 10 if stat == "hp" else core + 5
    return out


# ── Items ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Item:
    name: str
    pocket: str       # "Medicine" | "Berries"
    effect: str
    amount: int = 0
    target: str = "alive"   # "alive" | "fainted"
    desc: str = ""


ITEMS = {i.name: i for i in [
    Item("Potion", "Medicine", "heal", 20, desc="Restores 20 HP."),
    Item("Super Potion", "Medicine", "heal", 60, desc="Restores 60 HP."),
    Item("Full Heal", "Medicine", "cure_status",
         desc="Cures sleep, paralysis, freeze and burn."),
    Item("Revive", "Medicine", "revive_half", target="fainted",
         desc="Revives a fainted Pokémon with half its max HP."),
    Item("Oran Berry", "Berries", "heal", 10, desc="Restores 10 HP."),
]}

# What each player's Bag starts a battle with (tunable).
STARTING_BAG = {
    "Potion": 3,
    "Super Potion": 1,
    "Oran Berry": 3,
    "Full Heal": 1,
    "Revive": 1,
}

# Physical mat: 3 columns x 2 rows per player, slot 1 = lead.
GRID_COLS, GRID_ROWS = 3, 2
PARTY_SIZE = GRID_COLS * GRID_ROWS
