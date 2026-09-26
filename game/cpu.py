"""CPU opponent: greedy-ish, with a bit of randomness so it isn't robotic.

- At ≤25% HP it heals half the time if it has a healing item.
- Otherwise it scores each move (expected % of the foe's HP, KO bonus,
  simple heuristics for status moves) and plays the best one 75% of the
  time, a random legal move 25% of the time.
- It never switches voluntarily (v1); fainted Pokémon are replaced in
  slot order by the engine.
"""

from .battle import Fight, UseItem
from .data import MOVES

HEAL_ORDER = ("Super Potion", "Potion", "Oran Berry")
MISPLAY_CHANCE = 0.25


def choose_action(battle, side, rng):
    legal = battle.legal_actions(side)
    mon, foe = side.mon, battle.other(side).mon

    if mon.hp <= mon.max_hp // 4 and rng.random() < 0.5:
        for item in HEAL_ORDER:
            if UseItem(item, side.active) in legal:
                return UseItem(item, side.active)

    fights = [a for a in legal if isinstance(a, Fight)]
    if rng.random() < MISPLAY_CHANCE:
        return rng.choice(fights)
    return max(fights, key=lambda a: score(battle, side, a.move))


def score(battle, side, move_name):
    """Rough value of a move, on a 'fraction of the foe KO'd' scale."""
    if move_name == "Struggle":
        return 0
    mon, foe = side.mon, battle.other(side).mon
    move = MOVES[move_name]
    acc = (move.accuracy or 100) / 100
    if move.category != "status":
        dmg = battle.estimate_damage(mon, foe, move_name)
        frac = min(dmg / foe.hp, 1) if foe.hp else 0
        value = acc * (frac + (0.5 if frac >= 1 else 0))
        if move.effect == "self_ko":        # only worth it as a last gasp
            value *= 1.0 if mon.hp <= mon.max_hp // 4 else 0.2
        return value
    fx = move.effect
    if fx == "heal_1/2":
        return 0.6 if mon.hp < mon.max_hp / 2 else 0
    if fx == "sleep":
        return acc * 0.5 if foe.status is None else 0
    if fx == "leech_seed":
        ok = not foe.seeded and "Grass" not in foe.species.types
        return acc * 0.35 if ok else 0
    if fx == "eat_bag_berry" and not side.bag.get("Oran Berry", 0):
        return 0
    # Pure stat moves: worth something until the boost is already big.
    value = 0
    for stat, delta in move.boosts:
        if delta > 0 and mon.stages[stat] < 2:
            value += 0.15 * delta
    for stat, delta in move.target_boosts:
        if delta < 0 and foe.stages[stat] > -2:
            value += 0.15 * -delta
    return acc * value
