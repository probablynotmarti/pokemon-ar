"""Damage formula from the Switch/3DS games (Gen 5+), minus weather/abilities/items.

damage = ((2L/5 + 2) * Power * A/D / 50 + 2)
         * crit(1.5) * random(0.85..1.00) * STAB(1.5) * type
"""

from .data import MOVES, effectiveness, calc_stats

STAGE_MULT = {s: (2 + s) / 2 if s >= 0 else 2 / (2 - s) for s in range(-6, 7)}
CRIT_CHANCE = 1 / 24


def heavy_slam_power(user_kg, target_kg):
    ratio = user_kg / target_kg
    for threshold, power in ((5, 120), (4, 100), (3, 80), (2, 60)):
        if ratio >= threshold:
            return power
    return 40


def reversal_power(hp, max_hp):
    p = 48 * hp // max_hp
    for limit, power in ((1, 200), (4, 150), (9, 100), (16, 80), (32, 40)):
        if p <= limit:
            return power
    return 20


def move_power(move, attacker, defender, *, user_hp_frac=(1, 1),
               target_status=None):
    """attacker/defender are Species; HP/status passed in for live battles."""
    if move.effect == "weight_ratio_power":
        return heavy_slam_power(attacker.weight_kg, defender.weight_kg)
    if move.effect == "reversal_power":
        return reversal_power(*user_hp_frac)
    if move.effect == "hex_power" and target_status:
        return move.power * 2
    return move.power


def base_damage(level, power, atk, dfn):
    return (2 * level // 5 + 2) * power * atk // dfn // 50 + 2


def damage_range(attacker, defender, move_name, level=50,
                 atk_stage=0, def_stage=0, crit=False):
    """(min, max) damage for one hit; (0, 0) for status/fixed moves."""
    move = MOVES[move_name]
    if move.category == "status" or move.effect == "halve_target_hp":
        return (0, 0)
    a, d = calc_stats(attacker, level), calc_stats(defender, level)
    atk_key, def_key = (("atk", "def") if move.category == "physical"
                        else ("spa", "spd"))
    atk = a[atk_key] * STAGE_MULT[max(atk_stage, 0) if crit else atk_stage]
    dfn = d[def_key] * STAGE_MULT[min(def_stage, 0) if crit else def_stage]
    power = move_power(move, attacker, defender)
    stab = 1.5 if move.type in attacker.types else 1
    eff = effectiveness(move.type, defender.types)
    return tuple(final_damage(level, power, int(atk), int(dfn), crit=crit,
                              roll=roll, stab=stab, eff=eff)
                 for roll in (85, 100))


def final_damage(level, power, atk, dfn, *, crit, roll, stab, eff,
                 burned=False):
    """Games' modifier order: base → crit → roll → STAB → type → burn."""
    dmg = base_damage(level, power, atk, dfn)
    if crit:
        dmg = int(dmg * 1.5)
    dmg = int(int(int(dmg * roll / 100) * stab) * eff)
    return dmg // 2 if burned else dmg
