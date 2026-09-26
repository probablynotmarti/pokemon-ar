"""Print best-move damage (max-roll % of target max HP) for every matchup.

    python3 tools/balance_table.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from game.data import POKEDEX, calc_stats  # noqa: E402
from game.damage import damage_range  # noqa: E402

mons = list(POKEDEX.values())
print("HP at L50: " + ", ".join(f"{m.name} {calc_stats(m)['hp']}"
                                for m in mons))
print("\nBest move, max-roll % of target HP (rows attack columns). "
      "Explosion excluded.\n")
print("".ljust(11) + "".join(m.name[:8].rjust(9) for m in mons) + "   avg")
for a in mons:
    row, pcts = a.name.ljust(11), []
    for d in mons:
        hp = calc_stats(d)["hp"]
        best = max((damage_range(a, d, mv)[1] for mv in a.moves
                    if mv != "Explosion"), default=0)
        pcts.append(100 * best // hp)
        row += f"{pcts[-1]}%".rjust(9)
    print(row + f"{sum(pcts) // len(pcts)}%".rjust(6))
