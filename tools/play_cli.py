"""Play a battle in the terminal. Stand-in for the camera + gesture UI.

    python3 tools/play_cli.py
    python3 tools/play_cli.py --you 1=politoed,2=registeel --cpu 1=sudowoodo,3=skwovet

Menus mirror the gesture UI: FIGHT / BAG / POKÉMON / RUN, 0 = back (✊).
"""
import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from game.battle import (Battle, Fight, Run, Switch, UseItem,  # noqa: E402
                         STATUS_TAGS, build_side, parse_mat)
from game.data import MOVES, POKEDEX  # noqa: E402


def hp_bar(mon, width=20):
    filled = round(width * mon.hp / mon.max_hp)
    tag = STATUS_TAGS.get(mon.status, "")
    return f"{mon.name:<10} [{'█' * filled}{'·' * (width - filled)}] " \
           f"{mon.hp}/{mon.max_hp} {tag}"


def pick(prompt, options, allow_back=True):
    """options: list of (label, value, enabled)."""
    for i, (label, _, ok) in enumerate(options, 1):
        print(f"  {i}. {label}" + ("" if ok else "  (—)"))
    if allow_back:
        print("  0. back")
    while True:
        # accept "1", "1.", " 1) " — anything whose digits form a choice
        raw = "".join(ch for ch in input(prompt) if ch.isdigit())
        if allow_back and raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            label, value, ok = options[int(raw) - 1]
            if ok:
                return value
        print(f"  can't pick that; type a number 1–{len(options)}"
              + (" (0 = back)" if allow_back else "") + " and press Enter")


def party_options(side, legal, make):
    return [(f"slot {m.slot}: {hp_bar(m)}", make(i), make(i) in legal)
            for i, m in enumerate(side.party)]


def choose(b):
    side, legal = b.player, b.legal_actions(b.player)
    while True:
        top = pick("> ", [("FIGHT", "fight", True), ("BAG", "bag", True),
                          ("POKÉMON", "party", True), ("RUN", "run", True)],
                   allow_back=False)
        if top == "fight":
            opts = []
            for m in side.mon.species.moves:
                mv = MOVES[m]
                opts.append((f"{m:<14} {mv.type:<9} PP {side.mon.pp[m]}/{mv.pp}",
                             Fight(m), Fight(m) in legal))
            if Fight("Struggle") in legal:
                opts = [("Struggle", Fight("Struggle"), True)]
            act = pick("move> ", opts)
        elif top == "bag":
            item = pick("item> ", [(f"{k} x{v}", k, v > 0 and any(
                isinstance(a, UseItem) and a.item == k for a in legal))
                for k, v in side.bag.items()])
            if item is None:
                continue
            act = pick("on> ", party_options(side, legal,
                                             lambda i: UseItem(item, i)))
        elif top == "party":
            act = pick("switch to> ", party_options(side, legal, Switch))
        else:
            act = Run() if input("Forfeit? (y/n) ").lower() == "y" else None
        if act is not None:
            return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--you", default="1=politoed,2=registeel,3=ducklett")
    ap.add_argument("--cpu", default="1=sudowoodo,2=marshadow,4=registeel")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    b = Battle(build_side("player", parse_mat(args.you)),
               build_side("cpu", parse_mat(args.cpu)), random.Random(args.seed))
    print("Pokémon available:", ", ".join(POKEDEX))
    print(f"The CPU sent out {b.cpu.mon.name}! Go! {b.player.mon.name}!")
    while not b.winner:
        if b.needs_switch(b.player):
            print("\nChoose your next Pokémon:")
            act = pick("send out> ", party_options(
                b.player, b.legal_actions(b.player), Switch), allow_back=False)
            events = b.replace_fainted(b.player, act.target)
        else:
            print(f"\n── Turn {b.turn + 1} ──\n  CPU: {hp_bar(b.cpu.mon)}\n"
                  f"  You: {hp_bar(b.player.mon)}")
            events = b.play_turn(choose(b))
        for e in events:
            if e.kind == "damage":
                print(f"   (-{e.data['amount']} HP)")
            elif e.text:
                print("  " + e.text)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBattle abandoned.")
