"""Download the game art that can't live in the repo (see assets/ATTRIBUTION.md).

    python tools/fetch_assets.py

- Pokémon sprites: animated 3D-model renders from Project Pokémon, with the
  gaps (Gen 9, some backs) filled from Pokémon Showdown.
- Battle effects: Pokémon Showdown's /fx/ images.
Already-downloaded files are skipped. Kenney's CC0 particles ship with the repo.
Card reference photos are yours to make: tools/grid_cam.py --register <key>.
"""
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.data import POKEDEX  # noqa: E402

SPRITES = {
    "front": ["https://projectpokemon.org/images/normal-sprite/{}.gif",
              "https://play.pokemonshowdown.com/sprites/ani/{}.gif"],
    "back": ["https://projectpokemon.org/images/sprites-models/normal-back/{}.gif",
             "https://play.pokemonshowdown.com/sprites/ani-back/{}.gif"],
}
FX = ("alpha angry blackwisp bluefireball bone bottombite caltrop electroball "
      "energyball faintleft faintright feather fireball fist fist1 flareball foot "
      "gear greenmetal1 greenmetal2 heart hitmarker iceball icicle-pink icicle "
      "impact leaf1 leaf2 leftchop leftclaw leftslash lightning mistball moon "
      "mudwisp omega petal poisoncaltrop poisonwisp pokeball purplewisp rainbow "
      "rightchop rightclaw rightslash rock1 rock2 rock3 rocks shadowball shell "
      "shine sound stare sword topbite ultra waterwisp web wisp").split()
FX_URL = "https://play.pokemonshowdown.com/fx/{}.png"
HEADERS = {"User-Agent": "pokemon-ar asset fetcher (personal use)"}


def fetch(url, out):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    out.write_bytes(data)
    return len(data)


def main():
    got = skipped = 0
    failed = []
    for view, urls in SPRITES.items():
        folder = ROOT / "assets" / "sprites" / view
        folder.mkdir(parents=True, exist_ok=True)
        for key in POKEDEX:
            out = folder / f"{key}.gif"
            if out.exists():
                skipped += 1
                continue
            for url in urls:
                try:
                    fetch(url.format(key), out)
                    got += 1
                    break
                except Exception:
                    continue
            else:
                failed.append(f"sprite {view}/{key}")
    folder = ROOT / "assets" / "fx" / "showdown"
    folder.mkdir(parents=True, exist_ok=True)
    for name in FX:
        out = folder / f"{name}.png"
        if out.exists():
            skipped += 1
            continue
        try:
            fetch(FX_URL.format(name), out)
            got += 1
        except Exception:
            failed.append(f"effect {name}")
    print(f"assets: {got} downloaded, {skipped} already there"
          + (f", {len(failed)} failed: {', '.join(failed)}" if failed else ""))
    if failed:
        print("The game still runs: missing sprites fall back to built-in "
              "3D figurines, missing effects are skipped.")


if __name__ == "__main__":
    main()
