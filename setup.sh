#!/usr/bin/env bash
# One-time setup: Python 3.12 venv in .venv, pinned deps, game art.
set -euo pipefail
cd "$(dirname "$0")"
PY=""
for c in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Needs Python 3.12 (MediaPipe has no wheels for 3.13+)."
  echo "Install it with:  brew install python@3.12"
  exit 1
fi
echo "→ creating .venv with $("$PY" --version)"
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
echo "→ installing dependencies (MediaPipe is ~100 MB, give it a minute)"
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "→ downloading sprites + battle effects"
.venv/bin/python tools/fetch_assets.py
echo "→ checking"
.venv/bin/python -m unittest discover -s tests -t . 2>&1 | grep -E "^(Ran |OK|FAILED)" || true
cat <<MSG

Done. Next:
  1. Register your cards once (iPhone plugged in, card in YOUR slot 1):
       .venv/bin/python tools/grid_cam.py --register seedot     # etc.
  2. Play:   ./play.sh          (or ./play.sh --mouse, ./play.sh --no-grid)
MSG
