#!/usr/bin/env bash
# Start the game. Run it from macOS Terminal (it needs Camera permission).
# Extra flags pass through, e.g.  ./play.sh --mouse   ./play.sh --figurines
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Run ./setup.sh first."; exit 1; fi
exec .venv/bin/python tools/hand_ui.py "$@"
