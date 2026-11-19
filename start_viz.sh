#!/usr/bin/env bash
# Start the viz backend and the vite dev server with one command. Ctrl+C stops both.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Missing $PY - create the venv first" >&2; exit 1; }

cd "$ROOT/viz"
[ -d node_modules ] || npm install

"$PY" service_viz.py &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null || true' EXIT INT TERM

npm run dev
