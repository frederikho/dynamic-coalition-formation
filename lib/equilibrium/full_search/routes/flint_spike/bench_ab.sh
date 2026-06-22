#!/usr/bin/env bash
# Quiesced A/B bench: pause the big `find` run, bench one solver on the frozen corpus
# pinned to a core, then ALWAYS resume the run (trap on EXIT). Usage:
#   bash bench_ab.sh <solver:baseline|flint> <tag>
set -uo pipefail
cd /mnt/data1/Code/farsighted-coalitions
SOLVER="${1:-flint}"
TAG="${2:-$SOLVER}"
PAT="python -u scripts/full_mixing_sweep.py find"

resume() { pkill -CONT -f "$PAT" 2>/dev/null || true; echo "[bench_ab] run resumed"; }
trap resume EXIT

echo "[bench_ab] pausing big run..."
pkill -STOP -f "$PAT" 2>/dev/null || true
sleep 2
echo "[bench_ab] load while quiesced: $(cut -d' ' -f1 /proc/loadavg)"

.venv/bin/python -m lib.equilibrium.full_search.analysis.harness bench \
  --payoff burke_usaruschn_2035-2060 --tag "$TAG" --solver "$SOLVER" --repeats 3 --pin 15
echo "[bench_ab] bench exit: $?"
