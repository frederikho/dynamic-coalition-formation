#!/usr/bin/env bash
# Production run: cheapest-first FIND over the whole current cheap-50% (order.npy), using the
# in-process Julia Gröbner solver. Checkpointed (survives power-off: resumes from
# data/fullmix_<payoff>_find_progress.txt), self-logs every 30s with a trailing-window ETA to
# logs/fullmix_<payoff>_find.log. Re-running this script resumes; nothing is redone.
#   WORKERS env overrides the worker count (default 12; RAM-bound — each worker is a Julia runtime).
set -uo pipefail
cd /mnt/data1/Code/farsighted-coalitions
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export PYTHONPATH=/mnt/data1/Code/farsighted-coalitions
PAYOFF="${PAYOFF:-burke_usaruschn_2035-2060}"
WORKERS="${WORKERS:-12}"
echo "[run_cheap_half] $(date) payoff=$PAYOFF workers=$WORKERS"
exec .venv/bin/python -u -m lib.equilibrium.full_search.full_mixing_sweep find \
  --payoff "$PAYOFF" --workers "$WORKERS" --fraction 0.50 --max-nv 8 --solver julia \
  2>&1 | tee -a logs/cheaphalf_run_console.log
