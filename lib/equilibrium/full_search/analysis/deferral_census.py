#!/usr/bin/env python3
"""One-off: of the branches DEFERRED in the cheap-half run, which cause dominates?

Replicates the production run conditions exactly (Julia solver, max_nv=8) on a random
sample of the ALREADY-PROCESSED prefix of order.npy and tallies, per support-profile, the
precise deferral sub-cause:

  nv_capped        nv > max_nv (never solved; pure cost cap)
  fulldim_noeq     no equations -> variety full-dimensional (inequalities only)
  posdim           msolve/AlgebraicSolving reports dimension >= 1
  zerodim_irrat    finite real points but none certified as a rational witness
  (feasible / infeasible are reported too, for context)

Usage: python -m lib.equilibrium.full_search.analysis.deferral_census [N] [PREFIX]
"""
import os, sys, time, random
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lib.equilibrium.full_search.full_mixing_sweep import DATA
from lib.equilibrium.full_search.routes.julia_spike.julia_solver import FlintJuliaSolver

PAYOFF = "burke_usaruschn_2035-2060"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
PREFIX = int(sys.argv[2]) if len(sys.argv) > 2 else 29_934_000   # processed-so-far cut

s = FlintJuliaSolver(PAYOFF); s.max_nv = 8
NO = s.NO
order = np.load(DATA / f"fullmix_{PAYOFF}_order.npy")
prefix = order[:PREFIX]
rng = random.Random(20260622)
sample = [int(prefix[rng.randrange(len(prefix))]) for _ in range(N)]

# capture _msolve_flint's status without changing behaviour
_orig = s._msolve_flint
_hold = {}
def _spy(nv, eqs, names):
    st, sols = _orig(nv, eqs, names)
    _hold["status"] = st
    return st, sols
s._msolve_flint = _spy

causes = Counter()           # per-profile outcome
labels_with_defer = 0
t0 = time.time()
for li, pk in enumerate(sample):
    a, b, c = pk // (NO * NO), (pk // NO) % NO, pk % NO
    tiers = (s.orders[a], s.orders[b], s.orders[c])
    deferred_here = False
    for prof in s.candidate_profiles(tiers):
        acc, wk = s._vars_for_profile(tiers, prof)
        nv = len(acc) + len(wk)
        if nv > s.max_nv:
            causes["nv_capped"] += 1; deferred_here = True; continue
        _hold.clear()
        res, _ = s.solve_profile(tiers, prof)
        if res == "feasible":
            causes["feasible"] += 1
        elif res == "infeasible":
            causes["infeasible"] += 1
        else:  # deferred
            deferred_here = True
            st = _hold.get("status")
            if st is None:
                causes["fulldim_noeq"] += 1
            elif st == "posdim":
                causes["posdim"] += 1
            elif st == "zerodim_incomplete":
                causes["zerodim_irrat"] += 1
            elif st == "zerodim":
                causes["zerodim_unexp"] += 1     # complete zerodim should now never defer
            else:
                causes[f"other_{st}"] += 1
    if deferred_here:
        labels_with_defer += 1
    if (li + 1) % 250 == 0:
        print(f"  {li+1}/{N} labels, {time.time()-t0:.0f}s", flush=True)

defer_total = sum(v for k, v in causes.items() if k not in ("feasible", "infeasible"))
print(f"\n=== deferral census: {N} sampled labels from order.npy[:{PREFIX:,}] ===")
print(f"labels with >=1 deferred branch: {labels_with_defer}/{N} ({100*labels_with_defer/N:.1f}%)")
print(f"total deferred branches: {defer_total}")
for k in ("nv_capped", "fulldim_noeq", "posdim", "zerodim_irrat"):
    v = causes.get(k, 0)
    print(f"  {k:14s} {v:8d}  ({100*v/defer_total:5.1f}% of deferred)" if defer_total else f"  {k}: 0")
for k in causes:
    if k.startswith("other_"):
        print(f"  {k:14s} {causes[k]:8d}")
print(f"context: feasible={causes.get('feasible',0)}  infeasible={causes.get('infeasible',0)}")
print(f"elapsed {time.time()-t0:.0f}s")
