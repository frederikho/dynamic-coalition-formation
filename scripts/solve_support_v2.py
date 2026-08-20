#!/usr/bin/env python3
"""
Solve for the mixing probabilities given the support pattern -- robust version.

v1 used scipy `root(method="hybr")` from random starts and degraded with M
(10/10, 8/10, 8/10, 3/8).  The residual at the planted theta is machine-zero on
every table including the failures, so the equations are right and the root-finder
was the weak link.

Three changes:

1. CLOSED box.  v1 demanded 1e-9 < theta < 1-1e-9 and discarded any solution at a
   bound.  But a tie-resting PURE equilibrium is a legitimate endpoint of the
   feasible set -- measured on the M=1 tables, every feasible set is a single
   interval and several include 0 or 1.  Requiring interiority throws those away.

2. Several methods.  The Jacobian has a ZERO DIAGONAL (a mixer's own indifference
   is insensitive to their own probability), which is badly conditioned for a
   Newton step from a generic start.  least_squares/trf minimises ||r||^2 subject
   to bounds and copes with that far better than hybr.

3. Grid-seeded starts.  A coarse scan locates a low-residual basin, then a local
   solve polishes it, instead of hoping a random start lands in one.
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import root, least_squares
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
sp = importlib.util.spec_from_file_location("s", REPO/"scripts/solve_support_anyM.py")
S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)

EPS = 1e-12

def solve_one(d, g, comm, sig, alp, knobs, seed=0, grid_pts=6, rand_starts=24):
    M = len(knobs); delta = d["delta"]
    def resid(th):
        t = np.clip(th, 0.0, 1.0)
        _, _, V = S.state_at(g, comm, sig, alp, knobs, t, delta)
        return np.array([V[y, j] - V[x, j] for (x, y, j) in knobs])
    def accept(th):
        t = np.clip(th, 0.0, 1.0)
        a2, qs, V = S.state_at(g, comm, sig, alp, knobs, t, delta)
        return (t, True) if S.verify(g, sig, a2, qs, V) else (t, False)

    if M == 1:                       # solution is an interval; scan the CLOSED box
        for t in np.linspace(0.0, 1.0, 101):
            th, ok = accept([t])
            if ok:
                return th, "scan"
        return None, "scan"

    rng = np.random.default_rng(seed)
    starts = [np.full(M, 0.5)]
    coarse = np.linspace(0.1, 0.9, grid_pts)
    scored = []
    for pt in itertools.product(coarse, repeat=M):
        scored.append((np.max(np.abs(resid(np.array(pt)))), np.array(pt)))
    scored.sort(key=lambda z: z[0])
    starts += [p for _, p in scored[:8]]                    # grid-seeded basins
    starts += [rng.uniform(0, 1, M) for _ in range(rand_starts)]

    for th0 in starts:
        for tag in ("trf", "hybr", "lm"):
            try:
                if tag == "trf":
                    sol = least_squares(resid, np.clip(th0, 1e-9, 1-1e-9),
                                        bounds=(np.zeros(M), np.ones(M)),
                                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
                    x = sol.x
                else:
                    sol = root(resid, th0, method=tag)
                    x = sol.x
            except Exception:
                continue
            if not np.all(np.isfinite(x)):
                continue
            th, ok = accept(x)
            if ok:
                return th, tag
    return None, "none"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profiles", default="planted_profiles")
    a = ap.parse_args()
    PROF = REPO/"reports"/a.profiles
    print(f"{'table':<34}{'M':>2}{'solved':>8}{'via':>7}{'s':>8}   theta")
    agg = {}
    for f in sorted(PROF.glob("*.json")):
        d, g, comm, sig, alp, knobs = S.load(f)
        t0 = time.time(); th, via = solve_one(d, g, comm, sig, alp, knobs)
        dt = time.time()-t0
        agg.setdefault(d["M"], []).append((th is not None, dt))
        print(f"{d['file'][14:34]:<34}{d['M']:>2}{('YES' if th is not None else 'no'):>8}"
              f"{via:>7}{dt:>8.2f}   "
              f"{np.array2string(np.round(th,4)) if th is not None else '-'}", flush=True)
    print(f"\n{'M':>2}{'tables':>8}{'solved':>8}{'rate':>8}{'median s':>11}{'max s':>9}")
    for m in sorted(agg):
        v = agg[m]; ts = sorted(x[1] for x in v); n = sum(1 for x in v if x[0])
        print(f"{m:>2}{len(v):>8}{n:>8}{100*n/len(v):>7.0f}%{ts[len(ts)//2]:>11.2f}{max(ts):>9.2f}")

if __name__ == "__main__":
    main()
