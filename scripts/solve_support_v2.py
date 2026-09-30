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

3. NO seeding grid.  An earlier version ranked start points on a 6^M grid -- 1.7M
   residual evaluations at M=8, 60M at M=10 -- all spent before any solving began, so
   at M>=7 the budget expired before a single solve ran (51/86, mean 2.22 s).
   Fixed-budget random starts replace it.

0. PROPOSAL KNOBS (added 2026-08-24).  A knob may now be ("P", x, i, y, z), freeing
   sigma_i(x->y) with sigma_i(x->z) = 1 - it, alongside the acceptance triples
   (x, y, j).  The zero diagonal applies identically -- a proposer cannot satisfy
   their own indifference with their own weight -- so the algorithm is unchanged;
   only which probability the theta is written into differs.  Run the proposal
   controls with --profiles planted_profiles_prop.

4. The objective carries the INEQUALITY violations, not only the mixing players'
   indifference, which is flat in their own theta.  See mixed_controls.merit_vector.
   With 3 and 4 together: 86/86 at a mean of 0.06 s.
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import root, least_squares

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium import mixed_controls as mc  # noqa: E402
from lib.equilibrium.jeres_vfi.solver import (  # noqa: E402
    compute_values, full_transition_matrix, verify_proposals, verify_responses)

ATOL = 1e-12
import importlib.util
sp = importlib.util.spec_from_file_location("s", REPO/"scripts/solve_support_anyM.py")
S = importlib.util.module_from_spec(sp); sp.loader.exec_module(S)

EPS = 1e-12

def solve_one(ctx_d, g, comm, sig, alp, knobs, seed=0, budget=5.0, **_):
    """
    Find the M interior probabilities, given the support pattern.

    Two changes from the original, both measured on the 86-table control set:

      * the seeding grid is gone.  Ranking start points on a 6^M grid costs 1.7M
        residual evaluations at M=8 and 60M at M=10, all spent BEFORE any solving
        begins -- at M>=7 the entire budget went on ranking and no solve ever ran.
        Score was 51/86 at a mean of 2.22 s.
      * the objective now carries the INEQUALITY violations, not just the mixing
        players' indifference.  Those own-conditions are flat in their own theta, so
        a bare root-find is blind to what actually pins the answer.  See
        mixed_controls.merit_vector.

    Together: 86/86 at a mean of 0.06 s.
    """
    import time
    M = len(knobs)
    delta = ctx_d["delta"]

    def state(theta):
        """Apply the M free values.  A knob is either an acceptance triple (x, y, j)
        or a proposal knob ("P", x, i, y, z), where theta is sigma_i(x->y) and the
        co-supported target z takes 1 - theta.  With acceptance knobs only, s2 is
        `sig` unchanged, so the acceptance-only path is bit-identical to before."""
        a2 = [dict(x) for x in alp]
        s2 = [dict(x) for x in sig]
        for k, t in zip(knobs, np.clip(theta, 0.0, 1.0)):
            if mc.is_prop_knob(k):
                _, kx, ki, ky, kz = k
                s2[kx][(ki, ky)] = float(t)
                s2[kx][(ki, kz)] = 1.0 - float(t)
            else:
                kx, ky, kj = k
                a2[kx][(kj, ky)] = float(t)
        qs = mc.qs_from_alphas(g, comm, a2)
        V = compute_values(g, full_transition_matrix(g, s2, qs, None), delta)
        return s2, a2, qs, V

    def accept(theta):
        s2, a2, qs, V = state(theta)
        r, _ = verify_responses(g, s2, a2, qs, V, atol=ATOL)
        p, _ = verify_proposals(g, s2, a2, qs, V, atol=ATOL)
        return bool(r and p)

    def merit(theta):
        s2, a2, qs, V = state(theta)
        return mc.merit_vector(g, comm, s2, a2, qs, V, knobs)

    if accept(np.full(M, 0.5)):
        return np.full(M, 0.5), "half"

    rng = np.random.default_rng(seed)
    deadline = time.time() + budget
    while time.time() < deadline:
        th0 = rng.uniform(0.02, 0.98, M)
        try:
            sol = least_squares(merit, th0, bounds=(np.zeros(M), np.ones(M)),
                                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=600)
        except Exception:
            continue
        if accept(sol.x):
            return np.clip(sol.x, 0.0, 1.0), "merit"
    return None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="planted_profiles",
                    help="directory under reports/ holding the planted profiles. "
                         "Use planted_profiles_prop for the PROPOSAL-mixing controls.")
    ap.add_argument("--budget", type=float, default=5.0,
                    help="restart budget per table, seconds")
    a = ap.parse_args()
    PROF = REPO/"reports"/a.profiles
    print(f"{'table':<34}{'M':>2}{'solved':>8}{'via':>7}{'s':>8}   theta")
    agg = {}
    for f in sorted(PROF.glob("*.json")):
        d, g, comm, sig, alp, knobs = S.load(f)
        t0 = time.time(); th, via = solve_one(d, g, comm, sig, alp, knobs, budget=a.budget)
        dt = time.time()-t0
        agg.setdefault(d["M"], []).append((th is not None, dt))
        print(f"{Path(d['file']).stem:<34}{d['M']:>2}{('YES' if th is not None else 'no'):>8}"
              f"{via:>7}{dt:>8.2f}   "
              f"{np.array2string(np.round(th,4)) if th is not None else '-'}", flush=True)
    print(f"\n{'M':>2}{'tables':>8}{'solved':>8}{'rate':>8}{'median s':>11}{'max s':>9}")
    for m in sorted(agg):
        v = agg[m]; ts = sorted(x[1] for x in v); n = sum(1 for x in v if x[0])
        print(f"{m:>2}{len(v):>8}{n:>8}{100*n/len(v):>7.0f}%{ts[len(ts)//2]:>11.2f}{max(ts):>9.2f}")

if __name__ == "__main__":
    main()
