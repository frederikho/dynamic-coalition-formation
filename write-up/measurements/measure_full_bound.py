#!/usr/bin/env python3
"""Full proposal-mixing search: bracket on total time (Section 7.10).

We have NO working full per-label solver to measure, and the decisive quantity --
the number of REALISABLE proposal supports per triple -- is unmeasured. What we can
bound:

  lower bound  : the acceptance-class search (r = 1, proposals pure) ~ Section 7.9.
  upper-ish    : full cost ~ sum_R r(R) * c(m(R)), using the measured tie-restricted
                 r(R) = prod_cells (2^t - 1)   (t = #feasible at the proposer's best
                 tier; this already restricts to top-tier-tied targets, so it is the
                 realisable-support count UP TO the EV-tie refinement), and c(m) the
                 measured acceptance per-label cost as a proxy for one support-fixed
                 solve (a LOWER bound on per-support cost, which has extra proposal
                 variables).

The true full time lies between, gated by how far EV-realisability collapses r(R).
This script computes the bracket and shows it is r-tail-dominated.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "write-up" / "measurements"))
from bench_build_interp import cls, fg, n, S, orders, NO, m_of
TOTAL = NO ** 3

# measured acceptance per-label cost model c(m) (build+solve, Section 7.9; high m censored)
CM = {0:0.0001,1:0.0009,2:0.0017,3:0.0111,4:0.0436,5:0.6312,6:0.6482,
      7:9.926,8:18.61,9:25.06,10:58.25,11:60.0,12:60.0,13:60.0,14:60.0}
def c(m): return CM.get(m, 60.0)

def r_of(tiers):
    r = 1
    for ii in range(n):
        ti = tiers[ii]
        for xi in range(S):
            feas = fg.feasible[(ii, xi)]
            best = min(int(ti[y]) for y in feas)
            t = sum(1 for y in feas if int(ti[y]) == best)
            r *= (2 ** t - 1)
    return r

def main(N=100000, seed=21):
    rng = np.random.RandomState(seed)
    rc = np.empty(N); rr = np.empty(N); cc = np.empty(N)
    for i in range(N):
        tiers = tuple(orders[rng.randint(NO)] for _ in range(n))
        m = m_of(tiers); ri = r_of(tiers)
        rr[i] = ri; cc[i] = c(m); rc[i] = ri * c(m)
    # acceptance-class lower bound (r=1)
    acc_core = TOTAL * cc.mean()
    full_core = TOTAL * rc.mean()
    print(f"sample N={N}")
    print(f"  acceptance-class (r=1):   E[c]={cc.mean():.4g}s   -> {acc_core/3600:,.0f} core-h "
          f"= {acc_core/3600/14:,.0f} h/14w = {acc_core/3600/14/24:,.1f} d/14w")
    print(f"  full (r*c) UPPER model:   E[r*c]={rc.mean():.4g}s  -> {full_core/3600:,.3g} core-h "
          f"= {full_core/3600/14/24:,.3g} d/14w = {full_core/3600/14/24/365:,.3g} yr/14w")
    print(f"  E[r]={rr.mean():.4g} (median {np.median(rr):.0f}); "
          f"top 1% of triples carry {np.sort(rc)[-N//100:].sum()/rc.sum():.1%} of full work")
    print(f"  ratio full/acceptance = {rc.mean()/cc.mean():.4g}x")
    print("\nInterpretation: the full search is bracketed BELOW by the acceptance-class")
    print("(~27 d/14w) and ABOVE by the r-tail-dominated model (astronomically large).")
    print("The true value is gated by the UNMEASURED EV-realisable support count <= r(R).")

if __name__ == "__main__":
    main()
