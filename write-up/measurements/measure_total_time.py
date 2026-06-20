#!/usr/bin/env python3
"""Concrete total-time analysis for the acceptance-class certified search, with the
EXACT FLINT interpolation build, differentiated by case category (Section 7.9).

Per-label cost model (exact build + cheap solve):
   c(m) = (interpolation build) + (polynomial solve)
The interpolation build does 2^m exact FLINT vertex solves; the solve is ms-scale.
We MEASURE c(m) for m up to 8 (real interp build + sympy.solve), and for the rare
m>=9 extrapolate the build as 2^m * c_vertex (the production build uses the fast
O(m 2^m) Mobius transform, so vertex solves dominate). We then weight by the measured
m-distribution to get the total over 541^3 labels and the wall time on 14 (and 134)
workers, with a category breakdown.
"""
import sys, time, signal
from pathlib import Path
import numpy as np, sympy as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "write-up" / "measurements"))
from bench_build_interp import (cls, fg, n, S, orders, NO, interp_build,
                                V_at_vertex, free_vars_and_prop, m_of)

TOTAL = NO ** 3

# ---- m-distribution over a large cheap sample ----
def m_distribution(N=200000, seed=11):
    rng = np.random.RandomState(seed)
    import collections
    cnt = collections.Counter()
    for _ in range(N):
        t = tuple(orders[rng.randint(NO)] for _ in range(n))
        cnt[m_of(t)] += 1
    return cnt, N

# ---- c_vertex: one exact FLINT vertex solve+det ----
def measure_c_vertex(reps=3000, seed=1):
    rng = np.random.RandomState(seed)
    # use a triple with some mixing so keys is non-empty
    t = None
    for _ in range(10000):
        tt = tuple(orders[rng.randint(NO)] for _ in range(n))
        if m_of(tt) >= 2:
            t = tt; break
    keys, prop = free_vars_and_prop(t)
    bits = [0] * len(keys)
    t0 = time.time()
    for _ in range(reps):
        V_at_vertex(t, prop, keys, bits)
    return (time.time() - t0) / reps

# ---- per-m measured cost, build and solve timed SEPARATELY ----
class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))

def measure_cost_by_m(want=range(0, 12), per=4, seed=2, solve_timeout=120):
    rng = np.random.RandomState(seed)
    bins = {m: [] for m in want}
    for _ in range(600000):
        if all(len(bins[m]) >= per for m in want): break
        t = tuple(orders[rng.randint(NO)] for _ in range(n)); mm = m_of(t)
        if mm in bins and len(bins[mm]) < per: bins[mm].append(t)
    build_c, solve_c, solve_dnf = {}, {}, {}
    for m in sorted(bins):
        tb, tsv, dnf = [], [], 0
        for t in bins[m]:
            t0 = time.time(); polys, syms = interp_build(t); tb.append(time.time() - t0)
            if syms:
                signal.setitimer(signal.ITIMER_REAL, solve_timeout); s0 = time.time()
                try:
                    sp.solve(polys, syms, dict=True); tsv.append(time.time() - s0)
                except _TO:
                    dnf += 1
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        if tb: build_c[m] = float(np.median(tb))
        if tsv: solve_c[m] = float(np.median(tsv))
        solve_dnf[m] = dnf
    return build_c, solve_c, solve_dnf


def main():
    print(f"instance=burke_usaruschn  total labels = 541^3 = {TOTAL:,}")
    cnt, N = m_distribution()
    cvx = measure_c_vertex()
    build_c, solve_c, solve_dnf = measure_cost_by_m()
    print(f"c_vertex (one exact FLINT solve+det) = {cvx*1e6:.1f} us")
    mmax = max(cnt)
    # cost model c(m) = build(m) + solve(m); build from fast model 2^m*c_vertex if
    # not measured; solve measured (DNF flagged).
    def c(m):
        b = build_c.get(m, (2 ** m) * cvx)
        s = solve_c.get(m, 0.0)
        return b + s
    print("\nper-m: build = 2^m FLINT vertex solves (fast Mobius); solve = sympy")
    print(f"{'m':>2} {'build_s':>9} {'solve_s':>9} {'dnf':>4} {'c(m)':>10}")
    for m in sorted(set(list(build_c) + list(solve_c))):
        print(f"{m:>2} {build_c.get(m,float('nan')):>9.4f} "
              f"{solve_c.get(m,0.0):>9.4f} {solve_dnf.get(m,0):>4} {c(m):>10.4f}")
    print(f"\n{'m':>2} {'freq':>9} {'c(m) [s]':>11} {'labels(541^3)':>15} {'core-sec':>14}")
    cat = {"easy (m<=2)": [0, 1, 2], "medium (3-5)": [3, 4, 5],
           "hard (6-8)": [6, 7, 8], "extreme (>=9)": list(range(9, mmax + 1))}
    core_by_cat = {k: 0.0 for k in cat}; lab_by_cat = {k: 0.0 for k in cat}
    total_core = 0.0
    for m in sorted(cnt):
        freq = cnt[m] / N
        labels = freq * TOTAL
        core = labels * c(m)
        total_core += core
        for k, ms in cat.items():
            if m in ms: core_by_cat[k] += core; lab_by_cat[k] += labels
        print(f"{m:>2} {freq:>9.4f} {c(m):>11.4g} {labels:>15,.0f} {core:>14,.0f}")
    print(f"\n{'category':<16}{'%labels':>9}{'core-hours':>13}{'14-worker':>12}{'134-worker':>12}")
    for k in cat:
        ch = core_by_cat[k] / 3600
        print(f"{k:<16}{100*lab_by_cat[k]/TOTAL:>8.2f}%{ch:>13,.1f}{ch/14:>11.1f}h{ch/134:>11.1f}h")
    print(f"{'TOTAL':<16}{'100.00%':>9}{total_core/3600:>13,.1f}"
          f"{total_core/3600/14:>11,.1f}h{total_core/3600/134:>11,.1f}h")
    print("\n(acceptance-class certified search; exact FLINT interpolation build + exact solve.")
    print(" Full proposal-mixing search is NOT this -- it carries the heavy r-tail, Section 7.5.)")


if __name__ == "__main__":
    main()
