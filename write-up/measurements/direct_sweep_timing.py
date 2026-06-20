#!/usr/bin/env python3
"""Direct, faithful timing: run the ACTUAL certified acceptance-class solver on a
random sample of labels and extrapolate. No component model, no multiplication --
just draw N labels, solve each (exact FLINT interpolation build + exact solve),
record wall time, and report what a complete (and near-complete) sweep costs.

Per-label wall cap is a MEASUREMENT bound only (rare high-m labels); capped labels
are recorded as '>=cap' so the full-sweep figure is an explicit lower bound there,
while the cheapest-fraction figures (which never hit the cap) are exact.
"""
import sys, time, signal, csv
from pathlib import Path
import numpy as np, sympy as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "write-up" / "measurements"))
from bench_build_interp import cls, fg, n, orders, NO, interp_build
HERE = Path(__file__).resolve().parent
TOTAL = NO ** 3

class _TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))

def solve_one(tiers, cap):
    """Full per-label exact pipeline: interp build + exact solve. Returns (time, capped)."""
    t0 = time.time()
    signal.setitimer(signal.ITIMER_REAL, cap)
    try:
        polys, syms = interp_build(tiers)
        if syms:
            sp.solve(polys, syms, dict=True)
        return time.time() - t0, False
    except _TO:
        return float(cap), True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def main(N=10000, cap=20.0, seed=0):
    rng = np.random.RandomState(seed)
    times = np.empty(N); capped = np.zeros(N, bool)
    t0 = time.time()
    for i in range(N):
        tiers = tuple(orders[rng.randint(NO)] for _ in range(n))
        times[i], capped[i] = solve_one(tiers, cap)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{N} done, elapsed {time.time()-t0:.0f}s, capped {capped[:i+1].sum()}", flush=True)
    wall = time.time() - t0
    with open(HERE / "direct_timings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["time_s", "capped"]); w.writerows(zip(times, capped.astype(int)))

    scale = TOTAL / N
    order = np.sort(times); cum = np.cumsum(order)
    print(f"\nsample N={N}, measured in {wall:.0f}s wall; {capped.sum()} labels hit the {cap:.0f}s cap")
    print(f"per-label: mean={times.mean():.4f}s median={np.median(times):.4f}s  "
          f"(mean is a lower bound -- {capped.sum()} capped)")
    print(f"\nComplete & near-complete sweep of all {TOTAL:,} labels (acceptance class):")
    print(f"{'fraction (cheapest)':>22} {'core-hours':>12} {'14 workers':>12} {'134 workers':>12}")
    for p in [0.50, 0.90, 0.99, 0.999, 1.00]:
        k = max(1, int(p * N))
        core_h = cum[k - 1] * scale / 3600
        tag = "  (LB: capped)" if (p == 1.00 and capped.any()) else ""
        print(f"{('cheapest %.1f%%' % (100*p)):>22} {core_h:>12,.1f} "
              f"{core_h/14:>10,.1f}h {core_h/134:>10,.1f}h{tag}")
    full_core_h = cum[-1] * scale / 3600
    print(f"\nFULL sweep (acceptance class): {full_core_h/24/14:,.1f} days on 14 workers, "
          f"{full_core_h/24/134:,.1f} days on 134  (lower bound; tail capped at {cap:.0f}s)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--cap", type=float, default=20.0); a = ap.parse_args()
    main(N=a.N, cap=a.cap)
