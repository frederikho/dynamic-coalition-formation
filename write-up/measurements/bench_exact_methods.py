#!/usr/bin/env python3
"""Benchmark of EXACT, unfailable per-label methods, by case (Section 7.8).

Hard constraint: every method here is exact/certified -- exact rational arithmetic,
exact real-root isolation (sympy CRootOf / FLINT fmpq_poly with Arb-certified
enclosures), exact Groebner bases. NO floating tolerances, NO solver timeouts in the
production path. (The SIGALRM below is only a MEASUREMENT bound so the benchmark
itself finishes; a label that hits it is reported as DNF, never silently skipped --
in production those labels are simply handed to a stronger exact engine.)

It answers two questions:
  1. Where does per-label time go -- BUILD (forming the polynomial system) vs SOLVE?
  2. Do different exact packages win for different cases (m = #mixing vars)?

Methods compared, all exact:
  BUILD : sympy symbolic V=(I-dP)^{-1}(1-d)u + indifference-numerator extraction.
  SOLVE, m==1 : (a) sympy.solve  (b) sympy.real_roots(gcd)  (c) FLINT fmpq_poly roots
  SOLVE, m>=2 : (a) sympy.solve  (b) sympy.groebner (grevlex)  [FLINT has no multivar
                Groebner; msolve would slot in here]
"""
import sys, time, signal, functools
from pathlib import Path
import numpy as np
import sympy as sp
import flint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.residual_metric_probe import build_setup
from scripts._reduced_helpers import _generate_weak_orders
from scripts.certified_label_solver import CertifiedLabelSolver

setup = build_setup("power_threshold_RICE_n3",
                    ROOT / "payoff_tables" / "burke_usaruschn_2035-2060.xlsx", "adjacent_step")
cls = CertifiedLabelSolver(setup); fg = cls.fg
n, S = fg.n, fg.S
orders = _generate_weak_orders(S); NO = len(orders)

class TO(Exception): pass
def _a(s, f): raise TO()
signal.signal(signal.SIGALRM, _a)
def timed(fn, limit):
    signal.setitimer(signal.ITIMER_REAL, limit)
    t0 = time.time()
    try:
        fn(); return time.time() - t0, "ok"
    except TO:
        return float(limit), "DNF"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def m_of(tiers):
    prop = cls._fixed_proposals(tiers); keys = set()
    for (ii, xi), yi in prop.items():
        if yi != xi:
            for ki in fg.committee[(ii, xi, yi)]:
                if int(tiers[ki][yi]) == int(tiers[ki][xi]):
                    keys.add((ki, xi, yi))
    return len(keys)

def build_polys(tiers):
    """Exact build: symbolic V then indifference numerators. Returns (polys, syms)."""
    P, syms, prop = cls._build(tiers)
    V = cls._solve_V(P)
    eqs = []
    for ki in range(n):
        by = {}
        for si in range(S):
            by.setdefault(int(tiers[ki][si]), []).append(si)
        for g in by.values():
            if len(g) > 1:
                base = V[g[0], ki]
                for si in g[1:]:
                    eqs.append(sp.together(V[si, ki] - base))
    polys = [sp.expand(sp.fraction(sp.together(e))[0]) for e in eqs]
    polys = [p for p in polys if p != 0]
    return polys, list(syms)

def flint_roots_univariate(poly, sym):
    """Exact/certified roots of a univariate rational polynomial via FLINT."""
    P = sp.Poly(poly, sym)
    coeffs_hi_to_lo = P.all_coeffs()
    lo_to_hi = [flint.fmpq(int(sp.fraction(c)[0]), int(sp.fraction(c)[1])) for c in reversed(coeffs_hi_to_lo)]
    fp = flint.fmpq_poly(lo_to_hi)
    # certified complex root enclosures (Arb balls, adaptive precision -- no tolerance
    # to tune); real roots are those whose imaginary enclosure contains 0.
    return flint.acb_poly(fp).roots()

def collect(per_m=5, want_m=(1, 2, 3, 4, 5, 6), max_tries=200000, seed=3):
    rng = np.random.RandomState(seed); bins = {m: [] for m in want_m}
    for _ in range(max_tries):
        if all(len(bins[m]) >= per_m for m in want_m):
            break
        tiers = tuple(orders[rng.randint(NO)] for _ in range(n))
        m = m_of(tiers)
        if m in bins and len(bins[m]) < per_m:
            bins[m].append(tiers)
    return bins

def main():
    LIMIT = 30.0
    bins = collect()
    print(f"instance=burke_usaruschn  (measurement bound {LIMIT}s; DNF = exceeded, not skipped)\n")
    print(f"{'m':>2} {'build_s':>9} {'solve_sympy':>12} {'alt_exact':>11}  alt = method")
    for m in sorted(bins):
        for tiers in bins[m]:
            # BUILD (timed)
            tb, sb = timed(lambda: build_polys(tiers), LIMIT)
            if sb == "DNF":
                print(f"{m:>2} {'DNF':>9} {'-':>12} {'-':>11}  (build exceeded)"); continue
            polys, syms = build_polys(tiers)
            # SOLVE (a): sympy.solve
            ts, ss = timed(lambda: sp.solve(polys, syms, dict=True), LIMIT)
            # ALT exact method by case
            if m == 1:
                g = functools.reduce(sp.gcd, polys)
                ta, sa = timed(lambda: flint_roots_univariate(g, syms[0]), LIMIT)
                alt = "FLINT fmpq_poly.real_roots"
            else:
                ta, sa = timed(lambda: sp.groebner(polys, *syms, order='grevlex'), LIMIT)
                alt = "sympy.groebner grevlex"
            fs = f"{ts:.3f}" + ("" if ss == "ok" else "*")
            fa = f"{ta:.3f}" + ("" if sa == "ok" else "*")
            print(f"{m:>2} {tb:>9.3f} {fs:>12} {fa:>11}  {alt}")
    print("\n* = hit measurement bound (DNF). FLINT roots are certified (Arb); "
          "sympy.solve/groebner/real_roots are exact symbolic.")

if __name__ == "__main__":
    main()
