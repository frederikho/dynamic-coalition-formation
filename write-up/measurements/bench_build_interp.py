#!/usr/bin/env python3
"""EXACT interpolation build vs symbolic build (Section 7.8).

Finding from bench_exact_methods.py: per-label cost is dominated by the BUILD
(forming the indifference polynomials), specifically sympy's symbolic matrix inverse
V=(I-dP)^{-1}(1-d)u; the SOLVE is essentially free. This script implements and
validates the exact replacement:

  KEY FACT: each mixing acceptance variable appears to degree <= 1 everywhere in P,
  so det(I-dP) and the numerators are MULTILINEAR in the m mixing variables. A
  multilinear polynomial is determined by its values on the 2^m vertices {0,1}^m.
  We therefore evaluate V exactly at each vertex via a FLINT exact-rational linear
  solve (fmpq_mat), and reconstruct each indifference polynomial exactly by the
  Mobius (inclusion-exclusion) transform. Fully exact -- rational arithmetic only,
  no tolerance, no float. We verify the interpolated polynomials equal the symbolic
  ones, and compare build time.
"""
import sys, time, itertools, signal
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

# FLINT exact constants
DELTA = flint.fmpq(int(sp.fraction(cls.delta)[0]), int(sp.fraction(cls.delta)[1]))
PROTO = [flint.fmpq(int(sp.fraction(p)[0]), int(sp.fraction(p)[1])) for p in cls.proto]
U = [[flint.fmpq(int(sp.fraction(cls.u[s, j])[0]), int(sp.fraction(cls.u[s, j])[1])) for j in range(n)]
     for s in range(S)]
ONE = flint.fmpq(1)


def free_vars_and_prop(tiers):
    """Return (ordered list of mixing keys (ki,xi,yi), proposals dict)."""
    prop = cls._fixed_proposals(tiers)
    keys = []
    seen = set()
    for ii in range(n):
        for xi in range(S):
            yi = prop[(ii, xi)]
            if yi == xi:
                continue
            for ki in fg.committee[(ii, xi, yi)]:
                if int(tiers[ki][yi]) == int(tiers[ki][xi]) and (ki, xi, yi) not in seen:
                    seen.add((ki, xi, yi)); keys.append((ki, xi, yi))
    return keys, prop


def V_at_vertex(tiers, prop, keys, bits):
    """Exact V (list S x n of fmpq) with each mixing key set to bits[j] in {0,1}."""
    val = {k: bits[j] for j, k in enumerate(keys)}
    # build P (fmpq) S x S
    P = [[flint.fmpq(0) for _ in range(S)] for _ in range(S)]
    for ii in range(n):
        pr = PROTO[ii]
        for xi in range(S):
            yi = prop[(ii, xi)]
            if yi == xi:
                P[xi][xi] += pr; continue
            pa = ONE
            for ki in fg.committee[(ii, xi, yi)]:
                t_y, t_x = int(tiers[ki][yi]), int(tiers[ki][xi])
                if t_y < t_x:
                    a = ONE
                elif t_y > t_x:
                    a = flint.fmpq(0)
                else:
                    a = flint.fmpq(int(val[(ki, xi, yi)]))
                pa *= a
            P[xi][yi] += pr * pa
            P[xi][xi] += pr * (ONE - pa)
    # A = I - DELTA P ; b = (1-DELTA) u ; V = A^{-1} b
    A = flint.fmpq_mat([[ (ONE if i == j else flint.fmpq(0)) - DELTA * P[i][j] for j in range(S)] for i in range(S)])
    b = flint.fmpq_mat([[ (ONE - DELTA) * U[i][j] for j in range(n)] for i in range(S)])
    Vm = A.solve(b)
    detA = A.det()
    return [[Vm[i, j] for j in range(n)] for i in range(S)], detA


def interp_build(tiers):
    """Exact interpolation build: returns list of indifference polynomials (sympy)
    in the mixing symbols, via 2^m vertex evaluations + Mobius transform."""
    keys, prop = free_vars_and_prop(tiers)
    m = len(keys)
    syms = [sp.Symbol(f"a_{ki}_{xi}_{yi}", real=True) for (ki, xi, yi) in keys]
    # which (player, target s, base) indifference equations
    eqs = []
    for ki in range(n):
        by = {}
        for si in range(S):
            by.setdefault(int(tiers[ki][si]), []).append(si)
        for g in by.values():
            if len(g) > 1:
                for si in g[1:]:
                    eqs.append((ki, si, g[0]))
    if m == 0:
        # no mixing: evaluate once, equations are constants
        V, detA = V_at_vertex(tiers, prop, keys, [])
        out=[]
        for (ki, si, bi) in eqs:
            num=(V[si][ki]-V[bi][ki])*detA
            out.append(sp.Rational(int(num.p), int(num.q)))
        return out, syms
    # evaluate the numerator at all 2^m vertices, indexed by bitmask
    M = 1 << m
    fnum = [[None] * M for _ in eqs]
    for mask in range(M):
        bits = [(mask >> j) & 1 for j in range(m)]
        V, detA = V_at_vertex(tiers, prop, keys, bits)
        for e, (ki, si, bi) in enumerate(eqs):
            fnum[e][mask] = (V[si][ki] - V[bi][ki]) * detA
    # FAST Mobius (inverse subset-zeta) transform: O(m * 2^m), not O(4^m)
    polys = []
    for e in range(len(eqs)):
        g = fnum[e][:]
        for j in range(m):
            bit = 1 << j
            for mask in range(M):
                if mask & bit:
                    g[mask] = g[mask] - g[mask ^ bit]
        poly = sp.Integer(0)
        for mask in range(M):
            c = g[mask]
            if c != 0:
                mono = sp.Integer(1)
                for j in range(m):
                    if mask & (1 << j):
                        mono *= syms[j]
                poly += sp.Rational(int(c.p), int(c.q)) * mono
        polys.append(sp.expand(poly))
    return [p for p in polys if p != 0], syms


def symbolic_build(tiers):
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
    return [p for p in polys if p != 0], list(syms)


class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
def timed(fn, limit=30.0):
    signal.setitimer(signal.ITIMER_REAL, limit); t0 = time.time()
    try:
        r = fn(); return r, time.time() - t0, "ok"
    except TO:
        return None, float(limit), "DNF"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def m_of(tiers):
    return len(free_vars_and_prop(tiers)[0])


def main():
    rng = np.random.RandomState(5)
    want = (1, 2, 3, 4, 5, 6, 7, 8); per = 4
    bins = {k: [] for k in want}
    for _ in range(300000):
        if all(len(bins[k]) >= per for k in want): break
        t = tuple(orders[rng.randint(NO)] for _ in range(n)); mm = m_of(t)
        if mm in bins and len(bins[mm]) < per: bins[mm].append(t)
    print("verifying interpolation == symbolic, and timing build (s):")
    print(f"{'m':>2} {'symbolic':>10} {'interp(FLINT)':>14} {'speedup':>8} {'match':>6}")
    for m in sorted(bins):
        for t in bins[m]:
            (sp_polys), ts, ss = timed(lambda: symbolic_build(t))
            (ip), ti, si = timed(lambda: interp_build(t))
            if si != "ok":
                print(f"{m:>2} {('%.3f'%ts) if ss=='ok' else 'DNF':>10} {'DNF':>14}"); continue
            ip_polys, isyms = ip
            match = "n/a"
            if ss == "ok":
                sp_polys_, _ = sp_polys
                # compare as sets of polynomials up to sign/scalar: check each interp poly
                # is a scalar multiple match by comparing normalized (monic-ish) forms
                def norm(ps):
                    out = set()
                    for p in ps:
                        p = sp.expand(p)
                        if p == 0: continue
                        out.add(sp.srepr(sp.expand(p / sp.LC(sp.Poly(p, *isyms)))) if isyms else sp.srepr(p))
                    return out
                try:
                    match = "yes" if norm(sp_polys_) == norm(ip_polys) else "DIFF"
                except Exception:
                    match = "?"
            sp_str = ("%.3f" % ts) if ss == "ok" else "DNF"
            spd = ("%.1fx" % (ts / ti)) if ss == "ok" and ti > 0 else "-"
            print(f"{m:>2} {sp_str:>10} {ti:>14.3f} {spd:>8} {match:>6}")
    print("\ninterp is exact (rational Mobius over 2^m vertices, FLINT fmpq solves); "
          "match=yes confirms identical polynomials.")


if __name__ == "__main__":
    main()
