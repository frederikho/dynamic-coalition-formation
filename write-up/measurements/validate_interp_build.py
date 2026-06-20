#!/usr/bin/env python3
"""End-to-end correctness check of the EXACT FLINT interpolation build.

We do NOT rely on polynomial-identity comparison with the symbolic build (which
differs by a det-factor that sympy cancels). Instead we use the gold standard: build
the indifference system by interpolation, solve it, materialise a feasible point, and
check it with the real verify_equilibrium on a KNOWN game (the manifold game, which
has a verified mixed equilibrium / continuum). If verify passes, the interpolation
build is correct for the solver's purpose.
"""
import sys, itertools
from pathlib import Path
import numpy as np, sympy as sp, flint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mixed_merit_probe import build_manifold_setup
from scripts.residual_metric_probe import value_of_strategy, empty_strategy_df
from scripts.certified_label_solver import CertifiedLabelSolver
from lib.utils import verify_equilibrium


def fq(r):
    r = sp.nsimplify(r); p, q = sp.fraction(sp.Rational(r)); return flint.fmpq(int(p), int(q))


def interp_polys(cls, tiers):
    """Exact indifference polynomials via FLINT vertex interpolation (multilinear)."""
    fg = cls.fg; n, S = fg.n, fg.S
    DELTA = fq(cls.delta); PROTO = [fq(p) for p in cls.proto]
    U = [[fq(cls.u[s, j]) for j in range(n)] for s in range(S)]; ONE = flint.fmpq(1)
    prop = cls._fixed_proposals(tiers)
    keys, seen = [], set()
    for ii in range(n):
        for xi in range(S):
            yi = prop[(ii, xi)]
            if yi == xi: continue
            for ki in fg.committee[(ii, xi, yi)]:
                if int(tiers[ki][yi]) == int(tiers[ki][xi]) and (ki, xi, yi) not in seen:
                    seen.add((ki, xi, yi)); keys.append((ki, xi, yi))
    syms = [sp.Symbol(f"a_{ki}_{xi}_{yi}", real=True) for (ki, xi, yi) in keys]
    m = len(keys)

    def V_det(bits):
        val = {k: bits[j] for j, k in enumerate(keys)}
        P = [[flint.fmpq(0)] * S for _ in range(S)]
        P = [[flint.fmpq(0) for _ in range(S)] for _ in range(S)]
        for ii in range(n):
            pr = PROTO[ii]
            for xi in range(S):
                yi = prop[(ii, xi)]
                if yi == xi: P[xi][xi] += pr; continue
                pa = ONE
                for ki in fg.committee[(ii, xi, yi)]:
                    ty, tx = int(tiers[ki][yi]), int(tiers[ki][xi])
                    a = ONE if ty < tx else (flint.fmpq(0) if ty > tx else flint.fmpq(int(val[(ki, xi, yi)])))
                    pa *= a
                P[xi][yi] += pr * pa; P[xi][xi] += pr * (ONE - pa)
        A = flint.fmpq_mat([[(ONE if i == j else flint.fmpq(0)) - DELTA * P[i][j] for j in range(S)] for i in range(S)])
        b = flint.fmpq_mat([[(ONE - DELTA) * U[i][j] for j in range(n)] for i in range(S)])
        Vm = A.solve(b)
        return [[Vm[i, j] for j in range(n)] for i in range(S)], A.det()

    eqs = []
    for ki in range(n):
        by = {}
        for si in range(S): by.setdefault(int(tiers[ki][si]), []).append(si)
        for g in by.values():
            if len(g) > 1:
                for si in g[1:]: eqs.append((ki, si, g[0]))
    verts = list(itertools.product([0, 1], repeat=m)) if m else [()]
    fvals = {}
    for v in verts:
        V, det = V_det(list(v))
        for e, (ki, si, bi) in enumerate(eqs):
            fvals[(e, v)] = (V[si][ki] - V[bi][ki]) * det
    polys = []
    for e in range(len(eqs)):
        poly = sp.Integer(0)
        for T in verts:
            Tset = [j for j in range(m) if (T[j] if m else 0)]
            c = flint.fmpq(0)
            for Uu in itertools.product(*[([0, 1] if j in Tset else [0]) for j in range(m)]):
                c += (-1) ** (sum(T) - sum(Uu)) * fvals[(e, Uu)]
            if c != 0:
                mono = sp.Integer(1)
                for j in Tset: mono *= syms[j]
                poly += sp.Rational(int(c.p), int(c.q)) * mono
        polys.append(sp.expand(poly))
    return [p for p in polys if p != 0], syms, keys, prop


def main():
    setup = build_manifold_setup(); players = setup["players"]; states = setup["state_names"]
    si = {s: i for i, s in enumerate(states)}
    cls = CertifiedLabelSolver(setup); fg = cls.fg

    def mk(b):
        t = np.ones(len(states), dtype=np.int64); t[si[b]] = 0; return t
    tiers = (mk("A"), mk("B"))   # known manifold ranking
    polys, syms, keys, prop = interp_polys(cls, tiers)
    print("interp system (manifold ranking):")
    for p in polys: print("   ", p)
    sols = sp.solve(polys, syms, dict=True)
    print("solutions:", sols)

    # materialise: sweep free params on a small grid, find a feasible verified point
    exprs = {s: (sols[0].get(s, s) if sols else s) for s in syms}
    free = sorted(set().union(*[e.free_symbols for e in exprs.values()]) if exprs else set(), key=str)
    found = False
    grid = np.linspace(0, 1, 9)
    for combo in itertools.product(grid, repeat=len(free)):
        sub = {p: sp.Rational(float(v)).limit_denominator(1000) for p, v in zip(free, combo)}
        avals = {}
        ok = True
        for s in syms:
            val = sp.nsimplify(exprs[s].subs(sub)) if hasattr(exprs[s], "subs") else exprs[s]
            fv = float(val)
            if fv < -1e-9 or fv > 1 + 1e-9: ok = False; break
            avals[s] = min(1.0, max(0.0, fv))
        if not ok: continue
        # build strategy df from prop + acceptances (strict from tiers, mixing from avals)
        df = empty_strategy_df(players, states)
        for ii in range(fg.n):
            for xi in range(fg.S):
                ch = prop[(ii, xi)]
                for yi in range(fg.S):
                    df.loc[(states[xi], "Proposition", np.nan), (f"Proposer {players[ii]}", states[yi])] = 1.0 if yi == ch else 0.0
        for k in fg.appr_keys:
            ii, xi, yi, ki = k
            if xi == yi: a = 1.0
            elif tiers[ki][yi] < tiers[ki][xi]: a = 1.0
            elif tiers[ki][yi] > tiers[ki][xi]: a = 0.0
            else: a = avals.get(sp.Symbol(f"a_{ki}_{xi}_{yi}", real=True), 0.0)
            df.loc[(states[xi], "Acceptance", players[ki]), (f"Proposer {players[ii]}", states[yi])] = float(a)
        V, _, pp, pa = value_of_strategy(df, setup)
        res = {"players": players, "state_names": states, "V": V.astype(float), "P_proposals": pp,
               "P_approvals": pa, "effectivity": setup["effectivity"], "strategy_df": df.fillna(0.0),
               "forbidden_proposals": setup.get("forbidden_proposals", frozenset())}
        if verify_equilibrium(res, atol=1e-6)[0]:
            print(f"\nFEASIBLE point via interp build -> verify_equilibrium = True")
            print("  mixing:", {str(k): round(v, 4) for k, v in avals.items()})
            found = True; break
    if not found:
        print("\nno feasible point found (interp build may be wrong)")


if __name__ == "__main__":
    main()
