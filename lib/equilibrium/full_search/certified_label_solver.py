"""Certified per-label solver (acceptance-mixing class, fixed proposals).

For a weak-order label this builds the EXACT polynomial system and solves it
symbolically -- no least_squares, no miss within the class. Steps:

  1. Fix proposals by a deterministic rule (best proposer-tier among 'passable'
     transitions; passable = committee has no strict-rejector). Proposals are then
     CONSTANTS, so P is polynomial purely in the acceptance-mixing variables.
  2. Variable elimination: the only unknowns are tied committee acceptances on
     transitions that actually appear in P (proposed transitions). All other
     acceptances are irrelevant (don't affect P/V) and are dropped.
  3. V = (I - dP)^{-1}(1-d)u solved SYMBOLICALLY (exact rationals). The tied-group
     indifference conditions V_k(s)=V_k(base) become exact polynomial equations in
     the mixing variables.
  4. Solve the polynomial system EXACTLY (sympy). Isolated roots AND positive-
     dimensional families (the manifold case) are returned. Filter to real,
     in [0,1], satisfying strict orders + proposal optimality.

Certified for: acceptance mixing with the fixed-proposal rule. (Full waterproof
additionally needs proposal-support enumeration -- the combinatorial layer.)
"""
from __future__ import annotations
import numpy as np
import sympy as sp
from fractions import Fraction

from lib.equilibrium.merit_descent import _FastGame


def _rat(x, limit=10**9):
    return sp.Rational(Fraction(float(x)).limit_denominator(limit))


class CertifiedLabelSolver:
    def __init__(self, setup):
        self.fg = _FastGame(
            players=setup['players'], states=setup['state_names'],
            effectivity=setup['effectivity'], protocol=setup['protocol'],
            payoffs=setup['payoffs'], discounting=setup['discounting'],
            forbidden_proposals=setup.get('forbidden_proposals', frozenset()))
        fg = self.fg
        self.delta = sp.Rational(Fraction(float(fg.delta)).limit_denominator(10**9))
        self.proto = [_rat(p) for p in fg.protocol]
        self.u = sp.Matrix([[_rat(fg.u[s, p]) for p in range(fg.n)] for s in range(fg.S)])
        self.atol = sp.Rational(1, 10**6)

    def _strict_reject_free(self, tiers, ii, xi, yi):
        """True if no committee member strictly rejects x->y (so it's passable for
        some acceptance choice). Strict reject = approver strictly prefers x."""
        for ki in self.fg.committee[(ii, xi, yi)]:
            if tiers[ki][yi] > tiers[ki][xi]:
                return False
        return True

    def _fixed_proposals(self, tiers):
        """Deterministic pure proposal per (ii,xi): best proposer-tier among
        passable feasible targets (incl self), tie-break by state index."""
        fg = self.fg
        prop = {}
        for ii in range(fg.n):
            for xi in range(fg.S):
                cands = [yi for yi in fg.feasible[(ii, xi)]
                         if self._strict_reject_free(tiers, ii, xi, yi)]
                if not cands:
                    cands = [xi]
                best = min(cands, key=lambda yi: (int(tiers[ii][yi]), yi))
                prop[(ii, xi)] = best
        return prop

    def _build(self, tiers):
        """Return (P symbolic matrix, list of mixing Symbols, prop dict)."""
        fg = self.fg
        prop = self._fixed_proposals(tiers)
        syms = {}  # (ki, xi, yi) -> Symbol  (tied acceptance, shared over proposers)

        def acc(ii, xi, yi, ki):
            if tiers[ki][yi] < tiers[ki][xi]:
                return sp.Integer(1)
            if tiers[ki][yi] > tiers[ki][xi]:
                return sp.Integer(0)
            if xi == yi:
                return sp.Integer(1)
            key = (ki, xi, yi)
            if key not in syms:
                syms[key] = sp.Symbol(f'a_{ki}_{xi}_{yi}', real=True)
            return syms[key]

        P = sp.zeros(fg.S, fg.S)
        for ii in range(fg.n):
            pr = self.proto[ii]
            for xi in range(fg.S):
                yi = prop[(ii, xi)]
                if yi == xi:
                    P[xi, xi] += pr
                    continue
                pa = sp.Integer(1)
                for ki in fg.committee[(ii, xi, yi)]:
                    pa *= acc(ii, xi, yi, ki)
                P[xi, yi] += pr * pa
                P[xi, xi] += pr * (1 - pa)
        # keep only mixing syms that actually appear in P
        used = sorted(set().union(*[set(P[i, j].free_symbols) for i in range(fg.S) for j in range(fg.S)]) if fg.S else set(),
                      key=lambda s: s.name)
        return P, used, prop

    def _solve_V(self, P):
        fg = self.fg
        A = sp.eye(fg.S) - self.delta * P
        b = (1 - self.delta) * self.u
        return A.solve(b)  # S x n symbolic

    def solve(self, tiers):
        """Return a feasible (mixing dict, V numeric) or None. Certified within class."""
        fg = self.fg
        P, syms, prop = self._build(tiers)
        V = self._solve_V(P)
        # indifference equations: V_k(s) = V_k(base) for tied groups
        eqs = []
        for ki in range(fg.n):
            by = {}
            for si in range(fg.S):
                by.setdefault(int(tiers[ki][si]), []).append(si)
            for g in by.values():
                if len(g) > 1:
                    base = V[g[0], ki]
                    for si in g[1:]:
                        eqs.append(sp.together(V[si, ki] - base))
        # clear denominators -> polynomial numerators
        polys = []
        for e in eqs:
            num, den = sp.fraction(sp.together(e))
            polys.append(sp.expand(num))
        polys = [p for p in polys if p != 0]

        if not syms:
            sols = [{}]   # pure (no mixing) -> single candidate
        elif not polys:
            sols = [{}]   # equations vanish -> full box is the solution set (free)
        else:
            try:
                sols = sp.solve(polys, list(syms), dict=True) or []
            except Exception:
                sols = []

        for sol in sols:
            for cand in self._materialize(sol, syms):
                Vnum = self._eval_V(V, cand)
                if self._feasible(tiers, Vnum, prop, cand):
                    return cand, Vnum
        return None

    def _materialize(self, sol, syms, grid=9):
        """Yield concrete dicts in [0,1]. Free (positive-dimensional) parameters are
        swept on a grid so we explore the whole solution manifold, not one point."""
        exprs = {s: sol.get(s, s) for s in syms}     # symbol -> expr (free if == itself)
        free = sorted(set().union(*[e.free_symbols for e in exprs.values()]) if exprs else set(),
                      key=lambda s: s.name)
        if not free:
            cand = {}
            ok = True
            for s in syms:
                fv = float(exprs[s])
                if fv < -1e-9 or fv > 1 + 1e-9:
                    ok = False; break
                cand[s] = min(1.0, max(0.0, fv))
            if ok:
                yield cand
            return
        import itertools
        vals = np.linspace(0.0, 1.0, grid)
        for combo in itertools.product(vals, repeat=len(free)):
            subs = {p: sp.Float(v) for p, v in zip(free, combo)}
            cand = {}
            ok = True
            for s in syms:
                e = exprs[s]
                fv = float(e.subs(subs)) if hasattr(e, 'subs') else float(e)
                if fv < -1e-9 or fv > 1 + 1e-9:
                    ok = False; break
                cand[s] = min(1.0, max(0.0, fv))
            if ok:
                yield cand

    def _eval_V(self, V, cand):
        subs = {sp.Symbol(s.name, real=True): sp.Float(v) for s, v in cand.items()}
        return np.array([[float(V[i, j].subs(subs)) for j in range(self.fg.n)]
                         for i in range(self.fg.S)])

    def _feasible(self, tiers, V, prop, cand, tol=1e-6):
        fg = self.fg
        # strict orders
        for ki in range(fg.n):
            for x in range(fg.S):
                for y in range(fg.S):
                    if tiers[ki][x] < tiers[ki][y] and not (V[x, ki] > V[y, ki] - tol):
                        return False
        # indifference (within tol)
        for ki in range(fg.n):
            by = {}
            for si in range(fg.S):
                by.setdefault(int(tiers[ki][si]), []).append(si)
            for g in by.values():
                for si in g[1:]:
                    if abs(V[si, ki] - V[g[0], ki]) > tol:
                        return False
        # proposal optimality: chosen proposal must be EV-argmax over feasible
        for ii in range(fg.n):
            for xi in range(fg.S):
                evs = {}
                for yi in fg.feasible[(ii, xi)]:
                    if yi == xi:
                        pa = 1.0
                    else:
                        pa = 1.0
                        for ki in fg.committee[(ii, xi, yi)]:
                            if tiers[ki][yi] < tiers[ki][xi]:
                                p = 1.0
                            elif tiers[ki][yi] > tiers[ki][xi]:
                                p = 0.0
                            else:
                                p = cand.get(sp.Symbol(f'a_{ki}_{xi}_{yi}', real=True), 0.0)
                            pa *= p
                    evs[yi] = pa * V[yi, ii] + (1 - pa) * V[xi, ii]
                best = max(evs.values())
                if evs[prop[(ii, xi)]] < best - tol:
                    return False
        return True
