#!/usr/bin/env python3
"""Increment 1 of Fix #2: a SOUND witness-finder for positive-dimensional supports.

For a support whose indifference variety V = {g=0} is positive-dimensional, we still must
decide whether V meets the equilibrium feasibility conditions (strict>0, nonneg>=0, box).
This module samples candidate points on V via the CRITICAL-POINT METHOD and lets the caller
verify them EXACTLY. It is *sound* (any returned candidate that verify_witness accepts is a
true equilibrium) but not yet *complete* for proving emptiness -- that is a later increment.

Critical-point sampling (distance to a generic rational point p):
  minimise  ||x - p||^2  on  V = {g_1=0,...,g_c=0}
  =>  Lagrange/Jacobian condition: the gradients {grad(dist), grad(g_1),...,grad(g_c)} are
      linearly dependent, i.e. EVERY (c+1)x(c+1) minor of the (c+1) x nv Jacobian vanishes.
  Together with g=0 this is a zero-dimensional system (for generic p): its real solutions hit
  every connected component of the smooth part of V. We solve it exactly with rational_solutions
  (msolve, via Julia) and return the rational candidates for the caller to verify.

Free variables (vars in no equation) are handled uniformly: distance-to-p pins them, so no
special-casing. p is chosen as a fixed deterministic rational (genericity is validated by the
caller observing a zero-dimensional solve; on a degenerate p the solve returns posdim and we
re-perturb p before giving up).
"""
import os
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
import sympy as sp
from math import gcd
from pathlib import Path
from itertools import combinations

_JLP = None      # cached (juliacall Main, PosdimSolve.decide)


def _jl_posdim():
    global _JLP
    if _JLP is None:
        from juliacall import Main as jl
        src = (Path(__file__).resolve().parent / "posdim_solver.jl").read_text()
        jl.seval(src)
        _JLP = (jl, jl.seval("PosdimSolve.decide"))
    return _JLP


def _marshal(polys, syms):
    """list of sympy polys over `syms` -> (coeffs_list, exps_list): integer coeffs (denominators
    cleared) and flat per-term exponent rows, the marshalling PosdimSolve.decide expects."""
    C, E = [], []
    for e in polys:
        p = sp.Poly(sp.expand(e), *syms)
        cs, ms = [], []
        for monom, coef in p.terms():
            r = sp.Rational(coef)
            cs.append((r.p, r.q)); ms.append(monom)
        L = 1
        for _, d in cs:
            L = L * d // gcd(L, d)
        C.append([n * (L // d) for (n, d) in cs])
        flat = []
        for m in ms:
            flat.extend(int(v) for v in m)
        E.append(flat)
    return C, E

# deterministic "generic" probe points (rationals with distinct, non-symmetric coords); tried
# in order until one yields a zero-dimensional critical system.
_PROBES = [
    [sp.Rational(3 + 7 * i, 11 + 13 * i) for i in range(64)],
    [sp.Rational(5 + 2 * i, 17 + 3 * i) for i in range(64)],
    [sp.Rational(2 + 5 * i, 9 + 4 * i) for i in range(64)],
]


_GRAD_CACHE = {}


def _grad_row(g, syms):
    """Gradient row [dg/dx_j] of a polynomial g, memoised by structural identity. The base
    equations' Jacobian is identical across every boundary stratum, so this avoids recomputing
    it hundreds of times in decide_complete."""
    key = sp.srepr(g)
    row = _GRAD_CACHE.get(key)
    if row is None:
        row = [sp.diff(g, x) for x in syms]
        _GRAD_CACHE[key] = row
    return row


def critical_system(eqs, syms, probe, rho):
    """Polynomial system whose real solutions sample every connected component of the SMOOTH part
    of V={eqs=0}: eqs=0 AND x is a critical point of dist=sum (x-p)^2 restricted to V. The
    critical condition is grad(dist) in the row span of the Jacobian Jac_g, i.e. the augmented
    matrix [grad(dist); Jac_g] has rank <= rho = codim(V) = nv - dim(V): every (rho+1)x(rho+1)
    minor vanishes. Using the TRUE codimension rho (not the equation count) is essential -- the
    equations here are typically NOT a complete intersection (dependent, from free variables), so
    assuming rank = #eqs makes all minors vanish identically and constrains nothing."""
    nv = len(syms)
    dist = sum((x - probe[j]) ** 2 for j, x in enumerate(syms))
    grad = [sp.diff(dist, x) for x in syms]                # row 0
    jac = [_grad_row(g, syms) for g in eqs]                # rows 1..c (memoised across strata)
    c = len(jac)
    r = rho + 1
    minors = []
    # Only minors that INCLUDE the gradient-of-distance row: the pure-Jacobian (rho+1)-minors
    # vanish identically on V (rank Jac <= codim = rho everywhere on V), so they are redundant.
    if 1 <= r <= nv and r - 1 <= c:
        for jr in combinations(range(c), r - 1):
            M = sp.Matrix([grad] + [jac[i] for i in jr])   # r x nv
            for cs in combinations(range(nv), r):
                minors.append(M[:, list(cs)].det(method="bareiss"))
    sysm = [sp.expand(e) for e in eqs] + [sp.expand(m) for m in minors]
    return [e for e in sysm if e != 0]


def _jacobian_minors(eqs, syms, r):
    """All r x r minors of the Jacobian of `eqs` (the c x nv matrix d eqs_i / d x_j). These vanish
    exactly where rank(Jac) < r, i.e. on the SINGULAR locus of V={eqs=0} when r = codim(V)."""
    J = sp.Matrix([[sp.diff(g, x) for x in syms] for g in eqs])
    c, nv = J.rows, J.cols
    out = []
    if r <= min(c, nv):
        for rs in combinations(range(c), r):
            sub = J[list(rs), :]
            for cs in combinations(range(nv), r):
                out.append(sp.expand(sub[:, list(cs)].det()))
    return [m for m in out if m != 0]


class _Ctx:
    """Shared per-support solving context: the sign conditions (already marshalled) and the Julia
    handles, so the recursive sampler doesn't re-marshal/re-fetch them."""
    def __init__(self, cond):
        self.syms = cond["syms"]; self.nv = cond["nv"]
        strict = [sp.expand(e) for e in cond["strict"] if e != 0]
        box_nn = list(cond["box"]) + [sp.Integer(1) - b for b in cond["box"]]
        nonneg = [sp.expand(e) for e in (list(cond["nonneg"]) + box_nn) if e != 0]
        self.strict = strict; self.nonneg = nonneg
        jl, decide = _jl_posdim()
        self.decide = decide
        self.decide_sat = jl.seval("PosdimSolve.decide_sat")
        self.crit_decide = jl.seval("PosdimSolve.crit_decide")
        self.vdim = jl.seval("PosdimSolve.vdim")
        self.sc, self.se = _marshal(strict, self.syms) if strict else ([], [])
        self.nc, self.ne = _marshal(nonneg, self.syms) if nonneg else ([], [])

    def dim(self, eqsX):
        c, e = _marshal(eqsX, self.syms)
        return int(self.vdim(self.nv, c, e))


def _crit_feasible(eqsX, ctx):
    """Sample EVERY connected component of the real variety V={eqsX=0} and test the equilibrium
    sign conditions exactly. Smooth-locus components are reached by distance-critical points (the
    critical ideal SATURATED by the singular locus, so the singular part doesn't pollute it);
    singular components are reached by recursing onto the (lower-dimensional) singular locus.

    Returns ('feasible', witness) | ('feasible_irrational', None) | ('none', None) [no feasible
    point on V, certified] | ('degen', None) [a genericity self-check could not be discharged]."""
    syms, nv = ctx.syms, ctx.nv
    dX = ctx.dim(eqsX)
    if dX < 0:
        return ("none", None)                              # empty variety
    ec, ee = _marshal(eqsX, syms)
    if dX == 0:                                            # already zero-dim: decide directly
        st, _nr, fidx, wit, rat = ctx.decide(nv, ec, ee, ctx.sc, ctx.se, ctx.nc, ctx.ne)
        if int(st) == 1:
            return ("degen", None)
        return _verdict(fidx, wit, rat, syms)
    rho = nv - dX
    # FAST PATH: plain distance-critical points, built and solved entirely in Nemo (crit_decide:
    # Jacobian + (rho+1)-minors via compiled polynomial arithmetic -- no sympy determinants). If the
    # critical system is zero-dimensional it samples every component of V already (smooth components
    # via their critical points AND any ISOLATED singular points, where rank Jac < rho so the rank
    # condition holds trivially), so a 'none' verdict is a complete proof. Only a positive-dimensional
    # SINGULAR locus leaves it positive-dim, which needs the saturation slow path below.
    for probe in _PROBES:
        pnum = [int(probe[j].p) for j in range(nv)]
        pden = [int(probe[j].q) for j in range(nv)]
        st, _nr, fidx, wit, rat = ctx.crit_decide(nv, ec, ee, pnum, pden, rho,
                                                  ctx.sc, ctx.se, ctx.nc, ctx.ne)
        if int(st) == 1:                                   # positive-dim -> singular locus; slow path
            break
        return _verdict(fidx, wit, rat, syms)              # zero-dim: complete for V
    # SLOW PATH (positive-dimensional singular locus): isolate smooth critical points by saturating
    # the critical ideal by the singular locus, then recurse onto the (lower-dim) singular locus.
    singmins = _jacobian_minors(eqsX, syms, rho)
    sgc, sge = _marshal(singmins, syms) if singmins else ([], [])
    smooth_ok = False
    for probe in _PROBES:
        csys = critical_system(eqsX, syms, probe, rho)
        if not csys:
            continue
        cc, ce = _marshal(csys, syms)
        st, _nr, fidx, wit, rat = ctx.decide_sat(nv, ec, ee, cc, ce, sgc, sge,
                                                 ctx.sc, ctx.se, ctx.nc, ctx.ne)
        if int(st) == 1:
            continue
        smooth_ok = True
        v = _verdict(fidx, wit, rat, syms)
        if v[0] != "none":
            return v
        break
    if singmins:
        ds = ctx.dim(eqsX + singmins)
        if ds == dX:
            return ("degen", None)                         # singular locus not lower-dim: pathological
        if -1 < ds < dX:
            r = _crit_feasible(eqsX + singmins, ctx)
            if r[0] != "none":
                return r
    return ("none", None) if smooth_ok else ("degen", None)


def _verdict(fidx, wit, rat, syms):
    if int(fidx) == 0:
        return ("none", None)
    if bool(rat):
        return ("feasible", {str(s): sp.Rational(int(p[0]), int(p[1])) for s, p in zip(syms, wit)})
    return ("feasible_irrational", None)


def find_witness(cond):
    """SOUND witness-finder (increment 2b): sample the indifference variety and exact-verify the
    sign conditions. Returns ('feasible', dict) | ('feasible_irrational', None) |
    ('no_witness_critpts', None) [no feasible sample -- NOT a proof of emptiness] |
    ('degenerate', None). Never returns a false 'feasible' (every point is self-verified)."""
    ctx = _Ctx(cond)
    r = _crit_feasible(cond["eqs"], ctx)
    return {"none": ("no_witness_critpts", None), "degen": ("degenerate", None)}.get(r[0], r)


def decide_complete(cond):
    """COMPLETE posdim decision (increment 2c): return an exact witness, or PROVE no equilibrium
    exists on this support. For every subset (up to codim dim V) of the inequality-boundary family
    that actually vanishes on V, sample all components of V intersected with that boundary set
    (via _crit_feasible, which handles singular loci). A feasible sample anywhere => feasible;
    nothing across all strata => infeasible (proven). Returns ('feasible', dict) |
    ('feasible_irrational', None) | ('infeasible', None) | ('degenerate', None)."""
    ctx = _Ctx(cond)
    eqs = cond["eqs"]
    d = ctx.dim(eqs)
    # boundary family: dedup identical polynomials (strict orderings repeat), then keep only those
    # that actually vanish somewhere on V (others never cross 0 on V, so contribute no boundary).
    seen = {}
    for p in (ctx.strict + ctx.nonneg):
        seen.setdefault(sp.srepr(sp.expand(p)), p)
    Pfam = [p for p in seen.values() if ctx.dim(eqs + [p]) != -1]
    for k in range(0, d + 1):                              # strata A of size 0..dim V
        for A in combinations(range(len(Pfam)), k):
            r = _crit_feasible(eqs + [Pfam[i] for i in A], ctx)
            if r[0] in ("feasible", "feasible_irrational"):
                return r
            if r[0] == "degen":
                return ("degenerate", None)
    return ("infeasible", None)
