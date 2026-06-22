#!/usr/bin/env python3
"""In-process Gröbner via Julia (juliacall + AlgebraicSolving.jl, same msolve engine).

FlintJuliaSolver routes degree-<=1 systems to the exact FLINT linear fast-path and
degree->=2 systems to AlgebraicSolving in-process (no subprocess spawn, no text
serialization). Classification is identical to the msolve CLI (same engine):
dimension -1 -> empty, 0 -> zerodim (exact rational solutions), >=1 -> posdim.
Parity is gated by the harness fingerprint, not assumed.
"""
import os
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
import sympy as sp
from lib.equilibrium.full_search.routes.flint_spike.flint_solver import FlintMixingSolver

_JL = None          # cached (juliacall Main, solve_fn)

_JULIA_SRC = r"""
module FullmixSolve
using AlgebraicSolving
# polys given as parallel arrays: coeffs[i] (integers, denominators pre-cleared) and
# exps[i] (each a flat length-(n*t) row of exponents). Returns (dim, solutions) where
# solutions (only for dim==0) is a Vector of Vector of [num,den] BigInt pairs.
function solve(nv::Int, varnames, coeffs, exps)
    R, vars = polynomial_ring(QQ, collect(String, varnames))
    gens = typeof(zero(R))[]
    for k in eachindex(coeffs)
        cfs = coeffs[k]; ex = exps[k]
        p = zero(R)
        for ti in eachindex(cfs)
            term = R(BigInt(cfs[ti]))
            base = (ti - 1) * nv
            for j in 1:nv
                e = Int(ex[base + j])
                if e > 0
                    term *= vars[j]^e
                end
            end
            p += term
        end
        push!(gens, p)
    end
    I = Ideal(gens)
    # real_solutions throws iff dim>=1 (posdim). For zerodim it returns the real points (as
    # rational APPROXIMATIONS -- not exact, so they must not be used as witnesses). We use it
    # only to (a) detect posdim and (b) count the real roots. Exact witnesses come from
    # rational_solutions(I), which returns the EXACT rational points (Stern-Brocot recovery is
    # not needed and the old numerator/denominator-of-approximation path was a recovery bug).
    local reals
    try
        reals = real_solutions(I)
    catch e
        if occursin("Dimension", string(e))
            return (1, Vector{Vector{Vector{BigInt}}}())     # posdim -> deferred
        end
        rethrow(e)
    end
    if length(reals) == 0
        return (-1, Vector{Vector{Vector{BigInt}}}())        # no real roots -> infeasible
    end
    rats = rational_solutions(I)                             # EXACT rational real points
    out = [[[BigInt(numerator(x)), BigInt(denominator(x))] for x in sol] for sol in rats]
    # complete (code 0): every real root is rational, so failing to verify any => infeasible.
    # incomplete (code 2): some real roots are irrational (not in `out`) => caller must defer
    # rather than declare infeasible until the exact-algebraic verifier (Fix #2) handles them.
    return (length(rats) == length(reals) ? 0 : 2, out)
end
end
"""


def _jl():
    global _JL
    if _JL is None:
        from juliacall import Main as jl
        jl.seval(_JULIA_SRC)
        _JL = (jl, jl.seval("FullmixSolve.solve"))
    return _JL


class FlintJuliaSolver(FlintMixingSolver):

    def _msolve_flint(self, nv, eqs, names):
        # nonzero-constant eq -> inconsistent -> empty (matches the CLI's constant shortcut;
        # cheap, avoids a full solve on trivially-infeasible degree>=2 systems)
        if any(e.is_constant() and not e.is_zero() for e in eqs):
            return ("empty", [])
        # degree<=1: exact FLINT linear algebra (fastest, parity-proven)
        if all(e.total_degree() <= 1 for e in eqs):
            return self._solve_linear(nv, eqs, names)
        # degree>=2: in-process AlgebraicSolving (same msolve engine)
        jl, solve_fn = _jl()
        coeffs = []; exps = []
        for e in eqs:
            monoms = e.monoms(); cs = e.coeffs()
            den = 1
            for c in cs:
                den = den * int(c.q) // self._gcd(den, int(c.q))
            coeffs.append([int(c.p) * (den // int(c.q)) for c in cs])
            flat = []
            for m in monoms:
                flat.extend(int(v) for v in m)
            exps.append(flat)
        varnames = [f"x{i}" for i in range(nv)]
        d, sols = solve_fn(nv, varnames, coeffs, exps)
        d = int(d)
        if d == -1:
            return ("empty", [])
        if d == 1:
            return ("posdim", [])
        cands = []
        for sol in sols:                                   # each sol: nv [num,den] pairs
            cands.append({names[j]: sp.Rational(int(sol[j][0]), int(sol[j][1]))
                          for j in range(nv)})
        # d==0: every real root is rational (cands complete) -> none verifying => infeasible.
        # d==2: irrational real roots remain (not in cands) -> caller must defer, not declare
        # infeasible, until the exact-algebraic verifier handles them.
        return ("zerodim" if d == 0 else "zerodim_incomplete", cands)
