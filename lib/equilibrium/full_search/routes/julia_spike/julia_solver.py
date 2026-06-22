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
    # ONE solver pass: real_solutions throws iff dim>=1 (posdim). Otherwise it returns the
    # real rational points (possibly empty). empty-ideal and zerodim-with-no-real-point both
    # come back as [] and both map to "infeasible" downstream, so we don't need dimension().
    local sols
    try
        sols = real_solutions(I)
    catch e
        if occursin("Dimension", string(e))
            return (1, Vector{Vector{Vector{BigInt}}}())     # posdim -> deferred
        end
        rethrow(e)
    end
    out = [[[BigInt(numerator(x)), BigInt(denominator(x))] for x in sol] for sol in sols]
    return (length(out) == 0 ? -1 : 0, out)                  # [] -> infeasible
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
        if d >= 1:
            return ("posdim", [])
        cands = []
        for sol in sols:                                   # each sol: nv [num,den] pairs
            cands.append({names[j]: sp.Rational(int(sol[j][0]), int(sol[j][1]))
                          for j in range(nv)})
        return ("zerodim", cands)
