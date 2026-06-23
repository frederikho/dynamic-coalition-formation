module PosdimSolve
# Exact algebraic feasibility on a ZERO-DIMENSIONAL system, over the real algebraic numbers.
# Given equalities eqs=0, strict>0 and nonneg>=0 sign conditions, decide whether any REAL
# solution of eqs=0 satisfies all sign conditions -- exactly, via QQBar (no tolerance). Every
# reconstructed point is self-verified on-variety in exact arithmetic, so a "feasible" verdict
# is an exact certificate; an "infeasible" verdict means no real root of eqs=0 is feasible.
using AlgebraicSolving
const NM = AlgebraicSolving.Nemo

# polynomials are marshalled as parallel arrays: coeffs[k] (integers, denominators cleared) and
# exps[k] (flat length nv*t row of exponents), identical to the julia_solver convention.
function _build(R, vars, nv, cfs, ex)
    p = zero(R)
    for ti in eachindex(cfs)
        term = R(NM.ZZ(BigInt(cfs[ti]))); base = (ti - 1) * nv
        for j in 1:nv
            e = Int(ex[base + j]); e > 0 && (term *= vars[j]^e)
        end
        p += term
    end
    return p
end

# return: (status, nreal, feas_idx, witness) where status: 0 zerodim-decided, 1 posdim (genericity
# failure for this system -> caller perturbs), -1 empty (no complex solutions). feas_idx is the
# 1-based index of a feasible real solution or 0 if none. witness is that point's coords as
# [num,den] BigInt pairs when rational, else [] (irrational witness flagged by isrational=false).
# all size-k subsets of 1:n (ascending), as Vectors -- avoids a Combinatorics.jl dependency.
function _combs(n::Int, k::Int)
    res = Vector{Vector{Int}}()
    k < 0 && return res
    k == 0 && return [Int[]]
    idx = collect(1:k)
    while true
        push!(res, copy(idx))
        i = k
        while i >= 1 && idx[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        idx[i] += 1
        for j in i+1:k
            idx[j] = idx[j-1] + 1
        end
    end
    return res
end

# Build the distance-critical system of V={eqs} (codim rho) IN NEMO and decide feasibility, all in
# one in-process call: Jacobian via Nemo derivative, row-0-including (rho+1)-minors via Nemo det,
# then solve + exact sign-check. Replaces the slow sympy minor-determinant path. probe is the
# generic point as parallel numerator/denominator integer arrays (length nv).
function crit_decide(nv, eqc, eqe, pnum, pden, rho, strc, stre, nonc, none)
    R, vars = polynomial_ring(QQ, ["x$i" for i in 0:nv-1])
    eqs = [_build(R, vars, nv, eqc[k], eqe[k]) for k in eachindex(eqc)]
    grad = [2 * (vars[j] - QQ(BigInt(pnum[j]), BigInt(pden[j]))) for j in 1:nv]   # grad ||x-p||^2
    jac = [[NM.derivative(g, vars[j]) for j in 1:nv] for g in eqs]
    c = length(jac)
    r = rho + 1
    csys = copy(eqs)
    if 1 <= r <= nv && r - 1 <= c
        for jr in _combs(c, r - 1)
            rows = Vector{Vector{typeof(zero(R))}}()
            push!(rows, grad)
            for i in jr; push!(rows, jac[i]); end
            for cs in _combs(nv, r)
                M = NM.matrix(R, [rows[a][cs[b]] for a in 1:r, b in 1:r])
                m = NM.det(M)
                iszero(m) || push!(csys, m)
            end
        end
    end
    strs = [_build(R, vars, nv, strc[k], stre[k]) for k in eachindex(strc)]
    nons = [_build(R, vars, nv, nonc[k], none[k]) for k in eachindex(nonc)]
    return _solve_and_check(Ideal(csys), eqs, R, vars, nv, strs, nons)
end

function vdim(nv, eqc, eqe)
    R, vars = polynomial_ring(QQ, ["x$i" for i in 0:nv-1])
    gens = [_build(R, vars, nv, eqc[k], eqe[k]) for k in eachindex(eqc)]
    return dimension(Ideal(gens))
end

function decide(nv, eqc, eqe, strc, stre, nonc, none)
    R, vars = polynomial_ring(QQ, ["x$i" for i in 0:nv-1])
    gens = [_build(R, vars, nv, eqc[k], eqe[k]) for k in eachindex(eqc)]
    strs = [_build(R, vars, nv, strc[k], stre[k]) for k in eachindex(strc)]
    nons = [_build(R, vars, nv, nonc[k], none[k]) for k in eachindex(nonc)]
    return _solve_and_check(Ideal(gens), gens, R, vars, nv, strs, nons)
end

# Like decide, but the candidate points come from the CRITICAL system (csc) optionally SATURATED
# by the singular-locus generators (satc): solve  (<csys> : <singular minors>^inf)  to strip the
# singular components and recover the isolated smooth critical points. Recovered points are still
# self-verified against the VARIETY generators (eqc), so feasibility remains an exact certificate.
function decide_sat(nv, eqc, eqe, csc, cse, satc, sate, strc, stre, nonc, none)
    R, vars = polynomial_ring(QQ, ["x$i" for i in 0:nv-1])
    eqgens = [_build(R, vars, nv, eqc[k], eqe[k]) for k in eachindex(eqc)]
    csgens = [_build(R, vars, nv, csc[k], cse[k]) for k in eachindex(csc)]
    if length(satc) > 0
        satg = [_build(R, vars, nv, satc[k], sate[k]) for k in eachindex(satc)]
        csgens = AlgebraicSolving.saturate(csgens, satg)   # <csys> : <singular locus>^inf
    end
    strs = [_build(R, vars, nv, strc[k], stre[k]) for k in eachindex(strc)]
    nons = [_build(R, vars, nv, nonc[k], none[k]) for k in eachindex(nonc)]
    return _solve_and_check(Ideal(csgens), eqgens, R, vars, nv, strs, nons)
end

# Solve a zero-dimensional ideal exactly, reconstruct each real solution over QQBar, self-verify
# it lies on `verify_gens`=0, and exact-sign-check the conditions. status: 0 decided, 1 posdim
# (not zero-dimensional -> caller perturbs/recurses), -1 empty.
function _solve_and_check(I, verify_gens, R, vars, nv, strs, nons)
    d = dimension(I)
    d >= 1 && return (1, 0, 0, Any[], true)
    d == -1 && return (-1, 0, 0, Any[], true)
    rp = rational_parametrization(I)
    Rt, t = NM.polynomial_ring(NM.QQ, "t")
    realroots = [r for r in NM.roots(NM.QQBar, evaluate(rp.elim, t)) if NM.isreal(r)]
    varnames = [string(v) for v in rp.vars]
    ournames = ["x$i" for i in 0:nv-1]
    feas_idx = 0; witness = Any[]; wit_rat = true
    for (ri, T) in enumerate(realroots)
        denomT = evaluate(rp.denom, T)
        valmap = Dict{String,Any}()
        for i in eachindex(rp.param)
            valmap[varnames[i]] = evaluate(rp.param[i], T) / denomT
        end
        valmap[varnames[end]] = T
        pt = [valmap[nm] for nm in ournames]
        for g in verify_gens
            evaluate(g, pt) == 0 || error("reconstruction off-variety (RUR decode bug)")
        end
        feas = all(evaluate(s, pt) > 0 for s in strs) && all(evaluate(n, pt) >= 0 for n in nons)
        if feas && feas_idx == 0
            feas_idx = ri
            for v in pt
                if NM.is_rational(v)
                    q = NM.QQFieldElem(v)
                    push!(witness, [BigInt(NM.numerator(q)), BigInt(NM.denominator(q))])
                else
                    wit_rat = false
                end
            end
        end
    end
    return (0, length(realroots), feas_idx, witness, wit_rat)
end

end # module
