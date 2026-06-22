#!/usr/bin/env python3
"""FLINT spike: replace sympy in poly_build (and the msolve serialization) with
exact FLINT multivariate polynomials (fmpq_mpoly), keeping the decision and
verify_witness BIT-IDENTICAL to the baseline.

`FlintMixingSolver` subclasses the baseline and overrides only `solve_profile`:
  * NUM[s][k] = V[s,k]*det built as fmpq_mpoly via the same exact Mobius interpolation
    (no sympy), straight from the FLINT vertex evaluations;
  * the EV*det products and the variety equations are formed with fmpq_mpoly arithmetic;
  * the system is serialized to msolve directly from polynomial terms (LCM denominator
    clearing), no sympy Poly.clear_denoms.
DET is not built: it is dead in the decision path (verify_witness recomputes det), so
omitting it is a sound, parity-preserving win.

Everything else (_eval_vertex, _vars_for_profile, _parse_msolve, _simplest_rational,
verify_witness, the deferred/posdim/zerodim decision logic, MSTAT, _T stage names) is the
unchanged baseline machinery, so a stage-by-stage A/B and the parity fingerprint are valid.
"""
import os, tempfile, subprocess
import flint
import sympy as sp
from lib.equilibrium.full_search.full_mixing_sweep import FullMixingSolver, MSTAT, _T


class FlintMixingSolver(FullMixingSolver):

    @staticmethod
    def _mobius_mpoly(F, nv, ctx, gens):
        """Exact multilinear interpolant of F (mask->fmpq) as an fmpq_mpoly over `ctx`."""
        g = [F[m] for m in range(1 << nv)]
        for j in range(nv):
            bit = 1 << j
            for m in range(1 << nv):
                if m & bit:
                    g[m] = g[m] - g[m ^ bit]
        d = {}
        for m in range(1 << nv):
            if g[m] != 0:
                exp = tuple(1 if (m >> j) & 1 else 0 for j in range(nv))
                d[exp] = g[m]
        return ctx.from_dict(d)

    def solve_profile(self, tiers, profile):
        n, S, fg = self.n, self.S, self.fg
        with _T("setup"):
            acc, wk = self._vars_for_profile(tiers, profile)
            nv = len(acc) + len(wk)
            if self.max_nv is not None and nv > self.max_nv:
                return ("deferred", nv)
            # original-name list (a_/w_) in the canonical acc-then-wk order: keys for
            # _parse_msolve candidates and for verify_witness.
            names = ([f"a_{k[0]}_{k[1]}_{k[2]}" for k in acc] +
                     [f"w_{k[0]}_{k[1]}_{k[2]}" for k in wk])
            acc_index = {k: i for i, k in enumerate(acc)}      # acc key -> gen index
            if nv == 0:                                        # pure: no free vars
                MSTAT["nv0"] += 1
                return ("feasible", {}) if self.verify_witness(tiers, profile, {}) \
                    else ("infeasible", None)
            ctx = flint.fmpq_mpoly_ctx.get([f"x{i}" for i in range(nv)], flint.Ordering.lex)
            gens = ctx.gens()
            ONE = ctx.constant(flint.fmpq(1)); ZERO = ctx.constant(flint.fmpq(0))
            M = 1 << nv

        with _T("vertex_eval"):
            Fnum = [[{} for _ in range(n)] for _ in range(S)]
            for mask in range(M):
                bits = [(mask >> j) & 1 for j in range(nv)]
                V, det = self._eval_vertex(tiers, profile, acc, wk, bits)
                for s in range(S):
                    for k in range(n):
                        Fnum[s][k][mask] = V[s, k] * det

        with _T("poly_build"):
            NUM = [[self._mobius_mpoly(Fnum[s][k], nv, ctx, gens) for k in range(n)]
                   for s in range(S)]

            def pass_poly(ii, xi, yi, supported):
                if yi == xi:
                    return ONE
                p = ONE
                for ki in fg.committee[(ii, xi, yi)]:
                    ty, tx = int(tiers[ki][yi]), int(tiers[ki][xi])
                    if ty < tx:
                        continue                       # forced accept -> factor 1
                    if ty > tx:
                        return ZERO                    # forced reject -> 0
                    if not supported:
                        return ZERO
                    p = p * gens[acc_index[(ki, xi, yi)]]
                return p

            def EVdet(ii, xi, yi, supported):
                if yi == xi:
                    return NUM[xi][ii]
                pa = pass_poly(ii, xi, yi, supported)
                return pa * (NUM[yi][ii] - NUM[xi][ii]) + NUM[xi][ii]

            eqs = []
            for ki in range(n):                        # acceptance ties (multilinear)
                by = {}
                for si in range(S):
                    by.setdefault(int(tiers[ki][si]), []).append(si)
                for grp in by.values():
                    for si in grp[1:]:
                        eqs.append(NUM[si][ki] - NUM[grp[0]][ki])
            for (ii, xi), supp in profile.items():     # proposal indifference across support
                if len(supp) >= 2:
                    e0 = EVdet(ii, xi, supp[0], True)
                    for yj in supp[1:]:
                        eqs.append(e0 - EVdet(ii, xi, yj, True))
            eqs = [e for e in eqs if not e.is_zero()]

        if not eqs:
            MSTAT["posdim"] += 1
            return ("deferred", nv)
        with _T("msolve"):
            status, sols = self._msolve_flint(nv, eqs, names)
        MSTAT[status] += 1
        if status == "empty":
            return ("infeasible", None)
        if status in ("posdim", "timeout"):
            return ("deferred", nv)
        with _T("decide"):
            for cand in sols:                              # exact rational candidates
                if self.verify_witness(tiers, profile, cand):
                    return ("feasible", cand)
        # zerodim (complete): all real roots are rational and none is an equilibrium -> no real
        # solution is an equilibrium -> infeasible. zerodim_incomplete: irrational real roots
        # remain unchecked -> defer until the exact-algebraic verifier handles them.
        return ("deferred", 0) if status == "zerodim_incomplete" else ("infeasible", None)

    def _solve_linear(self, nv, eqs, names):
        """Exact in-process solve when ALL eqs are degree<=1. fmpq_mat rank analysis gives
        the SAME empty/posdim/zerodim classification msolve would on a linear system, so this
        is parity-preserving and avoids the subprocess. eqs are nonzero fmpq_mpoly."""
        rows = []; bs = []
        for e in eqs:
            coef = [0] * nv; const = flint.fmpq(0)
            for mono, c in zip(e.monoms(), e.coeffs()):
                nz = [j for j, ev in enumerate(mono) if ev]
                if not nz:
                    const = c
                else:                                   # degree<=1 -> exactly one var, exp 1
                    coef[nz[0]] = c
            rows.append(coef); bs.append(-const)
        A = flint.fmpq_mat(rows)
        aug = flint.fmpq_mat([rows[i] + [bs[i]] for i in range(len(rows))])
        rA = A.rank(); rAug = aug.rank()
        if rAug > rA:
            return ("empty", [])
        if rA < nv:
            return ("posdim", [])
        R, _ = aug.rref()                               # rank==nv: pivots in all nv columns
        cand = {names[j]: sp.Rational(int(R[j, nv].p), int(R[j, nv].q)) for j in range(nv)}
        return ("zerodim", [cand])

    def _msolve_flint(self, nv, eqs, names):
        """Serialize fmpq_mpoly equations to msolve directly (no sympy), then reuse the
        baseline subprocess call + _parse_msolve. msolve vars are x0..x{nv-1}.
        Fast path: degree<=1 systems are solved in-process (no subprocess)."""
        if all(e.total_degree() <= 1 for e in eqs):
            return self._solve_linear(nv, eqs, names)
        lines = []
        for e in eqs:
            monoms = e.monoms(); coeffs = e.coeffs()
            if not monoms:
                continue                               # zero (already filtered, defensive)
            if len(monoms) == 1 and all(v == 0 for v in monoms[0]):
                return ("empty", [])                   # nonzero constant: no solutions
            den = 1
            for c in coeffs:
                den = den * c.q // self._gcd(den, c.q)
            terms = []
            for exp, c in zip(monoms, coeffs):
                ic = int(c.p) * (den // int(c.q))      # integer coefficient
                mono = "*".join(f"x{j}^{ev}" if ev > 1 else f"x{j}"
                                for j, ev in enumerate(exp) if ev)
                terms.append(f"{ic}*{mono}" if mono else f"{ic}")
            lines.append("+".join(terms).replace("+-", "-"))
        if not lines:
            return ("posdim", [])
        body = ",".join(f"x{i}" for i in range(nv)) + "\n0\n" + ",\n".join(lines) + "\n"
        with tempfile.TemporaryDirectory() as d:
            inf = os.path.join(d, "in.ms"); outf = os.path.join(d, "out.ms")
            with open(inf, "w") as fh:
                fh.write(body)
            try:
                subprocess.run(["msolve", "-f", inf, "-o", outf],
                               timeout=self.msolve_timeout, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return ("timeout", [])
            txt = open(outf).read().strip()
        return self._parse_msolve(txt, names)

    @staticmethod
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
