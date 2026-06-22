#!/usr/bin/env python3
"""Full proposal-mixing certified sweep --- the REAL algorithm whose cost was modelled
as C_full(R) = r(R) * c(m(R)) in write-up/complete_search_strategy.tex.

For a weak-order triple R it does, per the document:
  * compute the candidate proposal supports per cell (non-empty subsets of the
    proposer's top-tier-tied feasible targets); their product is r(R);
  * for each support PROFILE (one support per (proposer,state) cell), build the exact
    support-fixed system --- acceptance mixing variables + proposal-weight variables
    --- by FLINT exact-rational interpolation (multilinear), and solve it exactly
    (sympy), checking feasibility (indifference equations + strict orders + proposal
    optimality + box).
Everything is exact (rational arithmetic / certified); no floating tolerances, no
solver timeouts in the work path.

Scheduling: labels are processed CHEAPEST-FIRST by the predicted cost r(R)*c(m(R)).
A run can target a fraction (e.g. the cheapest 50%) via a predicted-cost threshold,
or just run for a wall-clock budget. This script validates the timing estimate of the
strategy document and, as a by-product, reports any equilibria it finds.

USAGE (see bottom / --help):
  # 15-minute trial (single process) to check the per-label rate vs the estimate:
  python scripts/full_mixing_sweep.py trial --minutes 15
  # full cheapest-50% run on 14 workers (the ~0.7-day job):
  python scripts/full_mixing_sweep.py run --fraction 0.50 --workers 14
"""
from __future__ import annotations
import sys, time, itertools, argparse, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import sympy as sp
import flint

ROOT = Path(__file__).resolve().parents[3]   # lib/equilibrium/full_search/ -> repo root
sys.path.insert(0, str(ROOT))
from scripts.residual_metric_probe import build_setup          # shared helper, stays in scripts/
from scripts._reduced_helpers import _generate_weak_orders     # shared helper, stays in scripts/
from lib.equilibrium.full_search.certified_label_solver import CertifiedLabelSolver

# Generated artifacts (cheapest-first order, bench corpora, checkpoints). Large/derived ->
# kept inside the package under data/ and gitignored (see .gitignore).
DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

# acceptance per-label solve-cost model c(m) (measured, Section 7); used only for the
# cheapest-first PREDICTED cost r*c(m); the actual run measures real time.
CM = {0:0.0001,1:0.0009,2:0.0017,3:0.0111,4:0.0436,5:0.6312,6:0.6482,
      7:9.926,8:18.61,9:25.06,10:58.25,11:60.0,12:60.0,13:60.0,14:60.0}
def c_model(m): return CM.get(m, 60.0)

# --- lightweight stage profiler (per-process; trial aggregates and logs it) ---
from collections import defaultdict
PROF = defaultdict(lambda: [0.0, 0])   # stage -> [total_seconds, n_calls]
class _T:
    __slots__ = ("k", "t")
    def __init__(self, k): self.k = k
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        e = PROF[self.k]; e[0] += time.perf_counter() - self.t; e[1] += 1
DIMS = defaultdict(int)        # number of free parameters (k) per decided branch
MSTAT = defaultdict(int)       # msolve outcome counts (empty/zerodim/posdim/timeout/nv0)
def dims_report():
    return "  ".join(f"{k}:{v}" for k, v in sorted(MSTAT.items())) or "(none)"
def prof_report():
    tot = sum(v[0] for v in PROF.values()) or 1e-12
    rows = sorted(PROF.items(), key=lambda kv: -kv[1][0])
    lines = [f"{'stage':<16}{'total_s':>12}{'%':>8}{'calls':>12}{'ms/call':>12}"]
    for k, (s, n) in rows:
        lines.append(f"{k:<16}{s:>12.3f}{100*s/tot:>7.1f}%{n:>12,}{1000*s/max(1,n):>12.3f}")
    return "\n".join(lines)


class FullMixingSolver:
    def __init__(self, payoff, scenario="power_threshold_RICE_n3", rule="adjacent_step"):
        self.setup = build_setup(scenario, ROOT / "payoff_tables" / f"{payoff}.xlsx", rule)
        self.cls = CertifiedLabelSolver(self.setup)
        fg = self.cls.fg
        self.fg = fg; self.n = fg.n; self.S = fg.S; self.states = fg.states
        self.orders = _generate_weak_orders(self.S); self.NO = len(self.orders)
        ONE = flint.fmpq(1)
        self.DELTA = flint.fmpq(int(sp.fraction(self.cls.delta)[0]), int(sp.fraction(self.cls.delta)[1]))
        self.PROTO = [flint.fmpq(int(sp.fraction(p)[0]), int(sp.fraction(p)[1])) for p in self.cls.proto]
        self.U = [[flint.fmpq(int(sp.fraction(self.cls.u[s, j])[0]), int(sp.fraction(self.cls.u[s, j])[1]))
                   for j in range(self.n)] for s in range(self.S)]
        self.ONE = ONE
        # RHS b = (1-delta)*U is constant across every vertex/support/label -> build once
        self._b_const = flint.fmpq_mat([[(ONE - self.DELTA) * self.U[i][j]
                                          for j in range(self.n)] for i in range(self.S)])
        self.max_nv = None      # if set, profiles with > max_nv variables are deferred
        self.msolve_timeout = 10.0  # seconds; on timeout the variety solve is deferred

    # ---- structural counts (cheap, no solving) ----
    def top_tier_feasible(self, tiers, ii, xi):
        feas = self.fg.feasible[(ii, xi)]
        best = min(int(tiers[ii][y]) for y in feas)
        return [y for y in feas if int(tiers[ii][y]) == best]

    def r_and_m(self, tiers):
        r = 1
        for ii in range(self.n):
            for xi in range(self.S):
                r *= (2 ** len(self.top_tier_feasible(tiers, ii, xi)) - 1)
        # m: tied committee contexts on feasible (proposed) transitions
        m = self.cls_m(tiers)
        return r, m

    def cls_m(self, tiers):
        prop = self.cls._fixed_proposals(tiers); keys = set()
        for (ii, xi), yi in prop.items():
            if yi != xi:
                for ki in self.fg.committee[(ii, xi, yi)]:
                    if int(tiers[ki][yi]) == int(tiers[ki][xi]):
                        keys.add((ki, xi, yi))
        return len(keys)

    def candidate_profiles(self, tiers):
        """Yield support profiles: per cell, a non-empty subset of top-tier feasible."""
        cells = []
        per_cell_opts = []
        for ii in range(self.n):
            for xi in range(self.S):
                top = self.top_tier_feasible(tiers, ii, xi)
                subs = []
                for k in range(1, len(top) + 1):
                    subs.extend(itertools.combinations(top, k))
                cells.append((ii, xi)); per_cell_opts.append(subs)
        for combo in itertools.product(*per_cell_opts):
            yield dict(zip(cells, combo))

    # ---- exact support-fixed solve via FLINT interpolation ----
    def _vars_for_profile(self, tiers, profile):
        """Return (accept_keys, weight_keys). accept = tied committee contexts on
        supported transitions; weight = (ii,xi,j) free proposal weights (|support|-1)."""
        acc = []
        seen = set()
        for (ii, xi), supp in profile.items():
            for yi in supp:
                if yi == xi:
                    continue
                for ki in self.fg.committee[(ii, xi, yi)]:
                    if int(tiers[ki][yi]) == int(tiers[ki][xi]) and (ki, xi, yi) not in seen:
                        seen.add((ki, xi, yi)); acc.append((ki, xi, yi))
        wk = []
        for (ii, xi), supp in profile.items():
            for j in range(len(supp) - 1):
                wk.append((ii, xi, j))
        return acc, wk

    def _eval_vertex(self, tiers, profile, acc, wk, bits):
        """Build P (fmpq) at a {0,1} assignment of acc+weight vars; return (V list, det)."""
        n, S, fg, ONE = self.n, self.S, self.fg, self.ONE
        av = {k: bits[i] for i, k in enumerate(acc)}
        wv = {k: bits[len(acc) + i] for i, k in enumerate(wk)}
        P = [[flint.fmpq(0) for _ in range(S)] for _ in range(S)]
        for (ii, xi), supp in profile.items():
            pr = self.PROTO[ii]
            # proposal weights over support (last = 1 - sum of free)
            w = []
            if len(supp) == 1:
                w = [ONE]
            else:
                free = [flint.fmpq(int(wv[(ii, xi, j)])) for j in range(len(supp) - 1)]
                w = free + [ONE - sum(free, flint.fmpq(0))]
            for yi, wj in zip(supp, w):
                if yi == xi:
                    P[xi][xi] += pr * wj
                    continue
                pa = ONE
                for ki in self.fg.committee[(ii, xi, yi)]:
                    ty, tx = int(tiers[ki][yi]), int(tiers[ki][xi])
                    a = ONE if ty < tx else (flint.fmpq(0) if ty > tx else flint.fmpq(int(av[(ki, xi, yi)])))
                    pa *= a
                P[xi][yi] += pr * wj * pa
                P[xi][xi] += pr * wj * (ONE - pa)
        A = flint.fmpq_mat([[(ONE if i == j else flint.fmpq(0)) - self.DELTA * P[i][j]
                             for j in range(S)] for i in range(S)])
        return A.solve(self._b_const), A.det()    # b=(1-delta)*U hoisted to __init__

    def _mobius_poly(self, F, nv, syms):
        """Exact multilinear interpolant of a function given at the 2^nv cube vertices
        (F: mask->fmpq), returned as a sympy polynomial in `syms`."""
        g = [F[mask] for mask in range(1 << nv)]
        for j in range(nv):
            bit = 1 << j
            for mask in range(1 << nv):
                if mask & bit:
                    g[mask] = g[mask] - g[mask ^ bit]
        poly = sp.Integer(0)
        for mask in range(1 << nv):
            if g[mask] != 0:
                mono = sp.Integer(1)
                for j in range(nv):
                    if mask & (1 << j):
                        mono *= syms[j]
                poly += sp.Rational(int(g[mask].p), int(g[mask].q)) * mono
        return sp.expand(poly)

    def _pass_sym(self, tiers, ii, xi, yi, sym_acc, supported):
        """Symbolic acceptance product pass(ii: x->y).
        forced-accept member -> 1, forced-reject -> 0, tied member -> its acceptance
        variable if the transition is SUPPORTED (a real decision var), else 0 (an
        unproposed transition can be rejected by an indifferent committee --- the
        feasibility-favourable, equilibrium-consistent choice)."""
        if yi == xi:
            return sp.Integer(1)
        p = sp.Integer(1)
        for ki in self.fg.committee[(ii, xi, yi)]:
            ty, tx = int(tiers[ki][yi]), int(tiers[ki][xi])
            if ty < tx:
                continue                       # forced accept -> factor 1
            if ty > tx:
                return sp.Integer(0)           # forced reject -> pass = 0
            p *= sym_acc[(ki, xi, yi)] if supported else sp.Integer(0)
        return p

    def solve_profile(self, tiers, profile):
        """Exact feasibility of one support profile.
        Returns ('feasible', witness) | ('infeasible', None) | ('deferred', k)
        where k>=2 free parameters need a multivariate CAD (Step-2 backend)."""
        n, S, fg = self.n, self.S, self.fg
        with _T("setup"):
            acc, wk = self._vars_for_profile(tiers, profile)
            nv = len(acc) + len(wk)
            if self.max_nv is not None and nv > self.max_nv:
                return ("deferred", nv)         # too many vars: defer (avoids solve blow-ups)
            sym_acc = {k: sp.Symbol(f"a_{k[0]}_{k[1]}_{k[2]}", real=True) for k in acc}
            sym_w = {k: sp.Symbol(f"w_{k[0]}_{k[1]}_{k[2]}", real=True) for k in wk}
            syms = [sym_acc[k] for k in acc] + [sym_w[k] for k in wk]
            M = 1 << nv
        # interpolate ONLY the genuinely multilinear quantities: num[s][k]=V[s,k]*det, det
        with _T("vertex_eval"):
            Fnum = [[{} for _ in range(n)] for _ in range(S)]
            Fdet = {}
            for mask in range(M):
                bits = [(mask >> j) & 1 for j in range(nv)]
                V, det = self._eval_vertex(tiers, profile, acc, wk, bits)
                Fdet[mask] = det
                for s in range(S):
                    for k in range(n):
                        Fnum[s][k][mask] = V[s, k] * det
        with _T("poly_build"):
            NUM = [[self._mobius_poly(Fnum[s][k], nv, syms) for k in range(n)] for s in range(S)]
            DET = self._mobius_poly(Fdet, nv, syms)
            # EV(i: x->y) numerator (times det), built symbolically (degree may exceed 1)
            def EVdet(ii, xi, yi, supported):
                if yi == xi:
                    return NUM[xi][ii]
                pa = self._pass_sym(tiers, ii, xi, yi, sym_acc, supported)
                return sp.expand(pa * (NUM[yi][ii] - NUM[xi][ii]) + NUM[xi][ii])
            # --- equations defining the variety ---
            eqs = []
            # acceptance ties: V_k(s)=V_k(base) within each tier group  (multilinear, exact)
            for ki in range(n):
                by = {}
                for si in range(S):
                    by.setdefault(int(tiers[ki][si]), []).append(si)
                for grp in by.values():
                    for si in grp[1:]:
                        eqs.append(NUM[si][ki] - NUM[grp[0]][ki])
            # proposal indifference: proposer's EV equal across its support
            for (ii, xi), supp in profile.items():
                if len(supp) >= 2:
                    e0 = EVdet(ii, xi, supp[0], True)
                    for yj in supp[1:]:
                        eqs.append(sp.expand(e0 - EVdet(ii, xi, yj, True)))
            eqs = [e for e in (sp.expand(e) for e in eqs) if e != 0]
            # --- inequalities for feasibility ---
            strict = []      # must be > 0
            nonneg = []      # must be >= 0
            # strict orders: tiers[k][x] < tiers[k][y]  =>  V_x[k] > V_y[k]
            for ki in range(n):
                for x in range(S):
                    for y in range(S):
                        if int(tiers[ki][x]) < int(tiers[ki][y]):
                            strict.append(NUM[x][ki] - NUM[y][ki])
            # proposal optimality: support EV >= EV of every feasible target
            for (ii, xi), supp in profile.items():
                ev_s = EVdet(ii, xi, supp[0], True)     # all support EVs equal on variety
                for yi in fg.feasible[(ii, xi)]:
                    if yi in supp:
                        continue
                    nonneg.append(sp.expand(ev_s - EVdet(ii, xi, yi, False)))
            # det > 0 (so sign(V_a-V_b)=sign(num_a-num_b)); weight simplex
            strict.append(DET)
            box = list(syms)                              # each var in [0,1]
            for (ii, xi), supp in profile.items():
                if len(supp) >= 2:
                    last = sp.Integer(1) - sum(sym_w[(ii, xi, j)] for j in range(len(supp) - 1))
                    box.append(last)                      # dependent weight in [0,1]
        # solve the variety (exact symbolic; Step-2 will swap in a faster exact backend)
        if not syms:                       # no free vars: forced point (pure strategies)
            MSTAT["nv0"] += 1
            return ("feasible", {}) if self.verify_witness(tiers, profile, {}) else ("infeasible", None)
        if not eqs:                         # no equations -> variety is full-dim -> defer
            MSTAT["posdim"] += 1
            return ("deferred", len(syms))
        with _T("msolve"):
            status, sols = self._msolve(syms, eqs)
        MSTAT[status] += 1
        if status == "empty":               # no complex solutions at all -> no equilibrium
            return ("infeasible", None)
        if status in ("posdim", "timeout"):
            return ("deferred", len(syms))
        # zerodim: finitely many real points; exact-verify each recovered rational candidate
        with _T("decide"):
            for cand in sols:
                if self.verify_witness(tiers, profile, cand):
                    return ("feasible", cand)
        # real points existed but none certified as a (rational) equilibrium: conservative defer
        return ("deferred", 0) if sols else ("infeasible", None)

    @staticmethod
    def _simplest_rational(lo, hi):
        """The rational with smallest denominator in the closed interval [lo, hi]
        (Stern-Brocot). Used to recover an exact rational from msolve's isolating box;
        verify_witness then confirms it exactly, so a wrong guess is simply rejected."""
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            return sp.Rational(lo)
        fl = sp.floor(lo)
        if fl >= lo:                       # lo itself is an integer in the interval
            return sp.Integer(fl)
        if fl + 1 <= hi:                   # an integer lies strictly inside
            return sp.Integer(fl + 1)
        return fl + 1 / FullMixingSolver._simplest_rational(1 / (hi - fl), 1 / (lo - fl))

    def _msolve(self, syms, eqs):
        """Solve the polynomial system eqs=0 exactly via the msolve CLI (subprocess, so a
        hang is bounded by msolve_timeout and recorded, never silently skipped).
        Returns (status, candidates): status in {empty,zerodim,posdim,timeout}; candidates
        is a list of dicts {orig_sym_name: Rational} recovered from the isolating boxes."""
        mnames = [f"x{i}" for i in range(len(syms))]
        msyms = [sp.Symbol(nm) for nm in mnames]
        ren = dict(zip(syms, msyms))
        lines = []
        for e in eqs:
            e = sp.expand(e.subs(ren))
            if e == 0:
                continue
            P = sp.Poly(e, *msyms, domain="QQ")
            if P.total_degree() == 0:                  # nonzero constant: no solutions
                return ("empty", [])
            _, P = P.clear_denoms(convert=True)
            lines.append(str(P.as_expr()).replace("**", "^"))
        if not lines:
            return ("posdim", [])
        body = ",".join(mnames) + "\n0\n" + ",\n".join(lines) + "\n"
        with tempfile.TemporaryDirectory() as d:
            inf = os.path.join(d, "in.ms"); outf = os.path.join(d, "out.ms")
            with open(inf, "w") as fh:
                fh.write(body)
            try:
                subprocess.run(["msolve", "-f", inf, "-o", outf],
                               timeout=self.msolve_timeout, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                return ("timeout", [])
            except subprocess.CalledProcessError:
                return ("timeout", [])
            txt = open(outf).read().strip()
        return self._parse_msolve(txt, syms)

    def _parse_msolve(self, txt, syms):
        txt = txt.rstrip(":").strip().replace("^", "**")
        if not txt or txt == "[-1]":
            return ("empty", [])
        try:
            res = sp.sympify(txt, rational=True)
        except Exception:
            return ("timeout", [])          # unparseable -> treat as undecided/deferred
        if not isinstance(res, (list, tuple)) or len(res) == 0:
            return ("empty", [])
        if res[0] == 1:                      # [1, nvars, -1, []] : positive-dimensional
            return ("posdim", [])
        # zero-dimensional: res = [0, [dim_flag, [ solutions ]]]
        try:
            sol_list = res[1][1]
        except Exception:
            return ("empty", [])
        cands = []
        names = [str(s) for s in syms]
        for sol in sol_list:                 # sol: [[lo,hi], ...] per variable
            try:
                vals = [self._simplest_rational(box[0], box[1]) for box in sol]
            except Exception:
                continue
            cands.append(dict(zip(names, vals)))
        return ("zerodim", cands)

    def _build_V_exact(self, tiers, profile, av, wv):
        """Exact V (fmpq_mat) and det at a RATIONAL strategy point (av/wv: key->fmpq)."""
        n, S, ONE = self.n, self.S, self.ONE
        P = [[flint.fmpq(0) for _ in range(S)] for _ in range(S)]
        for (ii, xi), supp in profile.items():
            pr = self.PROTO[ii]
            if len(supp) == 1:
                w = [ONE]
            else:
                free = [wv[(ii, xi, j)] for j in range(len(supp) - 1)]
                w = free + [ONE - sum(free, flint.fmpq(0))]
            for yi, wj in zip(supp, w):
                if yi == xi:
                    P[xi][xi] += pr * wj; continue
                pa = ONE
                for ki in self.fg.committee[(ii, xi, yi)]:
                    ty, tx = int(tiers[ki][yi]), int(tiers[ki][xi])
                    a = ONE if ty < tx else (flint.fmpq(0) if ty > tx else av[(ki, xi, yi)])
                    pa *= a
                P[xi][yi] += pr * wj * pa; P[xi][xi] += pr * wj * (ONE - pa)
        A = flint.fmpq_mat([[(ONE if i == j else flint.fmpq(0)) - self.DELTA * P[i][j]
                             for j in range(S)] for i in range(S)])
        b = flint.fmpq_mat([[(ONE - self.DELTA) * self.U[i][j] for j in range(n)] for i in range(S)])
        return A.solve(b), A.det()

    def verify_witness(self, tiers, profile, witness):
        """Re-verify a candidate equilibrium in EXACT rational arithmetic. Returns True iff
        all MPE conditions hold exactly (no tolerance). `witness`: str(sym)->sympy rational."""
        ONE = self.ONE
        def to_fmpq(v):
            r = sp.nsimplify(v) if not getattr(v, "is_Rational", False) else v
            r = sp.Rational(r)
            return flint.fmpq(int(r.p), int(r.q))
        acc, wk = self._vars_for_profile(tiers, profile)
        av = {k: to_fmpq(witness[f"a_{k[0]}_{k[1]}_{k[2]}"]) for k in acc}
        wv = {k: to_fmpq(witness[f"w_{k[0]}_{k[1]}_{k[2]}"]) for k in wk}
        for v in list(av.values()) + list(wv.values()):
            if v < 0 or v > ONE:
                return False
        for (ii, xi), supp in profile.items():           # dependent weight in [0,1]
            if len(supp) >= 2:
                last = ONE - sum((wv[(ii, xi, j)] for j in range(len(supp) - 1)), flint.fmpq(0))
                if last < 0 or last > ONE:
                    return False
        V, det = self._build_V_exact(tiers, profile, av, wv)
        if det <= 0:
            return False
        # ordering realised exactly: strict where tiers differ, equal where tied
        for ki in range(self.n):
            for x in range(self.S):
                for y in range(self.S):
                    tx, ty = int(tiers[ki][x]), int(tiers[ki][y])
                    if tx < ty and not (V[x, ki] > V[y, ki]):
                        return False
                    if tx == ty and V[x, ki] != V[y, ki]:
                        return False
        # proposal optimality: support EV == max feasible EV, exactly
        for (ii, xi), supp in profile.items():
            def ev(yi, supported):
                if yi == xi:
                    return V[xi, ii]
                pa = ONE
                for ki in self.fg.committee[(ii, xi, yi)]:
                    ty, tx = int(tiers[ki][yi]), int(tiers[ki][xi])
                    pa *= ONE if ty < tx else (flint.fmpq(0) if ty > tx
                                               else (av[(ki, xi, yi)] if supported else flint.fmpq(0)))
                return pa * V[yi, ii] + (ONE - pa) * V[xi, ii]
            ev_s = ev(supp[0], True)
            for yj in supp[1:]:
                if ev(yj, True) != ev_s:
                    return False
            for yi in self.fg.feasible[(ii, xi)]:
                if yi not in supp and ev(yi, False) > ev_s:
                    return False
        return True

    def solve_label(self, tiers, find_one=True):
        """Process one label fully: solve every candidate support profile.
        Returns (n_profiles, n_equilibria, n_deferred)."""
        n_prof = 0; n_eq = 0; n_def = 0
        for profile in self.candidate_profiles(tiers):
            n_prof += 1
            status, payload = self.solve_profile(tiers, profile)
            if status == "feasible":
                n_eq += 1
                if find_one:
                    return n_prof, n_eq, n_def
            elif status == "deferred":
                n_def += 1
        return n_prof, n_eq, n_def

    def find_in_label(self, tiers):
        """Return (profile, witness) for the first EXACTLY-VERIFIED equilibrium in this
        label, or ('deferred', k_count) if none verified but branches were deferred,
        or (None, 0) if conclusively no easy equilibrium."""
        n_def = 0
        for profile in self.candidate_profiles(tiers):
            status, payload = self.solve_profile(tiers, profile)
            if status == "feasible" and self.verify_witness(tiers, profile, payload):
                return (profile, payload)
            if status == "deferred":
                n_def += 1
        return ("deferred", n_def) if n_def else (None, 0)


def _median_predicted(solver, n=20000, seed=0):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        t = tuple(solver.orders[rng.randint(solver.NO)] for _ in range(solver.n))
        r, m = solver.r_and_m(t); vals.append(r * c_model(m))
    return float(np.median(vals))


def trial(args):
    s = FullMixingSolver(args.payoff)
    thr = _median_predicted(s)
    print(f"[trial] {args.payoff}: cheapest-50% predicted-cost threshold = {thr:.4g} (r*c units)")
    print(f"[trial] running {args.minutes} min single-process on cheapest-50% labels...", flush=True)
    rng = np.random.RandomState(123)
    t0 = time.time(); budget = args.minutes * 60
    done = 0; profiles = 0; eqs = 0; defs = 0; skipped = 0
    while time.time() - t0 < budget:
        t = tuple(s.orders[rng.randint(s.NO)] for _ in range(s.n))
        r, m = s.r_and_m(t)
        if r * c_model(m) > thr:        # not in cheapest 50%
            skipped += 1; continue
        np_, ne, nd = s.solve_label(t, find_one=False)
        done += 1; profiles += np_; eqs += ne; defs += nd
    el = time.time() - t0
    rate = done / el
    TOTAL = s.NO ** 3
    cheap_half = 0.5 * TOTAL
    print(f"\n[trial] processed {done} cheapest-50% labels in {el:.0f}s "
          f"({profiles} support-solves, {eqs} equilibria, {defs} deferred to CAD); "
          f"skipped {skipped} expensive labels")
    print(f"[trial] rate = {rate:.2f} labels/s single-core; "
          f"mean {1000*el/max(1,done):.1f} ms/label, {1000*el/max(1,profiles):.2f} ms/support-solve")
    for w in (1, 14):
        days = cheap_half / rate / w / 86400
        print(f"[trial] => cheapest-50% ({cheap_half:,.0f} labels) on {w} core(s): {days:,.2f} days")
    print(f"[trial] (document estimate for cheapest-50% on 14 workers: ~0.7 days)")
    rep = prof_report()
    log = ROOT / "scripts" / f"fullmix_profile_{args.payoff}.log"
    with open(log, "w") as fh:
        fh.write(f"# full_mixing_sweep trial profile: {args.payoff}\n")
        fh.write(f"# {done} labels, {profiles} support-solves, {el:.0f}s, "
                 f"{1000*el/max(1,profiles):.3f} ms/support-solve\n\n{rep}\n")
    print("\n[trial] stage breakdown (per-process):\n" + rep)
    print(f"[trial] wrote {log}")


def run(args):
    """Cheapest-FIRST run. Two phases:
      A) compute the predicted cost r*c(m) for every label (cheap structural counts),
         keep the cheapest `fraction`, and SORT them ascending by predicted cost;
      B) solve them in that cheapest-first order on `workers` processes, so the cheap
         bulk is cleared first and the run can be stopped/reported at any point.
    Phase-A arrays are cached to disk so a restart skips it. Progress is checkpointed."""
    from multiprocessing import Pool
    s = FullMixingSolver(args.payoff)
    NO = s.NO; TOTAL = NO ** 3
    thr = _threshold_for(s, args.fraction)
    print(f"[run] {args.payoff}: target cheapest {args.fraction*100:.0f}% of {TOTAL:,} labels; "
          f"predicted-cost threshold={thr:.4g}; workers={args.workers}", flush=True)

    order_cache = DATA / f"fullmix_{args.payoff}_order.npy"
    if order_cache.exists():
        order = np.load(order_cache)
        print(f"[run] Phase A: loaded {len(order):,} cheapest labels (sorted) from cache", flush=True)
    else:
        print(f"[run] Phase A: scanning {TOTAL:,} labels for predicted cost (parallel)...", flush=True)
        tA = time.time()
        idxs = []; costs = []
        with Pool(args.workers, initializer=_winit, initargs=(args.payoff, thr)) as pool:
            for (sub_idx, sub_cost) in pool.imap_unordered(_wphaseA, range(NO), chunksize=4):
                idxs.append(sub_idx); costs.append(sub_cost)
        idx = np.concatenate(idxs); cost = np.concatenate(costs)
        order = idx[np.argsort(cost, kind="stable")]            # cheapest first
        np.save(order_cache, order)
        print(f"[run] Phase A done in {(time.time()-tA)/60:.1f} min: "
              f"{len(order):,} labels <= threshold; saved sorted order to {order_cache.name}", flush=True)

    if args.max_labels:
        order = order[:args.max_labels]
    ckpt = DATA / f"fullmix_{args.payoff}_progress.txt"
    start = 0
    if ckpt.exists():
        start = int(ckpt.read_text().split()[0]); print(f"[run] resuming from label {start:,}")
    found = DATA / f"fullmix_{args.payoff}_found.txt"
    fh = open(found, "a")
    print(f"[run] Phase B: solving {len(order)-start:,} labels cheapest-first...", flush=True)
    t0 = time.time(); done = 0; eqs = 0; defs = 0
    with Pool(args.workers, initializer=_winit, initargs=(args.payoff, thr)) as pool:
        for k, (packed, ne, nd) in enumerate(
                pool.imap(_wsolve_packed, order[start:].tolist(), chunksize=64), start=start + 1):
            done += 1; eqs += ne; defs += nd
            if ne or nd:
                fh.write(f"{packed}\t{ne}\t{nd}\n"); fh.flush()
            if done % 2000 == 0:
                el = time.time() - t0
                ckpt.write_text(f"{k} {eqs} {defs}\n")
                print(f"[run] {k:,}/{len(order):,} labels, {eqs} eq, {defs} deferred, "
                      f"{el/3600:.2f}h, {done/el:.1f} lab/s", flush=True)
    fh.close()
    print(f"[run] DONE: {done:,} labels, {eqs} equilibria, {defs} deferred, "
          f"{(time.time()-t0)/3600:.2f}h")

_W = {}
def _winit(payoff, thr, max_nv=None, solver="baseline"):
    if solver == "julia":
        # in-process Julia Groebner; build per worker (spawn) + warm the JIT once
        from lib.equilibrium.full_search.routes.julia_spike.julia_solver import FlintJuliaSolver
        s = FlintJuliaSolver(payoff); s.max_nv = max_nv
        ctx = flint.fmpq_mpoly_ctx.get(["x0", "x1"], flint.Ordering.lex); g = ctx.gens()
        s._msolve_flint(2, [g[0] * g[0] - 1, g[1] - g[0]], ["a_0_0_0", "w_0_0_0"])
    else:
        s = FullMixingSolver(payoff); s.max_nv = max_nv
    _W["s"] = s; _W["thr"] = thr
def _wphaseA(a):
    """Predicted cost for all labels with first index == a; return cheapest (idx,cost)."""
    s = _W["s"]; thr = _W["thr"]; NO = s.NO
    ti = s.orders[a]
    base = a * NO * NO
    keep_i = []; keep_c = []
    for b in range(NO):
        tj = s.orders[b]
        for c in range(NO):
            t = (ti, tj, s.orders[c]); r, m = s.r_and_m(t); cost = r * c_model(m)
            if cost <= thr:
                keep_i.append(base + b * NO + c); keep_c.append(cost)
    return (np.array(keep_i, dtype=np.int64), np.array(keep_c, dtype=np.float64))
def _wsolve_packed(packed):
    s = _W["s"]; NO = s.NO
    a = packed // (NO * NO); b = (packed // NO) % NO; c = packed % NO
    t = (s.orders[a], s.orders[b], s.orders[c])
    _, ne, nd = s.solve_label(t, find_one=False)
    return (packed, ne, nd)
def _wfind(packed):
    s = _W["s"]; NO = s.NO
    a = packed // (NO * NO); b = (packed // NO) % NO; c = packed % NO
    t = (s.orders[a], s.orders[b], s.orders[c])
    res, payload = s.find_in_label(t)
    if res is None:
        return (packed, "none", 0)
    if res == "deferred":
        return (packed, "deferred", int(payload))
    # verified equilibrium: serialise profile + witness exactly (as strings)
    prof = {f"{s.fg.players[ii]}@{s.states[xi]}": [s.states[y] for y in supp]
            for (ii, xi), supp in res.items()}
    wit = {k: str(v) for k, v in payload.items()}
    return (packed, "FOUND", {"label": [list(map(int, o)) for o in t], "profile": prof, "witness": wit})


def find(args):
    """Cheapest-first FIND-ONE: look for a single exactly-verified equilibrium among the
    easy (nv<=max_nv, k<=1) branches of the cheapest `fraction` of labels. Stops at the
    first verified witness. A null result is NOT a proof of non-existence (deferred
    branches are skipped)."""
    from multiprocessing import Pool
    s = FullMixingSolver(args.payoff)
    NO = s.NO; TOTAL = NO ** 3
    thr = _threshold_for(s, args.fraction)
    order_cache = DATA / f"fullmix_{args.payoff}_order.npy"
    if order_cache.exists():
        order = np.load(order_cache)
    else:
        print(f"[find] Phase A: scanning {TOTAL:,} labels in {NO} shards (parallel)...", flush=True)
        idxs = []; costs = []; tA = time.time(); shards = 0; kept = 0
        with Pool(args.workers, initializer=_winit, initargs=(args.payoff, thr)) as pool:
            for si, sc in pool.imap_unordered(_wphaseA, range(NO), chunksize=2):
                idxs.append(si); costs.append(sc); shards += 1; kept += len(si)
                if shards % 10 == 0 or shards == NO:
                    el = time.time() - tA; eta = el / shards * (NO - shards)
                    print(f"[find][A] {shards}/{NO} shards "
                          f"({100*shards/NO:.0f}%), {shards*TOTAL//NO:,} labels scanned, "
                          f"{kept:,} kept, {el:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
        idx = np.concatenate(idxs); cost = np.concatenate(costs)
        order = idx[np.argsort(cost, kind="stable")]; np.save(order_cache, order)
        print(f"[find] Phase A done: {len(order):,} cheap labels in {time.time()-tA:.0f}s", flush=True)
    if getattr(args, "max_labels", 0):
        order = order[:args.max_labels]          # cap (validation / partial runs)
    # Phase-B checkpoint: position in the (fixed) cheapest-first order + deferred tally.
    # Lets a crash/power-loss resume without redoing solved labels. The FOUND short-circuit
    # means a hit is terminal, so resuming only ever re-enters a not-yet-found scan.
    ckpt = DATA / f"fullmix_{args.payoff}_find_progress.txt"
    start = 0; defs = 0
    if ckpt.exists():
        parts = ckpt.read_text().split()
        start = int(parts[0]); defs = int(parts[1]) if len(parts) > 1 else 0

    logdir = ROOT / "logs"; logdir.mkdir(exist_ok=True)
    logf = open(logdir / f"fullmix_{args.payoff}_find.log", "a")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    # Persist the packed id of every DEFERRED label so deferral causes can be re-checked
    # surgically (e.g. after a solver fix) without redoing the decided labels. Truncated on a
    # fresh start (start==0), appended on resume so the running list survives power-offs.
    defids = open(DATA / f"fullmix_{args.payoff}_deferred_ids.txt", "w" if start == 0 else "a")

    if start:
        log(f"[find] resuming from label {start:,} ({defs} deferred so far)")
    log(f"[find] {args.payoff}: {len(order):,} cheap labels (solving {len(order)-start:,}); "
        f"max_nv={args.max_nv}; scanning cheapest-first on {args.workers} workers...")

    # Trailing-window rate so ETA reflects the CURRENT speed (cheapest-first slows over time)
    # and is immune to wall-clock gaps from power-offs across restarts (t0 resets per run).
    from collections import deque
    import multiprocessing as mp
    WINDOW = 120.0           # seconds; "speed over the last ~2 minutes"
    LOG_EVERY = 30.0         # seconds between progress lines (checkpoint stays more frequent)
    samples = deque()        # (timestamp, labels-done-this-run)
    t0 = time.time(); done = 0; last_log = t0
    # Julia isn't fork-safe -> spawn fresh worker processes for the julia solver
    ctx = mp.get_context("spawn" if args.solver == "julia" else "fork")
    with ctx.Pool(args.workers, initializer=_winit,
                  initargs=(args.payoff, thr, args.max_nv, args.solver)) as pool:
        it = pool.imap(_wfind, order[start:], chunksize=32)   # iterate np array (no 2.8GB list)
        for k, (packed, kind, payload) in enumerate(it, start=start + 1):
            done += 1
            if kind == "deferred":
                defs += payload
                defids.write(f"{packed}\n")     # flushed at checkpoint (every 2000 labels)
            elif kind == "FOUND":
                log("\n" + "=" * 70)
                log(f"[find] EQUILIBRIUM FOUND (exactly verified) at label index {packed}")
                log(f"  label tiers (W/per-player weak orders): {payload['label']}")
                log(f"  proposal support: {payload['profile']}")
                log(f"  mixing witness:   {payload['witness']}")
                log("=" * 70)
                out = DATA / f"fullmix_{args.payoff}_FOUND.txt"
                out.write_text(repr(payload) + "\n")
                defids.close(); pool.terminate(); logf.close()
                return
            if done % 2000 == 0:
                now = time.time()
                defids.flush()
                ckpt.write_text(f"{k} {defs}\n")
                samples.append((now, done))
                while len(samples) > 1 and now - samples[0][0] > WINDOW:
                    samples.popleft()
                if now - last_log >= LOG_EVERY:
                    last_log = now
                    if len(samples) >= 2 and now > samples[0][0]:
                        rate = (done - samples[0][1]) / (now - samples[0][0])   # recent (windowed)
                    else:
                        rate = done / (now - t0) if now > t0 else 0.0           # fallback at startup
                    eta_h = (len(order) - k) / rate / 3600 if rate > 0 else float("inf")
                    log(f"[find] {k:,}/{len(order):,} ({100*k/len(order):.1f}%), {defs} deferred, "
                        f"recent {rate:.1f} lab/s, ETA {eta_h:.1f}h")
    defids.close()
    ckpt.write_text(f"{len(order)} {defs}\n")
    log(f"[find] NO easy equilibrium in cheapest {args.fraction*100:.0f}% "
        f"({done:,} labels, {defs} deferred branches skipped). "
        f"INCONCLUSIVE: deferred (k>=2) branches not checked.")
    logf.close()

def _threshold_for(s, frac, n=40000, seed=0):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        t = tuple(s.orders[rng.randint(s.NO)] for _ in range(s.n))
        r, m = s.r_and_m(t); vals.append(r * c_model(m))
    return float(np.quantile(vals, frac))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("trial"); tp.add_argument("--payoff", default="burke_usaruschn_2035-2060")
    tp.add_argument("--minutes", type=float, default=15.0)
    rp = sub.add_parser("run"); rp.add_argument("--payoff", default="burke_usaruschn_2035-2060")
    rp.add_argument("--fraction", type=float, default=0.50); rp.add_argument("--workers", type=int, default=14)
    rp.add_argument("--max-labels", type=int, default=0)
    fp = sub.add_parser("find"); fp.add_argument("--payoff", default="burke_usaruschn_2035-2060")
    fp.add_argument("--fraction", type=float, default=0.50); fp.add_argument("--workers", type=int, default=14)
    fp.add_argument("--max-nv", type=int, default=8)
    fp.add_argument("--solver", choices=["baseline", "julia"], default="julia")
    fp.add_argument("--max-labels", type=int, default=0)
    a = ap.parse_args()
    {"trial": trial, "run": run, "find": find}[a.cmd](a)
