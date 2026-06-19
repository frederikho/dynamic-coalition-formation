"""Merit-descent equilibrium solver.

Reformulates the MPE search as continuous optimisation over the discrete pure-
strategy space. For a pure strategy sigma we solve the value matrix V exactly
(Bellman) and define a continuous merit M(sigma) >= 0 that is zero iff the two
equilibrium conditions hold:

    approval violation: for each committee bit a in {0,1} on x->y by player k,
        a==1 & V_k(y)<V_k(x) -> add V_k(x)-V_k(y)
        a==0 & V_k(y)>V_k(x) -> add V_k(y)-V_k(x)
    proposal violation: for each (proposer i, state x) choosing y*,
        add  max_y EV_i(x,y) - EV_i(x,y*),  EV = p_app*V(y) + (1-p_app)*V(x)

M(sigma)=0  <=>  verify_equilibrium passes.

Why this works where naive iteration cycles / annealing in value space fails:
the equilibrium sits in an ordinal cell whose radius is the minimum value gap
(~1e-4 for near-flat RICE payoffs) -- a needle in continuous V-space. But in the
discrete strategy space the M=0 set is LARGE and connected (many "don't-care"
decisions on never-proposed/rejected transitions). Strict-descent greedy stalls
on the flat (deltaM=0) plateaus; a plateau-aware walk (first improvement + equal-
merit sideways moves + random restarts) reaches M=0.

Assumes unanimity_required=True (the power_threshold_RICE regime). For majority
rules the transition builder would need extending; we fail loudly rather than
silently produce wrong transitions.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from lib.utils import get_approval_committee


class _FastGame:
    """Numpy-native pure-strategy game with O(0.1ms) merit evaluation."""

    def __init__(self, players, states, effectivity, protocol, payoffs,
                 discounting, forbidden_proposals):
        self.players = list(players)
        self.states = list(states)
        self.S = len(self.states)
        self.n = len(self.players)
        self.delta = float(discounting)
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.sidx = {s: i for i, s in enumerate(self.states)}
        self.protocol = np.array([float(protocol[p]) for p in self.players])
        self.u = np.array([[float(payoffs.loc[s, p]) for p in self.players]
                           for s in self.states])
        self.forbidden = forbidden_proposals or frozenset()
        self.committee = {}
        self.feasible = {}
        for ii, i in enumerate(self.players):
            for xi, x in enumerate(self.states):
                feas = []
                for yi, y in enumerate(self.states):
                    c = get_approval_committee(effectivity, self.players, i, x, y)
                    self.committee[(ii, xi, yi)] = [self.pidx[k] for k in c]
                    if (i, x, y) not in self.forbidden:
                        feas.append(yi)
                self.feasible[(ii, xi)] = feas
        self.appr_keys = [
            (ii, xi, yi, ki)
            for ii in range(self.n) for xi in range(self.S)
            for yi in range(self.S) for ki in self.committee[(ii, xi, yi)]
        ]

    def random_strategy(self, rng):
        approvals = {k: int(rng.randint(2)) for k in self.appr_keys}
        proposals = np.empty((self.n, self.S), dtype=int)
        for ii in range(self.n):
            for xi in range(self.S):
                feas = self.feasible[(ii, xi)]
                proposals[ii, xi] = feas[rng.randint(len(feas))]
        return approvals, proposals

    def p_app(self, approvals, ii, xi, yi):
        c = self.committee[(ii, xi, yi)]
        if not c:
            return 0.0
        for ki in c:
            if approvals[(ii, xi, yi, ki)] == 0:
                return 0.0
        return 1.0

    def build_P(self, approvals, proposals):
        P = np.zeros((self.S, self.S))
        for ii in range(self.n):
            pr = self.protocol[ii]
            for xi in range(self.S):
                yi = proposals[ii, xi]
                if yi != xi and self.p_app(approvals, ii, xi, yi) >= 1.0:
                    P[xi, yi] += pr
                else:
                    P[xi, xi] += pr
        return P

    def solve_V(self, P):
        A = np.eye(self.S) - self.delta * P
        b = (1 - self.delta) * self.u
        return np.linalg.solve(A, b)

    def merit(self, approvals, proposals):
        V = self.solve_V(self.build_P(approvals, proposals))
        m = 0.0
        for (ii, xi, yi, ki), a in approvals.items():
            diff = V[yi, ki] - V[xi, ki]
            if a == 1 and diff < 0:
                m += -diff
            elif a == 0 and diff > 0:
                m += diff
        for ii in range(self.n):
            for xi in range(self.S):
                best = -np.inf
                for yi in self.feasible[(ii, xi)]:
                    pa = self.p_app(approvals, ii, xi, yi)
                    ev = pa * V[yi, ii] + (1 - pa) * V[xi, ii]
                    if ev > best:
                        best = ev
                yi = proposals[ii, xi]
                pa = self.p_app(approvals, ii, xi, yi)
                m += best - (pa * V[yi, ii] + (1 - pa) * V[xi, ii])
        return m

    def neighbours(self, approvals, proposals):
        for k in self.appr_keys:
            a2 = dict(approvals)
            a2[k] = 1 - a2[k]
            yield a2, proposals
        for ii in range(self.n):
            for xi in range(self.S):
                cur = proposals[ii, xi]
                for yi in self.feasible[(ii, xi)]:
                    if yi != cur:
                        p2 = proposals.copy()
                        p2[ii, xi] = yi
                        yield approvals, p2

    # ---- continuous-acceptance (mixed-strategy) machinery ----
    # acceptances are a vector avec[len(appr_keys)] in [0,1]; proposals stay pure.
    def _kpos(self):
        return {k: idx for idx, k in enumerate(self.appr_keys)}

    def p_app_cont(self, avec, kpos, ii, xi, yi):
        c = self.committee[(ii, xi, yi)]
        if not c:
            return 0.0
        v = 1.0
        for ki in c:
            v *= avec[kpos[(ii, xi, yi, ki)]]
        return v

    def build_P_cont(self, avec, kpos, proposals):
        P = np.zeros((self.S, self.S))
        for ii in range(self.n):
            pr = self.protocol[ii]
            for xi in range(self.S):
                yi = proposals[ii, xi]
                pa = 1.0 if yi == xi else self.p_app_cont(avec, kpos, ii, xi, yi)
                P[xi, yi] += pr * pa
                if yi != xi:
                    P[xi, xi] += pr * (1 - pa)
        return P

    def merit_cont(self, avec, kpos, proposals):
        V = self.solve_V(self.build_P_cont(avec, kpos, proposals))
        m = 0.0
        for k in self.appr_keys:
            ii, xi, yi, ki = k
            a = avec[kpos[k]]
            diff = V[yi, ki] - V[xi, ki]
            m += (1 - a) * max(0.0, diff) + a * max(0.0, -diff)
        for ii in range(self.n):
            for xi in range(self.S):
                best = -np.inf
                for yi in self.feasible[(ii, xi)]:
                    pa = 1.0 if yi == xi else self.p_app_cont(avec, kpos, ii, xi, yi)
                    best = max(best, pa * V[yi, ii] + (1 - pa) * V[xi, ii])
                yi = proposals[ii, xi]
                pa = 1.0 if yi == xi else self.p_app_cont(avec, kpos, ii, xi, yi)
                m += best - (pa * V[yi, ii] + (1 - pa) * V[xi, ii])
        return m

    def best_response_proposals(self, avec, kpos, V):
        prop = np.empty((self.n, self.S), dtype=int)
        for ii in range(self.n):
            for xi in range(self.S):
                best_y, best_ev = self.feasible[(ii, xi)][0], -np.inf
                for yi in self.feasible[(ii, xi)]:
                    pa = 1.0 if yi == xi else self.p_app_cont(avec, kpos, ii, xi, yi)
                    ev = pa * V[yi, ii] + (1 - pa) * V[xi, ii]
                    if ev > best_ev + 1e-12:
                        best_ev, best_y = ev, yi
                prop[ii, xi] = best_y
        return prop


def _mixed_alternating_search(fg, seed, restarts, sweeps, n_starts, tol, log=None):
    """Block-coordinate search for MIXED equilibria: minimize the merit over
    continuous acceptance probabilities (L-BFGS-B), then set proposals to the
    best response given V; repeat with random restarts. Finds mixed equilibria
    (continuous acceptance probs at indifference) that pure search cannot."""
    from scipy.optimize import minimize as _minimize
    rng = np.random.RandomState(seed)
    kpos = fg._kpos()
    m = len(fg.appr_keys)
    bounds = [(0.0, 1.0)] * m
    best = None

    def random_prop():
        prop = np.empty((fg.n, fg.S), dtype=int)
        for ii in range(fg.n):
            for xi in range(fg.S):
                feas = fg.feasible[(ii, xi)]
                prop[ii, xi] = feas[rng.randint(len(feas))]
        return prop

    for r in range(restarts):
        prop = random_prop()
        for _ in range(sweeps):
            inner_best = None
            for _s in range(n_starts):
                x0 = rng.uniform(0, 1, m)
                res = _minimize(lambda a: fg.merit_cont(a, kpos, prop), x0,
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                if inner_best is None or res.fun < inner_best[1]:
                    inner_best = (res.x, res.fun)
                if inner_best[1] < tol:
                    break
            xvec, fval = inner_best
            if best is None or fval < best[2]:
                best = (xvec.copy(), prop.copy(), fval)
            if fval < tol:
                return best[0], best[1], best[2], True, r
            V = fg.solve_V(fg.build_P_cont(xvec, kpos, prop))
            new_prop = fg.best_response_proposals(xvec, kpos, V)
            if np.array_equal(new_prop, prop):
                break
            prop = new_prop
        if log and r % 10 == 0:
            log(f"  merit-descent (mixed) restart {r}: best M={best[2]:.3e}")
    return best[0], best[1], best[2], False, restarts


def _plateau_search(fg, seed, restarts, walk, tol, sideways_tol, log=None):
    rng = np.random.RandomState(seed)
    best = None
    for r in range(restarts):
        a, p = fg.random_strategy(rng)
        cur, cur_m = (a, p), fg.merit(a, p)
        stall = 0
        for _ in range(walk):
            if cur_m <= tol:
                return cur, cur_m, True, r
            nbs = list(fg.neighbours(*cur))
            rng.shuffle(nbs)
            moved = False
            for nb in nbs:
                m = fg.merit(*nb)
                if m < cur_m - 1e-12:
                    cur, cur_m, moved, stall = nb, m, True, 0
                    break
            if not moved:
                flat = [nb for nb in nbs[:30] if fg.merit(*nb) <= cur_m + sideways_tol]
                if not flat or stall > 200:
                    break
                cur = flat[rng.randint(len(flat))]
                cur_m = fg.merit(*cur)
                stall += 1
        if best is None or cur_m < best[1]:
            best = (cur, cur_m)
        if log and r % 25 == 0:
            log(f"  merit-descent restart {r}: best M={best[1]:.3e}")
    return best[0], best[1], False, restarts


def solve_with_merit_descent(
    solver,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Find an MPE by plateau-aware descent on the continuous merit M(sigma).

    Recognised params keys:
        merit_restarts      (int,   default 400)   random restarts
        merit_walk          (int,   default 4000)  max moves per restart
        merit_tol           (float, default 1e-9)  merit threshold for success
        merit_sideways_tol  (float, default 1e-9)  equal-merit plateau tolerance
        merit_seed          (int,   default solver.random_seed)

    Returns (strategy_df, result_dict) matching the other solver approaches.
    """
    params = params or {}
    if not getattr(solver, "unanimity_required", True):
        raise ValueError(
            "merit_descent solver currently supports unanimity_required=True only. "
            "Majority-rule transition building is not implemented."
        )

    restarts = int(params.get("merit_restarts", 400))
    walk = int(params.get("merit_walk", 4000))
    tol = float(params.get("merit_tol", 1e-9))
    sideways_tol = float(params.get("merit_sideways_tol", 1e-9))
    seed = int(params.get("merit_seed", getattr(solver, "random_seed", 0)))
    # 'pure'  : plateau-aware discrete search (fast, pure-strategy equilibria)
    # 'mixed' : block-coordinate continuous-acceptance search (finds mixed eq)
    mode = str(params.get("merit_mode", "pure"))

    logger = getattr(solver, "logger", None)
    log = (lambda m: logger.info(m)) if logger else None

    fg = _FastGame(
        players=solver.players,
        states=solver.states,
        effectivity=solver.effectivity,
        protocol=solver.protocol,
        payoffs=solver.payoffs,
        discounting=solver.discounting,
        forbidden_proposals=getattr(solver, "forbidden_proposals", frozenset()),
    )

    if log:
        log("=" * 70)
        log(f"Merit-descent solver (mode={mode})")
        log(f"  players={fg.players}, states={fg.S}, delta={fg.delta}")
        log(f"  restarts={restarts}, tol={tol}, seed={seed}")
        log("=" * 70)

    if mode == "mixed":
        mix_restarts = int(params.get("merit_mixed_restarts", 30))
        mix_sweeps = int(params.get("merit_mixed_sweeps", 8))
        mix_starts = int(params.get("merit_mixed_starts", 8))
        avec, proposals_arr, final_m, ok, used_restart = _mixed_alternating_search(
            fg, seed=seed, restarts=mix_restarts, sweeps=mix_sweeps,
            n_starts=mix_starts, tol=tol, log=log,
        )
        kpos = fg._kpos()
        approvals = {k: float(avec[kpos[k]]) for k in fg.appr_keys}
        proposals = proposals_arr
    else:
        (approvals, proposals), final_m, ok, used_restart = _plateau_search(
            fg, seed=seed, restarts=restarts, walk=walk, tol=tol,
            sideways_tol=sideways_tol, log=log,
        )

    # Write the best strategy back into the solver's dicts, then build the df
    # using the solver's own (canonical) layout helper.
    for ii in range(fg.n):
        for xi in range(fg.S):
            chosen = int(proposals[ii, xi])
            for yi in range(fg.S):
                solver.p_proposals[(fg.players[ii], fg.states[xi], fg.states[yi])] = \
                    1.0 if yi == chosen else 0.0
    for (ii, xi, yi, ki), a in approvals.items():
        solver.r_acceptances[
            (fg.players[ii], fg.states[xi], fg.states[yi], fg.players[ki])
        ] = float(a)

    strategy_df = solver._create_strategy_dataframe()

    result = {
        "converged": ok,
        "final_merit": float(final_m),
        "restarts_used": used_restart,
        "stopping_reason": "merit_zero" if ok else "merit_local_minimum",
    }
    if log:
        log(f"Merit-descent finished: M={final_m:.3e} converged={ok} "
            f"(restart {used_restart})")
    return strategy_df, result
