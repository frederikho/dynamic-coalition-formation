"""Continuous merit function over the discrete strategy/cell space, plus local search.

Why this parametrization
------------------------
The probe (residual_metric_probe.py) showed the equilibrium sits in an ordinal
cell of radius ~ the minimum value gap (~2.5e-4 on values ~100, i.e. relative
~1e-6). Searching continuous V hits a needle. So we instead search the DISCRETE
strategy space, where each pure strategy sigma maps -- via an exact closed-form
Bellman solve -- to one value matrix V(sigma). We define a CONTINUOUS merit:

    M(sigma) = approval_violation(sigma) + proposal_violation(sigma)  >= 0

    approval_violation: for each committee approval decision a in {0,1} on x->y by k,
        if a==1 (approve) but V_k(y) < V_k(x):  add  V_k(x) - V_k(y)
        if a==0 (reject)  but V_k(y) > V_k(x):  add  V_k(y) - V_k(x)
    proposal_violation: for each (proposer i, state x) with chosen next y*,
        add  max_y EV_i(x,y) - EV_i(x,y*),  EV = p_app*V(y) + (1-p_app)*V(x)

M(sigma)=0  <=>  the verifier passes (sigma is an MPE). M is exact (V solved
exactly per sigma), so there is no needle problem; we just navigate the discrete
strategy graph downhill to M=0.

Local search: from a candidate sigma, evaluate M; consider single-decision
neighbours (flip one approval bit, or change one proposal target) and move to the
best neighbour. Best-response (policy-iteration) cycles; M provides a Lyapunov-
style merit + simulated-annealing escape.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.residual_metric_probe import build_setup, value_of_strategy, empty_strategy_df
from lib.utils import get_approval_committee
from lib.probabilities import TransitionProbabilities
from lib.mdp import MDP


class Game:
    """Holds the static game and precomputed structure for fast M(sigma)."""

    def __init__(self, setup):
        self.setup = setup
        self.players = setup['players']
        self.states = setup['state_names']
        self.eff = setup['effectivity']
        self.protocol = setup['protocol']
        self.payoffs = setup['payoffs']
        self.delta = setup['discounting']
        self.unanimity = setup['unanimity_required']
        self.forbidden = setup.get('forbidden_proposals', frozenset())

        # committees and feasible proposal targets
        self.committee = {}
        for i in self.players:
            for x in self.states:
                for y in self.states:
                    self.committee[(i, x, y)] = get_approval_committee(
                        self.eff, self.players, i, x, y)
        self.feasible = {}
        for i in self.players:
            for x in self.states:
                self.feasible[(i, x)] = [
                    y for y in self.states if (i, x, y) not in self.forbidden]

    # ---- strategy representation: dict approvals {(i,x,y,k):0/1}, proposals {(i,x):y} ----
    def random_strategy(self, rng):
        approvals = {}
        for i in self.players:
            for x in self.states:
                for y in self.states:
                    for k in self.committee[(i, x, y)]:
                        approvals[(i, x, y, k)] = int(rng.randint(2))
        proposals = {}
        for i in self.players:
            for x in self.states:
                proposals[(i, x)] = self.feasible[(i, x)][
                    rng.randint(len(self.feasible[(i, x)]))]
        return approvals, proposals

    def to_df(self, approvals, proposals):
        df = empty_strategy_df(self.players, self.states)
        for (i, x), y_star in proposals.items():
            for y in self.states:
                df.loc[(x, 'Proposition', np.nan), (f'Proposer {i}', y)] = \
                    1.0 if y == y_star else 0.0
        for (i, x, y, k), a in approvals.items():
            df.loc[(x, 'Acceptance', k), (f'Proposer {i}', y)] = float(a)
        return df

    def value_and_approvals(self, approvals, proposals):
        df = self.to_df(approvals, proposals)
        V, P, P_proposals, P_approvals = value_of_strategy(df, self.setup)
        return V.astype(float), P_approvals, df

    def merit(self, approvals, proposals, return_parts=False):
        V, P_app, df = self.value_and_approvals(approvals, proposals)
        appr_v = 0.0
        for (i, x, y, k), a in approvals.items():
            diff = V.loc[y, k] - V.loc[x, k]   # >0 => k prefers y
            if a == 1 and diff < 0:
                appr_v += -diff
            elif a == 0 and diff > 0:
                appr_v += diff
        prop_v = 0.0
        for i in self.players:
            for x in self.states:
                evs = {}
                for y in self.feasible[(i, x)]:
                    pa = P_app[(i, x, y)]
                    evs[y] = pa * V.loc[y, i] + (1 - pa) * V.loc[x, i]
                best = max(evs.values())
                chosen = proposals[(i, x)]
                prop_v += best - evs[chosen]
        if return_parts:
            return appr_v + prop_v, appr_v, prop_v, V
        return appr_v + prop_v

    # ---- neighbourhood: single-decision changes ----
    def neighbours(self, approvals, proposals):
        for key in approvals:
            a2 = dict(approvals)
            a2[key] = 1 - a2[key]
            yield a2, proposals
        for (i, x), y_star in proposals.items():
            for y in self.feasible[(i, x)]:
                if y != y_star:
                    p2 = dict(proposals)
                    p2[(i, x)] = y
                    yield approvals, p2

    def best_response_step(self, approvals, proposals):
        """Policy-iteration step: set every decision to be rational given current V."""
        V, P_app, _ = self.value_and_approvals(approvals, proposals)
        na = {}
        for (i, x, y, k) in approvals:
            diff = V.loc[y, k] - V.loc[x, k]
            na[(i, x, y, k)] = 1 if diff > 0 else 0
        # recompute approvals -> approval probs change -> redo proposal BR with new committee approval
        df = self.to_df(na, proposals)
        _, _, _, P_app2 = (lambda r: r)(value_of_strategy(df, self.setup))
        np_ = {}
        for i in self.players:
            for x in self.states:
                best_y, best_ev = None, -np.inf
                for y in self.feasible[(i, x)]:
                    pa = P_app2[(i, x, y)]
                    ev = pa * V.loc[y, i] + (1 - pa) * V.loc[x, i]
                    if ev > best_ev:
                        best_ev, best_y = ev, y
                np_[(i, x)] = best_y
        return na, np_


def greedy_descent(g, approvals, proposals, max_iter=200):
    cur = (approvals, proposals)
    cur_m = g.merit(*cur)
    for _ in range(max_iter):
        if cur_m <= 1e-9:
            return cur, cur_m, True
        best_n, best_m = None, cur_m
        for n in g.neighbours(*cur):
            m = g.merit(*n)
            if m < best_m - 1e-12:
                best_n, best_m = n, m
        if best_n is None:
            return cur, cur_m, False  # local minimum
        cur, cur_m = best_n, best_m
    return cur, cur_m, cur_m <= 1e-9


def search(g, seed=0, restarts=40, anneal_iters=300, T0=2.0):
    """Random restarts of greedy descent; on stall, simulated-annealing kick."""
    rng = np.random.RandomState(seed)
    best_overall = None
    for r in range(restarts):
        a, p = g.random_strategy(rng)
        (a, p), m, ok = greedy_descent(g, a, p)
        if ok:
            return (a, p), m, True, r
        # annealing kick out of the local min
        cur, cur_m = (a, p), m
        T = T0
        for it in range(anneal_iters):
            neigh = list(g.neighbours(*cur))
            n = neigh[rng.randint(len(neigh))]
            nm = g.merit(*n)
            if nm < cur_m or rng.rand() < np.exp(-(nm - cur_m) / max(T, 1e-6)):
                cur, cur_m = n, nm
            T *= 0.99
            if cur_m <= 1e-9:
                return cur, cur_m, True, r
            if it % 30 == 0:
                (cur, cur_m, ok) = (*greedy_descent(g, *cur)[:2], False) if False else (cur, cur_m, False)
        if best_overall is None or cur_m < best_overall[1]:
            best_overall = (cur, cur_m)
    return best_overall[0], best_overall[1], False, restarts


if __name__ == '__main__':
    import time
    payoff = sys.argv[1] if len(sys.argv) > 1 else 'burke_usachnnde_2035-2060_summed_until_2060'
    scenario = sys.argv[2] if len(sys.argv) > 2 else 'power_threshold_RICE_n3'
    rule = sys.argv[3] if len(sys.argv) > 3 else 'adjacent_step'
    setup = build_setup(scenario, ROOT / 'payoff_tables' / f'{payoff}.xlsx', rule)
    g = Game(setup)
    print(f'{payoff}  states={g.states}')
    t0 = time.time()
    (a, p), m, ok, r = search(g, seed=0)
    dt = time.time() - t0
    print(f'search: M={m:.3e}  solved={ok}  restart={r}  time={dt:.1f}s')
    if ok:
        # confirm with the real verifier
        from lib.utils import verify_equilibrium
        V, _, pp, pa = value_of_strategy(g.to_df(a, p), setup)
        res = {'players': g.players, 'state_names': g.states, 'V': V.astype(float),
               'P_proposals': pp, 'P_approvals': pa, 'effectivity': g.eff,
               'strategy_df': g.to_df(a, p),
               'forbidden_proposals': g.forbidden}
        vok, vmsg = verify_equilibrium(res, atol=1e-6)
        print('verify_equilibrium:', vok)
