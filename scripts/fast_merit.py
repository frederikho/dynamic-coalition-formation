"""Fast numpy-native merit evaluator for pure strategies (unanimity rule).

Mirrors TransitionProbabilities (unanimity) + MDP + the merit definition in
merit_search.py, but with no pandas in the hot loop. Validated against the
pandas path before use.

Pure-strategy structure under unanimity:
  - each (proposer i, current x) proposes exactly one y* (proposals[i,x])
  - p_app(i,x,y) = AND over committee members of approval bits (in {0,1})
  - row x of P:  sum_i protocol[i] * ( e_{y*}  if approved else  e_x )
  - V[:,j] = solve(I - delta P, (1-delta) u[:,j])
"""
import numpy as np
from lib.utils import get_approval_committee


class FastGame:
    def __init__(self, setup):
        self.players = list(setup['players'])
        self.states = list(setup['state_names'])
        self.S = len(self.states)
        self.n = len(self.players)
        self.delta = setup['discounting']
        self.sidx = {s: i for i, s in enumerate(self.states)}
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.protocol = np.array([setup['protocol'][p] for p in self.players])
        # payoffs u[state, player]
        u = setup['payoffs']
        self.u = np.array([[float(u.loc[s, p]) for p in self.players]
                           for s in self.states])
        eff = setup['effectivity']
        self.forbidden = setup.get('forbidden_proposals', frozenset())
        # committee member indices per (i_idx, x_idx, y_idx)
        self.committee = {}
        self.feasible = {}
        for ii, i in enumerate(self.players):
            for xi, x in enumerate(self.states):
                feas = []
                for yi, y in enumerate(self.states):
                    c = get_approval_committee(eff, self.players, i, x, y)
                    self.committee[(ii, xi, yi)] = [self.pidx[k] for k in c]
                    if (i, x, y) not in self.forbidden:
                        feas.append(yi)
                self.feasible[(ii, xi)] = feas
        # enumerate approval decision keys (ii,xi,yi,ki) -> stable order
        self.appr_keys = []
        for ii in range(self.n):
            for xi in range(self.S):
                for yi in range(self.S):
                    for ki in self.committee[(ii, xi, yi)]:
                        self.appr_keys.append((ii, xi, yi, ki))

    # ---- strategy arrays ----
    # approvals: dict {(ii,xi,yi,ki): 0/1}, proposals: array[n, S] = chosen yi
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
        v = 1
        for ki in c:
            v &= approvals[(ii, xi, yi, ki)]
            if v == 0:
                return 0.0
        return float(v)

    def build_P(self, approvals, proposals):
        P = np.zeros((self.S, self.S))
        for ii in range(self.n):
            pr = self.protocol[ii]
            for xi in range(self.S):
                yi = proposals[ii, xi]
                approved = self.p_app(approvals, ii, xi, yi)
                if approved >= 1.0 and yi != xi:
                    P[xi, yi] += pr
                else:
                    P[xi, xi] += pr
        return P

    def solve_V(self, P):
        A = np.eye(self.S) - self.delta * P
        b = (1 - self.delta) * self.u           # (S, n)
        return np.linalg.solve(A, b)            # (S, n)

    def merit(self, approvals, proposals, return_parts=False):
        P = self.build_P(approvals, proposals)
        V = self.solve_V(P)                      # V[state, player]
        # approval violations
        appr_v = 0.0
        for (ii, xi, yi, ki), a in approvals.items():
            diff = V[yi, ki] - V[xi, ki]
            if a == 1 and diff < 0:
                appr_v += -diff
            elif a == 0 and diff > 0:
                appr_v += diff
        # proposal violations
        prop_v = 0.0
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
                ev_ch = pa * V[yi, ii] + (1 - pa) * V[xi, ii]
                prop_v += best - ev_ch
        if return_parts:
            return appr_v + prop_v, appr_v, prop_v, V
        return appr_v + prop_v

    # neighbours: flip one approval bit, or change one proposal target
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


def greedy(fg, approvals, proposals, max_iter=500):
    cur = (approvals, proposals); cur_m = fg.merit(*cur)
    for _ in range(max_iter):
        if cur_m <= 1e-9:
            return cur, cur_m, True
        best, best_m = None, cur_m
        for nb in fg.neighbours(*cur):
            m = fg.merit(*nb)
            if m < best_m - 1e-12:
                best, best_m = nb, m
        if best is None:
            return cur, cur_m, False
        cur, cur_m = best, best_m
    return cur, cur_m, cur_m <= 1e-9


def search(fg, seed=0, restarts=200, anneal_iters=800, T0=1.0, log=None):
    rng = np.random.RandomState(seed)
    best = None
    for r in range(restarts):
        a, p = fg.random_strategy(rng)
        (a, p), m, ok = greedy(fg, a, p)
        if ok:
            return (a, p), m, True, r
        cur, cur_m, T = (a, p), m, T0
        for it in range(anneal_iters):
            nbs = list(fg.neighbours(*cur))
            nb = nbs[rng.randint(len(nbs))]
            nm = fg.merit(*nb)
            if nm < cur_m or rng.rand() < np.exp(-(nm - cur_m) / max(T, 1e-9)):
                cur, cur_m = nb, nm
            T *= 0.997
            if cur_m <= 1e-9:
                (cur, cur_m, ok) = greedy(fg, *cur)
                if ok:
                    return cur, cur_m, True, r
        (cur, cur_m, ok) = greedy(fg, *cur)
        if ok:
            return cur, cur_m, True, r
        if best is None or cur_m < best[1]:
            best = (cur, cur_m)
        if log and r % 20 == 0:
            print(f'  restart {r}: best M so far={best[1]:.4f}', flush=True)
    return best[0], best[1], False, restarts


def plateau_search(fg, seed=0, restarts=300, walk=4000, sideways_tol=1e-9, log=None):
    """First-improvement local search that also drifts across equal-merit
    plateaus. Greedy strict-descent stalls because many decisions are
    'don't-care' (Delta M = 0); allowing sideways moves lets the walk traverse
    the large connected M=0 plateau region."""
    rng = np.random.RandomState(seed)
    best = None
    for r in range(restarts):
        a, p = fg.random_strategy(rng)
        cur, cur_m = (a, p), fg.merit(a, p)
        stall = 0
        for it in range(walk):
            if cur_m <= 1e-9:
                return cur, cur_m, True, r
            nbs = list(fg.neighbours(*cur))
            rng.shuffle(nbs)
            moved = False
            # first strict improvement
            for nb in nbs:
                m = fg.merit(*nb)
                if m < cur_m - 1e-12:
                    cur, cur_m, moved, stall = nb, m, True, 0
                    break
            if not moved:
                # take a random sideways (equal) move to escape the plateau
                cand = [(nb, fg.merit(*nb)) for nb in nbs[:30]]
                flat = [nb for nb, m in cand if m <= cur_m + sideways_tol]
                if flat:
                    cur = flat[rng.randint(len(flat))]
                    cur_m = fg.merit(*cur)
                    stall += 1
                else:
                    break
                if stall > 200:
                    break
        if best is None or cur_m < best[1]:
            best = (cur, cur_m)
        if log and r % 25 == 0:
            print(f'  restart {r}: best M={best[1]:.3e}', flush=True)
    return best[0], best[1], False, restarts
