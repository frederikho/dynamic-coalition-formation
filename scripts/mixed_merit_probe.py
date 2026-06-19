"""Prototype: mixed-strategy merit minimization.

Extends the merit to CONTINUOUS acceptance probabilities a in [0,1]:
    approval violation = (1-a)*max(0, V_k(y)-V_k(x)) + a*max(0, V_k(x)-V_k(y))
which is 0 for ANY a exactly at indifference V_k(y)=V_k(x). The mixing prob is
then pinned by the requirement that the induced V actually makes the relevant
player indifferent -- handled by continuous minimization of M over a (V is a
rational function of a, so M is piecewise-smooth).

Proposals are kept pure (one-hot); we search discrete proposal profiles and, for
each, minimize M over the acceptance vector with L-BFGS-B (multistart).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.equilibrium.scenarios import get_scenario, fill_players
from lib.equilibrium.find import setup_experiment, _parse_players_from_payoff_table
from lib.utils import get_approval_committee


def build_manifold_setup():
    pt = 'payoff_tables/simple_cycle_usachn-2-reduced-further.xlsx'
    cfg = get_scenario('power_threshold_RICE_n3')
    cfg['payoff_table'] = pt
    cfg['allow_non_canonical_states'] = True
    cfg['template_file'] = 'strategy_tables/manifold_lam5.000.xlsx'
    cfg = fill_players(cfg, _parse_players_from_payoff_table(Path(pt)))
    return setup_experiment(cfg)


class MixedGame:
    def __init__(self, setup):
        self.players = list(setup['players'])
        self.states = list(setup['state_names'])
        self.S, self.n = len(self.states), len(self.players)
        self.delta = float(setup['discounting'])
        self.pidx = {p: i for i, p in enumerate(self.players)}
        self.protocol = np.array([float(setup['protocol'][p]) for p in self.players])
        u = setup['payoffs']
        self.u = np.array([[float(u.loc[s, p]) for p in self.players] for s in self.states])
        self.forbidden = setup.get('forbidden_proposals', frozenset())
        eff = setup['effectivity']
        self.committee, self.feasible = {}, {}
        for ii, i in enumerate(self.players):
            for xi, x in enumerate(self.states):
                feas = []
                for yi, y in enumerate(self.states):
                    c = get_approval_committee(eff, self.players, i, x, y)
                    self.committee[(ii, xi, yi)] = [self.pidx[k] for k in c]
                    if (i, x, y) not in self.forbidden:
                        feas.append(yi)
                self.feasible[(ii, xi)] = feas
        self.appr_keys = [(ii, xi, yi, ki)
                          for ii in range(self.n) for xi in range(self.S)
                          for yi in range(self.S) for ki in self.committee[(ii, xi, yi)]]
        self.kpos = {k: idx for idx, k in enumerate(self.appr_keys)}

    def p_app(self, avec, ii, xi, yi):
        c = self.committee[(ii, xi, yi)]
        if not c:
            return 0.0
        v = 1.0
        for ki in c:
            v *= avec[self.kpos[(ii, xi, yi, ki)]]
        return v

    def build_P(self, avec, proposals):
        P = np.zeros((self.S, self.S))
        for ii in range(self.n):
            pr = self.protocol[ii]
            for xi in range(self.S):
                yi = proposals[ii, xi]
                pa = 1.0 if yi == xi else self.p_app(avec, ii, xi, yi)
                P[xi, yi] += pr * pa
                if yi != xi:
                    P[xi, xi] += pr * (1 - pa)
        return P

    def solve_V(self, P):
        return np.linalg.solve(np.eye(self.S) - self.delta * P, (1 - self.delta) * self.u)

    def merit(self, avec, proposals):
        V = self.solve_V(self.build_P(avec, proposals))
        m = 0.0
        for (ii, xi, yi, ki) in self.appr_keys:
            a = avec[self.kpos[(ii, xi, yi, ki)]]
            diff = V[yi, ki] - V[xi, ki]
            m += (1 - a) * max(0.0, diff) + a * max(0.0, -diff)
        for ii in range(self.n):
            for xi in range(self.S):
                best = -np.inf
                for yi in self.feasible[(ii, xi)]:
                    pa = 1.0 if yi == xi else self.p_app(avec, ii, xi, yi)
                    ev = pa * V[yi, ii] + (1 - pa) * V[xi, ii]
                    best = max(best, ev)
                yi = proposals[ii, xi]
                pa = 1.0 if yi == xi else self.p_app(avec, ii, xi, yi)
                m += best - (pa * V[yi, ii] + (1 - pa) * V[xi, ii])
        return m

    def random_proposals(self, rng):
        prop = np.empty((self.n, self.S), dtype=int)
        for ii in range(self.n):
            for xi in range(self.S):
                feas = self.feasible[(ii, xi)]
                prop[ii, xi] = feas[rng.randint(len(feas))]
        return prop


def minimize_over_acceptances(g, proposals, rng, n_starts=12):
    m = len(g.appr_keys)
    bounds = [(0.0, 1.0)] * m
    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(0, 1, m)
        r = minimize(lambda a: g.merit(a, proposals), x0, method='L-BFGS-B',
                     bounds=bounds, options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
        if best is None or r.fun < best[1]:
            best = (r.x, r.fun)
        if best[1] < 1e-10:
            break
    return best


def best_response_proposals(g, avec, V):
    prop = np.empty((g.n, g.S), dtype=int)
    for ii in range(g.n):
        for xi in range(g.S):
            best_y, best_ev = None, -np.inf
            for yi in g.feasible[(ii, xi)]:
                pa = 1.0 if yi == xi else g.p_app(avec, ii, xi, yi)
                ev = pa * V[yi, ii] + (1 - pa) * V[xi, ii]
                if ev > best_ev + 1e-12:
                    best_ev, best_y = ev, yi
            prop[ii, xi] = best_y
    return prop


def alternating_search(g, seed=0, restarts=30, sweeps=8, n_starts=8, tol=1e-9):
    """Block-coordinate: minimize M over continuous acceptances, then set
    proposals to best-response given V; repeat. Random restarts."""
    rng = np.random.RandomState(seed)
    best = None
    for r in range(restarts):
        prop = g.random_proposals(rng)
        for _ in range(sweeps):
            x, fval = minimize_over_acceptances(g, prop, rng, n_starts=n_starts)
            V = g.solve_V(g.build_P(x, prop))
            new_prop = best_response_proposals(g, x, V)
            if best is None or fval < best[2]:
                best = (prop.copy(), x.copy(), fval)
            if fval < tol:
                return best
            if np.array_equal(new_prop, prop):
                break
            prop = new_prop
    return best


def main():
    setup = build_manifold_setup()
    g = MixedGame(setup)
    print('states', g.states, 'players', g.players, '#acceptance vars', len(g.appr_keys))
    prop, x, fval = alternating_search(g, seed=0, restarts=30, sweeps=8, n_starts=8)
    print(f'\nFINAL merit M={fval:.3e}')
    V = g.solve_V(g.build_P(x, prop))
    print('V:');
    for si, s in enumerate(g.states):
        print(f'  {s}: ' + '  '.join(f'{p}={V[si,pi]:.5f}' for pi, p in enumerate(g.players)))
    print('mixing acceptance values (non 0/1):')
    for k in g.appr_keys:
        a = x[g.kpos[k]]
        if 1e-4 < a < 1 - 1e-4:
            ii, xi, yi, ki = k
            print(f'  {g.players[ii]} proposes {g.states[xi]}->{g.states[yi]}, '
                  f'{g.players[ki]} accepts w.p. {a:.4f}')
    return g, setup, prop, x, fval


if __name__ == '__main__':
    main()
