"""Targeted interior-mixing solve on a full 5-state n=3 hard table.

Reimplements the shared-parameter ansatz idea (reduced_weak_exact.py, which is
broken in the current tree) self-contained on FastGame:

  1. Get an informed value matrix V from a merit-descent plateau.
  2. Turn V into a per-player WEAK RANKING (states within eps -> same tier),
     i.e. a guess of the indifference structure.
  3. Given that ranking: fix every STRICT acceptance to 0/1; the only free
     variables are the acceptance probs on TIED transitions (the mixing). The
     proposal structure is pinned by best-response, so there is no proposal-
     support search (the bottleneck for the generic mixed solver).
  4. Solve the few free mixing probs with L-BFGS-B (merit -> 0), alternating
     with best-response proposals; verify with the real verifier.
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.residual_metric_probe import build_setup, value_of_strategy, empty_strategy_df
from scripts.fast_merit import plateau_search
from lib.equilibrium.merit_descent import _FastGame
from lib.utils import verify_equilibrium


def make_fastgame(setup):
    return _FastGame(
        players=setup['players'], states=setup['state_names'],
        effectivity=setup['effectivity'], protocol=setup['protocol'],
        payoffs=setup['payoffs'], discounting=setup['discounting'],
        forbidden_proposals=setup.get('forbidden_proposals', frozenset()))


def tiers_from_V(V, n_states, eps):
    tiers = []
    for p in range(V.shape[1]):
        vals = V[:, p]
        order = np.argsort(-vals)
        tier = np.zeros(n_states, dtype=np.int64)
        cur = 0
        for k in range(1, n_states):
            if vals[order[k-1]] - vals[order[k]] > eps:
                cur += 1
            tier[order[k]] = cur
        tiers.append(tier)
    return tiers


def classify_acceptances(fg, tiers):
    """Return (fixed_vals dict pos->{0,1}, free_positions list) for appr_keys."""
    kpos = fg._kpos()
    fixed, free = {}, []
    for k in fg.appr_keys:
        ii, xi, yi, ki = k
        pos = kpos[k]
        if xi == yi:
            fixed[pos] = 1.0
            continue
        tk = tiers[ki]
        if tk[yi] < tk[xi]:      # strictly prefers next -> approve
            fixed[pos] = 1.0
        elif tk[yi] > tk[xi]:    # strictly worse -> reject
            fixed[pos] = 0.0
        else:                    # indifferent -> free mixing variable
            free.append(pos)
    return kpos, fixed, free


def make_avec(fg, fixed, free, params):
    avec = np.zeros(len(fg.appr_keys))
    for pos, v in fixed.items():
        avec[pos] = v
    for j, pos in enumerate(free):
        avec[pos] = params[j]
    return avec


def solve_candidate(fg, tiers, n_starts=8, sweeps=10, seed=0):
    kpos, fixed, free = classify_acceptances(fg, tiers)
    if not free:
        return None, 0, len(free)   # no mixing -> pure (already known to fail)
    rng = np.random.RandomState(seed)
    m = len(free)
    bounds = [(0.0, 1.0)] * m
    # init proposals: best-response under acceptances at params=0.5, starting
    # from self-loop proposals to seed the value computation.
    avec0 = make_avec(fg, fixed, free, np.full(m, 0.5))
    self_prop = np.tile(np.arange(fg.S), (fg.n, 1))   # each (i,x) proposes x
    V0 = fg.solve_V(fg.build_P_cont(avec0, kpos, self_prop))
    prop = fg.best_response_proposals(avec0, kpos, V0)
    best = None
    for _sw in range(sweeps):
        for _s in range(n_starts):
            x0 = rng.uniform(0, 1, m)
            res = minimize(lambda q: fg.merit_cont(make_avec(fg, fixed, free, q), kpos, prop),
                           x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12})
            if best is None or res.fun < best[1]:
                best = (res.x.copy(), res.fun)
            if best[1] < 1e-10:
                break
        avec = make_avec(fg, fixed, free, best[0])
        V = fg.solve_V(fg.build_P_cont(avec, kpos, prop))
        new_prop = fg.best_response_proposals(avec, kpos, V)
        if best[1] < 1e-10 or np.array_equal(new_prop, prop):
            break
        prop = new_prop
    avec = make_avec(fg, fixed, free, best[0])
    return (avec, kpos, prop, best[1]), best[1], m


def run(payoff, scenario='power_threshold_RICE_n3', rule='adjacent_step',
        merit_restarts=80, seed=0):
    setup = build_setup(scenario, ROOT / 'payoff_tables' / f'{payoff}.xlsx', rule)
    players, states = setup['players'], setup['state_names']
    n = len(states)
    fg = make_fastgame(setup)
    print(f'[{payoff}] players={players} states={states}', flush=True)

    (a, p), m_pl, ok = plateau_search(fg, seed=seed, restarts=merit_restarts, walk=3000)[:3]
    Vp = fg.solve_V(fg.build_P(a, p))
    print(f'  merit plateau M={m_pl:.3e}', flush=True)

    seen = set()
    t0 = time.time()
    for eps in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
        tiers = tiers_from_V(Vp, n, eps)
        key = tuple(tuple(int(x) for x in ti) for ti in tiers)
        if key in seen:
            continue
        seen.add(key)
        ts = time.time()
        out, mbest, nfree = solve_candidate(fg, tiers, seed=seed)
        tag = f'M={mbest:.2e}' if out else 'no-ties(pure)'
        print(f'  eps={eps:<6} free_mix_vars={nfree:<3} -> {tag}  ({time.time()-ts:.1f}s)', flush=True)
        if out and mbest < 1e-9:
            avec, kpos, prop, _ = out
            # build df and verify with real verifier
            df = empty_strategy_df(players, states)
            for ii in range(fg.n):
                for xi in range(fg.S):
                    ch = int(prop[ii, xi])
                    for yi in range(fg.S):
                        df.loc[(states[xi], 'Proposition', np.nan), (f'Proposer {players[ii]}', states[yi])] = 1.0 if yi == ch else 0.0
            for k in fg.appr_keys:
                ii, xi, yi, ki = k
                df.loc[(states[xi], 'Acceptance', players[ki]), (f'Proposer {players[ii]}', states[yi])] = float(avec[kpos[k]])
            V, _, pp, pa = value_of_strategy(df, setup)
            vres = {'players': players, 'state_names': states, 'V': V.astype(float),
                    'P_proposals': pp, 'P_approvals': pa, 'effectivity': setup['effectivity'],
                    'strategy_df': df.fillna(0.0), 'forbidden_proposals': setup.get('forbidden_proposals', frozenset())}
            okv, msg = verify_equilibrium(vres, atol=1e-6)
            print(f'  ===> candidate M~0; verify_equilibrium={okv}', flush=True)
            if okv:
                out_f = ROOT / 'strategy_tables' / f'ansatz_{payoff}_{rule}.xlsx'
                df.to_excel(out_f)
                print(f'  SOLVED & SAVED -> {out_f}', flush=True)
                print(V.to_string(float_format=lambda z: f'{z:.5f}'), flush=True)
                return True
    print(f'  no candidate verified  (total {time.time()-t0:.1f}s)', flush=True)
    return False


if __name__ == '__main__':
    run(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
