"""A continuous residual metric for the equilibrium fixed-point.

Conceptual core
---------------
The verifier checks two BINARY conditions on a strategy profile sigma given its
induced value matrix V. We do NOT search sigma directly (the induced-strategy
map is discontinuous at ordinal ties). Instead we treat the VALUE MATRIX V as
the search object and define a continuous residual:

    sigma_hat(V)  : the deterministic rational strategy induced by V
    P(V)          : transition matrix from sigma_hat(V)   (polynomial, smooth in sigma)
    Vprime(V)     : (I - delta P)^-1 (1-delta) u          (Bellman re-solve)
    F(V)          = Vprime(V) - V                          (value-space residual)

An equilibrium is exactly a fixed point F(V)=0, and there the verifier passes.
F is piecewise-affine: constant ordinal cell -> constant sigma -> constant P ->
affine Vprime. So ||F(V)||^2 is a continuous (piecewise-smooth) objective whose
global minima (value 0) are the equilibria.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from lib.equilibrium.scenarios import get_scenario, fill_players
from lib.equilibrium.find import setup_experiment, _compute_verification, _parse_players_from_payoff_table
from lib.utils import get_approval_committee
from lib.probabilities import TransitionProbabilities


def build_setup(scenario, payoff_table, effectivity_rule):
    cfg = get_scenario(scenario)
    cfg['payoff_table'] = str(payoff_table)
    cfg['effectivity_rule'] = effectivity_rule
    players = _parse_players_from_payoff_table(Path(payoff_table))
    cfg = fill_players(cfg, players)
    setup = setup_experiment(cfg)
    return setup


def empty_strategy_df(players, states):
    columns = pd.MultiIndex.from_product(
        [[f'Proposer {p}' for p in players], states], names=['Proposer', 'Next State'])
    rows = []
    for s in states:
        rows.append((s, 'Proposition', np.nan))
        for p in players:
            rows.append((s, 'Acceptance', p))
    index = pd.MultiIndex.from_tuples(rows, names=['Current State', 'Type', 'Player'])
    df = pd.DataFrame(np.nan, index=index, columns=columns)
    return df.sort_index(axis=0).sort_index(axis=1)


def induced_strategy_df(V, setup, tie_tol=1e-12):
    """Deterministic rational strategy induced by value matrix V (DataFrame states x players).

    Approvals: 1 if V_k(next) > V_k(cur)+tol, 0 if < -tol, else 0.5 (tie -> mix allowed).
    Proposals: proposer picks argmax over feasible next states of
               p_app * V_i(next) + (1-p_app) * V_i(cur), where p_app is the
               committee-approval probability under these approval choices.
    """
    players, states = setup['players'], setup['state_names']
    eff, protocol = setup['effectivity'], setup['protocol']
    forbidden = setup.get('forbidden_proposals', frozenset())
    df = empty_strategy_df(players, states)

    # 1) approvals
    for proposer in players:
        for cur in states:
            for nxt in states:
                committee = get_approval_committee(eff, players, proposer, cur, nxt)
                for k in committee:
                    diff = V.loc[nxt, k] - V.loc[cur, k]
                    if diff > tie_tol:
                        a = 1.0
                    elif diff < -tie_tol:
                        a = 0.0
                    else:
                        a = 0.0  # break ties toward reject (deterministic pick)
                    df.loc[(cur, 'Acceptance', k), (f'Proposer {proposer}', nxt)] = a

    # need committee-approval probability for proposal step; reuse TransitionProbabilities
    # logic by reading the just-set acceptance entries through a temp TP for p_approved.
    # P_approvals is independent of proposals, but safety_checks needs valid proposal
    # rows -> set a temporary self-loop proposal for each (proposer, state).
    df_filled = df.fillna(0.0)
    for proposer in players:
        for cur in states:
            df_filled.loc[(cur, 'Proposition', np.nan), (f'Proposer {proposer}', cur)] = 1.0
    tp = TransitionProbabilities(df=df_filled, effectivity=eff, players=players,
                                 states=states, protocol=protocol,
                                 unanimity_required=setup['unanimity_required'])
    _, _, P_approvals = tp.get_probabilities()

    # 2) proposals: argmax of expected value
    for proposer in players:
        for cur in states:
            best_val, best_state = -np.inf, None
            for nxt in states:
                if (proposer, cur, nxt) in forbidden:
                    continue
                pa = P_approvals[(proposer, cur, nxt)]
                ev = pa * V.loc[nxt, proposer] + (1 - pa) * V.loc[cur, proposer]
                if ev > best_val + tie_tol:
                    best_val, best_state = ev, nxt
            for nxt in states:
                df.loc[(cur, 'Proposition', np.nan), (f'Proposer {proposer}', nxt)] = \
                    1.0 if nxt == best_state else 0.0
    return df


def value_of_strategy(df, setup):
    V, P, P_proposals, P_approvals = _compute_verification(df.fillna(0.0), setup)
    return V.astype(float), P, P_proposals, P_approvals


def residual(V, setup):
    """F(V) = Vprime(induced_strategy(V)) - V; returns (||F||, Vprime, df)."""
    df = induced_strategy_df(V, setup)
    Vp, P, _, _ = value_of_strategy(df, setup)
    F = Vp.values.astype(float) - V.values.astype(float)
    return float(np.linalg.norm(F)), Vp, df
