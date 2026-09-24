"""
Jere's SMPE solver (logit / quantal-response homotopy), vendored.

Upstream: coalition-game repo, commit ec55bf2 (2026-09-22), files
smpe.py, smpe_hard.py, smpe_sweep.py, check_equilibrium.py,
certify_equilibrium.py.  His README.md / TECHNICAL.md describe the method.
Local changes are marked "[farsighted-coalitions]" and are listed in UPSTREAM.md.

Pipeline per delta: policy iteration -> (if it cycles) logit homotopy to
identify which decisions mix -> exact solve of the indifference conditions ->
verification.  `solve_with_smpe` below is the adapter to find_equilibrium.

Committees come from OUR effectivity, not Jere's rule
-----------------------------------------------------
Jere hard-codes "everyone whose coalition changes, except the proposer; a
unilateral exit needs no vote".  That equals heyen_lehtomaa_2021 except on
own exits, but differs from unanimous_consent / free_exit / deployer_exit and
ignores adjacent_step's forbidden proposals.  So the adapter builds the
committees from `solver.effectivity` and blocks `solver.forbidden_proposals`.

Two translations are needed, both exact:

* Proposer on its own committee (heyen own exits, unanimous_consent, ...).
  Jere's model forbids it; we drop the proposer from the committee inside the
  solver and set the proposer's own vote on output by the cutoff rule.  This
  is exact: the proposer's own vote must follow sign(V_i(t) - V_i(s)).  A gain
  -> vote 1, the move passes as without the vote.  A loss -> vote 0, and the
  proposal is identical to staying; without the vote, the proposal has gain
  q*(V_i(t)-V_i(s)) < 0 and is never made.  A tie -> any vote is legal; we
  output 1, which reproduces the solver's q.  Our strategy table stores votes
  per proposer, so this override touches only the proposer's own column.

* Forbidden proposals.  Encoded as q = 0 (geom.blocked), i.e. as a proposal
  certain to be rejected, which is identical to staying and which the solver
  already folds into "stay".  On output any weight on a forbidden target is
  moved onto the status quo, so our verifier sees sigma = 0 there.

Jere's votes do not depend on the proposer; ours may.  That only narrows the
search: whatever it finds is an equilibrium of our game.

Not supported (raise, never approximate): majority approval
(unanimity_required=False), empty committees on a real move.

His `check_equilibrium.py` / `certify_equilibrium.py` re-derive committees
from HIS rule, so their verdicts are meaningful only for effectivity rules
equivalent to it (heyen_lehtomaa_2021).  Our verify_equilibrium is the arbiter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .smpe import CoalitionGeometry, Game, Solution
from .smpe_sweep import solve_point, sweep


# Own-vote tie threshold, in the solver's rescaled units (each player's payoffs
# have unit max absolute deviation there).  Exact ties in the solver's V come
# out at ~1e-16; its own verification tolerance is 1e-7.
OWN_VOTE_TIE_TOL = 1e-9


@dataclass
class Committees:
    voters: np.ndarray      # (S,S,N,N) bool, Jere state order, proposer excluded
    blocked: np.ndarray     # (S,S,N)   bool
    source: str


def _partition_of(name: str, players: List[str]) -> frozenset:
    from lib.equilibrium.jeres_vfi.game import fw_state_name_to_partition
    return frozenset(
        frozenset(players.index(i) if isinstance(i, str) else int(i) for i in b)
        for b in fw_state_name_to_partition(name, players)
    )


def state_maps(players: List[str], fw_states: List[str]
               ) -> Tuple[CoalitionGeometry, Dict[int, int], Dict[int, int]]:
    """(geometry, fw index -> Jere index, Jere index -> fw index)."""
    geom = CoalitionGeometry(players)
    jidx = {frozenset(frozenset(b) for b in st): k for k, st in enumerate(geom.states)}
    fw_to_j, j_to_fw = {}, {}
    for f, name in enumerate(fw_states):
        part = _partition_of(name, players)
        if part not in jidx:
            raise ValueError(f"framework state {name!r} is not a coalition structure "
                             f"over {players}")
        fw_to_j[f] = jidx[part]
        j_to_fw[jidx[part]] = f
    if len(fw_to_j) != geom.S:
        raise ValueError(f"framework has {len(fw_to_j)} states, the game has {geom.S}; "
                         "the SMPE solver needs the full set of coalition structures")
    return geom, fw_to_j, j_to_fw


def build_committees(players, fw_states, effectivity, forbidden_proposals,
                     fw_to_j, rule_name: str) -> Tuple[Committees, List[Tuple]]:
    """Committees in Jere's state order from our effectivity.

    Returns (Committees, own_votes) where own_votes lists the
    (proposer, fw_s, fw_t) at which the proposer sits on its own committee.
    """
    from lib.utils import get_approval_committee

    N, S = len(players), len(fw_states)
    voters = np.zeros((S, S, N, N), dtype=bool)
    blocked = np.zeros((S, S, N), dtype=bool)
    own_votes = []
    for i, proposer in enumerate(players):
        for fs, s_name in enumerate(fw_states):
            for ft, t_name in enumerate(fw_states):
                if fs == ft:
                    continue
                s, t = fw_to_j[fs], fw_to_j[ft]
                if (proposer, s_name, t_name) in forbidden_proposals:
                    blocked[s, t, i] = True
                    continue
                comm = get_approval_committee(effectivity, players, proposer,
                                              s_name, t_name)
                if not comm:
                    raise ValueError(
                        f"empty approval committee for {proposer}: {s_name} -> {t_name} "
                        f"under {rule_name}; the framework treats this as a bug in the "
                        "effectivity rule, and so does this adapter")
                for voter in comm:
                    j = players.index(voter)
                    if j == i:
                        own_votes.append((proposer, s_name, t_name))
                    else:
                        voters[s, t, i, j] = True
    return Committees(voters, blocked, rule_name), own_votes


def _payoff_matrix(solver, fw_to_j) -> np.ndarray:
    S, N = len(solver.states), len(solver.players)
    P = np.zeros((S, N))
    for f, name in enumerate(solver.states):
        P[fw_to_j[f]] = solver.payoffs.loc[name, solver.players].values.astype(float)
    return P


def solution_to_strategy_df(solver, sol: Solution, fw_to_j, own_votes):
    """Convert a Jere Solution into the framework strategy DataFrame."""
    from lib.equilibrium.mip_vfi import _arrays_to_strategy_df

    players, states = solver.players, solver.states
    S, N = len(states), len(players)
    sig = np.zeros((S, N, S))
    alp = np.zeros((S, N, S))
    for fs in range(S):
        for ft in range(S):
            s, t = fw_to_j[fs], fw_to_j[ft]
            sig[fs, :, ft] = sol.sigma[:, s, t]
            alp[fs, :, ft] = sol.alpha[:, s, t]
    # forbidden targets: weight belongs to the status quo (identical transition)
    forbidden = solver.forbidden_proposals
    for i, p in enumerate(players):
        for fs, s_name in enumerate(states):
            for ft, t_name in enumerate(states):
                if fs != ft and (p, s_name, t_name) in forbidden and sig[fs, i, ft] > 0:
                    sig[fs, i, fs] += sig[fs, i, ft]
                    sig[fs, i, ft] = 0.0
    df = _arrays_to_strategy_df(solver, sig, alp)
    # proposer's own vote: cutoff rule on the solver's V (see module docstring)
    fw_pos = {name: k for k, name in enumerate(states)}
    for proposer, s_name, t_name in own_votes:
        i = players.index(proposer)
        s, t = fw_to_j[fw_pos[s_name]], fw_to_j[fw_pos[t_name]]
        gain = sol.V[t, i] - sol.V[s, i]
        a = 1.0 if gain >= -OWN_VOTE_TIE_TOL else 0.0
        df.loc[(s_name, "Acceptance", proposer), (f"Proposer {proposer}", t_name)] = a
        solver.r_acceptances[(proposer, s_name, t_name, proposer)] = a
    return df


def solve_with_smpe(solver, params: Optional[Dict[str, Any]] = None):
    """Find an SMPE with Jere's homotopy solver.  Interface as the other
    `solve_with_*` adapters: returns (strategy_df, result_dict), or
    (None, result_dict) if no profile verified.

    Recognised params
    -----------------
    smpe_budget          (float, 15)   seconds per homotopy solve
    smpe_fallback_budget (float, 5)    seconds for the legacy support search
    smpe_anchor_deltas   (list, None)  if given, run Jere's sweep over these
                                       deltas plus the target and return the
                                       target point: branch following and gap
                                       filling from neighbouring deltas.
    smpe_walk_step       (float, 0.005) continuation step for the sweep
    """
    params = params or {}
    if not solver.unanimity_required:
        raise ValueError("solver_approach='smpe' supports unanimous approval only "
                         "(unanimity_required=True); majority approval is not in "
                         "Jere's model")
    players, states = list(solver.players), list(solver.states)
    delta = float(solver.discounting)
    rho = [float(solver.protocol[p]) for p in players]
    rule = getattr(solver, "effectivity_rule", "unknown")

    geom, fw_to_j, _ = state_maps(players, states)
    committees, own_votes = build_committees(
        players, states, solver.effectivity, solver.forbidden_proposals,
        fw_to_j, rule)
    payoffs = _payoff_matrix(solver, fw_to_j)

    budget = float(params.get("smpe_budget", 15.0))
    anchors = params.get("smpe_anchor_deltas")
    method, extra = None, {}
    if anchors:
        grid = sorted(set(float(d) for d in anchors) | {delta})
        res = sweep(players, payoffs, deltas=grid, rho=rho, budget=budget,
                    walk_step=float(params.get("smpe_walk_step", 0.005)),
                    committees=committees)
        point = next(p for p in res.points if abs(p.delta - delta) < 1e-12)
        sol, method = point.solution, point.method
        extra = dict(branch_change=point.branch_change,
                     change_interval=point.change_at,
                     anchor_differs=point.anchor_differs,
                     sweep_deltas=grid)
    else:
        sol, method = solve_point(
            players, payoffs, delta, rho=rho, budget=budget,
            fallback_budget=float(params.get("smpe_fallback_budget", 5.0)),
            committees=committees)

    base = dict(outer_iterations=0, final_tau_p=0.0, final_tau_r=0.0,
                committee_source=rule, n_own_votes=len(own_votes),
                n_blocked=int(committees.blocked.sum()), **extra)
    if sol is None:
        return None, dict(converged=False, stopping_reason="smpe_unsolved",
                          smpe_method=method, n_equilibria_found=0, **base)

    df = solution_to_strategy_df(solver, sol, fw_to_j, own_votes)
    return df, dict(
        converged=True, stopping_reason=f"smpe_{method}", smpe_method=method,
        smpe_status=sol.status, smpe_route=sol.route,
        smpe_max_proposal_regret=float(sol.report.max_proposal_regret),
        smpe_max_accept_violation=float(sol.report.max_accept_violation),
        n_equilibria_found=1, **base)


__all__ = ["solve_with_smpe", "build_committees", "state_maps", "Committees",
           "solution_to_strategy_df", "sweep", "solve_point", "OWN_VOTE_TIE_TOL"]
