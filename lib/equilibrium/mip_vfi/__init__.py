"""
MIP-VFI equilibrium solver — public API.

Entry point: solve_with_mip_vfi(solver, params) -> (strategy_df, result_dict)

Implements the Stationary Markov Perfect Equilibrium finder described in
README JERE.md:
  - Value Function Iteration (VFI) outer loop
  - Per-state Mixed Integer Program (MIP) via scipy.milp / HiGHS
  - V-interpolation bisection for cycle / mixed-strategy resolution
  - Multi-start search for multiple equilibria

The solver integrates with the existing EquilibriumSolver interface: it reads
players, states, effectivity, payoffs, discounting, protocol, and forbidden
proposals from the solver object, then writes back strategies and returns a
strategy DataFrame in the standard format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import get_approval_committee, list_members
from lib.equilibrium.mip_vfi.vfi import run_vfi
from lib.equilibrium.mip_vfi.search import find_equilibria


# ---------------------------------------------------------------------------
# Helpers: build approval_structure and forbidden arrays
# ---------------------------------------------------------------------------

def _compute_approval_structure(solver: EquilibriumSolver) -> List[List[List[Dict]]]:
    """
    Build per-(proposer_i, from_state_s, to_state_sp) approval information:
      {
        'formula': 'product' | 'at_least_one' | 'constant_1' | 'constant_0',
        'voters':  tuple of voter player indices,
      }

    Mirrors the logic of TransitionProbabilitiesOptimized so the MIP uses the
    same approval-probability formula as the rest of the framework.
    """
    player_idx = {p: i for i, p in enumerate(solver.players)}
    players    = solver.players
    states     = solver.states

    structure = [
        [
            [None] * len(states)
            for _ in states
        ]
        for _ in players
    ]

    for i, proposer in enumerate(players):
        for s, from_state in enumerate(states):
            for sp, to_state in enumerate(states):
                approver_names = get_approval_committee(
                    solver.effectivity, players, proposer, from_state, to_state
                )

                if len(approver_names) == 0:
                    # Empty committee: self-transition or unconditional exit.
                    # Existing code returns p_approved=0 for non-self empty committees
                    # (with a warning), so we replicate that.
                    if from_state == to_state:
                        structure[i][s][sp] = {'formula': 'constant_1', 'voters': ()}
                    else:
                        structure[i][s][sp] = {'formula': 'constant_0', 'voters': ()}
                    continue

                if solver.unanimity_required or len(approver_names) == 1:
                    # Unanimity (or single voter = same formula either way)
                    voters = tuple(player_idx[m] for m in approver_names)
                    structure[i][s][sp] = {'formula': 'product', 'voters': voters}
                    continue

                # Non-unanimity with multiple voters: mirror probabilities_optimized logic
                cur_members = list_members(from_state, players)
                nxt_members = list_members(to_state, players)

                new_members              = [c for c in nxt_members if c not in cur_members]
                new_non_proposer_members = [c for c in new_members if c != proposer]

                if new_non_proposer_members:
                    # Product of acceptances of new joiners (or exiters)
                    if (proposer not in cur_members) or \
                       (proposer in cur_members and proposer in nxt_members):
                        voters = tuple(player_idx[m] for m in new_non_proposer_members)
                    else:
                        # proposer leaving their coalition
                        voters = tuple(player_idx[m] for m in nxt_members)
                    structure[i][s][sp] = {'formula': 'product', 'voters': voters}
                else:
                    # No new non-proposer members: "at least one" of the committee approves
                    voters = tuple(player_idx[m] for m in approver_names)
                    structure[i][s][sp] = {'formula': 'at_least_one', 'voters': voters}

    return structure


def _build_forbidden(solver: EquilibriumSolver) -> np.ndarray:
    """
    forbidden[proposer_i, from_state_s, to_state_sp] = True if forbidden.
    Shape: (n_players, n_states, n_states).
    Used as forbidden[i, s, sp] inside per-state MIPs (indexing forbidden[i, s]).
    """
    n_p = len(solver.players)
    n_s = len(solver.states)
    mask = np.zeros((n_p, n_s, n_s), dtype=bool)
    for i, proposer in enumerate(solver.players):
        for s, from_state in enumerate(solver.states):
            for sp, to_state in enumerate(solver.states):
                if (proposer, from_state, to_state) in solver.forbidden_proposals:
                    mask[i, s, sp] = True
    return mask


# ---------------------------------------------------------------------------
# Strategy extraction: arrays → solver dicts → strategy DataFrame
# ---------------------------------------------------------------------------

def _arrays_to_strategy_df(
    solver: EquilibriumSolver,
    all_sigmas: np.ndarray,        # (n_states, n_players, n_states)
    all_alphas: np.ndarray,        # (n_states, n_players, n_states)  [voter, next_state]
    approval_structure=None,       # optional, unused here but kept for signature clarity
) -> pd.DataFrame:
    """
    Write the MIP-VFI arrays back into solver.p_proposals and
    solver.r_acceptances, then return the strategy DataFrame.

    all_sigmas[s, i, sp] = P(proposer i proposes sp | current state is s)
    all_alphas[s, j, sp] = P(voter j accepts s->sp)   (same for all proposers)
    """
    players = solver.players
    states = solver.states

    for i, proposer in enumerate(players):
        for s, from_state in enumerate(states):
            for sp, to_state in enumerate(states):
                solver.p_proposals[(proposer, from_state, to_state)] = float(all_sigmas[s, i, sp])

    for s, from_state in enumerate(states):
        for i, proposer in enumerate(players):
            for sp, to_state in enumerate(states):
                committee = get_approval_committee(
                    solver.effectivity, players, proposer, from_state, to_state
                )
                for j_name in committee:
                    j = players.index(j_name)
                    key = (proposer, from_state, to_state, j_name)
                    solver.r_acceptances[key] = float(all_alphas[s, j, sp])

    return solver._create_strategy_dataframe()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def solve_with_mip_vfi(
    solver: EquilibriumSolver,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Find a Stationary Markov Perfect Equilibrium using VFI + per-state MIP.

    Recognised params keys:
        mip_vfi_tol          (float, default 1e-6)  VFI convergence tolerance
        mip_vfi_max_iter     (int,   default 200)   max VFI iterations
        mip_vfi_cycle_window (int,   default 8)     cycle detection window
        mip_eps              (float, default 1e-8)  acceptance indifference threshold
        mip_time_limit       (float, default 30.0)  per-MIP time limit (s)
        mip_vfi_multi_start  (bool,  default False) run multi-start search
        mip_vfi_n_restarts   (int,   default 40)    random restarts per rho variant
        mip_vfi_seed         (int,   default 0)     random seed
        mip_vfi_atol_dedup   (float, default 0.01)  dedup tolerance for multi-start

    Returns:
        (strategy_df, result_dict) matching the interface of other solver approaches.
    """
    if params is None:
        params = {}

    tol           = float(params.get("mip_vfi_tol", 1e-6))
    max_iter      = int(params.get("mip_vfi_max_iter", 300))
    cycle_window  = int(params.get("mip_vfi_cycle_window", 16))
    mip_eps       = float(params.get("mip_eps", 1e-8))
    mip_time_limit= float(params.get("mip_time_limit", 30.0))
    damping_alpha = float(params.get("mip_vfi_damping_alpha", 0.0))
    multi_start   = bool(params.get("mip_vfi_multi_start", False))
    n_restarts    = int(params.get("mip_vfi_n_restarts", 40))
    seed          = int(params.get("mip_vfi_seed", 0))
    atol_dedup    = float(params.get("mip_vfi_atol_dedup", 0.01))

    logger = getattr(solver, "logger", None)

    def _log(msg):
        if logger:
            logger.info(msg)

    players = solver.players
    states  = solver.states
    n_players = len(players)
    n_states  = len(states)

    payoffs = solver.payoffs.loc[states, players].to_numpy(dtype=np.float64)
    rho = np.array([float(solver.protocol[p]) for p in players], dtype=np.float64)
    delta = float(solver.discounting)

    approval_structure = _compute_approval_structure(solver)
    forbidden_3d       = _build_forbidden(solver)

    _log("=" * 70)
    _log("MIP-VFI Solver")
    _log(f"  players={players}, states={len(states)}, delta={delta}")
    _log(f"  tol={tol}, max_iter={max_iter}, cycle_window={cycle_window}")
    _log(f"  multi_start={multi_start}, n_restarts={n_restarts}")
    _log("=" * 70)

    if multi_start:
        # --- Multi-start mode ---
        # Per-state MIP forbidden array: forbidden[i, sp] for each state s
        # We pass the 3-d mask; inside find_equilibria / run_vfi we index [i, s, sp].
        # Convert to the shape expected by run_vfi: (n_players, n_states, n_states)
        # run_vfi receives forbidden as 3-d and slices [i, s] inside the MIP call.
        # Actually run_vfi passes the 3-d mask straight through to _mip_at_state
        # which receives it as `forbidden` and indexes forbidden[i, sp] — meaning
        # it needs the s-specific slice (n_players, n_states).
        # We fix this by wrapping _mip_at_state to pass the correct slice.
        results = find_equilibria(
            payoffs=payoffs,
            approval_structure=approval_structure,
            forbidden=forbidden_3d,
            rho_base=rho,
            delta=delta,
            n_restarts=n_restarts,
            tol=tol,
            max_iter=max_iter,
            cycle_window=cycle_window,
            mip_eps=mip_eps,
            mip_time_limit=mip_time_limit,
            atol_dedup=atol_dedup,
            seed=seed,
            logger=logger,
        )
        if not results:
            _log("Multi-start: no equilibrium found.")
            return pd.DataFrame(), {
                "converged": False,
                "stopping_reason": "multi_start_no_solution",
                "outer_iterations": 0,
                "final_tau_p": 0.0,
                "final_tau_r": 0.0,
            }

        # Use first (or best) equilibrium
        best = results[0]
        all_sigmas = best["all_sigmas"]
        all_alphas = best["all_alphas"]
        stopping   = best["stopping"]
        n_iter     = best["iterations"]
        n_found    = len(results)
        _log(f"Multi-start: {n_found} distinct equilibria; using first.")

    else:
        # --- Single run ---
        result = run_vfi(
            payoffs=payoffs,
            approval_structure=approval_structure,
            forbidden=forbidden_3d,
            rho=rho,
            delta=delta,
            tol=tol,
            max_iter=max_iter,
            cycle_window=cycle_window,
            mip_eps=mip_eps,
            mip_time_limit=mip_time_limit,
            damping_alpha=damping_alpha,
            logger=logger,
        )
        all_sigmas = result["all_sigmas"]
        all_alphas = result["all_alphas"]
        stopping   = result["stopping"]
        n_iter     = result["iterations"]
        n_found    = 1 if result["success"] else 0

    strategy_df = _arrays_to_strategy_df(solver, all_sigmas, all_alphas)

    solver_result = {
        "converged": stopping in ("converged", "cycle_resolved"),
        "stopping_reason": stopping,
        "outer_iterations": n_iter,
        "final_tau_p": 0.0,  # not applicable to MIP-VFI
        "final_tau_r": 0.0,
        "n_equilibria_found": n_found,
    }

    return strategy_df, solver_result
