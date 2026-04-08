#!/usr/bin/env python3
"""Check one reduced 4-state weak ranking exactly and benchmark it.

This is a prototype for the reduced-game shared-parameter path.
It handles:
- deterministic weak rankings exactly
- shared-parameter ansatz on nontrivial weak rankings:
  one player/group-level parameterization for indifferent choices, reused
  across all committee contexts where that equality appears

This matches the intended restricted model more closely than the temporary
finite shared-refinement detour.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.find import setup_experiment
from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import get_approval_committee
from scripts.search_ordinal_rankings import (
    _build_payoff_config,
    _build_induced_arrays_weak,
    _format_weak_order,
    _generate_weak_orders,
    _resolve_payoff_file,
    _set_canonical_weak_profile,
    _solve_values_fast_array,
    _verify_equilibrium_fast,
    _weak_equality_groups,
    _weak_value_param_count,
    _weak_tie_structure,
)
from scripts.reduced_weak_exact import (
    exact_check_deterministic_resolutions,
    shared_variable_structure,
    solve_shared_parameter_ansatz,
)


def _committee_idxs(players: list[str], states: list[str], effectivity: dict[tuple, int]) -> list[list[list[tuple[int, ...]]]]:
    player_idx = {player: idx for idx, player in enumerate(players)}
    committee_idxs: list[list[list[tuple[int, ...]]]] = []
    for proposer in players:
        proposer_rows: list[list[tuple[int, ...]]] = []
        for current_state in states:
            row: list[tuple[int, ...]] = []
            for next_state in states:
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                row.append(tuple(player_idx[p] for p in committee))
            proposer_rows.append(row)
        committee_idxs.append(proposer_rows)
    return committee_idxs


def _exact_check_no_free_vars(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    committee_idxs: list[list[list[tuple[int, ...]]]],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[bool, str, dict | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proposal_probs, approval_action, approval_pass, P_array = _build_induced_arrays_weak(
        players=players,
        tiers=tiers,
        committee_idxs=committee_idxs,
        protocol_arr=protocol_arr,
    )
    V_array = _solve_values_fast_array(P_array, payoff_array, discounting)
    verified, message, detail = _verify_equilibrium_fast(
        players=players,
        states=states,
        effectivity=effectivity,
        P_proposals=None,
        P_approvals=None,
        V_df=None,
        proposal_probs=proposal_probs,
        approval_action=approval_action,
        approval_pass=approval_pass,
        V_array=V_array,
        committee_idxs=committee_idxs,
    )
    return verified, message, detail, proposal_probs, approval_action, approval_pass, V_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one reduced-game weak ranking exactly.")
    parser.add_argument("file", help="Payoff table path or basename under payoff_tables/")
    parser.add_argument("--scenario", required=True, help="Scenario used to load effectivity/protocol")
    parser.add_argument("--perm-a", type=int, required=True, help="Weak-order id for player 1")
    parser.add_argument("--perm-b", type=int, required=True, help="Weak-order id for player 2")
    parser.add_argument("--perm-c", type=int, required=True, help="Weak-order id for player 3")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the exact check N times to benchmark runtime")
    args = parser.parse_args()

    payoff_path = _resolve_payoff_file(args.file)
    config = _build_payoff_config(args.scenario, str(payoff_path))
    setup = setup_experiment(config)
    players = setup["players"]
    states = setup["state_names"]
    if len(states) != 4 or len(players) != 3:
        raise ValueError("This prototype expects the reduced 4-state, 3-player case.")

    orders = _generate_weak_orders(len(states))
    tiers = (orders[args.perm_a], orders[args.perm_b], orders[args.perm_c])
    committee_idxs = _committee_idxs(players, states, setup["effectivity"])

    solver = EquilibriumSolver(
        players=players,
        states=states,
        effectivity=setup["effectivity"],
        protocol=setup["protocol"],
        payoffs=setup["payoffs"],
        discounting=setup["discounting"],
        unanimity_required=setup["unanimity_required"],
        verbose=False,
        random_seed=0,
        initialization_mode="uniform",
        logger=None,
    )
    _set_canonical_weak_profile(solver, players, states, tiers, committee_idxs)
    free_approvals, proposal_rows = _weak_tie_structure(players, states, tiers, committee_idxs)
    n_strategy_vars = len(free_approvals) + sum(len(winners) - 1 for _p, _c, winners in proposal_rows)
    n_value_params = _weak_value_param_count(tiers)

    print("Reduced Weak Ranking Check")
    print("-" * 80)
    print(f"file: {payoff_path.relative_to(REPO_ROOT)}")
    print(f"players: {players}")
    print(f"states: {states}")
    print(f"perm_a/perm_b/perm_c: {args.perm_a}, {args.perm_b}, {args.perm_c}")
    for player, tier in zip(players, tiers):
        print(f"{player}: {_format_weak_order(states, tier)}")
    equality_groups = _weak_equality_groups(states, tiers)
    shared_structure = shared_variable_structure(
        players=players,
        states=states,
        tiers=tiers,
        effectivity=setup["effectivity"],
    )
    print("equality_groups:")
    for player, groups in zip(players, equality_groups):
        print(f"  {player}: {groups if groups else 'none'}")
    print(f"value_params: {n_value_params}")
    print(f"strategy_free_approval_vars: {len(free_approvals)}")
    print(f"strategy_free_proposal_rows: {len(proposal_rows)}")
    print(f"strategy_total_free_vars: {n_strategy_vars}")
    print(f"shared_group_params: {shared_structure['total_group_params']}")
    print(f"shared_approval_contexts: {len(shared_structure['approval_contexts'])}")
    print(f"shared_proposal_contexts: {len(shared_structure['proposal_contexts'])}")

    protocol_arr = np.array([float(setup["protocol"][player]) for player in players], dtype=np.float64)
    payoff_array = setup["payoffs"].loc[states, players].to_numpy(dtype=np.float64)

    start = time.perf_counter()
    verified = False
    message = ""
    detail = None
    last_V = None
    deterministic_resolution = None
    shared_resolution = None
    for _ in range(max(1, int(args.repeat))):
        if n_strategy_vars == 0:
            verified, message, detail, _proposal_probs, _approval_action, _approval_pass, last_V = _exact_check_no_free_vars(
                players=players,
                states=states,
                effectivity=setup["effectivity"],
                protocol_arr=protocol_arr,
                payoff_array=payoff_array,
                discounting=setup["discounting"],
                committee_idxs=committee_idxs,
                tiers=tiers,
            )
            deterministic_resolution = None
        else:
            deterministic_resolution = exact_check_deterministic_resolutions(
                players=players,
                states=states,
                effectivity=setup["effectivity"],
                protocol_arr=protocol_arr,
                payoff_array=payoff_array,
                discounting=setup["discounting"],
                tiers=tiers,
            )
            if deterministic_resolution is not None and "P_array" in deterministic_resolution:
                verified = True
                message = deterministic_resolution["message"]
                detail = deterministic_resolution["detail"]
                last_V = deterministic_resolution["V_array"]
                shared_resolution = None
            else:
                shared_resolution = solve_shared_parameter_ansatz(
                    players=players,
                    states=states,
                    effectivity=setup["effectivity"],
                    protocol_arr=protocol_arr,
                    payoff_array=payoff_array,
                    discounting=setup["discounting"],
                    tiers=tiers,
                )
                verified = shared_resolution is not None
                message = shared_resolution["message"] if shared_resolution is not None else (
                    deterministic_resolution["message"] if deterministic_resolution is not None else "no resolution verified"
                )
                detail = shared_resolution["detail"] if shared_resolution is not None else (
                    deterministic_resolution["detail"] if deterministic_resolution is not None else None
                )
                last_V = None if shared_resolution is None else shared_resolution["V_array"]
    elapsed = time.perf_counter() - start

    print()
    print("Result")
    print("-" * 80)
    print(f"verified: {verified}")
    print(f"message: {message}")
    print(f"repeat: {args.repeat}")
    print(f"avg_ms: {1000.0 * elapsed / max(1, int(args.repeat)):.6f}")
    if detail is not None:
        print(f"detail: {detail}")
    if deterministic_resolution is not None:
        print(f"candidate_count: {deterministic_resolution['candidate_count']}")
    if shared_resolution is not None:
        print(f"shared_params: {shared_resolution['params']}")
        print(f"shared_nfev: {shared_resolution['nfev']}")
    if last_V is not None:
        print("V:")
        for state_idx, state in enumerate(states):
            vals = "  ".join(f"{players[p_idx]}={last_V[state_idx, p_idx]:.6f}" for p_idx in range(len(players)))
            print(f"  {state}: {vals}")


if __name__ == "__main__":
    main()
