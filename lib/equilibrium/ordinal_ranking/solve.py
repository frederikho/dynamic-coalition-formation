"""Public entry: exhaustive / sampled search over ordinal value orders."""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from lib.equilibrium.ordinal_ranking.constants import LARGE_PERM_THRESHOLD
from lib.equilibrium.ordinal_ranking.induced_strategies import (
    _build_induced_arrays,
    _build_induced_arrays_weak_from_ids,
    _build_transition_matrix,
    _induce_profile_from_rankings,
    _induce_profile_from_weak_orders,
)
from lib.equilibrium.ordinal_ranking.progress import _print_progress
from lib.equilibrium.ordinal_ranking.ranking_orders import (
    _generate_weak_orders,
    _payoff_ordering,
    _payoff_ordering_weak,
)
from lib.equilibrium.ordinal_ranking.search import (
    _init_worker_ctx,
    _iter_batches,
    _iter_batches_large,
    _iter_rank_combos_large,
    _iter_tuples,
    _search_chunk,
    _search_chunk_large,
)
from lib.equilibrium.ordinal_ranking.value_mdp import _solve_values, _verify_fast
from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import get_approval_committee


def solve_with_ordinal_ranking_n3(
    solver: EquilibriumSolver,
    *,
    max_combinations: int | None = None,
    shuffle: bool = False,
    random_seed: int = 0,
    ranking_order: str = "lexicographic",
    progress_every: int = 0,
    workers: int = 8,
    batch_size: int = 20000,
    weak_orders: bool = False,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    players = solver.players
    states = solver.states
    n_players = len(players)
    if n_players < 2:
        raise ValueError("solver_approach='ordinal_ranking' expects at least 2 players.")
    if len(states) < 2:
        raise ValueError("solver_approach='ordinal_ranking' expects at least 2 states.")

    effectivity = solver.effectivity
    n_states = len(states)

    # Decide whether to enumerate all permutations or sample randomly.
    # Materialising n_states! arrays is infeasible for large n_states.
    perm_count = math.factorial(n_states)
    large_mode = not weak_orders and perm_count > LARGE_PERM_THRESHOLD

    if large_mode:
        if ranking_order not in ("payoff", "lexicographic"):
            raise ValueError(f"Unknown ordinal_ranking_order='{ranking_order}'. Expected 'lexicographic' or 'payoff'.")
        state_perms = None
        n_perms = None
        pos = None
        approve_lookup = None
        perm_orders = None
    elif weak_orders:
        state_perms = _generate_weak_orders(n_states)
        n_perms = state_perms.shape[0]
        pos = state_perms.copy()
        approve_lookup = pos[:, None, :] <= pos[:, :, None]
        perm_orders = None
        if ranking_order == "payoff":
            payoff_array_tmp = solver.payoffs.loc[states, players].to_numpy(dtype=np.float64)
            perm_orders = _payoff_ordering_weak(payoff_array_tmp, state_perms, players)
        elif ranking_order != "lexicographic":
            raise ValueError(f"Unknown ordinal_ranking_order='{ranking_order}'. Expected 'lexicographic' or 'payoff'.")
    else:
        state_perms = np.array(list(itertools.permutations(range(n_states))), dtype=np.int8)
        n_perms = state_perms.shape[0]
        pos = np.empty((n_perms, n_states), dtype=np.int8)
        for perm_idx in range(n_perms):
            for rank, state_idx in enumerate(state_perms[perm_idx]):
                pos[perm_idx, state_idx] = rank
        approve_lookup = None
        perm_orders = None
        if ranking_order == "payoff":
            payoff_array_tmp = solver.payoffs.loc[states, players].to_numpy(dtype=np.float64)
            perm_orders = _payoff_ordering(payoff_array_tmp, state_perms, players)
        elif ranking_order != "lexicographic":
            raise ValueError(f"Unknown ordinal_ranking_order='{ranking_order}'. Expected 'lexicographic' or 'payoff'.")

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

    payoff_array = solver.payoffs.loc[states, players].to_numpy(dtype=np.float64)
    protocol_arr = np.array([float(solver.protocol[player]) for player in players], dtype=np.float64)

    if large_mode:
        flat_total = perm_count ** n_players  # true total — astronomically large but displayable
        total = max_combinations  # None means run until interrupted
    else:
        flat_total = n_perms ** n_players
        total = min(flat_total, max_combinations) if max_combinations is not None else flat_total
    start_time = time.perf_counter()
    tested = 0
    interrupted = False
    first_success = None

    try:
        if large_mode:
            # Large-state mode: sample random orderings without materialising state_perms.
            # Parallel workers get batches of rank arrays (shape B×n_players×n_states).
            combos_iter = _iter_rank_combos_large(
                n_players=n_players,
                n_states=n_states,
                total=total,
                random_seed=random_seed,
                ranking_order=ranking_order,
                payoff_array=payoff_array,
            )
            if workers > 1:
                batches_large = _iter_batches_large(combos_iter, batch_size=max(1, int(batch_size)))
                max_workers = max(1, int(workers))
                max_in_flight = max_workers * 2
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker_ctx,
                    initargs=(
                        players, states, committee_idxs, protocol_arr, payoff_array,
                        solver.discounting, None, None, False, None,
                    ),
                ) as executor:
                    pending = {}

                    def submit_one_large() -> bool:
                        try:
                            batch = next(batches_large)
                        except StopIteration:
                            return False
                        future = executor.submit(_search_chunk_large, batch)
                        pending[future] = len(batch)
                        return True

                    for _ in range(max_in_flight):
                        if not submit_one_large():
                            break

                    while pending:
                        future = next(as_completed(pending))
                        pending.pop(future)
                        chunk_result = future.result()
                        tested += int(chunk_result["tested"])
                        if progress_every > 0:
                            _print_progress(tested, flat_total, start_time)
                        if chunk_result["success"] is not None:
                            first_success = chunk_result["success"]
                            for other in list(pending):
                                other.cancel()
                            pending.clear()
                            break
                        submit_one_large()
            else:
                for ranks in combos_iter:
                    if total is not None and tested >= total:
                        break
                    proposal_choice, approval_action, approval_pass = _build_induced_arrays(
                        players=players,
                        ranks=ranks,
                        committee_idxs=committee_idxs,
                        protocol_arr=protocol_arr,
                    )
                    P_array = _build_transition_matrix(proposal_choice, protocol_arr, n_states)
                    V_array = _solve_values(P_array, payoff_array, solver.discounting)
                    verified, message = _verify_fast(
                        players=players,
                        states=states,
                        V_array=V_array,
                        proposal_choice=proposal_choice,
                        approval_action=approval_action,
                        approval_pass=approval_pass,
                        committee_idxs=committee_idxs,
                    )
                    tested += 1
                    if verified:
                        first_success = {"ranks": ranks, "message": message}
                        break
                    if progress_every > 0 and tested % progress_every == 0:
                        _print_progress(tested, flat_total, start_time)
        elif workers > 1:
            triples_iter = _iter_tuples(
                n_players=n_players,
                n_perms=n_perms,
                total=total,
                shuffle=shuffle,
                random_seed=random_seed,
                perm_orders=perm_orders,
            )
            batches = _iter_batches(triples_iter, batch_size=max(1, int(batch_size)))
            max_workers = max(1, int(workers))
            max_in_flight = max_workers * 2
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker_ctx,
                initargs=(
                    players,
                    states,
                    committee_idxs,
                    protocol_arr,
                    payoff_array,
                    solver.discounting,
                    pos,
                    state_perms,
                    weak_orders,
                    approve_lookup,
                ),
            ) as executor:
                pending = {}

                def submit_one() -> bool:
                    try:
                        batch = next(batches)
                    except StopIteration:
                        return False
                    future = executor.submit(_search_chunk, batch)
                    pending[future] = len(batch)
                    return True

                for _ in range(max_in_flight):
                    if not submit_one():
                        break

                while pending:
                    future = next(as_completed(pending))
                    pending.pop(future)
                    chunk_result = future.result()
                    tested += int(chunk_result["tested"])
                    if progress_every > 0:
                        _print_progress(min(tested, total), total, start_time)
                    if chunk_result["success"] is not None:
                        first_success = chunk_result["success"]
                        for other in list(pending):
                            other.cancel()
                        pending.clear()
                        break
                    submit_one()
        else:
            for perm_tuple in _iter_tuples(
                n_players=n_players,
                n_perms=n_perms,
                total=total,
                shuffle=shuffle,
                random_seed=random_seed,
                perm_orders=perm_orders,
            ):
                if tested >= total:
                    break
                if weak_orders:
                    proposal_probs, approval_action, approval_pass = _build_induced_arrays_weak_from_ids(
                        perm_tuple,
                        pos,
                        approve_lookup,
                        committee_idxs,
                    )
                    proposal_choice = None
                    P_array = _build_transition_matrix(None, protocol_arr, n_states, proposal_probs=proposal_probs)
                else:
                    orders = tuple(pos[idx] for idx in perm_tuple)
                    proposal_choice, approval_action, approval_pass = _build_induced_arrays(
                        players=players,
                        ranks=orders,
                        committee_idxs=committee_idxs,
                        protocol_arr=protocol_arr,
                    )
                    proposal_probs = None
                    P_array = _build_transition_matrix(proposal_choice, protocol_arr, n_states)
                V_array = _solve_values(P_array, payoff_array, solver.discounting)
                verified, message = _verify_fast(
                    players=players,
                    states=states,
                    V_array=V_array,
                    proposal_choice=proposal_choice,
                    approval_action=approval_action,
                    approval_pass=approval_pass,
                    committee_idxs=committee_idxs,
                    proposal_probs=proposal_probs,
                )
                tested += 1
                if verified:
                    first_success = {
                        "perms": perm_tuple,
                        "rankings": tuple(state_perms[idx].copy() for idx in perm_tuple),
                        "message": message,
                    }
                    break
                if progress_every > 0 and tested % progress_every == 0:
                    _print_progress(tested, total, start_time)
    except KeyboardInterrupt:
        interrupted = True

    if progress_every > 0 and tested:
        _print_progress(tested, flat_total, start_time)
        print()

    elapsed = time.perf_counter() - start_time
    if first_success is None:
        strategy_df = solver._create_strategy_dataframe()
        solver_result = {
            "converged": False,
            "stopping_reason": "ordinal_ranking_exhausted" if (not interrupted and total is not None) else "interrupted",
            "tested_combinations": tested,
            "total_combinations": total,
            "runtime_seconds": elapsed,
            "final_tau_p": 0.0,
            "final_tau_r": 0.0,
            "outer_iterations": tested,
        }
        return strategy_df, solver_result

    if large_mode:
        # first_success["ranks"] is already a tuple of position arrays
        _induce_profile_from_rankings(solver, players, states, first_success["ranks"], committee_idxs)
    elif weak_orders:
        weak_tiers = tuple(np.asarray(order, dtype=np.int8) for order in first_success["rankings"])
        _induce_profile_from_weak_orders(solver, players, states, weak_tiers, committee_idxs)
    else:
        position_rankings: list[np.ndarray] = []
        for perm in first_success["rankings"]:
            perm = np.asarray(perm, dtype=np.int64)
            pos_arr = np.empty(len(perm), dtype=np.int8)
            for rank, state_idx in enumerate(perm):
                pos_arr[int(state_idx)] = rank
            position_rankings.append(pos_arr)
        _induce_profile_from_rankings(solver, players, states, tuple(position_rankings), committee_idxs)
    strategy_df = solver._create_strategy_dataframe()
    solver_result = {
        "converged": True,
        "stopping_reason": "ordinal_ranking_verified",
        "tested_combinations": tested,
        "total_combinations": total,
        "runtime_seconds": elapsed,
        "final_tau_p": 0.0,
        "final_tau_r": 0.0,
        "outer_iterations": tested,
    }
    if not large_mode:
        solver_result["ordinal_ranking_perms"] = first_success.get("perms")
    if logger is not None:
        logger.info(
            f"Ordinal ranking search tested {tested:,d}/{total:,d} combinations in {elapsed:.2f}s."
        )
    return strategy_df, solver_result
