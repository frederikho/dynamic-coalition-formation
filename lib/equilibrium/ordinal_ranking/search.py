"""Multiprocessing workers and iterators over ranking combinations."""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
from lib.equilibrium.ordinal_ranking.induced_strategies import (
    _build_induced_arrays,
    _build_induced_arrays_weak,
)
from lib.equilibrium.ordinal_ranking.value_mdp import _solve_values, _verify_fast
from lib.equilibrium.ordinal_ranking.numba_loops import (
    _NUMBA_AVAILABLE,
    _build_arrays_weak_nb,
    _verify_fast_nb,
    _solve_V_nb,
    _residuals_nb_core,
)
from lib.equilibrium.ordinal_ranking.weak_equality import (
    _solve_weak_equalities,
    _weak_tie_structure,
)

_WORKER_CTX: dict[str, Any] | None = None


def _init_worker_ctx(
    players: list[str],
    states: list[str],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    pos: np.ndarray | None,
    state_perms: np.ndarray | None,
    weak_orders: bool,
    approve_lookup: np.ndarray | None,
    weak_equality_solve: bool = False,
    weak_equality_max_vars: int | None = None,
    unanimity_required: bool = True,
    effectivity: dict | None = None,
    power_rule: str = "power_threshold",
    use_newton: bool = True,
    use_broyden: bool = False,
) -> None:
    global _WORKER_CTX
    n_players = len(players)
    n_states = len(states)
    max_k = 0
    for p_rows in committee_idxs:
        for c_row in p_rows:
            for comm in c_row:
                max_k = max(max_k, len(comm))

    comm_arr = np.full((n_players, n_states, n_states, max_k), -1, dtype=np.int8)
    comm_size = np.zeros((n_players, n_states, n_states), dtype=np.int8)
    for pi in range(n_players):
        for ci in range(n_states):
            for ni in range(n_states):
                comm = committee_idxs[pi][ci][ni]
                comm_size[pi, ci, ni] = len(comm)
                for k, ai in enumerate(comm):
                    comm_arr[pi, ci, ni, k] = ai

    _WORKER_CTX = {
        "players": players,
        "states": states,
        "committee_idxs": committee_idxs,
        "protocol_arr": protocol_arr,
        "payoff_array": payoff_array,
        "discounting": discounting,
        "pos": pos,
        "state_perms": state_perms,
        "weak_orders": weak_orders,
        "approve_lookup": approve_lookup,
        "weak_equality_solve": weak_equality_solve,
        "weak_equality_max_vars": weak_equality_max_vars,
        "unanimity_required": unanimity_required,
        "numba_comm_arr": comm_arr,
        "numba_comm_size": comm_size,
        "player_idx": {p: i for i, p in enumerate(players)},
        "state_idx": {s: i for i, s in enumerate(states)},
        "protocol": {p: float(protocol_arr[i]) for i, p in enumerate(players)},
        "effectivity": effectivity,
        "power_rule": power_rule,
        "use_newton": use_newton,
        "use_broyden": use_broyden,
    }

    # Workers ignore SIGINT so only the main process handles Ctrl+C.
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

    # Warm up Numba JIT functions so the first real batch isn't delayed by compilation.
    if _NUMBA_AVAILABLE and weak_orders:
        _tiers_buf = np.zeros((n_players, n_states), dtype=np.int8)
        _build_arrays_weak_nb(_tiers_buf, comm_arr, comm_size, protocol_arr)
        _dummy_P = np.eye(n_states, dtype=np.float64)
        _solve_V_nb(_dummy_P, payoff_array, discounting)
        _dummy_pp = np.zeros((n_players, n_states, n_states))
        _dummy_aa = np.zeros((n_players, n_players, n_states, n_states))
        _dummy_ap = np.ones((n_players, n_states, n_states))
        _dummy_V = np.zeros((n_states, n_players))
        _verify_fast_nb(_dummy_pp, _dummy_aa, _dummy_ap, _dummy_V, comm_arr, comm_size)
        if weak_equality_solve:
            from lib.equilibrium.ordinal_ranking.numba_loops import (
                _solve_broyden_nb as _broyden_warmup,
            )
            _dummy_raw = np.zeros(1, dtype=np.float64)
            _dummy_fa = np.zeros((1, 4), dtype=np.int8)
            _dummy_pt_pi = np.zeros(1, dtype=np.int8)
            _dummy_pt_si = np.zeros(1, dtype=np.int8)
            _dummy_pt_widxs = np.zeros((1, 1), dtype=np.int8)
            _dummy_pt_nwidxs = np.zeros(1, dtype=np.int8)
            _dummy_aff_pi = np.zeros(1, dtype=np.int8)
            _dummy_aff_ci = np.zeros(1, dtype=np.int8)
            _dummy_aff_is_pt = np.zeros(1, dtype=np.bool_)
            _nb_solver_args = (
                _dummy_raw,
                _dummy_pp, _dummy_aa, _dummy_ap,
                _dummy_fa, 1,
                _dummy_pt_pi, _dummy_pt_si, _dummy_pt_widxs, _dummy_pt_nwidxs, 0,
                _dummy_aff_pi, _dummy_aff_ci, _dummy_aff_is_pt, 0,
                comm_arr, comm_size, _tiers_buf,
                protocol_arr, payoff_array, discounting,
                n_players, n_states,
            )
            _residuals_nb_core(
                _dummy_raw, _dummy_pp, _dummy_aa, _dummy_ap,
                _dummy_fa, 1,
                _dummy_pt_pi, _dummy_pt_si, _dummy_pt_widxs, _dummy_pt_nwidxs, 0,
                _dummy_aff_pi, _dummy_aff_ci, _dummy_aff_is_pt, 0,
                comm_arr, comm_size, _tiers_buf,
                protocol_arr, payoff_array, discounting,
                n_players, n_states,
            )
            if use_broyden:
                _broyden_warmup(*_nb_solver_args)


def _make_solver_timing() -> dict[str, float]:
    return {
        "solver_root": 0.0, "solver_finalize": 0.0, "solver_setup": 0.0, "solver_check": 0.0,
        "solver_setup_copy": 0.0, "solver_setup_indices": 0.0, "solver_setup_numba": 0.0,
        "solver_root_v_solve": 0.0, "solver_root_p_agg": 0.0, "solver_root_other": 0.0,
        "solver_root_mapping": 0.0, "solver_root_residuals": 0.0, "solver_setup_guesses": 0.0,
        "finalize_rebuild": 0.0, "finalize_verify": 0.0, "finalize_solver_obj": 0.0,
        "solver_nb_newton": 0.0,
    }


def _search_chunk(batch_tuples: np.ndarray, stop_on_success: bool = True) -> dict[str, Any]:
    assert _WORKER_CTX is not None
    if len(batch_tuples) == 0:
        return {
            "tested": 0, "success": None, "all_successes": [],
            "t_numba": 0.0, "t_tie_struct": 0.0, "t_solver": 0.0,
            "t_solver_root": 0.0, "t_solver_finalize": 0.0,
            "t_solver_setup": 0.0, "t_solver_check": 0.0,
            "t_solver_setup_copy": 0.0,
            "t_solver_setup_indices": 0.0,
            "t_solver_setup_numba": 0.0,
            "t_solver_root_v_solve": 0.0,
            "t_solver_root_p_agg": 0.0,
            "t_solver_root_mapping": 0.0,
            "t_solver_root_residuals": 0.0,
            "t_solver_setup_guesses": 0.0,
            "t_solver_nb_newton": 0.0,
            "t_solver_finalize_rebuild": 0.0,
            "t_solver_finalize_verify": 0.0,
            "t_solver_finalize_solver_obj": 0.0,
            "n_solver_calls": 0, "n_skipped": 0, "n_hits": 0,
            "n_free_histogram": {},
            "solver_calls_by_n_free": {},
            "solver_time_by_n_free": {},
            "exit_stats_counts": np.zeros((7, 8), dtype=np.int64),
            "weak_solver_flow_stats": {},
            "weak_payload_returned": 0,
            "weak_payload_verified_true": 0,
            "weak_payload_verified_false": 0,
        }
    ctx = _WORKER_CTX
    players, states = ctx["players"], ctx["states"]
    n_states = len(states)
    protocol_arr, payoff_array = ctx["protocol_arr"], ctx["payoff_array"]
    discounting = ctx["discounting"]
    weak_orders = ctx["weak_orders"]
    weak_equality_solve = ctx["weak_equality_solve"]
    unanimity_required = ctx["unanimity_required"]
    comm_arr = ctx["numba_comm_arr"]
    comm_size = ctx["numba_comm_size"]

    all_successes = []
    tested = 0
    t_numba = 0.0
    t_tie_struct = 0.0
    t_solver = 0.0
    # Detailed solver sub-timings (accumulated over all calls)
    t_solver_root = 0.0
    t_solver_finalize = 0.0
    t_solver_setup = 0.0
    t_solver_check = 0.0
    t_solver_setup_copy = 0.0
    t_solver_setup_indices = 0.0
    t_solver_setup_numba = 0.0
    t_solver_root_v_solve = 0.0
    t_solver_root_p_agg = 0.0
    t_solver_root_mapping = 0.0
    t_solver_root_residuals = 0.0
    t_solver_setup_guesses = 0.0
    t_solver_nb_newton = 0.0
    t_solver_finalize_rebuild = 0.0
    t_solver_finalize_verify = 0.0
    t_solver_finalize_solver_obj = 0.0

    n_solver_calls = 0
    n_skipped = 0
    n_hits = 0
    n_free_histogram: dict[int, int] = {}
    solver_calls_by_n_free: dict[int, int] = {}
    solver_time_by_n_free: dict[int, float] = {}
    exit_stats_counts = np.zeros((7, 8), dtype=np.int64)
    weak_solver_flow_stats: dict[str, int] = {}
    weak_payload_returned = 0
    weak_payload_verified_true = 0
    weak_payload_verified_false = 0

    _tiers_buf = np.empty((len(players), n_states), dtype=np.int8) if weak_orders else None

    for row in batch_tuples:
        order_ids = tuple(int(x) for x in row)
        tested += 1

        verified = False
        solved_payload = None

        if weak_orders:
            tiers_tuple = tuple(ctx["pos"][idx] for idx in order_ids)
            tiers_arr = np.stack(tiers_tuple)

            _t0 = time.perf_counter()
            if _NUMBA_AVAILABLE:
                if _tiers_buf is not None:
                    for _pi, _perm_idx in enumerate(order_ids):
                        _tiers_buf[_pi] = ctx["pos"][_perm_idx]
                    proposal_probs, approval_action, approval_pass, P_array = _build_arrays_weak_nb(
                        _tiers_buf, comm_arr, comm_size, protocol_arr
                    )
                else:
                    proposal_probs, approval_action, approval_pass, P_array = _build_arrays_weak_nb(
                        tiers_arr, comm_arr, comm_size, protocol_arr
                    )
                V_array = _solve_V_nb(P_array, payoff_array, discounting)
                verified = _verify_fast_nb(
                    proposal_probs, approval_action, approval_pass, V_array,
                    comm_arr, comm_size,
                )
            else:
                proposal_probs, approval_action, approval_pass, P_array = _build_induced_arrays_weak(
                    players, tiers_tuple, ctx["committee_idxs"], protocol_arr
                )
                V_array = _solve_values(P_array, payoff_array, discounting)
                verified, _ = _verify_fast(
                    players=players, states=states, V_array=V_array,
                    proposal_choice=None,
                    approval_action=approval_action, approval_pass=approval_pass,
                    committee_idxs=ctx["committee_idxs"], proposal_probs=proposal_probs,
                )
            t_numba += time.perf_counter() - _t0

            if weak_equality_solve:
                # Calculate exact n_free for every combination to ensure an exhaustive histogram
                t_ts0 = time.perf_counter()
                tie_struct = _weak_tie_structure(
                    players, states, tiers_tuple, ctx["committee_idxs"]
                )
                n_free = len(tie_struct[0]) + sum(len(w) - 1 for _, _, w in tie_struct[1])
                t_tie_struct += time.perf_counter() - t_ts0
                n_free_histogram[n_free] = n_free_histogram.get(n_free, 0) + 1

                if not verified:
                    weak_equality_max_vars = ctx["weak_equality_max_vars"]
                    # Skip the heavy solver if we exceed the max variable threshold
                    if n_free > 0 and (weak_equality_max_vars is None or n_free <= weak_equality_max_vars):
                        n_solver_calls += 1
                        _call_exit_stats = np.zeros((7, 8), dtype=np.int64)
                        _call_flow_stats: dict[str, int] = {}
                        solver_timing = _make_solver_timing()

                        _nb_tiers = _tiers_buf if _NUMBA_AVAILABLE else None
                        t_sv0 = time.perf_counter()
                        solved_payload = _solve_weak_equalities(
                            players=players, states=states,
                            payoffs=_payoffs_df_from_ctx(ctx),
                            protocol=ctx["protocol"],
                            discounting=discounting,
                            unanimity_required=ctx.get("unanimity_required", True),
                            effectivity=ctx.get("effectivity"),
                            power_rule=ctx.get("power_rule", "power_threshold"),
                            tiers=tiers_tuple,
                            committee_idxs=ctx["committee_idxs"],
                            max_vars=weak_equality_max_vars,
                            use_newton=ctx.get("use_newton", True),
                            use_broyden=ctx.get("use_broyden", False),
                            _precomputed_tie_structure=tie_struct,
                            _precomputed_canon_arrays=(proposal_probs, approval_action, approval_pass),
                            _numba_comm_arr=comm_arr if _NUMBA_AVAILABLE else None,
                            _numba_comm_size=comm_size if _NUMBA_AVAILABLE else None,
                            _numba_tiers=_nb_tiers,
                            timing_data=solver_timing,
                            player_idx=ctx["player_idx"],
                            state_idx=ctx["state_idx"],
                            _precomputed_payoff_array=payoff_array,
                            _precomputed_protocol_arr=protocol_arr,
                            _exit_stats_counts=_call_exit_stats,
                            _flow_stats=_call_flow_stats,
                        )
                        exit_stats_counts += _call_exit_stats
                        for _k, _v in _call_flow_stats.items():
                            weak_solver_flow_stats[_k] = weak_solver_flow_stats.get(_k, 0) + int(_v)
                        if solved_payload is not None:
                            weak_payload_returned += 1
                            if bool(solved_payload.get("verification_success")):
                                weak_payload_verified_true += 1
                            else:
                                weak_payload_verified_false += 1
                        _d_sv = time.perf_counter() - t_sv0
                        solver_calls_by_n_free[n_free] = solver_calls_by_n_free.get(n_free, 0) + 1
                        solver_time_by_n_free[n_free] = solver_time_by_n_free.get(n_free, 0.0) + _d_sv
                        t_solver += _d_sv
                        t_solver_root += solver_timing["solver_root"]
                        t_solver_finalize += solver_timing["solver_finalize"]
                        t_solver_setup += solver_timing["solver_setup"]
                        t_solver_check += solver_timing["solver_check"]
                        t_solver_setup_copy += solver_timing["solver_setup_copy"]
                        t_solver_setup_indices += solver_timing["solver_setup_indices"]
                        t_solver_setup_numba += solver_timing["solver_setup_numba"]
                        t_solver_root_v_solve += solver_timing["solver_root_v_solve"]
                        t_solver_root_p_agg += solver_timing["solver_root_p_agg"]
                        t_solver_root_mapping += solver_timing["solver_root_mapping"]
                        t_solver_root_residuals += solver_timing["solver_root_residuals"]
                        t_solver_setup_guesses += solver_timing["solver_setup_guesses"]
                        t_solver_nb_newton += solver_timing["solver_nb_newton"]
                        t_solver_finalize_rebuild += solver_timing["finalize_rebuild"]
                        t_solver_finalize_verify += solver_timing["finalize_verify"]
                        t_solver_finalize_solver_obj += solver_timing["finalize_solver_obj"]

                        if solved_payload is not None and solved_payload.get("verification_success"):
                            verified = True
                    elif n_free > 0:
                        n_skipped += 1

        else:
            ranks = tuple(ctx["pos"][idx] for idx in order_ids)
            proposal_choice, approval_action, approval_pass, P_array = _build_induced_arrays(
                players, ranks, ctx["committee_idxs"], protocol_arr
            )
            V_array = _solve_values(P_array, payoff_array, discounting)
            verified, _ = _verify_fast(
                players=players, states=states, V_array=V_array,
                proposal_choice=proposal_choice, approval_action=approval_action,
                approval_pass=approval_pass, committee_idxs=ctx["committee_idxs"],
            )

        if verified:
            n_hits += 1
            success = {
                "perms": order_ids,
                "rankings": tuple(ctx["state_perms"][idx].copy() for idx in order_ids),
                "n_free": n_free if weak_equality_solve else 0,
            }
            if solved_payload is not None and solved_payload.get("verification_success"):
                success.update({
                    "source": "weak_equality_solve",
                    "payload": {
                        "P": solved_payload["P"],
                        "V": solved_payload["V"],
                        "strategy_df": solved_payload["strategy_df"],
                    },
                })
            else:
                success["source"] = "canonical"
            if stop_on_success:
                return {
                    "tested": tested, "success": success, "all_successes": [success],
                    "t_numba": t_numba, "t_tie_struct": t_tie_struct, "t_solver": t_solver,
                    "t_solver_root": t_solver_root, "t_solver_finalize": t_solver_finalize,
                    "t_solver_setup": t_solver_setup, "t_solver_check": t_solver_check,
                    "t_solver_setup_copy": t_solver_setup_copy,
                    "t_solver_setup_indices": t_solver_setup_indices,
                    "t_solver_setup_numba": t_solver_setup_numba,
                    "t_solver_root_v_solve": t_solver_root_v_solve,
                    "t_solver_root_p_agg": t_solver_root_p_agg,
                    "t_solver_root_mapping": t_solver_root_mapping,
                    "t_solver_root_residuals": t_solver_root_residuals,
                    "t_solver_setup_guesses": t_solver_setup_guesses,
                    "t_solver_nb_newton": t_solver_nb_newton,
                    "t_solver_finalize_rebuild": t_solver_finalize_rebuild,
                    "t_solver_finalize_verify": t_solver_finalize_verify,
                    "t_solver_finalize_solver_obj": t_solver_finalize_solver_obj,
                    "n_solver_calls": n_solver_calls, "n_skipped": n_skipped, "n_hits": n_hits,
                    "n_free_histogram": n_free_histogram,
                    "solver_calls_by_n_free": solver_calls_by_n_free,
                    "solver_time_by_n_free": solver_time_by_n_free,
                    "exit_stats_counts": exit_stats_counts,
                    "weak_solver_flow_stats": weak_solver_flow_stats,
                    "weak_payload_returned": weak_payload_returned,
                    "weak_payload_verified_true": weak_payload_verified_true,
                    "weak_payload_verified_false": weak_payload_verified_false,
                }
            all_successes.append(success)

    return {
        "tested": tested, "success": None, "all_successes": all_successes,
        "t_numba": t_numba, "t_tie_struct": t_tie_struct, "t_solver": t_solver,
        "t_solver_root": t_solver_root, "t_solver_finalize": t_solver_finalize,
        "t_solver_setup": t_solver_setup, "t_solver_check": t_solver_check,
        "t_solver_setup_copy": t_solver_setup_copy,
        "t_solver_setup_indices": t_solver_setup_indices,
        "t_solver_setup_numba": t_solver_setup_numba,
        "t_solver_root_v_solve": t_solver_root_v_solve,
        "t_solver_root_p_agg": t_solver_root_p_agg,
        "t_solver_root_mapping": t_solver_root_mapping,
        "t_solver_root_residuals": t_solver_root_residuals,
        "t_solver_setup_guesses": t_solver_setup_guesses,
        "t_solver_nb_newton": t_solver_nb_newton,
        "t_solver_finalize_rebuild": t_solver_finalize_rebuild,
        "t_solver_finalize_verify": t_solver_finalize_verify,
        "t_solver_finalize_solver_obj": t_solver_finalize_solver_obj,
        "n_solver_calls": n_solver_calls, "n_skipped": n_skipped, "n_hits": n_hits,
        "n_free_histogram": n_free_histogram,
        "solver_calls_by_n_free": solver_calls_by_n_free,
        "solver_time_by_n_free": solver_time_by_n_free,
        "exit_stats_counts": exit_stats_counts,
        "weak_solver_flow_stats": weak_solver_flow_stats,
        "weak_payload_returned": weak_payload_returned,
        "weak_payload_verified_true": weak_payload_verified_true,
        "weak_payload_verified_false": weak_payload_verified_false,
    }


def _payoffs_df_from_ctx(ctx: dict[str, Any]):
    """Reconstruct a minimal payoffs DataFrame from the worker context arrays."""
    import pandas as pd
    return pd.DataFrame(
        ctx["payoff_array"],
        index=ctx["states"],
        columns=ctx["players"],
    )


def _search_chunk_large(batch_ranks: np.ndarray, stop_on_success: bool = True) -> dict[str, Any]:
    assert _WORKER_CTX is not None
    if len(batch_ranks) == 0:
        return {
            "tested": 0, "success": None, "all_successes": [],
            "t_numba": 0.0, "t_tie_struct": 0.0, "t_solver": 0.0,
            "n_solver_calls": 0, "n_skipped": 0, "n_hits": 0,
        }
    ctx = _WORKER_CTX
    players, states = ctx["players"], ctx["states"]
    n_players, n_states = len(players), len(states)
    protocol_arr, payoff_array = ctx["protocol_arr"], ctx["payoff_array"]
    discounting = ctx["discounting"]

    all_successes = []
    tested = 0
    for ranks_arr in batch_ranks:
        tested += 1
        ranks = tuple(ranks_arr[i] for i in range(n_players))

        proposal_choice, approval_action, approval_pass, P_array = _build_induced_arrays(
            players, ranks, ctx["committee_idxs"], protocol_arr
        )
        V_array = _solve_values(P_array, payoff_array, discounting)
        verified, _ = _verify_fast(
            players=players, states=states, V_array=V_array,
            proposal_choice=proposal_choice, approval_action=approval_action,
            approval_pass=approval_pass, committee_idxs=ctx["committee_idxs"],
        )

        if verified:
            success = {"rankings": ranks}
            if stop_on_success:
                return {"tested": tested, "success": success, "all_successes": [success]}
            all_successes.append(success)

    return {"tested": tested, "success": None, "all_successes": all_successes}


def _iter_tuples(
    n_players: int, n_perms: int, total: int | None = None,
    shuffle: bool = False, random_seed: int = 0,
    perm_orders: list[np.ndarray] | None = None,
    valid_idx_per_player: list[np.ndarray] | None = None,
):
    count = 0
    if shuffle:
        rng = np.random.default_rng(random_seed)
        if valid_idx_per_player is not None:
            # Sample uniformly from each player's valid index set
            while total is None or count < total:
                yield np.array([
                    int(rng.choice(valid_idx_per_player[pi]))
                    for pi in range(n_players)
                ])
                count += 1
        else:
            while total is None or count < total:
                yield rng.integers(0, n_perms, size=n_players)
                count += 1
    elif perm_orders:
        for pt in itertools.product(*perm_orders):
            if total is not None and count >= total:
                return
            yield np.array(pt)
            count += 1
    else:
        for pt in itertools.product(range(n_perms), repeat=n_players):
            if total is not None and count >= total:
                return
            yield np.array(pt)
            count += 1


def _iter_rank_combos_large(
    n_players: int, n_states: int, total: int | None = None, random_seed: int = 0
):
    rng = np.random.default_rng(random_seed)
    count = 0
    while total is None or count < total:
        ranks_batch = np.zeros((n_players, n_states), dtype=np.int8)
        for pi in range(n_players):
            perm = rng.permutation(n_states)
            for rank, state_idx in enumerate(perm):
                ranks_batch[pi, state_idx] = rank
        yield ranks_batch
        count += 1


def _iter_batches(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield np.array(batch)
            batch = []
    if batch:
        yield np.array(batch)
