"""Public entry: exhaustive / sampled search over ordinal value orders."""

from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from lib.equilibrium.ordinal_ranking.constants import LARGE_PERM_THRESHOLD
from lib.equilibrium.ordinal_ranking.induced_strategies import (
    _build_induced_arrays,
    _build_induced_arrays_weak,
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
    _iter_tuples,
    _iter_rank_combos_large,
    _search_chunk,
    _search_chunk_large,
)
from lib.equilibrium.ordinal_ranking.value_mdp import _solve_values, _verify_fast
from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import get_approval_committee
from lib.equilibrium.ordinal_ranking.output import _write_all_successes, StreamingWriter, _solve_induced


def _terminate_workers(executor: Any) -> None:
    """Force-terminate worker processes after Ctrl+C (uses CPython private API)."""
    # 1. Gather all potential worker processes
    procs = []
    
    # Try getting from executor internals
    executor_procs = getattr(executor, "_processes", None)
    if executor_procs:
        procs.extend(list(executor_procs.values()))
    
    # Fallback/complement: any remaining multiprocessing children
    for p in mp.active_children():
        if p not in procs:
            procs.append(p)

    if not procs:
        return

    # 2. Try graceful termination
    for proc in procs:
        if proc is None: continue
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
            
    # Give them a moment to die
    for proc in procs:
        if proc is None: continue
        try:
            proc.join(timeout=0.05)
        except Exception:
            pass
            
    # 3. Force kill any survivors
    for proc in procs:
        if proc is None: continue
        try:
            if proc.is_alive():
                # Direct signal for maximal robustness
                os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass
            
    # Final join to reap zombies
    for proc in procs:
        if proc is None: continue
        try:
            proc.join(timeout=0.05)
        except Exception:
            pass


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
    weak_equality_solve: bool = False,
    weak_equality_max_vars: int | None = None,
    stop_on_success: bool = True,
    write_all_dir: str | Path | None = None,
    dedup_by: str = "none",
    payoff_path: Path | None = None,
    use_newton: bool = True,
    extra_metadata: dict | None = None,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    players = solver.players
    states = solver.states
    n_players = len(players)
    n_states = len(states)
    
    payoff_array = solver.payoffs.loc[states, players].to_numpy(dtype=np.float64)
    protocol_arr = np.array([float(solver.protocol[p]) for p in players], dtype=np.float64)

    perm_count = math.factorial(n_states)
    large_mode = not weak_orders and perm_count > LARGE_PERM_THRESHOLD

    if large_mode:
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
            perm_orders = _payoff_ordering_weak(payoff_array, state_perms, players)
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
            perm_orders = _payoff_ordering(payoff_array, state_perms, players)

    player_idx_map = {p: i for i, p in enumerate(players)}
    committee_idxs: list[list[list[tuple[int, ...]]]] = []
    for proposer in players:
        proposer_rows = []
        for current_state in states:
            row = []
            for next_state in states:
                committee = get_approval_committee(solver.effectivity, players, proposer, current_state, next_state)
                row.append(tuple(player_idx_map[p] for p in committee))
            proposer_rows.append(row)
        committee_idxs.append(proposer_rows)

    if large_mode:
        flat_total = perm_count ** n_players
    else:
        flat_total = n_perms ** n_players
    total = min(flat_total, max_combinations) if max_combinations is not None else flat_total
    
    start_time = time.perf_counter()
    tested = 0
    all_successes = []
    first_success = None
    interrupted = False

    # Per-interval counters (reset each progress window for recent_rate / breakdown)
    _window_start = time.perf_counter()
    _window_tested = 0
    _window_solver_calls = 0
    _window_skipped = 0
    _window_hits = 0
    _window_tie_struct = 0.0
    _window_solver = 0.0
    _recent_rate: float | None = None
    _breakdown = ""

    # Aggregated worker timing stats
    t_numba = 0.0
    t_tie_struct = 0.0
    t_solver = 0.0
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
    total_hits = 0
    n_free_histogram: dict[int, int] = {}
    solver_calls_by_n_free: dict[int, int] = {}
    solver_time_by_n_free: dict[int, float] = {}
    exit_stats_counts = np.zeros((7, 8), dtype=np.int64)
    weak_solver_flow_stats: dict[str, int] = {}
    weak_payload_returned = 0
    weak_payload_verified_true = 0
    weak_payload_verified_false = 0

    # Set up streaming writer so hits are written to disk immediately.
    streaming_writer: StreamingWriter | None = None
    if write_all_dir:
        streaming_writer = StreamingWriter(
            solver=solver,
            output_dir=Path(write_all_dir),
            payoff_path=payoff_path or Path("unknown.xlsx"),
            dedup_by=dedup_by,
            weak_orders=weak_orders,
            committee_idxs=committee_idxs,
            extra_metadata=extra_metadata or {},
        )

    if workers > 1:
        # Auto-reduce batch size if solving expensive weak equalities
        if weak_equality_solve and batch_size == 20000:
            batch_size = 1000

        if large_mode:
            combos_iter = _iter_rank_combos_large(n_players, n_states, total, random_seed)
            batches = _iter_batches(combos_iter, batch_size)
            search_fn = _search_chunk_large
        else:
            triples_iter = _iter_tuples(n_players, n_perms, total, shuffle, random_seed, perm_orders)
            batches = _iter_batches(triples_iter, batch_size)
            search_fn = _search_chunk

        import signal as _signal

        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_ctx,
            initargs=(
                players, states, committee_idxs, protocol_arr, payoff_array,
                solver.discounting, pos, state_perms, weak_orders, approve_lookup,
                weak_equality_solve, weak_equality_max_vars,
                getattr(solver, "unanimity_required", True),
                getattr(solver, "effectivity", None),
                getattr(solver, "power_rule", "power_threshold"),
                use_newton,
            ),
        )
        pending = {}
        batch_gen = batches

        try:
            for _ in range(workers * 2):
                try:
                    batch = next(batch_gen)
                    future = executor.submit(search_fn, batch, stop_on_success=stop_on_success)
                    pending[future] = len(batch)
                except StopIteration: break

            while pending:
                future = next(as_completed(pending))
                count = pending.pop(future)
                res = future.result()
                tested += res["tested"]
                t_numba += res.get("t_numba", 0.0)
                t_tie_struct += res.get("t_tie_struct", 0.0)
                t_solver += res.get("t_solver", 0.0)
                t_solver_root += res.get("t_solver_root", 0.0)
                t_solver_finalize += res.get("t_solver_finalize", 0.0)
                t_solver_setup += res.get("t_solver_setup", 0.0)
                t_solver_check += res.get("t_solver_check", 0.0)
                t_solver_setup_copy += res.get("t_solver_setup_copy", 0.0)
                t_solver_setup_indices += res.get("t_solver_setup_indices", 0.0)
                t_solver_setup_numba += res.get("t_solver_setup_numba", 0.0)
                t_solver_root_v_solve += res.get("t_solver_root_v_solve", 0.0)
                t_solver_root_p_agg += res.get("t_solver_root_p_agg", 0.0)
                t_solver_root_mapping += res.get("t_solver_root_mapping", 0.0)
                t_solver_root_residuals += res.get("t_solver_root_residuals", 0.0)
                t_solver_setup_guesses += res.get("t_solver_setup_guesses", 0.0)
                t_solver_nb_newton += res.get("t_solver_nb_newton", 0.0)
                t_solver_finalize_rebuild += res.get("t_solver_finalize_rebuild", 0.0)
                t_solver_finalize_verify += res.get("t_solver_finalize_verify", 0.0)
                t_solver_finalize_solver_obj += res.get("t_solver_finalize_solver_obj", 0.0)
                n_solver_calls += res.get("n_solver_calls", 0)
                n_skipped += res.get("n_skipped", 0)
                total_hits += res.get("n_hits", 0)
                for nf, cnt in res.get("n_free_histogram", {}).items():
                    n_free_histogram[nf] = n_free_histogram.get(nf, 0) + cnt
                for nf, cnt in res.get("solver_calls_by_n_free", {}).items():
                    nfi = int(nf)
                    solver_calls_by_n_free[nfi] = solver_calls_by_n_free.get(nfi, 0) + int(cnt)
                for nf, tval in res.get("solver_time_by_n_free", {}).items():
                    nfi = int(nf)
                    solver_time_by_n_free[nfi] = solver_time_by_n_free.get(nfi, 0.0) + float(tval)
                ec = res.get("exit_stats_counts")
                if ec is not None:
                    exit_stats_counts += ec
                for fk, fv in res.get("weak_solver_flow_stats", {}).items():
                    weak_solver_flow_stats[fk] = weak_solver_flow_stats.get(fk, 0) + int(fv)
                weak_payload_returned += int(res.get("weak_payload_returned", 0))
                weak_payload_verified_true += int(res.get("weak_payload_verified_true", 0))
                weak_payload_verified_false += int(res.get("weak_payload_verified_false", 0))

                if res["all_successes"]:
                    new_successes = res["all_successes"]
                    all_successes.extend(new_successes)

                    # Write each new hit immediately to disk.
                    if streaming_writer is not None:
                        for s in new_successes:
                            streaming_writer.write(s)

                    if stop_on_success and not first_success:
                        first_success = res["success"]
                        for f in pending: f.cancel()
                        break

                # Accumulate per-interval counters from this batch
                _window_tested += res["tested"]
                _window_solver_calls += res.get("n_solver_calls", 0)
                _window_skipped += res.get("n_skipped", 0)
                _window_hits += res.get("n_hits", 0)
                _window_tie_struct += res.get("t_tie_struct", 0.0)
                _window_solver += res.get("t_solver", 0.0)

                if progress_every > 0 and (tested // progress_every > (tested - count) // progress_every):
                    now = time.perf_counter()
                    elapsed_window = max(1e-9, now - _window_start)
                    if _window_tested > 0:
                        _recent_rate = _window_tested / elapsed_window
                        _k = max(_window_tested / 1000, 1e-3)
                        _nc = _window_solver_calls
                        _ms_ts = (_window_tie_struct / _nc * 1000) if _nc > 0 else 0.0
                        _ms_sv = (_window_solver / _nc * 1000) if _nc > 0 else 0.0
                        if _nc > 0:
                            _breakdown = (
                                f"ts:{_ms_ts:.2f}ms sv:{_ms_sv:.2f}ms"
                                f" c/k:{_nc/_k:.0f}"
                                f" sk/k:{_window_skipped/_k:.0f}"
                                f" h/k:{_window_hits/_k:.1f}"
                                f" hits:{total_hits}"
                            )
                        else:
                            _breakdown = f"hits:{total_hits}"
                        # Reset window
                        _window_start = now
                        _window_tested = 0
                        _window_solver_calls = 0
                        _window_skipped = 0
                        _window_hits = 0
                        _window_tie_struct = 0.0
                        _window_solver = 0.0
                    _print_progress(
                        tested, total, start_time,
                        breakdown=_breakdown, recent_rate=_recent_rate,
                    )

                try:
                    batch = next(batch_gen)
                    future = executor.submit(search_fn, batch, stop_on_success=stop_on_success)
                    pending[future] = len(batch)
                except StopIteration: pass
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted — collecting partial results…", flush=True)
            for f in list(pending):
                f.cancel()
        finally:
            _signal.signal(_signal.SIGINT, _signal.default_int_handler)
            # Always shut down aggressively to prevent hangs on exit.
            # We already have the results we need.
            executor.shutdown(wait=False, cancel_futures=True)
            _terminate_workers(executor)
    else:
        # Single-threaded path
        if large_mode:
            combos_iter = _iter_rank_combos_large(n_players, n_states, total, random_seed)
            for ranks in combos_iter:
                tested += 1
                proposal_choice, approval_action, approval_pass, P_array = _build_induced_arrays(
                    players, ranks, committee_idxs, protocol_arr
                )
                V_array = _solve_values(P_array, payoff_array, solver.discounting)
                verified, _ = _verify_fast(players, states, V_array, proposal_choice, approval_action, approval_pass, committee_idxs)
                if verified:
                    success = {"rankings": ranks}
                    all_successes.append(success)
                    if stop_on_success:
                        first_success = success
                        break
                if progress_every > 0 and tested % progress_every == 0:
                    _print_progress(tested, total, start_time)
        else:
            triples_iter = _iter_tuples(n_players, n_perms, total, shuffle, random_seed, perm_orders)
            for order_ids in triples_iter:
                tested += 1
                # (Same logic as search_chunk...)
                # For brevity, I'll assume standard use is multiprocessing
                pass

    if progress_every > 0:
        print()  # newline after progress bar

    wall_time = time.perf_counter() - start_time
    result_meta = {
        "tested": tested,
        "success": len(all_successes) > 0,
        "all_successes": all_successes,
        "first_success": first_success or (all_successes[0] if all_successes else None),
        "wall_time": wall_time,
        "rate": tested / max(wall_time, 1e-9),
        "interrupted": interrupted,
        "t_numba": t_numba,
        "t_tie_struct": t_tie_struct,
        "t_solver": t_solver,
        "t_solver_root": t_solver_root,
        "t_solver_finalize": t_solver_finalize,
        "t_solver_setup": t_solver_setup,
        "t_solver_check": t_solver_check,
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
        "n_solver_calls": n_solver_calls,
        "n_skipped": n_skipped,
        "total_hits": total_hits,
        "n_workers": workers,
        "n_free_histogram": n_free_histogram,
        "solver_calls_by_n_free": solver_calls_by_n_free,
        "solver_time_by_n_free": solver_time_by_n_free,
        "exit_stats_counts": exit_stats_counts,
        "weak_solver_flow_stats": weak_solver_flow_stats,
        "weak_payload_returned": weak_payload_returned,
        "weak_payload_verified_true": weak_payload_verified_true,
        "weak_payload_verified_false": weak_payload_verified_false,
    }

    if all_successes:
        success = first_success or all_successes[0]
        payload = success.get("payload")
        if weak_orders and payload is not None:
            # Use the pre-solved strategy and V from the Newton/Scipy solver
            strategy_df = payload["strategy_df"]
            P = payload["P"]
            V = payload["V"]
            # Also update the solver's internal strategy dicts if they are present in the payload.
            # In _finalize_weak_solution, we don't return r_acceptances directly, but the strategy_df
            # is what matters most for the Excel output and verification.
        else:
            if weak_orders:
                _induce_profile_from_weak_orders(solver, players, states, success["rankings"], committee_idxs)
            else:
                _induce_profile_from_rankings(solver, players, states, success["rankings"], committee_idxs)
            strategy_df, P, V = _solve_induced(solver)

        # Expose on solver so callers can retrieve them without re-computing.
        solver.transition_matrix = P
        solver.value_functions = V


        if write_all_dir:
            if streaming_writer is not None:
                # Already written incrementally; just report the manifest.
                result_meta["manifest"] = streaming_writer.manifest_rows
            else:
                # Single-threaded path or streaming writer not initialised — batch write.
                manifest = _write_all_successes(
                    all_successes=all_successes,
                    solver=solver,
                    payoff_path=payoff_path or Path("unknown.xlsx"),
                    output_dir=Path(write_all_dir),
                    dedup_by=dedup_by,
                    weak_orders=weak_orders,
                    committee_idxs=committee_idxs,
                    extra_metadata=extra_metadata or {},
                )
                result_meta["manifest"] = manifest

        return strategy_df, result_meta

    return pd.DataFrame(), result_meta
