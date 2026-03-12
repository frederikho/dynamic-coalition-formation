"""Cycle-guided support enumeration for 3-player equilibrium search."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from lib.equilibrium.solver import EquilibriumSolver


@dataclass
class SupportEnumerationDiagnostics:
    cycle_period: int | None
    unstable_rows: int
    candidate_support_combinations: int
    accepted_support_combination: int | None


def _powerset_product_size(options: List[List[Any]]) -> int:
    size = 1
    for option_list in options:
        size *= max(1, len(option_list))
    return size


def _get_strategy_key_orders(solver: EquilibriumSolver) -> Tuple[List[tuple], List[tuple]]:
    proposal_keys = []
    for proposer in solver.players:
        for current_state in solver.states:
            for next_state in solver.states:
                proposal_keys.append((proposer, current_state, next_state))

    acceptance_keys = []
    for proposer in solver.players:
        for current_state in solver.states:
            for next_state in solver.states:
                committee = solver.effectivity
                for approver in solver.players:
                    if committee.get((proposer, current_state, next_state, approver), 0) == 1:
                        acceptance_keys.append((proposer, current_state, next_state, approver))
    return proposal_keys, acceptance_keys


def _get_strategy_arrays(
    solver: EquilibriumSolver,
    proposal_keys: List[tuple],
    acceptance_keys: List[tuple],
) -> Tuple[np.ndarray, np.ndarray]:
    p_arr = np.fromiter((solver.p_proposals[k] for k in proposal_keys), dtype=np.float64)
    r_arr = np.fromiter((solver.r_acceptances[k] for k in acceptance_keys), dtype=np.float64)
    return p_arr, r_arr


def _set_strategy_from_arrays(
    solver: EquilibriumSolver,
    p_arr: np.ndarray,
    r_arr: np.ndarray,
    proposal_keys: List[tuple],
    acceptance_keys: List[tuple],
) -> None:
    for i, key in enumerate(proposal_keys):
        solver.p_proposals[key] = float(p_arr[i])
    for i, key in enumerate(acceptance_keys):
        solver.r_acceptances[key] = float(r_arr[i])


def _row_support_signature(solver: EquilibriumSolver, p_arr: np.ndarray) -> Dict[tuple, tuple]:
    row_supports = {}
    idx = 0
    for proposer in solver.players:
        for current_state in solver.states:
            support = []
            for next_state in solver.states:
                if p_arr[idx] > 1e-9:
                    support.append(next_state)
                idx += 1
            row_supports[(proposer, current_state)] = tuple(support)
    return row_supports


def _analyze_cycle(
    solver: EquilibriumSolver,
    seen_arrays: List[Tuple[np.ndarray, np.ndarray]],
    cycle_start_idx: int,
) -> Dict[str, Any]:
    cycle_arrays = seen_arrays[cycle_start_idx:]
    cycle_period = len(cycle_arrays)
    row_support_options: Dict[tuple, List[tuple]] = {}

    row_support_history = {}
    for p_arr, _ in cycle_arrays:
        row_supports = _row_support_signature(solver, p_arr)
        for row_key, support in row_supports.items():
            row_support_history.setdefault(row_key, [])
            if support not in row_support_history[row_key]:
                row_support_history[row_key].append(support)

    for row_key, supports in row_support_history.items():
        if len(supports) > 1:
            row_support_options[row_key] = supports

    return {
        "cycle_period": cycle_period,
        "cycle_arrays": cycle_arrays,
        "row_support_options": row_support_options,
    }


def _project_acceptances_only(solver: EquilibriumSolver, V: pd.DataFrame) -> bool:
    changed = False
    for proposer in solver.players:
        for current_state in solver.states:
            for next_state in solver.states:
                for approver in solver.players:
                    key = (proposer, current_state, next_state, approver)
                    if key not in solver.r_acceptances:
                        continue
                    current_val = solver.r_acceptances[key]
                    V_current = V.loc[current_state, approver]
                    V_next = V.loc[next_state, approver]
                    if np.isclose(V_next, V_current, rtol=0, atol=1e-9):
                        continue
                    new_val = 1.0 if V_next > V_current else 0.0
                    if abs(new_val - current_val) > 1e-12:
                        solver.r_acceptances[key] = new_val
                        changed = True
    return changed


def _capture_exact_cycle(
    solver: EquilibriumSolver,
    max_iter: int = 20,
    tol: float = 1e-9,
) -> Dict[str, Any] | None:
    proposal_keys, acceptance_keys = _get_strategy_key_orders(solver)
    seen_signatures = {}
    seen_arrays: List[Tuple[np.ndarray, np.ndarray]] = []
    prev_p = None
    prev_r = None

    for _ in range(1, max_iter + 1):
        strategy_df = solver._create_strategy_dataframe()
        P, _, _ = solver._compute_transition_probabilities(strategy_df)
        V = solver._solve_value_functions(P)
        solver._project_to_exact_equilibrium(V)

        p_arr, r_arr = _get_strategy_arrays(solver, proposal_keys, acceptance_keys)
        signature = (
            np.round(p_arr, 12).tobytes(),
            np.round(r_arr, 12).tobytes(),
        )

        if signature in seen_signatures:
            cycle_start_idx = seen_signatures[signature]
            return _analyze_cycle(solver, seen_arrays, cycle_start_idx)

        seen_signatures[signature] = len(seen_arrays)
        seen_arrays.append((p_arr.copy(), r_arr.copy()))

        if prev_p is not None and prev_r is not None:
            if np.allclose(p_arr, prev_p, rtol=0, atol=tol) and np.allclose(r_arr, prev_r, rtol=0, atol=tol):
                return None
        prev_p = p_arr
        prev_r = r_arr

    return None


def solve_with_support_enumeration_n3(
    solver: EquilibriumSolver,
    max_cycle_candidates: int = 512,
    acceptance_fixpoint_iter: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attempt a 3-player equilibrium solve using cycle-guided support enumeration."""
    cycle_info = _capture_exact_cycle(solver)
    if cycle_info is None:
        strategy_df = solver._create_strategy_dataframe()
        success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
        diagnostics = SupportEnumerationDiagnostics(
            cycle_period=None,
            unstable_rows=0,
            candidate_support_combinations=0,
            accepted_support_combination=0 if success else None,
        )
        return strategy_df, {
            "converged": success,
            "stopping_reason": "support_enumeration_direct" if success else "support_enumeration_failed",
            "outer_iterations": 0,
            "support_enumeration": diagnostics.__dict__,
            "verification_message": message,
        }

    row_items = sorted(cycle_info["row_support_options"].items())
    support_options = [supports for _, supports in row_items]
    candidate_count = _powerset_product_size(support_options)
    diagnostics = SupportEnumerationDiagnostics(
        cycle_period=cycle_info["cycle_period"],
        unstable_rows=len(row_items),
        candidate_support_combinations=candidate_count,
        accepted_support_combination=None,
    )
    if candidate_count > max_cycle_candidates:
        strategy_df = solver._create_strategy_dataframe()
        success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
        return strategy_df, {
            "converged": success,
            "stopping_reason": "support_enumeration_too_many_candidates",
            "outer_iterations": 0,
            "support_enumeration": diagnostics.__dict__,
            "verification_message": message,
        }

    proposal_keys, acceptance_keys = _get_strategy_key_orders(solver)
    baseline_cycle_p, baseline_cycle_r = cycle_info["cycle_arrays"][-1]

    for candidate_idx, support_choice in enumerate(product(*support_options), start=1):
        _set_strategy_from_arrays(
            solver,
            baseline_cycle_p.copy(),
            baseline_cycle_r.copy(),
            proposal_keys,
            acceptance_keys,
        )
        for ((proposer, current_state), _supports), support in zip(row_items, support_choice):
            support = tuple(support)
            for next_state in solver.states:
                key = (proposer, current_state, next_state)
                solver.p_proposals[key] = 1.0 / len(support) if next_state in support else 0.0

        for _ in range(acceptance_fixpoint_iter):
            strategy_df = solver._create_strategy_dataframe()
            P, _, _ = solver._compute_transition_probabilities(strategy_df)
            V = solver._solve_value_functions(P)
            changed = _project_acceptances_only(solver, V)
            if not changed:
                break

        strategy_df = solver._create_strategy_dataframe()
        success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
        if success:
            diagnostics.accepted_support_combination = candidate_idx
            return strategy_df, {
                "converged": True,
                "stopping_reason": "support_enumeration_verified",
                "outer_iterations": 0,
                "support_enumeration": diagnostics.__dict__,
                "verification_message": message,
            }

    strategy_df = solver._create_strategy_dataframe()
    success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
    return strategy_df, {
        "converged": success,
        "stopping_reason": "support_enumeration_failed",
        "outer_iterations": 0,
        "support_enumeration": diagnostics.__dict__,
        "verification_message": message,
    }
