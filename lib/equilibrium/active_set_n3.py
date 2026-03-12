"""Active-set equilibrium search for 3-player cases."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from lib.equilibrium.solver import EquilibriumSolver


@dataclass
class ActiveSetDiagnostics:
    cycle_period: int | None
    unstable_rows: int
    unstable_approvals: int
    candidate_combinations: int
    accepted_candidate: int | None


def _product_size(options: List[List[Any]]) -> int:
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
                for approver in solver.players:
                    key = (proposer, current_state, next_state, approver)
                    if key in solver.r_acceptances:
                        acceptance_keys.append(key)
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
            cycle_arrays = seen_arrays[cycle_start_idx:]
            return _analyze_cycle(solver, proposal_keys, acceptance_keys, cycle_arrays)

        seen_signatures[signature] = len(seen_arrays)
        seen_arrays.append((p_arr.copy(), r_arr.copy()))

        if prev_p is not None and prev_r is not None:
            if np.allclose(p_arr, prev_p, rtol=0, atol=tol) and np.allclose(r_arr, prev_r, rtol=0, atol=tol):
                return None
        prev_p = p_arr
        prev_r = r_arr

    return None


def _analyze_cycle(
    solver: EquilibriumSolver,
    proposal_keys: List[tuple],
    acceptance_keys: List[tuple],
    cycle_arrays: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    row_support_options: Dict[tuple, List[tuple]] = {}
    approval_options: Dict[tuple, List[float]] = {}

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

    r_matrix = np.stack([r_arr for _, r_arr in cycle_arrays], axis=0)
    for i, key in enumerate(acceptance_keys):
        vals = sorted({float(round(v, 12)) for v in r_matrix[:, i]})
        if len(vals) > 1:
            approval_options[key] = vals

    return {
        "cycle_period": len(cycle_arrays),
        "cycle_arrays": cycle_arrays,
        "row_support_options": row_support_options,
        "approval_options": approval_options,
    }


def _apply_candidate(
    solver: EquilibriumSolver,
    base_p: np.ndarray,
    base_r: np.ndarray,
    proposal_keys: List[tuple],
    acceptance_keys: List[tuple],
    row_items: List[Tuple[tuple, List[tuple]]],
    support_choice: Tuple[tuple, ...],
    approval_items: List[Tuple[tuple, List[float]]],
    approval_choice: Tuple[float, ...],
) -> None:
    _set_strategy_from_arrays(solver, base_p.copy(), base_r.copy(), proposal_keys, acceptance_keys)

    for ((proposer, current_state), _supports), support in zip(row_items, support_choice):
        support = tuple(support)
        total = 0.0
        for next_state in solver.states:
            key = (proposer, current_state, next_state)
            if next_state in support:
                total += solver.p_proposals[key]
        for next_state in solver.states:
            key = (proposer, current_state, next_state)
            if next_state in support:
                if total > 0:
                    solver.p_proposals[key] = solver.p_proposals[key] / total
                else:
                    solver.p_proposals[key] = 1.0 / len(support)
            else:
                solver.p_proposals[key] = 0.0

    for (key, _values), value in zip(approval_items, approval_choice):
        solver.r_acceptances[key] = float(value)


def _refine_candidate(
    solver: EquilibriumSolver,
    row_items: List[Tuple[tuple, List[tuple]]],
    support_choice: Tuple[tuple, ...],
    approval_items: List[Tuple[tuple, List[float]]],
    approval_choice: Tuple[float, ...],
    refinement_iter: int,
) -> None:
    support_map = {row_key: tuple(support) for (row_key, _supports), support in zip(row_items, support_choice)}
    approval_choice_map = {key: float(value) for (key, _values), value in zip(approval_items, approval_choice)}

    for _ in range(refinement_iter):
        strategy_df = solver._create_strategy_dataframe()
        P, _, P_approvals = solver._compute_transition_probabilities(strategy_df)
        V = solver._solve_value_functions(P)

        # Keep proposal supports fixed but update weights only within support using current expected values.
        for (proposer, current_state), support in support_map.items():
            expected_values = {}
            for next_state in solver.states:
                p_approved = P_approvals[(proposer, current_state, next_state)]
                p_rejected = 1.0 - p_approved
                V_current = V.loc[current_state, proposer]
                V_next = V.loc[next_state, proposer]
                expected_values[next_state] = p_approved * V_next + p_rejected * V_current

            support_vals = {ns: expected_values[ns] for ns in support}
            max_support = max(support_vals.values())
            winners = [ns for ns, val in support_vals.items() if np.isclose(val, max_support, atol=1e-9)]
            for next_state in solver.states:
                key = (proposer, current_state, next_state)
                if next_state in winners:
                    solver.p_proposals[key] = 1.0 / len(winners)
                else:
                    solver.p_proposals[key] = 0.0

        # Freeze enumerated unstable approvals; update all other approvals exactly.
        for proposer in solver.players:
            for current_state in solver.states:
                for next_state in solver.states:
                    for approver in solver.players:
                        key = (proposer, current_state, next_state, approver)
                        if key not in solver.r_acceptances:
                            continue
                        if key in approval_choice_map:
                            solver.r_acceptances[key] = approval_choice_map[key]
                            continue
                        V_current = V.loc[current_state, approver]
                        V_next = V.loc[next_state, approver]
                        if np.isclose(V_next, V_current, rtol=0, atol=1e-9):
                            continue
                        solver.r_acceptances[key] = 1.0 if V_next > V_current else 0.0


def solve_with_active_set_n3(
    solver: EquilibriumSolver,
    max_candidates: int = 1024,
    refinement_iter: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attempt a 3-player equilibrium solve using a cycle-guided active-set search."""
    cycle_info = _capture_exact_cycle(solver)
    if cycle_info is None:
        strategy_df = solver._create_strategy_dataframe()
        success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
        diagnostics = ActiveSetDiagnostics(
            cycle_period=None,
            unstable_rows=0,
            unstable_approvals=0,
            candidate_combinations=0,
            accepted_candidate=0 if success else None,
        )
        return strategy_df, {
            "converged": success,
            "stopping_reason": "active_set_direct" if success else "active_set_failed",
            "outer_iterations": 0,
            "active_set": diagnostics.__dict__,
            "verification_message": message,
        }

    row_items = sorted(cycle_info["row_support_options"].items())
    approval_items = sorted(cycle_info["approval_options"].items())
    candidate_count = _product_size([supports for _, supports in row_items]) * _product_size([vals for _, vals in approval_items])

    diagnostics = ActiveSetDiagnostics(
        cycle_period=cycle_info["cycle_period"],
        unstable_rows=len(row_items),
        unstable_approvals=len(approval_items),
        candidate_combinations=candidate_count,
        accepted_candidate=None,
    )

    if candidate_count > max_candidates:
        strategy_df = solver._create_strategy_dataframe()
        success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
        return strategy_df, {
            "converged": success,
            "stopping_reason": "active_set_too_many_candidates",
            "outer_iterations": 0,
            "active_set": diagnostics.__dict__,
            "verification_message": message,
        }

    proposal_keys, acceptance_keys = _get_strategy_key_orders(solver)
    base_p, base_r = cycle_info["cycle_arrays"][-1]
    support_space = [supports for _, supports in row_items] or [tuple()]
    approval_space = [vals for _, vals in approval_items] or [tuple()]

    candidate_idx = 0
    for support_choice in product(*support_space):
        for approval_choice in product(*approval_space):
            candidate_idx += 1
            _apply_candidate(
                solver,
                base_p=base_p,
                base_r=base_r,
                proposal_keys=proposal_keys,
                acceptance_keys=acceptance_keys,
                row_items=row_items,
                support_choice=support_choice,
                approval_items=approval_items,
                approval_choice=approval_choice,
            )
            _refine_candidate(
                solver,
                row_items=row_items,
                support_choice=support_choice,
                approval_items=approval_items,
                approval_choice=approval_choice,
                refinement_iter=refinement_iter,
            )
            strategy_df = solver._create_strategy_dataframe()
            success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
            if success:
                diagnostics.accepted_candidate = candidate_idx
                return strategy_df, {
                    "converged": True,
                    "stopping_reason": "active_set_verified",
                    "outer_iterations": 0,
                    "active_set": diagnostics.__dict__,
                    "verification_message": message,
                }

    strategy_df = solver._create_strategy_dataframe()
    success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
    return strategy_df, {
        "converged": success,
        "stopping_reason": "active_set_failed",
        "outer_iterations": 0,
        "active_set": diagnostics.__dict__,
        "verification_message": message,
    }
