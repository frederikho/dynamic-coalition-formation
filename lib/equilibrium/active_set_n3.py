"""Active-set equilibrium search."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import get_approval_committee


@dataclass
class ActiveSetDiagnostics:
    cycle_period: int | None
    unstable_rows: int
    unstable_approvals: int
    candidate_combinations: int
    accepted_candidate: int | None
    expansion_rounds: int = 0
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    seeded_row_supports: Dict[str, List[str]] = field(default_factory=dict)
    basin_signature: Dict[str, Any] = field(default_factory=dict)


def _format_violation_short(violation: Dict[str, Any] | None) -> str:
    if not violation:
        return "none"
    if violation["type"] == "proposal":
        return f"proposal {violation['proposer']} @ {violation['current_state']}"
    if violation["type"] == "approval":
        return (
            "approval "
            f"{violation['approver']} on "
            f"{violation['proposer']}:{violation['current_state']}->{violation['next_state']}"
        )
    return str(violation)


def _format_approval_key_short(key: tuple) -> str:
    proposer, current_state, next_state, approver = key
    return f"{approver} on {proposer}:{current_state}->{next_state}"


def _build_basin_signature(cycle_info: Dict[str, Any] | None) -> Dict[str, Any]:
    if cycle_info is None:
        return {
            "kind": "no_cycle",
            "cycle_period": None,
            "unstable_rows": [],
            "top_approval_flips": [],
        }

    unstable_rows = sorted(
        f"{proposer} @ {current_state}"
        for proposer, current_state in cycle_info.get("row_support_options", {}).keys()
    )
    top_approval_flips = [
        _format_approval_key_short(key)
        for key, _count in sorted(
            cycle_info.get("approval_flip_counts", {}).items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ]
    return {
        "kind": "cycle",
        "cycle_period": cycle_info.get("cycle_period"),
        "unstable_rows": unstable_rows,
        "top_approval_flips": top_approval_flips,
    }


def _minimal_proposal_supports(violation: Dict[str, Any]) -> List[tuple]:
    """Generate a small support menu for a violated proposal row.

    The verifier tells us the currently positive support and the argmax set.
    To avoid candidate explosion, only add:
    - singleton supports on each argmax state
    - repaired supports that keep any already-valid overlap and add one argmax
    """
    positive_states = list(violation.get("positive_states", []))
    argmax_states = list(violation.get("argmax_states", []))
    current_support = set(positive_states)
    argmax_set = set(argmax_states)
    valid_overlap = sorted(current_support & argmax_set)

    candidates: List[tuple] = []
    for state in argmax_states:
        singleton = (state,)
        if singleton not in candidates:
            candidates.append(singleton)

        repaired = tuple(sorted(set(valid_overlap) | {state}))
        if repaired and repaired not in candidates:
            candidates.append(repaired)

    return candidates


def _product_size(options: List[List[Any]]) -> int:
    size = 1
    for option_list in options:
        size *= max(1, len(option_list))
    return size


def _candidate_count(row_items: List[Tuple[tuple, List[tuple]]],
                     approval_items: List[Tuple[tuple, List[float]]]) -> int:
    return _product_size([supports for _, supports in row_items]) * _product_size([vals for _, vals in approval_items])


def _is_self_loop_approval_key(key: tuple) -> bool:
    _proposer, current_state, next_state, _approver = key
    return current_state == next_state


def _is_proposer_only_approval_key(solver: EquilibriumSolver, key: tuple) -> bool:
    proposer, current_state, next_state, approver = key
    if approver != proposer:
        return False
    committee = get_approval_committee(
        solver.effectivity,
        solver.players,
        proposer,
        current_state,
        next_state,
    )
    return len(committee) == 1 and committee[0] == proposer


def _compress_approval_items(
    solver: EquilibriumSolver,
    row_items: List[Tuple[tuple, List[tuple]]],
    approval_items: List[Tuple[tuple, List[float]]],
) -> List[Tuple[tuple, List[float]]]:
    active_rows = {row_key for row_key, _supports in row_items}
    compressed: List[Tuple[tuple, List[float]]] = []
    for key, values in approval_items:
        if _is_self_loop_approval_key(key):
            continue
        row_key = (key[0], key[1])
        if row_key in active_rows and _is_proposer_only_approval_key(solver, key):
            continue
        compressed.append((key, values))
    return compressed


def _apply_linked_proposer_only_approvals(
    solver: EquilibriumSolver,
    support_map: Dict[tuple, tuple],
) -> None:
    for (proposer, current_state), support in support_map.items():
        support_set = set(support)
        for next_state in solver.states:
            key = (proposer, current_state, next_state, proposer)
            if key not in solver.r_acceptances:
                continue
            if not _is_proposer_only_approval_key(solver, key):
                continue
            if _is_self_loop_approval_key(key):
                continue
            solver.r_acceptances[key] = 1.0 if next_state in support_set else 0.0


def _trim_initial_active_set(
    row_items: List[Tuple[tuple, List[tuple]]],
    approval_items: List[Tuple[tuple, List[float]]],
    row_counts: Dict[tuple, int],
    approval_counts: Dict[tuple, int],
    max_candidates: int,
) -> Tuple[List[Tuple[tuple, List[tuple]]], List[Tuple[tuple, List[float]]]]:
    """Greedily trim the seeded active set until it fits the candidate budget.

    Preference: keep high-frequency items unless they create disproportionate branching.
    """
    row_items = list(row_items)
    approval_items = list(approval_items)

    def item_priority(count: int, n_options: int) -> float:
        if n_options <= 1:
            return float("inf")
        return count / np.log2(n_options + 1.0)

    while _candidate_count(row_items, approval_items) > max_candidates:
        candidates = []
        for idx, (row_key, supports) in enumerate(row_items):
            if len(row_items) > 1 and len(supports) > 1:
                candidates.append(("row", idx, item_priority(row_counts.get(row_key, 1), len(supports))))
        for idx, (app_key, values) in enumerate(approval_items):
            if len(approval_items) > 1 and len(values) > 1:
                candidates.append(("approval", idx, item_priority(approval_counts.get(app_key, 1), len(values))))

        if not candidates:
            break

        kind, idx, _priority = min(candidates, key=lambda item: item[2])
        if kind == "row":
            row_items.pop(idx)
        else:
            approval_items.pop(idx)

    return row_items, approval_items


def _trim_support_options_per_row(
    row_items: List[Tuple[tuple, List[tuple]]],
    row_option_counts: Dict[tuple, Dict[tuple, int]],
    max_supports_per_row: int,
) -> List[Tuple[tuple, List[tuple]]]:
    trimmed = []
    for row_key, supports in row_items:
        if max_supports_per_row is None or len(supports) <= max_supports_per_row:
            trimmed.append((row_key, supports))
            continue
        counts = row_option_counts.get(row_key, {})
        ranked = sorted(
            supports,
            key=lambda support: (
                len(support) not in (1, 2),
                -counts.get(support, 0),
                len(support),
                support,
            ),
        )
        trimmed.append((row_key, ranked[:max_supports_per_row]))
    return trimmed


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


def _project_to_exact_equilibrium_fast(
    solver: EquilibriumSolver,
    V: pd.DataFrame,
) -> None:
    """Probe-only exact projection using array indexing instead of DataFrame lookups."""
    state_to_idx = {state: idx for idx, state in enumerate(solver.states)}
    player_to_idx = {player: idx for idx, player in enumerate(solver.players)}
    V_array = V.loc[solver.states, solver.players].to_numpy(dtype=np.float64, copy=False)

    # Project acceptances.
    for proposer in solver.players:
        for current_state in solver.states:
            current_idx = state_to_idx[current_state]
            for next_state in solver.states:
                next_idx = state_to_idx[next_state]
                committee = solver.tp.approval_committees[
                    player_to_idx[proposer]
                ][current_idx][next_idx] if solver.tp is not None else None
                if committee is None:
                    continue
                for approver_idx in committee:
                    approver = solver.players[approver_idx]
                    key = (proposer, current_state, next_state, approver)
                    V_current = V_array[current_idx, approver_idx]
                    V_next = V_array[next_idx, approver_idx]
                    if np.isclose(V_next, V_current, rtol=0, atol=1e-9):
                        continue
                    solver.r_acceptances[key] = 1.0 if V_next > V_current else 0.0

    # Recompute approvals after exact acceptance projection using the fast path.
    P, _, P_approvals = solver._compute_transition_probabilities_fast()

    # Project proposals.
    for proposer in solver.players:
        proposer_idx = player_to_idx[proposer]
        for current_state in solver.states:
            current_idx = state_to_idx[current_state]
            V_current = V_array[current_idx, proposer_idx]
            expected_values = {}
            for next_state in solver.states:
                next_idx = state_to_idx[next_state]
                p_approved = P_approvals[(proposer, current_state, next_state)]
                V_next = V_array[next_idx, proposer_idx]
                expected_values[next_state] = p_approved * V_next + (1.0 - p_approved) * V_current

            max_value = max(expected_values.values())
            argmax_states = [
                state for state, value in expected_values.items()
                if np.isclose(value, max_value, rtol=0.0, atol=1e-9)
            ]
            argmax_set = set(argmax_states)
            total_weight = sum(
                solver.p_proposals[(proposer, current_state, next_state)]
                for next_state in argmax_set
            )
            for next_state in solver.states:
                key = (proposer, current_state, next_state)
                if next_state in argmax_set:
                    if total_weight > 0:
                        solver.p_proposals[key] = solver.p_proposals[key] / total_weight
                    else:
                        solver.p_proposals[key] = 1.0 / len(argmax_set)
                else:
                    solver.p_proposals[key] = 0.0


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
        P, _, _ = solver._compute_transition_probabilities_fast()
        V = solver._solve_value_functions(P)
        _project_to_exact_equilibrium_fast(solver, V)

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


def _build_seed_row_items_from_current_profile(
    solver: EquilibriumSolver,
    seed_rows: List[Tuple[str, str]],
    max_supports_per_row: int | None,
) -> List[Tuple[tuple, List[tuple]]]:
    strategy_df = solver._create_strategy_dataframe()
    P, _P_proposals, P_approvals = solver._compute_transition_probabilities(strategy_df)
    V = solver._solve_value_functions(P)

    row_items: List[Tuple[tuple, List[tuple]]] = []
    row_option_counts: Dict[tuple, Dict[tuple, int]] = {}

    for proposer, current_state in seed_rows:
        positive_states = [
            next_state
            for next_state in solver.states
            if solver.p_proposals[(proposer, current_state, next_state)] > 1e-9
        ]
        expected_values = {}
        for next_state in solver.states:
            p_approved = P_approvals[(proposer, current_state, next_state)]
            p_rejected = 1.0 - p_approved
            V_current = float(V.loc[current_state, proposer])
            V_next = float(V.loc[next_state, proposer])
            expected_values[next_state] = p_approved * V_next + p_rejected * V_current

        max_value = max(expected_values.values())
        argmax_states = [
            next_state
            for next_state, value in expected_values.items()
            if np.isclose(value, max_value, rtol=0, atol=1e-9)
        ]

        violation = {
            "type": "proposal",
            "proposer": proposer,
            "current_state": current_state,
            "positive_states": positive_states,
            "argmax_states": argmax_states,
        }
        supports = _minimal_proposal_supports(violation)
        ranked_states = [
            state
            for state, _value in sorted(
                expected_values.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        top_states = ranked_states[: min(4, len(ranked_states))]
        for support_size in (1, 2, 3):
            if len(top_states) < support_size:
                continue
            for subset in combinations(top_states, support_size):
                subset = tuple(sorted(subset))
                if subset not in supports:
                    supports.append(subset)
        current_support = tuple(sorted(positive_states))
        if current_support and current_support not in supports:
            supports.append(current_support)
        if not supports and argmax_states:
            supports = [(argmax_states[0],)]

        row_key = (proposer, current_state)
        def support_score(support: tuple[str, ...]) -> tuple[float, int, tuple[str, ...]]:
            return (
                sum(expected_values[state] for state in support),
                -len(support),
                tuple(sorted(support)),
            )

        ranked_supports = sorted(
            {tuple(sorted(support)) for support in supports if support},
            key=support_score,
            reverse=True,
        )
        row_items.append((row_key, ranked_supports))
        row_option_counts[row_key] = {
            support: len(ranked_supports) - idx
            for idx, support in enumerate(ranked_supports)
        }

    return _trim_support_options_per_row(
        row_items,
        row_option_counts=row_option_counts,
        max_supports_per_row=max_supports_per_row,
    )


def _record_seeded_row_supports(
    diagnostics: ActiveSetDiagnostics,
    row_items: List[Tuple[tuple, List[tuple]]],
) -> None:
    diagnostics.seeded_row_supports = {
        f"{proposer} @ {current_state}": [
            ", ".join(support) if support else "(empty)"
            for support in supports
        ]
        for (proposer, current_state), supports in row_items
    }


def _analyze_cycle(
    solver: EquilibriumSolver,
    proposal_keys: List[tuple],
    acceptance_keys: List[tuple],
    cycle_arrays: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    row_support_options: Dict[tuple, List[tuple]] = {}
    row_support_option_counts: Dict[tuple, Dict[tuple, int]] = {}
    row_support_change_counts: Dict[tuple, int] = {}
    approval_options: Dict[tuple, List[float]] = {}
    approval_flip_counts: Dict[tuple, int] = {}

    row_support_history = {}
    row_support_signatures = []
    for p_arr, _ in cycle_arrays:
        row_supports = _row_support_signature(solver, p_arr)
        row_support_signatures.append(row_supports)
        for row_key, support in row_supports.items():
            row_support_history.setdefault(row_key, [])
            row_support_option_counts.setdefault(row_key, {})
            row_support_option_counts[row_key][support] = row_support_option_counts[row_key].get(support, 0) + 1
            if support not in row_support_history[row_key]:
                row_support_history[row_key].append(support)

    for idx in range(1, len(row_support_signatures)):
        prev = row_support_signatures[idx - 1]
        cur = row_support_signatures[idx]
        for row_key in prev:
            if prev[row_key] != cur[row_key]:
                row_support_change_counts[row_key] = row_support_change_counts.get(row_key, 0) + 1

    for row_key, supports in row_support_history.items():
        if len(supports) > 1:
            row_support_options[row_key] = supports

    r_matrix = np.stack([r_arr for _, r_arr in cycle_arrays], axis=0)
    for i, key in enumerate(acceptance_keys):
        vals = sorted({float(round(v, 12)) for v in r_matrix[:, i]})
        if len(vals) > 1:
            approval_options[key] = vals
    for idx in range(1, len(cycle_arrays)):
        prev_r = cycle_arrays[idx - 1][1]
        cur_r = cycle_arrays[idx][1]
        for i, key in enumerate(acceptance_keys):
            if abs(float(prev_r[i]) - float(cur_r[i])) > 1e-9:
                approval_flip_counts[key] = approval_flip_counts.get(key, 0) + 1

    return {
        "cycle_period": len(cycle_arrays),
        "cycle_arrays": cycle_arrays,
        "row_support_options": row_support_options,
        "row_support_option_counts": row_support_option_counts,
        "row_support_change_counts": row_support_change_counts,
        "approval_options": approval_options,
        "approval_flip_counts": approval_flip_counts,
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
    support_map: Dict[tuple, tuple] = {}

    for ((proposer, current_state), _supports), support in zip(row_items, support_choice):
        support = tuple(support)
        support_map[(proposer, current_state)] = support
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

    _apply_linked_proposer_only_approvals(solver, support_map)

    for (key, _values), value in zip(approval_items, approval_choice):
        solver.r_acceptances[key] = float(value)


def _refine_candidate(
    solver: EquilibriumSolver,
    row_items: List[Tuple[tuple, List[tuple]]],
    support_choice: Tuple[tuple, ...],
    approval_items: List[Tuple[tuple, List[float]]],
    approval_choice: Tuple[float, ...],
    refinement_iter: int,
    freeze_seeded_proposals: bool = False,
) -> None:
    support_map = {row_key: tuple(support) for (row_key, _supports), support in zip(row_items, support_choice)}
    approval_choice_map = {key: float(value) for (key, _values), value in zip(approval_items, approval_choice)}

    for _ in range(refinement_iter):
        strategy_df = solver._create_strategy_dataframe()
        P, _, P_approvals = solver._compute_transition_probabilities(strategy_df)
        V = solver._solve_value_functions(P)

        # Keep proposal supports fixed but update weights only within support using current expected values.
        for (proposer, current_state), support in support_map.items():
            if freeze_seeded_proposals:
                for next_state in solver.states:
                    key = (proposer, current_state, next_state)
                    solver.p_proposals[key] = 1.0 / len(support) if next_state in support else 0.0
                continue
            expected_values = {}
            for next_state in solver.states:
                p_approved = P_approvals[(proposer, current_state, next_state)]
                p_rejected = 1.0 - p_approved
                V_current = V.loc[current_state, proposer]
                V_next = V.loc[next_state, proposer]
                expected_values[next_state] = p_approved * V_next + p_rejected * V_current

            support_vals = {ns: expected_values[ns] for ns in support}
            max_support = max(support_vals.values())
            winners = [ns for ns, val in support_vals.items() if np.isclose(val, max_support, rtol=0.0, atol=1e-9)]
            for next_state in solver.states:
                key = (proposer, current_state, next_state)
                if next_state in winners:
                    solver.p_proposals[key] = 1.0 / len(winners)
                else:
                    solver.p_proposals[key] = 0.0

        _apply_linked_proposer_only_approvals(solver, support_map)

        # Freeze enumerated unstable approvals; update all other approvals exactly.
        for proposer in solver.players:
            for current_state in solver.states:
                for next_state in solver.states:
                    for approver in solver.players:
                        key = (proposer, current_state, next_state, approver)
                        if key not in solver.r_acceptances:
                            continue
                        if _is_self_loop_approval_key(key):
                            continue
                        if key in approval_choice_map:
                            solver.r_acceptances[key] = approval_choice_map[key]
                            continue
                        if (proposer, current_state) in support_map and _is_proposer_only_approval_key(solver, key):
                            continue
                        V_current = V.loc[current_state, approver]
                        V_next = V.loc[next_state, approver]
                        if np.isclose(V_next, V_current, rtol=0, atol=1e-9):
                            continue
                        solver.r_acceptances[key] = 1.0 if V_next > V_current else 0.0


def _enumerate_active_candidates(
    solver: EquilibriumSolver,
    row_items: List[Tuple[tuple, List[tuple]]],
    approval_items: List[Tuple[tuple, List[float]]],
    base_p: np.ndarray,
    base_r: np.ndarray,
    proposal_keys: List[tuple],
    acceptance_keys: List[tuple],
    refinement_iter: int,
    max_candidates: int,
    max_candidates_per_round: int | None,
    diagnostics: ActiveSetDiagnostics,
    freeze_seeded_proposals: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any] | None]:
    candidate_count = _product_size([supports for _, supports in row_items]) * _product_size([vals for _, vals in approval_items])
    diagnostics.unstable_rows = len(row_items)
    diagnostics.unstable_approvals = len(approval_items)
    diagnostics.candidate_combinations = candidate_count
    round_record = {
        "round": diagnostics.expansion_rounds,
        "rows": diagnostics.unstable_rows,
        "approvals": diagnostics.unstable_approvals,
        "candidates": candidate_count,
    }
    diagnostics.round_history.append(round_record)

    if solver.verbose:
        solver._log(
            f"Active-set round {diagnostics.expansion_rounds}: "
            f"rows={diagnostics.unstable_rows}, "
            f"approvals={diagnostics.unstable_approvals}, "
            f"candidates={candidate_count}"
        )

    if candidate_count > max_candidates:
        strategy_df = solver._create_strategy_dataframe()
        success, message, _V, detail = solver._verify_candidate_equilibrium_detailed(strategy_df)
        round_record["stopping_reason"] = "active_set_too_many_candidates"
        round_record["first_violation"] = detail
        if solver.verbose:
            solver._log(
                "Active-set stopped: "
                f"candidate_count={candidate_count} exceeds active_set_max_candidates={max_candidates}",
                level="warning",
            )
        return strategy_df, {
            "converged": success,
            "stopping_reason": "active_set_too_many_candidates",
            "outer_iterations": 0,
            "active_set": diagnostics.__dict__,
            "verification_message": message,
        }, detail

    support_space = [supports for _, supports in row_items] or [tuple()]
    approval_space = [vals for _, vals in approval_items] or [tuple()]
    candidate_idx = 0
    last_detail = None
    violation_counter: Counter[str] = Counter()
    violation_examples: Dict[str, Dict[str, Any]] = {}

    for support_choice in product(*support_space):
        for approval_choice in product(*approval_space):
            candidate_idx += 1
            if max_candidates_per_round is not None and candidate_idx > max_candidates_per_round:
                strategy_df = solver._create_strategy_dataframe()
                dominant_detail = None
                if violation_counter:
                    dominant_label, _count = violation_counter.most_common(1)[0]
                    dominant_detail = violation_examples.get(dominant_label)
                round_record["evaluated_candidates"] = candidate_idx - 1
                round_record["stopping_reason"] = "active_set_round_budget_exhausted"
                round_record["first_violation"] = dominant_detail
                if solver.verbose:
                    solver._log(
                        "Active-set round budget hit: "
                        f"evaluated={candidate_idx - 1}, "
                        f"budget={max_candidates_per_round}, "
                        f"dominant_violation={_format_violation_short(dominant_detail)}",
                        level="warning",
                    )
                return strategy_df, {
                    "converged": False,
                    "stopping_reason": "active_set_round_budget_exhausted",
                    "outer_iterations": 0,
                    "active_set": diagnostics.__dict__,
                    "verification_message": "Round budget exhausted before verification success.",
                }, dominant_detail
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
                freeze_seeded_proposals=freeze_seeded_proposals,
            )
            strategy_df = solver._create_strategy_dataframe()
            success, message, _V, detail = solver._verify_candidate_equilibrium_detailed(strategy_df)
            last_detail = detail
            if detail:
                label = _format_violation_short(detail)
                violation_counter[label] += 1
                violation_examples[label] = detail
            if success:
                diagnostics.accepted_candidate = candidate_idx
                round_record["evaluated_candidates"] = candidate_idx
                round_record["stopping_reason"] = "active_set_verified"
                round_record["first_violation"] = detail
                return strategy_df, {
                    "converged": True,
                    "stopping_reason": "active_set_verified",
                    "outer_iterations": 0,
                    "active_set": diagnostics.__dict__,
                    "verification_message": message,
                }, detail

    strategy_df = solver._create_strategy_dataframe()
    success, message, _V, detail = solver._verify_candidate_equilibrium_detailed(strategy_df)
    last_detail = detail or last_detail
    round_record["evaluated_candidates"] = candidate_idx
    round_record["stopping_reason"] = "active_set_failed"
    round_record["first_violation"] = last_detail
    return strategy_df, {
        "converged": success,
        "stopping_reason": "active_set_failed",
        "outer_iterations": 0,
        "active_set": diagnostics.__dict__,
        "verification_message": message,
    }, last_detail


def _expand_active_sets_from_violation(
    solver: EquilibriumSolver,
    strategy_df: pd.DataFrame,
    row_items: List[Tuple[tuple, List[tuple]]],
    approval_items: List[Tuple[tuple, List[float]]],
    violation: Dict[str, Any],
) -> bool:
    if not violation:
        return False

    if violation["type"] == "proposal":
        row_key = (violation["proposer"], violation["current_state"])
        existing = {key: list(options) for key, options in row_items}
        options = existing.get(row_key, [])
        changed = False
        for support in _minimal_proposal_supports(violation):
            if support and support not in options:
                options.append(support)
                changed = True
        if changed:
            existing[row_key] = options
            row_items[:] = sorted(existing.items())
        return changed

    if violation["type"] == "approval":
        key = (
            violation["proposer"],
            violation["current_state"],
            violation["next_state"],
            violation["approver"],
        )
        if _is_self_loop_approval_key(key):
            return False
        active_rows = {row_key for row_key, _supports in row_items}
        if (
            (violation["proposer"], violation["current_state"]) in active_rows
            and _is_proposer_only_approval_key(solver, key)
        ):
            return False
        existing = {akey: list(options) for akey, options in approval_items}
        if key not in existing:
            existing[key] = [0.0, 1.0]
            approval_items[:] = sorted(existing.items())
            return True
        return False

    return False


def solve_with_active_set_n3(
    solver: EquilibriumSolver,
    max_candidates: int = 1024,
    refinement_iter: int = 10,
    max_initial_approvals: int = 8,
    max_initial_rows: int = 2,
    max_supports_per_row: int = 2,
    max_candidates_per_round: int | None = 256,
    seed_rows: List[Tuple[str, str]] | None = None,
    freeze_seeded_proposals: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attempt an equilibrium solve using a cycle-guided active-set search."""
    if seed_rows:
        cycle_info = None
    else:
        cycle_info = _capture_exact_cycle(solver)

    if cycle_info is None and not seed_rows:
        strategy_df = solver._create_strategy_dataframe()
        success, message, _ = solver._verify_candidate_equilibrium(strategy_df)
        diagnostics = ActiveSetDiagnostics(
            cycle_period=None,
            unstable_rows=0,
            unstable_approvals=0,
            candidate_combinations=0,
            accepted_candidate=0 if success else None,
        )
        diagnostics.basin_signature = _build_basin_signature(None)
        return strategy_df, {
            "converged": success,
            "stopping_reason": "active_set_direct" if success else "active_set_failed",
            "outer_iterations": 0,
            "active_set": diagnostics.__dict__,
            "verification_message": message,
        }

    if cycle_info is None:
        diagnostics = ActiveSetDiagnostics(
            cycle_period=None,
            unstable_rows=0,
            unstable_approvals=0,
            candidate_combinations=0,
            accepted_candidate=None,
        )
        diagnostics.basin_signature = {
            "kind": "seeded_no_cycle",
            "cycle_period": None,
            "unstable_rows": [f"{p} @ {s}" for p, s in (seed_rows or [])],
            "top_approval_flips": [],
        }
        if solver.verbose:
            solver._log(
                "Active set start: "
                "cycle_period=None, "
                f"rows={len(seed_rows or [])}, "
                "approvals=0"
            )

        proposal_keys, acceptance_keys = _get_strategy_key_orders(solver)
        base_p, base_r = _get_strategy_arrays(solver, proposal_keys, acceptance_keys)
        row_items = _build_seed_row_items_from_current_profile(
            solver,
            seed_rows=seed_rows or [],
            max_supports_per_row=max_supports_per_row,
        )
        _record_seeded_row_supports(diagnostics, row_items)
        approval_items: List[Tuple[tuple, List[float]]] = []

        max_expansion_rounds = 3
        for expansion_round in range(max_expansion_rounds + 1):
            diagnostics.expansion_rounds = expansion_round
            strategy_df, result, detail = _enumerate_active_candidates(
                solver=solver,
                row_items=row_items,
                approval_items=approval_items,
                base_p=base_p,
                base_r=base_r,
                proposal_keys=proposal_keys,
                acceptance_keys=acceptance_keys,
                refinement_iter=refinement_iter,
                max_candidates=max_candidates,
                max_candidates_per_round=max_candidates_per_round,
                diagnostics=diagnostics,
                freeze_seeded_proposals=freeze_seeded_proposals,
            )
            if result["converged"] or result["stopping_reason"] in {"active_set_too_many_candidates", "active_set_round_budget_exhausted"}:
                return strategy_df, result

            expanded = _expand_active_sets_from_violation(
                solver=solver,
                strategy_df=strategy_df,
                row_items=row_items,
                approval_items=approval_items,
                violation=detail or {},
            )
            if diagnostics.round_history:
                diagnostics.round_history[-1]["post_round_action"] = "expanded" if expanded else "stalled"
                diagnostics.round_history[-1]["post_round_violation"] = detail
            if solver.verbose:
                action = "expanded" if expanded else "stalled"
                solver._log(
                    f"Active-set {action} after round {expansion_round}: "
                    f"{_format_violation_short(detail)}"
                )
            if not expanded:
                return strategy_df, result

        result["stopping_reason"] = "active_set_expansion_failed"
        result["active_set"] = diagnostics.__dict__
        return strategy_df, result

    diagnostics = ActiveSetDiagnostics(
        cycle_period=cycle_info["cycle_period"],
        unstable_rows=0,
        unstable_approvals=0,
        candidate_combinations=0,
        accepted_candidate=None,
    )
    diagnostics.basin_signature = _build_basin_signature(cycle_info)

    if solver.verbose:
        solver._log(
            "Active set start: "
            f"cycle_period={diagnostics.cycle_period}, "
            f"rows={len(cycle_info['row_support_options'])}, "
            f"approvals={len(cycle_info['approval_options'])}"
        )

    proposal_keys, acceptance_keys = _get_strategy_key_orders(solver)
    base_p, base_r = cycle_info["cycle_arrays"][-1]
    row_support_change_counts = cycle_info.get("row_support_change_counts", {})
    row_support_option_counts = cycle_info.get("row_support_option_counts", {})
    row_support_options = cycle_info["row_support_options"]
    if seed_rows:
        requested_seed_rows = [tuple(row_key) for row_key in seed_rows]
        recurrent_row_items = [
            (row_key, row_support_options[row_key])
            for row_key in requested_seed_rows
            if row_key in row_support_options
        ]
    else:
        recurrent_row_items = [
            (key, row_support_options[key])
            for key, count in sorted(
                row_support_change_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count > 1 and key in row_support_options
        ]
        if not recurrent_row_items:
            recurrent_row_items = sorted(row_support_options.items())
        if max_initial_rows is not None:
            recurrent_row_items = recurrent_row_items[:max_initial_rows]
    row_items = _trim_support_options_per_row(
        recurrent_row_items,
        row_option_counts=row_support_option_counts,
        max_supports_per_row=max_supports_per_row,
    )
    _record_seeded_row_supports(diagnostics, row_items)
    approval_flip_counts = cycle_info.get("approval_flip_counts", {})
    approval_options = cycle_info["approval_options"]
    recurrent_approval_items = [
        (key, approval_options[key])
        for key, count in sorted(
            approval_flip_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count > 1 and key in approval_options
    ]
    if not recurrent_approval_items:
        recurrent_approval_items = sorted(approval_options.items())
    if max_initial_approvals is not None:
        recurrent_approval_items = recurrent_approval_items[:max_initial_approvals]
    approval_items = _compress_approval_items(solver, row_items, recurrent_approval_items)
    row_items, approval_items = _trim_initial_active_set(
        row_items=row_items,
        approval_items=approval_items,
        row_counts=row_support_change_counts,
        approval_counts=approval_flip_counts,
        max_candidates=max_candidates,
    )

    max_expansion_rounds = 3
    for expansion_round in range(max_expansion_rounds + 1):
        diagnostics.expansion_rounds = expansion_round
        strategy_df, result, detail = _enumerate_active_candidates(
            solver=solver,
            row_items=row_items,
            approval_items=approval_items,
            base_p=base_p,
            base_r=base_r,
            proposal_keys=proposal_keys,
            acceptance_keys=acceptance_keys,
            refinement_iter=refinement_iter,
            max_candidates=max_candidates,
            max_candidates_per_round=max_candidates_per_round,
            diagnostics=diagnostics,
            freeze_seeded_proposals=freeze_seeded_proposals,
        )
        if result["converged"] or result["stopping_reason"] in {"active_set_too_many_candidates", "active_set_round_budget_exhausted"}:
            return strategy_df, result

        expanded = _expand_active_sets_from_violation(
            solver=solver,
            strategy_df=strategy_df,
            row_items=row_items,
            approval_items=approval_items,
            violation=detail or {},
        )
        if diagnostics.round_history:
            diagnostics.round_history[-1]["post_round_action"] = "expanded" if expanded else "stalled"
            diagnostics.round_history[-1]["post_round_violation"] = detail
        if solver.verbose:
            action = "expanded" if expanded else "stalled"
            solver._log(
                f"Active-set {action} after round {expansion_round}: "
                f"{_format_violation_short(detail)}"
            )
        if not expanded:
            return strategy_df, result

    result["stopping_reason"] = "active_set_expansion_failed"
    result["active_set"] = diagnostics.__dict__
    return strategy_df, result
