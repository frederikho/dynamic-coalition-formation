#!/usr/bin/env python3
"""Utilities for exact reduced-game weak-ranking solving.

This module does not yet solve the full exact problem. It provides the
compressed row-level objects the generic solver will need:

- player-level equality groups from a weak ranking
- committee structure for the reduced 4-state game
- row-level passability classification:
  forced pass / forced fail / potentially free

The goal is to work with row-level passability regimes rather than raw
approval cells, which overcounts the true complexity badly.

Current modeling decision for the next solver layer:

- We use a shared-parameter ansatz for indifferent choices.
- If a player's weak ranking has an equality group of size k, we assign
  k-1 latent parameters to that player/group and reuse those same parameters
  everywhere that equality appears in committee/approval contexts.
- Example: if CHN is indifferent across all 4 states, we do not allow a
  separate approval/pass probability for every proposer/current/next context.
  Instead, CHN gets 3 shared latent parameters for that 4-state equality
  group, and all CHN-indifferent approvals are mapped from those same
  parameters.

This keeps the dimension bounded by the weak ranking itself. It is a
restricted ansatz relative to the fully unrestricted strategy space, but it
is the intended approximation for the full weak-ranking sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.find import setup_experiment
from lib.utils import get_approval_committee
from lib.equilibrium.reduced_weak_helpers import (
    _build_payoff_config,
    _generate_weak_orders,
    _resolve_payoff_file,
    _solve_values_fast_array,
    _verify_equilibrium_fast,
    _weak_equality_groups,
)


@dataclass(frozen=True)
class RowEdgeStatus:
    proposer: str
    current_state: str
    next_state: str
    committee: tuple[str, ...]
    forced_pass: bool
    forced_fail: bool
    free_approvers: tuple[str, ...]


@dataclass(frozen=True)
class SharedApprovalContext:
    player: str
    proposer: str
    current_state: str
    next_state: str
    group_states: tuple[str, ...]


@dataclass(frozen=True)
class SharedProposalContext:
    player: str
    current_state: str
    winner_states: tuple[str, ...]
    group_states: tuple[str, ...]


def committee_idxs(players: list[str], states: list[str], effectivity: dict[tuple, int]) -> list[list[list[tuple[int, ...]]]]:
    player_idx = {player: idx for idx, player in enumerate(players)}
    out: list[list[list[tuple[int, ...]]]] = []
    for proposer in players:
        proposer_rows: list[list[tuple[int, ...]]] = []
        for current_state in states:
            row: list[tuple[int, ...]] = []
            for next_state in states:
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                row.append(tuple(player_idx[p] for p in committee))
            proposer_rows.append(row)
        out.append(proposer_rows)
    return out


def load_reduced_setup(payoff_file: str, scenario: str) -> dict[str, Any]:
    payoff_path = _resolve_payoff_file(payoff_file)
    config = _build_payoff_config(scenario, str(payoff_path))
    setup = setup_experiment(config)
    if len(setup["players"]) != 3 or len(setup["state_names"]) != 4:
        raise ValueError("This helper expects the reduced 4-state, 3-player case.")
    return setup


def weak_tiers_from_perm_ids(n_states: int, perm_a: int, perm_b: int, perm_c: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orders = _generate_weak_orders(n_states)
    return (orders[perm_a], orders[perm_b], orders[perm_c])


def classify_edge_statuses(
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    effectivity: dict[tuple, int],
) -> list[RowEdgeStatus]:
    statuses: list[RowEdgeStatus] = []
    player_idx = {player: idx for idx, player in enumerate(players)}
    for proposer in players:
        for current_state in states:
            current_idx = states.index(current_state)
            for next_state in states:
                next_idx = states.index(next_state)
                committee = tuple(get_approval_committee(effectivity, players, proposer, current_state, next_state))
                free_approvers: list[str] = []
                forced_fail = False
                for approver in committee:
                    i = player_idx[approver]
                    next_tier = int(tiers[i][next_idx])
                    current_tier = int(tiers[i][current_idx])
                    if next_tier > current_tier:
                        forced_fail = True
                        break
                    if next_idx != current_idx and next_tier == current_tier:
                        free_approvers.append(approver)
                forced_pass = (not forced_fail) and len(free_approvers) == 0
                statuses.append(
                    RowEdgeStatus(
                        proposer=proposer,
                        current_state=current_state,
                        next_state=next_state,
                        committee=committee,
                        forced_pass=forced_pass,
                        forced_fail=forced_fail,
                        free_approvers=tuple(free_approvers),
                    )
                )
    return statuses


def summarize_row_passability(
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    effectivity: dict[tuple, int],
) -> dict[tuple[str, str], dict[str, list[str]]]:
    statuses = classify_edge_statuses(players, states, tiers, effectivity)
    summary: dict[tuple[str, str], dict[str, list[str]]] = {}
    for status in statuses:
        row_key = (status.proposer, status.current_state)
        bucket = summary.setdefault(row_key, {"forced_pass": [], "forced_fail": [], "free": []})
        if status.forced_pass:
            bucket["forced_pass"].append(status.next_state)
        elif status.forced_fail:
            bucket["forced_fail"].append(status.next_state)
        else:
            bucket["free"].append(status.next_state)
    return summary


def ranking_summary(
    payoff_file: str,
    scenario: str,
    perm_a: int,
    perm_b: int,
    perm_c: int,
) -> dict[str, Any]:
    setup = load_reduced_setup(payoff_file, scenario)
    players = setup["players"]
    states = setup["state_names"]
    tiers = weak_tiers_from_perm_ids(len(states), perm_a, perm_b, perm_c)
    return {
        "players": players,
        "states": states,
        "tiers": tiers,
        "equality_groups": _weak_equality_groups(states, tiers),
        "row_passability": summarize_row_passability(players, states, tiers, setup["effectivity"]),
    }


def shared_variable_structure(
    *,
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    effectivity: dict[tuple, int],
) -> dict[str, Any]:
    """Describe the low-dimensional shared-variable space implied by the weak ranking.

    This captures only the user's intended restriction:
    each player/equality-group gets a small shared latent parameterization that
    is reused across all approval/proposal contexts where that indifference
    appears.
    """
    player_idx = {player: idx for idx, player in enumerate(players)}
    state_idx = {state: idx for idx, state in enumerate(states)}
    equality_groups = _weak_equality_groups(states, tiers)
    row_summary = summarize_row_passability(players, states, tiers, effectivity)

    group_param_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    approval_contexts: list[SharedApprovalContext] = []
    proposal_contexts: list[SharedProposalContext] = []

    for player_i, player in enumerate(players):
        group_by_state: dict[str, tuple[str, ...]] = {}
        for group in equality_groups[player_i]:
            for state in group:
                group_by_state[state] = group
            group_param_counts[(player, group)] = max(0, len(group) - 1)

        for proposer in players:
            for current_state in states:
                current_i = state_idx[current_state]
                bucket = row_summary[(proposer, current_state)]
                # Approval contexts where this player is indifferent and on committee.
                for next_state in bucket["free"]:
                    committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                    if player not in committee:
                        continue
                    next_i = state_idx[next_state]
                    if int(tiers[player_i][next_i]) != int(tiers[player_i][current_i]):
                        continue
                    group = group_by_state.get(current_state)
                    if group is None or next_state not in group:
                        continue
                    approval_contexts.append(
                        SharedApprovalContext(
                            player=player,
                            proposer=proposer,
                            current_state=current_state,
                            next_state=next_state,
                            group_states=group,
                        )
                    )

        for current_state in states:
            current_i = state_idx[current_state]
            # Proposal contexts where several tied best states exist for this player.
            bucket = row_summary[(player, current_state)]
            passable = list(bucket["forced_pass"]) + list(bucket["free"])
            if not passable:
                continue
            best_tier = min(int(tiers[player_i][state_idx[next_state]]) for next_state in passable)
            winners = tuple(next_state for next_state in passable if int(tiers[player_i][state_idx[next_state]]) == best_tier)
            if len(winners) <= 1:
                continue
            current_group = group_by_state.get(winners[0])
            if current_group is None or any(next_state not in current_group for next_state in winners):
                continue
            proposal_contexts.append(
                SharedProposalContext(
                    player=player,
                    current_state=current_state,
                    winner_states=winners,
                    group_states=current_group,
                )
            )

    return {
        "equality_groups": equality_groups,
        "group_param_counts": group_param_counts,
        "total_group_params": int(sum(group_param_counts.values())),
        "approval_contexts": tuple(approval_contexts),
        "proposal_contexts": tuple(proposal_contexts),
    }


def _player_refinement_tiers(
    states: list[str],
    player_tiers: np.ndarray,
) -> list[np.ndarray]:
    """Enumerate weak refinements within the player's equality groups.

    This is a finite exact search over latent shared-score weak orders:
    each original equality group may be further weakly ordered internally,
    while preserving the original order between groups.
    """
    tier_to_states: dict[int, list[str]] = {}
    for state, tier in zip(states, player_tiers):
        tier_to_states.setdefault(int(tier), []).append(state)
    ordered_groups = [tuple(tier_to_states[t]) for t in sorted(tier_to_states)]

    group_options: list[list[np.ndarray]] = []
    for group in ordered_groups:
        if len(group) == 1:
            group_options.append([np.zeros(1, dtype=np.int16)])
            continue
        group_orders = _generate_weak_orders(len(group))
        group_options.append([np.asarray(order, dtype=np.int16) for order in group_orders])

    state_idx = {state: idx for idx, state in enumerate(states)}
    refinements: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for choice in product(*group_options):
        refined = np.zeros(len(states), dtype=np.int16)
        offset = 0
        for group, local_tiers in zip(ordered_groups, choice):
            for local_idx, state in enumerate(group):
                refined[state_idx[state]] = np.int16(offset + int(local_tiers[local_idx]))
            offset += int(np.max(local_tiers)) + 1
        key = tuple(int(x) for x in refined)
        if key not in seen:
            seen.add(key)
            refinements.append(refined)
    return refinements


def weak_player_shape(groups: tuple[tuple[str, ...], ...]) -> str:
    sizes = sorted((len(group) for group in groups), reverse=True)
    if not sizes:
        return "strict"
    return "+".join(str(size) for size in sizes)


def deterministic_candidate_count(
    *,
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    effectivity: dict[tuple, int],
) -> int:
    player_idx = {player: idx for idx, player in enumerate(players)}
    state_idx = {state: idx for idx, state in enumerate(states)}
    row_summary = summarize_row_passability(players, states, tiers, effectivity)

    free_edges = 0
    base_count = 1
    for proposer in players:
        proposer_col = player_idx[proposer]
        for current_state in states:
            bucket = row_summary[(proposer, current_state)]
            free_edges += len(bucket["free"])
            passable = list(bucket["forced_pass"])
            best_tier = min(int(tiers[proposer_col][state_idx[next_state]]) for next_state in passable)
            winners = [next_state for next_state in passable if int(tiers[proposer_col][state_idx[next_state]]) == best_tier]
            base_count *= max(1, len(winners))
    return (2 ** free_edges) * base_count


def runtime_feature_summary(
    *,
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    effectivity: dict[tuple, int],
) -> dict[str, Any]:
    equality_groups = _weak_equality_groups(states, tiers)
    row_summary = summarize_row_passability(players, states, tiers, effectivity)
    statuses = classify_edge_statuses(players, states, tiers, effectivity)

    total_value_params = sum(sum(len(group) - 1 for group in groups) for groups in equality_groups)
    total_free_edges = sum(1 for status in statuses if (not status.forced_pass and not status.forced_fail))
    total_free_approver_slots = sum(len(status.free_approvers) for status in statuses)
    free_rows = sum(1 for bucket in row_summary.values() if bucket["free"])
    max_free_edges_in_row = max((len(bucket["free"]) for bucket in row_summary.values()), default=0)
    deterministic_count = deterministic_candidate_count(
        players=players,
        states=states,
        tiers=tiers,
        effectivity=effectivity,
    )

    shape_counts = {
        "1+1+1+1": 0,
        "2+1+1": 0,
        "2+2": 0,
        "3+1": 0,
        "4": 0,
    }
    player_value_params: list[int] = []
    player_shapes: list[str] = []
    for groups in equality_groups:
        shape = weak_player_shape(groups)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        player_shapes.append(shape)
        player_value_params.append(sum(len(group) - 1 for group in groups))

    return {
        "player_shapes": tuple(player_shapes),
        "total_value_params": int(total_value_params),
        "player_value_params": tuple(int(v) for v in player_value_params),
        "shape_all_strict_players": int(shape_counts.get("1+1+1+1", 0)),
        "shape_pair_players": int(shape_counts.get("2+1+1", 0)),
        "shape_two_pair_players": int(shape_counts.get("2+2", 0)),
        "shape_triple_players": int(shape_counts.get("3+1", 0)),
        "shape_all_equal_players": int(shape_counts.get("4", 0)),
        "free_rows": int(free_rows),
        "total_free_edges": int(total_free_edges),
        "total_free_approver_slots": int(total_free_approver_slots),
        "max_free_edges_in_row": int(max_free_edges_in_row),
        "deterministic_candidate_count": int(deterministic_count),
        "log2_deterministic_candidate_count": float(np.log2(max(1, deterministic_count))),
    }


def _sigmoid_scalar(value: float) -> float:
    z = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _build_group_score_index(
    states: list[str],
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[list[dict[str, float]], int]:
    equality_groups = _weak_equality_groups(states, tiers)
    score_maps: list[dict[str, float]] = []
    offset = 0
    for groups in equality_groups:
        state_to_param: dict[str, float] = {}
        for group in groups:
            for local_idx, state in enumerate(group):
                if local_idx == 0:
                    state_to_param[state] = -(offset + 1)  # negative sentinel for baseline 0.0
                else:
                    state_to_param[state] = float(offset)
                    offset += 1
        score_maps.append(state_to_param)
    return score_maps, offset


def _state_score(
    state: str,
    score_map: dict[str, float],
    params: np.ndarray,
) -> float:
    ref = score_map.get(state)
    if ref is None:
        return 0.0
    if ref < 0:
        return 0.0
    return float(params[int(ref)])


def induce_shared_parameter_profile(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol_arr: np.ndarray,
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    params: np.ndarray,
    forbidden_proposals: frozenset = frozenset(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_players = len(players)
    n_states = len(states)
    state_idx = {state: idx for idx, state in enumerate(states)}
    player_idx = {player: idx for idx, player in enumerate(players)}
    score_maps, _n_params = _build_group_score_index(states, tiers)

    proposal_probs = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    approval_action = np.zeros((n_players, n_players, n_states, n_states), dtype=np.float64)
    approval_pass = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    P_array = np.zeros((n_states, n_states), dtype=np.float64)

    # Build approval probabilities from shared state scores within equality groups.
    for proposer in players:
        proposer_i = player_idx[proposer]
        for current_state in states:
            current_i = state_idx[current_state]
            for next_state in states:
                next_i = state_idx[next_state]
                # Forbidden (e.g. non-adjacent under adjacent_step) proposals are
                # never proposable: zero their pass-probability and skip them.
                if next_i != current_i and (proposer, current_state, next_state) in forbidden_proposals:
                    approval_pass[proposer_i, current_i, next_i] = 0.0
                    continue
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                probs: list[float] = []
                for approver in committee:
                    approver_i = player_idx[approver]
                    next_tier = int(tiers[approver_i][next_i])
                    current_tier = int(tiers[approver_i][current_i])
                    if next_tier < current_tier:
                        prob = 1.0
                    elif next_tier > current_tier:
                        prob = 0.0
                    elif next_i == current_i:
                        prob = 1.0
                    else:
                        score_map = score_maps[approver_i]
                        prob = _sigmoid_scalar(
                            _state_score(next_state, score_map, params) - _state_score(current_state, score_map, params)
                        )
                    approval_action[proposer_i, approver_i, current_i, next_i] = prob
                    probs.append(prob)
                approval_pass[proposer_i, current_i, next_i] = 0.0 if len(probs) == 0 else float(np.prod(probs))

    # Build proposal probabilities from shared state scores on best-tier passable targets.
    for proposer in players:
        proposer_i = player_idx[proposer]
        proposer_scores = score_maps[proposer_i]
        for current_state in states:
            current_i = state_idx[current_state]
            passable = [next_state for next_state in states if approval_pass[proposer_i, current_i, state_idx[next_state]] > 1e-12]
            if not passable:
                proposal_probs[proposer_i, current_i, current_i] = 1.0
                P_array[current_i, current_i] += protocol_arr[proposer_i]
                continue
            best_tier = min(int(tiers[proposer_i][state_idx[next_state]]) for next_state in passable)
            winners = [next_state for next_state in passable if int(tiers[proposer_i][state_idx[next_state]]) == best_tier]
            winner_scores = np.array([_state_score(next_state, proposer_scores, params) for next_state in winners], dtype=np.float64)
            probs = _softmax(winner_scores)
            for next_state, prob in zip(winners, probs):
                next_i = state_idx[next_state]
                proposal_probs[proposer_i, current_i, next_i] = float(prob)
                P_array[current_i, next_i] += protocol_arr[proposer_i] * float(prob)

    return proposal_probs, approval_action, approval_pass, P_array


def solve_shared_parameter_ansatz(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    max_nfev: int = 200,
    forbidden_proposals: frozenset = frozenset(),
) -> dict[str, Any] | None:
    state_idx = {state: idx for idx, state in enumerate(states)}
    score_maps, n_params = _build_group_score_index(states, tiers)

    if n_params == 0:
        proposal_probs, approval_action, approval_pass, P_array = induce_shared_parameter_profile(
            players=players,
            states=states,
            effectivity=effectivity,
            protocol_arr=protocol_arr,
            tiers=tiers,
            params=np.zeros(0, dtype=np.float64),
            forbidden_proposals=forbidden_proposals,
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
            committee_idxs=committee_idxs(players, states, effectivity),
            forbidden_proposals=forbidden_proposals,
        )
        if verified:
            return {"params": np.zeros(0, dtype=np.float64), "P_array": P_array, "V_array": V_array, "message": message, "detail": detail}
        return None

    def residuals(param_vec: np.ndarray) -> np.ndarray:
        proposal_probs, approval_action, approval_pass, P_array = induce_shared_parameter_profile(
            players=players,
            states=states,
            effectivity=effectivity,
            protocol_arr=protocol_arr,
            tiers=tiers,
            params=param_vec,
            forbidden_proposals=forbidden_proposals,
        )
        V_array = _solve_values_fast_array(P_array, payoff_array, discounting)
        res: list[float] = []

        # Enforce equality groups in V.
        for player_i, score_map in enumerate(score_maps):
            groups: dict[float, list[str]] = {}
            for state, ref in score_map.items():
                key = ref if ref < 0 else float(int(ref))
                groups.setdefault(key, [])
            # Recover actual groups from tiers
        equality_groups = _weak_equality_groups(states, tiers)
        for player_i, groups in enumerate(equality_groups):
            for group in groups:
                base = group[0]
                base_v = float(V_array[state_idx[base], player_i])
                for state in group[1:]:
                    res.append(float(V_array[state_idx[state], player_i]) - base_v)

        # Penalize violated strict tier inequalities.
        margin = 1e-6
        for player_i in range(len(players)):
            for i, current_state in enumerate(states):
                for j, next_state in enumerate(states):
                    if i == j:
                        continue
                    if int(tiers[player_i][i]) < int(tiers[player_i][j]):
                        diff = float(V_array[i, player_i]) - float(V_array[j, player_i])
                        res.append(min(0.0, diff - margin))

        # Proposal optimality residual: mass should be on argmax expected values.
        for proposer_i, proposer in enumerate(players):
            for current_state in states:
                current_i = state_idx[current_state]
                current_v = float(V_array[current_i, proposer_i])
                expected = []
                for next_state in states:
                    next_i = state_idx[next_state]
                    p_app = float(approval_pass[proposer_i, current_i, next_i])
                    expected.append(p_app * float(V_array[next_i, proposer_i]) + (1.0 - p_app) * current_v)
                best = max(expected)
                for next_i, next_state in enumerate(states):
                    prob = float(proposal_probs[proposer_i, current_i, next_i])
                    res.append(prob * (expected[next_i] - best))

        return np.array(res, dtype=np.float64)

    guesses = [np.zeros(n_params, dtype=np.float64), np.full(n_params, 1.0), np.full(n_params, -1.0)]
    for guess in guesses:
        opt = least_squares(residuals, guess, max_nfev=max_nfev)
        if np.max(np.abs(opt.fun)) > 1e-6:
            continue
        proposal_probs, approval_action, approval_pass, P_array = induce_shared_parameter_profile(
            players=players,
            states=states,
            effectivity=effectivity,
            protocol_arr=protocol_arr,
            tiers=tiers,
            params=np.asarray(opt.x, dtype=np.float64),
            forbidden_proposals=forbidden_proposals,
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
            committee_idxs=committee_idxs(players, states, effectivity),
            forbidden_proposals=forbidden_proposals,
        )
        if verified:
            return {
                "params": np.asarray(opt.x, dtype=np.float64),
                "P_array": P_array,
                "V_array": V_array,
                "message": message,
                "detail": detail,
                "cost": float(opt.cost),
                "nfev": int(opt.nfev),
            }
    return None


def _approval_value_from_tiers(
    *,
    approver_idx: int,
    current_idx: int,
    next_idx: int,
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    chosen_pass: bool | None,
) -> float:
    next_tier = int(tiers[approver_idx][next_idx])
    current_tier = int(tiers[approver_idx][current_idx])
    if next_tier < current_tier:
        return 1.0
    if next_tier > current_tier:
        return 0.0
    if next_idx == current_idx:
        return 1.0
    return 1.0 if chosen_pass else 0.0


def exact_check_shared_refinements(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    max_refinements: int = 200000,
    max_deterministic_candidates: int = 200000,
) -> dict[str, Any] | None:
    """Finite exact search over per-player weak refinements, then deterministic resolution.

    This replaces the previous numerical shared-parameter fallback in the probe path.
    It is exhaustive over:
    - weak refinements inside each player's original equality groups
    - deterministic row/edge resolutions for each refined ranking

    It is still a restriction relative to fully interior mixed behavior.
    """
    player_refinements = [_player_refinement_tiers(states, np.asarray(player_tiers, dtype=np.int16)) for player_tiers in tiers]
    total_refinements = 1
    for options in player_refinements:
        total_refinements *= max(1, len(options))
    if total_refinements > max_refinements:
        return {
            "message": "shared refinement cap exceeded",
            "total_refinements": total_refinements,
            "refinement_counts": tuple(len(options) for options in player_refinements),
        }

    tested_refinements = 0
    for refined_a, refined_b, refined_c in product(*player_refinements):
        tested_refinements += 1
        det = exact_check_deterministic_resolutions(
            players=players,
            states=states,
            effectivity=effectivity,
            protocol_arr=protocol_arr,
            payoff_array=payoff_array,
            discounting=discounting,
            tiers=(refined_a, refined_b, refined_c),
            max_candidates=max_deterministic_candidates,
        )
        if det is not None and "P_array" in det:
            return {
                "message": "verified via shared refinements",
                "refinement_counts": tuple(len(options) for options in player_refinements),
                "total_refinements": total_refinements,
                "tested_refinements": tested_refinements,
                "refined_tiers": (refined_a.copy(), refined_b.copy(), refined_c.copy()),
                "deterministic_detail": det,
                "P_array": det["P_array"],
                "V_array": det["V_array"],
            }
    return {
        "message": "no shared refinement verified",
        "refinement_counts": tuple(len(options) for options in player_refinements),
        "total_refinements": total_refinements,
        "tested_refinements": tested_refinements,
    }


def exact_check_deterministic_resolutions(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    tiers: tuple[np.ndarray, np.ndarray, np.ndarray],
    max_candidates: int = 200000,
) -> dict[str, Any] | None:
    state_idx = {state: idx for idx, state in enumerate(states)}
    player_idx = {player: idx for idx, player in enumerate(players)}
    row_summary = summarize_row_passability(players, states, tiers, effectivity)

    free_edges: list[tuple[str, str, str]] = []
    for (proposer, current_state), bucket in sorted(row_summary.items()):
        for next_state in sorted(bucket["free"]):
            free_edges.append((proposer, current_state, next_state))

    row_winner_options: list[tuple[tuple[str, str], list[str]]] = []
    for proposer in players:
        proposer_col = player_idx[proposer]
        for current_state in states:
            bucket = row_summary[(proposer, current_state)]
            passable = list(bucket["forced_pass"])
            current_i = state_idx[current_state]
            best_tier = min(int(tiers[proposer_col][state_idx[next_state]]) for next_state in passable)
            winners = [next_state for next_state in passable if int(tiers[proposer_col][state_idx[next_state]]) == best_tier]
            row_winner_options.append(((proposer, current_state), winners))

    base_count = 1
    for _row_key, winners in row_winner_options:
        base_count *= max(1, len(winners))
    candidate_count = (2 ** len(free_edges)) * base_count
    if candidate_count > max_candidates:
        return None

    free_edge_order = {edge: idx for idx, edge in enumerate(free_edges)}

    for free_bits in product((0, 1), repeat=len(free_edges)):
        chosen_free_pass = {edge: bool(free_bits[idx]) for edge, idx in free_edge_order.items()}

        row_choices: list[list[str]] = []
        for (proposer, current_state), _winners in row_winner_options:
            proposer_col = player_idx[proposer]
            bucket = row_summary[(proposer, current_state)]
            passable = list(bucket["forced_pass"]) + [
                next_state
                for next_state in bucket["free"]
                if chosen_free_pass[(proposer, current_state, next_state)]
            ]
            best_tier = min(int(tiers[proposer_col][state_idx[next_state]]) for next_state in passable)
            winners = [next_state for next_state in passable if int(tiers[proposer_col][state_idx[next_state]]) == best_tier]
            row_choices.append(winners)

        for winner_choice in product(*row_choices):
            proposal_probs = np.zeros((len(players), len(states), len(states)), dtype=np.float64)
            approval_action = np.zeros((len(players), len(players), len(states), len(states)), dtype=np.float64)
            approval_pass = np.zeros((len(players), len(states), len(states)), dtype=np.float64)
            P_array = np.zeros((len(states), len(states)), dtype=np.float64)

            for row_idx, ((proposer, current_state), _winners) in enumerate(row_winner_options):
                proposer_i = player_idx[proposer]
                current_i = state_idx[current_state]
                chosen_next = winner_choice[row_idx]
                next_i = state_idx[chosen_next]
                proposal_probs[proposer_i, current_i, next_i] = 1.0
                P_array[current_i, next_i] += protocol_arr[proposer_i]

            for proposer in players:
                proposer_i = player_idx[proposer]
                for current_state in states:
                    current_i = state_idx[current_state]
                    bucket = row_summary[(proposer, current_state)]
                    for next_state in states:
                        next_i = state_idx[next_state]
                        committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                        if next_state in bucket["forced_pass"]:
                            chosen_pass = True
                        elif next_state in bucket["forced_fail"]:
                            chosen_pass = False
                        else:
                            chosen_pass = chosen_free_pass[(proposer, current_state, next_state)]
                        individual_probs = []
                        for approver in committee:
                            approver_i = player_idx[approver]
                            prob = _approval_value_from_tiers(
                                approver_idx=approver_i,
                                current_idx=current_i,
                                next_idx=next_i,
                                tiers=tiers,
                                chosen_pass=chosen_pass,
                            )
                            approval_action[proposer_i, approver_i, current_i, next_i] = prob
                            individual_probs.append(prob)
                        if len(individual_probs) == 0:
                            approval_pass[proposer_i, current_i, next_i] = 0.0
                        else:
                            approval_pass[proposer_i, current_i, next_i] = float(np.prod(individual_probs))

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
                committee_idxs=committee_idxs(players, states, effectivity),
            )
            if verified:
                return {
                    "message": message,
                    "detail": detail,
                    "free_edges": free_edges,
                    "chosen_free_pass": chosen_free_pass,
                    "winner_choice": winner_choice,
                    "P_array": P_array,
                    "V_array": V_array,
                    "candidate_count": candidate_count,
                }

    return {
        "message": "no deterministic resolution verified",
        "detail": None,
        "free_edges": free_edges,
        "candidate_count": candidate_count,
    }


if __name__ == "__main__":
    summary = ranking_summary(
        payoff_file="simple_cycle_usachnnde-65-reduced.xlsx",
        scenario="power_threshold_RICE_n3",
        perm_a=73,
        perm_b=22,
        perm_c=66,
    )
    print(summary)
