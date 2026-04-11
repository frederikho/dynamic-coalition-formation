"""Induced proposition/approval strategies from ordinal rankings."""

from __future__ import annotations

import numpy as np

from lib.equilibrium.solver import EquilibriumSolver


def _build_induced_arrays(
    players: list[str],
    ranks: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_players = len(players)
    n_states = len(ranks[0])
    proposal_choice = np.zeros((n_players, n_states), dtype=np.int8)
    approval_action = np.zeros((n_players, n_players, n_states, n_states), dtype=np.float64)
    approval_pass = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    for proposer_idx in range(n_players):
        rank_prop = ranks[proposer_idx]
        for current_idx in range(n_states):
            committees_row = committee_idxs[proposer_idx][current_idx]
            best_target = current_idx
            best_rank = int(rank_prop[current_idx])
            for next_idx in range(n_states):
                committee = committees_row[next_idx]
                if next_idx == current_idx:
                    approved = True
                    for approver_idx in committee:
                        approval_action[proposer_idx, approver_idx, current_idx, next_idx] = 1.0
                else:
                    approved = True
                    for approver_idx in committee:
                        individual = 1.0 if int(ranks[approver_idx][next_idx]) < int(ranks[approver_idx][current_idx]) else 0.0
                        approval_action[proposer_idx, approver_idx, current_idx, next_idx] = individual
                        if individual == 0.0:
                            approved = False
                approval_pass[proposer_idx, current_idx, next_idx] = 1.0 if approved else 0.0
                if approved and int(rank_prop[next_idx]) < best_rank:
                    best_target = next_idx
                    best_rank = int(rank_prop[next_idx])
            proposal_choice[proposer_idx, current_idx] = best_target
    return proposal_choice, approval_action, approval_pass


def _build_induced_arrays_weak(
    players: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_players = len(players)
    n_states = len(tiers[0])
    proposal_probs = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    approval_action = np.zeros((n_players, n_players, n_states, n_states), dtype=np.float64)
    approval_pass = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    for proposer_idx in range(n_players):
        tier_prop = tiers[proposer_idx]
        for current_idx in range(n_states):
            committees_row = committee_idxs[proposer_idx][current_idx]
            approved_targets: list[int] = []
            best_tier = int(tier_prop[current_idx])
            for next_idx in range(n_states):
                committee = committees_row[next_idx]
                approved = True
                for approver_idx in committee:
                    individual = 1.0 if int(tiers[approver_idx][next_idx]) <= int(tiers[approver_idx][current_idx]) else 0.0
                    approval_action[proposer_idx, approver_idx, current_idx, next_idx] = individual
                    if individual == 0.0:
                        approved = False
                approval_pass[proposer_idx, current_idx, next_idx] = 1.0 if approved else 0.0
                if approved:
                    approved_targets.append(next_idx)
                    best_tier = min(best_tier, int(tier_prop[next_idx]))
            winners = [next_idx for next_idx in approved_targets if int(tier_prop[next_idx]) == best_tier]
            mass = 1.0 / len(winners)
            for next_idx in winners:
                proposal_probs[proposer_idx, current_idx, next_idx] = mass
    return proposal_probs, approval_action, approval_pass


def _build_induced_arrays_weak_from_ids(
    order_ids: tuple[int, ...],
    order_arrays: np.ndarray,
    approve_lookup: np.ndarray,
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_players = len(order_ids)
    n_states = int(order_arrays.shape[1])
    proposal_probs = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    approval_action = np.zeros((n_players, n_players, n_states, n_states), dtype=np.float64)
    approval_pass = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    for proposer_idx in range(n_players):
        tier_prop = order_arrays[order_ids[proposer_idx]]
        for current_idx in range(n_states):
            committees_row = committee_idxs[proposer_idx][current_idx]
            approved_targets: list[int] = []
            best_tier = int(tier_prop[current_idx])
            for next_idx in range(n_states):
                committee = committees_row[next_idx]
                approved = True
                for approver_idx in committee:
                    individual = 1.0 if approve_lookup[order_ids[approver_idx], current_idx, next_idx] else 0.0
                    approval_action[proposer_idx, approver_idx, current_idx, next_idx] = individual
                    if individual == 0.0:
                        approved = False
                approval_pass[proposer_idx, current_idx, next_idx] = 1.0 if approved else 0.0
                if approved:
                    approved_targets.append(next_idx)
                    tier_value = int(tier_prop[next_idx])
                    if tier_value < best_tier:
                        best_tier = tier_value
            winners = [next_idx for next_idx in approved_targets if int(tier_prop[next_idx]) == best_tier]
            mass = 1.0 / len(winners)
            for next_idx in winners:
                proposal_probs[proposer_idx, current_idx, next_idx] = mass
    return proposal_probs, approval_action, approval_pass


def _build_transition_matrix(
    proposal_choice: np.ndarray | None,
    protocol_arr: np.ndarray,
    n_states: int,
    proposal_probs: np.ndarray | None = None,
) -> np.ndarray:
    P = np.zeros((n_states, n_states), dtype=np.float64)
    if proposal_probs is not None:
        for proposer_idx in range(proposal_probs.shape[0]):
            P += protocol_arr[proposer_idx] * proposal_probs[proposer_idx]
        return P
    assert proposal_choice is not None
    for proposer_idx in range(proposal_choice.shape[0]):
        for current_idx in range(n_states):
            P[current_idx, int(proposal_choice[proposer_idx, current_idx])] += protocol_arr[proposer_idx]
    return P


def _induce_profile_from_rankings(
    solver: EquilibriumSolver,
    players: list[str],
    states: list[str],
    ranks: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> None:
    for proposer_idx, proposer in enumerate(players):
        for current_idx, current_state in enumerate(states):
            rank_prop = ranks[proposer_idx]
            best_target = current_idx
            best_rank = int(rank_prop[current_idx])
            committees_row = committee_idxs[proposer_idx][current_idx]
            for next_idx, next_state in enumerate(states):
                committee = committees_row[next_idx]
                if next_idx == current_idx:
                    approved = True
                else:
                    approved = True
                    for approver_idx in committee:
                        if int(ranks[approver_idx][next_idx]) >= int(ranks[approver_idx][current_idx]):
                            approved = False
                            break
                if approved and int(rank_prop[next_idx]) < best_rank:
                    best_target = next_idx
                    best_rank = int(rank_prop[next_idx])
            for next_idx, next_state in enumerate(states):
                solver.p_proposals[(proposer, current_state, next_state)] = 1.0 if next_idx == best_target else 0.0
                committee = committees_row[next_idx]
                for approver_idx in committee:
                    approver = players[approver_idx]
                    approve = 1.0 if next_idx == current_idx or int(ranks[approver_idx][next_idx]) < int(ranks[approver_idx][current_idx]) else 0.0
                    solver.r_acceptances[(proposer, current_state, next_state, approver)] = approve


def _induce_profile_from_weak_orders(
    solver: EquilibriumSolver,
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> None:
    for proposer_idx, proposer in enumerate(players):
        for current_idx, current_state in enumerate(states):
            tier_prop = tiers[proposer_idx]
            committees_row = committee_idxs[proposer_idx][current_idx]
            approved_targets: list[int] = []
            best_tier = int(tier_prop[current_idx])
            for next_idx, _next_state in enumerate(states):
                committee = committees_row[next_idx]
                approved = True
                if next_idx != current_idx:
                    for approver_idx in committee:
                        if int(tiers[approver_idx][next_idx]) > int(tiers[approver_idx][current_idx]):
                            approved = False
                            break
                if approved:
                    approved_targets.append(next_idx)
                    best_tier = min(best_tier, int(tier_prop[next_idx]))
            winners = [next_idx for next_idx in approved_targets if int(tier_prop[next_idx]) == best_tier]
            mass = 1.0 / len(winners)
            for next_idx, next_state in enumerate(states):
                solver.p_proposals[(proposer, current_state, next_state)] = mass if next_idx in winners else 0.0
                committee = committees_row[next_idx]
                for approver_idx in committee:
                    approver = players[approver_idx]
                    approve = 1.0 if int(tiers[approver_idx][next_idx]) <= int(tiers[approver_idx][current_idx]) else 0.0
                    solver.r_acceptances[(proposer, current_state, next_state, approver)] = approve
