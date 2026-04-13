"""Induced proposition/approval strategies from ordinal rankings."""

from __future__ import annotations

import numpy as np
from lib.equilibrium.solver import EquilibriumSolver

def _build_induced_arrays(
    players: list[str],
    ranks: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pure strategy profile and transition matrix from strict rankings."""
    n_players = len(players)
    n_states = len(ranks[0])
    proposal_choice = np.zeros((n_players, n_states), dtype=np.int8)
    approval_action = np.zeros((n_players, n_players, n_states, n_states), dtype=np.float64)
    approval_pass = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    P = np.zeros((n_states, n_states), dtype=np.float64)

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
            P[current_idx, best_target] += protocol_arr[proposer_idx]

    return proposal_choice, approval_action, approval_pass, P

def _build_induced_arrays_weak(
    players: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build canonical (uniform) strategy profile and transition matrix from weak rankings."""
    n_players = len(players)
    n_states = len(tiers[0])
    proposal_probs = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    approval_action = np.zeros((n_players, n_players, n_states, n_states), dtype=np.float64)
    approval_pass = np.zeros((n_players, n_states, n_states), dtype=np.float64)
    P = np.zeros((n_states, n_states), dtype=np.float64)

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
                P[current_idx, next_idx] += protocol_arr[proposer_idx] * mass

    return proposal_probs, approval_action, approval_pass, P

def _induce_profile_from_rankings(
    solver: EquilibriumSolver,
    players: list[str],
    states: list[str],
    ranks: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> None:
    """Populate solver strategy objects from strict rankings."""
    n_players = len(players)
    n_states = len(states)
    for proposer_idx, proposer in enumerate(players):
        rank_prop = ranks[proposer_idx]
        for current_idx, current_state in enumerate(states):
            committees_row = committee_idxs[proposer_idx][current_idx]
            best_target = current_state
            best_rank = int(rank_prop[current_idx])
            for next_idx, next_state in enumerate(states):
                committee = committees_row[next_idx]
                approved = True
                for approver_idx in committee:
                    approver = players[approver_idx]
                    approve = 1.0 if int(ranks[approver_idx][next_idx]) < int(ranks[approver_idx][current_idx]) else 0.0
                    solver.r_acceptances[(proposer, current_state, next_state, approver)] = approve
                    if approve == 0.0:
                        approved = False
                if approved and int(rank_prop[next_idx]) < best_rank:
                    best_target = next_state
                    best_rank = int(rank_prop[next_idx])
            for next_state in states:
                solver.p_proposals[(proposer, current_state, next_state)] = 1.0 if next_state == best_target else 0.0

def _induce_profile_from_weak_orders(
    solver: EquilibriumSolver,
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> None:
    """Populate solver strategy objects from weak rankings (canonical ties)."""
    n_players = len(players)
    n_states = len(states)
    for proposer_idx, proposer in enumerate(players):
        tier_prop = tiers[proposer_idx]
        for current_idx, current_state in enumerate(states):
            committees_row = committee_idxs[proposer_idx][current_idx]
            approved_targets: list[str] = []
            best_tier = int(tier_prop[current_idx])
            for next_idx, next_state in enumerate(states):
                committee = committees_row[next_idx]
                approved = True
                for approver_idx in committee:
                    approver = players[approver_idx]
                    approve = 1.0 if int(tiers[approver_idx][next_idx]) <= int(tiers[approver_idx][current_idx]) else 0.0
                    solver.r_acceptances[(proposer, current_state, next_state, approver)] = approve
                    if approve == 0.0:
                        approved = False
                if approved:
                    approved_targets.append(next_state)
                    best_tier = min(best_tier, int(tier_prop[next_idx]))
            winners = [s for s in approved_targets if int(tiers[proposer_idx][states.index(s)]) == best_tier]
            mass = 1.0 / len(winners)
            for next_state in states:
                solver.p_proposals[(proposer, current_state, next_state)] = mass if next_state in winners else 0.0
