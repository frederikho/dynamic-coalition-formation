"""Bellman linear system and fast equilibrium checks for induced strategies."""

from __future__ import annotations

import numpy as np


def _solve_values(P: np.ndarray, payoff_array: np.ndarray, discounting: float) -> np.ndarray:
    A = np.eye(P.shape[0], dtype=np.float64) - discounting * P
    B = (1.0 - discounting) * payoff_array
    return np.linalg.solve(A, B)


def _verify_fast(
    players: list[str],
    states: list[str],
    V_array: np.ndarray,
    proposal_choice: np.ndarray | None,
    approval_action: np.ndarray | None,
    approval_pass: np.ndarray | None,
    committee_idxs: list[list[list[tuple[int, ...]]]],
    proposal_probs: np.ndarray | None = None,
) -> tuple[bool, str]:
    player_idx = {player: idx for idx, player in enumerate(players)}
    state_idx = {state: idx for idx, state in enumerate(states)}

    for proposer in players:
        proposer_col = player_idx[proposer]
        for current_state in states:
            current_i = state_idx[current_state]
            best_value = -np.inf
            argmaxes: list[str] = []
            chosen_states: list[str] = []
            v_current = float(V_array[current_i, proposer_col])
            for next_state in states:
                next_i = state_idx[next_state]
                if proposal_probs is not None:
                    proposed = float(proposal_probs[proposer_col, current_i, next_i])
                else:
                    assert proposal_choice is not None
                    proposed = 1.0 if int(proposal_choice[proposer_col, current_i]) == next_i else 0.0
                if proposed > 0.0:
                    chosen_states.append(next_state)
                assert approval_pass is not None
                p_approved = float(approval_pass[proposer_col, current_i, next_i])
                expected = p_approved * float(V_array[next_i, proposer_col]) + (1.0 - p_approved) * v_current
                if expected > best_value + 1e-9:
                    best_value = expected
                    argmaxes = [next_state]
                elif abs(expected - best_value) <= 1e-9:
                    argmaxes.append(next_state)
            if not set(chosen_states).issubset(argmaxes):
                return False, (
                    f"Proposal strategy error with player {proposer}! In state {current_state}, "
                    f"positive probability on state(s) {chosen_states}, but the argmax states are: {argmaxes}."
                )

    for proposer in players:
        proposer_idx = player_idx[proposer]
        for current_state in states:
            current_i = state_idx[current_state]
            for next_state in states:
                next_i = state_idx[next_state]
                committee = [players[i] for i in committee_idxs[proposer_idx][current_i][next_i]]
                for approver in committee:
                    approver_col = player_idx[approver]
                    v_current = float(V_array[current_i, approver_col])
                    v_next = float(V_array[next_i, approver_col])
                    assert approval_action is not None
                    p_approve = float(approval_action[proposer_idx, approver_col, current_i, next_i])
                    if abs(v_next - v_current) <= 1e-9:
                        passed = 0.0 <= p_approve <= 1.0
                    elif v_next > v_current:
                        passed = abs(p_approve - 1.0) <= 1e-12
                    else:
                        passed = abs(p_approve - 0.0) <= 1e-12
                    if not passed:
                        return False, (
                            f"Approval strategy error with player {approver}! "
                            f"When player {proposer} proposes the transition {current_state} -> {next_state}, "
                            f"the values are V(current) = {v_current:.6f} and V(next) = {v_next:.6f}, "
                            f"but approval probability is {p_approve}."
                        )
    return True, "All tests passed."
