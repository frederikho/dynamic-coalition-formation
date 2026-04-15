"""Weak orders, permutations, and payoff-based ranking enumeration order."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def _generate_weak_orders(n_states: int) -> np.ndarray:
    partitions: list[list[list[int]]] = []

    def rec(i: int, blocks: list[list[int]]) -> None:
        if i == n_states:
            partitions.append([block.copy() for block in blocks])
            return
        for block in blocks:
            block.append(i)
            rec(i + 1, blocks)
            block.pop()
        blocks.append([i])
        rec(i + 1, blocks)
        blocks.pop()

    rec(0, [])
    tiers: list[np.ndarray] = []
    for blocks in partitions:
        for ordered_blocks in itertools.permutations(blocks):
            tier = np.empty(n_states, dtype=np.int8)
            for rank, block in enumerate(ordered_blocks):
                for state_idx in block:
                    tier[state_idx] = rank
            tiers.append(tier)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for tier in tiers:
        unique.setdefault(tuple(int(x) for x in tier), tier)
    return np.array(list(unique.values()), dtype=np.int8)


def _payoff_ordering(payoff_array: np.ndarray, state_perms: np.ndarray, players: list[str]) -> tuple[np.ndarray, ...]:
    del players
    orders: list[np.ndarray] = []
    for player_idx in range(payoff_array.shape[1]):
        baseline_perm = np.array([
            idx for idx, _value in sorted(
                enumerate(payoff_array[:, player_idx]),
                key=lambda pair: (-float(pair[1]), pair[0]),
            )
        ], dtype=np.int64)
        baseline_pos = np.empty(len(baseline_perm), dtype=np.int64)
        for rank, state_idx in enumerate(baseline_perm):
            baseline_pos[int(state_idx)] = rank
        scored: list[tuple[int, int, int]] = []
        for perm_idx, perm in enumerate(state_perms):
            perm = np.asarray(perm, dtype=np.int64)
            kendall = 0
            for i in range(len(perm)):
                pi = int(perm[i])
                for j in range(i + 1, len(perm)):
                    pj = int(perm[j])
                    if baseline_pos[pi] > baseline_pos[pj]:
                        kendall += 1
            footrule = int(sum(abs(rank - int(baseline_pos[int(state_idx)])) for rank, state_idx in enumerate(perm)))
            scored.append((kendall, footrule, perm_idx))
        scored.sort()
        orders.append(np.array([perm_idx for _k, _f, perm_idx in scored], dtype=np.int64))
    return tuple(orders)


def _payoff_ordering_weak(payoff_array: np.ndarray, weak_orders: np.ndarray, players: list[str]) -> tuple[np.ndarray, ...]:
    del players
    orders: list[np.ndarray] = []
    for player_idx in range(payoff_array.shape[1]):
        baseline_perm = np.array([
            idx for idx, _value in sorted(
                enumerate(payoff_array[:, player_idx]),
                key=lambda pair: (-float(pair[1]), pair[0]),
            )
        ], dtype=np.int64)
        baseline_pos = np.empty(len(baseline_perm), dtype=np.int64)
        for rank, state_idx in enumerate(baseline_perm):
            baseline_pos[int(state_idx)] = rank
        scored: list[tuple[int, int, int]] = []
        for order_idx, order in enumerate(weak_orders):
            perm = np.array(sorted(range(len(order)), key=lambda idx: (int(order[idx]), int(baseline_pos[idx]))), dtype=np.int64)
            kendall = 0
            for i in range(len(perm)):
                pi = int(perm[i])
                for j in range(i + 1, len(perm)):
                    pj = int(perm[j])
                    if baseline_pos[pi] > baseline_pos[pj]:
                        kendall += 1
            footrule = int(sum(abs(rank - int(baseline_pos[int(state_idx)])) for rank, state_idx in enumerate(perm)))
            scored.append((kendall, footrule, order_idx))
        scored.sort()
        orders.append(np.array([order_idx for _k, _f, order_idx in scored], dtype=np.int64))
    return tuple(orders)


def _compute_absorbing_pruning_masks(
    weak_orders: np.ndarray,
    payoff_array: np.ndarray,
    state_names: list[str],
    players: list[str],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    absorbing_state_name: str,
    forbidden_proposals: frozenset | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Filter weak orders per player using LCCS absorbing-state pruning.

    Two safe necessary conditions are applied (both provably safe: no valid
    equilibrium is discarded):

    Rule 1 – Unilateral Exit
        If player i can unilaterally exit absorbing state A to state B
        (approval committee ⊆ {i}) AND the proposal is not forbidden, and
        ui(B) < ui(A), then Vi(B) < Vi(A) = ui(A) in any equilibrium.
        → discard weak orders where tier_i[B] ≤ tier_i[A].
        Forbidden proposals (e.g. non-adjacent under adjacent_step) are skipped
        because the player can never actually make that proposal.

    Rule 2 – Strict Dominance
        If A has strictly the highest payoff for player i among ALL states,
        then Vi(X) < Vi(A) for every transient state X (since all paths
        eventually reach A, spending time at lower payoffs only).
        → discard weak orders where tier_i[X] ≤ tier_i[A] for ANY X ≠ A.
        This subsumes Rule 1 when it applies.  Forbidden proposals do not
        affect Rule 2 (it is a pure payoff comparison, not transition-based).

    Args:
        weak_orders:          (n_orders, n_states) int8 tier matrix from
                              _generate_weak_orders.  Smaller tier = better.
        payoff_array:         (n_states, n_players) static payoffs.
        state_names:          list of state names (length n_states).
        players:              list of player names (length n_players).
        committee_idxs:       committee_idxs[pi][ci][ni] = tuple of approver
                              indices for (proposer pi, current ci, next ni).
        absorbing_state_name: state identified as unique absorbing by LCCS.
        forbidden_proposals:  frozenset of (proposer_name, current_state_name,
                              next_state_name) triples that are blocked by the
                              effectivity rule (e.g. non-adjacent under
                              adjacent_step).  If None, no proposals are
                              considered forbidden.

    Returns:
        valid_idx_per_player: list[np.ndarray] — per-player arrays of valid
                              weak-order indices (into weak_orders rows).
        report:               dict with stats and constraint details.
    """
    n_orders, n_states = weak_orders.shape
    n_players = len(players)
    absorbing_idx = state_names.index(absorbing_state_name)
    absorbing_name = state_names[absorbing_idx]
    _forbidden = forbidden_proposals or frozenset()

    # Collect (player_idx, state_b_idx, rule) constraints
    constraints: list[dict[str, Any]] = []

    for pi in range(n_players):
        u_absorbing = float(payoff_array[absorbing_idx, pi])

        # Rule 2: absorbing state has strictly max payoff for this player
        # Use a small epsilon to be robust against numerical noise in payoff tables
        eps = 1e-10
        all_others_lower = all(
            float(payoff_array[bi, pi]) < u_absorbing - eps
            for bi in range(n_states)
            if bi != absorbing_idx
        )
        if all_others_lower:
            for bi in range(n_states):
                if bi != absorbing_idx:
                    constraints.append({"pi": pi, "bi": bi, "rule": "strict_dominance"})
            continue  # Rule 2 covers everything; no need to add Rule 1 separately

        # Rule 1: unilateral exits with lower payoff (skip forbidden proposals)
        for bi in range(n_states):
            if bi == absorbing_idx:
                continue
            if (players[pi], absorbing_name, state_names[bi]) in _forbidden:
                continue
            committee = committee_idxs[pi][absorbing_idx][bi]
            is_unilateral = (len(committee) == 0) or all(c == pi for c in committee)
            if not is_unilateral:
                continue
            u_b = float(payoff_array[bi, pi])
            if u_b < u_absorbing - eps:
                constraints.append({"pi": pi, "bi": bi, "rule": "unilateral_exit"})

    # Deduplicate
    seen: set[tuple[int, int]] = set()
    unique_constraints: list[dict[str, Any]] = []
    for c in constraints:
        key = (c["pi"], c["bi"])
        if key not in seen:
            seen.add(key)
            unique_constraints.append(c)

    # Apply masks per player
    valid_idx_per_player: list[np.ndarray] = []
    report_per_player: list[dict[str, Any]] = []

    for pi in range(n_players):
        player_cs = [c for c in unique_constraints if c["pi"] == pi]
        if not player_cs:
            valid_idx_per_player.append(np.arange(n_orders, dtype=np.int64))
            report_per_player.append({"n_valid": n_orders, "n_pruned": 0, "constraints": []})
            continue

        # Require tier[bi] > tier[absorbing_idx] for every constrained state bi
        mask = np.ones(n_orders, dtype=bool)
        for c in player_cs:
            bi = c["bi"]
            mask &= weak_orders[:, bi] > weak_orders[:, absorbing_idx]

        valid = np.where(mask)[0].astype(np.int64)
        valid_idx_per_player.append(valid)
        report_per_player.append({
            "n_valid": len(valid),
            "n_pruned": n_orders - len(valid),
            "constraints": player_cs,
        })

    total_original = n_orders ** n_players
    total_after: int = 1
    for v in valid_idx_per_player:
        total_after *= len(v)

    report: dict[str, Any] = {
        "absorbing_state": absorbing_state_name,
        "n_orders_original": n_orders,
        "total_original": total_original,
        "total_after_pruning": total_after,
        "reduction_factor": total_original / max(total_after, 1),
        "per_player": report_per_player,
        "constraints": unique_constraints,
    }

    return valid_idx_per_player, report
