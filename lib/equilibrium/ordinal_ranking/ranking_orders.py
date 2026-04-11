"""Weak orders, permutations, and payoff-based ranking enumeration order."""

from __future__ import annotations

import itertools

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
