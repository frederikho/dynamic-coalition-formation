"""Multiprocessing workers and iterators over ranking combinations."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from lib.equilibrium.ordinal_ranking.induced_strategies import (
    _build_induced_arrays,
    _build_induced_arrays_weak_from_ids,
    _build_transition_matrix,
)
from lib.equilibrium.ordinal_ranking.value_mdp import _solve_values, _verify_fast

_WORKER_CTX: dict[str, Any] | None = None


def _init_worker_ctx(
    players: list[str],
    states: list[str],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    pos: np.ndarray | None,
    state_perms: np.ndarray | None,
    weak_orders: bool,
    approve_lookup: np.ndarray | None,
) -> None:
    global _WORKER_CTX
    _WORKER_CTX = {
        "players": players,
        "states": states,
        "committee_idxs": committee_idxs,
        "protocol_arr": protocol_arr,
        "payoff_array": payoff_array,
        "discounting": discounting,
        "pos": pos,
        "state_perms": state_perms,
        "weak_orders": weak_orders,
        "approve_lookup": approve_lookup,
    }


def _search_chunk(batch_tuples: np.ndarray) -> dict[str, Any]:
    assert _WORKER_CTX is not None
    players = _WORKER_CTX["players"]
    states = _WORKER_CTX["states"]
    committee_idxs = _WORKER_CTX["committee_idxs"]
    protocol_arr = _WORKER_CTX["protocol_arr"]
    payoff_array = _WORKER_CTX["payoff_array"]
    discounting = _WORKER_CTX["discounting"]
    pos = _WORKER_CTX["pos"]
    state_perms = _WORKER_CTX["state_perms"]
    weak_orders = _WORKER_CTX["weak_orders"]
    approve_lookup = _WORKER_CTX["approve_lookup"]
    tested = 0
    for row in batch_tuples:
        order_ids = tuple(int(x) for x in row)
        if weak_orders:
            proposal_probs, approval_action, approval_pass = _build_induced_arrays_weak_from_ids(
                order_ids,
                pos,
                approve_lookup,
                committee_idxs,
            )
            proposal_choice = None
            P_array = _build_transition_matrix(None, protocol_arr, len(states), proposal_probs=proposal_probs)
        else:
            orders = tuple(pos[idx] for idx in order_ids)
            proposal_choice, approval_action, approval_pass = _build_induced_arrays(
                players=players,
                ranks=orders,
                committee_idxs=committee_idxs,
                protocol_arr=protocol_arr,
            )
            proposal_probs = None
            P_array = _build_transition_matrix(proposal_choice, protocol_arr, len(states))
        V_array = _solve_values(P_array, payoff_array, discounting)
        verified, _message = _verify_fast(
            players=players,
            states=states,
            V_array=V_array,
            proposal_choice=proposal_choice,
            approval_action=approval_action,
            approval_pass=approval_pass,
            committee_idxs=committee_idxs,
            proposal_probs=proposal_probs,
        )
        tested += 1
        if verified:
            return {
                "tested": tested,
                "success": {
                    "perms": order_ids,
                    "rankings": tuple(state_perms[idx].copy() for idx in order_ids),
                },
            }
    return {"tested": tested, "success": None}


def _iter_tuples(
    *,
    n_players: int,
    n_perms: int,
    total: int,
    shuffle: bool,
    random_seed: int,
    perm_orders: tuple[np.ndarray, ...] | None,
):
    flat_total = n_perms ** n_players
    if shuffle:
        rng = np.random.RandomState(random_seed)
        if total == flat_total:
            order = np.arange(flat_total, dtype=np.int64)
            rng.shuffle(order)
        else:
            order = rng.choice(flat_total, size=total, replace=False)
        for idx in order:
            digits = []
            remaining = int(idx)
            for _ in range(n_players):
                digits.append(remaining % n_perms)
                remaining //= n_perms
            yield tuple(reversed(digits))
        return

    orders = perm_orders if perm_orders is not None else tuple(
        np.arange(n_perms, dtype=np.int64) for _ in range(n_players)
    )
    done = 0
    for indices in itertools.product(*[range(n_perms) for _ in range(n_players)]):
        if done >= total:
            return
        yield tuple(int(orders[i][indices[i]]) for i in range(n_players))
        done += 1


def _iter_batches(triples_iter, batch_size: int):
    batch: list[tuple[int, int, int]] = []
    for triple in triples_iter:
        batch.append(triple)
        if len(batch) >= batch_size:
            yield np.asarray(batch, dtype=np.int16)
            batch = []
    if batch:
        yield np.asarray(batch, dtype=np.int16)


def _payoff_base_ranks(payoff_array: np.ndarray, n_players: int, n_states: int) -> tuple[np.ndarray, ...]:
    """Return per-player position arrays sorted by descending payoff (rank 0 = best state)."""
    result = []
    for p in range(n_players):
        sorted_states = np.argsort(-payoff_array[:, p], kind='stable')
        rank_arr = np.empty(n_states, dtype=np.int8)
        for r, s in enumerate(sorted_states):
            rank_arr[int(s)] = r
        result.append(rank_arr)
    return tuple(result)


def _iter_rank_combos_large(
    *,
    n_players: int,
    n_states: int,
    total: int | None,
    random_seed: int,
    ranking_order: str,
    payoff_array: np.ndarray,
):
    """Yield n_players-tuples of position arrays (rank[state]=rank) without
    materialising all n_states! permutations.  Used when n_states is too
    large for exhaustive enumeration.  With ranking_order='payoff', the first
    combination is the exact payoff-sorted ordering for each player; all
    subsequent combinations are uniformly random permutations.
    If total is None, yields indefinitely until the caller stops."""
    rng = np.random.RandomState(random_seed)
    emitted = 0

    if ranking_order == "payoff" and (total is None or total > 0):
        yield _payoff_base_ranks(payoff_array, n_players, n_states)
        emitted += 1

    while total is None or emitted < total:
        ranks = []
        for _ in range(n_players):
            perm = rng.permutation(n_states)
            rank_arr = np.empty(n_states, dtype=np.int8)
            for r, s in enumerate(perm):
                rank_arr[int(s)] = r
            ranks.append(rank_arr)
        yield tuple(ranks)
        emitted += 1


def _iter_batches_large(combos_iter, batch_size: int):
    """Batch rank-array combos into np arrays of shape (B, n_players, n_states)."""
    batch: list[np.ndarray] = []
    for combo in combos_iter:
        batch.append(np.stack(combo))  # (n_players, n_states)
        if len(batch) >= batch_size:
            yield np.array(batch, dtype=np.int8)
            batch = []
    if batch:
        yield np.array(batch, dtype=np.int8)


def _search_chunk_large(batch_ranks: np.ndarray) -> dict[str, Any]:
    """Worker for large-mode: batch_ranks is (B, n_players, n_states) int8."""
    assert _WORKER_CTX is not None
    players = _WORKER_CTX["players"]
    states = _WORKER_CTX["states"]
    committee_idxs = _WORKER_CTX["committee_idxs"]
    protocol_arr = _WORKER_CTX["protocol_arr"]
    payoff_array = _WORKER_CTX["payoff_array"]
    discounting = _WORKER_CTX["discounting"]
    tested = 0
    for combo in batch_ranks:  # combo: (n_players, n_states)
        ranks = tuple(combo[i] for i in range(combo.shape[0]))
        proposal_choice, approval_action, approval_pass = _build_induced_arrays(
            players=players,
            ranks=ranks,
            committee_idxs=committee_idxs,
            protocol_arr=protocol_arr,
        )
        P_array = _build_transition_matrix(proposal_choice, protocol_arr, len(states))
        V_array = _solve_values(P_array, payoff_array, discounting)
        verified, _message = _verify_fast(
            players=players,
            states=states,
            V_array=V_array,
            proposal_choice=proposal_choice,
            approval_action=approval_action,
            approval_pass=approval_pass,
            committee_idxs=committee_idxs,
        )
        tested += 1
        if verified:
            return {"tested": tested, "success": {"ranks": ranks}}
    return {"tested": tested, "success": None}
