"""Fast ordinal-ranking equilibrium search for small n=3 payoff tables."""

from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import get_approval_committee

_WORKER_CTX: dict[str, Any] | None = None


def _print_progress(done: int, total: int, start_time: float) -> None:
    elapsed = max(1e-9, time.perf_counter() - start_time)
    frac = done / total if total else 1.0
    width = 30
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else float("inf")
    if math.isfinite(remaining) and remaining >= 0:
        secs = int(round(remaining))
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        eta = f"{hours:d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
    else:
        eta = "?"
    print(
        f"\r[{bar}] {done:>9,d}/{total:>9,d}  {100.0 * frac:5.1f}%  "
        f"rate={rate:8.0f}/s  eta={eta}",
        end="",
        flush=True,
    )


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
    order_ids: tuple[int, int, int],
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


def _init_worker_ctx(
    players: list[str],
    states: list[str],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    discounting: float,
    pos: np.ndarray,
    state_perms: np.ndarray,
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


def _search_chunk(batch_triples: np.ndarray) -> dict[str, Any]:
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
    for perm_a, perm_b, perm_c in batch_triples:
        if weak_orders:
            proposal_probs, approval_action, approval_pass = _build_induced_arrays_weak_from_ids(
                (int(perm_a), int(perm_b), int(perm_c)),
                pos,
                approve_lookup,
                committee_idxs,
            )
            proposal_choice = None
            P_array = _build_transition_matrix(None, protocol_arr, len(states), proposal_probs=proposal_probs)
        else:
            orders = (pos[int(perm_a)], pos[int(perm_b)], pos[int(perm_c)])
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
                    "perms": (int(perm_a), int(perm_b), int(perm_c)),
                    "rankings": (
                        state_perms[int(perm_a)].copy(),
                        state_perms[int(perm_b)].copy(),
                        state_perms[int(perm_c)].copy(),
                    ),
                },
            }
    return {"tested": tested, "success": None}


def _iter_triples(
    *,
    n_perms: int,
    total: int,
    shuffle: bool,
    random_seed: int,
    perm_orders: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
):
    if shuffle:
        rng = np.random.RandomState(random_seed)
        flat_total = n_perms ** 3
        if total == flat_total:
            order = np.arange(flat_total, dtype=np.int64)
            rng.shuffle(order)
        else:
            order = rng.choice(flat_total, size=total, replace=False)
        for idx in order:
            yield (
                int(idx // (n_perms * n_perms)),
                int((idx % (n_perms * n_perms)) // n_perms),
                int(idx % n_perms),
            )
        return

    order_a, order_b, order_c = perm_orders or (
        np.arange(n_perms, dtype=np.int64),
        np.arange(n_perms, dtype=np.int64),
        np.arange(n_perms, dtype=np.int64),
    )
    done = 0
    for pos_a in range(n_perms):
        perm_a = int(order_a[pos_a])
        for pos_b in range(n_perms):
            perm_b = int(order_b[pos_b])
            for pos_c in range(n_perms):
                perm_c = int(order_c[pos_c])
                yield (perm_a, perm_b, perm_c)
                done += 1
                if done >= total:
                    return


def _iter_batches(triples_iter, batch_size: int):
    batch: list[tuple[int, int, int]] = []
    for triple in triples_iter:
        batch.append(triple)
        if len(batch) >= batch_size:
            yield np.asarray(batch, dtype=np.int16)
            batch = []
    if batch:
        yield np.asarray(batch, dtype=np.int16)


def solve_with_ordinal_ranking_n3(
    solver: EquilibriumSolver,
    *,
    max_combinations: int | None = None,
    shuffle: bool = False,
    random_seed: int = 0,
    ranking_order: str = "lexicographic",
    progress_every: int = 0,
    workers: int = 8,
    batch_size: int = 20000,
    weak_orders: bool = False,
    logger=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    players = solver.players
    states = solver.states
    if len(players) != 3:
        raise ValueError("solver_approach='ordinal_ranking' currently expects exactly 3 players.")
    if len(states) < 2:
        raise ValueError("solver_approach='ordinal_ranking' expects at least 2 states.")

    effectivity = solver.effectivity
    if weak_orders:
        state_perms = _generate_weak_orders(len(states))
    else:
        state_perms = np.array(list(itertools.permutations(range(len(states)))), dtype=np.int8)
    n_perms = state_perms.shape[0]
    if weak_orders:
        pos = state_perms.copy()
    else:
        pos = np.empty((n_perms, len(states)), dtype=np.int8)
        for perm_idx in range(n_perms):
            for rank, state_idx in enumerate(state_perms[perm_idx]):
                pos[perm_idx, state_idx] = rank

    player_idx = {player: idx for idx, player in enumerate(players)}
    committee_idxs: list[list[list[tuple[int, ...]]]] = []
    for proposer in players:
        proposer_rows: list[list[tuple[int, ...]]] = []
        for current_state in states:
            row: list[tuple[int, ...]] = []
            for next_state in states:
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                row.append(tuple(player_idx[p] for p in committee))
            proposer_rows.append(row)
        committee_idxs.append(proposer_rows)

    payoff_array = solver.payoffs.loc[states, players].to_numpy(dtype=np.float64)
    protocol_arr = np.array([float(solver.protocol[player]) for player in players], dtype=np.float64)
    approve_lookup = (pos[:, None, :] <= pos[:, :, None]) if weak_orders else None
    perm_orders = None
    if ranking_order == "payoff":
        perm_orders = _payoff_ordering_weak(payoff_array, state_perms, players) if weak_orders else _payoff_ordering(payoff_array, state_perms, players)
    elif ranking_order != "lexicographic":
        raise ValueError(f"Unknown ordinal_ranking_order='{ranking_order}'. Expected 'lexicographic' or 'payoff'.")

    flat_total = n_perms ** 3
    total = min(flat_total, max_combinations) if max_combinations is not None else flat_total
    start_time = time.perf_counter()
    tested = 0
    interrupted = False
    first_success = None

    try:
        if workers > 1:
            triples_iter = _iter_triples(
                n_perms=n_perms,
                total=total,
                shuffle=shuffle,
                random_seed=random_seed,
                perm_orders=perm_orders,
            )
            batches = _iter_batches(triples_iter, batch_size=max(1, int(batch_size)))
            max_workers = max(1, int(workers))
            max_in_flight = max_workers * 2
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker_ctx,
                initargs=(
                    players,
                    states,
                    committee_idxs,
                    protocol_arr,
                    payoff_array,
                    solver.discounting,
                    pos,
                    state_perms,
                    weak_orders,
                    approve_lookup,
                ),
            ) as executor:
                pending = {}

                def submit_one() -> bool:
                    try:
                        batch = next(batches)
                    except StopIteration:
                        return False
                    future = executor.submit(_search_chunk, batch)
                    pending[future] = len(batch)
                    return True

                for _ in range(max_in_flight):
                    if not submit_one():
                        break

                while pending:
                    future = next(as_completed(pending))
                    batch_len = pending.pop(future)
                    chunk_result = future.result()
                    tested += int(chunk_result["tested"])
                    if progress_every > 0:
                        _print_progress(min(tested, total), total, start_time)
                    if chunk_result["success"] is not None:
                        first_success = chunk_result["success"]
                        for other in list(pending):
                            other.cancel()
                        pending.clear()
                        break
                    submit_one()
        elif shuffle:
            rng = np.random.RandomState(random_seed)
            if total == flat_total:
                order = np.arange(flat_total, dtype=np.int64)
                rng.shuffle(order)
            else:
                order = rng.choice(flat_total, size=total, replace=False)
            iterator = (
                (int(idx // (n_perms * n_perms)), int((idx % (n_perms * n_perms)) // n_perms), int(idx % n_perms))
                for idx in order
            )
            for perm_a, perm_b, perm_c in iterator:
                if tested >= total:
                    break
                if weak_orders:
                    proposal_probs, approval_action, approval_pass = _build_induced_arrays_weak_from_ids(
                        (int(perm_a), int(perm_b), int(perm_c)),
                        pos,
                        approve_lookup,
                        committee_idxs,
                    )
                    proposal_choice = None
                    P_array = _build_transition_matrix(None, protocol_arr, len(states), proposal_probs=proposal_probs)
                else:
                    orders = (pos[perm_a], pos[perm_b], pos[perm_c])
                    proposal_choice, approval_action, approval_pass = _build_induced_arrays(
                        players=players,
                        ranks=orders,
                        committee_idxs=committee_idxs,
                        protocol_arr=protocol_arr,
                    )
                    proposal_probs = None
                    P_array = _build_transition_matrix(proposal_choice, protocol_arr, len(states))
                V_array = _solve_values(P_array, payoff_array, solver.discounting)
                verified, message = _verify_fast(
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
                    first_success = {
                        "perms": (perm_a, perm_b, perm_c),
                        "rankings": (
                            state_perms[perm_a].copy(),
                            state_perms[perm_b].copy(),
                            state_perms[perm_c].copy(),
                        ),
                        "message": message,
                    }
                    break
                if progress_every > 0 and tested % progress_every == 0:
                    _print_progress(tested, total, start_time)
        else:
            order_a, order_b, order_c = perm_orders or (
                np.arange(n_perms, dtype=np.int64),
                np.arange(n_perms, dtype=np.int64),
                np.arange(n_perms, dtype=np.int64),
            )
            iterator = (
                (int(order_a[pos_a]), int(order_b[pos_b]), int(order_c[pos_c]))
                for pos_a in range(n_perms)
                for pos_b in range(n_perms)
                for pos_c in range(n_perms)
            )
            for perm_a, perm_b, perm_c in iterator:
                if tested >= total:
                    break
                if weak_orders:
                    proposal_probs, approval_action, approval_pass = _build_induced_arrays_weak_from_ids(
                        (int(perm_a), int(perm_b), int(perm_c)),
                        pos,
                        approve_lookup,
                        committee_idxs,
                    )
                    proposal_choice = None
                    P_array = _build_transition_matrix(None, protocol_arr, len(states), proposal_probs=proposal_probs)
                else:
                    orders = (pos[perm_a], pos[perm_b], pos[perm_c])
                    proposal_choice, approval_action, approval_pass = _build_induced_arrays(
                        players=players,
                        ranks=orders,
                        committee_idxs=committee_idxs,
                        protocol_arr=protocol_arr,
                    )
                    proposal_probs = None
                    P_array = _build_transition_matrix(proposal_choice, protocol_arr, len(states))
                V_array = _solve_values(P_array, payoff_array, solver.discounting)
                verified, message = _verify_fast(
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
                    first_success = {
                        "perms": (perm_a, perm_b, perm_c),
                        "rankings": (
                            state_perms[perm_a].copy(),
                            state_perms[perm_b].copy(),
                            state_perms[perm_c].copy(),
                        ),
                        "message": message,
                    }
                    break
                if progress_every > 0 and tested % progress_every == 0:
                    _print_progress(tested, total, start_time)
    except KeyboardInterrupt:
        interrupted = True

    if progress_every > 0 and tested:
        _print_progress(tested, total, start_time)
        print()

    elapsed = time.perf_counter() - start_time
    if first_success is None:
        strategy_df = solver._create_strategy_dataframe()
        solver_result = {
            "converged": False,
            "stopping_reason": "ordinal_ranking_exhausted" if not interrupted else "interrupted",
            "tested_combinations": tested,
            "total_combinations": total,
            "runtime_seconds": elapsed,
            "final_tau_p": 0.0,
            "final_tau_r": 0.0,
            "outer_iterations": tested,
        }
        return strategy_df, solver_result

    if weak_orders:
        weak_tiers = tuple(np.asarray(order, dtype=np.int8) for order in first_success["rankings"])
        _induce_profile_from_weak_orders(solver, players, states, weak_tiers, committee_idxs)
    else:
        position_rankings: list[np.ndarray] = []
        for perm in first_success["rankings"]:
            perm = np.asarray(perm, dtype=np.int64)
            pos_arr = np.empty(len(perm), dtype=np.int8)
            for rank, state_idx in enumerate(perm):
                pos_arr[int(state_idx)] = rank
            position_rankings.append(pos_arr)
        _induce_profile_from_rankings(solver, players, states, tuple(position_rankings), committee_idxs)
    strategy_df = solver._create_strategy_dataframe()
    solver_result = {
        "converged": True,
        "stopping_reason": "ordinal_ranking_verified",
        "tested_combinations": tested,
        "total_combinations": total,
        "runtime_seconds": elapsed,
        "final_tau_p": 0.0,
        "final_tau_r": 0.0,
        "outer_iterations": tested,
        "ordinal_ranking_perms": first_success["perms"],
    }
    if logger is not None:
        logger.info(
            f"Ordinal ranking search tested {tested:,d}/{total:,d} combinations in {elapsed:.2f}s."
        )
    return strategy_df, solver_result
