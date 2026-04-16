"""Weak orders, permutations, and payoff-based ranking enumeration order."""

from __future__ import annotations

import itertools
from collections import deque
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


# ---------------------------------------------------------------------------
# Theorem 1 + 2 topology-based pruning
# ---------------------------------------------------------------------------

def _find_lcs_connected_components(
    lcs_states: frozenset[str],
    state_names: list[str],
    players: list[str],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    forbidden_proposals: frozenset | None = None,
) -> list[frozenset[str]]:
    """Find weakly connected components of LCS states under permitted transitions.

    An undirected edge exists between LCS states A and B when at least one
    non-forbidden proposal for the transition A→B (or B→A) exists.  Two states
    that share no edge-path at all are provably in different Closed Communicating
    Classes (Theorem 2).
    """
    lcs_idx_set = {state_names.index(s) for s in lcs_states}
    n_players = len(players)
    _forbidden = forbidden_proposals or frozenset()

    # Build undirected adjacency among LCS state indices.
    adj: dict[int, set[int]] = {i: set() for i in lcs_idx_set}
    for pi in range(n_players):
        proposer = players[pi]
        for ci in lcs_idx_set:
            for ni in lcs_idx_set:
                if ci == ni:
                    continue
                if (proposer, state_names[ci], state_names[ni]) in _forbidden:
                    continue
                # At least one non-forbidden proposer → permitted edge.
                adj[ci].add(ni)
                adj[ni].add(ci)  # undirected: B→A counts as well

    # BFS to collect weakly-connected components.
    visited: set[int] = set()
    components: list[frozenset[str]] = []
    for start in sorted(lcs_idx_set):
        if start in visited:
            continue
        component: set[int] = set()
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for nb in adj.get(node, set()):
                if nb not in visited:
                    queue.append(nb)
        components.append(frozenset(state_names[i] for i in component))

    return components


def _find_reachable_lcs_components(
    non_lcs_state_idx: int,
    lcs_idx_set: set[int],
    component_of: dict[int, int],
    n_states: int,
    state_names: list[str],
    players: list[str],
    forbidden_proposals: frozenset | None = None,
) -> set[int]:
    """Find which LCS component indices are reachable from a non-LCS state.

    BFS through the full state graph (non-LCS states are traversed, LCS states
    are collected as destinations but not expanded further).  This implements
    Theorem 1: a non-LCS state is transitory, and its long-run value is bounded
    by the payoff range of the LCS components it can reach.
    """
    n_players = len(players)
    _forbidden = forbidden_proposals or frozenset()

    visited: set[int] = set()
    reachable_components: set[int] = set()
    queue: deque[int] = deque([non_lcs_state_idx])

    while queue:
        curr = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)

        for pi in range(n_players):
            proposer = players[pi]
            for ni in range(n_states):
                if ni == curr:
                    continue
                if (proposer, state_names[curr], state_names[ni]) in _forbidden:
                    continue
                if ni in lcs_idx_set:
                    reachable_components.add(component_of[ni])
                elif ni not in visited:
                    queue.append(ni)

    return reachable_components


def _compute_lcs_topology_pruning_masks(
    weak_orders: np.ndarray,
    payoff_array: np.ndarray,
    state_names: list[str],
    players: list[str],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    lcs_states: frozenset[str],
    forbidden_proposals: frozenset | None = None,
    eps: float = 1e-10,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Prune weak orders using Theorems 1 and 2 (topology-based).

    Theorem 2 (Topological Disconnection): LCS states that lie in different
    weakly-connected components of the permitted-transition graph restricted to
    the LCS belong to separate Closed Communicating Classes.  The Pain Barrier
    then implies: if max u_i over component A < min u_i over component B, then
    V_i(any state in A) < V_i(any state in B) for every equilibrium.

    Theorem 1 (Farsighted Bounding Box): Non-LCS states are transitory — the
    system eventually moves to the LCS.  Hence V_i(T) is bounded by the payoff
    range of the LCS components reachable from T.  If max reachable payoff for
    player i < min payoff in some unreachable component C, then
    V_i(T) < V_i(any state in C).

    Both theorems generate safe ordering constraints:
        tier_i[worse_state] > tier_i[better_state]
    that are used to prune the per-player weak-order search space.

    Args:
        weak_orders:       (n_orders, n_states) int8 tier matrix.
        payoff_array:      (n_states, n_players) static payoffs.
        state_names:       list of state names.
        players:           list of player names.
        committee_idxs:    committee_idxs[pi][ci][ni] = tuple of approver indices.
        lcs_states:        frozenset of state names that are in the LCS.
        forbidden_proposals: frozenset of (proposer, current, next) triples blocked
                           by the effectivity rule (e.g. non-adjacent under
                           adjacent_step).
        eps:               numerical tolerance for strict payoff comparisons.

    Returns:
        valid_idx_per_player: per-player arrays of valid weak-order indices.
        report:              dict with stats and explanatory constraint details.
    """
    n_orders, n_states = weak_orders.shape
    n_players = len(players)
    _empty = np.arange(n_orders, dtype=np.int64)

    lcs_idx_set: set[int] = {state_names.index(s) for s in lcs_states}
    non_lcs_idx_list: list[int] = sorted(i for i in range(n_states) if i not in lcs_idx_set)

    # ── Step 1: Connected components within the LCS ───────────────────────────
    components = _find_lcs_connected_components(
        lcs_states, state_names, players, committee_idxs, forbidden_proposals
    )
    n_components = len(components)

    if n_components <= 1:
        # Single component → Theorem 2 yields nothing; Theorem 1 also cannot
        # produce cross-component constraints.
        valid_idx = [_empty.copy() for _ in range(n_players)]
        return valid_idx, {
            "lcs_size": len(lcs_states),
            "n_components": 1,
            "components": [sorted(c) for c in components],
            "n_constraints": 0,
            "n_orders_original": n_orders,
            "total_original": n_orders ** n_players,
            "total_after_pruning": n_orders ** n_players,
            "reduction_factor": 1.0,
            "per_player": [{"n_valid": n_orders, "n_pruned": 0, "constraints": []} for _ in range(n_players)],
            "message": "LCS is fully connected — no Theorem 2 / Theorem 1 cross-component pruning applicable.",
        }

    # Map each LCS state index → component index.
    component_of: dict[int, int] = {}
    for comp_idx, comp in enumerate(components):
        for s in comp:
            component_of[state_names.index(s)] = comp_idx

    # ── Step 2: Payoff bounds per component per player ────────────────────────
    comp_min = np.full((n_components, n_players), np.inf)
    comp_max = np.full((n_components, n_players), -np.inf)
    for comp_idx, comp in enumerate(components):
        for s in comp:
            si = state_names.index(s)
            for pi in range(n_players):
                u = float(payoff_array[si, pi])
                if u < comp_min[comp_idx, pi]:
                    comp_min[comp_idx, pi] = u
                if u > comp_max[comp_idx, pi]:
                    comp_max[comp_idx, pi] = u

    # ── Step 3: Reachable LCS components for each non-LCS state ──────────────
    non_lcs_reachable: dict[int, set[int]] = {}
    for ti in non_lcs_idx_list:
        non_lcs_reachable[ti] = _find_reachable_lcs_components(
            ti, lcs_idx_set, component_of, n_states,
            state_names, players, forbidden_proposals,
        )

    # Effective payoff ceiling for each non-LCS state per player
    # (max over reachable component payoffs).
    non_lcs_max: dict[int, list[float]] = {}
    for ti in non_lcs_idx_list:
        r = non_lcs_reachable[ti]
        row = []
        for pi in range(n_players):
            if r:
                ceiling = max(float(comp_max[c, pi]) for c in r)
            else:
                ceiling = float(payoff_array[ti, pi])  # stays in own payoff
            row.append(ceiling)
        non_lcs_max[ti] = row

    # ── Step 4: Generate ordering constraints ─────────────────────────────────
    # Each constraint: "for player pi, state worse_idx must rank strictly worse
    #                   than state better_idx".
    raw_constraints: list[dict[str, Any]] = []

    # Theorem 2 — cross-component LCS pairs.
    for ca in range(n_components):
        states_a = sorted(components[ca])
        for cb in range(n_components):
            if ca == cb:
                continue
            states_b = sorted(components[cb])
            for pi in range(n_players):
                if comp_max[ca, pi] < comp_min[cb, pi] - eps:
                    # V_i(any state in C_a) < V_i(any state in C_b).
                    reason = (
                        f"Theorem 2: disconnected LCS components → "
                        f"max u_{players[pi]}(C{ca}={{{', '.join(states_a)}}}) "
                        f"= {comp_max[ca, pi]:.6g} < "
                        f"min u_{players[pi]}(C{cb}={{{', '.join(states_b)}}}) "
                        f"= {comp_min[cb, pi]:.6g}"
                    )
                    for a_name in states_a:
                        for b_name in states_b:
                            raw_constraints.append({
                                "pi": pi,
                                "worse_idx": state_names.index(a_name),
                                "better_idx": state_names.index(b_name),
                                "theorem": "Theorem 2",
                                "worse_state": a_name,
                                "better_state": b_name,
                                "reason": reason,
                            })

    # Theorem 1 — non-LCS (transitory) states vs LCS components they can't reach.
    for ti in non_lcs_idx_list:
        t_name = state_names[ti]
        reachable = non_lcs_reachable[ti]
        for cb in range(n_components):
            if cb in reachable:
                continue  # T can reach this component → no pruning vs it
            states_b = sorted(components[cb])
            for pi in range(n_players):
                t_ceil = non_lcs_max[ti][pi]
                if t_ceil < comp_min[cb, pi] - eps:
                    reachable_names = [
                        f"C{c}={{{', '.join(sorted(components[c]))}}}"
                        for c in sorted(reachable)
                    ]
                    reason = (
                        f"Theorem 1: {t_name} is transitory (non-LCS); "
                        f"reachable LCS: {reachable_names or 'none'}; "
                        f"max reachable u_{players[pi]} = {t_ceil:.6g} < "
                        f"min u_{players[pi]}(C{cb}={{{', '.join(states_b)}}}) "
                        f"= {comp_min[cb, pi]:.6g}"
                    )
                    for b_name in states_b:
                        raw_constraints.append({
                            "pi": pi,
                            "worse_idx": ti,
                            "better_idx": state_names.index(b_name),
                            "theorem": "Theorem 1",
                            "worse_state": t_name,
                            "better_state": b_name,
                            "reason": reason,
                        })

    # Deduplicate by (pi, worse_idx, better_idx).
    seen_keys: set[tuple[int, int, int]] = set()
    unique_constraints: list[dict[str, Any]] = []
    for c in raw_constraints:
        key = (c["pi"], c["worse_idx"], c["better_idx"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_constraints.append(c)

    # ── Step 5: Build per-player masks ────────────────────────────────────────
    valid_idx_per_player: list[np.ndarray] = []
    report_per_player: list[dict[str, Any]] = []

    for pi in range(n_players):
        pcs = [c for c in unique_constraints if c["pi"] == pi]
        if not pcs:
            valid_idx_per_player.append(_empty.copy())
            report_per_player.append({"n_valid": n_orders, "n_pruned": 0, "constraints": []})
            continue

        mask = np.ones(n_orders, dtype=bool)
        for c in pcs:
            wi = c["worse_idx"]
            bi = c["better_idx"]
            # Require: tier of worse_state > tier of better_state
            # (smaller tier = preferred, so better_state must have smaller tier).
            mask &= weak_orders[:, wi] > weak_orders[:, bi]

        valid = np.where(mask)[0].astype(np.int64)
        valid_idx_per_player.append(valid)
        report_per_player.append({
            "n_valid": len(valid),
            "n_pruned": n_orders - len(valid),
            "constraints": pcs,
        })

    total_original = n_orders ** n_players
    total_after: int = 1
    for v in valid_idx_per_player:
        total_after *= len(v)

    return valid_idx_per_player, {
        "lcs_size": len(lcs_states),
        "n_components": n_components,
        "components": [sorted(c) for c in components],
        "comp_min": comp_min.tolist(),
        "comp_max": comp_max.tolist(),
        "non_lcs_reachable": {state_names[ti]: sorted(r) for ti, r in non_lcs_reachable.items()},
        "n_constraints": len(unique_constraints),
        "n_orders_original": n_orders,
        "total_original": total_original,
        "total_after_pruning": total_after,
        "reduction_factor": total_original / max(total_after, 1),
        "per_player": report_per_player,
        "constraints": unique_constraints,
    }
