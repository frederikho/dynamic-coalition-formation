#!/usr/bin/env python3
"""Search ordinal ranking triples against a payoff table.

Given an n=3 payoff table, enumerate triples of strict state rankings
(one ranking per player). For each triple:

- approvals are deterministic: approve iff next state ranks strictly above
  current state for the approver; self-loops are canonically approved
- proposals are deterministic: each proposer chooses the highest-ranked target
  that the approval committee would pass; if none improve on staying, stay put

This yields a canonical pure strategy profile and an induced transition matrix.
The script solves V, verifies equilibrium, and can stop on the first verified
success.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import tempfile
import time
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.excel_writer import write_strategy_table_excel
from lib.equilibrium.solver import softmax
from lib.equilibrium.scenarios import fill_players, get_scenario
from lib.equilibrium.solver import EquilibriumSolver
from lib.effectivity import heyen_lehtomaa_2021
from lib.utils import get_approval_committee, verify_equilibrium_detailed
from lib.verify_cli import _run_verification


def _resolve_payoff_file(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = REPO_ROOT / "payoff_tables" / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find payoff file: {value}")


def _infer_players_from_payoff_table(path: Path) -> list[str]:
    df = pd.read_excel(path, sheet_name="Payoffs", header=1, index_col=0)
    excluded_prefixes = ("W_SAI",)
    excluded_names = {"Source file"}
    players = [
        str(col) for col in df.columns
        if not any(str(col).startswith(prefix) for prefix in excluded_prefixes)
        and str(col) not in excluded_names
    ]
    if not players:
        raise ValueError(f"Could not infer players from payoff table columns in {path.name}")
    return players


def _absorbing_states(P: np.ndarray, states: list[str], edge_threshold: float = 1e-12) -> tuple[str, ...]:
    n = len(states)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        row = P[i]
        for j in range(n):
            if row[j] > edge_threshold:
                adj[i].append(j)

    index_counter = [0]
    stack: list[int] = []
    on_stack = [False] * n
    index = [-1] * n
    lowlink = [-1] * n
    sccs: list[list[int]] = []

    def strongconnect(v: int) -> None:
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj[v]:
            if index[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), n * 10 + 100))
    for v in range(n):
        if index[v] == -1:
            strongconnect(v)

    absorbing: list[str] = []
    for scc in sccs:
        scc_set = set(scc)
        has_outgoing = any(w not in scc_set for v in scc for w in adj[v])
        if has_outgoing:
            continue
        if len(scc) > 1:
            absorbing.extend(states[v] for v in scc)
        else:
            v = scc[0]
            if v in adj[v]:
                absorbing.append(states[v])
    return tuple(sorted(absorbing))


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _print_progress(done: int, total: int, start_time: float, width: int = 30) -> None:
    elapsed = max(1e-9, time.perf_counter() - start_time)
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else float("inf")
    msg = (
        f"\r[{bar}] {done:>9,d}/{total:>9,d}  "
        f"{100.0 * frac:5.1f}%  rate={rate:8.0f}/s  "
        f"eta={_format_eta(remaining)}"
    )
    print(msg, end="", flush=True)


def _build_payoff_config(scenario_name: str, payoff_table: str) -> dict[str, Any]:
    config = get_scenario(scenario_name)
    config["payoff_table"] = payoff_table
    if config.get("players") is None:
        from lib.equilibrium.find import _parse_players_from_payoff_table
        config = fill_players(config, _parse_players_from_payoff_table(Path(payoff_table)))
    return config


def _build_inferred_payoff_config(payoff_path: Path) -> dict[str, Any]:
    players = _infer_players_from_payoff_table(payoff_path)
    uniform = 1.0 / len(players)
    default_scalar = {player: 0.0 for player in players}
    return {
        "scenario_name": f"ordinal_search_{payoff_path.stem}",
        "players": players,
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "discounting": 0.99,
        "unanimity_required": True,
        "protocol": {player: uniform for player in players},
        "base_temp": default_scalar.copy(),
        "ideal_temp": default_scalar.copy(),
        "delta_temp": default_scalar.copy(),
        "m_damage": {player: 1.0 for player in players},
        "power": {player: uniform for player in players},
        "payoff_table": str(payoff_path),
    }


def _induce_profile_from_rankings(
    solver: EquilibriumSolver,
    players: list[str],
    states: list[str],
    ranks: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> None:
    n_players = len(players)
    n_states = len(states)

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


def _build_induced_arrays(
    players: list[str],
    ranks: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    protocol_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                    # non-committee approvers remain 0.0 but are ignored
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


def _sigmoid_scalar(value: float) -> float:
    z = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-z))


def _set_canonical_weak_profile(
    solver: EquilibriumSolver,
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> None:
    _induce_profile_from_weak_orders(solver, players, states, tiers, committee_idxs)


def _weak_tie_structure(
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, tuple[str, ...]]]]:
    free_approvals: list[tuple[str, str, str, str]] = []
    proposal_rows: list[tuple[str, str, tuple[str, ...]]] = []

    for proposer_idx, proposer in enumerate(players):
        for current_idx, current_state in enumerate(states):
            committees_row = committee_idxs[proposer_idx][current_idx]
            approved_targets: list[int] = []
            best_tier = int(tiers[proposer_idx][current_idx])

            for next_idx, next_state in enumerate(states):
                committee = committees_row[next_idx]
                approved = True
                for approver_idx in committee:
                    next_tier = int(tiers[approver_idx][next_idx])
                    current_tier = int(tiers[approver_idx][current_idx])
                    if next_idx != current_idx and next_tier == current_tier:
                        key = (proposer, current_state, next_state, players[approver_idx])
                        if key not in free_approvals:
                            free_approvals.append(key)
                    if next_tier > current_tier:
                        approved = False
                if approved:
                    approved_targets.append(next_idx)
                    best_tier = min(best_tier, int(tiers[proposer_idx][next_idx]))

            winners = tuple(
                states[next_idx]
                for next_idx in approved_targets
                if int(tiers[proposer_idx][next_idx]) == best_tier
            )
            if len(winners) > 1:
                proposal_rows.append((proposer, current_state, winners))

    return free_approvals, proposal_rows


def _weak_free_var_count(
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> int:
    free_approvals, proposal_rows = _weak_tie_structure(players, states, tiers, committee_idxs)
    return len(free_approvals) + sum(len(winners) - 1 for _proposer, _current, winners in proposal_rows)


def _weak_value_param_count(tiers: tuple[np.ndarray, ...]) -> int:
    total = 0
    for player_tiers in tiers:
        counts: dict[int, int] = {}
        for tier in player_tiers:
            counts[int(tier)] = counts.get(int(tier), 0) + 1
        total += sum(size - 1 for size in counts.values() if size > 1)
    return total


def _weak_equality_groups(
    states: list[str],
    tiers: tuple[np.ndarray, ...],
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    grouped: list[tuple[tuple[str, ...], ...]] = []
    for player_tiers in tiers:
        buckets: dict[int, list[str]] = {}
        for state_idx, tier in enumerate(player_tiers):
            buckets.setdefault(int(tier), []).append(states[state_idx])
        groups = tuple(
            tuple(bucket)
            for _tier, bucket in sorted(
                ((tier, members) for tier, members in buckets.items() if len(members) > 1),
                key=lambda item: item[0],
            )
        )
        grouped.append(groups)
    return tuple(grouped)


def _update_proposals_from_values(
    solver: EquilibriumSolver,
    V: pd.DataFrame,
    players: list[str],
    states: list[str],
    P_approvals: dict[tuple, float],
    fixed_supports: dict[tuple[str, str], tuple[str, ...]],
) -> bool:
    changed = False
    tol = 1e-9

    for proposer in players:
        for current_state in states:
            support = fixed_supports.get((proposer, current_state))
            current_value = float(V.loc[current_state, proposer])
            expected_values: dict[str, float] = {}
            best_value = -np.inf
            for next_state in states:
                p_approved = float(P_approvals[(proposer, current_state, next_state)])
                expected = p_approved * float(V.loc[next_state, proposer]) + (1.0 - p_approved) * current_value
                expected_values[next_state] = expected
                if expected > best_value:
                    best_value = expected

            if support is None:
                winners = [ns for ns, value in expected_values.items() if abs(value - best_value) <= tol]
            else:
                support_values = {ns: expected_values[ns] for ns in support}
                local_best = max(support_values.values())
                winners = [ns for ns, value in support_values.items() if abs(value - local_best) <= tol]

            target_mass = 1.0 / len(winners)
            for next_state in states:
                key = (proposer, current_state, next_state)
                new_value = target_mass if next_state in winners else 0.0
                if abs(solver.p_proposals[key] - new_value) > 1e-12:
                    solver.p_proposals[key] = new_value
                    changed = True

    return changed


def _solve_weak_equalities(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol: dict[str, float],
    payoffs: pd.DataFrame,
    discounting: float,
    unanimity_required: bool,
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    max_vars: int | None,
    max_br_iters: int = 50,
) -> dict[str, Any] | None:
    try:
        from scipy.optimize import root
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Weak equality refinement requires SciPy. Install it in the active environment "
            "or run without --weak-equality-solve."
        ) from exc

    solver = EquilibriumSolver(
        players=players,
        states=states,
        effectivity=effectivity,
        protocol=protocol,
        payoffs=payoffs,
        discounting=discounting,
        unanimity_required=unanimity_required,
        verbose=False,
        random_seed=0,
        initialization_mode="uniform",
        logger=None,
    )
    _set_canonical_weak_profile(solver, players, states, tiers, committee_idxs)
    free_approvals, proposal_rows = _weak_tie_structure(players, states, tiers, committee_idxs)

    n_vars = len(free_approvals) + sum(len(winners) - 1 for _proposer, _current, winners in proposal_rows)
    if n_vars == 0:
        return None
    if max_vars is not None and n_vars > max_vars:
        return None

    fixed_supports = {(proposer, current_state): winners for proposer, current_state, winners in proposal_rows}

    def _compute_current() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple, float], dict[tuple, float]]:
        strategy_df = solver._create_strategy_dataframe()
        P, _P_proposals, P_approvals = solver._compute_transition_probabilities(strategy_df)
        V = solver._solve_value_functions(P)
        return strategy_df, P, P_approvals, V

    def _apply_params(params: np.ndarray) -> None:
        _set_canonical_weak_profile(solver, players, states, tiers, committee_idxs)
        idx = 0
        for key in free_approvals:
            solver.r_acceptances[key] = _sigmoid_scalar(float(params[idx]))
            idx += 1
        for proposer, current_state, winners in proposal_rows:
            logits = np.zeros(len(winners), dtype=np.float64)
            if len(winners) > 1:
                logits[:-1] = params[idx:idx + len(winners) - 1]
                idx += len(winners) - 1
            probs = softmax(logits, temperature=1.0)
            for next_state in states:
                key = (proposer, current_state, next_state)
                solver.p_proposals[key] = 0.0
            for winner, prob in zip(winners, probs):
                solver.p_proposals[(proposer, current_state, winner)] = float(prob)

    def _residuals(params: np.ndarray) -> np.ndarray:
        _apply_params(params)
        for _ in range(max_br_iters):
            _strategy_df, _P, P_approvals, V = _compute_current()
            changed = _update_proposals_from_values(
                solver,
                V,
                players,
                states,
                P_approvals,
                fixed_supports=fixed_supports,
            )
            if not changed:
                break
        _strategy_df, _P, P_approvals, V = _compute_current()

        residuals: list[float] = []
        for proposer, current_state, winners in proposal_rows:
            baseline = winners[0]
            baseline_value = float(P_approvals[(proposer, current_state, baseline)]) * float(V.loc[baseline, proposer]) + (
                1.0 - float(P_approvals[(proposer, current_state, baseline)])
            ) * float(V.loc[current_state, proposer])
            for next_state in winners[1:]:
                expected = float(P_approvals[(proposer, current_state, next_state)]) * float(V.loc[next_state, proposer]) + (
                    1.0 - float(P_approvals[(proposer, current_state, next_state)])
                ) * float(V.loc[current_state, proposer])
                residuals.append(expected - baseline_value)

        for proposer, current_state, next_state, approver in free_approvals:
            residuals.append(float(V.loc[next_state, approver]) - float(V.loc[current_state, approver]))

        return np.array(residuals, dtype=np.float64)

    guesses = [np.zeros(n_vars, dtype=np.float64)]
    if n_vars:
        guesses.append(np.full(n_vars, 2.0, dtype=np.float64))
        guesses.append(np.full(n_vars, -2.0, dtype=np.float64))

    for guess in guesses:
        solved = root(_residuals, guess, method="hybr")
        if not solved.success:
            continue
        residual = _residuals(np.asarray(solved.x, dtype=np.float64))
        if residual.size and float(np.max(np.abs(residual))) > 1e-7:
            continue
        _apply_params(np.asarray(solved.x, dtype=np.float64))
        for _ in range(max_br_iters):
            strategy_df, P, P_approvals, V = _compute_current()
            changed = _update_proposals_from_values(
                solver,
                V,
                players,
                states,
                P_approvals,
                fixed_supports=fixed_supports,
            )
            if not changed:
                break
        strategy_df, P, P_approvals, V = _compute_current()
        verified, message, detail = verify_equilibrium_detailed(
            {
                "players": players,
                "states": states,
                "state_names": states,
                "effectivity": effectivity,
                "P": P,
                "P_proposals": solver.p_proposals.copy(),
                "P_approvals": P_approvals,
                "V": V,
                "strategy_df": strategy_df,
            }
        )
        if verified:
            return {
                "strategy_df": strategy_df.copy(),
                "P": P.copy(),
                "V": V.copy(),
                "P_proposals": solver.p_proposals.copy(),
                "P_approvals": dict(P_approvals),
                "verification_success": verified,
                "verification_message": message,
                "verification_detail": detail,
                "free_approvals": list(free_approvals),
                "proposal_rows": list(proposal_rows),
            }

    return None


def _solve_values_fast_array(P: np.ndarray, payoff_array: np.ndarray, discounting: float) -> np.ndarray:
    A = np.eye(P.shape[0], dtype=np.float64) - discounting * P
    B = (1.0 - discounting) * payoff_array
    return np.linalg.solve(A, B)


def _solve_values_fast(P_df: pd.DataFrame, payoffs: pd.DataFrame, states: list[str], players: list[str], discounting: float) -> pd.DataFrame:
    P = P_df.loc[states, states].to_numpy(dtype=np.float64)
    U = payoffs.loc[states, players].to_numpy(dtype=np.float64)
    V = _solve_values_fast_array(P, U, discounting)
    return pd.DataFrame(V, index=states, columns=players)


def _verify_equilibrium_fast(
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    P_proposals: dict[tuple, float] | None,
    P_approvals: dict[tuple, float] | None,
    V_df: pd.DataFrame | None,
    *,
    proposal_choice: np.ndarray | None = None,
    proposal_probs: np.ndarray | None = None,
    approval_action: np.ndarray | None = None,
    approval_pass: np.ndarray | None = None,
    V_array: np.ndarray | None = None,
    committee_idxs: list[list[list[tuple[int, ...]]]] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    player_idx = {player: idx for idx, player in enumerate(players)}
    state_idx = {state: idx for idx, state in enumerate(states)}
    if V_array is None:
        assert V_df is not None
        V = V_df.loc[states, players].to_numpy(dtype=np.float64)
    else:
        V = V_array

    use_array_path = (
        (proposal_choice is not None or proposal_probs is not None)
        and approval_action is not None
        and approval_pass is not None
        and committee_idxs is not None
    )

    for proposer in players:
        proposer_col = player_idx[proposer]
        proposer_idx = proposer_col
        for current_state in states:
            current_i = state_idx[current_state]
            pos_prob_next_states: list[str] = []
            best_value = -np.inf
            argmaxes: list[str] = []
            v_current = float(V[current_i, proposer_col])

            for next_state in states:
                next_i = state_idx[next_state]
                if use_array_path:
                    if proposal_probs is not None:
                        proposed = float(proposal_probs[proposer_idx, current_i, next_i])
                    else:
                        proposed = 1.0 if int(proposal_choice[proposer_idx, current_i]) == next_i else 0.0
                    p_approved = float(approval_pass[proposer_idx, current_i, next_i])
                else:
                    proposed = float(P_proposals[(proposer, current_state, next_state)])
                    p_approved = float(P_approvals[(proposer, current_state, next_state)])
                if proposed > 0.0:
                    pos_prob_next_states.append(next_state)
                expected = p_approved * float(V[next_i, proposer_col]) + (1.0 - p_approved) * v_current
                if expected > best_value + 1e-9:
                    best_value = expected
                    argmaxes = [next_state]
                elif abs(expected - best_value) <= 1e-9:
                    argmaxes.append(next_state)

            if not set(pos_prob_next_states).issubset(argmaxes):
                return False, (
                    f"Proposal strategy error with player {proposer}! In state {current_state}, "
                    f"positive probability on state(s) {pos_prob_next_states}, but the argmax states are: {argmaxes}."
                ), {
                    "type": "proposal",
                    "proposer": proposer,
                    "current_state": current_state,
                    "positive_states": pos_prob_next_states,
                    "argmax_states": argmaxes,
                }

    for proposer in players:
        proposer_idx = player_idx[proposer]
        for current_state in states:
            current_i = state_idx[current_state]
            for next_state in states:
                next_i = state_idx[next_state]
                if use_array_path:
                    committee = [players[i] for i in committee_idxs[proposer_idx][current_i][next_i]]
                else:
                    committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                for approver in committee:
                    approver_col = player_idx[approver]
                    v_current = float(V[current_i, approver_col])
                    v_next = float(V[next_i, approver_col])
                    if use_array_path:
                        p_approve = float(approval_action[proposer_idx, approver_col, current_i, next_i])
                    else:
                        p_approve = float(P_approvals[(proposer, current_state, next_state)])
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
                        ), {
                            "type": "approval",
                            "approver": approver,
                            "proposer": proposer,
                            "current_state": current_state,
                            "next_state": next_state,
                            "V_current": v_current,
                            "V_next": v_next,
                            "approval_probability": p_approve,
                        }

    return True, "All tests passed.", None


def _ranking_triplets(
    state_perms: np.ndarray,
    max_combinations: int | None,
    shuffle: bool = False,
    random_seed: int = 0,
    perm_orders: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
):
    n_perms = state_perms.shape[0]
    if perm_orders is None:
        default_order = np.arange(n_perms, dtype=np.int64)
        perm_orders = (default_order, default_order, default_order)
    total = n_perms ** 3
    if max_combinations is not None:
        total = min(total, max_combinations)
    if shuffle:
        rng = np.random.RandomState(random_seed)
        flat_total = n_perms ** 3
        if total == flat_total:
            order = np.arange(flat_total, dtype=np.int64)
            rng.shuffle(order)
        else:
            order = rng.choice(flat_total, size=total, replace=False)
        for done, flat_idx in enumerate(order):
            perm_a = int(flat_idx // (n_perms * n_perms))
            rem = int(flat_idx % (n_perms * n_perms))
            perm_b = int(rem // n_perms)
            perm_c = int(rem % n_perms)
            yield done, total, perm_a, perm_b, perm_c
        return

    done = 0
    order_a, order_b, order_c = perm_orders
    for pos_a in range(n_perms):
        perm_a = int(order_a[pos_a])
        for pos_b in range(n_perms):
            perm_b = int(order_b[pos_b])
            for pos_c in range(n_perms):
                perm_c = int(order_c[pos_c])
                yield done, total, perm_a, perm_b, perm_c
                done += 1
                if done >= total:
                    return


def _payoff_ordering(
    payoff_array: np.ndarray,
    states: list[str],
    players: list[str],
    order_arrays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del states
    orders: list[np.ndarray] = []
    for player_idx, _player in enumerate(players):
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
        for perm_idx, order in enumerate(order_arrays):
            order = np.asarray(order, dtype=np.int64)
            perm = np.array(sorted(range(len(order)), key=lambda idx: (int(order[idx]), int(baseline_pos[idx]))), dtype=np.int64)

            # Primary key: Kendall distance to payoff ranking.
            kendall = 0
            for i in range(len(perm)):
                pi = int(perm[i])
                for j in range(i + 1, len(perm)):
                    pj = int(perm[j])
                    if baseline_pos[pi] > baseline_pos[pj]:
                        kendall += 1

            # Secondary key: Spearman footrule distance.
            footrule = int(sum(abs(rank - int(baseline_pos[int(state_idx)])) for rank, state_idx in enumerate(perm)))

            scored.append((kendall, footrule, perm_idx))

        scored.sort()
        orders.append(np.array([perm_idx for _k, _f, perm_idx in scored], dtype=np.int64))
    return tuple(orders)  # type: ignore[return-value]


def _search_payoff_table(
    config: dict[str, Any],
    max_combinations: int | None,
    progress_every: int,
    stop_on_success: bool,
    use_reference_verifier: bool,
    shuffle: bool,
    random_seed: int,
    ranking_order: str,
    weak_orders: bool,
    weak_exact_reduced: bool,
    weak_equality_solve: bool,
    weak_equality_max_vars: int | None,
) -> dict[str, Any]:
    setup = __import__("lib.equilibrium.find", fromlist=["setup_experiment"]).setup_experiment(config)
    players = setup["players"]
    states = setup["state_names"]
    protocol = setup["protocol"]
    payoffs = setup["payoffs"]
    effectivity = setup["effectivity"]
    discounting = setup["discounting"]
    unanimity_required = setup["unanimity_required"]

    n_players = len(players)
    n_states = len(states)
    if n_players != 3:
        raise ValueError("This script currently expects exactly 3 players.")

    if weak_orders:
        order_arrays = _generate_weak_orders(n_states)
    else:
        order_arrays = np.array(list(itertools.permutations(range(n_states))), dtype=np.int8)
    n_orders = order_arrays.shape[0]
    pos = np.empty((n_orders, n_states), dtype=np.int8)
    for order_idx in range(n_orders):
        if weak_orders:
            pos[order_idx] = order_arrays[order_idx]
        else:
            for rank, state_idx in enumerate(order_arrays[order_idx]):
                pos[order_idx, state_idx] = rank

    committee_idxs: list[list[list[tuple[int, ...]]]] = []
    player_idx = {player: idx for idx, player in enumerate(players)}
    for proposer in players:
        proposer_rows: list[list[tuple[int, ...]]] = []
        for current_state in states:
            row: list[tuple[int, ...]] = []
            for next_state in states:
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                row.append(tuple(player_idx[p] for p in committee))
            proposer_rows.append(row)
        committee_idxs.append(proposer_rows)

    payoff_array = payoffs.loc[states, players].to_numpy(dtype=np.float64)
    protocol_arr = np.array([float(protocol[player]) for player in players], dtype=np.float64)
    perm_orders = None
    if ranking_order == "payoff":
        perm_orders = _payoff_ordering(payoff_array, states, players, order_arrays)
    solver = None
    if use_reference_verifier:
        solver = EquilibriumSolver(
            players=players,
            states=states,
            effectivity=effectivity,
            protocol=protocol,
            payoffs=payoffs,
            discounting=discounting,
            unanimity_required=unanimity_required,
            verbose=False,
            random_seed=0,
            initialization_mode="uniform",
            logger=None,
        )

    start_time = time.perf_counter()
    tested = 0
    verified_successes = 0
    first_success: dict[str, Any] | None = None
    all_successes: list[dict[str, Any]] = []
    weak_exact_zero_params = 0
    weak_exact_deterministic = 0
    weak_exact_nontrivial = 0

    interrupted = False
    try:
        for done, total, perm_a, perm_b, perm_c in _ranking_triplets(
            order_arrays,
            max_combinations,
            shuffle=shuffle,
            random_seed=random_seed,
            perm_orders=perm_orders,
        ):
            ranks = (pos[perm_a], pos[perm_b], pos[perm_c])
            if weak_orders and weak_exact_reduced:
                n_value_params = _weak_value_param_count(ranks)
                n_strategy_free = _weak_free_var_count(players, states, ranks, committee_idxs)
                if n_value_params == 0:
                    weak_exact_zero_params += 1
                if n_strategy_free == 0:
                    weak_exact_deterministic += 1
                else:
                    weak_exact_nontrivial += 1
            if weak_orders:
                proposal_probs, approval_action, approval_pass, P_array = _build_induced_arrays_weak(
                    players=players,
                    tiers=ranks,
                    committee_idxs=committee_idxs,
                    protocol_arr=protocol_arr,
                )
                proposal_choice = None
            else:
                proposal_choice, approval_action, approval_pass, P_array = _build_induced_arrays(
                    players=players,
                    ranks=ranks,
                    committee_idxs=committee_idxs,
                    protocol_arr=protocol_arr,
                )
                proposal_probs = None
            V_array = _solve_values_fast_array(P_array, payoff_array, discounting)
            if use_reference_verifier:
                assert solver is not None
                if weak_orders:
                    _induce_profile_from_weak_orders(solver, players, states, ranks, committee_idxs)
                else:
                    _induce_profile_from_rankings(solver, players, states, ranks, committee_idxs)
                P, P_proposals, P_approvals = solver._compute_transition_probabilities_fast()
                V = pd.DataFrame(V_array, index=states, columns=players)
                strategy_df = solver._create_strategy_dataframe()
                result = {
                    "players": players,
                    "states": states,
                    "state_names": states,
                    "effectivity": effectivity,
                    "P": P,
                    "P_proposals": P_proposals,
                    "P_approvals": P_approvals,
                    "V": V,
                    "strategy_df": strategy_df,
                }
                verified, message, detail = verify_equilibrium_detailed(result)
            else:
                verified, message, detail = _verify_equilibrium_fast(
                    players=players,
                    states=states,
                    effectivity=effectivity,
                    P_proposals=None,
                    P_approvals=None,
                    V_df=None,
                    proposal_choice=proposal_choice,
                    proposal_probs=proposal_probs,
                    approval_action=approval_action,
                    approval_pass=approval_pass,
                    V_array=V_array,
                    committee_idxs=committee_idxs,
                )
            solved_payload: dict[str, Any] | None = None
            if (not verified) and weak_orders and weak_equality_solve:
                solved_payload = _solve_weak_equalities(
                    players=players,
                    states=states,
                    effectivity=effectivity,
                    protocol=protocol,
                    payoffs=payoffs,
                    discounting=discounting,
                    unanimity_required=unanimity_required,
                    tiers=ranks,
                    committee_idxs=committee_idxs,
                    max_vars=weak_equality_max_vars,
                )
                if solved_payload is not None:
                    verified = True
                    message = solved_payload["verification_message"]
                    detail = solved_payload["verification_detail"]
            tested += 1
            if verified:
                verified_successes += 1
                success: dict[str, Any] = {
                    "perms": (perm_a, perm_b, perm_c),
                    "rankings": (
                        order_arrays[perm_a].copy(),
                        order_arrays[perm_b].copy(),
                        order_arrays[perm_c].copy(),
                    ),
                }
                if solved_payload is not None:
                    success.update({
                        "source": "weak_equality_solve",
                        "P": solved_payload["P"],
                        "V": solved_payload["V"],
                        "strategy_df": solved_payload["strategy_df"],
                        "P_proposals": solved_payload["P_proposals"],
                        "P_approvals": solved_payload["P_approvals"],
                    })
                else:
                    success.update({
                        "source": "canonical",
                        "P": pd.DataFrame(P_array, index=states, columns=states),
                        "V": pd.DataFrame(V_array, index=states, columns=players),
                    })
                all_successes.append(success)
                if first_success is None:
                    first_success = success
                if stop_on_success:
                    break
            if progress_every > 0 and tested % progress_every == 0:
                _print_progress(tested, total, start_time)
    except KeyboardInterrupt:
        interrupted = True

    if tested:
        _print_progress(tested, total, start_time)
        print()

    elapsed = time.perf_counter() - start_time
    return {
        "players": players,
        "states": states,
        "effectivity": effectivity,
        "protocol": protocol,
        "payoffs": payoffs,
        "discounting": discounting,
        "unanimity_required": unanimity_required,
        "config": config,
        "total": total,
        "tested": tested,
        "elapsed": elapsed,
        "rate": tested / elapsed if elapsed > 0 else float("nan"),
        "verified_successes": verified_successes,
        "first_success": first_success,
        "all_successes": all_successes,
        "order_arrays": order_arrays,
        "weak_orders": weak_orders,
        "weak_exact_reduced": weak_exact_reduced,
        "weak_exact_zero_params": weak_exact_zero_params,
        "weak_exact_deterministic": weak_exact_deterministic,
        "weak_exact_nontrivial": weak_exact_nontrivial,
        "weak_equality_solve": weak_equality_solve,
        "interrupted": interrupted,
    }


def _rebuild_reference_success(
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    protocol: dict[str, float],
    payoffs: pd.DataFrame,
    discounting: float,
    unanimity_required: bool,
    rankings: tuple[np.ndarray, np.ndarray, np.ndarray],
    weak_orders: bool = False,
    committee_idxs: list[list[list[tuple[int, ...]]]] | None = None,
) -> dict[str, Any]:
    if committee_idxs is None:
        player_idx = {player: idx for idx, player in enumerate(players)}
        committee_idxs = []
        for proposer in players:
            rows = []
            for current_state in states:
                row = []
                for next_state in states:
                    comm = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                    row.append(tuple(player_idx[a] for a in comm))
                rows.append(row)
            committee_idxs.append(rows)

    solver = EquilibriumSolver(
        players=players,
        states=states,
        effectivity=effectivity,
        protocol=protocol,
        payoffs=payoffs,
        discounting=discounting,
        unanimity_required=unanimity_required,
        verbose=False,
        random_seed=0,
        initialization_mode="uniform",
        logger=None,
    )
    if weak_orders:
        weak_tiers = tuple(np.asarray(order, dtype=np.int8) for order in rankings)
        _induce_profile_from_weak_orders(solver, players, states, weak_tiers, committee_idxs)
    else:
        position_rankings: list[np.ndarray] = []
        for perm in rankings:
            perm = np.asarray(perm, dtype=np.int64)
            pos = np.empty(len(perm), dtype=np.int8)
            for rank, state_idx in enumerate(perm):
                pos[int(state_idx)] = rank
            position_rankings.append(pos)
        _induce_profile_from_rankings(solver, players, states, tuple(position_rankings), committee_idxs)
    P, P_proposals, P_approvals = solver._compute_transition_probabilities_fast()
    V = solver._solve_value_functions(P)
    strategy_df = solver._create_strategy_dataframe()
    result = {
        "players": players,
        "states": states,
        "state_names": states,
        "effectivity": effectivity,
        "P": P,
        "P_proposals": P_proposals,
        "P_approvals": P_approvals,
        "V": V,
        "strategy_df": strategy_df,
    }
    verified, message, detail = verify_equilibrium_detailed(result)
    result["verification_success"] = verified
    result["verification_message"] = message
    result["verification_detail"] = detail
    return result


def _materialize_success_result(result: dict[str, Any], success: dict[str, Any]) -> dict[str, Any]:
    if "strategy_df" in success:
        materialized = {
            "players": result["players"],
            "states": result["states"],
            "state_names": result["states"],
            "effectivity": result["effectivity"],
            "P": success["P"],
            "P_proposals": success["P_proposals"],
            "P_approvals": success["P_approvals"],
            "V": success["V"],
            "strategy_df": success["strategy_df"],
        }
        verified, message, detail = verify_equilibrium_detailed(materialized)
        materialized["verification_success"] = verified
        materialized["verification_message"] = message
        materialized["verification_detail"] = detail
        return materialized
    return _rebuild_reference_success(
        players=result["players"],
        states=result["states"],
        effectivity=result["effectivity"],
        protocol=result["protocol"],
        payoffs=result["payoffs"],
        discounting=result["discounting"],
        unanimity_required=result["unanimity_required"],
        rankings=success["rankings"],
        weak_orders=result["weak_orders"],
    )


def _build_excel_metadata(config: dict[str, Any], payoff_table_path: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "scenario_name": config.get("scenario_name", "ordinal_search_candidate"),
        "players": ",".join(config["players"]),
        "n_players": len(config["players"]),
        "power_rule": config["power_rule"],
        "unanimity_required": config["unanimity_required"],
        "min_power": config.get("min_power"),
        "discounting": config["discounting"],
        "payoff_source": "precomputed_table",
        "payoff_table": payoff_table_path,
    }
    for field in ("base_temp", "ideal_temp", "delta_temp", "m_damage", "power", "protocol"):
        for player in config["players"]:
            metadata[f"{field}_{player}"] = config[field][player]
    return metadata


def _verify_via_old_cli_pipeline(
    config: dict[str, Any],
    payoff_table_path: str,
    strategy_df: pd.DataFrame,
    players: list[str],
    states: list[str],
    effectivity: dict[tuple, int],
    V: pd.DataFrame,
    P: pd.DataFrame,
) -> tuple[bool, str, dict[str, Any]]:
    metadata = _build_excel_metadata(config, payoff_table_path)
    with tempfile.NamedTemporaryFile(prefix="ordinal_candidate_", suffix=".xlsx", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        write_strategy_table_excel(
            df=strategy_df,
            excel_file_path=str(temp_path),
            players=players,
            effectivity=effectivity,
            states=states,
            metadata=metadata,
            value_functions=V,
            geo_levels=None,
            deploying_coalitions=None,
            static_payoffs=None,
            transition_matrix=P,
        )
        return _run_verification(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _resolve_output_path(payoff_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return REPO_ROOT / "strategy_tables" / f"ordinal_{payoff_path.stem}.xlsx"


def _serialize_transition_matrix(P: pd.DataFrame, states: list[str]) -> tuple[float, ...]:
    return tuple(round(float(x), 12) for x in P.loc[states, states].to_numpy().flatten())


def _serialize_strategy_df(strategy_df: pd.DataFrame) -> tuple[float, ...]:
    return tuple(round(float(x), 12) for x in strategy_df.fillna(0.0).to_numpy().flatten())


def _resolve_all_output_dir(payoff_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return REPO_ROOT / "strategy_tables" / f"ordinal_all_{payoff_path.stem}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_all_successes(
    *,
    result: dict[str, Any],
    payoff_path: Path,
    output_dir: Path,
    dedup_by: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_base = _build_excel_metadata(result["config"], result["config"]["payoff_table"])
    manifest_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()

    for idx, success in enumerate(result["all_successes"], start=1):
        reference = _materialize_success_result(result, success)
        if dedup_by == "transition":
            dedup_key = _serialize_transition_matrix(reference["P"], result["states"])
        elif dedup_by == "strategy":
            dedup_key = _serialize_strategy_df(reference["strategy_df"])
        else:
            dedup_key = ("ordinal", idx)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        perm_a, perm_b, perm_c = success["perms"]
        suffix = f"{idx:04d}_a{perm_a}_b{perm_b}_c{perm_c}"
        output_path = output_dir / f"{payoff_path.stem}_{suffix}.xlsx"
        metadata = dict(metadata_base)
        metadata["ordinal_ranking_weak_orders"] = result["weak_orders"]
        metadata["ordinal_perm_a"] = perm_a
        metadata["ordinal_perm_b"] = perm_b
        metadata["ordinal_perm_c"] = perm_c
        write_strategy_table_excel(
            df=reference["strategy_df"],
            excel_file_path=str(output_path),
            players=result["players"],
            effectivity=result["effectivity"],
            states=result["states"],
            metadata=metadata,
            value_functions=reference["V"],
            geo_levels=None,
            deploying_coalitions=None,
            static_payoffs=result["payoffs"],
            transition_matrix=reference["P"],
        )
        row = {
            "index": idx,
            "output_file": _display_path(output_path),
            "perm_a": perm_a,
            "perm_b": perm_b,
            "perm_c": perm_c,
        }
        for player, ranking in zip(result["players"], success["rankings"]):
            row[f"ranking_{player}"] = (
                _format_weak_order(result["states"], ranking)
                if result["weak_orders"]
                else _format_ranking(result["states"], ranking)
            )
        row["absorbing_states"] = ", ".join(
            _absorbing_states(
                reference["P"].loc[result["states"], result["states"]].to_numpy(dtype=float),
                result["states"],
            )
        )
        manifest_rows.append(row)

    manifest_path = output_dir / "manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    return manifest_rows


def _format_ranking(states: list[str], perm: np.ndarray) -> str:
    return " > ".join(states[int(idx)] for idx in perm)


def _format_weak_order(states: list[str], tiers: np.ndarray) -> str:
    groups: dict[int, list[str]] = {}
    for state_idx, tier in enumerate(tiers):
        groups.setdefault(int(tier), []).append(states[int(state_idx)])
    ordered = []
    for tier in sorted(groups):
        ordered.append(" = ".join(groups[tier]))
    return " > ".join(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate strict ranking triples against a payoff table and verify them."
    )
    parser.add_argument("file", help="Payoff table path or basename under payoff_tables/")
    parser.add_argument("--scenario", type=str, default=None, help="Optional scenario override for payoff-table search")
    parser.add_argument("--max-combinations", type=int, default=None, help="Cap the number of ranking triples to test")
    parser.add_argument("--progress-every", type=int, default=10000, help="Update progress every N combinations")
    parser.add_argument("--stop-on-success", action="store_true", help="Stop immediately on first verified equilibrium in payoff-table mode")
    parser.add_argument("--write-output", type=str, default=None, help="Write the found strategy table to this path (default: strategy_tables/ordinal_<payoff>.xlsx)")
    parser.add_argument("--write-all", action="store_true", help="Write every verified equilibrium to the default directory under strategy_tables/")
    parser.add_argument("--write-all-output-dir", type=str, default=None, help="Write every verified equilibrium to this directory and include a manifest.csv")
    parser.add_argument(
        "--dedup-by",
        choices=("none", "transition", "strategy"),
        default="none",
        help="When writing all outputs, optionally deduplicate by induced transition matrix or full strategy table",
    )
    parser.add_argument("--shuffle", action="store_true", help="Enumerate ranking triples in random order")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for --shuffle")
    parser.add_argument(
        "--ranking-order",
        choices=("lexicographic", "payoff"),
        default="lexicographic",
        help="Deterministic ranking enumeration order for payoff-table mode",
    )
    parser.add_argument(
        "--weak-orders",
        action="store_true",
        help="Search weak orders (ties allowed) instead of only strict rankings",
    )
    parser.add_argument(
        "--weak-exact-reduced",
        action="store_true",
        help="Track coverage of the exact reduced-game zero-free-variable weak-ranking checker during the sweep",
    )
    parser.add_argument(
        "--weak-equality-solve",
        action="store_true",
        help="In weak-order mode, solve tie equalities numerically instead of testing only the canonical weak profile",
    )
    parser.add_argument(
        "--weak-equality-max-vars",
        type=int,
        default=None,
        help="Optional cap on free variables for weak equality refinement; default is no cap",
    )
    parser.add_argument(
        "--use-reference-verifier",
        action="store_true",
        help="Use the existing DataFrame-based verifier in payoff-table mode (slower, for cross-checking)",
    )
    args = parser.parse_args()
    payoff_path = _resolve_payoff_file(args.file)
    if args.scenario:
        config = _build_payoff_config(args.scenario, str(payoff_path))
        config_source = f"scenario:{args.scenario}"
    else:
        config = _build_inferred_payoff_config(payoff_path)
        config_source = "inferred"
    players = config["players"]
    print("Ordinal Ranking Verification Search")
    print("-" * 80)
    print(f"file: {payoff_path.relative_to(REPO_ROOT)}")
    print(f"config_source: {config_source}")
    print(f"players: {players}")
    setup_preview = __import__("lib.equilibrium.find", fromlist=["setup_experiment"]).setup_experiment(config)
    state_count = len(setup_preview["state_names"])
    if args.weak_orders:
        total_orders = len(_generate_weak_orders(state_count))
    else:
        total_orders = math.factorial(state_count)
    print(f"total_ranking_triples: {total_orders ** len(players):,d}")
    if args.max_combinations is not None:
        print(f"max_combinations: {args.max_combinations:,d}")
    print(f"stop_on_success: {args.stop_on_success}")
    print(f"shuffle: {args.shuffle}")
    print(f"ranking_order: {args.ranking_order}")
    print(f"weak_orders: {args.weak_orders}")
    if args.weak_orders:
        print(f"weak_exact_reduced: {args.weak_exact_reduced}")
        print(f"weak_equality_solve: {args.weak_equality_solve}")
        if args.weak_equality_solve and args.weak_equality_max_vars is not None:
            print(f"weak_equality_max_vars: {args.weak_equality_max_vars}")
    if args.shuffle:
        print(f"random_seed: {args.random_seed}")
    if args.write_output:
        print(f"write_output: {args.write_output}")
    if args.write_all or args.write_all_output_dir:
        print(f"write_all: {args.write_all or bool(args.write_all_output_dir)}")
    if args.write_all_output_dir:
        print(f"write_all_output_dir: {args.write_all_output_dir}")
        print(f"dedup_by: {args.dedup_by}")
    print()

    result = _search_payoff_table(
        config=config,
        max_combinations=args.max_combinations,
        progress_every=args.progress_every,
        stop_on_success=args.stop_on_success,
        use_reference_verifier=args.use_reference_verifier,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        ranking_order=args.ranking_order,
        weak_orders=args.weak_orders,
        weak_exact_reduced=args.weak_exact_reduced,
        weak_equality_solve=args.weak_equality_solve,
        weak_equality_max_vars=args.weak_equality_max_vars,
    )
    states = result["states"]
    order_arrays = result["order_arrays"]
    print("Summary")
    print("-" * 80)
    print(f"tested_combinations:  {result['tested']:>12,d}")
    print(f"wall_time:            {result['elapsed']:>12.2f}s")
    print(f"rate:                 {result['rate']:>12.0f}/s")
    print(f"interrupted:          {str(result['interrupted']):>12s}")
    print(f"verified_successes:   {result['verified_successes']:>12,d}")
    print(f"stored_successes:     {len(result['all_successes']):>12,d}")
    if args.weak_orders and args.weak_exact_reduced:
        print(f"weak_exact_zero_params:{result['weak_exact_zero_params']:>12,d}")
        print(f"weak_exact_deterministic:{result['weak_exact_deterministic']:>8,d}")
        print(f"weak_exact_nontrivial:{result['weak_exact_nontrivial']:>12,d}")

    if args.write_all or args.write_all_output_dir:
        output_dir = _resolve_all_output_dir(payoff_path, args.write_all_output_dir)
        manifest_rows = _write_all_successes(
            result=result,
            payoff_path=payoff_path,
            output_dir=output_dir,
            dedup_by=args.dedup_by,
        )
        print(f"written_successes:    {len(manifest_rows):>12,d}")
        print(f"write_all_dir:        {str(output_dir):>12s}")
        print(f"manifest:             {str((output_dir / 'manifest.csv')):>12s}")

    if result["first_success"] is not None:
        success = result["first_success"]
        perm_a, perm_b, perm_c = success["perms"]
        reference = _materialize_success_result(result, success)
        cli_success, cli_message, cli_details = _verify_via_old_cli_pipeline(
            config=result["config"],
            payoff_table_path=result["config"]["payoff_table"],
            strategy_df=reference["strategy_df"],
            players=result["players"],
            states=result["states"],
            effectivity=result["effectivity"],
            V=reference["V"],
            P=reference["P"],
        )
        print()
        print("First Verified Success")
        print("-" * 80)
        if args.weak_orders:
            print(f"{players[0]}: {_format_weak_order(states, order_arrays[perm_a])}")
            print(f"{players[1]}: {_format_weak_order(states, order_arrays[perm_b])}")
            print(f"{players[2]}: {_format_weak_order(states, order_arrays[perm_c])}")
        else:
            print(f"{players[0]}: {_format_ranking(states, order_arrays[perm_a])}")
            print(f"{players[1]}: {_format_ranking(states, order_arrays[perm_b])}")
            print(f"{players[2]}: {_format_ranking(states, order_arrays[perm_c])}")
        print(f"reference_verification: {reference['verification_success']}")
        print(f"reference_message: {reference['verification_message']}")
        print(f"cli_verification: {cli_success}")
        print(f"cli_message: {cli_message}")
        output_path = None
        if cli_success:
            output_path = _resolve_output_path(payoff_path, args.write_output)
            metadata = _build_excel_metadata(result["config"], result["config"]["payoff_table"])
            write_strategy_table_excel(
                df=reference["strategy_df"],
                excel_file_path=str(output_path),
                players=result["players"],
                effectivity=result["effectivity"],
                states=result["states"],
                metadata=metadata,
                value_functions=reference["V"],
                geo_levels=None,
                deploying_coalitions=None,
                static_payoffs=result["payoffs"],
                transition_matrix=reference["P"],
            )
            print(f"written: {output_path}")
        print()
        print("Transition Matrix")
        print("-" * 80)
        print(reference["P"].loc[states, states].to_string(float_format=lambda x: f"{x:.6f}"))
        print()
        print("Value Functions")
        print("-" * 80)
        print(reference["V"].loc[states, players].to_string(float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
