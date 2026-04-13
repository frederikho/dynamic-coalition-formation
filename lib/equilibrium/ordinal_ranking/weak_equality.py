"""Numerical solver for tie-breaking in weak ordinal rankings."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import root as scipy_root

from lib.equilibrium.ordinal_ranking.induced_strategies import (
    _build_induced_arrays_weak,
    _induce_profile_from_weak_orders,
)
from lib.equilibrium.ordinal_ranking.value_mdp import _solve_values, _verify_fast
from lib.equilibrium.ordinal_ranking.numba_loops import (
    _NUMBA_AVAILABLE,
    _solve_weak_equalities_nb,
    _residuals_nb_prep,
    _residuals_nb_p_agg,
    _residuals_nb_residuals,
    _solve_V_nb,
)

# Cache of pre-allocated guess arrays indexed by n_vars.
_WEAK_GUESS_CACHE: dict[int, list[np.ndarray]] = {}
_NEWTON_GUESS_LIMIT = 4


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax implementation."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def _sigmoid_scalar(value: float) -> float:
    z = max(-50.0, min(50.0, float(value)))
    return 1.0 / (1.0 + math.exp(-z))


def _weak_tie_structure(
    players: list[str],
    states: list[str],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, tuple[str, ...]]]]:
    free_approvals: list[tuple[str, str, str, str]] = []
    free_approvals_set: set[tuple[str, str, str, str]] = set()
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
                        if key not in free_approvals_set:
                            free_approvals_set.add(key)
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
    return len(free_approvals) + sum(len(w) - 1 for _, _, w in proposal_rows)


def _build_P_direct(
    best_phys: np.ndarray,
    canon_action: np.ndarray,
    canon_pass: np.ndarray,
    canon_probs: np.ndarray,
    fa_idx: list[tuple[int, int, int, int]],
    pt_idx: list[tuple[int, int, list[int]]],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    players: list[str],
    states: list[str],
    protocol_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_players = len(players)
    n_states = len(states)
    action = canon_action.copy()
    pass_ = canon_pass.copy()
    probs = canon_probs.copy()
    n_free = len(fa_idx)
    pt_set = {(pi, si) for pi, si, _ in pt_idx}

    for k, (pi, ai, ci, ni) in enumerate(fa_idx):
        action[pi, ai, ci, ni] = best_phys[k]

    var_idx = n_free
    for pi, si, widxs in pt_idx:
        if len(widxs) > 1:
            logits = np.zeros(len(widxs))
            logits[:-1] = best_phys[var_idx: var_idx + len(widxs) - 1]
            var_idx += len(widxs) - 1
            pw = softmax(logits, temperature=1.0)
            probs[pi, si, :] = 0.0
            for wk, wk_p in zip(widxs, pw):
                probs[pi, si, wk] = float(wk_p)

    affected = set((fa[0], fa[2]) for fa in fa_idx).union(set((pt[0], pt[1]) for pt in pt_idx))
    for pi, si in affected:
        for ni in range(n_states):
            prob = 1.0
            for ai in committee_idxs[pi][si][ni]:
                prob *= action[pi, ai, si, ni]
            pass_[pi, si, ni] = prob

        if (pi, si) not in pt_set:
            tier_p = tiers[pi]
            approved = [ni for ni in range(n_states) if pass_[pi, si, ni]]
            if approved:
                best_t = min(int(tier_p[ni]) for ni in approved)
                winners = [ni for ni in approved if int(tier_p[ni]) == best_t]
                probs[pi, si, :] = 0.0
                m = 1.0 / len(winners)
                for ni in winners:
                    probs[pi, si, ni] = m

    P = np.einsum('i,ijk,ijk->jk', protocol_arr, probs, pass_)
    row_sums = P.sum(axis=1)
    np.fill_diagonal(P, P.diagonal() + (1.0 - row_sums))
    return P, probs, pass_, action


def _finalize_weak_solution(
    best_phys: np.ndarray,
    canon_action: np.ndarray,
    canon_pass: np.ndarray,
    canon_probs: np.ndarray,
    fa_idx: list[tuple[int, int, int, int]],
    pt_idx: list[tuple[int, int, list[int]]],
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    players: list[str],
    states: list[str],
    protocol: dict[str, float],
    payoffs: pd.DataFrame,
    discounting: float,
    unanimity_required: bool,
    free_approvals: list[tuple[str, str, str, str]],
    proposal_rows: list[tuple[str, str, tuple[str, ...]]],
    protocol_arr: np.ndarray,
    payoff_array: np.ndarray,
    timing_data: dict[str, float] | None = None,
    effectivity: dict | None = None,
) -> dict[str, Any]:
    from lib.equilibrium.solver import EquilibriumSolver

    t_rebuild0 = time.perf_counter()
    P_arr, probs_arr, pass_arr, action_arr = _build_P_direct(
        best_phys, canon_action, canon_pass, canon_probs,
        fa_idx, pt_idx, tiers, committee_idxs, players, states, protocol_arr,
    )
    V_arr = _solve_values(P_arr, payoff_array, discounting)
    if timing_data is not None:
        timing_data["finalize_rebuild"] += (time.perf_counter() - t_rebuild0)

    t_verify0 = time.perf_counter()
    verified, message = _verify_fast(
        players=players, states=states, V_array=V_arr,
        proposal_probs=probs_arr, approval_action=action_arr,
        approval_pass=pass_arr, committee_idxs=committee_idxs,
        proposal_choice=None,
    )
    if timing_data is not None:
        timing_data["finalize_verify"] += (time.perf_counter() - t_verify0)

    t_obj0 = time.perf_counter()
    # Build strategy DataFrame only when effectivity is available
    strategy_df = None
    P_df = None
    P_proposals = None
    P_approvals = None
    if effectivity is not None:
        solver = EquilibriumSolver(
            players=players, states=states,
            effectivity=effectivity,
            protocol=protocol, payoffs=payoffs, discounting=discounting,
            unanimity_required=unanimity_required, verbose=False,
            random_seed=0, initialization_mode="uniform", logger=None,
        )
        _induce_profile_from_weak_orders(solver, players, states, tiers, committee_idxs)
        n_free = len(free_approvals)
        for k, (prop, src, dst, appr) in enumerate(free_approvals):
            solver.r_acceptances[(prop, src, dst, appr)] = float(best_phys[k])
        var_idx = n_free
        for pi, si, widxs in pt_idx:
            prop = players[pi]
            src = states[si]
            if len(widxs) > 1:
                logits = np.zeros(len(widxs))
                logits[:-1] = best_phys[var_idx: var_idx + len(widxs) - 1]
                var_idx += len(widxs) - 1
                pw = softmax(logits, temperature=1.0)
            else:
                pw = np.array([1.0])
            for ns in states:
                solver.p_proposals[(prop, src, ns)] = 0.0
            for wk, wk_p in zip(widxs, pw):
                solver.p_proposals[(prop, src, states[wk])] = float(wk_p)

        strategy_df = solver._create_strategy_dataframe()
        P_df, P_proposals, P_approvals = solver._compute_transition_probabilities_fast()

    V_df = pd.DataFrame(V_arr, index=states, columns=players)
    if timing_data is not None:
        timing_data["finalize_solver_obj"] += (time.perf_counter() - t_obj0)

    return {
        "strategy_df": strategy_df.copy() if strategy_df is not None else None,
        "P": P_df.copy() if P_df is not None else None,
        "V": V_df.copy(),
        "P_proposals": P_proposals.copy() if P_proposals is not None else None,
        "P_approvals": dict(P_approvals) if P_approvals is not None else None,
        "verification_success": verified,
        "verification_message": message,
        "verification_detail": None,
        "free_approvals": list(free_approvals),
        "proposal_rows": list(proposal_rows),
    }


def _solve_weak_equalities(
    *,
    players: list[str],
    states: list[str],
    payoffs: pd.DataFrame,
    protocol: dict[str, float],
    discounting: float,
    unanimity_required: bool = True,
    tiers: tuple[np.ndarray, ...],
    committee_idxs: list[list[list[tuple[int, ...]]]],
    effectivity: dict | None = None,
    max_vars: int | None = None,
    use_newton: bool = True,
    _precomputed_tie_structure: tuple[
        list[tuple[str, str, str, str]],
        list[tuple[str, str, tuple[str, ...]]],
    ] | None = None,
    _precomputed_canon_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    _numba_comm_arr: np.ndarray | None = None,
    _numba_comm_size: np.ndarray | None = None,
    _numba_tiers: np.ndarray | None = None,
    timing_data: dict[str, float] | None = None,
    player_idx: dict[str, int] | None = None,
    state_idx: dict[str, int] | None = None,
    _precomputed_payoff_array: np.ndarray | None = None,
    _precomputed_protocol_arr: np.ndarray | None = None,
    _exit_stats_counts: np.ndarray | None = None,
    _flow_stats: dict[str, int] | None = None,
    _source_path: str | None = None,
) -> dict[str, Any] | None:
    """Solve for free approval/proposal probabilities that make V self-consistent."""
    t_start = time.perf_counter()

    if _precomputed_tie_structure is not None:
        free_approvals, proposal_rows = _precomputed_tie_structure
    else:
        free_approvals, proposal_rows = _weak_tie_structure(players, states, tiers, committee_idxs)
    n_free_approvals = len(free_approvals)
    n_proposal_vars = sum(len(winners) - 1 for _p, _s, winners in proposal_rows)
    n_vars = n_free_approvals + n_proposal_vars

    if n_vars == 0:
        return None
    if max_vars is not None and n_vars > max_vars:
        if timing_data is not None:
            timing_data["n_skipped"] = timing_data.get("n_skipped", 0) + 1
        return None

    n_players = len(players)
    n_states = len(states)
    if player_idx is None:
        player_idx = {p: i for i, p in enumerate(players)}
    if state_idx is None:
        state_idx = {s: i for i, s in enumerate(states)}

    if _precomputed_protocol_arr is not None:
        protocol_arr = _precomputed_protocol_arr
    else:
        protocol_arr = np.array([protocol[p] for p in players], dtype=np.float64)

    if _precomputed_payoff_array is not None:
        payoff_array = _precomputed_payoff_array
    else:
        payoff_array = payoffs.loc[states, players].to_numpy(dtype=np.float64)

    t_s_copy = time.perf_counter()
    if _precomputed_canon_arrays is not None:
        canon_probs, canon_action, canon_pass = (
            _precomputed_canon_arrays[0].copy(),
            _precomputed_canon_arrays[1].copy(),
            _precomputed_canon_arrays[2].copy(),
        )
    else:
        canon_probs, canon_action, canon_pass, _ = _build_induced_arrays_weak(
            players=players, tiers=tiers,
            committee_idxs=committee_idxs, protocol_arr=protocol_arr,
        )
    if timing_data is not None:
        timing_data["solver_setup_copy"] += (time.perf_counter() - t_s_copy)

    t_s_indices = time.perf_counter()
    fa_idx: list[tuple[int, int, int, int]] = [
        (player_idx[prop], player_idx[appr], state_idx[src], state_idx[dst])
        for prop, src, dst, appr in free_approvals
    ]
    pt_idx: list[tuple[int, int, list[int]]] = [
        (player_idx[prop], state_idx[src], [state_idx[w] for w in winners])
        for prop, src, winners in proposal_rows
    ]
    pt_set: set[tuple[int, int]] = {(pi, si) for pi, si, _ in pt_idx}
    if timing_data is not None:
        timing_data["solver_setup_indices"] += (time.perf_counter() - t_s_indices)

    _use_nb = (
        use_newton
        and _NUMBA_AVAILABLE
        and _numba_comm_arr is not None
        and _numba_comm_size is not None
        and _numba_tiers is not None
    )

    t_s_numba = time.perf_counter()
    _n_fa = n_free_approvals
    _fa_arr = np.array(fa_idx, dtype=np.int8).reshape(max(_n_fa, 1), 4)
    _n_pt = len(pt_idx)
    _max_nw = max((len(w) for _, _, w in pt_idx), default=1)
    _pt_pi = np.zeros(max(_n_pt, 1), dtype=np.int8)
    _pt_si = np.zeros(max(_n_pt, 1), dtype=np.int8)
    _pt_widxs = np.zeros((max(_n_pt, 1), max(_max_nw, 1)), dtype=np.int8)
    _pt_nwidxs = np.zeros(max(_n_pt, 1), dtype=np.int8)
    for _t, (_tpi, _tsi, _twidxs) in enumerate(pt_idx):
        _pt_pi[_t] = _tpi; _pt_si[_t] = _tsi; _pt_nwidxs[_t] = len(_twidxs)
        for _j, _wj in enumerate(_twidxs):
            _pt_widxs[_t, _j] = _wj

    _affected: set[tuple[int, int]] = set()
    for _pi, _ai, _ci, _ni in fa_idx:
        _affected.add((_pi, _ci))
    for _pi, _si, _ in pt_idx:
        _affected.add((_pi, _si))
    _aff_list = sorted(_affected)
    _n_aff = len(_aff_list)
    if _n_aff == 0:
        _aff_pi = np.zeros(1, dtype=np.int8)
        _aff_ci = np.zeros(1, dtype=np.int8)
        _aff_is_pt = np.zeros(1, dtype=np.bool_)
    else:
        _aff_pi = np.array([_p for _p, _ in _aff_list], dtype=np.int8)
        _aff_ci = np.array([_c for _, _c in _aff_list], dtype=np.int8)
        _aff_is_pt = np.array([(_p, _c) in pt_set for _p, _c in _aff_list], dtype=np.bool_)
    if timing_data is not None:
        timing_data["solver_setup_numba"] += (time.perf_counter() - t_s_numba)

    if _use_nb:
        def _nb_residuals(raw: np.ndarray) -> np.ndarray:
            t_prep0 = time.perf_counter()
            probs, pass_, action = _residuals_nb_prep(
                raw, canon_probs, canon_action, canon_pass,
                _fa_arr, _n_fa, _pt_pi, _pt_si, _pt_widxs, _pt_nwidxs, _n_pt,
                _aff_pi, _aff_ci, _aff_is_pt, _n_aff,
                _numba_comm_arr, _numba_comm_size, _numba_tiers,
                n_players, n_states,
            )
            t_prep1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_mapping"] += (t_prep1 - t_prep0)

            t_p_agg0 = time.perf_counter()
            P_mat = _residuals_nb_p_agg(probs, pass_, protocol_arr, n_players, n_states)
            t_p_agg1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_p_agg"] += (t_p_agg1 - t_p_agg0)

            t_v_solve0 = time.perf_counter()
            V_mat = _solve_V_nb(P_mat, payoff_array, discounting)
            t_v_solve1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_v_solve"] += (t_v_solve1 - t_v_solve0)

            t_res0 = time.perf_counter()
            res = _residuals_nb_residuals(
                V_mat, pass_, _fa_arr, _n_fa,
                _pt_pi, _pt_si, _pt_widxs, _pt_nwidxs, _n_pt, n_vars,
            )
            t_res1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_residuals"] += (t_res1 - t_res0)
            return res
        _scipy_fn = _nb_residuals
    else:
        def _residuals_sigmoid(raw: np.ndarray) -> np.ndarray:
            t_prep0 = time.perf_counter()
            phys = raw.copy()
            for k in range(n_free_approvals):
                phys[k] = _sigmoid_scalar(float(raw[k]))
            action = canon_action.copy()
            pass_ = canon_pass.copy()
            probs = canon_probs.copy()
            for k, (pi, ai, ci, ni) in enumerate(fa_idx):
                action[pi, ai, ci, ni] = phys[k]
            var_idx = n_free_approvals
            for pi, si, widxs in pt_idx:
                if len(widxs) > 1:
                    logits = np.zeros(len(widxs))
                    logits[:-1] = phys[var_idx: var_idx + len(widxs) - 1]
                    var_idx += len(widxs) - 1
                    pw = softmax(logits, temperature=1.0)
                else:
                    pw = np.array([1.0])
                probs[pi, si, :] = 0.0
                for wk, wk_p in zip(widxs, pw):
                    probs[pi, si, wk] = float(wk_p)
            affected: set[tuple[int, int]] = {(pi, ci) for pi, _ai, ci, _ni in fa_idx}
            for pi, si, _ in pt_idx:
                affected.add((pi, si))
            for pi, ci in affected:
                for ni in range(n_states):
                    p = 1.0
                    for ai in committee_idxs[pi][ci][ni]:
                        p *= action[pi, ai, ci, ni]
                    pass_[pi, ci, ni] = p
                if (pi, ci) not in pt_set:
                    tier_p = tiers[pi]
                    approved = [ni for ni in range(n_states) if pass_[pi, ci, ni]]
                    if approved:
                        best_t = min(int(tier_p[ni]) for ni in approved)
                        winners_l = [ni for ni in approved if int(tier_p[ni]) == best_t]
                        probs[pi, ci, :] = 0.0
                        if winners_l:
                            m = 1.0 / len(winners_l)
                            for ni in winners_l:
                                probs[pi, ci, ni] = m
            t_prep1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_mapping"] += (t_prep1 - t_prep0)

            t_p_agg0 = time.perf_counter()
            P_mat = np.einsum('i,ijk,ijk->jk', protocol_arr, probs, pass_)
            np.fill_diagonal(P_mat, P_mat.diagonal() + (1.0 - P_mat.sum(axis=1)))
            t_p_agg1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_p_agg"] += (t_p_agg1 - t_p_agg0)

            t_v_solve0 = time.perf_counter()
            V_mat = _solve_values(P_mat, payoff_array, discounting)
            t_v_solve1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_v_solve"] += (t_v_solve1 - t_v_solve0)

            t_res0 = time.perf_counter()
            res = []
            for _k, (_pi, ai, ci, ni) in enumerate(fa_idx):
                res.append(float(V_mat[ni, ai]) - float(V_mat[ci, ai]))
            for pi, si, widxs in pt_idx:
                ev0 = (float(pass_[pi, si, widxs[0]]) * float(V_mat[widxs[0], pi])
                       + (1.0 - float(pass_[pi, si, widxs[0]])) * float(V_mat[si, pi]))
                for wk in widxs[1:]:
                    evk = (float(pass_[pi, si, wk]) * float(V_mat[wk, pi])
                           + (1.0 - float(pass_[pi, si, wk])) * float(V_mat[si, pi]))
                    res.append(evk - ev0)
            t_res1 = time.perf_counter()
            if timing_data is not None:
                timing_data["solver_root_residuals"] += (t_res1 - t_res0)
            return np.array(res, dtype=np.float64)
        _scipy_fn = _residuals_sigmoid

    t_guesses0 = time.perf_counter()
    if n_vars not in _WEAK_GUESS_CACHE:
        rng = np.random.RandomState(42)
        _WEAK_GUESS_CACHE[n_vars] = [
            np.zeros(n_vars), np.full(n_vars, -2.2), np.full(n_vars, 2.2),
            np.full(n_vars, -4.0), np.full(n_vars, 4.0),
        ] + [rng.uniform(-3.0, 3.0, size=n_vars) for _ in range(2)]
    guesses = _WEAK_GUESS_CACHE[n_vars]
    if _use_nb:
        guesses = guesses[:_NEWTON_GUESS_LIMIT]
    if timing_data is not None:
        timing_data["solver_setup_guesses"] += (time.perf_counter() - t_guesses0)

    best_phys: np.ndarray | None = None
    best_resid = np.inf
    best_source_path: str | None = None  # Track whether best came from Newton or Scipy

    if timing_data is not None:
        timing_data["solver_setup"] += (time.perf_counter() - t_start)

    _OUTCOME_NB_HIT = 0; _OUTCOME_SUCCESS = 1; _OUTCOME_CONV_BADRESID = 2
    _OUTCOME_MAXFEV = 3; _OUTCOME_XTOL = 4; _OUTCOME_BADPROG = 5
    _OUTCOME_EXCEPTION = 6; _OUTCOME_NB_SKIP = 7

    def _bump_flow(key: str, value: int = 1) -> None:
        if _flow_stats is None:
            return
        _flow_stats[key] = _flow_stats.get(key, 0) + value

    for guess_idx, guess in enumerate(guesses):
        raw: np.ndarray | None = None
        _nb_hit = False
        _seeded_from_newton = False
        scipy_start = guess

        if _use_nb:
            _bump_flow("newton_attempted")
            t_nb0 = time.perf_counter()
            nb_alpha, nb_converged, nb_progress = _solve_weak_equalities_nb(
                guess,
                canon_probs, canon_action, canon_pass,
                _fa_arr, _n_fa,
                _pt_pi, _pt_si, _pt_widxs, _pt_nwidxs, _n_pt,
                _aff_pi, _aff_ci, _aff_is_pt, _n_aff,
                _numba_comm_arr, _numba_comm_size, _numba_tiers,
                protocol_arr, payoff_array, discounting,
                n_players, n_states,
            )
            if timing_data is not None:
                timing_data["solver_nb_newton"] += (time.perf_counter() - t_nb0)
            if nb_converged:
                raw = nb_alpha
                _nb_hit = True
                _bump_flow("newton_converged")
            elif nb_progress > 1.0 + 1e-6:
                scipy_start = nb_alpha
                _seeded_from_newton = True
                _bump_flow("newton_progress_seeded")
            else:
                _bump_flow("newton_no_progress")
                if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                    _exit_stats_counts[guess_idx, _OUTCOME_NB_SKIP] += 1

        if raw is None:
            _bump_flow("scipy_attempted")
            try:
                t_r0 = time.perf_counter()
                sol = scipy_root(_scipy_fn, scipy_start, method="hybr", options={"maxfev": 200})
                if timing_data is not None:
                    timing_data["solver_root"] += (time.perf_counter() - t_r0)
            except Exception:
                _bump_flow("scipy_exception")
                if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                    _exit_stats_counts[guess_idx, _OUTCOME_EXCEPTION] += 1
                continue
            if not sol.success:
                _bump_flow("scipy_unsuccessful")
                if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                    st = int(getattr(sol, "status", 0))
                    if st == 2:
                        _exit_stats_counts[guess_idx, _OUTCOME_MAXFEV] += 1
                    elif st == 3:
                        _exit_stats_counts[guess_idx, _OUTCOME_XTOL] += 1
                    elif st in (4, 5):
                        _exit_stats_counts[guess_idx, _OUTCOME_BADPROG] += 1
                    else:
                        _exit_stats_counts[guess_idx, _OUTCOME_EXCEPTION] += 1
                continue
            _bump_flow("scipy_success_flag")
            raw = np.asarray(sol.x, dtype=np.float64)

        phys = raw.copy()
        for k in range(n_free_approvals):
            phys[k] = _sigmoid_scalar(float(raw[k]))

        # Final residuals check
        t_c0 = time.perf_counter()
        action = canon_action.copy()
        pass_ = canon_pass.copy()
        probs = canon_probs.copy()
        for k, (pi, ai, ci, ni) in enumerate(fa_idx):
            action[pi, ai, ci, ni] = phys[k]
        var_idx = n_free_approvals
        for pi, si, widxs in pt_idx:
            if len(widxs) > 1:
                logits = np.zeros(len(widxs))
                logits[:-1] = phys[var_idx: var_idx + len(widxs) - 1]
                var_idx += len(widxs) - 1
                pw = softmax(logits, temperature=1.0)
            else:
                pw = np.array([1.0])
            probs[pi, si, :] = 0.0
            for wk, wk_p in zip(widxs, pw):
                probs[pi, si, wk] = float(wk_p)
        affected2 = {(pi, ci) for pi, _ai, ci, _ni in fa_idx}
        for pi, si, _ in pt_idx:
            affected2.add((pi, si))
        for pi, ci in affected2:
            for ni in range(n_states):
                p = 1.0
                for ai in committee_idxs[pi][ci][ni]:
                    p *= action[pi, ai, ci, ni]
                pass_[pi, ci, ni] = p
            if (pi, ci) not in pt_set:
                tier_p = tiers[pi]
                approved = [ni for ni in range(n_states) if pass_[pi, ci, ni]]
                if approved:
                    best_t = min(int(tier_p[ni]) for ni in approved)
                    winners_l = [ni for ni in approved if int(tier_p[ni]) == best_t]
                    probs[pi, ci, :] = 0.0
                    if winners_l:
                        m = 1.0 / len(winners_l)
                        for ni in winners_l:
                            probs[pi, ci, ni] = m
        P_mat = np.einsum('i,ijk,ijk->jk', protocol_arr, probs, pass_)
        np.fill_diagonal(P_mat, P_mat.diagonal() + (1.0 - P_mat.sum(axis=1)))
        V_mat = _solve_values(P_mat, payoff_array, discounting)
        res_check = []
        for _k, (_pi, ai, ci, ni) in enumerate(fa_idx):
            res_check.append(float(V_mat[ni, ai]) - float(V_mat[ci, ai]))
        for pi, si, widxs in pt_idx:
            ev0 = (float(pass_[pi, si, widxs[0]]) * float(V_mat[widxs[0], pi])
                   + (1.0 - float(pass_[pi, si, widxs[0]])) * float(V_mat[si, pi]))
            for wk in widxs[1:]:
                evk = (float(pass_[pi, si, wk]) * float(V_mat[wk, pi])
                       + (1.0 - float(pass_[pi, si, wk])) * float(V_mat[si, pi]))
                res_check.append(evk - ev0)
        r = float(np.max(np.abs(np.array(res_check))))
        if timing_data is not None:
            timing_data["solver_check"] += (time.perf_counter() - t_c0)

        if r < best_resid:
            best_resid = r
            best_phys = phys
            if _nb_hit:
                best_source_path = "newton"
            elif _seeded_from_newton:
                best_source_path = "scipy_seeded"
            else:
                best_source_path = "scipy_unseeded"
        if best_resid < 1e-7:
            _bump_flow("final_residual_success")
            if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                if _nb_hit:
                    _exit_stats_counts[guess_idx, _OUTCOME_NB_HIT] += 1
                    _bump_flow("final_valid_from_newton")
                else:
                    _exit_stats_counts[guess_idx, _OUTCOME_SUCCESS] += 1
                    _bump_flow("final_valid_from_scipy")
                    if _seeded_from_newton:
                        _bump_flow("final_valid_from_scipy_seeded")
                    else:
                        _bump_flow("final_valid_from_scipy_unseeded")
            break
        else:
            _bump_flow("final_residual_fail")
            if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                _exit_stats_counts[guess_idx, _OUTCOME_CONV_BADRESID] += 1
                if _nb_hit:
                    _bump_flow("final_invalid_from_newton")
                else:
                    _bump_flow("final_invalid_from_scipy")
                    if _seeded_from_newton:
                        _bump_flow("final_invalid_from_scipy_seeded")
                    else:
                        _bump_flow("final_invalid_from_scipy_unseeded")

    if best_phys is None or best_resid > 1e-7:
        return None

    t_f0 = time.perf_counter()
    res_final = _finalize_weak_solution(
        best_phys, canon_action, canon_pass, canon_probs,
        fa_idx, pt_idx, tiers, committee_idxs,
        players, states, protocol, payoffs, discounting, unanimity_required,
        free_approvals, proposal_rows, protocol_arr, payoff_array,
        timing_data=timing_data,
        effectivity=effectivity,
    )
    if timing_data is not None:
        timing_data["solver_finalize"] += (time.perf_counter() - t_f0)
    
    # Track finalize verification outcome split by solver path origin.
    # Keep aggregate SciPy counters for backward-compatible reporting.
    if res_final is not None and best_source_path is not None:
        if bool(res_final.get("verification_success")):
            _bump_flow(f"finalize_verify_true_from_{best_source_path}")
            if best_source_path.startswith("scipy"):
                _bump_flow("finalize_verify_true_from_scipy")
        else:
            _bump_flow(f"finalize_verify_false_from_{best_source_path}")
            if best_source_path.startswith("scipy"):
                _bump_flow("finalize_verify_false_from_scipy")
    
    return res_final
