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
from scipy.optimize import root as scipy_root

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

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


# ---------------------------------------------------------------------------
# Numba-accelerated inner-loop functions
# ---------------------------------------------------------------------------
# These functions accept only numpy arrays (no Python lists/dicts) so they can
# be JIT-compiled to native code.  Compiled artifacts are cached to disk so
# subsequent runs incur no recompilation overhead.
#
# Inputs:
#   tiers_arr  : (n_players, n_states)              int8  — tier rank per player/state
#   comm_arr   : (n_players, n_states, n_states, K) int8  — approver indices, -1 = padding
#   comm_size  : (n_players, n_states, n_states)    int8  — number of actual committee members
#   protocol   : (n_players,)                       float64
#   payoffs    : (n_states, n_players)              float64
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    @_numba.njit(cache=True)
    def _build_arrays_weak_nb(tiers_arr, comm_arr, comm_size, protocol_arr):
        """Numba JIT version of _build_induced_arrays_weak.

        Returns (proposal_probs, approval_action, approval_pass, P).
        """
        n_players = tiers_arr.shape[0]
        n_states = tiers_arr.shape[1]

        proposal_probs = np.zeros((n_players, n_states, n_states))
        approval_action = np.zeros((n_players, n_players, n_states, n_states))
        approval_pass = np.zeros((n_players, n_states, n_states))
        P = np.zeros((n_states, n_states))

        for pi in range(n_players):
            for ci in range(n_states):
                best_tier = tiers_arr[pi, ci]
                approved_mask = np.zeros(n_states, dtype=np.bool_)

                for ni in range(n_states):
                    all_approve = True
                    for k in range(comm_size[pi, ci, ni]):
                        ai = comm_arr[pi, ci, ni, k]
                        if tiers_arr[ai, ni] <= tiers_arr[ai, ci]:
                            approval_action[pi, ai, ci, ni] = 1.0
                        else:
                            all_approve = False
                            # approval_action stays 0.0

                    if all_approve:
                        approval_pass[pi, ci, ni] = 1.0
                        approved_mask[ni] = True
                        if tiers_arr[pi, ni] < best_tier:
                            best_tier = tiers_arr[pi, ni]

                winner_count = 0
                for ni in range(n_states):
                    if approved_mask[ni] and tiers_arr[pi, ni] == best_tier:
                        winner_count += 1

                if winner_count > 0:
                    mass = 1.0 / winner_count
                    for ni in range(n_states):
                        if approved_mask[ni] and tiers_arr[pi, ni] == best_tier:
                            proposal_probs[pi, ci, ni] = mass
                            P[ci, ni] += protocol_arr[pi] * mass

        return proposal_probs, approval_action, approval_pass, P

    @_numba.njit(cache=True)
    def _verify_fast_nb(proposal_probs, approval_action, approval_pass, V, comm_arr, comm_size):
        """Numba JIT equilibrium verifier.  Returns True iff strategy is an equilibrium.

        Only returns a bool (no error details).  Call the Python verifier for
        diagnostics when this returns False and you need to know why.
        """
        n_players = V.shape[1]
        n_states = V.shape[0]
        tol = 1e-9

        # --- Proposal check ---
        for pi in range(n_players):
            for ci in range(n_states):
                v_current = V[ci, pi]
                best_ev = -1e18

                for ni in range(n_states):
                    p_pass = approval_pass[pi, ci, ni]
                    ev = p_pass * V[ni, pi] + (1.0 - p_pass) * v_current
                    if ev > best_ev + tol:
                        best_ev = ev

                for ni in range(n_states):
                    if proposal_probs[pi, ci, ni] > 0.0:
                        p_pass = approval_pass[pi, ci, ni]
                        ev = p_pass * V[ni, pi] + (1.0 - p_pass) * v_current
                        if abs(ev - best_ev) > tol:
                            return False

        # --- Approval check ---
        for pi in range(n_players):
            for ci in range(n_states):
                for ni in range(n_states):
                    for k in range(comm_size[pi, ci, ni]):
                        ai = comm_arr[pi, ci, ni, k]
                        v_c = V[ci, ai]
                        v_n = V[ni, ai]
                        p_approve = approval_action[pi, ai, ci, ni]
                        diff = v_n - v_c
                        if diff > tol:
                            if abs(p_approve - 1.0) > 1e-12:
                                return False
                        elif diff < -tol:
                            if abs(p_approve) > 1e-12:
                                return False
                        # else indifferent — any value in [0,1] is fine

        return True

    @_numba.njit(cache=True)
    def _solve_V_nb(P, payoff_array, discounting):
        """Numba JIT value-function solver.  Equivalent to _solve_values_fast_array."""
        n = P.shape[0]
        A = np.eye(n) - discounting * P
        B = (1.0 - discounting) * payoff_array
        return np.linalg.solve(A, B)

    @_numba.njit(cache=True)
    def _residuals_nb_prep(
        alpha_raw,
        canon_probs, canon_action, canon_pass,
        fa_arr, n_fa,
        pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
        aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
        comm_arr, comm_size,
        tiers,
        n_players, n_states,
    ):
        """Part 1 of _residuals_nb_core: mapping and strategy building."""
        n_vars = alpha_raw.shape[0]
        # Step 1: sigmoid/softmax mapping
        alpha_phys = alpha_raw.copy()
        for k in range(n_fa):
            z = alpha_raw[k]
            if z > 50.0: z = 50.0
            elif z < -50.0: z = -50.0
            if z >= 0.0:
                s = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                s = ez / (1.0 + ez)
            alpha_phys[k] = s

        # Step 2: patch arrays
        action = canon_action.copy()
        pass_ = canon_pass.copy()
        probs = canon_probs.copy()

        for k in range(n_fa):
            action[int(fa_arr[k, 0]), int(fa_arr[k, 1]), int(fa_arr[k, 2]), int(fa_arr[k, 3])] = alpha_phys[k]

        var_idx = n_fa
        for t in range(n_pt):
            nw = int(pt_nwidxs_arr[t])
            nl = nw - 1
            max_logit = 0.0
            for j in range(nl):
                if alpha_phys[var_idx + j] > max_logit: max_logit = alpha_phys[var_idx + j]
            sum_exp = math.exp(-max_logit)
            for j in range(nl): sum_exp += math.exp(alpha_phys[var_idx + j] - max_logit)
            
            pi = int(pt_pi_arr[t]); si = int(pt_si_arr[t])
            for ns in range(n_states): probs[pi, si, ns] = 0.0
            for j in range(nl):
                wj = int(pt_widxs_arr[t, j])
                probs[pi, si, wj] = math.exp(alpha_phys[var_idx + j] - max_logit) / sum_exp
            wlast = int(pt_widxs_arr[t, nw - 1])
            probs[pi, si, wlast] = math.exp(-max_logit) / sum_exp
            var_idx += nl

        # Step 3: rebuild pass_ and probs for affected
        for a in range(n_aff):
            pi = int(aff_pi_arr[a]); ci = int(aff_ci_arr[a])
            for ni in range(n_states):
                p = 1.0
                for k in range(int(comm_size[pi, ci, ni])):
                    p *= action[pi, int(comm_arr[pi, ci, ni, k]), ci, ni]
                pass_[pi, ci, ni] = p
            if not aff_is_pt_arr[a]:
                best_t = int(tiers[pi, ci])
                for ni in range(n_states):
                    if pass_[pi, ci, ni] > 0.0:
                        if int(tiers[pi, ni]) < best_t: best_t = int(tiers[pi, ni])
                for ns in range(n_states): probs[pi, ci, ns] = 0.0
                count_w = 0
                for ni in range(n_states):
                    if pass_[pi, ci, ni] > 0.0 and int(tiers[pi, ni]) == best_t: count_w += 1
                if count_w > 0:
                    m = 1.0 / count_w
                    for ni in range(n_states):
                        if pass_[pi, ci, ni] > 0.0 and int(tiers[pi, ni]) == best_t: probs[pi, ci, ni] = m
        return probs, pass_, action

    @_numba.njit(cache=True)
    def _residuals_nb_p_agg(probs, pass_, protocol_arr, n_players, n_states):
        """Part 2 of _residuals_nb_core: Aggregate P."""
        P = np.zeros((n_states, n_states))
        for pi in range(n_players):
            for ci in range(n_states):
                for ni in range(n_states):
                    P[ci, ni] += protocol_arr[pi] * probs[pi, ci, ni] * pass_[pi, ci, ni]
        for ci in range(n_states):
            P[ci, ci] += 1.0 - np.sum(P[ci, :])
        return P

    @_numba.njit(cache=True)
    def _residuals_nb_residuals(
        V, pass_,
        fa_arr, n_fa,
        pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
        n_vars,
    ):
        """Part 3 of _residuals_nb_core: Compute residual values."""
        res = np.empty(n_vars)
        for k in range(n_fa):
            res[k] = V[int(fa_arr[k, 3]), int(fa_arr[k, 1])] - V[int(fa_arr[k, 2]), int(fa_arr[k, 1])]
        res_idx = n_fa
        for t in range(n_pt):
            pi = int(pt_pi_arr[t]); si = int(pt_si_arr[t]); nw = int(pt_nwidxs_arr[t]); w0 = int(pt_widxs_arr[t, 0])
            ev0 = pass_[pi, si, w0] * V[w0, pi] + (1.0 - pass_[pi, si, w0]) * V[si, pi]
            for j in range(1, nw):
                wj = int(pt_widxs_arr[t, j])
                evj = pass_[pi, si, wj] * V[wj, pi] + (1.0 - pass_[pi, si, wj]) * V[si, pi]
                res[res_idx] = evj - ev0
                res_idx += 1
        return res

    @_numba.njit(cache=True)
    def _residuals_nb_core(
        alpha_raw,
        canon_probs, canon_action, canon_pass,
        fa_arr, n_fa,
        pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
        aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
        comm_arr, comm_size,
        tiers,
        protocol_arr, payoff_array, discounting,
        n_players, n_states,
        compute_jac=False,
    ):
        """Numba JIT residual and Jacobian for _solve_weak_equalities.

        If compute_jac=True, returns (res, jac).
        """
        n_vars = alpha_raw.shape[0]

        # Use split components for consistency (though it's one function here)
        probs, pass_, action = _residuals_nb_prep(
            alpha_raw, canon_probs, canon_action, canon_pass,
            fa_arr, n_fa, pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
            aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
            comm_arr, comm_size, tiers, n_players, n_states
        )
        
        P = _residuals_nb_p_agg(probs, pass_, protocol_arr, n_players, n_states)
        # Fused: compute A_mat once, reuse for both V solve and Jacobian RHS solve.
        A_mat = np.eye(n_states) - discounting * P
        V = np.linalg.solve(A_mat, (1.0 - discounting) * payoff_array)
        res = _residuals_nb_residuals(V, pass_, fa_arr, n_fa, pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt, n_vars)

        # Step 6: Jacobian J = d_res / d_raw (only if compute_jac=True)
        if not compute_jac:
            return res, np.zeros((1, 1))

        # Re-map alpha for d_phys_d_raw
        d_phys_d_raw = np.ones(n_vars)
        for k in range(n_fa):
            z = alpha_raw[k]
            if z > 50.0: z = 50.0
            elif z < -50.0: z = -50.0
            if z >= 0.0:
                s = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                s = ez / (1.0 + ez)
            d_phys_d_raw[k] = s * (1.0 - s)

        # Vectorized Adjoint Jacobian:
        # dV/dx = (I - delta*P)^-1 * (delta * dP/dx * V)
        # We solve for all x (n_vars) at once by stacking RHS columns.
        jac = np.zeros((n_vars, n_vars))
        # A_mat already computed above — reuse it here.

        # RHS_all will have n_players * n_vars columns
        RHS_all = np.zeros((n_states, n_players * n_vars))

        for k in range(n_vars):
            if k < n_fa:
                # Sparse dP: only row p_ci has nonzero entries (p_ci→p_ni and p_ci→p_ci).
                # (dP @ V)[p_ci, pl] = coeff*(V[p_ni,pl] - V[p_ci,pl]); all other rows = 0.
                # Write directly into RHS_all without allocating a full dP matrix.
                p_pi = int(fa_arr[k, 0]); p_ai = int(fa_arr[k, 1]); p_ci = int(fa_arr[k, 2]); p_ni = int(fa_arr[k, 3])
                d_pass = 0.0
                if action[p_pi, p_ai, p_ci, p_ni] > 1e-12:
                    d_pass = pass_[p_pi, p_ci, p_ni] / action[p_pi, p_ai, p_ci, p_ni]
                elif int(comm_size[p_pi, p_ci, p_ni]) == 1:
                    d_pass = 1.0
                coeff = discounting * protocol_arr[p_pi] * probs[p_pi, p_ci, p_ni] * d_pass
                for pl in range(n_players):
                    RHS_all[p_ci, k*n_players + pl] = coeff * (V[p_ni, pl] - V[p_ci, pl])
            else:
                dP = np.zeros((n_states, n_states))
                curr = n_fa
                for t in range(n_pt):
                    nw = int(pt_nwidxs_arr[t]); nl = nw - 1
                    if curr <= k < curr + nl:
                        pi = int(pt_pi_arr[t]); si = int(pt_si_arr[t]); idx_in_pt = k - curr
                        p_j = probs[pi, si, int(pt_widxs_arr[t, idx_in_pt])]
                        for j_nw in range(nw):
                            wj = int(pt_widxs_arr[t, j_nw]); p_i = probs[pi, si, wj]
                            delta_ij = 1.0 if j_nw == idx_in_pt else 0.0
                            dp_i = p_i * (delta_ij - p_j)
                            dP[si, wj] += protocol_arr[pi] * dp_i * pass_[pi, si, wj]
                        dP[si, si] -= np.sum(dP[si, :])
                        break
                    curr += nl
                # Dense dP for proposal-tie variables
                RHS_k = discounting * (dP @ V)
                RHS_all[:, k*n_players : (k+1)*n_players] = RHS_k

        # Solve all at once: O(N^3 + N^2 * n_vars * n_players)
        dV_all = np.linalg.solve(A_mat, RHS_all)

        for k in range(n_vars):
            dV_d_phys = dV_all[:, k*n_players : (k+1)*n_players]
            
            # 1. dV contribution to fa residuals
            for j in range(n_fa):
                ai_j = int(fa_arr[j, 1]); ci_j = int(fa_arr[j, 2]); ni_j = int(fa_arr[j, 3])
                jac[j, k] = (dV_d_phys[ni_j, ai_j] - dV_d_phys[ci_j, ai_j]) * d_phys_d_raw[k]
            
            # 2. dV and d_pass contribution to pt residuals
            res_idx_j = n_fa
            for t in range(n_pt):
                pi_t = int(pt_pi_arr[t]); si_t = int(pt_si_arr[t]); nw_t = int(pt_nwidxs_arr[t]); w0_t = int(pt_widxs_arr[t, 0])
                dk_pass_t = 0.0
                if k < n_fa:
                    p_pi = int(fa_arr[k, 0]); p_ci = int(fa_arr[k, 2])
                    if p_pi == pi_t and p_ci == si_t:
                        p_ai = int(fa_arr[k, 1]); p_ni = int(fa_arr[k, 3])
                        if action[p_pi, p_ai, p_ci, p_ni] > 1e-12:
                            dk_pass_t = pass_[p_pi, p_ci, p_ni] / action[p_pi, p_ai, p_ci, p_ni]
                        elif int(comm_size[p_pi, p_ci, p_ni]) == 1:
                            dk_pass_t = 1.0

                dk_pass_w0 = dk_pass_t if (k < n_fa and int(fa_arr[k, 3]) == w0_t and int(fa_arr[k, 0]) == pi_t and int(fa_arr[k, 2]) == si_t) else 0.0
                d_ev0 = (pass_[pi_t, si_t, w0_t] * dV_d_phys[w0_t, pi_t] + 
                         (1.0 - pass_[pi_t, si_t, w0_t]) * dV_d_phys[si_t, pi_t] +
                         dk_pass_w0 * (V[w0_t, pi_t] - V[si_t, pi_t]))
                
                for j in range(1, nw_t):
                    wj = int(pt_widxs_arr[t, j])
                    dk_pass_wj = dk_pass_t if (k < n_fa and int(fa_arr[k, 3]) == wj and int(fa_arr[k, 0]) == pi_t and int(fa_arr[k, 2]) == si_t) else 0.0
                    d_evj = (pass_[pi_t, si_t, wj] * dV_d_phys[wj, pi_t] + 
                             (1.0 - pass_[pi_t, si_t, wj]) * dV_d_phys[si_t, pi_t] +
                             dk_pass_wj * (V[wj, pi_t] - V[si_t, pi_t]))
                    jac[res_idx_j, k] = (d_evj - d_ev0) * d_phys_d_raw[k]
                    res_idx_j += 1

        return res, jac

    @_numba.njit(cache=True)
    def _solve_weak_equalities_nb(
        alpha_init,
        canon_probs, canon_action, canon_pass,
        fa_arr, n_fa,
        pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
        aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
        comm_arr, comm_size, tiers,
        protocol_arr, payoff_array, discounting,
        n_players, n_states
    ):
        alpha = alpha_init.copy()
        tol = 1e-9
        max_iters = 50

        for i in range(max_iters):
            res, jac = _residuals_nb_core(
                alpha, canon_probs, canon_action, canon_pass,
                fa_arr, n_fa,
                pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
                aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
                comm_arr, comm_size, tiers,
                protocol_arr, payoff_array, discounting,
                n_players, n_states, compute_jac=True
            )

            if np.max(np.abs(res)) < tol:
                return alpha, True

            try:
                delta = -np.linalg.solve(jac, res)
                alpha += 0.5 * delta
            except Exception:
                break

        return alpha, False


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


def _print_progress(done: int, total: int, start_time: float, width: int = 30, recent_rate: float | None = None, breakdown: str = "") -> None:
    import shutil
    elapsed = max(1e-9, time.perf_counter() - start_time)
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else float("inf")
    recent_str = f"  recent={recent_rate:8.0f}/s" if recent_rate is not None else ""
    msg = (
        f"[{bar}] {done:>9,d}/{total:>9,d}  "
        f"{100.0 * frac:5.1f}%  rate={rate:8.0f}/s{recent_str}  "
        f"eta={_format_eta(remaining)}{breakdown}"
    )
    cols = shutil.get_terminal_size(fallback=(200, 24)).columns
    msg = msg[:cols]
    print(f"\r\033[2K{msg}", end="", flush=True)


def _build_payoff_config(
    scenario_name: str,
    payoff_table: str,
    *,
    allow_non_canonical_states: bool = False,
    effectivity_rule: str | None = None,
) -> dict[str, Any]:
    config = get_scenario(scenario_name)
    config["payoff_table"] = payoff_table
    if config.get("players") is None:
        from lib.equilibrium.find import _parse_players_from_payoff_table
        config = fill_players(config, _parse_players_from_payoff_table(Path(payoff_table)))
    if allow_non_canonical_states:
        config["allow_non_canonical_states"] = True
    if effectivity_rule is not None:
        config["effectivity_rule"] = effectivity_rule
    return config


def _build_inferred_payoff_config(
    payoff_path: Path,
    *,
    allow_non_canonical_states: bool = False,
    effectivity_rule: str | None = None,
) -> dict[str, Any]:
    players = _infer_players_from_payoff_table(payoff_path)
    uniform = 1.0 / len(players)
    default_scalar = {player: 0.0 for player in players}
    config = {
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
    if allow_non_canonical_states:
        config["allow_non_canonical_states"] = True
    if effectivity_rule is not None:
        config["effectivity_rule"] = effectivity_rule
    return config


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
    free_approvals_set: set[tuple[str, str, str, str]] = set()  # O(1) membership
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


# Cache of pre-allocated guess arrays indexed by n_vars.
# The 7 guess vectors are always the same for a given n_vars (fixed seed=42),
# so we build them once and reuse across candidates.
_WEAK_GUESS_CACHE: dict[int, list[np.ndarray]] = {}


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
    _precomputed_tie_structure: tuple[
        list[tuple[str, str, str, str]],
        list[tuple[str, str, tuple[str, ...]]],
    ] | None = None,
    _precomputed_canon_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    _numba_comm_arr: "np.ndarray | None" = None,
    _numba_comm_size: "np.ndarray | None" = None,
    _numba_tiers: "np.ndarray | None" = None,
    timing_data: dict[str, float] | None = None,
    player_idx: dict[str, int] | None = None,
    state_idx: dict[str, int] | None = None,
    _precomputed_payoff_array: np.ndarray | None = None,
    _precomputed_protocol_arr: np.ndarray | None = None,
    _exit_stats_counts: "np.ndarray | None" = None,
) -> dict[str, Any] | None:
    """
    Solve for the free approval/proposal probabilities that make V self-consistent
    with the assumed weak ordinal ranking.
    """
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
        return None

    n_players = len(players)
    n_states = len(states)
    if player_idx is None: player_idx = {p: i for i, p in enumerate(players)}
    if state_idx is None: state_idx = {s: i for i, s in enumerate(states)}
    
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
    # Numeric indices for free approvals: (proposer_idx, approver_idx, src_idx, dst_idx)
    fa_idx: list[tuple[int, int, int, int]] = [
        (player_idx[prop], player_idx[appr], state_idx[src], state_idx[dst])
        for prop, src, dst, appr in free_approvals
    ]
    # Numeric indices for proposal ties: (proposer_idx, src_idx, [dst_idx, ...])
    pt_idx: list[tuple[int, int, list[int]]] = [
        (player_idx[prop], state_idx[src], [state_idx[w] for w in winners])
        for prop, src, winners in proposal_rows
    ]
    # Which (proposer, src) pairs have a proposal tie (so we don't double-update)
    pt_set: set[tuple[int, int]] = {(pi, si) for pi, si, _ in pt_idx}
    if timing_data is not None:
        timing_data["solver_setup_indices"] += (time.perf_counter() - t_s_indices)

    _use_nb = (
        _NUMBA_AVAILABLE
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
        for _j, _wj in enumerate(_twidxs): _pt_widxs[_t, _j] = _wj

    _affected: set[tuple[int, int]] = set()
    for _pi, _ai, _ci, _ni in fa_idx: _affected.add((_pi, _ci))
    for _pi, _si, _ in pt_idx: _affected.add((_pi, _si))
    _aff_list = sorted(_affected)
    _n_aff = len(_aff_list)
    if _n_aff == 0:
        _aff_pi = np.zeros(1, dtype=np.int8); _aff_ci = np.zeros(1, dtype=np.int8)
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
                n_players, n_states
            )
            t_prep1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_mapping"] += (t_prep1 - t_prep0)

            t_p_agg0 = time.perf_counter()
            P_mat = _residuals_nb_p_agg(probs, pass_, protocol_arr, n_players, n_states)
            t_p_agg1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_p_agg"] += (t_p_agg1 - t_p_agg0)
            
            t_v_solve0 = time.perf_counter()
            V_mat = _solve_V_nb(P_mat, payoff_array, discounting)
            t_v_solve1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_v_solve"] += (t_v_solve1 - t_v_solve0)
            
            t_res0 = time.perf_counter()
            res = _residuals_nb_residuals(V_mat, pass_, _fa_arr, _n_fa, _pt_pi, _pt_si, _pt_widxs, _pt_nwidxs, _n_pt, n_vars)
            t_res1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_residuals"] += (t_res1 - t_res0)
            return res
        _scipy_fn = _nb_residuals
    else:
        # Wrap with sigmoid for unconstrained optimisation of the approval vars;
        # proposal logits stay unconstrained as-is.
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
                else: pw = np.array([1.0])
                probs[pi, si, :] = 0.0
                for wk, wk_p in zip(widxs, pw): probs[pi, si, wk] = float(wk_p)
            affected: set[tuple[int, int]] = {(pi, ci) for pi, _ai, ci, _ni in fa_idx}
            for pi, si, _ in pt_idx: affected.add((pi, si))
            for pi, ci in affected:
                for ni in range(n_states):
                    p = 1.0
                    for ai in committee_idxs[pi][ci][ni]: p *= action[pi, ai, ci, ni]
                    pass_[pi, ci, ni] = p
                if (pi, ci) not in pt_set:
                    tier_p = tiers[pi]; best_t = int(tier_p[ci]); approved = []
                    for ni in range(n_states):
                        if pass_[pi, ci, ni]:
                            approved.append(ni); best_t = min(best_t, int(tier_p[ni]))
                    winners_l = [ni for ni in approved if int(tier_p[ni]) == best_t]
                    probs[pi, ci, :] = 0.0
                    if winners_l:
                        m = 1.0 / len(winners_l)
                        for ni in winners_l: probs[pi, ci, ni] = m
            t_prep1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_mapping"] += (t_prep1 - t_prep0)

            t_p_agg0 = time.perf_counter()
            P_mat = np.einsum('i,ijk,ijk->jk', protocol_arr, probs, pass_)
            np.fill_diagonal(P_mat, P_mat.diagonal() + (1.0 - P_mat.sum(axis=1)))
            t_p_agg1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_p_agg"] += (t_p_agg1 - t_p_agg0)
            
            t_v_solve0 = time.perf_counter()
            V_mat = _solve_values_fast_array(P_mat, payoff_array, discounting)
            t_v_solve1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_v_solve"] += (t_v_solve1 - t_v_solve0)
            
            t_res0 = time.perf_counter()
            res = []
            for _k, (_pi, ai, ci, ni) in enumerate(fa_idx): res.append(float(V_mat[ni, ai]) - float(V_mat[ci, ai]))
            for pi, si, widxs in pt_idx:
                ev0 = float(pass_[pi, si, widxs[0]]) * float(V_mat[widxs[0], pi]) + (1.0 - float(pass_[pi, si, widxs[0]])) * float(V_mat[si, pi])
                for wk in widxs[1:]:
                    evk = float(pass_[pi, si, wk]) * float(V_mat[wk, pi]) + (1.0 - float(pass_[pi, si, wk])) * float(V_mat[si, pi])
                    res.append(evk - ev0)
            t_res1 = time.perf_counter()
            if timing_data is not None: timing_data["solver_root_residuals"] += (t_res1 - t_res0)
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
    if timing_data is not None:
        timing_data["solver_setup_guesses"] += (time.perf_counter() - t_guesses0)

    best_phys: np.ndarray | None = None
    best_resid = np.inf

    if timing_data is not None:
        timing_data["solver_setup"] += (time.perf_counter() - t_start)

    # _exit_stats_counts shape: (n_guesses, 7)
    # outcome indices: 0=nb_newton_hit, 1=success(scipy), 2=converged+bad_resid,
    #                  3=maxfev, 4=xtol, 5=bad_progress(status 4 or 5), 6=exception
    _OUTCOME_NB_HIT = 0; _OUTCOME_SUCCESS = 1; _OUTCOME_CONV_BADRESID = 2
    _OUTCOME_MAXFEV = 3; _OUTCOME_XTOL = 4; _OUTCOME_BADPROG = 5; _OUTCOME_EXCEPTION = 6

    for guess_idx, guess in enumerate(guesses):
        raw: np.ndarray | None = None
        _nb_hit = False

        # Primary path: Numba Newton with analytical Jacobian (fully JIT, zero Python overhead)
        if _use_nb:
            t_nb0 = time.perf_counter()
            nb_alpha, nb_converged = _solve_weak_equalities_nb(
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

        # Fallback: scipy hybr (also sole path when Numba unavailable)
        if raw is None:
            try:
                t_r0 = time.perf_counter()
                sol = scipy_root(_scipy_fn, guess, method="hybr", options={"maxfev": 200})
                if timing_data is not None:
                    timing_data["solver_root"] += (time.perf_counter() - t_r0)
            except Exception:
                if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                    _exit_stats_counts[guess_idx, _OUTCOME_EXCEPTION] += 1
                continue
            if not sol.success:
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
            raw = np.asarray(sol.x, dtype=np.float64)
        phys = raw.copy()
        for k in range(n_free_approvals):
            phys[k] = _sigmoid_scalar(float(raw[k]))
        
        # 3. Final residuals check to ensure it's a true root
        t_c0 = time.perf_counter()
        action = canon_action.copy(); pass_ = canon_pass.copy(); probs = canon_probs.copy()
        for k, (pi, ai, ci, ni) in enumerate(fa_idx): action[pi, ai, ci, ni] = phys[k]
        var_idx = n_free_approvals
        for pi, si, widxs in pt_idx:
            if len(widxs) > 1:
                logits = np.zeros(len(widxs)); logits[:-1] = phys[var_idx: var_idx + len(widxs) - 1]
                var_idx += len(widxs) - 1; pw = softmax(logits, temperature=1.0)
            else: pw = np.array([1.0])
            probs[pi, si, :] = 0.0
            for wk, wk_p in zip(widxs, pw): probs[pi, si, wk] = float(wk_p)
        affected = {(pi, ci) for pi, _ai, ci, _ni in fa_idx}
        for pi, si, _ in pt_idx: affected.add((pi, si))
        for pi, ci in affected:
            for ni in range(n_states):
                p = 1.0
                for ai in committee_idxs[pi][ci][ni]: p *= action[pi, ai, ci, ni]
                pass_[pi, ci, ni] = p
            if (pi, ci) not in pt_set:
                tier_p = tiers[pi]; best_t = int(tier_p[ci]); approved = []
                for ni in range(n_states):
                    if pass_[pi, ci, ni]: approved.append(ni); best_t = min(best_t, int(tier_p[ni]))
                winners_l = [ni for ni in approved if int(tier_p[ni]) == best_t]
                probs[pi, ci, :] = 0.0
                if winners_l:
                    m = 1.0 / len(winners_l)
                    for ni in winners_l: probs[pi, ci, ni] = m
        P_mat = np.einsum('i,ijk,ijk->jk', protocol_arr, probs, pass_)
        np.fill_diagonal(P_mat, P_mat.diagonal() + (1.0 - P_mat.sum(axis=1)))
        V_mat = _solve_values_fast_array(P_mat, payoff_array, discounting)
        res_check = []
        for _k, (_pi, ai, ci, ni) in enumerate(fa_idx): res_check.append(float(V_mat[ni, ai]) - float(V_mat[ci, ai]))
        for pi, si, widxs in pt_idx:
            ev0 = float(pass_[pi, si, widxs[0]]) * float(V_mat[widxs[0], pi]) + (1.0 - float(pass_[pi, si, widxs[0]])) * float(V_mat[si, pi])
            for wk in widxs[1:]:
                evk = float(pass_[pi, si, wk]) * float(V_mat[wk, pi]) + (1.0 - float(pass_[pi, si, wk])) * float(V_mat[si, pi])
                res_check.append(evk - ev0)
        r = float(np.max(np.abs(np.array(res_check))))
        if timing_data is not None: timing_data["solver_check"] += (time.perf_counter() - t_c0)

        if r < best_resid:
            best_resid = r
            best_phys = phys
        if best_resid < 1e-7:
            if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                if _nb_hit:
                    _exit_stats_counts[guess_idx, _OUTCOME_NB_HIT] += 1
                else:
                    _exit_stats_counts[guess_idx, _OUTCOME_SUCCESS] += 1
            break
        else:
            if _exit_stats_counts is not None and guess_idx < _exit_stats_counts.shape[0]:
                _exit_stats_counts[guess_idx, _OUTCOME_CONV_BADRESID] += 1

    if best_phys is None or best_resid > 1e-7:
        return None

    t_f0 = time.perf_counter()
    res_final = _finalize_weak_solution(best_phys, canon_action, canon_pass, canon_probs, fa_idx, pt_idx, tiers, committee_idxs, players, states, effectivity, protocol, payoffs, discounting, unanimity_required, free_approvals, proposal_rows, timing_data=timing_data)
    if timing_data is not None:
        timing_data["solver_finalize"] += (time.perf_counter() - t_f0)
    return res_final

    # ------------------------------------------------------------------
    # Build Numba arrays for fast residual evaluation (scipy calls these)
    # ------------------------------------------------------------------
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
        for _j, _wj in enumerate(_twidxs): _pt_widxs[_t, _j] = _wj

    _affected: set[tuple[int, int]] = set()
    for _pi, _ai, _ci, _ni in fa_idx: _affected.add((_pi, _ci))
    for _pi, _si, _ in pt_idx: _affected.add((_pi, _si))
    _aff_list = sorted(_affected)
    _n_aff = len(_aff_list)
    if _n_aff == 0:
        _aff_pi = np.zeros(1, dtype=np.int8); _aff_ci = np.zeros(1, dtype=np.int8)
        _aff_is_pt = np.zeros(1, dtype=np.bool_)
    else:
        _aff_pi = np.array([_p for _p, _ in _aff_list], dtype=np.int8)
        _aff_ci = np.array([_c for _, _c in _aff_list], dtype=np.int8)
        _aff_is_pt = np.array([(_p, _c) in pt_set for _p, _c in _aff_list], dtype=np.bool_)

    if _use_nb:
        def _nb_residuals(raw: np.ndarray) -> np.ndarray:
            res, _ = _residuals_nb_core(
                raw, canon_probs, canon_action, canon_pass,
                _fa_arr, _n_fa,
                _pt_pi, _pt_si, _pt_widxs, _pt_nwidxs, _n_pt,
                _aff_pi, _aff_ci, _aff_is_pt, _n_aff,
                _numba_comm_arr, _numba_comm_size, _numba_tiers,
                protocol_arr, payoff_array, discounting,
                n_players, n_states, compute_jac=False,
            )
            return res
        _scipy_fn = _nb_residuals
    else:
        _scipy_fn = _residuals_sigmoid

    best_phys: np.ndarray | None = None
    best_resid = np.inf

    for guess in guesses:
        try:
            sol = scipy_root(_scipy_fn, guess, method="hybr", options={"maxfev": 200})
        except Exception:
            continue
        if not sol.success:
            continue
        raw = np.asarray(sol.x, dtype=np.float64)
        phys = raw.copy()
        for k in range(n_free_approvals):
            phys[k] = _sigmoid_scalar(float(raw[k]))
        r = float(np.max(np.abs(_residuals(phys))))
        if r < best_resid:
            best_resid = r
            best_phys = phys
        if best_resid < 1e-7:
            break

    if best_phys is None or best_resid > 1e-7:
        return None

    return _finalize_weak_solution(best_phys, canon_action, canon_pass, canon_probs, fa_idx, pt_idx, tiers, committee_idxs, players, states, effectivity, protocol, payoffs, discounting, unanimity_required, free_approvals, proposal_rows)


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


def _ranking_tuples(
    state_perms: np.ndarray,
    n_players: int,
    max_combinations: int | None,
    shuffle: bool = False,
    random_seed: int = 0,
    perm_orders: tuple[np.ndarray, ...] | None = None,
):
    n_perms = state_perms.shape[0]
    if perm_orders is None:
        default_order = np.arange(n_perms, dtype=np.int64)
        perm_orders = tuple(default_order for _ in range(n_players))
    total = n_perms ** n_players
    if max_combinations is not None:
        total = min(total, max_combinations)
    if shuffle:
        rng = np.random.RandomState(random_seed)
        flat_total = n_perms ** n_players
        if total == flat_total:
            order = np.arange(flat_total, dtype=np.int64)
            rng.shuffle(order)
        else:
            order = rng.choice(flat_total, size=total, replace=False)
        for done, flat_idx in enumerate(order):
            digits = [0] * n_players
            value = int(flat_idx)
            for pos in range(n_players - 1, -1, -1):
                digits[pos] = value % n_perms
                value //= n_perms
            yield done, total, tuple(digits)
        return

    done = 0
    for positions in itertools.product(range(n_perms), repeat=n_players):
        perm_tuple = tuple(int(perm_orders[player_idx][pos]) for player_idx, pos in enumerate(positions))
        yield done, total, perm_tuple
        done += 1
        if done >= total:
            return


def _random_ordering(
    n_orders: int,
    n_players: int,
    random_seed: int = 0,
) -> tuple[np.ndarray, ...]:
    rng = np.random.RandomState(random_seed)
    orders: list[np.ndarray] = []
    for _ in range(n_players):
        order = np.arange(n_orders, dtype=np.int64)
        rng.shuffle(order)
        orders.append(order)
    return tuple(orders)


def _payoff_ordering(
    payoff_array: np.ndarray,
    states: list[str],
    players: list[str],
    order_arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
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
    return tuple(orders)


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
    worker_id: int = 0,
    n_workers: int = 1,
    progress_queue: Any = None,
    order_arrays: np.ndarray | None = None,
    perm_orders: tuple[np.ndarray, ...] | None = None,
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
    if order_arrays is None:
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
    state_idx = {state: idx for idx, state in enumerate(states)}
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

    # Pre-compute committee arrays in numpy format for the Numba fast path.
    # comm_arr[pi, ci, ni, k] = approver index k for (proposer pi, from ci, to ni); -1 = padding.
    # comm_size[pi, ci, ni]   = number of committee members for that transition.
    _max_comm = max(
        len(committee_idxs[pi][ci][ni])
        for pi in range(n_players)
        for ci in range(n_states)
        for ni in range(n_states)
    )
    comm_arr_nb = np.full((n_players, n_states, n_states, max(_max_comm, 1)), -1, dtype=np.int8)
    comm_size_nb = np.zeros((n_players, n_states, n_states), dtype=np.int8)
    for _pi in range(n_players):
        for _ci in range(n_states):
            for _ni in range(n_states):
                for _k, _ai in enumerate(committee_idxs[_pi][_ci][_ni]):
                    comm_arr_nb[_pi, _ci, _ni, _k] = _ai
                comm_size_nb[_pi, _ci, _ni] = len(committee_idxs[_pi][_ci][_ni])

def _finalize_weak_solution(best_phys, canon_action, canon_pass, canon_probs, fa_idx, pt_idx, tiers, committee_idxs, players, states, effectivity, protocol, payoffs, discounting, unanimity_required, free_approvals, proposal_rows, timing_data=None):
    # This rebuilds the solver result after Numba Newton finds a root.
    t_rebuild0 = time.perf_counter()
    P_arr, probs_arr, pass_arr, action_arr = _build_P_direct(best_phys, canon_action, canon_pass, canon_probs, fa_idx, pt_idx, tiers, committee_idxs, players, states)
    payoff_array = payoffs.loc[states, players].to_numpy(dtype=np.float64)
    V_arr = _solve_values_fast_array(P_arr, payoff_array, discounting)
    if timing_data is not None: timing_data["finalize_rebuild"] += (time.perf_counter() - t_rebuild0)

    # Verify directly using the computed arrays — bypasses TransitionProbabilitiesOptimized
    # which uses a different code path and can produce V-inconsistent P_approvals.
    t_verify0 = time.perf_counter()
    verified, message, detail = _verify_equilibrium_fast(
        players=players, states=states, effectivity=effectivity,
        P_proposals=None, P_approvals=None, V_df=None,
        proposal_probs=probs_arr, approval_action=action_arr,
        approval_pass=pass_arr, V_array=V_arr,
        committee_idxs=committee_idxs,
    )
    if timing_data is not None: timing_data["finalize_verify"] += (time.perf_counter() - t_verify0)

    # Build strategy_df and DataFrames for the returned payload (used when writing Excel).
    t_obj0 = time.perf_counter()
    solver = EquilibriumSolver(
        players=players, states=states, effectivity=effectivity,
        protocol=protocol, payoffs=payoffs, discounting=discounting,
        unanimity_required=unanimity_required, verbose=False,
        random_seed=0, initialization_mode="uniform", logger=None,
    )
    _set_canonical_weak_profile(solver, players, states, tiers, committee_idxs)
    n_free = len(free_approvals)
    for k, (prop, src, dst, appr) in enumerate(free_approvals):
        solver.r_acceptances[(prop, src, dst, appr)] = float(best_phys[k])
    var_idx = n_free
    for pi, si, widxs in pt_idx:
        prop = players[pi]; src = states[si]
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
    if timing_data is not None: timing_data["finalize_solver_obj"] += (time.perf_counter() - t_obj0)

    return {
        "strategy_df": strategy_df.copy(),
        "P": P_df.copy(),
        "V": V_df.copy(),
        "P_proposals": P_proposals.copy(),
        "P_approvals": dict(P_approvals),
        "verification_success": verified,
        "verification_message": message,
        "verification_detail": detail,
        "free_approvals": list(free_approvals),
        "proposal_rows": list(proposal_rows),
    }

def _build_P_direct(best_phys, canon_action, canon_pass, canon_probs, fa_idx, pt_idx, tiers, committee_idxs, players, states):
    n_players = len(players)
    n_states = len(states)
    action = canon_action.copy()
    pass_ = canon_pass.copy()
    probs = canon_probs.copy()
    n_free = len(fa_idx)
    
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

    for pi, si in set((fa[0], fa[2]) for fa in fa_idx).union(set((pt[0], pt[1]) for pt in pt_idx)):
        for ni in range(n_states):
            prob = 1.0
            for ai in committee_idxs[pi][si][ni]:
                prob *= action[pi, ai, si, ni]
            pass_[pi, si, ni] = prob
        
        # Proposal probs logic (only if not PT)
        if not any(pi == pt[0] and si == pt[1] for pt in pt_idx):
            tier_p = tiers[pi]
            best_t = int(tier_p[si])
            approved = [ni for ni in range(n_states) if pass_[pi, si, ni]]
            if approved:
                best_t = min(int(tier_p[ni]) for ni in approved)
                winners = [ni for ni in approved if int(tier_p[ni]) == best_t]
                probs[pi, si, :] = 0.0
                m = 1.0 / len(winners)
                for ni in winners: probs[pi, si, ni] = m
                
    protocol_arr = np.array([1.0/n_players for _ in range(n_players)])
    P = np.einsum('i,ijk,ijk->jk', protocol_arr, probs, pass_)
    row_sums = P.sum(axis=1)
    np.fill_diagonal(P, P.diagonal() + (1.0 - row_sums))
    return P, probs, pass_, action

    # Warm up the Numba JIT on first call (compiles once, then cached).
    _use_numba = _NUMBA_AVAILABLE and weak_orders  # strict-order path uses a different function
    # Pre-allocate tier buffer (avoid np.stack per iteration)
    _tiers_buf = np.empty((n_players, n_states), dtype=np.int8)

    if _use_numba:
        _build_arrays_weak_nb(_tiers_buf, comm_arr_nb, comm_size_nb, protocol_arr)
        _dummy_P = np.eye(n_states)
        _solve_V_nb(_dummy_P, payoff_array, discounting)
        _dummy_pp = np.zeros((n_players, n_states, n_states))
        _dummy_aa = np.zeros((n_players, n_players, n_states, n_states))
        _dummy_ap = np.ones((n_players, n_states, n_states))
        _dummy_V = np.zeros((n_states, n_players))
        _verify_fast_nb(_dummy_pp, _dummy_aa, _dummy_ap, _dummy_V, comm_arr_nb, comm_size_nb)
        # Warm up the Numba residual function for _solve_weak_equalities
        _dummy_raw = np.zeros(1, dtype=np.float64)
        _dummy_fa = np.zeros((1, 4), dtype=np.int8)
        _dummy_pt_pi = np.zeros(1, dtype=np.int8)
        _dummy_pt_si = np.zeros(1, dtype=np.int8)
        _dummy_pt_widxs = np.zeros((1, 1), dtype=np.int8)
        _dummy_pt_nwidxs = np.zeros(1, dtype=np.int8)
        _dummy_aff_pi = np.zeros(1, dtype=np.int8)
        _dummy_aff_ci = np.zeros(1, dtype=np.int8)
        _dummy_aff_is_pt = np.zeros(1, dtype=np.bool_)
        _residuals_nb_core(
            _dummy_raw, _dummy_pp, _dummy_aa, _dummy_ap,
            _dummy_fa, 1,
            _dummy_pt_pi, _dummy_pt_si, _dummy_pt_widxs, _dummy_pt_nwidxs, 0,
            _dummy_aff_pi, _dummy_aff_ci, _dummy_aff_is_pt, 0,
            comm_arr_nb, comm_size_nb, _tiers_buf,
            protocol_arr, payoff_array, discounting,
            n_players, n_states,
        )

    if perm_orders is None and n_workers <= 1:
        if ranking_order == "payoff":
            perm_orders = _payoff_ordering(payoff_array, states, players, order_arrays)
        elif ranking_order == "random":
            perm_orders = _random_ordering(n_orders, n_players, random_seed)
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
        for done, total, perm_tuple in _ranking_tuples(
            order_arrays,
            n_players,
            max_combinations,
            shuffle=shuffle,
            random_seed=random_seed,
            perm_orders=perm_orders,
        ):
            if n_workers > 1 and done % n_workers != worker_id:
                continue
            ranks = tuple(pos[perm_idx] for perm_idx in perm_tuple)
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
                if _use_numba:
                    # Fast path: Numba JIT — fill pre-allocated buffer (avoids np.stack per iter)
                    for _pi, _perm_idx in enumerate(perm_tuple):
                        _tiers_buf[_pi] = pos[_perm_idx]
                    proposal_probs, approval_action, approval_pass, P_array = _build_arrays_weak_nb(
                        _tiers_buf, comm_arr_nb, comm_size_nb, protocol_arr,
                    )
                    V_array = _solve_V_nb(P_array, payoff_array, discounting)
                else:
                    proposal_probs, approval_action, approval_pass, P_array = _build_induced_arrays_weak(
                        players=players,
                        tiers=ranks,
                        committee_idxs=committee_idxs,
                        protocol_arr=protocol_arr,
                    )
                    V_array = _solve_values_fast_array(P_array, payoff_array, discounting)
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
                if _use_numba:
                    # Fast path: Numba verifier returns just a bool.
                    # If it passes, great. If it fails, the Python verifier is only
                    # called when we need the detailed error message (i.e. never in the
                    # hot loop — failures are simply skipped).
                    verified = _verify_fast_nb(
                        proposal_probs, approval_action, approval_pass,
                        V_array, comm_arr_nb, comm_size_nb,
                    )
                    message = "All tests passed." if verified else ""
                    detail = None
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
                # Fast pre-check: count how many (approver, transition) pairs are tied
                # across all players' orderings. This gives a cheap lower bound on n_free
                # without calling the full _weak_tie_structure. If the lower bound already
                # exceeds max_vars, skip the expensive exact computation entirely.
                _skip = False
                if weak_equality_max_vars is not None:
                    _lb = 0
                    for _ai, _r in enumerate(ranks):
                        # Count transitions within the same tier for this player
                        # (each is a potential free approval in some committee)
                        _tiers_arr = _r
                        for _ci in range(n_states):
                            _tc = int(_tiers_arr[_ci])
                            for _ni in range(n_states):
                                if _ni != _ci and int(_tiers_arr[_ni]) == _tc:
                                    _lb += 1
                                    if _lb > weak_equality_max_vars:
                                        _skip = True
                                        break
                            if _skip:
                                break
                        if _skip:
                            break

                if not _skip:
                    # Compute exact tie structure and pass it in to avoid recomputation
                    # inside _solve_weak_equalities.
                    tie_struct = _weak_tie_structure(players, states, ranks, committee_idxs)
                    _n_free = len(tie_struct[0]) + sum(
                        len(w) - 1 for _, _, w in tie_struct[1]
                    )
                    if _n_free > 0 and (weak_equality_max_vars is None or _n_free <= weak_equality_max_vars):
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
                            _precomputed_tie_structure=tie_struct,
                            _precomputed_canon_arrays=(proposal_probs, approval_action, approval_pass),
                            _numba_comm_arr=comm_arr_nb if _use_numba else None,
                            _numba_comm_size=comm_size_nb if _use_numba else None,
                            _numba_tiers=_tiers_buf if _use_numba else None,
                        )
                if solved_payload is not None and solved_payload["verification_success"]:
                    verified = True
                    message = solved_payload["verification_message"]
                    detail = solved_payload["verification_detail"]
            tested += 1
            if progress_every > 0 and tested % progress_every == 0:
                if progress_queue is not None:
                    try:
                        progress_queue.put(progress_every)
                    except Exception:
                        pass
                else:
                    _print_progress(tested, total, start_time)
            if verified:
                verified_successes += 1
                success: dict[str, Any] = {
                    "perms": perm_tuple,
                    "rankings": tuple(order_arrays[perm_idx].copy() for perm_idx in perm_tuple),
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
                    success["source"] = "canonical"
                all_successes.append(success)
                if first_success is None:
                    first_success = success
                if stop_on_success:
                    break
    except KeyboardInterrupt:
        interrupted = True

    if progress_every > 0 and progress_queue is not None:
        try:
            rem = tested % progress_every
            if rem > 0:
                progress_queue.put(rem)
        except Exception:
            pass

    if tested and progress_queue is None:
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
    effectivity_rule: str = "heyen_lehtomaa_2021",
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
        return _run_verification(temp_path, effectivity_rule=effectivity_rule)
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


def _terminate_process_pool_workers(executor: Any) -> None:
    """Force worker processes to exit (private CPython API; used only after Ctrl+C)."""
    procs = getattr(executor, "_processes", None)
    if not procs:
        return
    for proc in list(procs.values()):
        if proc is None:
            continue
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    for proc in list(procs.values()):
        if proc is None:
            continue
        try:
            proc.join(timeout=2.0)
        except Exception:
            pass


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
    total_n = len(result["all_successes"])
    progress_iv = 25 if total_n > 400 else 10 if total_n > 80 else 1
    print(f"  output_dir: {output_dir.resolve()}", flush=True)
    metadata_base = _build_excel_metadata(result["config"], result["config"]["payoff_table"])
    manifest_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    written = 0
    skipped_dedup = 0

    def _write_progress(msg: str) -> None:
        print(f"\r\033[2K{msg}", end="", flush=True)

    def _flush_manifest() -> None:
        if not manifest_rows:
            return
        manifest_path = output_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    successes_list = result["all_successes"]

    def _process_one(idx: int, success: dict[str, Any]) -> None:
        nonlocal written, skipped_dedup
        if idx == 1 or idx % progress_iv == 0:
            _write_progress(
                f"  write_all: row {idx}/{total_n} (written={written}, skipped_dedup={skipped_dedup}) - materializing..."
            )
        reference = _materialize_success_result(result, success)
        if dedup_by == "transition":
            dedup_key = _serialize_transition_matrix(reference["P"], result["states"])
        elif dedup_by == "strategy":
            dedup_key = _serialize_strategy_df(reference["strategy_df"])
        else:
            dedup_key = ("ordinal", idx)
        if dedup_key in seen_keys:
            skipped_dedup += 1
            if idx % progress_iv == 0 or idx == total_n:
                _write_progress(
                    f"  write_all: row {idx}/{total_n} (written={written}, skipped_dedup={skipped_dedup})"
                )
            return
        seen_keys.add(dedup_key)

        perm_tuple = tuple(int(perm) for perm in success["perms"])
        suffix = f"{idx:04d}_" + "_".join(f"p{player_idx}{perm_idx}" for player_idx, perm_idx in enumerate(perm_tuple))
        output_path = output_dir / f"{payoff_path.stem}_{suffix}.xlsx"
        metadata = dict(metadata_base)
        metadata["ordinal_ranking_weak_orders"] = result["weak_orders"]
        metadata["ordinal_perms"] = ",".join(str(perm) for perm in perm_tuple)
        for player_idx, perm_idx in enumerate(perm_tuple):
            metadata[f"ordinal_perm_{player_idx}"] = perm_idx
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
        written += 1
        if written % progress_iv == 0 or written <= 3:
            _write_progress(
                f"  write_all: row {idx}/{total_n} wrote #{written} -> {_display_path(output_path)}"
            )
        row = {
            "index": idx,
            "output_file": _display_path(output_path),
            "ordinal_perms": ",".join(str(perm) for perm in perm_tuple),
        }
        for player_idx, perm_idx in enumerate(perm_tuple):
            row[f"perm_{player_idx}"] = perm_idx
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
        if idx <= 5:
            _write_progress(
                f"  write_all: finished row {idx}/{total_n} (written={written}, skipped_dedup={skipped_dedup})"
            )

    try:
        for idx, success in enumerate(successes_list, start=1):
            _process_one(idx, success)

        _flush_manifest()
        if total_n:
            print(
                f"\n  write_all done: {written} files, {skipped_dedup} skipped (dedup_by={dedup_by}), "
                f"manifest rows={len(manifest_rows)}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\n\n  write_all interrupted by user (Ctrl+C).", flush=True)
        try:
            _flush_manifest()
            if manifest_rows:
                print(
                    f"  Partial manifest saved ({len(manifest_rows)} rows): "
                    f"{_display_path(output_dir / 'manifest.csv')}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  Could not save partial manifest: {exc}", flush=True)
        raise
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


# Set by ProcessPoolExecutor initializer (parallel search). Queues/Events must not be
# passed inside submit() args on the "spawn" start method — only via initargs/inheritance.
_WORKER_STOP_EVENT: Any = None
_WORKER_PROGRESS_QUEUE: Any = None


def _worker_init(stop_event: Any = None, progress_queue: Any = None) -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _WORKER_STOP_EVENT, _WORKER_PROGRESS_QUEUE
    _WORKER_STOP_EVENT = stop_event
    _WORKER_PROGRESS_QUEUE = progress_queue


def _worker_search_batch(args: tuple) -> dict[str, Any]:
    """Search a specific list of ranking perms."""
    (
        config, perm_tuples, use_reference_verifier,
        weak_orders_flag, weak_exact_reduced,
        weak_equality_solve, weak_equality_max_vars,
        progress_every, order_arrays,
    ) = args
    progress_queue = _WORKER_PROGRESS_QUEUE

    # Internal setup (same as _search_payoff_table but without the ranking_tuples loop)
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
    if order_arrays is None:
        if weak_orders_flag:
            order_arrays = _generate_weak_orders(n_states)
        else:
            order_arrays = np.array(list(itertools.permutations(range(n_states))), dtype=np.int8)

    n_orders = order_arrays.shape[0]
    pos = np.empty((n_orders, n_states), dtype=np.int8)
    for order_idx in range(n_orders):
        if weak_orders_flag:
            pos[order_idx] = order_arrays[order_idx]
        else:
            for rank, state_idx in enumerate(order_arrays[order_idx]):
                pos[order_idx, state_idx] = rank

    committee_idxs: list[list[list[tuple[int, ...]]]] = []
    player_idx = {player: idx for idx, player in enumerate(players)}
    state_idx = {state: idx for idx, state in enumerate(states)}
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

    _max_comm = max(len(committee_idxs[pi][ci][ni]) for pi in range(n_players) for ci in range(n_states) for ni in range(n_states))
    comm_arr_nb = np.full((n_players, n_states, n_states, max(_max_comm, 1)), -1, dtype=np.int8)
    comm_size_nb = np.zeros((n_players, n_states, n_states), dtype=np.int8)
    for _pi in range(n_players):
        for _ci in range(n_states):
            for _ni in range(n_states):
                for _k, _ai in enumerate(committee_idxs[_pi][_ci][_ni]):
                    comm_arr_nb[_pi, _ci, _ni, _k] = _ai
                comm_size_nb[_pi, _ci, _ni] = len(committee_idxs[_pi][_ci][_ni])

    _use_numba = _NUMBA_AVAILABLE and weak_orders_flag
    _tiers_buf = np.empty((n_players, n_states), dtype=np.int8)

    tested = 0
    verified_successes = 0
    all_successes: list[dict[str, Any]] = []
    weak_exact_zero_params = 0
    weak_exact_deterministic = 0
    weak_exact_nontrivial = 0

    solver = None
    if use_reference_verifier:
        solver = EquilibriumSolver(players=players, states=states, effectivity=effectivity, protocol=protocol, payoffs=payoffs, discounting=discounting, unanimity_required=unanimity_required, verbose=False, random_seed=0, initialization_mode="uniform", logger=None)

    import time as _time
    t_start = _time.perf_counter()
    t_numba = 0.0
    t_tie_struct = 0.0
    t_solver = 0.0
    t_solver_root = 0.0
    t_solver_finalize = 0.0
    t_solver_setup = 0.0
    t_solver_check = 0.0
    t_solver_setup_copy = 0.0
    t_solver_setup_indices = 0.0
    t_solver_setup_numba = 0.0
    t_solver_root_v_solve = 0.0
    t_solver_root_p_agg = 0.0
    t_solver_root_other = 0.0
    t_solver_root_mapping = 0.0
    t_solver_root_residuals = 0.0
    t_solver_setup_guesses = 0.0
    t_solver_nb_newton = 0.0

    t_solver_finalize_rebuild = 0.0
    t_solver_finalize_verify = 0.0
    t_solver_finalize_solver_obj = 0.0

    total_solver_calls = 0
    total_skipped_max_vars = 0
    n_free_histogram: dict[int, int] = {}
    # Exit stats: shape (7, 7) — 7 guesses × 7 outcome codes
    # outcomes: 0=nb_newton_hit, 1=success(scipy), 2=converged+bad_resid,
    #           3=maxfev, 4=xtol, 5=bad_progress, 6=exception
    exit_stats_counts = np.zeros((7, 7), dtype=np.int64)

    # Per-progress-interval counters (reset after each send)
    iv_numba_t = 0.0
    iv_tie_struct_t = 0.0
    iv_solver_t = 0.0
    iv_solver_calls = 0
    iv_lb_skipped = 0
    iv_hits = 0

    for perm_tuple in perm_tuples:
        if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
            break
        # 1. Canonical check (Numba accelerated)
        t0 = _time.perf_counter()
        ranks = tuple(pos[perm_idx] for perm_idx in perm_tuple)
        if weak_orders_flag and weak_exact_reduced:
            n_value_params = _weak_value_param_count(ranks)
            n_strategy_free = _weak_free_var_count(players, states, ranks, committee_idxs)
            if n_value_params == 0: weak_exact_zero_params += 1
            if n_strategy_free == 0: weak_exact_deterministic += 1
            else: weak_exact_nontrivial += 1

        if weak_orders_flag:
            if _use_numba:
                for _pi, _perm_idx in enumerate(perm_tuple): _tiers_buf[_pi] = pos[_perm_idx]
                proposal_probs, approval_action, approval_pass, P_array = _build_arrays_weak_nb(_tiers_buf, comm_arr_nb, comm_size_nb, protocol_arr)
                V_array = _solve_V_nb(P_array, payoff_array, discounting)
            else:
                proposal_probs, approval_action, approval_pass, P_array = _build_induced_arrays_weak(players=players, tiers=ranks, committee_idxs=committee_idxs, protocol_arr=protocol_arr)
                V_array = _solve_values_fast_array(P_array, payoff_array, discounting)
            proposal_choice = None
        else:
            proposal_choice, approval_action, approval_pass, P_array = _build_induced_arrays(players=players, ranks=ranks, committee_idxs=committee_idxs, protocol_arr=protocol_arr)
            proposal_probs = None
            V_array = _solve_values_fast_array(P_array, payoff_array, discounting)

        if use_reference_verifier:
            assert solver is not None
            if weak_orders_flag: _induce_profile_from_weak_orders(solver, players, states, ranks, committee_idxs)
            else: _induce_profile_from_rankings(solver, players, states, ranks, committee_idxs)
            P, P_proposals, P_approvals = solver._compute_transition_probabilities_fast()
            V = pd.DataFrame(V_array, index=states, columns=players)
            strategy_df = solver._create_strategy_dataframe()
            verified, message, detail = verify_equilibrium_detailed({"players": players, "states": states, "state_names": states, "effectivity": effectivity, "P": P, "P_proposals": P_proposals, "P_approvals": P_approvals, "V": V, "strategy_df": strategy_df})
        else:
            if _use_numba:
                verified = _verify_fast_nb(proposal_probs, approval_action, approval_pass, V_array, comm_arr_nb, comm_size_nb)
            else:
                verified, _, _ = _verify_equilibrium_fast(players=players, states=states, effectivity=effectivity, P_proposals=None, P_approvals=None, V_df=None, proposal_choice=proposal_choice, proposal_probs=proposal_probs, approval_action=approval_action, approval_pass=approval_pass, V_array=V_array, committee_idxs=committee_idxs)
        t1 = _time.perf_counter()
        t_numba += (t1 - t0)
        iv_numba_t += (t1 - t0)

        # 2. Numerical refinement (if needed)
        solved_payload = None
        if (not verified) and weak_orders_flag and weak_equality_solve:
            if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
                break
            # Cheap _lb pre-filter: count same-tier pairs; if already > max_vars, skip
            _skip = False
            if weak_equality_max_vars is not None:
                _lb = 0
                for _r in ranks:
                    for _ci in range(n_states):
                        _tc = int(_r[_ci])
                        for _ni in range(n_states):
                            if _ni != _ci and int(_r[_ni]) == _tc:
                                _lb += 1
                                if _lb > weak_equality_max_vars:
                                    _skip = True
                                    break
                        if _skip:
                            break
                    if _skip:
                        break
            if _skip:
                iv_lb_skipped += 1
            else:
                t_ts0 = _time.perf_counter()
                tie_struct = _weak_tie_structure(players, states, ranks, committee_idxs)
                n_free = len(tie_struct[0]) + sum(len(w) - 1 for _, _, w in tie_struct[1])
                iv_tie_struct_t += (_time.perf_counter() - t_ts0)
                t_tie_struct += (_time.perf_counter() - t_ts0)
                
                n_free_histogram[n_free] = n_free_histogram.get(n_free, 0) + 1

                if n_free > 0 and (weak_equality_max_vars is None or n_free <= weak_equality_max_vars):
                    iv_solver_calls += 1
                    total_solver_calls += 1
                    
                    _call_exit_stats = np.zeros((7, 7), dtype=np.int64)
                    solver_timing = {
                        "solver_root": 0.0, "solver_finalize": 0.0, "solver_setup": 0.0, "solver_check": 0.0,
                        "solver_setup_copy": 0.0, "solver_setup_indices": 0.0, "solver_setup_numba": 0.0,
                        "solver_root_v_solve": 0.0, "solver_root_p_agg": 0.0, "solver_root_other": 0.0,
                        "solver_root_mapping": 0.0, "solver_root_residuals": 0.0, "solver_setup_guesses": 0.0,
                        "finalize_rebuild": 0.0, "finalize_verify": 0.0, "finalize_solver_obj": 0.0,
                        "solver_nb_newton": 0.0,
                    }
                    t_sv0 = _time.perf_counter()
                    solved_payload = _solve_weak_equalities(
                        players=players, states=states, effectivity=effectivity,
                        protocol=protocol, payoffs=payoffs, discounting=discounting,
                        unanimity_required=unanimity_required, tiers=ranks,
                        committee_idxs=committee_idxs, max_vars=weak_equality_max_vars,
                        _precomputed_tie_structure=tie_struct,
                        _precomputed_canon_arrays=(proposal_probs, approval_action, approval_pass),
                        _numba_comm_arr=comm_arr_nb if _use_numba else None,
                        _numba_comm_size=comm_size_nb if _use_numba else None,
                        _numba_tiers=_tiers_buf if _use_numba else None,
                        timing_data=solver_timing,
                        player_idx=player_idx,
                        state_idx=state_idx,
                        _precomputed_payoff_array=payoff_array,
                        _precomputed_protocol_arr=protocol_arr,
                        _exit_stats_counts=_call_exit_stats,
                    )
                    exit_stats_counts += _call_exit_stats
                    _d_sv = _time.perf_counter() - t_sv0
                    iv_solver_t += _d_sv
                    t_solver += _d_sv
                    t_solver_root += solver_timing["solver_root"]
                    t_solver_finalize += solver_timing["solver_finalize"]
                    t_solver_setup += solver_timing["solver_setup"]
                    t_solver_check += solver_timing["solver_check"]
                    
                    t_solver_setup_copy += solver_timing["solver_setup_copy"]
                    t_solver_setup_indices += solver_timing["solver_setup_indices"]
                    t_solver_setup_numba += solver_timing["solver_setup_numba"]
                    t_solver_root_v_solve += solver_timing["solver_root_v_solve"]
                    t_solver_root_p_agg += solver_timing["solver_root_p_agg"]
                    t_solver_root_other += solver_timing["solver_root_other"]
                    
                    t_solver_root_mapping += solver_timing["solver_root_mapping"]
                    t_solver_root_residuals += solver_timing["solver_root_residuals"]
                    t_solver_setup_guesses += solver_timing["solver_setup_guesses"]
                    
                    t_solver_finalize_rebuild += solver_timing["finalize_rebuild"]
                    t_solver_finalize_verify += solver_timing["finalize_verify"]
                    t_solver_finalize_solver_obj += solver_timing["finalize_solver_obj"]
                    t_solver_nb_newton += solver_timing["solver_nb_newton"]
                elif n_free > 0:
                    iv_lb_skipped += 1
                    total_skipped_max_vars += 1

        tested += 1
        if verified or (solved_payload is not None and solved_payload["verification_success"]):
            verified_successes += 1
            iv_hits += 1
            success = {"perms": perm_tuple, "rankings": tuple(order_arrays[perm_idx].copy() for perm_idx in perm_tuple)}
            if solved_payload is not None and solved_payload["verification_success"]:
                success.update({"source": "weak_equality_solve", "P": solved_payload["P"], "V": solved_payload["V"], "strategy_df": solved_payload["strategy_df"], "P_proposals": solved_payload["P_proposals"], "P_approvals": solved_payload["P_approvals"]})
            else:
                success["source"] = "canonical"
            all_successes.append(success)

        if progress_every > 0 and tested % progress_every == 0:
            if progress_queue is not None:
                try:
                    progress_queue.put((
                        progress_every,
                        iv_numba_t, iv_tie_struct_t, iv_solver_t,
                        iv_solver_calls, iv_lb_skipped, iv_hits,
                    ))
                except Exception:
                    pass
            iv_numba_t = 0.0
            iv_tie_struct_t = 0.0
            iv_solver_t = 0.0
            iv_solver_calls = 0
            iv_lb_skipped = 0
            iv_hits = 0

    if progress_every > 0 and progress_queue is not None:
        rem = tested % progress_every
        if rem > 0:
            try:
                progress_queue.put((
                    rem,
                    iv_numba_t, iv_tie_struct_t, iv_solver_t,
                    iv_solver_calls, iv_lb_skipped, iv_hits,
                ))
            except Exception:
                pass

    t_total = _time.perf_counter() - t_start
    t_overhead = t_total - t_numba - t_tie_struct - t_solver

    return {
        "tested": tested,
        "verified_successes": verified_successes,
        "all_successes": all_successes,
        "first_success": all_successes[0] if all_successes else None,
        "weak_exact_zero_params": weak_exact_zero_params,
        "weak_exact_deterministic": weak_exact_deterministic,
        "weak_exact_nontrivial": weak_exact_nontrivial,
        "players": players,
        "states": states,
        "effectivity": effectivity,
        "protocol": protocol,
        "payoffs": payoffs,
        "discounting": discounting,
        "unanimity_required": unanimity_required,
        "config": config,
        "order_arrays": order_arrays,
        "weak_orders": weak_orders_flag,
        "weak_exact_reduced": weak_exact_reduced,
        "weak_equality_solve": weak_equality_solve,
        "t_numba": t_numba,
        "t_tie_struct": t_tie_struct,
        "t_solver": t_solver,
        "t_solver_root": t_solver_root,
        "t_solver_finalize": t_solver_finalize,
        "t_solver_setup": t_solver_setup,
        "t_solver_check": t_solver_check,
        "t_solver_setup_copy": t_solver_setup_copy,
        "t_solver_setup_indices": t_solver_setup_indices,
        "t_solver_setup_numba": t_solver_setup_numba,
        "t_solver_root_v_solve": t_solver_root_v_solve,
        "t_solver_root_p_agg": t_solver_root_p_agg,
        "t_solver_root_other": t_solver_root_other,
        "t_solver_root_mapping": t_solver_root_mapping,
        "t_solver_root_residuals": t_solver_root_residuals,
        "t_solver_setup_guesses": t_solver_setup_guesses,
        "t_solver_finalize_rebuild": t_solver_finalize_rebuild,
        "t_solver_finalize_verify": t_solver_finalize_verify,
        "t_solver_finalize_solver_obj": t_solver_finalize_solver_obj,
        "t_solver_nb_newton": t_solver_nb_newton,
        "t_overhead": t_overhead,
        "total_solver_calls": total_solver_calls,
        "total_skipped_max_vars": total_skipped_max_vars,
        "n_free_histogram": n_free_histogram,
        "exit_stats_counts": exit_stats_counts,
        "elapsed": t_total,
    }


def _merge_worker_results(worker_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge results from multiple parallel workers into a single result dict."""
    base = worker_results[0]
    merged: dict[str, Any] = {
        k: base[k]
        for k in ("players", "states", "effectivity", "protocol", "payoffs",
                  "discounting", "unanimity_required", "config",
                  "order_arrays", "weak_orders", "weak_exact_reduced",
                  "weak_equality_solve")
    }
    merged["total"] = sum(r.get("total", 0) for r in worker_results)
    merged["interrupted"] = any(r.get("interrupted", False) for r in worker_results)
    merged["tested"] = sum(r["tested"] for r in worker_results)
    merged["elapsed"] = max((r.get("elapsed", 0.0) for r in worker_results), default=0.0)
    merged["rate"] = merged["tested"] / merged["elapsed"] if merged["elapsed"] > 0 else float("nan")
    merged["verified_successes"] = sum(r["verified_successes"] for r in worker_results)
    merged["all_successes"] = [s for r in worker_results for s in r["all_successes"]]
    merged["first_success"] = next(
        (r.get("first_success") for r in worker_results if r.get("first_success") is not None), None
    )
    merged["weak_exact_zero_params"] = sum(r.get("weak_exact_zero_params", 0) for r in worker_results)
    merged["weak_exact_deterministic"] = sum(r.get("weak_exact_deterministic", 0) for r in worker_results)
    merged["weak_exact_nontrivial"] = sum(r.get("weak_exact_nontrivial", 0) for r in worker_results)
    merged["t_numba"] = sum(r.get("t_numba", 0.0) for r in worker_results)
    merged["t_tie_struct"] = sum(r.get("t_tie_struct", 0.0) for r in worker_results)
    merged["t_solver"] = sum(r.get("t_solver", 0.0) for r in worker_results)
    merged["t_solver_root"] = sum(r.get("t_solver_root", 0.0) for r in worker_results)
    merged["t_solver_finalize"] = sum(r.get("t_solver_finalize", 0.0) for r in worker_results)
    merged["t_solver_setup"] = sum(r.get("t_solver_setup", 0.0) for r in worker_results)
    merged["t_solver_check"] = sum(r.get("t_solver_check", 0.0) for r in worker_results)
    merged["t_solver_setup_copy"] = sum(r.get("t_solver_setup_copy", 0.0) for r in worker_results)
    merged["t_solver_setup_indices"] = sum(r.get("t_solver_setup_indices", 0.0) for r in worker_results)
    merged["t_solver_setup_numba"] = sum(r.get("t_solver_setup_numba", 0.0) for r in worker_results)
    merged["t_solver_root_v_solve"] = sum(r.get("t_solver_root_v_solve", 0.0) for r in worker_results)
    merged["t_solver_root_p_agg"] = sum(r.get("t_solver_root_p_agg", 0.0) for r in worker_results)
    merged["t_solver_root_other"] = sum(r.get("t_solver_root_other", 0.0) for r in worker_results)
    merged["t_solver_root_mapping"] = sum(r.get("t_solver_root_mapping", 0.0) for r in worker_results)
    merged["t_solver_root_residuals"] = sum(r.get("t_solver_root_residuals", 0.0) for r in worker_results)
    merged["t_solver_setup_guesses"] = sum(r.get("t_solver_setup_guesses", 0.0) for r in worker_results)
    merged["t_solver_finalize_rebuild"] = sum(r.get("t_solver_finalize_rebuild", 0.0) for r in worker_results)
    merged["t_solver_finalize_verify"] = sum(r.get("t_solver_finalize_verify", 0.0) for r in worker_results)
    merged["t_solver_finalize_solver_obj"] = sum(r.get("t_solver_finalize_solver_obj", 0.0) for r in worker_results)
    merged["t_solver_nb_newton"] = sum(r.get("t_solver_nb_newton", 0.0) for r in worker_results)
    merged["t_overhead"] = sum(r.get("t_overhead", 0.0) for r in worker_results)
    merged["total_solver_calls"] = sum(r.get("total_solver_calls", 0) for r in worker_results)
    merged["total_skipped_max_vars"] = sum(r.get("total_skipped_max_vars", 0) for r in worker_results)
    
    merged_hist: dict[int, int] = {}
    for r in worker_results:
        hist = r.get("n_free_histogram", {})
        for k, v in hist.items():
            merged_hist[k] = merged_hist.get(k, 0) + v
    merged["n_free_histogram"] = merged_hist

    merged_exit = np.zeros((7, 7), dtype=np.int64)
    for r in worker_results:
        ec = r.get("exit_stats_counts")
        if ec is not None:
            merged_exit += np.asarray(ec, dtype=np.int64)
    merged["exit_stats_counts"] = merged_exit

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate strict ranking triples against a payoff table and verify them."
    )
    parser.add_argument("file", help="Payoff table path or basename under payoff_tables/")
    parser.add_argument("--scenario", type=str, default=None, help="Optional scenario override for payoff-table search")
    parser.add_argument(
        "--allow-non-canonical-states",
        action="store_true",
        help="Allow non-canonical state names from reduced payoff tables.",
    )
    parser.add_argument(
        "--effectivity-rule",
        type=str,
        default=None,
        choices=("heyen_lehtomaa_2021", "unanimous_consent", "deployer_exit", "free_exit"),
        help="Effectivity rule used to generate approval committees.",
    )
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
        choices=("lexicographic", "payoff", "random"),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). Use --workers 0 to auto-detect CPU count.",
    )
    args = parser.parse_args()

    if getattr(args, "weak_equality_solve", False) and not _NUMBA_AVAILABLE:
        print("ERROR: --weak-equality-solve requires Numba. Install it with: pip install numba", file=sys.stderr)
        sys.exit(1)

    payoff_path = _resolve_payoff_file(args.file)
    if args.scenario:
        config = _build_payoff_config(
            args.scenario,
            str(payoff_path),
            allow_non_canonical_states=args.allow_non_canonical_states,
            effectivity_rule=args.effectivity_rule,
        )
        config_source = f"scenario:{args.scenario}"
    else:
        config = _build_inferred_payoff_config(
            payoff_path,
            allow_non_canonical_states=args.allow_non_canonical_states,
            effectivity_rule=args.effectivity_rule,
        )
        config_source = "inferred"
    players = config["players"]
    print("Ordinal Ranking Verification Search")
    print("-" * 80)
    print(f"file: {payoff_path.relative_to(REPO_ROOT)}")
    print(f"config_source: {config_source}")
    print(f"players: {players}")
    print(f"allow_non_canonical_states: {args.allow_non_canonical_states}")
    if args.effectivity_rule is not None:
        print(f"effectivity_rule: {args.effectivity_rule}")
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
    import multiprocessing as _mp
    n_workers = args.workers if args.workers > 0 else _mp.cpu_count()
    if n_workers > 1:
        print(f"workers: {n_workers}")

    # Pre-compute rankings and their sorted order
    if args.weak_orders:
        order_arrays = _generate_weak_orders(state_count)
    else:
        order_arrays = np.array(list(itertools.permutations(range(state_count))), dtype=np.int8)

    payoff_array_preview = setup_preview["payoffs"].loc[setup_preview["state_names"], setup_preview["players"]].to_numpy(dtype=np.float64)
    perm_orders = None
    if args.ranking_order == "payoff":
        perm_orders = _payoff_ordering(payoff_array_preview, setup_preview["state_names"], players, order_arrays)
    elif args.ranking_order == "random":
        perm_orders = _random_ordering(len(order_arrays), len(players), args.random_seed)

    # When ranking_order="random", force flat-index shuffle to break the block
    # structure of itertools.product (otherwise player 0's ranking is fixed for
    # n_perms^(n_players-1) consecutive combinations, causing large speed swings).
    # Set False to opt out.
    _RANDOM_ORDER_IMPLIES_SHUFFLE = True
    _effective_shuffle = args.shuffle or (args.ranking_order == "random" and _RANDOM_ORDER_IMPLIES_SHUFFLE)
    print(f"shuffle: {_effective_shuffle}")
    print()

    if n_workers <= 1:
        result = _search_payoff_table(
            config=config,
            max_combinations=args.max_combinations,
            progress_every=args.progress_every,
            stop_on_success=args.stop_on_success,
            use_reference_verifier=args.use_reference_verifier,
            shuffle=_effective_shuffle,
            random_seed=args.random_seed,
            ranking_order=args.ranking_order,
            weak_orders=args.weak_orders,
            weak_exact_reduced=args.weak_exact_reduced,
            weak_equality_solve=args.weak_equality_solve,
            weak_equality_max_vars=args.weak_equality_max_vars,
            order_arrays=order_arrays,
            perm_orders=perm_orders,
        )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import threading
        import queue

        # Plain Queue avoids multiprocessing.Manager (its shutdown often hangs waiting on the server).
        progress_queue = _mp.Queue()
        stop_event = _mp.Event()

        total_to_test = total_orders ** len(players)
        if args.max_combinations is not None:
            total_to_test = min(total_to_test, args.max_combinations)

        def progress_listener(q, total_val):
            done = 0
            start_time = time.perf_counter()
            _window_done = 0
            _window_start = time.perf_counter()
            _recent_rate: float | None = None
            _w_numba_t = 0.0
            _w_ts_t = 0.0
            _w_solver_t = 0.0
            _w_solver_calls = 0
            _w_lb_skipped = 0
            _w_hits = 0
            _total_hits = 0
            _breakdown: str = ""
            _PRINT_INTERVAL = 0.1
            _last_print = 0.0

            def _accumulate_msg(msg: object) -> bool:
                """Update counters from one queue item. Returns True if listener should stop (DONE)."""
                nonlocal done, _window_done, _total_hits
                nonlocal _w_numba_t, _w_ts_t, _w_solver_t, _w_solver_calls, _w_lb_skipped, _w_hits
                if msg == "DONE":
                    return True
                if isinstance(msg, tuple) and len(msg) == 7:
                    inc, iv_numba, iv_ts, iv_sv, iv_calls, iv_skip, iv_hits = msg
                elif isinstance(msg, tuple) and len(msg) == 5:
                    inc, iv_numba, iv_sv, iv_calls, iv_skip = msg
                    iv_ts = 0.0
                    iv_hits = 0
                else:
                    try:
                        inc = int(msg)  # noqa: TRY004 — int from workers or legacy single-value puts
                    except (TypeError, ValueError):
                        print(f"\nprogress_listener: skip bad message: {type(msg).__name__}", file=sys.stderr, flush=True)
                        return False
                    iv_numba = iv_ts = iv_sv = 0.0
                    iv_calls = iv_skip = iv_hits = 0
                inc_i = int(inc)
                done += inc_i
                _window_done += inc_i
                _w_numba_t += float(iv_numba)
                _w_ts_t += float(iv_ts)
                _w_solver_t += float(iv_sv)
                _w_solver_calls += int(iv_calls)
                _w_lb_skipped += int(iv_skip)
                _w_hits += int(iv_hits)
                _total_hits += int(iv_hits)
                return False

            def _maybe_print(*, force: bool) -> None:
                nonlocal _last_print, _recent_rate, _window_done, _window_start, _breakdown
                nonlocal _w_numba_t, _w_ts_t, _w_solver_t, _w_solver_calls, _w_lb_skipped, _w_hits
                now = time.perf_counter()
                if not force and (now - _last_print) < _PRINT_INTERVAL:
                    return

                # Update recent stats every ~1.0 second or if forced (DONE)
                elapsed = now - _window_start
                if _window_done > 0 and (elapsed >= 1.0 or force):
                    _recent_rate = _window_done / max(1e-9, elapsed)
                    _k = max(1e-3, _window_done / 1000)
                    _n_calls = _w_solver_calls
                    _ms_ts = (_w_ts_t / _n_calls * 1000) if _n_calls > 0 else 0.0
                    _ms_sv = (_w_solver_t / _n_calls * 1000) if _n_calls > 0 else 0.0
                    _breakdown = (
                        f"  [ts:{_ms_ts:.2f}ms sv:{_ms_sv:.2f}ms"
                        f" c/k:{_n_calls/_k:.0f}"
                        f" sk/k:{_w_lb_skipped/_k:.0f}"
                        f" h/k:{_w_hits/_k:.1f}"
                        f" hits:{_total_hits}]"
                    )
                    _window_done = 0
                    _window_start = now
                    _w_numba_t = _w_ts_t = _w_solver_t = 0.0
                    _w_solver_calls = _w_lb_skipped = _w_hits = 0

                _print_progress(done, total_val, start_time, recent_rate=_recent_rate, breakdown=_breakdown)
                _last_print = now

            while True:
                now = time.perf_counter()
                time_until_print = _PRINT_INTERVAL - (now - _last_print)
                # Short wait when due for a draw so we can batch several queue items into one update
                timeout = min(0.1, time_until_print) if time_until_print > 0 else 0.05
                try:
                    msg = q.get(timeout=timeout)
                except queue.Empty:
                    # No worker progress yet (e.g. Numba JIT): avoid redrawing 0% every 100ms
                    if done > 0 or _last_print == 0.0:
                        _maybe_print(force=False)
                    continue
                if _accumulate_msg(msg):
                    break
                while True:
                    try:
                        msg2 = q.get_nowait()
                    except queue.Empty:
                        break
                    if _accumulate_msg(msg2):
                        _maybe_print(force=True)
                        print()
                        return
                _maybe_print(force=False)
                # Avoid busy-spinning when the queue refills faster than the print throttle
                if time.perf_counter() - _last_print < _PRINT_INTERVAL:
                    time.sleep(max(0.0, _last_print + _PRINT_INTERVAL - time.perf_counter()))
            _maybe_print(force=True)
            print()

        listener = threading.Thread(target=progress_listener, args=(progress_queue, total_to_test))
        listener.daemon = True
        listener.start()

        all_perms_gen = _ranking_tuples(
            order_arrays, len(players), args.max_combinations,
            shuffle=_effective_shuffle, random_seed=args.random_seed, perm_orders=perm_orders
        )

        chunk_size = 2000
        def chunked_worker_args():
            batch = []
            for _, _, pt in all_perms_gen:
                batch.append(pt)
                if len(batch) >= chunk_size:
                    yield (config, batch, args.use_reference_verifier, args.weak_orders, args.weak_exact_reduced, args.weak_equality_solve, args.weak_equality_max_vars, args.progress_every, order_arrays)
                    batch = []
            if batch:
                yield (config, batch, args.use_reference_verifier, args.weak_orders, args.weak_exact_reduced, args.weak_equality_solve, args.weak_equality_max_vars, args.progress_every, order_arrays)

        import time as _time
        _par_start = _time.perf_counter()
        worker_results = []
        interrupted = False
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(stop_event, progress_queue),
        )
        active_futures = set()
        arg_gen = chunked_worker_args()
        max_pending = n_workers * 2

        try:
            # Initial fill
            for _ in range(max_pending):
                try:
                    active_futures.add(executor.submit(_worker_search_batch, next(arg_gen)))
                except StopIteration:
                    break

            while active_futures:
                done_fs = [f for f in active_futures if f.done()]
                for f in done_fs:
                    active_futures.remove(f)
                    try:
                        worker_results.append(f.result())
                    except Exception as e:
                        print(f"\nWorker error: {e}")
                    try:
                        active_futures.add(executor.submit(_worker_search_batch, next(arg_gen)))
                    except StopIteration:
                        pass
                if not done_fs:
                    _time.sleep(0.1)
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            import signal

            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\nInterrupted by user. Gathering partial results...")
        finally:
            _par_elapsed = _time.perf_counter() - _par_start
            stop_event.set()
            for f in list(active_futures):
                if f.done():
                    try:
                        worker_results.append(f.result())
                    except Exception as e:
                        print(f"\nWorker error: {e}")
            # Signal progress listener to stop
            try:
                progress_queue.put("DONE")
            except Exception:
                pass
            listener.join(timeout=3.0)
            if interrupted:
                executor.shutdown(wait=False, cancel_futures=True)
                _terminate_process_pool_workers(executor)
            else:
                executor.shutdown(wait=True, cancel_futures=False)
            # Allow Ctrl+C again during write_all / CLI output (except handler used SIG_IGN).
            import signal

            signal.signal(signal.SIGINT, signal.default_int_handler)

        if not worker_results:
            print("No results collected.")
            return

        result = _merge_worker_results(worker_results)
        result["elapsed"] = _par_elapsed
        result["total"] = total_to_test
        if interrupted:
            result["interrupted"] = True
        
        # Recalculate rate based on actual wall time and tested count
        if result["elapsed"] > 0:
            result["rate"] = result["tested"] / result["elapsed"]
        else:
            result["rate"] = float("nan")
    states = result["states"]
    order_arrays = result["order_arrays"]
    print("Summary")
    print("-" * 80)
    print(f"tested_combinations:  {result['tested']:>12,d}")
    print(f"wall_time:            {result['elapsed']:>12.2f}s")
    print(f"rate:                 {result['rate']:>12.0f}/s")
    print(f"  - numba_time:       {result.get('t_numba', 0.0) / n_workers:>12.2f}s (avg per worker)")
    print(f"  - tie_struct_time:  {result.get('t_tie_struct', 0.0) / n_workers:>12.2f}s (avg per worker)")
    t_solver = result.get('t_solver', 0.0) / n_workers
    print(f"  - solver_time:      {t_solver:>12.2f}s (avg per worker)")
    if t_solver > 0:
        def _fmt_pct(val):
            return f" ({100.0 * val / t_solver:5.1f}%)" if t_solver > 0 else ""

        print(f"    - nb_newton:      {result.get('t_solver_nb_newton', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_nb_newton', 0.0) / n_workers)}")
        print(f"    - root_search:    {result.get('t_solver_root', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_root', 0.0) / n_workers)}")
        if result.get("t_solver_root_v_solve", 0.0) > 0:
            print(f"      - V_solve:      {result.get('t_solver_root_v_solve', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_root_v_solve', 0.0) / n_workers)}")
            print(f"      - P_aggregate:  {result.get('t_solver_root_p_agg', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_root_p_agg', 0.0) / n_workers)}")
            print(f"      - mapping:      {result.get('t_solver_root_mapping', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_root_mapping', 0.0) / n_workers)}")
            print(f"      - residuals:    {result.get('t_solver_root_residuals', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_root_residuals', 0.0) / n_workers)}")
        print(f"    - finalize:       {result.get('t_solver_finalize', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_finalize', 0.0) / n_workers)}")
        if result.get("t_solver_finalize_rebuild", 0.0) > 0:
            print(f"      - rebuild:      {result.get('t_solver_finalize_rebuild', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_finalize_rebuild', 0.0) / n_workers)}")
            print(f"      - verify:       {result.get('t_solver_finalize_verify', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_finalize_verify', 0.0) / n_workers)}")
            print(f"      - solver_obj:   {result.get('t_solver_finalize_solver_obj', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_finalize_solver_obj', 0.0) / n_workers)}")
        print(f"    - check:          {result.get('t_solver_check', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_check', 0.0) / n_workers)}")
        print(f"    - setup/other:    {result.get('t_solver_setup', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_setup', 0.0) / n_workers)}")
        if result.get("t_solver_setup_copy", 0.0) > 0:
            print(f"      - copy_arrays:  {result.get('t_solver_setup_copy', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_setup_copy', 0.0) / n_workers)}")
            print(f"      - find_indices: {result.get('t_solver_setup_indices', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_setup_indices', 0.0) / n_workers)}")
            print(f"      - build_numba:  {result.get('t_solver_setup_numba', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_setup_numba', 0.0) / n_workers)}")
            print(f"      - guesses:      {result.get('t_solver_setup_guesses', 0.0) / n_workers:>12.2f}s{_fmt_pct(result.get('t_solver_setup_guesses', 0.0) / n_workers)}")
    print(f"  - python_overhead:  {result.get('t_overhead', 0.0) / n_workers:>12.2f}s (avg per worker)")
    print(f"interrupted:          {str(result['interrupted']):>12s}")
    print(f"verified_successes:   {result['verified_successes']:>12,d}")
    print(f"stored_successes:     {len(result['all_successes']):>12,d}")
    
    if args.weak_orders and args.weak_equality_solve:
        print(f"total_solver_calls:   {result.get('total_solver_calls', 0):>12,d}")
        print(f"skipped_max_vars:     {result.get('total_skipped_max_vars', 0):>12,d}")

        hist = result.get("n_free_histogram", {})
        if hist:
            print("\nFree Variables Distribution (n_free: count)")
            # Sort by n_free
            for n_free in sorted(hist.keys()):
                count = hist[n_free]
                skip_str = ""
                if args.weak_equality_max_vars is not None and n_free > args.weak_equality_max_vars:
                    skip_str = " (SKIPPED)"
                print(f"  {n_free:>2d} vars: {count:>12,d}{skip_str}")

        ec = result.get("exit_stats_counts")
        if ec is not None and np.any(ec):
            # outcomes: 0=nb_newton_hit, 1=success(scipy), 2=converged+bad_resid,
            #           3=maxfev, 4=xtol, 5=bad_progress, 6=exception
            _outcome_labels = ["nb_newton_hit", "success(scipy)", "conv_badresid", "maxfev", "xtol", "bad_progress", "exception"]
            print("\nSolver Exit Reasons (rows=guess_idx, cols=outcome, counts across all workers)")
            header = f"  {'guess':>5s}" + "".join(f"  {lbl:>13s}" for lbl in _outcome_labels)
            print(header)
            for gi in range(ec.shape[0]):
                if np.any(ec[gi]):
                    row_str = f"  {gi:>5d}" + "".join(f"  {int(ec[gi, oi]):>13,d}" for oi in range(ec.shape[1]))
                    print(row_str)

    if args.weak_orders and args.weak_exact_reduced:
        print(f"weak_exact_zero_params:{result['weak_exact_zero_params']:>12,d}")
        print(f"weak_exact_deterministic:{result['weak_exact_deterministic']:>8,d}")
        print(f"weak_exact_nontrivial:{result['weak_exact_nontrivial']:>12,d}")

    if args.write_all or args.write_all_output_dir:
        n_write = len(result["all_successes"])
        if n_write >= 200:
            print(f"\nWriting up to {n_write} strategy tables (dedup_by={args.dedup_by})...", flush=True)
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
        perm_tuple = tuple(int(perm) for perm in success["perms"])
        reference = _materialize_success_result(result, success)
        cli_details: dict[str, Any] | None = None
        try:
            cli_success, cli_message, cli_details = _verify_via_old_cli_pipeline(
                config=result["config"],
                payoff_table_path=result["config"]["payoff_table"],
                strategy_df=reference["strategy_df"],
                players=result["players"],
                states=result["states"],
            effectivity=result["effectivity"],
            V=reference["V"],
            P=reference["P"],
            effectivity_rule=args.effectivity_rule or "heyen_lehtomaa_2021",
        )
        except Exception as exc:
            cli_success = False
            cli_message = f"legacy verifier unavailable: {exc}"
            cli_details = {"exception": repr(exc)}
        print()
        print("First Verified Success")
        print("-" * 80)
        for player_idx, player in enumerate(players):
            order = order_arrays[perm_tuple[player_idx]]
            if args.weak_orders:
                print(f"{player}: {_format_weak_order(states, order)}")
            else:
                print(f"{player}: {_format_ranking(states, order)}")
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
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
