"""
Value Function Iteration (VFI) outer loop.

Outer loop:
  1. Solve per-state MIP for all states s given current V.
  2. Build transition matrix T from (sigma, q, rho).
  3. Update V = (I - δT)^{-1} (1-δ) π.
  4. Repeat until ||ΔV|| < tol or cycle detected.
  5. On cycle: delegate to cycle resolver (V-bisection → MIP at V*).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from typing import Dict, List, Optional, Tuple

from lib.equilibrium.mip_vfi.mip import _mip_at_state
from lib.equilibrium.mip_vfi.cycle import detect_vfi_cycle, resolve_cycle


def _build_transition_matrix(
    all_sigmas: np.ndarray,   # (n_states, n_players, n_states)
    all_qs: np.ndarray,        # (n_states, n_players, n_states)
    rho: np.ndarray,           # (n_players,) proposer probabilities
    n_states: int,
    n_players: int,
) -> np.ndarray:
    """
    T[s, sp] = sum_i rho[i] * sigma[s, i, sp] * q[s, i, sp]
    Diagonal absorbs remaining mass so rows sum to 1.
    """
    T = np.zeros((n_states, n_states))
    for s in range(n_states):
        for sp in range(n_states):
            if sp == s:
                continue
            for i in range(n_players):
                T[s, sp] += rho[i] * all_sigmas[s, i, sp] * all_qs[s, i, sp]
        T[s, s] = max(0.0, 1.0 - T[s, :].sum())
    return T


def _solve_values(
    T: np.ndarray,
    payoffs: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    V = (I - δT)^{-1} (1-δ) π
    Solved exactly via scipy.linalg.solve.

    payoffs: (n_states, n_players)
    Returns: (n_states, n_players)
    """
    n = T.shape[0]
    A = np.eye(n) - delta * T
    rhs = (1.0 - delta) * payoffs  # (n_states, n_players)
    return scipy.linalg.solve(A, rhs)


def _correct_alphas_at_V(
    all_alphas: np.ndarray,  # (n_states, n_players, n_states)
    approval_structure,
    V: np.ndarray,           # (n_states, n_players)
    n_states: int,
    n_players: int,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Re-fix alpha[s, j, sp] based on sign of V[sp, j] - V[s, j].
    Only touches entries where j is actually a voter for at least one proposer.
    Near-zero gains (|dv| <= tol) are left unchanged (mixed-strategy region).
    """
    alphas = all_alphas.copy()
    for i in range(n_players):
        for s in range(n_states):
            for sp in range(n_states):
                if s == sp:
                    continue
                info = approval_structure[i][s][sp]
                if info is None:
                    continue
                if info['formula'] not in ('product', 'at_least_one'):
                    continue
                for j in info['voters']:
                    dv = float(V[sp, j]) - float(V[s, j])
                    if dv > tol:
                        alphas[s, j, sp] = 1.0
                    elif dv < -tol:
                        alphas[s, j, sp] = 0.0
    return alphas


def _recompute_qs_from_alphas(
    all_alphas: np.ndarray,
    approval_structure,
    n_players: int,
    n_states: int,
) -> np.ndarray:
    """
    Recompute q[s,i,sp] = product(alpha[s,j,sp] for j in voters(i,s,sp))
    from blended alphas, using the approval structure.

    This is necessary after damped updates because blended qs are not equal
    to the product of blended alphas (nonlinearity).
    """
    qs = np.zeros((n_states, n_players, n_states))
    for i in range(n_players):
        for s in range(n_states):
            for sp in range(n_states):
                info = approval_structure[i][s][sp]
                if info is None:
                    continue
                formula = info['formula']
                voters  = info['voters']
                if formula == 'constant_1':
                    qs[s, i, sp] = 1.0
                elif formula == 'constant_0':
                    qs[s, i, sp] = 0.0
                elif formula == 'product':
                    q = 1.0
                    for j in voters:
                        q *= float(all_alphas[s, j, sp])
                    qs[s, i, sp] = q
                elif formula == 'at_least_one':
                    # 1 - prod(1 - alpha_j)
                    q = 1.0
                    for j in voters:
                        q *= (1.0 - float(all_alphas[s, j, sp]))
                    qs[s, i, sp] = 1.0 - q
    return qs


def run_vfi(
    payoffs: np.ndarray,         # (n_states, n_players)
    approval_structure,              # [proposer][from_state][to_state] -> tuple of voter idxs
    forbidden: np.ndarray,       # (n_players, n_states) bool - forbidden[i, sp]
    rho: np.ndarray,             # (n_players,) proposer probabilities
    delta: float,
    V_init: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 300,
    cycle_window: int = 16,
    mip_eps: float = 1e-8,
    mip_time_limit: float = 30.0,
    damping_alpha: float = 0.0,
    logger=None,
) -> Dict:
    """
    Run Value Function Iteration until convergence or cycle.

    Returns a dict with keys:
        success:    True if VFI converged or cycle was resolved.
        V:          (n_states, n_players) final value functions.
        all_sigmas: (n_states, n_players, n_states) proposal probabilities.
        all_alphas: (n_states, n_players, n_states) acceptance probabilities.
        all_qs:     (n_states, n_players, n_states) acceptance products.
        T:          (n_states, n_states) transition matrix.
        iterations: number of VFI iterations.
        stopping:   one of 'converged', 'cycle_resolved', 'max_iter', 'mip_infeasible'.

    If damping_alpha > 0, uses damped strategy update to resolve cycles:
        sigma_new = (1-alpha)*sigma_old + alpha*sigma_mip
    Cycle detection is disabled in damped mode; convergence is the only stopping
    condition. After convergence, qs are recomputed from blended alphas so that
    the strategy DataFrame passes verification correctly.
    """
    n_states, n_players = payoffs.shape

    def _log(msg):
        if logger:
            logger.info(msg)

    # Initialise V
    V = payoffs.copy() if V_init is None else V_init.copy()
    V_history: List[np.ndarray] = [V.copy()]

    # Initialise sigmas/alphas/qs from first MIP solve so they sum to 1 from iter 0.
    # (Initialising from zeros would cause sigma sums to be (1-(1-alpha)^k) < 1.)
    all_sigmas = np.zeros((n_states, n_players, n_states))
    all_alphas = np.zeros((n_states, n_players, n_states))
    all_qs = np.zeros((n_states, n_players, n_states))
    if damping_alpha > 0.0:
        for s in range(n_states):
            sig0, alp0, qs0 = _mip_at_state(s, V, approval_structure, forbidden, n_players, n_states,
                                             eps=mip_eps, time_limit=mip_time_limit)
            if sig0 is None:
                _log(f"  VFI damped init: MIP infeasible at state {s}")
                return {
                    "success": False, "V": V, "all_sigmas": all_sigmas,
                    "all_alphas": all_alphas, "all_qs": all_qs,
                    "T": None, "iterations": 0, "stopping": "mip_infeasible",
                }
            all_sigmas[s] = sig0
            all_alphas[s] = alp0
            all_qs[s] = qs0

    def _mip_fn(s, V_arg, committees, forb, np_, ns_):
        return _mip_at_state(s, V_arg, committees, forb, np_, ns_, eps=mip_eps, time_limit=mip_time_limit)

    for iteration in range(max_iter):
        # --- Per-state MIP ---
        mip_failed = False
        for s in range(n_states):
            sigmas, alphas, qs = _mip_fn(s, V, approval_structure, forbidden, n_players, n_states)
            if sigmas is None:
                _log(f"  VFI iter {iteration}: MIP infeasible at state {s}")
                mip_failed = True
                break
            if damping_alpha > 0.0:
                a = damping_alpha
                all_sigmas[s] = (1.0 - a) * all_sigmas[s] + a * sigmas
                all_alphas[s] = (1.0 - a) * all_alphas[s] + a * alphas
                # Blend qs during iteration so T is self-consistent with blended sigma.
                # At convergence, qs are recomputed from blended alphas for verification.
                all_qs[s] = (1.0 - a) * all_qs[s] + a * qs
            else:
                all_sigmas[s] = sigmas
                all_alphas[s] = alphas
                all_qs[s] = qs

        if mip_failed:
            return {
                "success": False,
                "V": V,
                "all_sigmas": all_sigmas,
                "all_alphas": all_alphas,
                "all_qs": all_qs,
                "T": None,
                "iterations": iteration,
                "stopping": "mip_infeasible",
            }

        # --- Build T and update V ---
        # During damped iteration, use MIP-pure qs (all_qs[s] = qs from last MIP)
        # so T is consistent with the MIP best-response at the current V.
        T = _build_transition_matrix(all_sigmas, all_qs, rho, n_states, n_players)
        V_new = _solve_values(T, payoffs, delta)

        delta_V = np.max(np.abs(V_new - V))
        _log(f"  VFI iter {iteration}: ||ΔV|| = {delta_V:.2e}")

        # --- Convergence check ---
        if delta_V < tol:
            if damping_alpha > 0.0:
                # Recompute qs from blended alphas so that the strategy DataFrame
                # passes verification in TransitionProbabilitiesOptimized.
                # (Blended qs ≠ product(blended alphas) due to nonlinearity.)
                all_qs = _recompute_qs_from_alphas(
                    all_alphas, approval_structure, n_players, n_states
                )
                T = _build_transition_matrix(all_sigmas, all_qs, rho, n_states, n_players)
                V_new = _solve_values(T, payoffs, delta)
                _log(f"  VFI (damped) converged after {iteration + 1} iterations")
            else:
                # Re-solve MIPs at V_new so strategies are consistent with the fixed-point V.
                # Without this, tiny V^t vs V^{t+1} gaps can flip optimality in verification.
                _log(f"  VFI ||ΔV|| < tol; re-solving MIPs at V_new for consistency...")
                for s in range(n_states):
                    sigmas, alphas, qs = _mip_fn(s, V_new, approval_structure, forbidden, n_players, n_states)
                    if sigmas is None:
                        _log(f"  MIP infeasible at state {s} during final re-solve; returning as-is.")
                        break
                    all_sigmas[s] = sigmas
                    all_alphas[s] = alphas
                    all_qs[s] = qs
                else:
                    T = _build_transition_matrix(all_sigmas, all_qs, rho, n_states, n_players)
                    V_new = _solve_values(T, payoffs, delta)
                _log(f"  VFI converged after {iteration + 1} iterations")
            return {
                "success": True,
                "V": V_new,
                "all_sigmas": all_sigmas,
                "all_alphas": all_alphas,
                "all_qs": all_qs,
                "T": T,
                "iterations": iteration + 1,
                "stopping": "converged",
            }

        # --- Cycle detection (disabled in damped mode) ---
        if damping_alpha > 0.0:
            V = V_new
            V_history.append(V.copy())
            continue

        cycle_k = detect_vfi_cycle(V_history, V_new, tol=tol, window=cycle_window)
        if cycle_k is not None:
            _log(f"  VFI cycle of period {cycle_k} detected at iteration {iteration}; resolving...")
            V_star, star_sigmas, star_alphas, star_qs = resolve_cycle(
                V_history=V_history,
                cycle_k=cycle_k,
                mip_fn=_mip_fn,
                n_players=n_players,
                n_states=n_states,
                approval_structure=approval_structure,
                forbidden=forbidden,
                eps=mip_eps,
            )
            if V_star is None:
                _log("  Cycle resolution failed; continuing VFI...")
            else:
                T_star = _build_transition_matrix(star_sigmas, star_qs, rho, n_states, n_players)
                V_final = _solve_values(T_star, payoffs, delta)
                _log(f"  Cycle resolved (||V_final - V*|| = "
                     f"{np.max(np.abs(V_final - V_star)):.2e})")
                return {
                    "success": True,
                    "V": V_star,
                    "all_sigmas": star_sigmas,
                    "all_alphas": star_alphas,
                    "all_qs": star_qs,
                    "T": T_star,
                    "iterations": iteration + 1,
                    "stopping": "cycle_resolved",
                }

        V = V_new
        V_history.append(V.copy())

    _log(f"  VFI reached max_iter={max_iter} without convergence")
    T = _build_transition_matrix(all_sigmas, all_qs, rho, n_states, n_players)
    return {
        "success": False,
        "V": V,
        "all_sigmas": all_sigmas,
        "all_alphas": all_alphas,
        "all_qs": all_qs,
        "T": T,
        "iterations": max_iter,
        "stopping": "max_iter",
    }
