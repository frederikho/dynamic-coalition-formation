"""
Cycle detection and resolution for VFI.

Two resolution strategies:
  1. V-interpolation bisection (detect_vfi_cycle / resolve_cycle):
     Classic approach — bisects between two cycle iterates to find V* where a
     dominant oscillating gain equals zero, then solves MIPs at V*.
     Works for pure-strategy equilibria but can fail when the equilibrium
     requires simultaneous mixed strategies.

  2. Mixed-strategy refinement (refine_mixing_probabilities):
     Post-processing step run after step 1.  Identifies every transition
     (proposer i, state s → state sp) that still has near-zero gain at V*
     (i.e., the MIP result is carrying a free mixing probability).  Treats
     each such probability pⱼ as a free variable, writes T(p₁, …, pₖ) linearly,
     and solves the fixed-point system
         V(p)[spⱼ, iⱼ] − V(p)[sⱼ, iⱼ] = 0   for all j
     using scipy.optimize (brentq for k=1, fsolve for k>1).
     This correctly finds simultaneous mixed-strategy equilibria.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, fsolve
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers shared with vfi.py (duplicated here to avoid circular imports)
# ---------------------------------------------------------------------------

def _solve_values_local(T: np.ndarray, payoffs: np.ndarray, delta: float) -> np.ndarray:
    """V = (I − δT)⁻¹ (1−δ) π  (n_states × n_players)."""
    import scipy.linalg
    n = T.shape[0]
    A = np.eye(n) - delta * T
    rhs = (1.0 - delta) * payoffs
    return scipy.linalg.solve(A, rhs)


def _build_T_from_sigma(
    sigma: np.ndarray,   # (n_states, n_players, n_states)
    qs: np.ndarray,      # (n_states, n_players, n_states)
    rho: np.ndarray,     # (n_players,)
    n_states: int,
    n_players: int,
) -> np.ndarray:
    T = np.zeros((n_states, n_states))
    for s in range(n_states):
        for sp in range(n_states):
            if sp == s:
                continue
            for i in range(n_players):
                T[s, sp] += rho[i] * sigma[s, i, sp] * qs[s, i, sp]
        T[s, s] = max(0.0, 1.0 - T[s, :].sum())
    return T


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

def detect_vfi_cycle(
    V_history: List[np.ndarray],
    V_new: np.ndarray,
    tol: float = 1e-6,
    window: int = 8,
) -> Optional[int]:
    """
    Return the cycle period k if V_new matches a previous iterate within threshold,
    else None.

    Threshold = max(500*tol, 1e-4 * max|V|) so it scales with payoff magnitude.
    Scans k = 2 .. min(window, len(V_history)).
    """
    v_scale = max(np.max(np.abs(V_new)), 1.0)
    threshold = max(500.0 * tol, 1e-4 * v_scale)
    n = len(V_history)
    for k in range(2, min(window, n) + 1):
        if np.max(np.abs(V_new - V_history[-k])) < threshold:
            return k
    return None


# ---------------------------------------------------------------------------
# Mixed-strategy fixed-point refinement
# ---------------------------------------------------------------------------

def refine_mixing_probabilities(
    V_star: np.ndarray,          # (n_states, n_players) — cycle-resolved V
    all_sigmas: np.ndarray,      # (n_states, n_players, n_states) — strategies at V_star
    all_alphas: np.ndarray,      # (n_states, n_players, n_states)
    all_qs: np.ndarray,          # (n_states, n_players, n_states) — approval products
    payoffs: np.ndarray,         # (n_states, n_players)
    rho: np.ndarray,             # (n_players,)
    delta: float,
    n_players: int,
    n_states: int,
    mix_threshold: float = 5e-3,  # |gain| below this → treat as free variable
    osc_transitions: Optional[set] = None,  # (i, s, sp) triples that were oscillating
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine mixing probabilities so that V_star is a true fixed point.

    Two sources of free variables:
    1. Transitions where sigma > 0 AND |gain| ≤ mix_threshold (near-indifferent support).
    2. Oscillating transitions (from cycle detection) where the MIP at V* placed sigma=0
       (it chose the other side of the oscillation), but |gain| ≤ mix_threshold at V*
       and q > 0 (the transition can actually occur).

    The second source is the key extension beyond Jere's description: VFI oscillation
    means a player is indifferent, but the MIP (with C3: sum z=1) snaps to a pure
    strategy.  Without knowing the oscillation history, refine_mixing_probabilities
    would miss the sigma=0 oscillating transition entirely.

    The transition matrix decomposes as:
        T(p) = T₀ + Σⱼ pⱼ · ΔTⱼ
    where T₀ is T with all free vars zeroed out (mass moved to diagonal),
    and ΔTⱼ is the incremental T contribution of pⱼ = 1.

    We solve the fixed-point system:
        V(p)[spⱼ, iⱼ] − V(p)[sⱼ, iⱼ] = 0   for all j

    using brentq (k=1) or fsolve (k>1).

    Returns refined (V, sigmas, alphas, qs).  Falls back to inputs on failure.
    """
    # --- 1. Identify free mixing variables ---
    free_vars: List[Tuple[int, int, int]] = []  # (s, i, sp)
    seen: set = set()  # (i, s, sp) already added

    # Source 1: current support with near-zero gain
    for s in range(n_states):
        for i in range(n_players):
            for sp in range(n_states):
                if sp == s:
                    continue
                if all_sigmas[s, i, sp] < 1e-9:
                    continue
                gain = float(V_star[sp, i]) - float(V_star[s, i])
                if abs(gain) <= mix_threshold:
                    free_vars.append((s, i, sp))
                    seen.add((i, s, sp))

    # Source 2: oscillating transitions where MIP chose sigma=0 (the other side)
    if osc_transitions:
        for (i, s, sp) in osc_transitions:
            if (i, s, sp) in seen:
                continue
            if s == sp:
                continue
            gain = float(V_star[sp, i]) - float(V_star[s, i])
            if abs(gain) > mix_threshold:
                continue  # gain resolved clearly at V*; not a free variable here
            if float(all_qs[s, i, sp]) < 1e-9:
                continue  # q=0: transition can never occur regardless of sigma
            free_vars.append((s, i, sp))
            seen.add((i, s, sp))

    if not free_vars:
        return V_star, all_sigmas, all_alphas, all_qs

    k = len(free_vars)

    # --- 2. Build T₀ and ΔTⱼ matrices ---
    # T₀: set free vars' sigma contributions to 0 (move mass to diagonal).
    sigma_base = all_sigmas.copy()
    for (s, i, sp) in free_vars:
        sigma_base[s, i, s] += sigma_base[s, i, sp]
        sigma_base[s, i, sp] = 0.0

    T0 = _build_T_from_sigma(sigma_base, all_qs, rho, n_states, n_players)

    # ΔTⱼ: incremental T change when pⱼ goes from 0 to 1 (moves mass from diagonal).
    delta_Ts = []
    for (s_j, i_j, sp_j) in free_vars:
        dT = np.zeros((n_states, n_states))
        contribution = rho[i_j] * all_qs[s_j, i_j, sp_j]
        dT[s_j, sp_j] += contribution
        dT[s_j, s_j] -= contribution
        delta_Ts.append(dT)

    # --- 3. Gain function ---
    def _T_from_probs(probs):
        T = T0.copy()
        for j, p in enumerate(probs):
            T += p * delta_Ts[j]
        return T

    def gains(probs):
        T = _T_from_probs(probs)
        V = _solve_values_local(T, payoffs, delta)
        return np.array([
            V[sp_j, i_j] - V[s_j, i_j]
            for (s_j, i_j, sp_j) in free_vars
        ])

    # --- 4. Solve ---
    p0 = np.array([all_sigmas[s_j, i_j, sp_j] for (s_j, i_j, sp_j) in free_vars],
                  dtype=np.float64)

    try:
        if k == 1:
            g0 = float(gains([0.0])[0])
            g1 = float(gains([1.0])[0])
            if g0 * g1 > 0:
                # No sign change — refinement can't fix this; return as-is
                return V_star, all_sigmas, all_alphas, all_qs
            p_star = brentq(lambda p: gains([p])[0], 0.0, 1.0, xtol=1e-12)
            probs_star = np.array([p_star])
        else:
            # Multi-dimensional: use fsolve with bounds check
            result, info, ier, msg = fsolve(gains, p0, full_output=True)
            if ier != 1:
                # fsolve failed; try a few random starts
                from numpy.random import default_rng
                rng = default_rng(0)
                best_residual = np.inf
                best_result = None
                for _ in range(20):
                    p_try = rng.uniform(0.0, 1.0, size=k)
                    res, info2, ier2, _ = fsolve(gains, p_try, full_output=True)
                    if ier2 == 1:
                        residual = np.max(np.abs(gains(res)))
                        if residual < best_residual:
                            best_residual = residual
                            best_result = res
                if best_result is None or best_residual > 1e-6:
                    return V_star, all_sigmas, all_alphas, all_qs
                result = best_result
            probs_star = np.clip(result, 0.0, 1.0)
    except Exception:
        return V_star, all_sigmas, all_alphas, all_qs

    # --- 5. Build refined strategies ---
    sigma_refined = sigma_base.copy()
    for j, (s_j, i_j, sp_j) in enumerate(free_vars):
        p = float(probs_star[j])
        sigma_refined[s_j, i_j, sp_j] = p
        sigma_refined[s_j, i_j, s_j] -= p

    # Oscillating transitions added via Source 2 had sigma=0 originally, so
    # sigma_base moved 0 mass to the diagonal — subtracting p above can make
    # the status-quo entry negative.  Clip and renormalize to restore [0,1].
    for s in range(n_states):
        for i in range(n_players):
            row = np.clip(sigma_refined[s, i, :], 0.0, None)
            total = row.sum()
            if total > 1e-12:
                sigma_refined[s, i, :] = row / total
            else:
                sigma_refined[s, i, :] = 0.0
                sigma_refined[s, i, s] = 1.0

    T_refined = _T_from_probs(probs_star)
    V_refined = _solve_values_local(T_refined, payoffs, delta)

    return V_refined, sigma_refined, all_alphas, all_qs


# ---------------------------------------------------------------------------
# V-interpolation bisection (original approach, now used as first pass)
# ---------------------------------------------------------------------------

def resolve_cycle(
    V_history: List[np.ndarray],
    cycle_k: int,
    mip_fn,
    n_players: int,
    n_states: int,
    approval_structure,
    forbidden: np.ndarray,
    eps: float = 1e-8,
    bisect_tol: float = 1e-10,
    payoffs: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    delta: Optional[float] = None,
    mix_threshold: float = 5e-3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Resolve a detected VFI cycle.

    Pipeline:
      1. Find dominant oscillating gain → bisect V(t) = (1−t)V_a + t·V_b
         to get V* where that gain = 0.
      2. Solve MIP at V* for all states.
      3. (Optional, when payoffs/rho/delta provided) Refine mixing probabilities
         via the fixed-point system so V* is a true equilibrium fixed point.

    Returns:
        (V_star, all_sigmas, all_alphas, all_qs)  or  (None, None, None, None).

    mip_fn: callable(s, V, approval_structure, forbidden, n_players, n_states) →
                (sigmas, alphas, qs) or (None, None, None).
    """
    # Gather cycle iterates
    cycle_Vs = V_history[-cycle_k:]

    # ------------------------------------------------------------------
    # 1. Find dominant oscillating gain; collect ALL oscillating transitions
    # ------------------------------------------------------------------
    best_bracket = None  # (|g_a − g_b|, V_a, V_b, i, s, sp)
    all_osc: set = set()  # (i, s, sp) triples with sign-changing gain

    for a_idx in range(len(cycle_Vs)):
        for b_idx in range(a_idx + 1, len(cycle_Vs)):
            V_a = cycle_Vs[a_idx]
            V_b = cycle_Vs[b_idx]
            for i in range(n_players):
                for s in range(n_states):
                    for sp in range(n_states):
                        if s == sp:
                            continue
                        g_a = float(V_a[sp, i]) - float(V_a[s, i])
                        g_b = float(V_b[sp, i]) - float(V_b[s, i])
                        if g_a * g_b < 0:  # sign change
                            all_osc.add((i, s, sp))
                            width = abs(g_a - g_b)
                            if best_bracket is None or width > best_bracket[0]:
                                best_bracket = (width, V_a.copy(), V_b.copy(), i, s, sp)

    if best_bracket is None:
        return None, None, None, None

    _, V_a, V_b, osc_i, osc_s, osc_sp = best_bracket

    # ------------------------------------------------------------------
    # 2a. Try multi-dimensional V*: find t = (t_0,...,t_{K-1}) such that
    #     V* = sum_k t_k * V_k has all oscillating gains ≈ 0 simultaneously.
    #
    #     Minimize ||A t||^2  subject to  1^T t = 1
    #     where A[m,k] = V_k[sp_m, i_m] - V_k[s_m, i_m].
    #
    #     The KKT augmented system has size (K+1) × (K+1) — tiny.
    # ------------------------------------------------------------------
    V_star = None
    if len(all_osc) > 0:
        K = len(cycle_Vs)
        osc_list = sorted(all_osc)
        M = len(osc_list)

        A = np.zeros((M, K))
        for m, (oi, os, osp) in enumerate(osc_list):
            for k, Vk in enumerate(cycle_Vs):
                A[m, k] = float(Vk[osp, oi]) - float(Vk[os, oi])

        # Augmented KKT system: [2 A^T A  1; 1^T  0] [t; λ] = [0; 1]
        AtA = A.T @ A
        one = np.ones(K)
        aug = np.zeros((K + 1, K + 1))
        aug[:K, :K] = 2.0 * AtA
        aug[:K, K] = one
        aug[K, :K] = one
        rhs = np.zeros(K + 1)
        rhs[K] = 1.0

        try:
            sol = np.linalg.solve(aug, rhs)
            t_vec = sol[:K]
            V_star_cand = sum(t_vec[k] * cycle_Vs[k] for k in range(K))
            residuals = A @ t_vec
            max_res = float(np.max(np.abs(residuals)))
            if max_res < mix_threshold:
                V_star = V_star_cand
        except np.linalg.LinAlgError:
            pass

    # ------------------------------------------------------------------
    # 2b. Fall back: 1-D bisection on the dominant oscillation
    # ------------------------------------------------------------------
    if V_star is None:
        def gain(t: float) -> float:
            Vt = (1.0 - t) * V_a + t * V_b
            return float(Vt[osc_sp, osc_i]) - float(Vt[osc_s, osc_i])

        try:
            t_star = brentq(gain, 0.0, 1.0, xtol=bisect_tol)
        except ValueError:
            return None, None, None, None

        V_star = (1.0 - t_star) * V_a + t_star * V_b

    # ------------------------------------------------------------------
    # 3. Solve MIP at V* for all states
    # ------------------------------------------------------------------
    all_sigmas = np.zeros((n_states, n_players, n_states))
    all_alphas = np.zeros((n_states, n_players, n_states))
    all_qs = np.zeros((n_states, n_players, n_states))

    for s in range(n_states):
        sigmas, alphas, qs = mip_fn(s, V_star, approval_structure, forbidden, n_players, n_states)
        if sigmas is None:
            return None, None, None, None
        all_sigmas[s] = sigmas
        all_alphas[s] = alphas
        all_qs[s] = qs

    # ------------------------------------------------------------------
    # 4. Refine mixing probabilities (if game data provided)
    # ------------------------------------------------------------------
    if payoffs is not None and rho is not None and delta is not None:
        V_star, all_sigmas, all_alphas, all_qs = refine_mixing_probabilities(
            V_star=V_star,
            all_sigmas=all_sigmas,
            all_alphas=all_alphas,
            all_qs=all_qs,
            payoffs=payoffs,
            rho=rho,
            delta=delta,
            n_players=n_players,
            n_states=n_states,
            mix_threshold=mix_threshold,
            osc_transitions=all_osc,
        )

    return V_star, all_sigmas, all_alphas, all_qs
