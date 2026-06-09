"""
Per-state MIP solver.

For a fixed current state s and given value-function matrix V (shape n_states x n_players),
finds strategies (sigma, alpha, q) that satisfy the SMPE equilibrium conditions at s.

The approval probability q[i,s'] for proposer i proposing s' can use two formulas:
  'product'      : q = product(alpha_j for j in voters)            [unanimity]
  'at_least_one' : q = 1 - product(1-alpha_j for j in voters)     [majority]

Both are linearised via sequential McCormick chains.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csc_matrix
from typing import Dict, List, Tuple, Optional


def _mip_at_state(
    s: int,
    V: np.ndarray,
    approval_structure: List[List[List[Dict]]],
    forbidden: np.ndarray,
    n_players: int,
    n_states: int,
    eps: float = 1e-8,
    time_limit: float = 30.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve the per-state MIP for state s given value function V.

    Args:
        s:                 Index of the current state.
        V:                 (n_states, n_players) value function array.
        approval_structure: [proposer_idx][from_state_idx][to_state_idx] ->
                            {'formula': 'product'|'at_least_one'|'constant_1'|'constant_0',
                             'voters': tuple of voter indices}
        forbidden:         (n_players, n_states, n_states) or (n_players, n_states) bool.
                           If 3-d, sliced as forbidden[:, s, :].
        n_players, n_states: dimensions.
        eps:               Indifference threshold for pre-solving acceptances.
        time_limit:        HiGHS time limit in seconds per MIP.

    Returns:
        sigmas:  (n_players, n_states) proposal probabilities.
        alphas:  (n_players, n_states) acceptance probabilities (voter x next_state).
        qs:      (n_players, n_states) acceptance product probabilities (proposer x next_state).
        Any is None when the MIP is infeasible.
    """
    if forbidden.ndim == 3:
        forbidden = forbidden[:, s, :]

    # ------------------------------------------------------------------
    # 1. Pre-solve acceptance decisions
    # ------------------------------------------------------------------
    # alpha_status[(j, sp)] = 0.0 | 1.0 | 'free'
    alpha_status: dict = {}
    for j in range(n_players):
        V_s_j = float(V[s, j])
        for sp in range(n_states):
            gain = float(V[sp, j]) - V_s_j
            if gain > eps:
                alpha_status[(j, sp)] = 1.0
            elif gain < -eps:
                alpha_status[(j, sp)] = 0.0
            else:
                alpha_status[(j, sp)] = "free"

    # ------------------------------------------------------------------
    # 2. Compute q-structure per (proposer i, next state sp)
    # ------------------------------------------------------------------
    # q_info[(i, sp)] stores everything needed to set up q constraints.
    q_info: dict = {}
    free_set: set = set()   # (j, sp) pairs needing a free alpha variable

    for i in range(n_players):
        for sp in range(n_states):
            # Forbidden proposals have zero effective gain in C4 (player can't propose them)
            if forbidden[i, sp]:
                q_info[(i, sp)] = {'formula': 'constant_0'}
                continue

            entry = approval_structure[i][s][sp]
            formula = entry['formula']
            voters  = entry['voters']

            if formula == 'constant_1':
                q_info[(i, sp)] = {'formula': 'constant_1'}
                continue
            if formula == 'constant_0':
                q_info[(i, sp)] = {'formula': 'constant_0'}
                continue

            if formula == 'product':
                fp = 1.0
                fv = []
                for j in voters:
                    st = alpha_status[(j, sp)]
                    if st == 'free':
                        fv.append(j)
                        free_set.add((j, sp))
                    else:
                        fp *= float(st)   # 0.0 or 1.0
                if fp == 0.0:
                    q_info[(i, sp)] = {'formula': 'constant_0'}
                else:
                    q_info[(i, sp)] = {'formula': 'product', 'fp': fp, 'fv': fv}

            elif formula == 'at_least_one':
                # q = 1 - product(1 - alpha_j for j in voters)
                # If any voter strictly accepts: q = 1 immediately.
                has_strict_accept = any(alpha_status[(j, sp)] == 1.0 for j in voters)
                if has_strict_accept:
                    q_info[(i, sp)] = {'formula': 'constant_1'}
                    continue
                # Track free betas = (1-alpha_j) for j in voters
                # Strict rejects: beta_j = 1 (contribute fixed factor 1 to product → no effect)
                free_betas = []
                for j in voters:
                    if alpha_status[(j, sp)] == 'free':
                        free_betas.append(j)
                        free_set.add((j, sp))
                    # strict reject (alpha=0): beta=1, fixed factor → skip
                if not free_betas:
                    # All strict rejects → product(1-0) = 1 → q = 0
                    q_info[(i, sp)] = {'formula': 'constant_0'}
                else:
                    q_info[(i, sp)] = {'formula': 'at_least_one', 'free_betas': free_betas}

    # ------------------------------------------------------------------
    # 3. Variable layout
    # ------------------------------------------------------------------
    # sigma[i, sp]  : continuous [0,1]   at offset 0
    # z[i, sp]      : binary             at offset NS
    # alpha[j, sp]  : continuous [0,1]   at offset 2*NS  (free pairs only)
    # q[i, sp]      : continuous [0,1]   at offset 2*NS + n_free
    # w (McCormick) : continuous [0,1]   at offset 3*NS + n_free
    NS = n_players * n_states
    sigma_off = 0
    z_off     = NS
    alpha_off = 2 * NS

    free_list = sorted(free_set)
    free_idx  = {k: idx for idx, k in enumerate(free_list)}
    n_free    = len(free_list)
    q_off     = 2 * NS + n_free

    # Plan McCormick auxiliaries
    mcc_start: dict = {}
    w_count = 0
    for i in range(n_players):
        for sp in range(n_states):
            info = q_info[(i, sp)]
            fmt = info['formula']
            if fmt == 'product':
                k = len(info['fv'])
                if k >= 2:
                    mcc_start[(i, sp, 'product')] = q_off + NS + w_count
                    w_count += k - 1
            elif fmt == 'at_least_one':
                k = len(info['free_betas'])
                if k >= 2:
                    mcc_start[(i, sp, 'atleast')] = q_off + NS + w_count
                    w_count += k - 1

    n_vars = q_off + NS + w_count

    def _sigma(i, sp): return sigma_off + i * n_states + sp
    def _z(i, sp):     return z_off + i * n_states + sp
    def _alpha(j, sp): return alpha_off + free_idx[(j, sp)]
    def _q(i, sp):     return q_off + i * n_states + sp

    # ------------------------------------------------------------------
    # 4. Build constraints
    # ------------------------------------------------------------------
    V_abs_max = np.max(np.abs(V))
    big_M = 2.0 * V_abs_max if V_abs_max > 1e-12 else 1.0

    gains = np.empty((n_players, n_states))
    for i in range(n_players):
        for sp in range(n_states):
            gains[i, sp] = float(V[sp, i]) - float(V[s, i])

    max_rows = (
        2 * n_players       # C1 + C3
        + NS                # C2
        + NS * n_states     # C4
        + 2 * NS            # q equality / constant constraints
        + 4 * w_count       # McCormick 4-inequalities per auxiliary (generous)
        + 20
    )

    A  = lil_matrix((max_rows, n_vars))
    lo = np.full(max_rows, -np.inf)
    hi = np.full(max_rows, np.inf)
    r  = 0

    # C1: sum_sp sigma[i, sp] = 1  for each i
    for i in range(n_players):
        for sp in range(n_states):
            A[r, _sigma(i, sp)] = 1.0
        lo[r] = hi[r] = 1.0
        r += 1

    # C2: sigma[i, sp] - z[i, sp] <= 0
    for i in range(n_players):
        for sp in range(n_states):
            A[r, _sigma(i, sp)] = 1.0
            A[r, _z(i, sp)]     = -1.0
            hi[r] = 0.0
            r += 1

    # C3: sum_sp z[i, sp] = 1  for each i
    for i in range(n_players):
        for sp in range(n_states):
            A[r, _z(i, sp)] = 1.0
        lo[r] = hi[r] = 1.0
        r += 1

    # C4: g[i,sp'']*q[i,sp''] - g[i,sp']*q[i,sp'] + M*z[i,sp'] <= M
    for i in range(n_players):
        for sp in range(n_states):
            g_sp = gains[i, sp]
            for spp in range(n_states):
                if sp == spp:
                    continue
                g_spp = gains[i, spp]
                A[r, _q(i, spp)] = g_spp
                A[r, _q(i, sp)]  = -g_sp
                A[r, _z(i, sp)]  = big_M
                hi[r] = big_M
                r += 1

    # ------------------------------------------------------------------
    # q constraints
    # ------------------------------------------------------------------
    def _mcc_product(r_, w_var, a_alpha_idx, b_alpha_idx):
        """McCormick for w = alpha_a * alpha_b ∈ [0,1]²."""
        A[r_, w_var] = 1.0;  A[r_, a_alpha_idx] = -1.0;                    hi[r_] = 0.0; r_ += 1
        A[r_, w_var] = 1.0;  A[r_, b_alpha_idx] = -1.0;                    hi[r_] = 0.0; r_ += 1
        A[r_, w_var] = -1.0; A[r_, a_alpha_idx] = 1.0; A[r_, b_alpha_idx] = 1.0; hi[r_] = 1.0; r_ += 1
        return r_

    def _mcc_atleast(r_, w_var, a_alpha_idx, b_alpha_idx):
        """McCormick for w = (1-alpha_a)*(1-alpha_b):
           w <= 1-alpha_a, w <= 1-alpha_b, w >= 1-alpha_a-alpha_b."""
        A[r_, w_var] = 1.0; A[r_, a_alpha_idx] = 1.0;  hi[r_] = 1.0; r_ += 1   # w <= 1-alpha_a
        A[r_, w_var] = 1.0; A[r_, b_alpha_idx] = 1.0;  hi[r_] = 1.0; r_ += 1   # w <= 1-alpha_b
        A[r_, w_var] = -1.0; A[r_, a_alpha_idx] = 1.0; A[r_, b_alpha_idx] = 1.0
        lo[r_] = 0.0; r_ += 1                                                    # w >= 1-alpha_a-alpha_b (i.e., -w+alpha_a+alpha_b <= 1 ← lo bound for -w+a+b >= 0)
        return r_

    for i in range(n_players):
        for sp in range(n_states):
            info = q_info[(i, sp)]
            fmt  = info['formula']
            qv   = _q(i, sp)

            if fmt == 'constant_1':
                A[r, qv] = 1.0; lo[r] = hi[r] = 1.0; r += 1

            elif fmt == 'constant_0':
                A[r, qv] = 1.0; lo[r] = hi[r] = 0.0; r += 1

            elif fmt == 'product':
                fp = info['fp']   # always 1.0 here (fp=0 → constant_0)
                fv = info['fv']
                k  = len(fv)
                if k == 0:
                    A[r, qv] = 1.0; lo[r] = hi[r] = fp; r += 1
                elif k == 1:
                    # q = alpha[fv[0], sp]
                    A[r, qv] = 1.0; A[r, _alpha(fv[0], sp)] = -fp; lo[r] = hi[r] = 0.0; r += 1
                else:
                    base = mcc_start[(i, sp, 'product')]
                    r = _mcc_product(r, base, _alpha(fv[0], sp), _alpha(fv[1], sp))
                    for m in range(1, k - 1):
                        r = _mcc_product(r, base + m, base + m - 1, _alpha(fv[m + 1], sp))
                    # q = fp * w_{k-2}
                    A[r, qv] = 1.0; A[r, base + k - 2] = -fp; lo[r] = hi[r] = 0.0; r += 1

            elif fmt == 'at_least_one':
                fb = info['free_betas']  # indices j where alpha_j is free
                k  = len(fb)
                if k == 0:
                    # All strict rejects → product = 1 → q = 0
                    A[r, qv] = 1.0; lo[r] = hi[r] = 0.0; r += 1
                elif k == 1:
                    # q = 1 - (1 - alpha[fb[0], sp]) = alpha[fb[0], sp]
                    A[r, qv] = 1.0; A[r, _alpha(fb[0], sp)] = -1.0; lo[r] = hi[r] = 0.0; r += 1
                else:
                    base = mcc_start[(i, sp, 'atleast')]
                    # w0 = (1-alpha_j0)*(1-alpha_j1)
                    r = _mcc_atleast(r, base, _alpha(fb[0], sp), _alpha(fb[1], sp))
                    for m in range(1, k - 1):
                        # w_m = w_{m-1} * (1 - alpha_{j_{m+1}})
                        r = _mcc_atleast(r, base + m, base + m - 1, _alpha(fb[m + 1], sp))
                    # q + w_{k-2} = 1  → q = 1 - product(1-alpha_j)
                    A[r, qv] = 1.0; A[r, base + k - 2] = 1.0; lo[r] = hi[r] = 1.0; r += 1

    # Trim
    A  = A[:r, :]
    lo = lo[:r]
    hi = hi[:r]

    # ------------------------------------------------------------------
    # 5. Variable bounds and integrality
    # ------------------------------------------------------------------
    var_lo = np.zeros(n_vars)
    var_hi = np.ones(n_vars)
    integrality = np.zeros(n_vars)

    for i in range(n_players):
        for sp in range(n_states):
            integrality[_z(i, sp)] = 1
            if forbidden[i, sp]:
                var_hi[_sigma(i, sp)] = 0.0
                var_hi[_z(i, sp)]     = 0.0

    # ------------------------------------------------------------------
    # 6. Solve
    # ------------------------------------------------------------------
    c = np.zeros(n_vars)
    res = milp(
        c,
        constraints=LinearConstraint(csc_matrix(A), lo, hi),
        integrality=integrality,
        bounds=Bounds(var_lo, var_hi),
        options={"disp": False, "time_limit": time_limit},
    )

    if res.status != 0:
        return None, None, None

    x = res.x

    # Extract sigmas
    sigmas = np.clip(x[sigma_off: sigma_off + NS].reshape(n_players, n_states), 0.0, 1.0)

    # Extract alphas (fixed + free from MIP)
    alphas = np.zeros((n_players, n_states))
    for j in range(n_players):
        for sp in range(n_states):
            st = alpha_status[(j, sp)]
            if st == "free":
                if (j, sp) in free_idx:
                    alphas[j, sp] = float(np.clip(x[_alpha(j, sp)], 0.0, 1.0))
                else:
                    alphas[j, sp] = 0.5  # free but not in any committee
            else:
                alphas[j, sp] = float(st)

    # Extract qs
    qs = np.clip(x[q_off: q_off + NS].reshape(n_players, n_states), 0.0, 1.0)

    return sigmas, alphas, qs
