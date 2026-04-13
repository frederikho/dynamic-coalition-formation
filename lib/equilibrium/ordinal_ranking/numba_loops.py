"""Numba-accelerated inner loops for ordinal ranking search."""

from __future__ import annotations

import math
import numpy as np

try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:
    @_numba.njit(cache=True)
    def _build_arrays_weak_nb(tiers_arr, comm_arr, comm_size, protocol_arr):
        """Numba JIT version of _build_induced_arrays_weak."""
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
        """Numba JIT equilibrium verifier."""
        n_players = V.shape[1]
        n_states = V.shape[0]
        tol = 1e-9

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

        return True

    @_numba.njit(cache=True)
    def _solve_V_nb(P, payoff_array, discounting):
        """Numba JIT value-function solver."""
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
        n_fa = int(n_fa)
        n_pt = int(n_pt)
        n_aff = int(n_aff)
        
        # Step 1: sigmoid/softmax mapping
        alpha_phys = alpha_raw.copy()
        for k in range(int(alpha_raw.shape[0])):
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
        n_fa = int(n_fa)
        n_pt = int(n_pt)
        n_vars = int(n_vars)
        
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
        """Numba JIT residual and Jacobian for _solve_weak_equalities."""
        n_vars = alpha_raw.shape[0]
        n_fa = int(n_fa)
        n_pt = int(n_pt)
        n_aff = int(n_aff)

        probs, pass_, action = _residuals_nb_prep(
            alpha_raw, canon_probs, canon_action, canon_pass,
            fa_arr, n_fa, pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
            aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
            comm_arr, comm_size, tiers, n_players, n_states
        )
        
        P = _residuals_nb_p_agg(probs, pass_, protocol_arr, n_players, n_states)
        A_mat = np.eye(n_states) - discounting * P
        V = np.linalg.solve(A_mat, (1.0 - discounting) * payoff_array)
        res = _residuals_nb_residuals(V, pass_, fa_arr, n_fa, pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt, n_vars)

        if not compute_jac:
            return res, np.zeros((1, 1))

        d_phys_d_raw = np.ones(n_vars)
        for k in range(n_vars):
            z = alpha_raw[k]
            if z > 50.0: z = 50.0
            elif z < -50.0: z = -50.0
            if z >= 0.0:
                s = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                s = ez / (1.0 + ez)
            d_phys_d_raw[k] = s * (1.0 - s)

        jac = np.zeros((n_vars, n_vars))
        RHS_all = np.zeros((n_states, n_players * n_vars))

        for k in range(n_vars):
            if k < n_fa:
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
                RHS_k = discounting * (dP @ V)
                RHS_all[:, k*n_players : (k+1)*n_players] = RHS_k

        dV_all = np.linalg.solve(A_mat, RHS_all)

        for k in range(n_vars):
            dV_d_phys = dV_all[:, k*n_players : (k+1)*n_players]
            for j in range(n_fa):
                ai_j = int(fa_arr[j, 1]); ci_j = int(fa_arr[j, 2]); ni_j = int(fa_arr[j, 3])
                jac[j, k] = (dV_d_phys[ni_j, ai_j] - dV_d_phys[ci_j, ai_j]) * d_phys_d_raw[k]
            
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
        """Numba JIT regularised Newton solver for weak-equality constraints."""
        n_v = alpha_init.shape[0]
        alpha = alpha_init.copy()
        best_alpha = alpha.copy()
        tol = 1e-8
        max_iters = 50
        max_step = 5.0
        n_fa = int(n_fa)
        n_pt = int(n_pt)
        n_aff = int(n_aff)

        res, jac = _residuals_nb_core(
            alpha, canon_probs, canon_action, canon_pass,
            fa_arr, n_fa, pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
            aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
            comm_arr, comm_size, tiers,
            protocol_arr, payoff_array, discounting,
            n_players, n_states, compute_jac=True,
        )
        initial_res = np.max(np.abs(res))
        if initial_res < tol:
            return alpha, True, np.inf

        max_jac = np.max(np.abs(jac))
        if max_jac < 1e-9:
            return alpha, False, 1.0

        best_res = initial_res

        for i in range(max_iters):
            curr_res = np.max(np.abs(res))
            if curr_res < best_res:
                best_res = curr_res
                best_alpha[:] = alpha[:]

            if curr_res < tol:
                return best_alpha, True, initial_res / (curr_res + 1e-300)

            lam = max_jac * 1e-4
            if lam < 1e-9: lam = 1e-9
            for j in range(n_v): jac[j, j] += lam

            try:
                delta = -np.linalg.solve(jac, res)
            except:
                break

            step_sq = 0.0
            for j in range(n_v): step_sq += delta[j] * delta[j]
            if step_sq > max_step * max_step:
                scale = max_step / math.sqrt(step_sq)
                for j in range(n_v): delta[j] *= scale
            
            alpha += delta
            res, jac = _residuals_nb_core(
                alpha, canon_probs, canon_action, canon_pass,
                fa_arr, n_fa, pt_pi_arr, pt_si_arr, pt_widxs_arr, pt_nwidxs_arr, n_pt,
                aff_pi_arr, aff_ci_arr, aff_is_pt_arr, n_aff,
                comm_arr, comm_size, tiers,
                protocol_arr, payoff_array, discounting,
                n_players, n_states, compute_jac=True,
            )
            max_jac = np.max(np.abs(jac))

        final_res = np.max(np.abs(res))
        if final_res < best_res:
            best_res = final_res
            best_alpha[:] = alpha[:]
        
        return best_alpha, (best_res < tol), initial_res / (best_res + 1e-300)
else:
    # Minimal stubs if numba is missing
    def _build_arrays_weak_nb(*args): raise ImportError("numba required")
    def _verify_fast_nb(*args): raise ImportError("numba required")
    def _solve_V_nb(*args): raise ImportError("numba required")
    def _residuals_nb_prep(*args): raise ImportError("numba required")
    def _residuals_nb_p_agg(*args): raise ImportError("numba required")
    def _residuals_nb_residuals(*args): raise ImportError("numba required")
    def _residuals_nb_core(*args): raise ImportError("numba required")
    def _solve_weak_equalities_nb(*args): raise ImportError("numba required")
