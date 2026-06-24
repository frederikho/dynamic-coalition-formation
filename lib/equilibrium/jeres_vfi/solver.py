"""
MIP-VFI solver for farsighted coalition formation.

All functions take a Game as their first argument — no module-level globals.
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, brentq
from scipy.linalg import solve

from lib.equilibrium.jeres_vfi.game import Game, EPS_IND, voters, is_unilateral_exit


# ---------------------------------------------------------------------------
# Per-state MIP
# ---------------------------------------------------------------------------

def solve_state_mip(game: Game, state_idx: int, V: np.ndarray):
    """
    Solve the per-state equilibrium MIP given continuation values V.

    Variables
    ---------
    sigma[i, ns]     ∈ [0,1]  — proposal probability
    z[i, ns]         ∈ {0,1}  — best-response indicator
    alpha[j, ns]     ∈ [0,1]  — acceptance probability (free voters only)
    q[i, ns]         ∈ [0,1]  — product of alphas over voter set
    aux[i, ns, k]    ∈ [0,1]  — McCormick auxiliary for chains of length >2

    Returns (sigma_dict, alpha_dict, q_dict, success_bool).
    """
    state = game.states[state_idx]
    big_M = max(float(np.abs(V).max()) * 2, 1.0)

    # Build (proposer_i, next_state_idx) → voter frozenset
    trans = {}
    for i in range(game.n_players):
        for ns_idx in range(game.n_states):
            trans[(i, ns_idx)] = voters(game, state, game.states[ns_idx], i)

    # Pre-compute acceptance cutoffs: 1.0 / 0.0 / None (free)
    alpha_fixed = {}
    for (i, ns_idx), v_set in trans.items():
        for j in v_set:
            if (j, ns_idx) in alpha_fixed:
                continue
            dv = V[ns_idx, j] - V[state_idx, j]
            if dv > EPS_IND:
                alpha_fixed[(j, ns_idx)] = 1.0
            elif dv < -EPS_IND:
                alpha_fixed[(j, ns_idx)] = 0.0
            else:
                alpha_fixed[(j, ns_idx)] = None  # indifferent — free variable

    # Variable index allocation
    var_idx = 0
    sigma_vars = {}
    z_vars = {}
    alpha_vars = {}
    q_vars = {}
    aux_vars = {}

    for key in trans:
        sigma_vars[key] = var_idx; var_idx += 1
    for key in trans:
        z_vars[key] = var_idx; var_idx += 1
    for (j, ns_idx), val in alpha_fixed.items():
        if val is None:
            alpha_vars[(j, ns_idx)] = var_idx; var_idx += 1
    for key in trans:
        q_vars[key] = var_idx; var_idx += 1
    for (i, ns_idx), v_set in trans.items():
        for m in range(len(sorted(v_set)) - 2):
            aux_vars[(i, ns_idx, m)] = var_idx; var_idx += 1

    N_VARS = var_idx

    def get_alpha_info(j, ns_idx):
        val = alpha_fixed[(j, ns_idx)]
        if val is not None:
            return True, val, None
        return False, None, alpha_vars[(j, ns_idx)]

    ineq_A, ineq_b = [], []
    eq_A, eq_b = [], []

    def add_ineq(row_dict, rhs):
        row = np.zeros(N_VARS)
        for k, v in row_dict.items():
            row[k] = v
        ineq_A.append(row); ineq_b.append(rhs)

    def add_eq(row_dict, rhs):
        row = np.zeros(N_VARS)
        for k, v in row_dict.items():
            row[k] = v
        eq_A.append(row); eq_b.append(rhs)

    # C1. Proposals sum to 1 per proposer
    for i in range(game.n_players):
        add_eq({sigma_vars[(i, ns)]: 1.0 for (i2, ns) in trans if i2 == i}, 1.0)

    # C2. sigma[i,ns] ≤ z[i,ns]
    for key in trans:
        add_ineq({sigma_vars[key]: 1.0, z_vars[key]: -1.0}, 0.0)

    # C3. Exactly one z=1 per proposer
    for i in range(game.n_players):
        add_eq({z_vars[(i, ns)]: 1.0 for (i2, ns) in trans if i2 == i}, 1.0)

    # C4. Best-response big-M (gain formulation)
    for i in range(game.n_players):
        v_cur_i = V[state_idx, i]
        my_ns = [ns for (i2, ns) in trans if i2 == i]
        for ns in my_ns:
            g_ns = V[ns, i] - v_cur_i
            for ns2 in my_ns:
                if ns2 == ns:
                    continue
                g_ns2 = V[ns2, i] - v_cur_i
                add_ineq(
                    {q_vars[(i, ns2)]: g_ns2, q_vars[(i, ns)]: -g_ns, z_vars[(i, ns)]: big_M},
                    big_M,
                )

    # C5. Linearise q = ∏_{j ∈ voters} alpha_j via sequential McCormick chain
    for (i, ns_idx), v_set in trans.items():
        v_list = sorted(v_set)
        k = len(v_list)
        qi = q_vars[(i, ns_idx)]

        if k == 0:
            add_eq({qi: 1.0}, 1.0)

        elif k == 1:
            j = v_list[0]
            fixed, fval, fvar = get_alpha_info(j, ns_idx)
            if fixed:
                add_eq({qi: 1.0}, fval)
            else:
                add_eq({qi: 1.0, fvar: -1.0}, 0.0)

        else:
            # chain[m] = partial product after first m+2 voters (last entry = qi)
            chain = [aux_vars[(i, ns_idx, m)] for m in range(k - 2)] + [qi]

            for step in range(k - 1):
                result_var = chain[step]

                if step == 0:
                    jL = v_list[0]
                    fixedL, fvalL, fvarL = get_alpha_info(jL, ns_idx)
                else:
                    fixedL, fvalL, fvarL = False, None, chain[step - 1]

                jR = v_list[step + 1]
                fixedR, fvalR, fvarR = get_alpha_info(jR, ns_idx)

                if fixedL and fixedR:
                    add_eq({result_var: 1.0}, fvalL * fvalR)
                elif fixedL:
                    add_eq({result_var: 1.0, fvarR: -fvalL}, 0.0)
                elif fixedR:
                    add_eq({result_var: 1.0, fvarL: -fvalR}, 0.0)
                else:
                    # Both free — McCormick relaxation for z = a * b, a,b ∈ [0,1]
                    a_var, b_var = fvarL, fvarR
                    add_ineq({result_var: 1.0, a_var: -1.0}, 0.0)
                    add_ineq({result_var: 1.0, b_var: -1.0}, 0.0)
                    add_ineq({result_var: -1.0, a_var: 1.0, b_var: 1.0}, 1.0)
                    add_ineq({result_var: -1.0}, 0.0)

    lb = np.zeros(N_VARS)
    ub = np.ones(N_VARS)
    integrality = np.zeros(N_VARS)
    for key in trans:
        integrality[z_vars[key]] = 1

    constraints = []
    if ineq_A:
        constraints.append(LinearConstraint(np.array(ineq_A), -np.inf, np.array(ineq_b)))
    if eq_A:
        constraints.append(LinearConstraint(np.array(eq_A), np.array(eq_b), np.array(eq_b)))

    result = milp(np.zeros(N_VARS), constraints=constraints,
                  integrality=integrality, bounds=Bounds(lb, ub))

    if not result.success:
        return None, None, None, False

    x = result.x
    sigma_out = {key: x[sigma_vars[key]] for key in trans}
    q_out = {key: x[q_vars[key]] for key in trans}
    alpha_out = {
        (j, ns_idx): (val if val is not None else x[alpha_vars[(j, ns_idx)]])
        for (j, ns_idx), val in alpha_fixed.items()
    }
    return sigma_out, alpha_out, q_out, True


# ---------------------------------------------------------------------------
# Transition matrix and value function
# ---------------------------------------------------------------------------

def full_transition_matrix(game: Game, sigmas: list, qs: list,
                            proposer_probs=None) -> np.ndarray:
    rho = (np.ones(game.n_players) / game.n_players
           if proposer_probs is None else np.array(proposer_probs))
    T = np.zeros((game.n_states, game.n_states))
    for si in range(game.n_states):
        sigma, q = sigmas[si], qs[si]
        for (i, ns_idx), sig in sigma.items():
            T[si, ns_idx] += rho[i] * sig * q[(i, ns_idx)]
        T[si, si] += max(0.0, 1.0 - T[si].sum())
    return T


def compute_values(game: Game, T: np.ndarray, delta: float,
                   payoffs: np.ndarray | None = None) -> np.ndarray:
    u = game.payoffs if payoffs is None else payoffs
    A = np.eye(game.n_states) - delta * T
    V = np.zeros((game.n_states, game.n_players))
    for p in range(game.n_players):
        V[:, p] = solve(A, (1 - delta) * u[:, p])
    return V


# ---------------------------------------------------------------------------
# VFI inner machinery
# ---------------------------------------------------------------------------

def _vfi_step(game: Game, V: np.ndarray, proposer_probs):
    sigmas = [None] * game.n_states
    alphas = [None] * game.n_states
    qs = [None] * game.n_states
    for si in range(game.n_states):
        sigma, alpha, q, ok = solve_state_mip(game, si, V)
        if not ok:
            warnings.warn(f"MIP infeasible at state {si}")
            sigma = {(i, si): 1.0 for i in range(game.n_players)}
            alpha = {}
            q = {(i, si): 1.0 for i in range(game.n_players)}
        sigmas[si], alphas[si], qs[si] = sigma, alpha, q
    T = full_transition_matrix(game, sigmas, qs, proposer_probs)
    return sigmas, alphas, qs, T


def _resolve_cycle(game: Game, V_cycle: np.ndarray, sigmas_cycle: list,
                   qs_cycle: list, delta: float, proposer_probs,
                   tol: float, verbose: bool):
    """
    Bisect on the interpolated value function to find a mixed-strategy equilibrium
    at the boundary between two pure-strategy phases of the VFI cycle.
    """
    sign_changes = []
    for k in range(len(V_cycle) - 1):
        V_a, V_b = V_cycle[k], V_cycle[k + 1]
        for si in range(game.n_states):
            for i in range(game.n_players):
                for ns in range(game.n_states):
                    if ns == si:
                        continue
                    g_a = V_a[ns, i] - V_a[si, i]
                    g_b = V_b[ns, i] - V_b[si, i]
                    if g_a * g_b < 0:
                        sign_changes.append((i, si, ns, k, g_a, g_b, abs(g_a - g_b)))

    if not sign_changes:
        return None, None, None, None

    sign_changes.sort(key=lambda x: x[6], reverse=True)
    i_dom, si_dom, ns_dom = sign_changes[0][:3]

    all_Vs = list(V_cycle)
    best_pair = None
    best_width = 0.0
    for ka in range(len(all_Vs)):
        for kb in range(len(all_Vs)):
            if ka == kb:
                continue
            ga = all_Vs[ka][ns_dom, i_dom] - all_Vs[ka][si_dom, i_dom]
            gb = all_Vs[kb][ns_dom, i_dom] - all_Vs[kb][si_dom, i_dom]
            if ga * gb < 0 and abs(ga - gb) > best_width:
                best_width = abs(ga - gb)
                best_pair = (ka, kb)

    if best_pair is None:
        return None, None, None, None

    V_a = all_Vs[best_pair[0]]
    V_b = all_Vs[best_pair[1]]

    def gain_at_t(t):
        V_t = (1 - t) * V_a + t * V_b
        return V_t[ns_dom, i_dom] - V_t[si_dom, i_dom]

    g0, g1 = gain_at_t(0.0), gain_at_t(1.0)
    if verbose:
        print(f"  V-bisect: g(0)={g0:+.6f}  g(1)={g1:+.6f}")
    if g0 * g1 > 0:
        return None, None, None, None

    t_star = brentq(gain_at_t, 0.0, 1.0, xtol=1e-10)
    V_star = (1 - t_star) * V_a + t_star * V_b
    if verbose:
        print(f"  t*={t_star:.6f}  gain={gain_at_t(t_star):+.2e}")

    sigmas_star = [None] * game.n_states
    alphas_star = [None] * game.n_states
    qs_star = [None] * game.n_states
    for si in range(game.n_states):
        sigma, alpha, q, ok = solve_state_mip(game, si, V_star)
        if not ok:
            sigma = {(i, si): 1.0 for i in range(game.n_players)}
            alpha, q = {}, {(i, si): 1.0 for i in range(game.n_players)}
        sigmas_star[si] = sigma
        alphas_star[si] = alpha
        qs_star[si] = q

    return V_star, sigmas_star, alphas_star, qs_star


# ---------------------------------------------------------------------------
# Main VFI loop
# ---------------------------------------------------------------------------

def vfi(game: Game, delta: float = 0.95, max_iter: int = 200, tol: float = 1e-8,
        cycle_window: int = 8, proposer_probs=None, verbose: bool = True,
        V_init: np.ndarray | None = None):
    """
    Value Function Iteration with cycle-breaking bisection.

    Parameters
    ----------
    game          : Game instance
    delta         : discount factor
    max_iter      : maximum VFI iterations
    tol           : convergence tolerance on max |ΔV|
    cycle_window  : number of past V's to check for cycles
    proposer_probs: proposal probability per player (uniform if None)
    verbose       : print iteration progress
    V_init        : starting value function; defaults to game.payoffs

    Returns
    -------
    (V, sigmas, alphas, qs) at equilibrium (or at max_iter if not converged)
    """
    V = game.payoffs.copy() if V_init is None else np.array(V_init, dtype=float)
    sigmas = alphas = qs = None
    V_history: list = []
    sigma_history: list = []
    qs_history: list = []

    for iteration in range(max_iter):
        V_old = V.copy()
        sigmas, alphas, qs, T = _vfi_step(game, V, proposer_probs)
        V = compute_values(game, T, delta)
        diff = np.max(np.abs(V - V_old))
        V_history.append(V.copy())
        sigma_history.append([dict(s) for s in sigmas])
        qs_history.append([dict(q) for q in qs])

        if verbose:
            print(f"  Iter {iteration + 1:3d}  |ΔV| = {diff:.2e}")

        if diff < tol:
            if verbose:
                print(f"  Converged in {iteration + 1} iterations.\n")
            return V, sigmas, alphas, qs

        if len(V_history) >= cycle_window:
            for period in range(2, cycle_window):
                if np.max(np.abs(V_history[-1] - V_history[-1 - period])) < tol * 500:
                    if verbose:
                        print(f"  *** Cycle (period {period}) at iter {iteration + 1}."
                              f" Attempting bisection. ***")
                    V_cycle = np.array(V_history[-period:])
                    V_star, sigmas_star, alphas_star, qs_star = _resolve_cycle(
                        game, V_cycle, sigma_history[-period:], qs_history[-period:],
                        delta, proposer_probs, tol, verbose,
                    )
                    if V_star is not None:
                        T_star = full_transition_matrix(game, sigmas_star, qs_star, proposer_probs)
                        V_check = compute_values(game, T_star, delta)
                        final_diff = np.max(np.abs(V_check - V_star))
                        r_ok, _ = verify_responses(game, sigmas_star, alphas_star, qs_star, V_star)
                        p_ok, _ = verify_proposals(game, sigmas_star, alphas_star, qs_star, V_star)
                        if r_ok and p_ok:
                            if verbose:
                                print(f"  Mixed-strategy equilibrium resolved"
                                      f" (|ΔV|={final_diff:.2e}).\n")
                            return V_star, sigmas_star, alphas_star, qs_star
                        if verbose:
                            print(f"  Resolution failed; continuing from V*.")
                        V = V_star
                    else:
                        V = np.mean(np.array(V_history[-period:]), axis=0)

                    V_history.clear(); sigma_history.clear(); qs_history.clear()
                    V_history.append(V.copy())
                    if V_star is not None:
                        sigma_history.append([dict(s) for s in sigmas_star])
                        qs_history.append([dict(q) for q in qs_star])
                    break

    warnings.warn("VFI did not converge within max_iter.")
    return V, sigmas, alphas, qs


# ---------------------------------------------------------------------------
# Equilibrium verification
# ---------------------------------------------------------------------------

def verify_responses(game: Game, sigmas, alphas, qs, V):
    """Check that acceptance strategies are best responses given V."""
    violations = []
    for si in range(game.n_states):
        for (j, ns_idx), a in alphas[si].items():
            dv = V[ns_idx, j] - V[si, j]
            if dv > EPS_IND and not np.isclose(a, 1.0):
                violations.append(
                    f"  RESP: {game.players[j]} should ACCEPT "
                    f"state {si}→{ns_idx} (ΔV={dv:+.6f}) but alpha={a:.4f}"
                )
            elif dv < -EPS_IND and not np.isclose(a, 0.0):
                violations.append(
                    f"  RESP: {game.players[j]} should REJECT "
                    f"state {si}→{ns_idx} (ΔV={dv:+.6f}) but alpha={a:.4f}"
                )
    return len(violations) == 0, violations


def verify_proposals(game: Game, sigmas, alphas, qs, V):
    """Check that proposal strategies are best responses given V and q."""
    violations = []
    for si in range(game.n_states):
        sigma, q = sigmas[si], qs[si]
        for i in range(game.n_players):
            exp_val = {
                ns: q[(i, ns)] * V[ns, i] + (1 - q[(i, ns)]) * V[si, i]
                for (i2, ns) in sigma if i2 == i
            }
            best = max(exp_val.values())
            for (i2, ns), p in sigma.items():
                if i2 != i:
                    continue
                if p > EPS_IND and not np.isclose(exp_val[ns], best, atol=EPS_IND):
                    violations.append(
                        f"  PROP: {game.players[i]} at state {si} proposes"
                        f" suboptimal state {ns}"
                        f" (exp={exp_val[ns]:.6f} vs best={best:.6f})"
                    )
    return len(violations) == 0, violations


def verify_equilibrium(game: Game, sigmas, alphas, qs, V, verbose: bool = True) -> bool:
    r_pass, r_viol = verify_responses(game, sigmas, alphas, qs, V)
    p_pass, p_viol = verify_proposals(game, sigmas, alphas, qs, V)
    if verbose:
        print("=" * 60)
        print("EQUILIBRIUM VERIFICATION")
        print("=" * 60)
        status = "✓ valid" if r_pass else f"✗ {len(r_viol)} violations"
        print(f"  Acceptance strategies: {status}")
        for v in r_viol:
            print(v)
        status = "✓ valid" if p_pass else f"✗ {len(p_viol)} violations"
        print(f"  Proposal strategies:   {status}")
        for v in p_viol:
            print(v)
    return r_pass and p_pass


# ---------------------------------------------------------------------------
# Multi-start search
# ---------------------------------------------------------------------------

def _equilibria_are_distinct(V1: np.ndarray, V2: np.ndarray, atol: float = 1e-2) -> bool:
    return float(np.max(np.abs(V1 - V2))) > atol


def find_equilibria(
    game: Game,
    delta: float,
    proposer_probs=None,
    n_restarts: int = 40,
    tol: float = 1e-6,
    max_iter: int = 300,
    cycle_window: int = 8,
    seed: int = 42,
    verbose: bool = True,
    verbose_each: bool = False,
    dedup_atol: float = 1e-2,
) -> list:
    """
    Search for multiple SMPE via multi-start VFI from randomised initialisations.

    Parameters
    ----------
    game          : Game instance
    delta         : discount factor
    proposer_probs: per-player proposal probabilities (uniform if None)
    n_restarts    : number of random V_init draws (first run always uses game.payoffs)
    tol           : VFI convergence tolerance
    max_iter      : max VFI iterations per run
    cycle_window  : cycle-detection window passed to vfi
    seed          : numpy random seed
    verbose_each  : print VFI progress for every individual run
    dedup_atol    : max-abs V distance to consider two equilibria identical

    Returns
    -------
    list of dicts with keys: V, sigmas, alphas, qs, V_init_tag, verified
    """
    rng = np.random.default_rng(seed)
    payoff_scale = max(float(np.abs(game.payoffs).max()), 1.0)

    V_inits = [("payoffs", game.payoffs.copy())]
    for k in range(n_restarts - 1):
        noise = rng.normal(0, 10 * payoff_scale, size=game.payoffs.shape)
        V_inits.append((f"random-{k}", game.payoffs + noise))

    equilibria = []
    total = len(V_inits)

    for run_idx, (tag, V0) in enumerate(V_inits, start=1):
        if verbose and not verbose_each:
            print(f"  Run {run_idx:3d}/{total}  init={tag}", end="", flush=True)
        try:
            V, sigmas, alphas, qs = vfi(
                game, delta=delta, max_iter=max_iter, tol=tol,
                cycle_window=cycle_window, proposer_probs=proposer_probs,
                V_init=V0, verbose=verbose_each,
            )
            r_ok, _ = verify_responses(game, sigmas, alphas, qs, V)
            p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V)
            verified = r_ok and p_ok
        except Exception as exc:
            if verbose and not verbose_each:
                print(f"  → ERROR: {exc}")
            continue

        if not verified:
            if verbose and not verbose_each:
                print("  → not verified, skipping")
            continue

        is_new = all(_equilibria_are_distinct(V, eq["V"], atol=dedup_atol) for eq in equilibria)
        if is_new:
            equilibria.append(dict(V=V, sigmas=sigmas, alphas=alphas, qs=qs,
                                   V_init_tag=tag, verified=verified))
            if verbose and not verbose_each:
                print(f"  → NEW equilibrium #{len(equilibria)}")
        else:
            if verbose and not verbose_each:
                print("  → duplicate")

    return equilibria
