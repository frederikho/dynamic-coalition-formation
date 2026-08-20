"""
MIP-VFI solver for farsighted coalition formation.

All functions take a Game as their first argument — no module-level globals.
"""

from __future__ import annotations

import warnings
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, brentq, root, least_squares
from scipy.linalg import solve

from lib.equilibrium.jeres_vfi.game import Game, EPS_IND, voters, is_unilateral_exit


# ---------------------------------------------------------------------------
# Per-state MIP
# ---------------------------------------------------------------------------

# HiGHS reports solutions to within its own primal feasibility tolerance, so a
# variable bounded to [0,1] can come back as -1e-14 or 1+1e-14.  Anything this
# close to a bound is indistinguishable from the bound to the solver that
# produced it, so treating it as the bound loses no information.
MIP_BOUND_TOL = 1e-9


def _clean_probability(value: float) -> float:
    """
    Snap a MIP-reported probability onto [0,1], and onto the bounds themselves
    when it is within solver noise of them.

    Both halves are load-bearing, because the framework verifier
    (lib/utils.verify_approvals) tests probabilities EXACTLY:

        if |V_next - V_current| <= atol:      passed = (0. <= p <= 1.)
        elif V_next > V_current:              passed = (p == 1.)
        elif V_next < V_current:              passed = (p == 0.)

    A raw MIP value of -1.0e-14 fails the first branch (it is not >= 0) and the
    third (it is not == 0), so a perfectly good equilibrium is rejected over a
    rounding artefact.  This was observed on kalkuhl_eurndeusa_2035-2100: two
    acceptance cells came back at -1.0011e-14, and the profile was reported
    unsolvable at verify_atol=1e-2 while the same game solved at 1e-10 purely
    because the stricter tolerance routed verification down a different branch.

    Clipping alone is not enough: a value of 1 - 1e-14 satisfies 0 <= p <= 1 but
    still fails `p == 1.`, so near-bound values must be snapped, not merely
    clamped.
    """
    if not np.isfinite(value):
        raise ValueError(f"MIP returned a non-finite probability: {value!r}")
    if value <= MIP_BOUND_TOL:
        return 0.0
    if value >= 1.0 - MIP_BOUND_TOL:
        return 1.0
    return float(value)


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
    # Use framework approval_committees when provided so the MIP matches the
    # framework's effectivity rule instead of Jere's default changed_players rule.
    # Transitions the effectivity rule forbids are dropped from the candidate set
    # entirely, so no variables are allocated for them and the best-response
    # constraints (C4) optimise over permitted targets only.  Constraining them to
    # sigma=0 instead would contradict C1/C3 whenever the unconstrained argmax is a
    # forbidden target, making the whole state infeasible.
    forbidden = game.forbidden_transitions or frozenset()
    trans = {}
    for i in range(game.n_players):
        for ns_idx in range(game.n_states):
            if ns_idx != state_idx and (i, state_idx, ns_idx) in forbidden:
                continue
            if game.approval_committees is not None:
                trans[(i, ns_idx)] = game.approval_committees.get(
                    (i, state_idx, ns_idx), frozenset()
                )
            else:
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

    # C0 is no longer needed: forbidden transitions never enter `trans`, so they have
    # no sigma/z/q variables at all (see the candidate-set construction above).

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
    sigma_out = {key: _clean_probability(x[sigma_vars[key]]) for key in trans}
    q_out = {key: _clean_probability(x[q_vars[key]]) for key in trans}
    alpha_out = {
        (j, ns_idx): (val if val is not None
                      else _clean_probability(x[alpha_vars[(j, ns_idx)]]))
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

def _freed_alphas(game: Game, V: np.ndarray) -> list:
    """(state, target, voter) triples where the voter is exactly indifferent.

    These are the acceptance probabilities the equilibrium conditions leave FREE.
    solve_state_mip allocates them as variables but optimises a ZERO objective, so
    the LP returns them at a bound -- measured 570/570 at 0 on one table.  Nothing
    in the MIP determines them, because a voter's own indifference is insensitive
    to their own probability; what pins them is the requirement that the V they
    generate still carries the tie.
    """
    out = []
    for x in range(game.n_states):
        for i in range(game.n_players):
            for y in range(game.n_states):
                if y == x:
                    continue
                for j in _committee(game, x, y, i):
                    if abs(V[y, j] - V[x, j]) <= EPS_IND and (x, y, j) not in out:
                        out.append((x, y, j))
    return sorted(set(out))


def _step_with_solved_alphas(game: Game, V: np.ndarray, proposer_probs, freed: list,
                             verify_atol: float = EPS_IND):
    """A VFI step that SOLVES the freed acceptance probabilities.

    Ordinary _vfi_step takes whatever the MIP returns for a freed alpha -- a bound,
    since the MIP optimises a zero objective -- which gives the wrong q, hence the
    wrong proposal argmax, hence a V without the tie.  The iteration therefore walks
    away from any V carrying ties, which is why seeding it with a mixed equilibrium's
    own V recovers nothing.

    The proposal support is ENUMERATED rather than recomputed inside the solve.
    Recomputing the argmax while theta moves makes the residual discontinuous -- it
    jumps whenever the argmax switches -- and no root-finder handles that.  Sweeping
    theta first shows only a handful of distinct proposal patterns are reachable from
    a given V (measured: ~3, and ~3.5 even at M=4), so fixing each in turn and solving
    a SMOOTH residual is both cheap and well posed.
    """
    n = len(freed)
    rho = (np.ones(game.n_players) / game.n_players
           if proposer_probs is None else np.asarray(proposer_probs, dtype=float))
    delta = _DELTA_HOLDER[0]

    def alphas_at(theta):
        out = []
        for x in range(game.n_states):
            a = {}
            for i in range(game.n_players):
                for y in range(game.n_states):
                    if y == x:
                        continue
                    for j in _committee(game, x, y, i):
                        a[(j, y)] = 1.0 if V[y, j] - V[x, j] > EPS_IND else 0.0
            out.append(a)
        for (x, y, j), t in zip(freed, theta):
            out[x][(j, y)] = float(np.clip(t, 0.0, 1.0))
        return out

    def proposals_at(qs):
        sigmas = []
        for x in range(game.n_states):
            sig = {}
            for i in range(game.n_players):
                best, best_val = x, 0.0
                for y in range(game.n_states):
                    if y == x:
                        continue
                    val = qs[x][(i, y)] * (V[y, i] - V[x, i])
                    # Compare at the VERIFIER's tolerance, not at machine epsilon.
                    # verify_proposals treats gains within atol as tied and accepts
                    # either choice; a stricter rule here emits only one of the tied
                    # patterns, and if the equilibrium uses the other it is never
                    # generated.  Measured on m1_07: 8 of 15 slots are tied at
                    # q*g = 0, and the strict rule missed the verifying pattern.
                    if val > best_val + verify_atol:
                        best, best_val = y, val
                for y in range(game.n_states):
                    sig[(i, y)] = 1.0 if y == best else 0.0
            sigmas.append(sig)
        return sigmas

    # 1. collect the reachable proposal patterns by sweeping theta
    rng = np.random.default_rng(0)
    probes = [np.full(n, 0.5)]
    probes += [np.array(c, dtype=float) for c in itertools.product((0.0, 1.0), repeat=n)]
    probes += [rng.uniform(0.0, 1.0, n) for _ in range(60)]
    patterns = {}
    for th in probes:
        qs = _rebuild_qs(game, alphas_at(th))
        sig = proposals_at(qs)
        key = tuple(y for x in range(game.n_states) for i in range(game.n_players)
                    for y in range(game.n_states) if sig[x].get((i, y), 0.0) > 0.5)
        patterns.setdefault(key, (sig, np.array(th, dtype=float)))

    # 2. For each pattern, solve exactly as the standalone solver does: fix sigma,
    #    root-find the indifference conditions, verify.
    #
    # One subtlety decides whether that works.  A tie between x and y for voter j frees
    # BOTH directions, so the naive unknown vector has 2 entries per tie -- and their
    # residuals are negatives of one another, leaving the system rank-deficient and flat,
    # which stalls any root-finder (this is why an earlier version fell back to grids).
    # The equilibrium only needs ONE direction free; the other takes the sign rule's
    # value, which at a tie is 0.  Which direction that is is not known a priori, so try
    # both -- 2 per tie, and the number of ties is small.
    pairs = {}
    for k, (x, y, j) in enumerate(freed):
        pairs.setdefault((frozenset((x, y)), j), []).append(k)
    pair_keys = sorted(pairs, key=lambda t: (sorted(t[0]), t[1]))

    def attempt(sigmas, theta):
        alphas = alphas_at(theta)
        qs = _rebuild_qs(game, alphas)
        T = full_transition_matrix(game, sigmas, qs, rho)
        V_new = compute_values(game, T, delta)
        r_ok, _ = verify_responses(game, sigmas, alphas, qs, V_new, atol=verify_atol)
        p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V_new, atol=verify_atol)
        return (sigmas, alphas, qs, T) if (r_ok and p_ok) else None

    for sigmas, th0 in patterns.values():
        for choice in itertools.product(*[range(len(pairs[k])) for k in pair_keys]):
            active = [pairs[k][c] for k, c in zip(pair_keys, choice)]
            m = len(active)
            if m == 0:
                continue

            def embed(t_active):
                # inactive directions keep the sign rule's value (0 at a tie)
                th = np.zeros(n)
                for idx, val in zip(active, t_active):
                    th[idx] = float(np.clip(val, 0.0, 1.0))
                return th

            def residual(t_active, _sig=sigmas, _act=active):
                th = embed(t_active)
                qs = _rebuild_qs(game, alphas_at(th))
                V_new = compute_values(
                    game, full_transition_matrix(game, _sig, qs, rho), delta)
                return np.array([V_new[freed[k][1], freed[k][2]]
                                 - V_new[freed[k][0], freed[k][2]] for k in _act])

            if m == 1:
                # One mixer's own indifference is flat in their own probability, so the
                # residual carries no information -- the standalone scans here too.
                for t in np.linspace(0.0, 1.0, 401):
                    got = attempt(sigmas, embed([t]))
                    if got is not None:
                        return got
            else:
                for z0 in ([np.full(m, 0.5)]
                           + [rng.uniform(0.0, 1.0, m) for _ in range(11)]):
                    try:
                        sol = least_squares(residual, np.clip(z0, 1e-9, 1 - 1e-9),
                                            bounds=(np.zeros(m), np.ones(m)),
                                            method="trf", xtol=1e-15, ftol=1e-15,
                                            gtol=1e-15, max_nfev=1000)
                    except Exception:
                        continue
                    got = attempt(sigmas, embed(sol.x))
                    if got is not None:
                        return got
    return None


_DELTA_HOLDER = [0.95]   # set by vfi(); the step signature predates needing delta


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
                   alphas_cycle: list, qs_cycle: list, delta: float,
                   proposer_probs, tol: float, verbose: bool,
                   verify_atol: float = EPS_IND):
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

    # Primary result from bisection: return V_star along with the derived strategy.
    # The caller will verify against the true V (compute_values from T_star).
    return V_star, sigmas_star, alphas_star, qs_star


def _committee(game: Game, state_idx: int, ns_idx: int, i: int) -> frozenset:
    """Approval committee for proposer i moving state_idx -> ns_idx."""
    if game.approval_committees is not None:
        return game.approval_committees.get((i, state_idx, ns_idx), frozenset())
    return voters(game, game.states[state_idx], game.states[ns_idx], i)


def _rebuild_qs(game: Game, alphas: list) -> list:
    """q[(i,ns)] = product of the committee's acceptance probabilities."""
    qs = []
    for si in range(game.n_states):
        q = {}
        for i in range(game.n_players):
            for ns in range(game.n_states):
                key = (i, ns)
                comm = _committee(game, si, ns, i)
                val = 1.0
                for j in comm:
                    val *= alphas[si].get((j, ns), 1.0)
                q[key] = val
        qs.append(q)
    return qs


def _collect_knobs(game: Game, V_cycle: list, sigmas_cycle: list):
    """
    Ranked candidate mixing knobs, read off the cycle.

    Each knob is one unknown and one indifference equation:

      ('acc', si, ns, j)      voter j indifferent about si -> ns; their acceptance
                              probability is free.  Equation V_j(ns) - V_j(si) = 0.
      ('prop', si, i, y1, y2) proposer i's target flips between y1 and y2 across the
                              cycle.  Equation q(y1)*gain(y1) - q(y2)*gain(y2) = 0.

    Acceptance knobs are deduplicated on the UNORDERED state pair: si -> ns and
    ns -> si express the same indifference for player j, so keeping both makes the
    system singular.  Ranked by how wide the sign change is across the cycle, which
    is the solver's own measure of which transition is driving the cycle.
    """
    acc = {}
    for k in range(len(V_cycle) - 1):
        Va, Vb = V_cycle[k], V_cycle[k + 1]
        for si in range(game.n_states):
            for ns in range(game.n_states):
                if ns == si:
                    continue
                for i in range(game.n_players):
                    # LIVENESS is checked NUMERICALLY in _solve_indifference, not
                    # structurally here.  alpha_j(si->ns) reaches V only through
                    # T[si, ns] += rho_i * sigma_i(si->ns) * q_i(si->ns), so it is
                    # inert wherever sigma = 0 -- but this search ALTERNATES, and a
                    # later pass recomputes sigma as a best response to V(theta), so
                    # sigma can become positive on a transition that was unproposed
                    # in every cycle phase.  Filtering on the cycle's sigmas here
                    # discards knobs that are live against the baseline actually in
                    # use, so the test belongs where that baseline is known.
                    for j in _committee(game, si, ns, i):
                        ga = Va[ns, j] - Va[si, j]
                        gb = Vb[ns, j] - Vb[si, j]
                        if ga * gb >= 0:
                            continue
                        key = (frozenset((si, ns)), j)
                        width = abs(ga - gb)
                        if key not in acc or width > acc[key][0]:
                            acc[key] = (width, ("acc", si, ns, j))

    prop = []
    for si in range(game.n_states):
        for i in range(game.n_players):
            targets = set()
            for sig in sigmas_cycle:
                for (i2, ns), pr in sig[si].items():
                    # ns == si is "stay": its gain V(si)-V(si) is identically zero,
                    # so a knob built on it has a residual that cannot depend on
                    # theta.  Such a knob is not a mixing decision.
                    if i2 == i and ns != si and pr > 0.5:
                        targets.add(ns)
            if len(targets) == 2:
                y1, y2 = sorted(targets)
                spread = max(abs(V[y1, i] - V[y2, i]) for V in V_cycle)
                prop.append((spread, ("prop", si, i, y1, y2)))

    ranked = sorted(list(acc.values()) + prop, key=lambda t: -t[0])
    return [k for _, k in ranked]


def _solve_indifference(game: Game, knobs: list, sigmas0: list, alphas0: list,
                        delta: float, proposer_probs, theta0=None,
                        inner_iter: int = 40):
    """
    Solve the indifference system F(theta) = 0 for the mixing probabilities.

    This is what `_resolve_cycle` does not do.  Bisection interpolates two phases'
    VALUE functions and then asks the MIP for strategies at the midpoint, but the
    equilibrium condition couples the mixing probabilities back to V, so the loop
    has to be closed in STRATEGY space.  Here theta ARE the mixing probabilities,
    V(theta) is recomputed from them at every evaluation, and the residuals are the
    indifference conditions themselves.

    Returns (theta, sigmas, alphas, qs, V) or None.
    """
    if not knobs:
        return None

    def build(theta):
        """Substitute theta into the given baseline strategies and recompute V."""
        sig = [dict(d) for d in sigmas0]
        alp = [dict(d) for d in alphas0]
        for t, kn in zip(theta, knobs):
            if kn[0] == "acc":
                _, si, ns, j = kn
                alp[si][(j, ns)] = float(t)
            else:
                _, si, i, y1, y2 = kn
                for (i2, ns) in list(sig[si].keys()):
                    if i2 == i:
                        sig[si][(i2, ns)] = 0.0
                sig[si][(i, y1)] = float(t)
                sig[si][(i, y2)] = 1.0 - float(t)
        qs = _rebuild_qs(game, alp)
        T = full_transition_matrix(game, sig, qs, proposer_probs)
        return sig, alp, qs, compute_values(game, T, delta)

    def residual(theta):
        # A mixing probability outside the open unit box is not a mixed strategy;
        # push the search back inside rather than letting it wander.
        clipped = np.clip(theta, 1e-12, 1 - 1e-12)
        _, _, qs, V = build(clipped)
        out = []
        for kn in knobs:
            if kn[0] == "acc":
                _, si, ns, j = kn
                out.append(V[ns, j] - V[si, j])
            else:
                _, si, i, y1, y2 = kn
                g1 = V[y1, i] - V[si, i]
                g2 = V[y2, i] - V[si, i]
                out.append(qs[si][(i, y1)] * g1 - qs[si][(i, y2)] * g2)
        return np.array(out) + np.sum(np.abs(theta - clipped))

    if theta0 is None:
        theta0 = np.full(len(knobs), 0.5)
    theta0 = np.asarray(theta0, dtype=float)
    k = len(knobs)

    # WELL-POSEDNESS.  Build the k x k Jacobian by finite differences and require
    # full rank before spending a solve on the support.
    #
    # This replaces a two-point liveness probe, which was not enough on two counts.
    # It compared the whole residual VECTOR, so in a support with one live knob and
    # one inert one the vector still moved: the guard passed, the dead knob stayed
    # in the system, and the Jacobian was singular anyway -- one live knob masks any
    # number of dead ones.  And no probe of that shape can see COLLINEARITY: two
    # knobs can each move the residual while their Jacobian columns are parallel,
    # which is equally unsolvable and equally invisible.  A rank test catches both,
    # and subsumes liveness exactly, since an inert knob is a zero column.
    base = residual(theta0)
    J = np.empty((k, k))
    for m in range(k):
        tp = theta0.copy()
        step = 1e-6 if tp[m] <= 1 - 1e-6 else -1e-6
        tp[m] += step
        J[:, m] = (residual(tp) - base) / step
    if not np.all(np.isfinite(J)) or np.linalg.matrix_rank(J) < k:
        return None

    try:
        sol = root(residual, theta0, method="hybr")
    except Exception:
        return None
    if not sol.success:
        return None
    theta = np.clip(sol.x, 0.0, 1.0)
    if not np.all(np.isfinite(theta)):
        return None
    if np.max(np.abs(residual(theta))) > 1e-9:
        return None
    sig, alp, qs, V = build(theta)
    return theta, sig, alp, qs, V


def _try_indifference_solve(game, V_cycle, sigmas_cycle, alphas_cycle,
                            delta, proposer_probs, verify_atol, verbose,
                            max_support: int = 3, max_candidates: int = 8,
                            outer_passes: int = 12, budget=None):
    """
    Search small mixed supports, solving the indifference system on each.

    A mixed equilibrium generically has a SMALL support: a few players are exactly
    indifferent and everyone else strictly prefers their action.  Making every
    cycling transition indifferent at once is both over-determined and wrong, so we
    try supports of size 1, then 2, then 3, drawn from the highest-ranked knobs, and
    over each phase of the cycle as the pure baseline.
    """
    from itertools import combinations

    # Deterministic given the cycle, so re-running it on every cycle detection
    # just repeats the same work.  The budget is owned by the vfi() call: a
    # module-level dict keyed on id(game) is shared across restarts, and since
    # CPython recycles ids, a later table in a batch can inherit an exhausted one.
    if budget is not None:
        if budget[0] <= 0:
            return None
        budget[0] -= 1

    knobs = _collect_knobs(game, V_cycle, sigmas_cycle)
    if not knobs:
        return None
    knobs = knobs[:max_candidates]

    def best_responses(V, theta, subset):
        """Best responses to V, with the mixed support clamped at theta."""
        sig = [None] * game.n_states
        alp = [None] * game.n_states
        for si in range(game.n_states):
            a_, b_, _c, ok = solve_state_mip(game, si, V)
            if not ok:
                a_ = {(i, si): 1.0 for i in range(game.n_players)}
                b_ = {}
            sig[si], alp[si] = dict(a_), dict(b_)
        for t, kn in zip(theta, subset):
            if kn[0] == "acc":
                _, si, ns, j = kn
                alp[si][(j, ns)] = float(t)
            else:
                _, si, i, y1, y2 = kn
                for (i2, ns) in list(sig[si].keys()):
                    if i2 == i:
                        sig[si][(i2, ns)] = 0.0
                sig[si][(i, y1)] = float(t)
                sig[si][(i, y2)] = 1.0 - float(t)
        return sig, alp

    tried = 0
    for size in range(1, min(max_support, len(knobs)) + 1):
        for subset in combinations(knobs, size):
            for base in range(len(sigmas_cycle)):
                sig0 = sigmas_cycle[base]
                alp0 = alphas_cycle[base]
                theta = None
                # Alternate: solve theta against a fixed pure part, then refresh the
                # pure part as best responses to the resulting V.  Nesting a full
                # clamped VFI inside every residual evaluation is ~200x more
                # expensive and times out; alternating converges in a few passes.
                for _ in range(outer_passes):
                    out = _solve_indifference(game, list(subset), sig0, alp0,
                                              delta, proposer_probs, theta0=theta)
                    tried += 1
                    if out is None:
                        break
                    theta, sig, alp, qs, V = out
                    if not all(1e-9 < t < 1 - 1e-9 for t in theta):
                        break
                    r_ok, _ = verify_responses(game, sig, alp, qs, V, atol=verify_atol)
                    p_ok, _ = verify_proposals(game, sig, alp, qs, V, atol=verify_atol)
                    if r_ok and p_ok:
                        if verbose:
                            print(f"  Mixed equilibrium found: support={subset}, "
                                  f"theta={np.round(theta, 8)} ({tried} solves)\n")
                        return V, sig, alp, qs
                    new_sig, new_alp = best_responses(V, theta, subset)
                    if new_sig == sig0 and new_alp == alp0:
                        break
                    sig0, alp0 = new_sig, new_alp
    if verbose:
        print(f"  indifference search: {tried} supports tried, none verified")
    return None


def _mean_strategy_fallback(
    game: Game, sigmas_cycle: list, alphas_cycle: list, qs_cycle: list,
    delta: float, proposer_probs, verify_atol: float, verbose: bool,
):
    """
    Mean-strategy cycle resolution for games with multiple coupled cycling transitions.

    When a single bisection can't resolve all cycling transitions simultaneously,
    we average the strategies across the VFI cycle.  The mean sigmas/alphas/qs
    represent the time-averaged mixed strategy, which is the correct equilibrium
    when all cycling transitions are near-indifferent.

    Returns (V_mean, sigmas_mean, alphas_mean, qs_mean) if the mean strategy
    passes verify_responses at the given verify_atol, else (None, None, None, None).
    """
    p = len(sigmas_cycle)
    if p == 0:
        return None, None, None, None

    # Average proposal (sigma) probabilities across cycle iterations
    sigmas_mean = [{} for _ in range(game.n_states)]
    for si in range(game.n_states):
        keys = set()
        for k in range(p):
            keys |= sigmas_cycle[k][si].keys()
        for key in keys:
            sigmas_mean[si][key] = sum(
                sigmas_cycle[k][si].get(key, 0.0) for k in range(p)
            ) / p

    # Average approval (alpha) probabilities across cycle iterations
    alphas_mean = [{} for _ in range(game.n_states)]
    for si in range(game.n_states):
        keys = set()
        for k in range(p):
            keys |= alphas_cycle[k][si].keys()
        for key in keys:
            alphas_mean[si][key] = sum(
                alphas_cycle[k][si].get(key, 0.0) for k in range(p)
            ) / p

    # Recompute qs from mean alphas to maintain q = product(alpha_j for j in committee)
    qs_mean = [{} for _ in range(game.n_states)]
    for si in range(game.n_states):
        for (i, ns_idx) in sigmas_mean[si]:
            if game.approval_committees is not None:
                v_set = game.approval_committees.get((i, si, ns_idx), frozenset())
            else:
                from lib.equilibrium.jeres_vfi.game import voters
                v_set = voters(game, game.states[si], game.states[ns_idx], i)
            q_val = 1.0
            for j in sorted(v_set):
                q_val *= alphas_mean[si].get((j, ns_idx), 0.0)
            qs_mean[si][(i, ns_idx)] = q_val

    T_mean = full_transition_matrix(game, sigmas_mean, qs_mean, proposer_probs)
    V_mean = compute_values(game, T_mean, delta)

    r_ok, _ = verify_responses(game, sigmas_mean, alphas_mean, qs_mean, V_mean,
                                atol=verify_atol)
    p_ok, _ = verify_proposals(game, sigmas_mean, alphas_mean, qs_mean, V_mean,
                                atol=verify_atol)
    if r_ok and p_ok:
        if verbose:
            print("  Mean-strategy fallback: verified.")
        return V_mean, sigmas_mean, alphas_mean, qs_mean

    if verbose:
        print("  Mean-strategy fallback: failed verification.")
    return None, None, None, None


# ---------------------------------------------------------------------------
# Main VFI loop
# ---------------------------------------------------------------------------

def vfi(game: Game, delta: float = 0.95, max_iter: int = 200, tol: float = 1e-8,
        cycle_window: int = 8, proposer_probs=None, verbose: bool = True,
        V_init: np.ndarray | None = None, verify_atol: float = EPS_IND,
        mixed_solve: bool = False):
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
    # Budget is per vfi() call: a module-level dict keyed on id(game) is shared
    # across restarts and, since ids are recycled, across tables in a batch.
    mixed_budget = [8]
    sigmas = alphas = qs = None
    V_history: list = []
    sigma_history: list = []
    alpha_history: list = []
    qs_history: list = []

    _DELTA_HOLDER[0] = delta
    _mixed_tries = [1]
    for iteration in range(max_iter):
        V_old = V.copy()
        sigmas, alphas, qs, T = _vfi_step(game, V, proposer_probs)

        # Where the current V has EXACT ties, the MIP's freed alphas came back on a
        # bound (zero objective), which yields the wrong q, the wrong proposal
        # argmax, and a V without the tie -- so the iteration walks off any V that
        # carries one.  Solve those alphas instead, keeping the tie intact.
        if mixed_solve and _mixed_tries[0] > 0:
            freed = _freed_alphas(game, V)
            if freed:
                # Budgeted: a failed attempt costs ~1200 candidate evaluations, and
                # retrying it at every one of max_iter iterations turns a millisecond
                # solve into hours.  A few attempts are enough -- if the current V
                # carries the right ties the first one finds it.
                _mixed_tries[0] -= 1
                got = _step_with_solved_alphas(game, V, proposer_probs, freed,
                                               verify_atol=verify_atol)
                if got is not None:
                    # _step_with_solved_alphas only returns a profile it has already
                    # VERIFIED, so this IS an equilibrium -- return it.  Continuing the
                    # iteration recomputes V from it and discards it.
                    sigmas, alphas, qs, T = got
                    if verbose:
                        print(f"  Mixed equilibrium found at iteration {iteration} "
                              f"({len(freed)} freed alphas).\n")
                    return compute_values(game, T, delta), sigmas, alphas, qs
        V = compute_values(game, T, delta)
        diff = np.max(np.abs(V - V_old))
        V_history.append(V.copy())
        sigma_history.append([dict(s) for s in sigmas])
        alpha_history.append([dict(a) for a in alphas])
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
                        game, V_cycle,
                        sigma_history[-period:],
                        alpha_history[-period:],
                        qs_history[-period:],
                        delta, proposer_probs, tol, verbose,
                        verify_atol=verify_atol,
                    )
                    if V_star is not None:
                        T_star = full_transition_matrix(game, sigmas_star, qs_star, proposer_probs)
                        V_check = compute_values(game, T_star, delta)
                        final_diff = np.max(np.abs(V_check - V_star))
                        # Verify against V_check (the true V from T_star) using verify_atol
                        # so that near-indifferent transitions (|ΔV| ≤ verify_atol) don't
                        # cause spurious failures.  This must match the framework verifier's
                        # tolerance to avoid false positives.
                        r_ok, _ = verify_responses(game, sigmas_star, alphas_star, qs_star,
                                                    V_check, atol=verify_atol)
                        p_ok, _ = verify_proposals(game, sigmas_star, alphas_star, qs_star,
                                                    V_check, atol=verify_atol)
                        if r_ok and p_ok:
                            if verbose:
                                print(f"  Mixed-strategy equilibrium resolved"
                                      f" (|ΔV|={final_diff:.2e}).\n")
                            return V_check, sigmas_star, alphas_star, qs_star

                        # Bisection fixed the dominant cycle but left coupled transitions
                        # inconsistent.  Try the mean-strategy fallback: average all
                        # strategies over the cycle to represent the time-averaged mixing.
                        if mixed_solve:
                            got = _try_indifference_solve(
                                game, V_history[-period:], sigma_history[-period:],
                                alpha_history[-period:], delta, proposer_probs,
                                verify_atol, verbose, budget=mixed_budget,
                            )
                            if got is not None:
                                return got

                        if verbose:
                            print(f"  Bisection left residual inconsistency; "
                                  f"trying mean-strategy fallback.")
                        V_ms, s_ms, a_ms, q_ms = _mean_strategy_fallback(
                            game, sigma_history[-period:], alpha_history[-period:],
                            qs_history[-period:], delta, proposer_probs, verify_atol, verbose,
                        )
                        if V_ms is not None:
                            if verbose:
                                print(f"  Mean-strategy equilibrium accepted.\n")
                            return V_ms, s_ms, a_ms, q_ms

                        if verbose:
                            print(f"  All resolutions failed; continuing from V_check.")
                        V = V_check
                    else:
                        if mixed_solve:
                            got = _try_indifference_solve(
                                game, V_history[-period:], sigma_history[-period:],
                                alpha_history[-period:], delta, proposer_probs,
                                verify_atol, verbose, budget=mixed_budget,
                            )
                            if got is not None:
                                return got

                        # No sign changes found; try mean strategy directly.
                        V_ms, s_ms, a_ms, q_ms = _mean_strategy_fallback(
                            game, sigma_history[-period:], alpha_history[-period:],
                            qs_history[-period:], delta, proposer_probs, verify_atol, verbose,
                        )
                        if V_ms is not None:
                            if verbose:
                                print(f"  Mean-strategy equilibrium accepted (no sign changes).\n")
                            return V_ms, s_ms, a_ms, q_ms
                        V = np.mean(np.array(V_history[-period:]), axis=0)

                    V_history.clear()
                    sigma_history.clear()
                    alpha_history.clear()
                    qs_history.clear()
                    V_history.append(V.copy())
                    if V_star is not None:
                        sigma_history.append([dict(s) for s in sigmas_star])
                        alpha_history.append([dict(a) for a in alphas_star])
                        qs_history.append([dict(q) for q in qs_star])
                    break

    warnings.warn("VFI did not converge within max_iter.")
    return V, sigmas, alphas, qs


# ---------------------------------------------------------------------------
# Equilibrium verification
# ---------------------------------------------------------------------------

def verify_responses(game: Game, sigmas, alphas, qs, V, atol: float = EPS_IND):
    """Check that acceptance strategies are best responses given V.

    atol : absolute tolerance for treating a voter as indifferent (default EPS_IND).
           Pass a looser value (e.g. 1e-4) when verifying near-flat payoffs where V
           was computed to limited precision; matches the framework's verify_atol.
    """
    violations = []
    for si in range(game.n_states):
        for (j, ns_idx), a in alphas[si].items():
            dv = V[ns_idx, j] - V[si, j]
            if abs(dv) <= atol:
                continue  # indifferent: any alpha is consistent
            if dv > atol and not np.isclose(a, 1.0, atol=atol):
                violations.append(
                    f"  RESP: {game.players[j]} should ACCEPT "
                    f"state {si}→{ns_idx} (ΔV={dv:+.6f}) but alpha={a:.4f}"
                )
            elif dv < -atol and not np.isclose(a, 0.0, atol=atol):
                violations.append(
                    f"  RESP: {game.players[j]} should REJECT "
                    f"state {si}→{ns_idx} (ΔV={dv:+.6f}) but alpha={a:.4f}"
                )
    return len(violations) == 0, violations


def verify_proposals(game: Game, sigmas, alphas, qs, V, atol: float = EPS_IND):
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
                if p > EPS_IND and not np.isclose(exp_val[ns], best, atol=atol):
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
    verify_atol: float = EPS_IND,
    restart_scaling: str = "spread",
    extra_v_inits: list | None = None,
    mixed_solve: bool = False,
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
    extra_v_inits : optional [(tag, V array)] tried BEFORE the random restarts.
                    Random restarts draw from a diffuse ball around game.payoffs
                    and so never land on a strategically coherent point; an
                    equilibrium V of a NEARBY game (e.g. the same table at an
                    adjacent delta) is exactly such a point, and reaching it is
                    not implied by the inertness of random multi-start.

    Returns
    -------
    list of dicts with keys: V, sigmas, alphas, qs, V_init_tag, verified
    """
    rng = np.random.default_rng(seed)

    # Perturb each player by the SPREAD of their payoffs across states, not by the
    # LEVEL of those payoffs.
    #
    # Value functions are convex combinations of a player's own static payoffs, so
    # V_i always lies inside [min_x u_i(x), max_x u_i(x)].  The spread is therefore
    # the whole region a restart could usefully explore; the level says nothing
    # about it.  Scaling by the level is harmless when payoffs straddle zero (in
    # the 2021 paper's example, level and spread are both ~118, giving a sane 10x
    # exploration radius), but it silently destroys the multi-start on payoffs with
    # a large level and a tiny spread.  RICE welfare payoffs sit near -13 with a
    # spread near 1e-3, so the old rule perturbed V by ~130 across a meaningful
    # range of 0.001 -- a radius 30,000x too large.  Every random restart landed
    # nowhere near the value function, and the solver degenerated to a single
    # deterministic start from game.payoffs.
    #
    # Per-player rather than global: one player's spread can exceed another's by a
    # factor of 50 within the same game, and a shared radius would over-perturb the
    # narrow player while under-perturbing the wide one.
    spread = game.payoffs.max(axis=0) - game.payoffs.min(axis=0)
    if not np.any(spread > 0):
        raise ValueError(
            "Every player has identical payoffs in all states, so there is no "
            "value-function structure to search and no scale for restarts."
        )

    if restart_scaling == "spread":
        # A flat player has no range to explore; leave their V_init unperturbed
        # rather than substituting an arbitrary radius.
        restart_scale = 10.0 * spread
    elif restart_scaling == "level":
        # Historical behaviour, kept so the two can be benchmarked against each
        # other without editing code. Correct only when level ~ spread.
        restart_scale = 10.0 * max(float(np.abs(game.payoffs).max()), 1.0)
    else:
        raise ValueError(
            f"restart_scaling must be 'spread' or 'level', got {restart_scaling!r}"
        )

    V_inits = [("payoffs", game.payoffs.copy())]
    for tag, V0 in (extra_v_inits or []):
        V0 = np.asarray(V0, dtype=float)
        if V0.shape != game.payoffs.shape:
            raise ValueError(
                f"warm-start V {tag!r} has shape {V0.shape}, expected "
                f"{game.payoffs.shape} (n_states, n_players) in Jere state order"
            )
        V_inits.append((tag, V0))
    for k in range(n_restarts - 1):
        noise = rng.normal(0, 1.0, size=game.payoffs.shape) * restart_scale
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
                V_init=V0, verbose=verbose_each, verify_atol=verify_atol,
                mixed_solve=mixed_solve,
            )
            r_ok, _ = verify_responses(game, sigmas, alphas, qs, V, atol=verify_atol)
            p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V, atol=verify_atol)
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
