"""
Farsighted Coalition Formation MIP (Ray-Vohra style)
=====================================================

Committee rule:
  1. Unilateral exit: proposer i splits off as a singleton — free, no consent.
  2. All other proposals: voters = changed_players(state, next_state) \\ {i}.

Strategy objects:
  sigma[i, next_state_idx]  : prob proposer i proposes that next-state
  alpha[j, next_state_idx]  : prob voter j accepts transition to next-state
  q[i, next_state_idx]      : product of alpha over voters(state, next_state, i)

Solved via Value Function Iteration (outer loop) with a per-state MIP
(inner loop) using scipy.milp.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.linalg import solve
import warnings

Coalition = frozenset[str]
State = tuple[Coalition]

# ===========================================================================
# USER CONFIGURATION  ← only edit this block
# ===========================================================================




# Optional: restrict the state space to a custom subset of partitions.
# Each entry is a tuple of frozensets, e.g.:
#   (frozenset({0,1,2}),)                           -- grand coalition
#   (frozenset({0,1}), frozenset({2}))              -- pair + singleton
#   (frozenset({0}), frozenset({1}), frozenset({2}))-- all singletons
# Leave as None to use all Bell-number partitions (default behaviour).
#
# Transitions are automatically restricted to next-states that exist in
# your list. Unilateral exit is free only if the result state is in the
# list; otherwise it is treated as a regular proposal requiring consent.
STATES_OVERRIDE = None

# ===========================================================================
# 1.  COMBINATORIAL SETUP  (derived from PLAYERS — do not edit)
# ===========================================================================
# Set SEARCH_MODE = True to run multi-start equilibrium search.
# Set SEARCH_MODE = False for a single solve.

# SOLUTION CONFIG PARAMS
########################
EPS_IND = 1e-12  # indifference thresh for numerical stability
SEARCH_MODE = True
RESCALE_PAYOFFS = False
MAX_ITER = 1000
N_RESTARTS = 250
DELTA = 0.99   # discount factor


# GAME DEFINITION
#################



PLAYERS = ["CHN", "NDE", "USA"]
PROPOSER_PROBS = [1/3, 1/3, 1/3]

payoffs = {
    #  state_label       CHN            NDE            USA
    "()":          [-99.979961,  -186.659000,  -38.747981],
    "(CHNNDE)":    [-99.906491,  -186.278502,  -38.718375],
    "(CHNUSA)":    [-99.901092,  -186.337493,  -38.717686],
    "(NDEUSA)":    [-99.896071,  -186.266076,  -38.729995],
    "(CHNNDEUSA)": [-99.901493,  -186.285119,  -38.719403],
}





PLAYERS = ["W", "T", "C"]
PROPOSER_PROBS = [1/3, 1/3, 1/3]
payoffs = {
#  state_label  W         T         C
    "()":      [0.00,     0.00,     0.00],
    "(TC)":    [55.6875,  14.4375,  0.6875],
    "(WC)":    [107.25,   9.75,    -22.75],
    "(WT)":    [118.1875, 1.9375,  -36.8125],
    "(WTC)":   [98.22222222222221,  13.222222222222221,  -15.111111111111114],
}








####################### NO EDITS BELOW ############


N_PLAYERS = len(PLAYERS)

def pname(coalition):
    return "{" + ",".join(PLAYERS[p] for p in sorted(coalition)) + "}"


def partition_name(partition):
    return " + ".join(pname(c) for c in sorted(partition, key=lambda s: sorted(s)))

def gen_partitions(players):
    players = list(players)
    if not players:
        yield ()
        return
    first, rest = players[0], players[1:]
    for p in gen_partitions(rest):
        yield (frozenset([first]),) + p
        for i, block in enumerate(p):
            new_p = list(p)
            new_p[i] = block | frozenset([first])
            yield tuple(new_p)

def canon_partition(p):
    return tuple(sorted((frozenset(c) for c in p), key=lambda s: sorted(s)))

def _validate_states(states):
    """Check every state covers each player exactly once."""
    player_set = frozenset(range(N_PLAYERS))
    for s in states:
        covered = frozenset().union(*s)
        if covered != player_set:
            raise ValueError(
                f"State {partition_name(s)} does not cover all players. "
                f"Missing: {player_set - covered}, Extra: {covered - player_set}")
        if sum(len(c) for c in s) != len(player_set):
            raise ValueError(f"State {partition_name(s)} has overlapping blocks.")

if STATES_OVERRIDE is not None:
    STATES = [canon_partition(s) for s in STATES_OVERRIDE]
    _validate_states(STATES)
else:
    STATES = sorted(
        set(canon_partition(p) for p in gen_partitions(range(N_PLAYERS))),
        key=lambda p: (len(p), sorted(len(c) for c in p))
    )
    _validate_states(STATES)


def rescale_payoffs(payoffs, target_spread=10.0):
    """
    Affine rescaling per player column: subtract mean, scale max-abs to
    target_spread.  Preserves all ordinal rankings and equilibrium strategies.
    Returns (scaled_matrix, list_of_(mean, raw_spread) per player).
    """
    scaled = payoffs.copy().astype(float)
    info = []
    for i in range(payoffs.shape[1]):
        col = scaled[:, i]
        m   = col.mean()
        col -= m
        sp  = np.abs(col).max()
        if sp > 1e-12:
            col *= target_spread / sp
        scaled[:, i] = col
        info.append((m, sp))
    return scaled, info



def canon(p):
    return tuple(sorted((frozenset(c) for c in p), key=lambda s: sorted(s)))

def parse_state_label(label, player_names):
    """
    "(CHNNDE)" with players ["CHN","NDE","USA"]
    -> coalition {0,1} and singletons {2}
    -> partition (frozenset({0,1}), frozenset({2}))
 
    "()" -> all singletons
    """
    inner = label.strip("()")
    if not inner:
        return tuple(frozenset({i}) for i in range(len(player_names)))
 
    in_coal = set()
    s = inner
    while s:
        matched = False
        for idx, name in enumerate(player_names):
            if s.startswith(name):
                in_coal.add(idx)
                s = s[len(name):]
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse '{s}' in label '{label}'")
 
    out_of_coal = set(range(len(player_names))) - in_coal
    parts = [frozenset(in_coal)] + [frozenset({p}) for p in sorted(out_of_coal)]
    return tuple(sorted(parts, key=lambda c: sorted(c)))
 

def canon(p):
    return tuple(sorted((frozenset(c) for c in p), key=lambda s: sorted(s)))
 
label_to_canon = {lbl: canon(parse_state_label(lbl, PLAYERS))
                  for lbl in payoffs}
canon_to_label = {v: k for k, v in label_to_canon.items()}

PAYOFFS = np.array([payoffs[canon_to_label[s]] for s in STATES])

if RESCALE_PAYOFFS:
    PAYOFFS, _ = rescale_payoffs(PAYOFFS)



N_STATES  = len(STATES)
STATE_IDX = {s: i for i, s in enumerate(STATES)}

print("=" * 60)
print(f"N={N_PLAYERS} players: " + ", ".join(PLAYERS))
print(f"{N_STATES} states (partitions):")
for idx, s in enumerate(STATES):
    print(f"  [{idx}]  {partition_name(s)}")
print()

# ===========================================================================
# 2.  GAME MECHANICS
# ===========================================================================

def player_block(state, player):
    for c in state:
        if player in c:
            return c
    raise ValueError(f"Player {player} not in state {state}")

def changed_players(state, next_state):
    return frozenset(
        p for p in range(N_PLAYERS)
        if player_block(state, p) != player_block(next_state, p)
    )

def apply_proposal(state, proposed_coalition):
    proposed = frozenset(proposed_coalition)
    new_partition = [frozenset(b - proposed) for b in state if b - proposed]
    new_partition.append(proposed)
    return canon_partition(new_partition)

def is_unilateral_exit(state, next_state, proposer):
    """
    True iff next_state is exactly what results from proposer leaving their
    coalition alone AND that result is an allowed state.
    If the exit-result partition is not in the state list (e.g. it was
    excluded by game rules), the exit is not free — the proposer must obtain
    consent like any other proposal.
    """
    coal = player_block(state, proposer)
    if len(coal) == 1:
        return False
    exit_result = apply_proposal(state, frozenset([proposer]))
    return next_state == exit_result and exit_result in STATE_IDX

def voters(state, next_state, proposer):
    if is_unilateral_exit(state, next_state, proposer):
        return frozenset()
    return changed_players(state, next_state) - {proposer}

def reachable_next_states(state, proposer):
    """All states in the allowed state list, including status quo."""
    return list(range(N_STATES))



# ===========================================================================
# 4.  PER-STATE MIP
# ===========================================================================




def solve_state_mip(state_idx, V):
    """
    Solve per-state equilibrium MIP given continuation values V.

    Variables
    ---------
    sigma[i, ns]   ∈ [0,1]  — proposal probability
    z[i, ns]       ∈ {0,1}  — best-response indicator
    alpha[j, ns]   ∈ [0,1]  — acceptance probability (free voters only)
    q[i, ns]       ∈ [0,1]  — product of alphas over voter set
    aux[i, ns, k]  ∈ [0,1]  — McCormick auxiliary for sequential product
                               (needed when |voters| > 2)

    Returns sigma, alpha, q dicts and success flag.
    """
    state   = STATES[state_idx]
    big_M   = max(float(np.abs(V).max()) * 2, 1.0)

    # Build (proposer, next_state_idx) → voter frozenset
    trans = {}
    for i in range(N_PLAYERS):
        for ns_idx in reachable_next_states(state, i):
            trans[(i, ns_idx)] = voters(state, STATES[ns_idx], i)

    # Acceptance cutoffs
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
                alpha_fixed[(j, ns_idx)] = None  # free

    # Variable index allocation
    var_idx    = 0
    sigma_vars = {}
    z_vars     = {}
    alpha_vars = {}
    q_vars     = {}
    aux_vars   = {}   # (i, ns_idx, k) for McCormick chain

    for key in trans:
        sigma_vars[key] = var_idx; var_idx += 1
    for key in trans:
        z_vars[key]     = var_idx; var_idx += 1
    for (j, ns_idx), val in alpha_fixed.items():
        if val is None:
            alpha_vars[(j, ns_idx)] = var_idx; var_idx += 1
    for key in trans:
        q_vars[key]     = var_idx; var_idx += 1
    # McCormick auxiliaries: for voter list of length k, need k-2 auxiliaries
    # aux[i, ns, m] = product of first (m+2) voters' alphas, m = 0…k-3
    for (i, ns_idx), v_set in trans.items():
        v_list = sorted(v_set)
        for m in range(len(v_list) - 2):
            aux_vars[(i, ns_idx, m)] = var_idx; var_idx += 1

    N_VARS = var_idx

    def get_alpha_info(j, ns_idx):
        """Returns (is_fixed, fixed_val_or_None, var_idx_or_None)."""
        val = alpha_fixed[(j, ns_idx)]
        if val is not None:
            return True, val, None
        return False, None, alpha_vars[(j, ns_idx)]

    ineq_A, ineq_b = [], []
    eq_A,   eq_b   = [], []

    def add_ineq(row_dict, rhs):
        row = np.zeros(N_VARS)
        for k, v in row_dict.items(): row[k] = v
        ineq_A.append(row); ineq_b.append(rhs)

    def add_eq(row_dict, rhs):
        row = np.zeros(N_VARS)
        for k, v in row_dict.items(): row[k] = v
        eq_A.append(row); eq_b.append(rhs)

    # C1. Proposals sum to 1 per proposer
    for i in range(N_PLAYERS):
        add_eq({sigma_vars[(i, ns)]: 1.0
                for (i2, ns) in trans if i2 == i}, 1.0)

    # C2. sigma[i,ns] ≤ z[i,ns]
    for key in trans:
        add_ineq({sigma_vars[key]: 1.0, z_vars[key]: -1.0}, 0.0)

    # C3. Exactly one z=1 per proposer
    for i in range(N_PLAYERS):
        add_eq({z_vars[(i, ns)]: 1.0
                for (i2, ns) in trans if i2 == i}, 1.0)

    # C4. Best-response big-M (gain formulation)
    for i in range(N_PLAYERS):
        v_cur_i = V[state_idx, i]
        my_ns   = [ns for (i2, ns) in trans if i2 == i]
        for ns in my_ns:
            g_ns = V[ns, i] - v_cur_i
            for ns2 in my_ns:
                if ns2 == ns:
                    continue
                g_ns2 = V[ns2, i] - v_cur_i
                add_ineq({q_vars[(i, ns2)]:  g_ns2,
                          q_vars[(i, ns)]:  -g_ns,
                          z_vars[(i, ns)]:   big_M},
                         big_M)

    # C5. Linearise q = ∏_{j ∈ voters} alpha_j
    # For k voters we use a sequential McCormick chain:
    #   k=0 : q = 1
    #   k=1 : q = alpha_1
    #   k=2 : q = alpha_1 * alpha_2  (one McCormick)
    #   k≥3 : introduce w_m = (product of first m+2 voters)
    #          w_0 = alpha_1 * alpha_2          (McCormick)
    #          w_1 = w_0    * alpha_3           (McCormick)
    #          …
    #          w_{k-3} = w_{k-4} * alpha_{k-1}  (McCormick)
    #          q       = w_{k-3} * alpha_k       (McCormick)
    for (i, ns_idx), v_set in trans.items():
        v_list = sorted(v_set)   # deterministic order
        k      = len(v_list)
        qi     = q_vars[(i, ns_idx)]

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
            # Build chain: chain[m] is the variable index for the partial product
            # after including voters v_list[0..m+1]
            # chain[-1] is qi (the final product)
            chain = []  # chain[m] = var index of w_m for m = 0 .. k-3
            for m in range(k - 2):
                chain.append(aux_vars[(i, ns_idx, m)])
            chain.append(qi)   # last link is qi itself

            for step in range(k - 1):
                # Multiply chain[step-1] (or alpha[v_list[0]] when step=0)
                # by alpha[v_list[step+1]]
                result_var = chain[step]

                # Left operand
                if step == 0:
                    jL = v_list[0]
                    fixedL, fvalL, fvarL = get_alpha_info(jL, ns_idx)
                else:
                    fixedL, fvalL, fvarL = False, None, chain[step - 1]

                # Right operand
                jR = v_list[step + 1]
                fixedR, fvalR, fvarR = get_alpha_info(jR, ns_idx)

                if fixedL and fixedR:
                    add_eq({result_var: 1.0}, fvalL * fvalR)
                elif fixedL:
                    add_eq({result_var: 1.0, fvarR: -fvalL}, 0.0)
                elif fixedR:
                    add_eq({result_var: 1.0, fvarL: -fvalR}, 0.0)
                else:
                    # Both free — McCormick for z = a * b, a,b ∈ [0,1]
                    a_var = fvarL
                    b_var = fvarR
                    add_ineq({result_var:  1.0, a_var: -1.0},           0.0)
                    add_ineq({result_var:  1.0, b_var: -1.0},           0.0)
                    add_ineq({result_var: -1.0, a_var:  1.0, b_var: 1.0}, 1.0)
                    add_ineq({result_var: -1.0},                         0.0)

    # Assemble and solve
    lb = np.zeros(N_VARS)
    ub = np.ones(N_VARS)
    integrality = np.zeros(N_VARS)
    for key in trans:
        integrality[z_vars[key]] = 1

    constraints = []
    if ineq_A:
        constraints.append(LinearConstraint(
            np.array(ineq_A), -np.inf, np.array(ineq_b)))
    if eq_A:
        constraints.append(LinearConstraint(
            np.array(eq_A), np.array(eq_b), np.array(eq_b)))

    result = milp(np.zeros(N_VARS),
                  constraints=constraints,
                  integrality=integrality,
                  bounds=Bounds(lb, ub))

    if not result.success:
        return None, None, None, False

    x         = result.x
    sigma_out = {key: x[sigma_vars[key]] for key in trans}
    q_out     = {key: x[q_vars[key]]     for key in trans}
    alpha_out = {}
    for (j, ns_idx), val in alpha_fixed.items():
        alpha_out[(j, ns_idx)] = val if val is not None else x[alpha_vars[(j, ns_idx)]]

    return sigma_out, alpha_out, q_out, True


# ===========================================================================
# 5.  TRANSITION MATRIX
# ===========================================================================

def full_transition_matrix(all_sigmas, all_qs, proposer_probs=None):
    rho = (np.ones(N_PLAYERS) / N_PLAYERS
           if proposer_probs is None else np.array(proposer_probs))
    T   = np.zeros((N_STATES, N_STATES))
    for si in range(N_STATES):
        sigma, q = all_sigmas[si], all_qs[si]
        for (i, ns_idx), sig in sigma.items():
            T[si, ns_idx] += rho[i] * sig * q[(i, ns_idx)]
        T[si, si] += max(0.0, 1.0 - T[si].sum())
    return T

# ===========================================================================
# 6.  VALUE FUNCTION
# ===========================================================================

def compute_values(T, delta):
    A = np.eye(N_STATES) - delta * T
    V = np.zeros((N_STATES, N_PLAYERS))
    for p in range(N_PLAYERS):
        V[:, p] = solve(A, (1 - delta) * PAYOFFS[:, p])
    return V

# ===========================================================================
# 7.  VALUE FUNCTION ITERATION
# ===========================================================================

def _vfi_step(V, proposer_probs):
    sigmas = [None] * N_STATES
    alphas = [None] * N_STATES
    qs     = [None] * N_STATES
    for si in range(N_STATES):
        sigma, alpha, q, ok = solve_state_mip(si, V)
        if not ok:
            warnings.warn(f"MIP infeasible at state {si}")
            sigma = {(i, si): 1.0 for i in range(N_PLAYERS)}
            alpha = {}
            q     = {(i, si): 1.0 for i in range(N_PLAYERS)}
        sigmas[si], alphas[si], qs[si] = sigma, alpha, q
    T = full_transition_matrix(sigmas, qs, proposer_probs)
    return sigmas, alphas, qs, T


def _resolve_cycle(V_cycle, sigmas_cycle, qs_cycle, delta, proposer_probs, tol, verbose):
    """
    Resolve a VFI cycle by bisecting on the interpolated value function.

    Finds V* = (1-t*)*V_A + t*V_B where the dominant oscillating gain crosses
    zero, then runs the full MIP at V* to recover the correct mixed strategies.
    """
    from scipy.optimize import brentq

    # Collect all sign-changing gains across consecutive V pairs.
    # No T-contribution filter: acceptance-driven oscillations have sigma=0
    # for the oscillating player (they are voters, not proposers), so the
    # filter would incorrectly remove them. The V-bisect is safe without it.
    sign_changes = []
    for k in range(len(V_cycle) - 1):
        V_a, V_b = V_cycle[k], V_cycle[k + 1]
        for si in range(N_STATES):
            for i in range(N_PLAYERS):
                for ns in range(N_STATES):
                    if ns == si:
                        continue
                    g_a = V_a[ns, i] - V_a[si, i]
                    g_b = V_b[ns, i] - V_b[si, i]
                    if g_a * g_b < 0:
                        sign_changes.append((i, si, ns, k, g_a, g_b,
                                             abs(g_a - g_b)))

    if not sign_changes:
        return None, None, None, None

    # Pick the widest bracket (most numerically stable to bisect)
    sign_changes.sort(key=lambda x: x[6], reverse=True)
    i_dom, si_dom, ns_dom, k_br, g_a, g_b, _ = sign_changes[0]

    if verbose:
        print(f"  Oscillating: {PLAYERS[i_dom]}"
              f" at {partition_name(STATES[si_dom])}"
              f" → {partition_name(STATES[ns_dom])}"
              f"  gain: {g_a:+.4f} → {g_b:+.4f}")

    # Try all V pairs from cycle, find widest bracket for the dominant gain
    all_Vs     = list(V_cycle)
    best_pair  = None
    best_width = 0.0
    for ka in range(len(all_Vs)):
        for kb in range(len(all_Vs)):
            if ka == kb:
                continue
            ga = all_Vs[ka][ns_dom, i_dom] - all_Vs[ka][si_dom, i_dom]
            gb = all_Vs[kb][ns_dom, i_dom] - all_Vs[kb][si_dom, i_dom]
            if ga * gb < 0 and abs(ga - gb) > best_width:
                best_width = abs(ga - gb)
                best_pair  = (ka, kb)

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

    # Run full MIP at V_star — indifferent player(s) will now mix
    sigmas_star = [None] * N_STATES
    alphas_star = [None] * N_STATES
    qs_star     = [None] * N_STATES
    for si in range(N_STATES):
        sigma, alpha, q, ok = solve_state_mip(si, V_star)
        if not ok:
            sigma = {(i, si): 1.0 for i in range(N_PLAYERS)}
            alpha, q = {}, {(i, si): 1.0 for i in range(N_PLAYERS)}
        sigmas_star[si] = sigma
        alphas_star[si] = alpha
        qs_star[si]     = q

    return V_star, sigmas_star, alphas_star, qs_star


def vfi(delta=0.95, max_iter=200, tol=1e-8, cycle_window=8,
        proposer_probs=None, verbose=True, V_init=None):
    """
    Value Function Iteration with cycle-breaking.

    Parameters
    ----------
    V_init : array (N_STATES, N_PLAYERS) or None
        Starting value function. Defaults to PAYOFFS (myopic).
        Pass a random matrix to explore different basins of attraction.

    Returns V, sigmas, alphas, qs at equilibrium.
    """
    V             = PAYOFFS.copy().astype(float) if V_init is None else np.array(V_init, dtype=float)
    sigmas        = alphas = qs = None
    V_history     = []
    sigma_history = []
    qs_history    = []

    for iteration in range(max_iter):
        V_old = V.copy()
        sigmas, alphas, qs, T = _vfi_step(V, proposer_probs)
        V    = compute_values(T, delta)
        diff = np.max(np.abs(V - V_old))
        V_history.append(V.copy())
        sigma_history.append([dict(s) for s in sigmas])
        qs_history.append([dict(q) for q in qs])

        if verbose:
            print(f"  Iter {iteration+1:3d}  |ΔV| = {diff:.2e}")

        if diff < tol:
            if verbose:
                print(f"  Converged in {iteration+1} iterations.\n")
            return V, sigmas, alphas, qs

        if len(V_history) >= cycle_window:
            for period in range(2, cycle_window):
                if np.max(np.abs(V_history[-1] - V_history[-1-period])) < tol * 500:
                    if verbose:
                        print(f"  *** Cycle (period {period}) at iter {iteration+1}."
                              f" Resolving mixed-strategy equilibrium. ***")
                    V_cycle      = np.array(V_history[-period:])
                    sigmas_cycle = sigma_history[-period:]
                    qs_cycle_    = qs_history[-period:]

                    V_star, sigmas_star, alphas_star, qs_star = _resolve_cycle(
                        V_cycle, sigmas_cycle, qs_cycle_, delta,
                        proposer_probs, tol, verbose)

                    if V_star is not None:
                        T_star     = full_transition_matrix(sigmas_star, qs_star, proposer_probs)
                        V_check    = compute_values(T_star, delta)
                        final_diff = np.max(np.abs(V_check - V_star))
                        if verbose:
                            print(f"  Resolution |ΔV| = {final_diff:.2e}")
                        r_ok, _ = verify_responses(sigmas_star, alphas_star, qs_star, V_star)
                        p_ok, _ = verify_proposals(sigmas_star, alphas_star, qs_star, V_star)
                        if r_ok and p_ok:
                            if verbose:
                                print(f"  Mixed-strategy equilibrium resolved.\n")
                            return V_star, sigmas_star, alphas_star, qs_star
                        if verbose:
                            print(f"  Secondary near-indifferences remain; continuing VFI from V*.")
                        V = V_star
                        V_history.clear()
                        sigma_history.clear()
                        qs_history.clear()
                        V_history.append(V.copy())
                        sigma_history.append([dict(s) for s in sigmas_star])
                        qs_history.append([dict(q) for q in qs_star])
                        break

                    V = np.mean(V_cycle, axis=0)
                    V_history.clear()
                    sigma_history.clear()
                    qs_history.clear()
                    V_history.append(V.copy())
                    break

    warnings.warn("VFI did not converge within max_iter.")
    return V, sigmas, alphas, qs

# ===========================================================================
# 8.  EQUILIBRIUM VERIFICATION
# ===========================================================================

def verify_responses(sigmas, alphas, qs, V):
    violations = []
    for si in range(N_STATES):
        for (j, ns_idx), a in alphas[si].items():
            dv = V[ns_idx, j] - V[si, j]
            if dv > EPS_IND and not np.isclose(a, 1.0):
                violations.append(
                    f"  RESP: {PLAYERS[j]} should ACCEPT "
                    f"{partition_name(STATES[si])}->{partition_name(STATES[ns_idx])} "
                    f"(ΔV={dv:+.6f}) but alpha={a:.4f}")
            elif dv < -EPS_IND and not np.isclose(a, 0.0):
                violations.append(
                    f"  RESP: {PLAYERS[j]} should REJECT "
                    f"{partition_name(STATES[si])}->{partition_name(STATES[ns_idx])} "
                    f"(ΔV={dv:+.6f}) but alpha={a:.4f}")
    return len(violations) == 0, violations


def verify_proposals(sigmas, alphas, qs, V):
    violations = []
    for si in range(N_STATES):
        sigma, q = sigmas[si], qs[si]
        for i in range(N_PLAYERS):
            exp_val = {
                ns: q[(i, ns)] * V[ns, i] + (1 - q[(i, ns)]) * V[si, i]
                for (i2, ns) in sigma if i2 == i
            }
            best = max(exp_val.values())
            for ns, p in [(ns, sigma[(i, ns)]) for (i2, ns) in sigma if i2 == i]:
                if p > EPS_IND and not np.isclose(exp_val[ns], best, atol=EPS_IND):
                    violations.append(
                        f"  PROP: {PLAYERS[i]} at {partition_name(STATES[si])} "
                        f"proposes suboptimal {partition_name(STATES[ns])} "
                        f"(exp={exp_val[ns]:.6f} vs best={best:.6f})")
    return len(violations) == 0, violations


def verify_equilibrium(sigmas, alphas, qs, V, verbose=True):
    r_pass, r_viol = verify_responses(sigmas, alphas, qs, V)
    p_pass, p_viol = verify_proposals(sigmas, alphas, qs, V)
    if verbose:
        print("=" * 60)
        print("EQUILIBRIUM VERIFICATION")
        print("=" * 60)
        print(f"  Acceptance strategies: {'✓ valid' if r_pass else f'✗ {len(r_viol)} violations'}")
        for v in r_viol:
            print(v)
        print(f"  Proposal strategies:   {'✓ valid' if p_pass else f'✗ {len(p_viol)} violations'}")
        for v in p_viol:
            print(v)
    return r_pass and p_pass

# ===========================================================================
# 9.  RESULTS DISPLAY
# ===========================================================================

def print_results(V, sigmas, alphas, qs):
    # --- Equilibrium values ---
    print("=" * 60)
    print("EQUILIBRIUM VALUES")
    print("=" * 60)
    col_hdr = "".join(f"{PLAYERS[p]:>9}" for p in range(N_PLAYERS))
    print(f"  {'State':<25}{col_hdr}")
    print("  " + "-" * (25 + 9 * N_PLAYERS))
    for si, state in enumerate(STATES):
        vals = "".join(f"{V[si, p]:>9.4f}" for p in range(N_PLAYERS))
        print(f"  {partition_name(state):<25}{vals}")

    # --- Strategies ---
    print()
    print("=" * 60)
    print("EQUILIBRIUM STRATEGIES")
    print("=" * 60)
    for si, state in enumerate(STATES):
        print(f"\n--- State: {partition_name(state)} ---")
        sigma, alpha = sigmas[si], alphas[si]

        for i in range(N_PLAYERS):
            props = sorted(
                [(ns, p) for (i2, ns), p in sigma.items() if i2 == i and p > 1e-6],
                key=lambda x: -x[1])
            prop_str = ", ".join(
                f"{partition_name(STATES[ns])}:{p:.3f}" for ns, p in props)
            print(f"  {PLAYERS[i]} proposes: {prop_str}")

        for (j, ns_idx), a in sorted(alpha.items()):
            label = ("ACCEPT" if a > 1 - 1e-6
                     else "REJECT" if a < 1e-6
                     else f"MIX({a:.3f})")
            print(f"  {PLAYERS[j]} votes on "
                  f"{partition_name(STATES[ns_idx])}: {label}")

    # --- Transition matrix ---
    print()
    print("=" * 60)
    print("TRANSITION MATRIX  T[s -> s']")
    print("=" * 60)
    T  = full_transition_matrix(sigmas, qs)
    cw = 14
    hdr = "".join(f"{partition_name(STATES[j]):>{cw}}" for j in range(N_STATES))
    print(f"  {'':25}{hdr}")
    for si in range(N_STATES):
        row = "".join(f"{T[si,j]:>{cw}.4f}" for j in range(N_STATES))
        print(f"  {partition_name(STATES[si]):<25}{row}")

    # --- Stationary distribution (eigenvalue method — always works) ---
    print()
    print("=" * 60)
    print("STATIONARY DISTRIBUTION")
    print("=" * 60)
    try:
        eigvals, eigvecs = np.linalg.eig(T.T)
        unit_idx = np.where(np.abs(eigvals - 1.0) < 1e-8)[0]
        if len(unit_idx) == 0:
            unit_idx = np.array([np.argmin(np.abs(eigvals - 1.0))])
        pi_raw = eigvecs[:, unit_idx[0]].real
        pi_raw = np.abs(pi_raw)
        pi     = pi_raw / pi_raw.sum()
        for si, state in enumerate(STATES):
            bar = "█" * int(round(pi[si] * 40))
            print(f"  {partition_name(state):<25} {pi[si]:.4f}  {bar}")
    except Exception as exc:
        print(f"  (could not compute: {exc})")

# ===========================================================================
# 10.  MAIN
# ===========================================================================

def equilibria_are_distinct(V1, V2, atol=1e-2):
    """
    Two equilibria are considered the same if their value functions agree
    within atol at every state and player. Using V rather than T or sigma
    because V is the canonical payoff-relevant object; strategy profiles
    that differ only off the equilibrium path are treated as identical.
    """
    return np.max(np.abs(V1 - V2)) > atol


def find_equilibria(delta, proposer_probs, n_restarts=40,
                    tol=1e-8, max_iter=200, seed=0, verbose_each=False):
    """
    Search for multiple SMPE by running VFI from diverse starting conditions.

    Three complementary sources of variation are used simultaneously:

    1. Random V_init — sample starting value functions from a scaled normal
       distribution centred on PAYOFFS. Different starting points can lie in
       different basins of attraction and converge to different fixed points.

    2. Proposer probability vectors rho — who gets to propose (and how often)
       affects which transitions are available at each state, steering VFI
       toward different equilibria. Each rho is tried with every V_init.

    3. MIP tie-breaking is deterministic given (V_init, rho), so we don't
       need to randomize it separately — the variation above is sufficient.

    Parameters
    ----------
    delta      : float
    n_restarts : int   — number of random V_init draws per rho variant
    rho_variants : list of arrays or None
        Proposer probability vectors to try. If None, uses:
          - uniform (default)
          - one vector emphasising each player in turn
    tol        : convergence tolerance passed to vfi
    max_iter   : passed to vfi
    seed       : numpy random seed for reproducibility
    verbose_each : print VFI progress for every run (very verbose)

    Returns
    -------
    list of dicts, each with keys:
        'V'          : equilibrium value function
        'sigmas'     : proposal strategies
        'alphas'     : acceptance strategies
        'qs'         : q values
        'rho'        : proposer probs used
        'V_init_tag' : 'payoffs' | 'random-k'
        'verified'   : bool
    """
    rng = np.random.default_rng(seed)


    # Build V_init pool: first entry is always the canonical PAYOFFS start
    payoff_scale = max(np.abs(PAYOFFS).max(), 1.0)
    V_inits = [("payoffs", PAYOFFS.copy())]
    for k in range(n_restarts - 1):
        noise = rng.normal(0, 10*payoff_scale, size=PAYOFFS.shape)
        V_inits.append((f"random-{k}", PAYOFFS + noise))

    equilibria = []
    total_runs  =  len(V_inits)
    run_idx     = 0


    for tag, V0 in V_inits:
        run_idx += 1
        if not verbose_each:
            print(f"  Run {run_idx:3d}/{total_runs}  init={tag}",
                    end="", flush=True)
        try:
            V, sigmas, alphas, qs = vfi(
                delta=delta, max_iter=max_iter, tol=tol,
                proposer_probs=proposer_probs, V_init=V0,
                verbose=verbose_each)
            r_ok, _ = verify_responses(sigmas, alphas, qs, V)
            p_ok, _ = verify_proposals(sigmas, alphas, qs, V)
            verified = r_ok and p_ok
        except Exception as exc:
            if not verbose_each:
                print(f"  → ERROR: {exc}")
            continue

        if not verified:
            if not verbose_each:
                print("  → not verified, skipping")
            continue

        # Check whether this V is genuinely new
        is_new = all(equilibria_are_distinct(V, eq["V"]) for eq in equilibria)
        if is_new:
            equilibria.append(dict(V=V, sigmas=sigmas, alphas=alphas,
                                    qs=qs, V_init_tag=tag,
                                    verified=verified))
            if not verbose_each:
                print(f"  → NEW equilibrium #{len(equilibria)}")
        else:
            if not verbose_each:
                print("  → duplicate")

    return equilibria


def print_equilibrium_summary(equilibria, delta):
    """Print a compact comparison table of all found equilibria."""
    n = len(equilibria)
    print()
    print("=" * 70)
    print(f"EQUILIBRIUM SUMMARY  (delta={delta},  {n} distinct equilibri{'um' if n==1 else 'a'} found)")
    print("=" * 70)

    # Header: one column per equilibrium
    col = 14
    hdr = "".join(f"  Eq {i+1:>2}" for i in range(n))
    print(f"  {'State':<25}{'Player':<8}{hdr}")
    print("  " + "-" * (33 + (col) * n))

    for si, state in enumerate(STATES):
        for pi, player in enumerate(range(N_PLAYERS)):
            row = "".join(f"{eq['V'][si, player]:>{col}.4f}" for eq in equilibria)
            label = partition_name(state) if pi == 0 else ""
            print(f"  {label:<25}{PLAYERS[player]:<8}{row}")

    print()
    print("  Stationary distributions:")
    print(f"  {'State':<25}" + "".join(f"{'Eq '+str(i+1):>{col}}" for i in range(n)))
    print("  " + "-" * (25 + col * n))

    for i, eq in enumerate(equilibria):
        T = full_transition_matrix(eq['sigmas'], eq['qs'], PROPOSER_PROBS)
        eigvals, eigvecs = np.linalg.eig(T.T)
        unit_idx = np.where(np.abs(eigvals - 1.0) < 1e-8)[0]
        if len(unit_idx) == 0:
            unit_idx = np.array([np.argmin(np.abs(eigvals - 1.0))])
        pi_raw = np.abs(eigvecs[:, unit_idx[0]].real)
        eq['pi'] = pi_raw / pi_raw.sum()

    for si, state in enumerate(STATES):
        row = "".join(f"{eq['pi'][si]:>{col}.4f}" for eq in equilibria)
        print(f"  {partition_name(state):<25}{row}")

    print()
    print("  Discovery info:")
    for i, eq in enumerate(equilibria):
        print(f"  Eq {i+1}: first found with init={eq['V_init_tag']}")

# ===========================================================================
# 10.  MAIN
# ===========================================================================



if __name__ == "__main__":
    print(f"Solving {N_PLAYERS}-player farsighted coalition formation MIP\n")

    if not SEARCH_MODE:
        # ── Single solve (original behaviour) ──────────────────────────────
        print(f"Running VFI with delta={DELTA} ...")
        V, sigmas, alphas, qs = vfi(delta=DELTA, max_iter=MAX_ITER, tol=1e-8, verbose=True, proposer_probs=PROPOSER_PROBS)
        print_results(V, sigmas, alphas, qs)
        verify_equilibrium(sigmas, alphas, qs, V)

    else:
        # ── Multi-start equilibrium search ─────────────────────────────────
        print(f"Searching for multiple equilibria at delta={DELTA} ...")
        print(f"(set SEARCH_MODE = False for single-solve output)\n")
        equilibria = find_equilibria(
            delta      = DELTA,
            n_restarts = N_RESTARTS,      # random V_init draws per rho variant
            seed       = 42,
            verbose_each = False,
            proposer_probs=PROPOSER_PROBS,
        )
        print()
        if not equilibria:
            print("No verified equilibria found.")
        else:
            print_equilibrium_summary(equilibria, DELTA)
            print()
            print("Full details of each equilibrium:")
            for i, eq in enumerate(equilibria):
                print(f"\n{'='*60}")
                print(f"EQUILIBRIUM {i+1}")
                print(f"{'='*60}")
                print_results(eq['V'], eq['sigmas'], eq['alphas'], eq['qs'])
                verify_equilibrium(eq['sigmas'], eq['alphas'], eq['qs'], eq['V'])

