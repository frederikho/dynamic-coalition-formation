"""
Jere's MIP-VFI coalition formation solver — clean port without module-level globals.

Usage
-----
    from lib.equilibrium.jeres_vfi import Game, find_equilibria, vfi

    game = Game.from_payoffs(players, payoffs_np)
    equilibria = find_equilibria(game, delta=0.99, proposer_probs=[1/3, 1/3, 1/3])
    V, sigmas, alphas, qs = vfi(game, delta=0.99)

Public API
----------
Game            — dataclass holding players, states, payoffs
find_equilibria — multi-start search returning list of equilibrium dicts
vfi             — single VFI run from a given V_init
verify_responses / verify_proposals / verify_equilibrium — consistency checks
all_partitions / canon_partition / gen_partitions — partition combinatorics
"""

from lib.equilibrium.jeres_vfi.game import (
    Game,
    EPS_IND,
    gen_partitions,
    canon_partition,
    all_partitions,
    changed_players,
    voters,
    is_unilateral_exit,
    fw_state_name_to_partition,
)

from lib.equilibrium.jeres_vfi.solver import (
    solve_state_mip,
    full_transition_matrix,
    compute_values,
    vfi,
    find_equilibria,
    verify_responses,
    verify_proposals,
    verify_equilibrium,
)

def solve_with_jeres_vfi(solver, params=None):
    """
    Find SMPE using the ported Jere's MIP-VFI implementation.

    Takes an EquilibriumSolver and returns (strategy_df, result_dict) matching
    the interface of all other solver approaches in find_equilibrium.py.

    Recognised params keys
    ----------------------
    jeres_vfi_n_restarts  (int,   default 40)    random V_init restarts
    jeres_vfi_max_iter    (int,   default 300)   max VFI iterations per run
    jeres_vfi_tol         (float, default 1e-6)  VFI convergence tolerance
    jeres_vfi_seed        (int,   default 42)    random seed for multi-start
    jeres_vfi_single      (bool,  default False) single run instead of multi-start
    jeres_vfi_cycle_window(int,   default 8)     cycle-detection window
    """
    import numpy as np
    from lib.equilibrium.mip_vfi import _arrays_to_strategy_df
    from lib.utils import get_approval_committee

    if params is None:
        params = {}

    n_restarts   = int(params.get("jeres_vfi_n_restarts", 40))
    max_iter     = int(params.get("jeres_vfi_max_iter", 300))
    tol          = float(params.get("jeres_vfi_tol", 1e-6))
    seed         = int(params.get("jeres_vfi_seed", 42))
    single       = bool(params.get("jeres_vfi_single", False))
    cycle_window = int(params.get("jeres_vfi_cycle_window", 8))
    verify_atol  = float(params.get("jeres_vfi_verify_atol", tol * 100))

    players     = solver.players
    state_names = solver.states
    n_players   = len(players)
    n_states    = len(state_names)
    delta       = float(solver.discounting)
    rho         = [float(solver.protocol[p]) for p in players]

    # Build Jere state list and bidirectional index maps
    jere_states   = all_partitions(n_players)
    jere_state_idx = {s: i for i, s in enumerate(jere_states)}

    fw_to_jere = {}  # framework idx → jere idx
    jere_to_fw = {}  # jere idx     → framework idx
    for fw_idx, name in enumerate(state_names):
        jkey  = fw_state_name_to_partition(name, players)
        j_idx = jere_state_idx[jkey]
        fw_to_jere[fw_idx] = j_idx
        jere_to_fw[j_idx]  = fw_idx

    # Build payoffs in Jere order: (n_jere_states, n_players)
    payoffs_df = solver.payoffs
    payoffs_np = np.zeros((len(jere_states), n_players))
    for fw_idx, name in enumerate(state_names):
        payoffs_np[fw_to_jere[fw_idx]] = payoffs_df.loc[name, players].values

    # Build approval committees from the framework's effectivity so the MIP
    # uses the same committee structure as the framework verifier.
    player_idx = {p: i for i, p in enumerate(players)}
    approval_committees = {}
    for fw_s, s_name in enumerate(state_names):
        for i, proposer in enumerate(players):
            for fw_sp, sp_name in enumerate(state_names):
                committee_names = get_approval_committee(
                    solver.effectivity, players, proposer, s_name, sp_name
                )
                j_s  = fw_to_jere[fw_s]
                j_sp = fw_to_jere[fw_sp]
                approval_committees[(i, j_s, j_sp)] = frozenset(
                    player_idx[name] for name in committee_names
                )

    # Identify structurally impossible transitions: empty committee for non-self transitions.
    # The framework treats these as p_approved=0 (impossible), but Jere's default q=1
    # for empty voter sets would make them auto-approve.  Mark them forbidden so the MIP
    # forces sigma=0 for such transitions, keeping Jere's T consistent with the framework's T.
    forbidden_transitions: set = set()
    for fw_s, s_name in enumerate(state_names):
        for i, proposer in enumerate(players):
            for fw_sp, sp_name in enumerate(state_names):
                if s_name == sp_name:
                    continue  # self-transitions: constant_1 in both models
                j_s  = fw_to_jere[fw_s]
                j_sp = fw_to_jere[fw_sp]
                committee = approval_committees.get((i, j_s, j_sp), frozenset())
                if len(committee) == 0:
                    forbidden_transitions.add((i, j_s, j_sp))

    # Transitions the effectivity rule forbids outright (e.g. non-adjacent moves under
    # 'adjacent_step').  These can still have a NON-empty committee, so the check above
    # misses them; without this the MIP may propose them and the framework verifier then
    # rejects the profile with a proposal-strategy error.
    fw_state_pos = {name: idx for idx, name in enumerate(state_names)}
    for proposer, s_name, sp_name in solver.forbidden_proposals:
        i    = player_idx[proposer]
        j_s  = fw_to_jere[fw_state_pos[s_name]]
        j_sp = fw_to_jere[fw_state_pos[sp_name]]
        if j_s == j_sp:
            continue
        forbidden_transitions.add((i, j_s, j_sp))

    game = Game.from_payoffs(players, payoffs_np)
    game.approval_committees = approval_committees
    game.forbidden_transitions = forbidden_transitions

    # Run solver.  vfi() warns once per restart that fails to converge, and
    # solve_state_mip once per infeasible state — per-restart diagnostics that would
    # flood stderr across a multi-start run.  Count them here and report the totals
    # in the result dict; anything else is re-emitted so unexpected warnings still
    # reach the caller.
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")

        if single:
            V, sigmas, alphas, qs = vfi(
                game, delta=delta, max_iter=max_iter, tol=tol,
                cycle_window=cycle_window, proposer_probs=rho, verbose=False,
                verify_atol=verify_atol,
            )
            r_ok, _ = verify_responses(game, sigmas, alphas, qs, V, atol=verify_atol)
            p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V, atol=verify_atol)
            equilibria = [dict(V=V, sigmas=sigmas, alphas=alphas, qs=qs,
                               V_init_tag="payoffs", verified=r_ok and p_ok)]
            stopping = "jeres_vfi_single"
            n_found  = 1
        else:
            equilibria = find_equilibria(
                game, delta=delta, proposer_probs=rho,
                n_restarts=n_restarts, tol=tol, max_iter=max_iter,
                cycle_window=cycle_window, seed=seed, verbose=False,
                verify_atol=verify_atol,
            )
            stopping = "jeres_vfi_multistart"
            n_found  = len(equilibria)

    n_nonconverged = 0
    n_mip_infeasible = 0
    for w in caught:
        text = str(w.message)
        if "VFI did not converge" in text:
            n_nonconverged += 1
        elif "MIP infeasible" in text:
            n_mip_infeasible += 1
        else:
            _warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    counts = {
        "vfi_nonconverged_restarts": n_nonconverged,
        "mip_infeasible_states": n_mip_infeasible,
    }

    if not equilibria:
        return None, {
            "converged": False,
            "stopping_reason": "no_equilibrium_found",
            "outer_iterations": 0,
            "final_tau_p": 0.0,
            "final_tau_r": 0.0,
            "n_equilibria_found": 0,
            "found_at_restart": None,
            "n_restarts_run": 1 if single else n_restarts,
            **counts,
        }

    # Convert first equilibrium to framework strategy DataFrame
    eq      = equilibria[0]
    sigmas_ = eq["sigmas"]
    alphas_ = eq["alphas"]

    all_sigmas = np.zeros((n_states, n_players, n_states))
    all_alphas = np.zeros((n_states, n_players, n_states))

    for fw_s in range(n_states):
        j_s = fw_to_jere[fw_s]
        for (i, j_sp), prob in sigmas_[j_s].items():
            if j_sp in jere_to_fw:
                all_sigmas[fw_s, i, jere_to_fw[j_sp]] = prob
        for (j_voter, j_sp), prob in alphas_[j_s].items():
            if j_sp in jere_to_fw:
                all_alphas[fw_s, j_voter, jere_to_fw[j_sp]] = prob

    strategy_df = _arrays_to_strategy_df(solver, all_sigmas, all_alphas)

    # Which multi-start run produced the exported equilibrium: 0 = the deterministic
    # payoff-initialised run, k = the k-th random restart.  Reveals whether a hit was
    # immediate or needed most of the restart budget, which pass/fail alone hides.
    tag = eq["V_init_tag"]
    found_at_restart = 0 if tag == "payoffs" else int(tag.split("-")[1]) + 1

    return strategy_df, {
        "converged": True,
        "stopping_reason": stopping,
        "outer_iterations": max_iter,
        "final_tau_p": 0.0,
        "final_tau_r": 0.0,
        "n_equilibria_found": n_found,
        "found_at_restart": found_at_restart,
        "n_restarts_run": 1 if single else n_restarts,
        **counts,
    }


__all__ = [
    "Game",
    "EPS_IND",
    "gen_partitions",
    "canon_partition",
    "all_partitions",
    "changed_players",
    "voters",
    "is_unilateral_exit",
    "solve_state_mip",
    "full_transition_matrix",
    "compute_values",
    "vfi",
    "find_equilibria",
    "verify_responses",
    "verify_proposals",
    "verify_equilibrium",
    "fw_state_name_to_partition",
    "solve_with_jeres_vfi",
]
