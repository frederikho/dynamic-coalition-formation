"""
Find equilibrium strategy profiles using the smoothed fixed-point iteration algorithm.

This script allows you to find equilibria for different game configurations
and save the resulting strategy profiles to Excel files.
"""

import argparse
import inspect
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from typing import Any, Dict, List

from lib.logging import get_logger
from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.equilibrium.solver import EquilibriumSolver
from lib.equilibrium.active_set_n3 import solve_with_active_set_n3
from lib.equilibrium.ordinal_ranking import solve_with_ordinal_ranking_n3
from lib.equilibrium.support_enumeration_n3 import solve_with_support_enumeration_n3
from lib.equilibrium.scenarios import get_scenario, list_scenarios
from lib.equilibrium.excel_writer import (
    write_strategy_table_excel,
    generate_filename,
    generate_config_hash,
    write_payoff_table_excel
)
from lib.effectivity import heyen_lehtomaa_2021
from lib.utils import (
    derive_effectivity,
    get_payoff_matrix,
    verify_equilibrium,
    verify_equilibrium_detailed,
    get_geoengineering_levels,
    get_deploying_coalitions,
    list_members,
    get_approval_committee,
)
from lib.probabilities_optimized import TransitionProbabilitiesOptimized as TransitionProbabilities
from lib.mdp import MDP
from lib.coalition_structures import generate_coalition_structures, generate_all_coalition_maps


def _deployer_key(state) -> str:
    """
    Compute the payoff table lookup key for a state.

    Returns '( )' when G=0 (no deployment), or '(MEMBERS)' with members
    sorted alphabetically for the coalition that actually deploys.
    """
    if state.geo_deployment_level == 0:
        return "( )"
    members = sorted(country.name for country in state.strongest_coalition.members)
    return "(" + "".join(members) + ")"


def _load_payoff_table(path: Path, states: list, players: list) -> tuple:
    """
    Load a precomputed payoff table from an Excel file (as produced by lib.ingest_payoffs).

    Rows in the table are keyed by deployer set (e.g. '( )', '(IND)', '(INDRUS)').
    For each framework state, the deploying coalition is determined from the State
    object (via strongest_coalition / geo_deployment_level) and used as the lookup key.
    This works for both power_threshold and weak_governance scenarios.

    The file must have a 'Payoffs' sheet where:
    - Row 1 is a title (skipped)
    - Row 2 is the header: State, <player1>, <player2>, ..., W_SAI_sum_≤YYYY, (other columns)
    - Column A is the deployer key index

    Path resolution: tries the path as given first, then payoff_tables/<basename>.

    Returns a tuple (payoffs, geo_levels) where:
    - payoffs: DataFrame indexed by framework state names, columns=players, dtype float64
    - geo_levels: DataFrame indexed by framework state names, column="G", with W_SAI values
                  (or None if no W_SAI_sum_* column is found in the table)
    """
    # Resolve path: as given, or under payoff_tables/
    _default_dir = Path(__file__).parent.parent.parent / "payoff_tables"
    if not path.exists():
        fallback = _default_dir / path.name
        if fallback.exists():
            path = fallback
        else:
            searched = [str(path.resolve()), str(fallback.resolve())]
            raise FileNotFoundError(
                f"Payoff table '{path.name}' not found.\n"
                f"Searched:\n" + "\n".join(f"  {p}" for p in searched) + "\n"
                f"Tip: place the file in payoff_tables/ or provide the full path."
            )

    # header=1 uses the second row (0-indexed) as column names; index_col=0 uses State column as index
    df = pd.read_excel(str(path), sheet_name="Payoffs", header=1, index_col=0)

    # Verify all required players are present
    missing_players = [p for p in players if p not in df.columns]
    if missing_players:
        raise ValueError(
            f"Payoff table {path.name} is missing player columns: {missing_players}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Find W_SAI column (named W_SAI_sum_≤YYYY by ingest_payoffs)
    sai_col = next((c for c in df.columns if str(c).startswith("W_SAI")), None)

    # Build result DataFrames indexed by framework state names
    state_names = [s.name for s in states]
    payoffs = pd.DataFrame(index=state_names, columns=players, dtype=np.float64)
    geo_levels = pd.DataFrame(index=state_names, columns=["G"], dtype=np.float64) if sai_col else None

    for state in states:
        # Prefer direct state-name rows when available (used by --save-payoffs).
        # Fallback to deployer-key rows for ingest_payoffs-style tables.
        if state.name in df.index:
            row_key = state.name
        else:
            row_key = _deployer_key(state)
        if row_key not in df.index:
            needed_key = _deployer_key(state)
            raise ValueError(
                f"Payoff table {path.name} has no row for framework state '{state.name}' "
                f"or deployer key '{needed_key}'.\n"
                f"Available keys: {df.index.tolist()}"
            )
        payoffs.loc[state.name] = df.loc[row_key, players].values
        if sai_col is not None:
            geo_levels.loc[state.name, "G"] = float(df.loc[row_key, sai_col])

    return payoffs, geo_levels


def _read_payoff_table_index(path: Path) -> list[str]:
    """
    Read the row index from a payoff table without interpreting it as framework states.

    Path resolution matches _load_payoff_table.
    """
    _default_dir = Path(__file__).parent.parent.parent / "payoff_tables"
    if not path.exists():
        fallback = _default_dir / path.name
        if fallback.exists():
            path = fallback
        else:
            searched = [str(path.resolve()), str(fallback.resolve())]
            raise FileNotFoundError(
                f"Payoff table '{path.name}' not found.\n"
                f"Searched:\n" + "\n".join(f"  {p}" for p in searched)
            )
    df = pd.read_excel(str(path), sheet_name="Payoffs", header=1, index_col=0)
    return [str(idx) for idx in df.index.tolist()]


def _synthetic_coalition_map(state_name: str, players: list) -> list | None:
    """
    Build a coalition map for a non-canonical state name such as '(CHN)' in a
    2-player game where the standard Bell-number enumeration does not produce it.

    Players named inside the parentheses form one coalition; remaining players
    are singletons.  Returns None if no players can be parsed (e.g. '( )').

    This is used when a payoff table defines a custom state space (e.g. a
    reduced 3-state / 2-player model).  Because payoffs and geo levels are read
    directly from the table, the synthetic map only serves to construct the
    State object and does not affect computed payoffs.
    """
    coalition = sorted(list_members(state_name, players))
    if not coalition:
        # No player names found — treat as all-singletons (e.g. 'N', 'none', 'empty').
        return [[p] for p in sorted(players)]
    singletons = [[p] for p in sorted(players) if p not in coalition]
    return [coalition] + singletons


def setup_experiment(config):
    """
    Setup experiment configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with game setup (players, states, effectivity, payoffs, etc.)
    """
    # Initialize countries
    all_countries = []
    country_dict = {}
    for player in config["players"]:
        country = Country(
            name=player,
            base_temp=config["base_temp"][player],
            delta_temp=config["delta_temp"][player],
            ideal_temp=config["ideal_temp"][player],
            m_damage=config["m_damage"][player],
            power=config["power"][player]
        )
        all_countries.append(country)
        country_dict[player] = country

    # Generate coalition structures dynamically based on number of players
    players = config["players"]

    allow_non_canonical = config.get("allow_non_canonical_states", False)

    # Generate state names if not provided. When a payoff table supplies coalition
    # rows that form a subset of the scenario's canonical state set, use that
    # subset directly. This also supports reduced tables that still contain extra
    # helper rows such as singleton states.
    if "state_names" not in config or config["state_names"] is None:
        canonical_state_names = generate_coalition_structures(players)
        payoff_table_path = config.get("payoff_table", None)
        if payoff_table_path is not None:
            payoff_rows = _read_payoff_table_index(Path(payoff_table_path))
            canonical_row_set = set(canonical_state_names)
            payoff_state_subset = [
                state_name for state_name in canonical_state_names if state_name in set(payoff_rows)
            ]
            if allow_non_canonical:
                # Use all table rows that are either canonical or parseable as
                # custom coalition structures, preserving table order.
                non_canonical = [
                    r for r in payoff_rows
                    if r not in canonical_row_set
                    and _synthetic_coalition_map(r, players) is not None
                ]
                active = set(payoff_state_subset) | set(non_canonical)
                state_names = [r for r in payoff_rows if r in active]
                if not state_names:
                    state_names = canonical_state_names
            elif payoff_state_subset:
                state_names = payoff_state_subset
            else:
                state_names = canonical_state_names
        else:
            state_names = canonical_state_names
        config["state_names"] = state_names
    else:
        state_names = config["state_names"]

    # Generate coalition maps for all states
    all_coalition_maps = generate_all_coalition_maps(players)

    # Create State objects for each coalition structure
    states = []
    for state_name in state_names:
        if state_name not in all_coalition_maps:
            if not allow_non_canonical:
                raise ValueError(f"Unknown state: {state_name}")
            # Non-canonical state (e.g. '(CHN)' in a 2-player reduced model).
            # Build a synthetic coalition map so a State object can be created;
            # actual payoffs and geo levels will be overridden from the table.
            synthetic = _synthetic_coalition_map(state_name, players)
            if synthetic is None:
                raise ValueError(f"Unknown state: {state_name}")
            all_coalition_maps[state_name] = synthetic

        # Get the coalition map for this state
        coalition_player_lists = all_coalition_maps[state_name]

        # Convert player name lists to Coalition objects
        coalitions = [
            Coalition([country_dict[player_name] for player_name in player_list])
            for player_list in coalition_player_lists
        ]

        state = State(
            name=state_name,
            coalitions=coalitions,
            all_countries=all_countries,
            power_rule=config["power_rule"],
            min_power=config.get("min_power", None)
        )
        states.append(state)

    # Load payoffs: either from precomputed table or by computing from state parameters
    payoff_table_path = config.get("payoff_table", None)
    if payoff_table_path is not None:
        payoffs, table_geo_levels = _load_payoff_table(Path(payoff_table_path), states, config["players"])
        geoengineering = table_geo_levels if table_geo_levels is not None else get_geoengineering_levels(states=states)
    else:
        payoffs = get_payoff_matrix(states=states, columns=config["players"])
        geoengineering = get_geoengineering_levels(states=states)
    deploying_coalitions = get_deploying_coalitions(states=states)

    # Derive effectivity from template or generate
    forbidden_proposals: frozenset = frozenset()
    template_file = config.get("template_file", None)
    if template_file and Path(template_file).exists():
        template_df = pd.read_excel(template_file, header=[0, 1], index_col=[0, 1, 2])
        effectivity = derive_effectivity(
            df=template_df,
            players=config["players"],
            states=state_names
        )
    else:
        effectivity_rule = config.get("effectivity_rule", "heyen_lehtomaa_2021")
        from lib.effectivity import get_effectivity, get_forbidden_proposals
        effectivity = get_effectivity(effectivity_rule, players, state_names)
        forbidden_proposals = get_forbidden_proposals(effectivity_rule, players, state_names)

    return {
        'players': config["players"],
        'states': states,
        'state_names': state_names,
        'effectivity': effectivity,
        'forbidden_proposals': forbidden_proposals,
        'protocol': config["protocol"],
        'payoffs': payoffs,
        'geoengineering': geoengineering,
        'deploying_coalitions': deploying_coalitions,
        'discounting': config["discounting"],
        'unanimity_required': config["unanimity_required"],
        'power_rule': config["power_rule"],
        'min_power': config.get("min_power", None)
    }


def _get_solver_params(config, user_params=None):
    """
    Get solver parameters with scenario-specific defaults.

    Args:
        config: Configuration dictionary
        user_params: User-provided solver parameters (optional)

    Returns:
        Dictionary of solver parameters
    """
    if user_params is None:
        user_params = {}
    """
    # Explanation parameters:

    'tau_p_init': any positive value. if this is small, the softmax ...
    'tau_r_init':  any positive value. if this is small, the sigmoid for acceptance becomes step-like
    'tau_decay':  set between 0 and 1. if this is close to 1, the annealing is slow
    'tau_min':  the temperature that has to be reached for convergence
    'max_outer_iter': Safety valve - convergence criterion will stop earlier
    'max_inner_iter':        
    'damping': 1 means full damping, 0 means no damping
    'inner_tol': 
    Convergence precision for the fixed-point iteration at a given temperature. Measures how well we solve "for this temperature τ, find strategies that satisfy the smoothed best-response conditions."
    
    'outer_tol': max_change has be lower than this to trigger expensive early verification.
    'consecutive_tol': how many consecutive outer iterations must meet to trigger early verification
    'verify_every_n': only run early verification every nth stable outer iteration (1 = every iter). Reduces runtime when verification dominates.
    'tau_margin': ,
    'project_to_exact': should always be True actually
    """
    
    #   One subtlety with the current update_k_players/update_k_acceptances setup: new_proposals[k] and new_acceptances[k] for non-updated keys are copied from old_*, so their change is     
    #   exactly 0. This means max_change is only driven by the values that were actually updated this iteration. With update_k_acceptances=1, only 1 acceptance key changes per inner         
    #   iteration, so max_change will often be small even if the overall strategy is far from a fixed point — inner convergence becomes easier to hit spuriously. Something to watch out for
    #   when interpreting "Converged after N iterations" with very granular updates. 
    
    # Default parameters resembling Jere's implementation
    default_params = {
        'tau_p_init': 1e-6,
        'tau_r_init': 1e-6,
        'tau_decay': 0.6,
        'tau_min': 1e-8,
        'max_outer_iter': 10,
        'max_inner_iter': 100,
        'damping': 0,
        'inner_tol': 1e-10,
        'outer_tol': 1e-9,
        'consecutive_tol': 1,
        'tau_margin': 0.01,
        'project_to_exact': True,
        'cycle_break_tau_threshold': 0.01, # Tau gate for the cycle stop; None means tau_min * (1 + tau_margin).
        'max_cycles_at_tau_min': 8, # Stop if low-tau cycling persists this many outer iterations in a row.
        'update_k_players': None,
        'link_player_updates': True,  # False makes the updating more granular
        'update_k_acceptances': None,
        'inner_cycle_check_interval': 20,
        'active_set_max_candidates': 1024,
        'active_set_refinement_iter': 10,
        'active_set_max_initial_approvals': 8,
        'active_set_max_initial_rows': 2,
        'active_set_max_supports_per_row': 2,
        'active_set_max_candidates_per_round': 256,
        'support_enumeration_max_candidates': 512,
        'support_enumeration_acceptance_fixpoint_iter': 20,
        'ordinal_ranking_max_combinations': None,
        'ordinal_ranking_shuffle': False,
        'ordinal_ranking_random_seed': 0,
        'ordinal_ranking_order': 'payoff',
        'ordinal_ranking_progress_every': 100,
        'ordinal_ranking_workers': 15,
        'ordinal_ranking_batch_size': 20000,
        'ordinal_ranking_weak_orders': False,
        'initialization_mode': 'uniform',
    }

    if len(config['players']) == 3:
        default_params.update({
            'tau_p_init': 0.2,
            'tau_r_init': 0.2,
            'tau_decay': 0.99,
            'tau_min': 0.001,
            'damping': 0.95,
            'max_inner_iter': 150,
            'max_outer_iter': 1000,
            'inner_tol': 2e-2,
            'outer_tol': 2e-2,
        }) 
        
        
    # for RICE case
    if len(config['players']) == 3:
        default_params.update({
            'tau_p_init': 1e-10,
            'tau_r_init': 1e-10,
            'tau_decay': 1,
            'tau_min': 1e-10,
            'damping': 0.0,
            'max_inner_iter': 30,
            'max_outer_iter': 20,
            'inner_tol': 5e-3,
            'outer_tol': 5e-3,
            'consecutive_tol': 3,
            'verify_every_n': 4,
            'update_k_players': 3,
            'link_player_updates': True,
            'update_k_acceptances': None,
            # tau_decay=1 means tau stays at tau_p_init=1e-6 forever.
            # The default cycle_break_tau_threshold=0.01 would fire immediately
            # (1e-6 < 0.01), falsely treating every saturated max_change=1.0 as
            # a low-tau cycle.  Set the threshold below the actual tau so the
            # cycle-stopping logic never triggers.
            'cycle_break_tau_threshold': 5e-10,
        })
        
    # Standard parameters for 4-player scenarios. Works well for some of them. Commented out, do not delete yet.
    # if len(config['players']) == 4:
    #     default_params.update({
    #         'tau_p_init': 1,
    #         'tau_r_init': 1,
    #         'tau_decay': 0.90,
    #         'tau_min': 0.001,
    #         'damping': 0.3,
    #         'max_inner_iter': 100,
    #         'max_outer_iter': 1000,
    #         'inner_tol': 2e-2,
    #         'outer_tol': 2e-2,
    #         'consecutive_tol': 2,
    #     })  
        
    # Special parameters for n==4
    if len(config['players']) == 4:
        default_params.update({
            'tau_p_init': 0.000002,
            'tau_r_init': 0.000002,
            'tau_decay': 0.99,
            'tau_min': 0.0000001,
            'damping': 0.0,
            'max_inner_iter': 100,
            'max_outer_iter': 1000,
            'inner_tol': 2e-2,
            'outer_tol': 2e-2,
            'consecutive_tol': 1,
            'verify_every_n': 1,
        })        

    if len(config['players']) >= 5:
        default_params.update({
            'tau_p_init': 2,
            'tau_r_init': 2,
            'tau_decay': 0.90,
            'tau_min': 0.01,
            'damping': 0.95,
            'max_inner_iter': 300,
            'max_outer_iter': 1000,
            'inner_tol': 1e-2,
            'outer_tol': 1e-2,
            'consecutive_tol': 1,
        }) 
        
    # Update with user-provided parameters
    default_params.update(user_params)

    return default_params


def _print_solver_params(params, logger):
    """Print solver parameters in consistent order."""
    param_order = ['tau_p_init', 'tau_r_init', 'tau_decay', 'tau_min',
                  'max_outer_iter', 'max_inner_iter', 'damping',
                  'inner_tol', 'outer_tol', 'consecutive_tol', 'verify_every_n',
                  'tau_margin', 'max_cycles_at_tau_min', 'cycle_break_tau_threshold',
                  'project_to_exact', 'update_k_players', 'link_player_updates',
                  'update_k_acceptances', 'inner_cycle_check_interval',
                  'active_set_max_candidates',
                  'active_set_refinement_iter',
                  'active_set_max_initial_approvals',
                  'active_set_max_initial_rows',
                  'active_set_max_supports_per_row',
                  'active_set_max_candidates_per_round',
                  'active_set_seed_rows',
                  'active_set_freeze_seeded_proposals',
                  'support_enumeration_max_candidates',
                  'support_enumeration_acceptance_fixpoint_iter',
                  'ordinal_ranking_max_combinations',
                  'ordinal_ranking_shuffle',
                  'ordinal_ranking_random_seed',
                  'ordinal_ranking_order',
                  'ordinal_ranking_progress_every',
                  'ordinal_ranking_workers',
                  'ordinal_ranking_batch_size',
                  'ordinal_ranking_weak_orders',
                  'initialization_mode']

    logger.info("Solver parameters:")
    for key in param_order:
        if key in params:
            logger.info(f"  {key}: {params[key]}")
    logger.info("")


def _run_solver(solver, params, checkpoint_dir='./checkpoints', load_from_checkpoint=False, config_hash=None, logger=None):
    """
    Run the equilibrium solver with KeyboardInterrupt handling.

    Args:
        solver: EquilibriumSolver instance
        params: Solver parameters
        checkpoint_dir: Directory for checkpoint files
        load_from_checkpoint: Whether to load from checkpoint
        config_hash: Configuration hash for checkpoint identification
        logger: Logger instance

    Returns:
        Tuple of (strategy_df, solver_result)
    """
    try:
        solve_signature = inspect.signature(solver.solve)
        allowed_param_names = {
            name
            for name, parameter in solve_signature.parameters.items()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        filtered_params = {
            key: value for key, value in params.items() if key in allowed_param_names
        }
        return solver.solve(
            **filtered_params,
            checkpoint_dir=checkpoint_dir,
            load_from_checkpoint=load_from_checkpoint,
            config_hash=config_hash
        )
    except KeyboardInterrupt:
        if logger:
            logger.warning("Solver stopped. No output file saved.")
        import sys
        sys.exit(0)


def _run_active_set_solver(solver, params, logger=None):
    """Run the cycle-guided active-set solver for n=3 cases."""
    try:
        max_candidates = int(params.get("active_set_max_candidates", 1024))
        refinement_iter = int(params.get("active_set_refinement_iter", 10))
        max_initial_approvals = int(params.get("active_set_max_initial_approvals", 8))
        max_initial_rows = int(params.get("active_set_max_initial_rows", 2))
        max_supports_per_row = int(params.get("active_set_max_supports_per_row", 2))
        max_candidates_per_round = int(params.get("active_set_max_candidates_per_round", 256))
        seed_rows = params.get("active_set_seed_rows")
        freeze_seeded_proposals = bool(params.get("active_set_freeze_seeded_proposals", False))
        return solve_with_active_set_n3(
            solver,
            max_candidates=max_candidates,
            refinement_iter=refinement_iter,
            max_initial_approvals=max_initial_approvals,
            max_initial_rows=max_initial_rows,
            max_supports_per_row=max_supports_per_row,
            max_candidates_per_round=max_candidates_per_round,
            seed_rows=seed_rows,
            freeze_seeded_proposals=freeze_seeded_proposals,
        )
    except KeyboardInterrupt:
        if logger:
            logger.warning("Solver stopped. No output file saved.")
        import sys
        sys.exit(0)


def _run_support_enumeration_solver(solver, params, logger=None):
    """Run the cycle-guided support-enumeration solver for n=3 cases."""
    try:
        max_candidates = int(params.get("support_enumeration_max_candidates", 512))
        acceptance_fixpoint_iter = int(params.get("support_enumeration_acceptance_fixpoint_iter", 20))
        return solve_with_support_enumeration_n3(
            solver,
            max_cycle_candidates=max_candidates,
            acceptance_fixpoint_iter=acceptance_fixpoint_iter,
        )
    except KeyboardInterrupt:
        if logger:
            logger.warning("Solver stopped. No output file saved.")
        import sys
        sys.exit(0)


def _run_ordinal_ranking_solver(solver, params, logger=None):
    """Run the exhaustive ordinal-ranking search for small n=3 cases."""
    try:
        max_combinations = params.get("ordinal_ranking_max_combinations")
        if max_combinations is not None:
            max_combinations = int(max_combinations)
        shuffle = bool(params.get("ordinal_ranking_shuffle", False))
        random_seed = int(params.get("ordinal_ranking_random_seed", 0))
        ranking_order = str(params.get("ordinal_ranking_order", "lexicographic"))
        progress_every = int(params.get("ordinal_ranking_progress_every", 0))
        workers = int(params.get("ordinal_ranking_workers", 8))
        batch_size = int(params.get("ordinal_ranking_batch_size", 20000))
        weak_orders = bool(params.get("ordinal_ranking_weak_orders", False))
        return solve_with_ordinal_ranking_n3(
            solver,
            max_combinations=max_combinations,
            shuffle=shuffle,
            random_seed=random_seed,
            ranking_order=ranking_order,
            progress_every=progress_every,
            workers=workers,
            batch_size=batch_size,
            weak_orders=weak_orders,
            logger=logger,
        )
    except KeyboardInterrupt:
        if logger:
            logger.warning("Solver stopped. No output file saved.")
        import sys
        sys.exit(0)


def _compute_verification(strategy_df, setup):
    """
    Compute transition probabilities, MDP, and value functions.

    Args:
        strategy_df: Strategy DataFrame (filled, no NaN)
        setup: Setup dictionary from setup_experiment()

    Returns:
        Tuple of (V, P, P_proposals, P_approvals)
    """
    # Compute transition probabilities
    tp = TransitionProbabilities(
        df=strategy_df,
        effectivity=setup['effectivity'],
        players=setup['players'],
        states=setup['state_names'],
        protocol=setup['protocol'],
        unanimity_required=setup['unanimity_required']
    )
    P, P_proposals, P_approvals = tp.get_probabilities()

    # Solve MDP for value functions
    mdp = MDP(
        n_states=len(setup['state_names']),
        transition_probs=P,
        discounting=setup['discounting']
    )

    V = pd.DataFrame(index=setup['state_names'], columns=setup['players'])
    for player in setup['players']:
        V.loc[:, player] = mdp.solve_value_func(setup['payoffs'].loc[:, player])

    return V, P, P_proposals, P_approvals


def _build_metadata(config, setup, solver_params, solver_result,
                    verification_success, runtime_seconds, start_time, end_time,
                    description=None, random_seed=None):
    """
    Build metadata dictionary for Excel file.

    Args:
        config: Configuration dictionary
        setup: Setup dictionary
        solver_params: Solver parameters used
        solver_result: Result from solver
        verification_success: Boolean verification result
        runtime_seconds: Runtime in seconds
        start_time: Start timestamp string
        end_time: End timestamp string
        description: Optional description string
        random_seed: Random seed used for initialization

    Returns:
        Metadata dictionary
    """
    metadata = {
        '--- RUN INFO ---': '',
        'start_time': start_time,
        'end_time': end_time,
        'runtime_seconds': f'{runtime_seconds:.2f}',
        'runtime_formatted': f'{runtime_seconds//60:.0f}m {runtime_seconds%60:.1f}s',
        'verification_success': verification_success,
        '': '',
        '--- SCENARIO INFO ---': '',
        'scenario_name': config.get('scenario_name', 'N/A'),
        'scenario_description': config.get('scenario_description', description or ''),
        ' ': '',
        '--- GAME CONFIG ---': '',
        'n_players': len(setup['players']),
        'players': ', '.join(setup['players']),
        'n_states': len(setup['state_names']),
        'states': ', '.join(setup['state_names']),
        'power_rule': config['power_rule'],
        'min_power': config.get('min_power', 'N/A'),
        'unanimity_required': config['unanimity_required'],
        'discounting': config['discounting'],
        '  ': '',
        '--- PLAYER PARAMETERS ---': '',
    }

    # Add player-specific parameters
    payoff_table_path = config.get("payoff_table", None)
    if payoff_table_path is not None:
        metadata['payoff_source'] = 'precomputed_table'
        metadata['payoff_table'] = str(payoff_table_path)
    for player in setup['players']:
        metadata[f'base_temp_{player}'] = config['base_temp'][player]
        metadata[f'ideal_temp_{player}'] = config['ideal_temp'][player]
        metadata[f'delta_temp_{player}'] = config['delta_temp'][player]
        metadata[f'm_damage_{player}'] = config['m_damage'][player]
        metadata[f'power_{player}'] = config['power'][player]

    # Add protocol
    metadata['  '] = ''
    metadata['--- PROTOCOL ---'] = ''
    for player in setup['players']:
        metadata[f'protocol_{player}'] = config['protocol'][player]

    # Add solver info
    metadata['   '] = ''
    metadata['--- SOLVER INFO ---'] = ''
    metadata['converged'] = solver_result.get('converged', 'N/A')
    metadata['outer_iterations'] = solver_result.get('outer_iterations', 'N/A')
    metadata['final_tau_p'] = f"{solver_result.get('final_tau_p', 'N/A'):.6f}" if 'final_tau_p' in solver_result else 'N/A'
    metadata['final_tau_r'] = f"{solver_result.get('final_tau_r', 'N/A'):.6f}" if 'final_tau_r' in solver_result else 'N/A'
    if random_seed is not None:
        metadata['random_seed'] = random_seed

    # Add solver parameters used
    metadata['     '] = ''
    metadata['--- SOLVER PARAMETERS ---'] = ''
    param_order = ['tau_p_init', 'tau_r_init', 'tau_decay', 'tau_min',
                  'max_outer_iter', 'max_inner_iter', 'damping',
                  'inner_tol', 'outer_tol', 'consecutive_tol', 'verify_every_n',
                  'tau_margin', 'max_cycles_at_tau_min', 'cycle_break_tau_threshold',
                  'project_to_exact', 'update_k_players', 'link_player_updates',
                  'update_k_acceptances', 'inner_cycle_check_interval']
    param_order.extend([
        'ordinal_ranking_max_combinations',
        'ordinal_ranking_shuffle',
        'ordinal_ranking_random_seed',
        'ordinal_ranking_order',
        'ordinal_ranking_progress_every',
    ])
    for key in param_order:
        if key in solver_params:
            metadata[f'solver_{key}'] = solver_params[key]

    # Add config hash for reference
    metadata['    '] = ''
    metadata['config_hash'] = generate_config_hash(config, length=10)

    # Add description if provided
    if description:
        metadata['custom_description'] = description

    return metadata


def _save_to_file(strategy_df, output_file, setup, metadata, V=None, transition_matrix=None, verbose=True, logger=None):
    """
    Save strategy profile to Excel file with metadata.

    Args:
        strategy_df: Strategy DataFrame
        output_file: Output file path
        setup: Setup dictionary
        metadata: Metadata dictionary
        V: Value functions DataFrame (optional)
        transition_matrix: Transition probability matrix DataFrame (optional)
        verbose: Whether to print status
        logger: Logger instance
    """
    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with custom Excel writer to match original format exactly
    write_strategy_table_excel(
        strategy_df, output_file, setup['players'],
        setup['effectivity'], setup['state_names'], metadata=metadata,
        value_functions=V, geo_levels=setup['geoengineering'],
        deploying_coalitions=setup.get('deploying_coalitions', None),
        static_payoffs=setup['payoffs'],
        transition_matrix=transition_matrix
    )

    if verbose and logger:
        logger.info(f"\nEquilibrium strategy saved to: {output_file}")
        runtime_seconds = float(metadata['runtime_seconds'])
        logger.info(f"Runtime: {runtime_seconds//60:.0f}m {runtime_seconds%60:.1f}s")


def _build_saved_payoff_table(setup: dict) -> pd.DataFrame:
    """
    Build a payoff table indexed by deployer key from framework state payoffs.

    Multiple framework states can map to the same deployer key. In that case we
    require payoff and geoengineering values to match (within tolerance).
    """
    players = setup["players"]
    ordered_states = setup["state_names"]
    rows = []
    for state_name in ordered_states:
        payload = {p: float(setup["payoffs"].loc[state_name, p]) for p in players}
        payload["G"] = float(setup["geoengineering"].loc[state_name, "G"])
        rows.append(payload)
    return pd.DataFrame(rows, index=ordered_states)


def find_equilibrium(config, output_file=None, solver_params=None, verbose=True, description=None,
                     load_from_checkpoint=False, random_seed=None, logger=None, save_payoffs=False,
                     save_unverified=False, diagnostics=False,
                     approval_margin_threshold: float = 1e-3,
                     solver_approach: str = "annealing"):
    """
    Find equilibrium for a given configuration.

    Args:
        config: Configuration dictionary
        output_file: Path to save the equilibrium strategy profile (optional)
                    Use 'auto' to generate filename automatically
        solver_params: Dictionary of solver parameters (optional)
        verbose: Whether to print progress
        description: Optional description/tag for filename generation
        load_from_checkpoint: Whether to load from checkpoint if it exists
        random_seed: Random seed for initialization (if None, generates one)
        logger: Logger instance (optional, will be created if not provided)
        save_payoffs: If True, save derived payoff table under payoff_tables/
        save_unverified: If True, save the strategy profile even when final
                        equilibrium verification fails
        diagnostics: If True, attach machine-readable run diagnostics
        approval_margin_threshold: Threshold used in diagnostics for
                                   classifying small approval margins
        solver_approach: One of 'annealing', 'support_enumeration',
                         'active_set', or 'ordinal_ranking'

    Returns:
        Dictionary with equilibrium results
    """
    # Setup logger if not provided
    if logger is None:
        scenario_name = config.get('scenario_name', 'equilibrium')
        log_file = Path('./logs') / f"{scenario_name}.log"
        logger = get_logger(log_file=log_file)

    # Track runtime
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Setup experiment
    setup = setup_experiment(config)

    # Generate config hash for checkpoint identification
    config_hash = generate_config_hash(config, length=10)

    if save_payoffs:
        repo_root = Path(__file__).resolve().parents[2]
        payoff_dir = repo_root / "payoff_tables"
        scenario_name = config.get("scenario_name", "scenario")
        payoff_hash = config_hash[:6]
        payoff_path = payoff_dir / f"payoff_{scenario_name}_{payoff_hash}.xlsx"
        payoff_df = _build_saved_payoff_table(setup)
        source_label = Path(config["payoff_table"]).name if config.get("payoff_table") else "computed_from_scenario"
        payoff_metadata = {
            "scenario_name": scenario_name,
            "config_hash": payoff_hash,
            "players": ", ".join(setup["players"]),
            "source": source_label,
        }
        write_payoff_table_excel(
            payoff_df=payoff_df,
            excel_file_path=str(payoff_path),
            players=setup["players"],
            metadata=payoff_metadata,
            source_label=source_label,
        )
        if verbose:
            logger.info(f"Saved payoff table to: {payoff_path}")

    selected_solver_approach = solver_approach

    # Check if checkpoint exists and inform user
    checkpoint_dir = './checkpoints'
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{config_hash}.pkl"
    checkpoint_exists = checkpoint_path.exists()

    if verbose and selected_solver_approach == "annealing" and checkpoint_exists:
        if load_from_checkpoint:
            logger.info(f"Found existing checkpoint: {checkpoint_path}")
            logger.info("Will resume from checkpoint. Use --fresh to start from scratch.")
            logger.info("")
        else:
            logger.info(f"Found existing checkpoint: {checkpoint_path}")
            logger.info("Starting fresh (--fresh flag used). Checkpoint will be overwritten.")
            logger.info("")
    elif verbose and selected_solver_approach == "annealing" and not checkpoint_exists and load_from_checkpoint:
        logger.info("No existing checkpoint found. Starting fresh.")
        logger.info("")

    # Get solver parameters
    solver_params = _get_solver_params(config, solver_params)

    # Print solver parameters
    if verbose:
        _print_solver_params(solver_params, logger)

    # Create and run equilibrium solver
    solver = EquilibriumSolver(
        players=setup['players'],
        states=setup['state_names'],
        effectivity=setup['effectivity'],
        protocol=setup['protocol'],
        payoffs=setup['payoffs'],
        discounting=setup['discounting'],
        unanimity_required=setup['unanimity_required'],
        power_rule=setup['power_rule'],
        forbidden_proposals=setup.get('forbidden_proposals', frozenset()),
        verbose=verbose,
        random_seed=random_seed,
        initialization_mode=solver_params.get("initialization_mode", "uniform"),
        logger=logger
    )

    if verbose:
        logger.info(f"Random seed for initialization: {solver.random_seed}")
        logger.info("")

    if selected_solver_approach == "support_enumeration" and len(setup["players"]) != 3:
        raise ValueError(
            "solver_approach='support_enumeration' is currently only supported for 3-player cases."
        )

    if selected_solver_approach == "active_set":
        if verbose:
            logger.info("Using cycle-guided active-set solver.")
            logger.info("")
        found_strategy_df, solver_result = _run_active_set_solver(
            solver,
            solver_params,
            logger=logger,
        )
    elif selected_solver_approach == "support_enumeration":
        if verbose:
            logger.info("Using cycle-guided support-enumeration solver.")
            logger.info("")
        found_strategy_df, solver_result = _run_support_enumeration_solver(
            solver,
            solver_params,
            logger=logger,
        )
    elif selected_solver_approach == "ordinal_ranking":
        if verbose:
            logger.info("Using ordinal-ranking solver.")
            logger.info("")
        found_strategy_df, solver_result = _run_ordinal_ranking_solver(
            solver,
            solver_params,
            logger=logger,
        )
    elif selected_solver_approach == "annealing":
        found_strategy_df, solver_result = _run_solver(
            solver,
            solver_params,
            checkpoint_dir='./checkpoints',
            load_from_checkpoint=load_from_checkpoint,
            config_hash=config_hash,
            logger=logger
        )
    else:
        raise ValueError(
            f"Unknown solver_approach='{selected_solver_approach}'. "
            "Expected one of: annealing, support_enumeration, active_set, ordinal_ranking."
        )

    # Fill NaN values for non-committee members
    found_strategy_df_filled = found_strategy_df.copy()
    found_strategy_df_filled.fillna(0., inplace=True)

    # Compute verification
    V, P, P_proposals, P_approvals = _compute_verification(found_strategy_df_filled, setup)

    # Build result dictionary
    result = {
        'scenario_name': config.get('scenario_name', 'equilibrium'),
        'V': V,
        'P': P,
        'geoengineering': setup['geoengineering'],
        'payoffs': setup['payoffs'],
        'P_proposals': P_proposals,
        'P_approvals': P_approvals,
        'players': setup['players'],
        'state_names': setup['state_names'],
        'effectivity': setup['effectivity'],
        'strategy_df': found_strategy_df_filled,
        'solver_result': solver_result,
        'stopping_reason': solver_result.get('stopping_reason'),
        'random_seed': solver.random_seed,
        'solver_approach': selected_solver_approach,
    }

    # Verify equilibrium
    success, message, verification_detail = verify_equilibrium_detailed(result)
    result['verification_success'] = success
    result['verification_message'] = message
    result['verification_detail'] = verification_detail

    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("EQUILIBRIUM VERIFICATION")
        logger.info("=" * 80)
        logger.info(f"Status: {message}")
        logger.info("\nValue functions:")
        logger.info(f"\n{V}")
        logger.info("\nTransition probabilities:")
        logger.info(f"\n{P}")
        logger.info("\nGeoengineering levels:")
        logger.info(f"\n{setup['geoengineering']}")

    # Calculate runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save to file if requested. By default, skip unverified profiles unless
    # explicitly requested via save_unverified.
    should_save_output = output_file is not None and (success or save_unverified)
    final_output_file = output_file

    if output_file is not None and not should_save_output:
        if verbose:
            logger.warning("Skipping file output: equilibrium verification failed.")

    if should_save_output:
        # Generate filename if not provided explicitly
        if output_file == 'auto':
            output_file = generate_filename(config, description=description)
        final_output_file = output_file

        # Build metadata
        metadata = _build_metadata(
            config, setup, solver_params, solver_result, success,
            runtime_seconds, start_timestamp, end_timestamp, description,
            random_seed=solver.random_seed
        )

        # Save to file with value functions, transition matrix, and geo levels
        _save_to_file(found_strategy_df, output_file, setup, metadata, V, P, verbose, logger)

        if verbose and not success and save_unverified:
            logger.warning("Saved strategy profile even though equilibrium verification failed.")

    result["output_written"] = should_save_output
    result["output_file"] = final_output_file
    result["runtime_seconds"] = runtime_seconds
    if diagnostics:
        result["diagnostics"] = build_result_diagnostics(
            config=config,
            solver_params=solver_params,
            result=result,
            runtime_seconds=runtime_seconds,
            output_file=final_output_file,
            output_written=should_save_output,
            approval_margin_threshold=approval_margin_threshold,
        )

    return result


def _parse_players_from_payoff_table(path: Path) -> list[str]:
    """
    Parse RICE50x player codes from a payoff table filename.

    Strategy:
      - Strip extension and year suffix (trailing _YYYY segment).
      - Split the remainder by '_'.
      - Try to parse each segment as a concatenation of RICE50x codes.
      - Return the sorted union of all successfully parsed player names.

    Example: 'burke_usachnnde_2060.xlsx' → ['CHN', 'NDE', 'USA']
    """
    from lib.ingest_payoffs import _GLOBAL_TOKEN_MAP, _parse_deployers

    stem = path.stem  # e.g. 'burke_usachnnde_2060' or 'burke_usachnnde_2060-2080'
    if _parse_year_range_from_stem(stem) is not None:
        stem = stem.rsplit("_", 1)[0]  # strip year/range → 'burke_usachnnde'

    players: set[str] = set()
    for segment in stem.split("_"):
        # Try the segment as-is, then also strip a trailing '-...' suffix
        # (e.g. 'usachnnde-100' → try 'usachnnde' as well).
        candidates = [segment]
        if "-" in segment:
            candidates.append(segment.split("-")[0])
        for candidate in candidates:
            try:
                players.update(_parse_deployers(candidate, _GLOBAL_TOKEN_MAP))
                break
            except ValueError:
                pass  # not a country combination (e.g. 'burke', '-100'), skip

    if not players:
        raise ValueError(
            f"Could not parse any RICE50x player codes from filename '{path.name}'. "
            "Encode player codes in the stem, e.g. burke_usachnnde_2060.xlsx → CHN, NDE, USA."
        )

    return sorted(players)


def _infer_players_from_payoff_table(path: Path) -> list[str]:
    """
    Infer player names by reading the 'Payoffs' sheet column headers.

    Falls back to this when the filename does not encode RICE50x player codes
    (e.g. 'simple_cycle_usachnnde-60-reduced.xlsx').

    Path resolution matches _load_payoff_table: tries path as given, then
    payoff_tables/<basename>.
    """
    _default_dir = Path(__file__).parent.parent.parent / "payoff_tables"
    resolved = path
    if not resolved.exists():
        fallback = _default_dir / path.name
        if fallback.exists():
            resolved = fallback
        else:
            raise FileNotFoundError(
                f"Payoff table '{path.name}' not found. "
                f"Searched: {path.resolve()}, {fallback.resolve()}"
            )

    df = pd.read_excel(str(resolved), sheet_name="Payoffs", header=1, index_col=0)
    excluded_prefixes = ("W_SAI",)
    excluded_names = {"Source file"}
    players = [
        str(col) for col in df.columns
        if not any(str(col).startswith(p) for p in excluded_prefixes)
        and str(col) not in excluded_names
    ]
    if not players:
        raise ValueError(f"Could not infer players from payoff table columns in {path.name}")
    return players


def _infer_or_parse_players_from_payoff_table(path: Path) -> list[str]:
    """
    Infer player names from a payoff table, trying filename parsing first and
    falling back to reading the Excel sheet if the filename is non-standard.
    """
    try:
        return _parse_players_from_payoff_table(path)
    except (ValueError, ImportError):
        return _infer_players_from_payoff_table(path)


def is_valid_rice_payoff_table_filename(path: Path) -> bool:
    """Return True if the payoff table filename encodes a player set."""
    if path.suffix.lower() != ".xlsx":
        return False
    try:
        _parse_players_from_payoff_table(path)
    except ValueError:
        return False
    return True


def iter_valid_rice_payoff_tables(directory: Path) -> list[Path]:
    """Return valid RICE payoff tables in a directory, sorted by filename."""
    return sorted(
        path for path in directory.glob("*.xlsx")
        if is_valid_rice_payoff_table_filename(path)
    )


def _compute_hardness_metrics(result: Dict[str, Any], approval_margin_threshold: float = 1e-3) -> Dict[str, Any]:
    """Compute runtime-independent fragility metrics from a solved result."""
    players = result["players"]
    states = result["state_names"]
    V = result["V"]
    P_approvals = result["P_approvals"]
    effectivity = result["effectivity"]

    approval_margins: List[float] = []
    proposal_ambiguous_rows = 0
    proposal_min_best_gap = None

    for proposer in players:
        for current_state in states:
            expected_values = []
            for next_state in states:
                p_approved = P_approvals[(proposer, current_state, next_state)]
                p_rejected = 1.0 - p_approved
                V_current = float(V.loc[current_state, proposer])
                V_next = float(V.loc[next_state, proposer])
                expected_values.append(
                    p_approved * V_next + p_rejected * V_current
                )

                approvers = get_approval_committee(
                    effectivity, players, proposer, current_state, next_state
                )
                for approver in approvers:
                    margin = abs(float(V.loc[next_state, approver]) - float(V.loc[current_state, approver]))
                    approval_margins.append(margin)

            sorted_values = sorted(expected_values, reverse=True)
            if len(sorted_values) >= 2:
                best_gap = sorted_values[0] - sorted_values[1]
                proposal_min_best_gap = best_gap if proposal_min_best_gap is None else min(proposal_min_best_gap, best_gap)

            max_value = max(expected_values)
            argmax_count = sum(np.isclose(val, max_value, rtol=0.0, atol=1e-9) for val in expected_values)
            if argmax_count > 1:
                proposal_ambiguous_rows += 1

    nonzero_margins = [m for m in approval_margins if m > 0]
    min_nonzero_approval_margin = min(nonzero_margins) if nonzero_margins else None
    num_small_approval_margins = sum(m < approval_margin_threshold for m in nonzero_margins)

    return {
        "min_nonzero_approval_margin": min_nonzero_approval_margin,
        "num_small_approval_margins": num_small_approval_margins,
        "proposal_ambiguous_rows": proposal_ambiguous_rows,
        "proposal_min_best_gap": proposal_min_best_gap,
        "approval_margin_threshold": approval_margin_threshold,
    }


def build_result_diagnostics(
    config: Dict[str, Any],
    solver_params: Dict[str, Any],
    result: Dict[str, Any],
    runtime_seconds: float,
    output_file: str | None,
    output_written: bool,
    approval_margin_threshold: float = 1e-3,
) -> Dict[str, Any]:
    """Build machine-readable diagnostics for a single solver run."""
    diagnostics = {
        "scenario_name": config.get("scenario_name", "equilibrium"),
        "payoff_table": config.get("payoff_table"),
        "players": result["players"],
        "n_players": len(result["players"]),
        "solver_approach": result.get("solver_approach"),
        "runtime_seconds": runtime_seconds,
        "verification_success": result["verification_success"],
        "verification_message": result.get("verification_message", ""),
        "verification_message_first_line": result.get("verification_message", "").splitlines()[0] if result.get("verification_message") else "",
        "random_seed": result.get("random_seed"),
        "output_file": output_file,
        "output_written": output_written,
        "solver_params": solver_params.copy(),
    }

    solver_result = result.get("solver_result", {})
    diagnostics.update({
        "stopping_reason": solver_result.get("stopping_reason"),
        "outer_iterations": solver_result.get("outer_iterations"),
        "converged": solver_result.get("converged"),
        "final_tau_p": solver_result.get("final_tau_p"),
        "final_tau_r": solver_result.get("final_tau_r"),
    })
    diagnostics.update(_compute_hardness_metrics(result, approval_margin_threshold=approval_margin_threshold))
    return diagnostics


def _parse_year_range_from_stem(stem: str) -> tuple[int | None, int] | None:
    """
    Extract (start_year, end_year) from the last '_'-separated segment of a stem.

    Accepts:
      '...._2060'       → (None, 2060)   — no lower bound, sum up to 2060
      '...._2060-2080'  → (2060, 2080)   — inclusive range

    Returns None if the last segment is not a recognised year/range pattern.
    """
    import re
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    m = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", parts[1])
    if not m:
        return None
    if m.group(2):
        return int(m.group(1)), int(m.group(2))
    return None, int(m.group(1))


def _parse_year_range_arg(year_range: str, parser) -> tuple[int | None, int]:
    """
    Parse a year range CLI argument.

    Accepts:
      'YYYY'       → (None, YYYY)   — no lower bound, sum up to YYYY
      'YYYY-YYYY'  → (YYYY, YYYY)   — inclusive range
    """
    import re

    raw = year_range.strip()
    m = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", raw)
    if not m:
        parser.error(
            f"Invalid --payoff-year-range '{year_range}'. "
            "Use 'YYYY' or 'YYYY-YYYY' (e.g. 2300 or 2035-2300)."
        )
    if m.group(2):
        start_year = int(m.group(1))
        end_year = int(m.group(2))
        if start_year > end_year:
            parser.error(
                f"Invalid --payoff-year-range '{year_range}': start year must be <= end year."
            )
        return start_year, end_year
    return None, int(m.group(1))


def _run_auto_ingest(payoff_table_arg, parser) -> str:
    """
    Produce a payoff table file via lib.ingest_payoffs if it does not already exist.

    The year range is inferred from the filename stem:
      burke_usachnnde_2060.xlsx      → up to 2060
      burke_usachnnde_2060-2080.xlsx → years 2060–2080 inclusive

    Path resolution mirrors _load_payoff_table: a bare filename is looked up under
    payoff_tables/ in the repo root.

    Args:
        payoff_table_arg: Value of --payoff-table (filename or path).
        parser: ArgumentParser instance used for error reporting.

    Returns:
        Resolved absolute path to the payoff table file as a string.
    """
    if payoff_table_arg is None:
        parser.error(
            "--auto-ingest requires --payoff-table to name the target file; "
            "the year range is inferred from the filename "
            "(e.g. burke_usachnnde_2060.xlsx or burke_usachnnde_2060-2080.xlsx)"
        )

    pt_path = Path(payoff_table_arg)
    if not pt_path.suffix:
        pt_path = pt_path.with_suffix(".xlsx")
    if pt_path.parent == Path("."):
        repo_root = Path(__file__).parent.parent.parent
        pt_path = repo_root / "payoff_tables" / pt_path

    if pt_path.exists():
        print(f"[auto-ingest] {pt_path.name} already exists — skipping ingest")
    else:
        year_range = _parse_year_range_from_stem(pt_path.stem)
        if year_range is None:
            parser.error(
                f"Cannot infer year range from '{pt_path.name}'. "
                "Encode the year in the stem, e.g. burke_usachnnde_2060.xlsx "
                "or burke_usachnnde_2060-2080.xlsx"
            )
        start_year, end_year = year_range
        folder_name = pt_path.stem.rsplit("_", 1)[0]
        from lib.ingest_payoffs import ingest, DEFAULT_RESULTS_DIR
        input_dir = DEFAULT_RESULTS_DIR / folder_name
        year_desc = f"{start_year}-{end_year}" if start_year else f"up to {end_year}"
        print(f"[auto-ingest] {pt_path.name} not found — running ingest (years {year_desc})")
        print(f"[auto-ingest] {input_dir} → {pt_path}")
        try:
            ingest(input_dir=input_dir, output_path=pt_path,
                   players=None, start_year=start_year, end_year=end_year)
        except FileNotFoundError as e:
            parser.error(
                f"auto-ingest failed: {e}\n"
                f"  Expected GDX files in {input_dir}"
            )

    return str(pt_path)


def main():
    """Command-line interface for finding equilibria."""
    # Setup logger early so it's available for all output
    logger = None

    # Get available scenarios
    available_scenarios = list_scenarios()
    
    parser = argparse.ArgumentParser(
        description='Find equilibrium strategy profiles for coalition formation games',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Find equilibrium for a single scenario
  python -m lib.equilibrium.find weak_governance_n3

  # Run multiple scenarios sequentially
  python -m lib.equilibrium.find weak_governance_n4 power_threshold_n4 power_threshold_no_unanimity_n4

  # With custom output file (single scenario only)
  python -m lib.equilibrium.find power_threshold_n3 -o my_results.xlsx

  # Start fresh (ignore existing checkpoints)
  python -m lib.equilibrium.find weak_governance_n3 --fresh

Available scenarios (use --list-scenarios to see all):
  {', '.join(available_scenarios[:10])}...
        """
    )
    parser.add_argument(
        'scenario',
        nargs='*',
        help='One or more scenarios to run sequentially (see --list-scenarios for options)'
    )
    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List all available scenarios and exit'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='auto',
        help='Output file path for equilibrium strategy. Use "auto" to generate filename automatically (default: auto)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Optional custom description/tag for filename (e.g., "RICE50", "lowdiscount")'
    )
    parser.add_argument(
        '--max-outer-iter',
        type=int,
        default=None,
        help='Maximum number of outer (annealing) iterations (optional)'
    )
    parser.add_argument(
        '--max-inner-iter',
        type=int,
        default=None,
        help='Maximum number of inner (fixed-point) iterations (optional)'
    )
    parser.add_argument(
        '--verify-every-n',
        type=int,
        default=None,
        help='Run early verification only every nth stable outer iteration (default: scenario-specific, typically 1 or 10)'
    )
    parser.add_argument(
        '--max-cycles-at-tau-min',
        type=int,
        default=None,
        help='Stop if this many low-tau cycles are detected consecutively (default: scenario-specific)'
    )
    parser.add_argument(
        '--cycle-break-tau-threshold',
        type=float,
        default=None,
        help='Tau threshold for cycle stop gating (default: tau_min * (1 + tau_margin))'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for strategy initialization (if not provided, generates randomly)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Force starting from scratch, ignoring any existing checkpoints (default: auto-resume from checkpoint if exists)'
    )
    parser.add_argument(
        '--payoff-table',
        type=str,
        default=None,
        help=(
            'Path to a precomputed payoff table Excel file (as produced by lib.ingest_payoffs). '
            'When provided, payoffs are loaded from this file instead of being computed from '
            'the scenario\'s temperature/damage parameters. '
            'State names in the table must match the scenario\'s state names exactly.'
        )
    )
    parser.add_argument(
        '--allow-non-canonical-states',
        action='store_true',
        help=(
            'Allow non-canonical state names from the payoff table (e.g. "(CHN)" in a '
            '2-player reduced model where "(CHN)" and "(USA)" replace the grand coalition). '
            'Synthetic coalition maps are built from the state names. '
            'Only use this when the payoff table defines a custom, reduced state space.'
        )
    )
    parser.add_argument(
        '--effectivity-rule',
        type=str,
        default=None,
        help=(
            'Effectivity rule to use when generating approval committees. '
            'Available rules: heyen_lehtomaa_2021 (default), unanimous_consent, deployer_exit, free_exit. '
            'deployer_exit: the named deployer can exit unilaterally to ( ), all other transitions require unanimity. '
            'free_exit: any player can exit to the no-coordination state unilaterally; entering/switching coordination requires unanimity.'
        )
    )
    parser.add_argument(
        '--auto-ingest',
        action='store_true',
        help=(
            'Run lib.ingest_payoffs with its default parameters before solving and use the '
            'resulting payoff table as --payoff-table. '
            'Output is placed in payoff_tables/<input_dir_name>_<cutoff_year>.xlsx. '
            'Mutually exclusive with --payoff-table.'
        )
    )
    parser.add_argument(
        '--payoff-year-range',
        type=str,
        default=None,
        help=(
            "Year range to use when (re)ingesting payoffs before solving. "
            "Accepts 'YYYY' (sum up to year) or 'YYYY-YYYY' (inclusive range). "
            "Requires --payoff-table. Overrides year inference from the payoff filename."
        )
    )
    parser.add_argument(
        '--payoff-input-dir',
        type=str,
        default=None,
        help=(
            "Directory containing source .gdx files for payoff ingestion. "
            "Default: inferred from payoff filename under DEFAULT_RESULTS_DIR."
        )
    )
    parser.add_argument(
        '--save-payoffs',
        action='store_true',
        help='Save the derived payoff table to payoff_tables/ before solving'
    )
    parser.add_argument(
        '--save-unverified',
        action='store_true',
        help='Save the strategy profile even if final equilibrium verification fails'
    )
    parser.add_argument(
        '--diagnostics-json',
        action='store_true',
        help='Print machine-readable diagnostics as one JSON object per scenario after each run'
    )
    parser.add_argument(
        '--approval-margin-threshold',
        type=float,
        default=1e-3,
        help='Threshold used in diagnostics for counting small approval margins (default: 1e-3)'
    )
    parser.add_argument(
        '--solver-approach',
        type=str,
        choices=['annealing', 'support_enumeration', 'active_set', 'ordinal_ranking'],
        default='annealing',
        help=(
            "Solver approach to use: 'annealing' for the legacy smoothed solver, "
            "'support_enumeration' for the cycle-guided support search, "
            "'active_set' for the stricter cycle-guided active-set search, "
            "'ordinal_ranking' for exhaustive search over ordinal value orders."
        )
    )
    parser.add_argument(
        '--initialization-mode',
        type=str,
        choices=['uniform', 'one_hot', 'payoff_structured'],
        default=None,
        help=(
            "Initialization mode for the starting strategy profile: "
            "'uniform' for dense random proposals and continuous approvals, "
            "'one_hot' for one-hot proposal rows and binary approvals, "
            "'payoff_structured' for static-payoff argmax proposals and sign-based binary approvals."
        )
    )
    parser.add_argument(
        '--ordinal-ranking-max-combinations',
        type=int,
        default=None,
        help='Optional cap on ranking triples tested by solver_approach=ordinal_ranking'
    )
    parser.add_argument(
        '--ordinal-ranking-order',
        type=str,
        choices=['lexicographic', 'payoff'],
        default=None,
        help="Deterministic ranking enumeration order for solver_approach=ordinal_ranking (default: payoff)"
    )
    parser.add_argument(
        '--ordinal-ranking-shuffle',
        action='store_true',
        help='Enumerate ordinal-ranking triples in random order'
    )
    parser.add_argument(
        '--ordinal-ranking-random-seed',
        type=int,
        default=None,
        help='Random seed used with --ordinal-ranking-shuffle'
    )
    parser.add_argument(
        '--ordinal-ranking-progress-every',
        type=int,
        default=None,
        help='Print ordinal-ranking progress every N combinations (default: 100; 0 disables progress bar)'
    )
    parser.add_argument(
        '--ordinal-ranking-workers',
        type=int,
        default=None,
        help='Process workers for solver_approach=ordinal_ranking (default: 8)'
    )
    parser.add_argument(
        '--ordinal-ranking-batch-size',
        type=int,
        default=None,
        help='Batch size for solver_approach=ordinal_ranking multiprocessing (default: 5000)'
    )
    parser.add_argument(
        '--ordinal-ranking-weak-orders',
        action='store_true',
        help='Allow tied value orders (weak orders) in solver_approach=ordinal_ranking'
    )

    args = parser.parse_args()

    if args.auto_ingest or args.payoff_year_range is not None:
        if args.payoff_table is None:
            parser.error(
                "--auto-ingest/--payoff-year-range requires --payoff-table to name the target file."
            )
        # Resolve path the same way _load_payoff_table does
        pt_path = Path(args.payoff_table)
        if not pt_path.suffix:
            pt_path = pt_path.with_suffix(".xlsx")
        if pt_path.parent == Path("."):
            repo_root = Path(__file__).parent.parent.parent
            pt_path = repo_root / "payoff_tables" / pt_path

        if args.payoff_year_range is not None:
            start_year, end_year = _parse_year_range_arg(args.payoff_year_range, parser)
            force_reingest = True
        else:
            year_range = _parse_year_range_from_stem(pt_path.stem)
            if year_range is None:
                parser.error(
                    f"Cannot infer year range from '{pt_path.name}'. "
                    "Encode the year in the stem, e.g. burke_usachnnde_2060.xlsx "
                    "or burke_usachnnde_2060-2080.xlsx, "
                    "or pass --payoff-year-range explicitly."
                )
            start_year, end_year = year_range
            force_reingest = args.fresh

        if args.payoff_input_dir is not None:
            input_dir = Path(args.payoff_input_dir)
        else:
            # Infer input folder from payoff filename by stripping a trailing year/range segment.
            folder_name = pt_path.stem
            if _parse_year_range_from_stem(pt_path.stem) is not None:
                folder_name = pt_path.stem.rsplit("_", 1)[0]
            from lib.ingest_payoffs import DEFAULT_RESULTS_DIR
            input_dir = DEFAULT_RESULTS_DIR / folder_name

        if pt_path.exists() and not force_reingest:
            print(f"[auto-ingest] {pt_path.name} already exists — skipping ingest")
        else:
            from lib.ingest_payoffs import ingest
            year_desc = f"{start_year}-{end_year}" if start_year else f"up to {end_year}"
            action = "re-ingesting" if pt_path.exists() else "running ingest"
            print(f"[auto-ingest] {pt_path.name} {'exists' if pt_path.exists() else 'not found'} — {action} (years {year_desc})")
            print(f"[auto-ingest] {input_dir} → {pt_path}")
            try:
                ingest(input_dir=input_dir, output_path=pt_path,
                       players=None, start_year=start_year, end_year=end_year)
            except FileNotFoundError as e:
                parser.error(
                    f"auto-ingest failed: {e}\n"
                    f"  Expected GDX files in {input_dir}"
                )
        args.payoff_table = str(pt_path)

    # Handle --list-scenarios flag
    if args.list_scenarios:
        print("Available scenarios:")
        print("\n3-player scenarios:")
        for name in list_scenarios(filter_players=3):
            print(f"  - {name}")
        print("\n4-player scenarios:")
        for name in list_scenarios(filter_players=4):
            print(f"  - {name}")
        print("\n5-player scenarios:")
        for name in list_scenarios(filter_players=5):
            print(f"  - {name}")
        return

    # Require at least one scenario if not listing
    if not args.scenario:
        parser.error("at least one scenario is required (use --list-scenarios to see available options)")

    # --output only makes sense for a single scenario
    if len(args.scenario) > 1 and args.output != 'auto':
        parser.error("--output cannot be used with multiple scenarios (filenames are auto-generated)")

    # By default, load from checkpoint unless --fresh is specified
    load_from_checkpoint = not args.fresh

    # Solver parameters: only include when provided on CLI so file defaults remain
    solver_params = {}
    if args.max_outer_iter is not None:
        solver_params['max_outer_iter'] = args.max_outer_iter
    if args.max_inner_iter is not None:
        solver_params['max_inner_iter'] = args.max_inner_iter
    if args.verify_every_n is not None:
        solver_params['verify_every_n'] = args.verify_every_n
    if args.max_cycles_at_tau_min is not None:
        solver_params['max_cycles_at_tau_min'] = args.max_cycles_at_tau_min
    if args.cycle_break_tau_threshold is not None:
        solver_params['cycle_break_tau_threshold'] = args.cycle_break_tau_threshold
    if args.initialization_mode is not None:
        solver_params['initialization_mode'] = args.initialization_mode
    if args.ordinal_ranking_max_combinations is not None:
        solver_params['ordinal_ranking_max_combinations'] = args.ordinal_ranking_max_combinations
    if args.ordinal_ranking_order is not None:
        solver_params['ordinal_ranking_order'] = args.ordinal_ranking_order
    if args.ordinal_ranking_shuffle:
        solver_params['ordinal_ranking_shuffle'] = True
    if args.ordinal_ranking_random_seed is not None:
        solver_params['ordinal_ranking_random_seed'] = args.ordinal_ranking_random_seed
    if args.ordinal_ranking_progress_every is not None:
        solver_params['ordinal_ranking_progress_every'] = args.ordinal_ranking_progress_every
    if args.ordinal_ranking_workers is not None:
        solver_params['ordinal_ranking_workers'] = args.ordinal_ranking_workers
    if args.ordinal_ranking_batch_size is not None:
        solver_params['ordinal_ranking_batch_size'] = args.ordinal_ranking_batch_size
    if args.ordinal_ranking_weak_orders:
        solver_params['ordinal_ranking_weak_orders'] = True

    results_summary = []

    for scenario_name in args.scenario:
        # Get scenario configuration
        try:
            config = get_scenario(scenario_name)
        except KeyError as e:
            parser.error(str(e))

        # Inject payoff table path into config if provided
        if args.payoff_table is not None:
            config['payoff_table'] = args.payoff_table

        # Inject non-canonical states flag if set
        if args.allow_non_canonical_states:
            config['allow_non_canonical_states'] = True

        # Inject effectivity rule if specified
        if args.effectivity_rule is not None:
            config['effectivity_rule'] = args.effectivity_rule

        # When players are not hardcoded in the scenario, derive them from the filename
        if config.get('players') is None:
            if args.payoff_table is None:
                parser.error(
                    f"Scenario '{scenario_name}' requires --payoff-table: "
                    "players are inferred from the filename "
                    "(e.g. burke_usachnnde_2060.xlsx → CHN, NDE, USA)."
                )
            try:
                players = _parse_players_from_payoff_table(Path(args.payoff_table))
            except ValueError as e:
                parser.error(str(e))
            from lib.equilibrium.scenarios import fill_players
            config = fill_players(config, players)

        # Setup logger with scenario-specific log file
        log_file = Path('./logs') / f"{config['scenario_name']}.log"
        logger = get_logger(log_file=log_file)

        logger.info("=" * 80)
        logger.info(f"FINDING EQUILIBRIUM FOR: {scenario_name}")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Power rule: {config['power_rule']}")
        logger.info(f"  Minimum power: {config.get('min_power', 'N/A')}")
        logger.info(f"  Unanimity required: {config['unanimity_required']}")
        logger.info(f"  Discounting: {config['discounting']}")
        if config.get('payoff_table'):
            logger.info(f"  Payoff source: precomputed table ({config['payoff_table']})")
        if args.description:
            logger.info(f"  Description: {args.description}")

        # Print player parameters table
        players = config['players']
        logger.info("")
        logger.info("Player parameters:")

        header = f"  {'Parameter':<15}" + "".join(f"{p:>8}" for p in players)
        logger.info(header)
        logger.info("  " + "-" * (15 + 8 * len(players)))

        base_temps = config['base_temp']
        logger.info(f"  {'Base Temp':<15}" + "".join(f"{base_temps[p]:>8.1f}" for p in players))
        ideal_temps = config['ideal_temp']
        logger.info(f"  {'Ideal Temp':<15}" + "".join(f"{ideal_temps[p]:>8.1f}" for p in players))
        delta_temps = config['delta_temp']
        logger.info(f"  {'ΔTemp':<15}" + "".join(f"{delta_temps[p]:>8.2f}" for p in players))
        damages = config['m_damage']
        logger.info(f"  {'Damage Coeff':<15}" + "".join(f"{damages[p]:>8.2f}" for p in players))
        powers = config['power']
        logger.info(f"  {'Power':<15}" + "".join(f"{powers[p]:>8.3f}" for p in players))
        protocols = config['protocol']
        logger.info(f"  {'Protocol':<15}" + "".join(f"{protocols[p]:>8.3f}" for p in players))
        logger.info("")

        result = find_equilibrium(
            config,
            output_file=args.output,
            solver_params=solver_params,
            verbose=not args.quiet,
            description=args.description,
            load_from_checkpoint=load_from_checkpoint,
            random_seed=args.seed,
            logger=logger,
            save_payoffs=args.save_payoffs,
            save_unverified=args.save_unverified,
            diagnostics=args.diagnostics_json,
            approval_margin_threshold=args.approval_margin_threshold,
            solver_approach=args.solver_approach,
        )

        results_summary.append((scenario_name, result['verification_success'],
                                 result.get('verification_message', '')))

        if args.diagnostics_json and 'diagnostics' in result:
            print(json.dumps(result['diagnostics'], default=str))

        if result['verification_success']:
            logger.info("\n" + "=" * 80)
            logger.info(f"Scenario: {scenario_name}")
            logger.info(f"Runtime: {result['runtime_seconds']:.2f}s")
            logger.success("SUCCESS: Found valid equilibrium!")
            logger.info("=" * 80)
        else:
            logger.warning("\n" + "=" * 80)
            logger.warning(f"Scenario: {scenario_name}")
            logger.warning(f"Runtime: {result['runtime_seconds']:.2f}s")
            logger.warning("WARNING: Equilibrium verification failed!")
            stopping_reason = result.get('stopping_reason')
            if stopping_reason:
                logger.warning(f"Stopping reason: {stopping_reason}")
            logger.warning(result['verification_message'])
            logger.warning("=" * 80)

    # Print summary when multiple scenarios were run
    if len(args.scenario) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for name, success, _ in results_summary:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  {status}  {name}")
        print("=" * 80)


if __name__ == "__main__":
    main()
