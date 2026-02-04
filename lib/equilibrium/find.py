"""
Find equilibrium strategy profiles using the smoothed fixed-point iteration algorithm.

This script allows you to find equilibria for different game configurations
and save the resulting strategy profiles to Excel files.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

from lib.logging import get_logger
from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.equilibrium.solver import EquilibriumSolver
from lib.equilibrium.excel_writer import (
    write_strategy_table_excel,
    generate_filename,
    generate_config_hash
)
from lib.effectivity import heyen_lehtomaa_2021
from lib.utils import (
    derive_effectivity,
    get_payoff_matrix,
    verify_equilibrium,
    get_geoengineering_levels,
    list_members
)
from lib.probabilities_optimized import TransitionProbabilitiesOptimized as TransitionProbabilities
from lib.mdp import MDP
from lib.coalition_structures import generate_coalition_structures, generate_all_coalition_maps


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

    # Generate state names if not provided
    if "state_names" not in config or config["state_names"] is None:
        state_names = generate_coalition_structures(players)
        config["state_names"] = state_names
    else:
        state_names = config["state_names"]

    # Generate coalition maps for all states
    all_coalition_maps = generate_all_coalition_maps(players)

    # Create State objects for each coalition structure
    states = []
    for state_name in state_names:
        if state_name not in all_coalition_maps:
            raise ValueError(f"Unknown state: {state_name}")

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

    payoffs = get_payoff_matrix(states=states, columns=config["players"])
    geoengineering = get_geoengineering_levels(states=states)

    # Derive effectivity from template or generate
    template_file = config.get("template_file", None)
    if template_file and Path(template_file).exists():
        template_df = pd.read_excel(template_file, header=[0, 1], index_col=[0, 1, 2])
        effectivity = derive_effectivity(
            df=template_df,
            players=config["players"],
            states=state_names
        )
    else:
        # Generate effectivity based on Heyen & Lehtomaa (2021) rules
        effectivity = heyen_lehtomaa_2021(players, state_names)

    return {
        'players': config["players"],
        'state_names': state_names,
        'effectivity': effectivity,
        'protocol': config["protocol"],
        'payoffs': payoffs,
        'geoengineering': geoengineering,
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

    # Default parameters resembling Jere's implementation
    default_params = {
        'tau_p_init': 1e-6,
        'tau_r_init': 1e-6,
        'tau_decay': 0.6,
        'tau_min': 1e-8,
        'max_outer_iter': 400,
        'max_inner_iter': 100,
        'damping': 1,
        'inner_tol': 1e-10,
        'outer_tol': 1e-9,
        'consecutive_tol': 1,
        'tau_margin': 0.01,
        'project_to_exact': True
    }

    # Special parameters for non-unanimity scenarios (more complex, needs more smoothing)
    if not config['unanimity_required']:
        default_params.update({
            'tau_p_init': 1.0,
            'tau_r_init': 1.0,
            'tau_decay': 0.95,
            'tau_min': 0.01,
            'damping': 0.5,
            'max_inner_iter': 100,
        })

    # Special parameters for n>=4 (larger state space needs more smoothing)
    if len(config['players']) >= 4:
        default_params.update({
            'tau_p_init': 1,
            'tau_r_init': 1,
            'tau_decay': 0.9,
            'tau_min': 0.01,
            'damping': 0.6,
            'max_inner_iter': 250,
            'max_outer_iter': 500,
            'inner_tol': 1e-8,
            'outer_tol': 1e-8,
            'consecutive_tol': 1,
            'project_to_exact': False
        })

    # Update with user-provided parameters
    default_params.update(user_params)

    return default_params


def _print_solver_params(params, logger):
    """Print solver parameters in consistent order."""
    param_order = ['tau_p_init', 'tau_r_init', 'tau_decay', 'tau_min',
                  'max_outer_iter', 'max_inner_iter', 'damping',
                  'inner_tol', 'outer_tol', 'consecutive_tol',
                  'tau_margin', 'project_to_exact']

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
        return solver.solve(
            **params,
            checkpoint_dir=checkpoint_dir,
            load_from_checkpoint=load_from_checkpoint,
            config_hash=config_hash
        )
    except KeyboardInterrupt:
        if logger:
            logger.warning("\n\n" + "="*80)
            logger.warning("INTERRUPTED BY USER")
            logger.warning("="*80)
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
                    description=None):
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
        '--- GAME CONFIG ---': '',
        'n_players': len(setup['players']),
        'players': ', '.join(setup['players']),
        'n_states': len(setup['state_names']),
        'states': ', '.join(setup['state_names']),
        'power_rule': config['power_rule'],
        'min_power': config.get('min_power', 'N/A'),
        'unanimity_required': config['unanimity_required'],
        'discounting': config['discounting'],
        ' ': '',
        '--- PLAYER PARAMETERS ---': '',
    }

    # Add player-specific parameters
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

    # Add solver parameters used
    metadata['     '] = ''
    metadata['--- SOLVER PARAMETERS ---'] = ''
    param_order = ['tau_p_init', 'tau_r_init', 'tau_decay', 'tau_min',
                  'max_outer_iter', 'max_inner_iter', 'damping',
                  'inner_tol', 'outer_tol', 'consecutive_tol',
                  'tau_margin', 'project_to_exact']
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


def _save_to_file(strategy_df, output_file, setup, metadata, verbose=True, logger=None):
    """
    Save strategy profile to Excel file with metadata.

    Args:
        strategy_df: Strategy DataFrame
        output_file: Output file path
        setup: Setup dictionary
        metadata: Metadata dictionary
        verbose: Whether to print status
        logger: Logger instance
    """
    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with custom Excel writer to match original format exactly
    write_strategy_table_excel(
        strategy_df, output_file, setup['players'],
        setup['effectivity'], setup['state_names'], metadata=metadata
    )

    if verbose and logger:
        logger.info(f"\nEquilibrium strategy saved to: {output_file}")
        runtime_seconds = float(metadata['runtime_seconds'])
        logger.info(f"Runtime: {runtime_seconds//60:.0f}m {runtime_seconds%60:.1f}s")


def find_equilibrium(config, output_file=None, solver_params=None, verbose=True, description=None, load_from_checkpoint=False, logger=None):
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
        logger: Logger instance (optional, will be created if not provided)

    Returns:
        Dictionary with equilibrium results
    """
    # Setup logger if not provided
    if logger is None:
        scenario_name = config.get('experiment_name', 'equilibrium')
        log_file = Path('./logs') / f"{scenario_name}.log"
        logger = get_logger(log_file=log_file)

    # Track runtime
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Setup experiment
    setup = setup_experiment(config)

    # Generate config hash for checkpoint identification
    config_hash = generate_config_hash(config, length=10)

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
        verbose=verbose
    )

    found_strategy_df, solver_result = _run_solver(
        solver,
        solver_params,
        checkpoint_dir='./checkpoints',
        load_from_checkpoint=load_from_checkpoint,
        config_hash=config_hash,
        logger=logger
    )

    # Fill NaN values for non-committee members
    found_strategy_df_filled = found_strategy_df.copy()
    found_strategy_df_filled.fillna(0., inplace=True)

    # Compute verification
    V, P, P_proposals, P_approvals = _compute_verification(found_strategy_df_filled, setup)

    # Build result dictionary
    result = {
        'experiment_name': config.get('experiment_name', 'equilibrium'),
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
        'solver_result': solver_result
    }

    # Verify equilibrium
    success, message = verify_equilibrium(result)
    result['verification_success'] = success
    result['verification_message'] = message

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

    # Save to file if requested
    if output_file is not None:
        # Generate filename if not provided explicitly
        if output_file == 'auto':
            output_file = generate_filename(config, description=description)

        # Build metadata
        metadata = _build_metadata(
            config, setup, solver_params, solver_result, success,
            runtime_seconds, start_timestamp, end_timestamp, description
        )

        # Save to file
        _save_to_file(found_strategy_df, output_file, setup, metadata, verbose, logger)

    return result


def main():
    """Command-line interface for finding equilibria."""
    # Setup logger early so it's available for all output
    logger = None

    parser = argparse.ArgumentParser(
        description='Find equilibrium strategy profiles for coalition formation games'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['weak_governance', 'power_threshold', 'power_threshold_no_unanimity',
                 'weak_governance_n4', 'power_threshold_n4', 'custom'],
        default='power_threshold',
        help='Predefined scenario to use'
    )
    parser.add_argument(
        '--n-players',
        type=int,
        default=None,
        help='Number of players (overrides scenario default)'
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
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    parser.add_argument(
        '--load-from-checkpoint',
        action='store_true',
        help='Resume from checkpoint if it exists for this configuration'
    )

    args = parser.parse_args()

    # Determine number of players
    if args.scenario.endswith('_n4'):
        default_n_players = 4
    else:
        default_n_players = 3

    n_players = args.n_players if args.n_players is not None else default_n_players

    # Setup configuration based on number of players
    if n_players == 3:
        players = ["W", "T", "C"]
        base_config = dict(
            base_temp={"W": 21.5, "T": 14.0, "C": 11.5},
            ideal_temp={player: 13. for player in players},
            delta_temp={player: 3. for player in players},
            power={player: 1/n_players for player in players},
            protocol={player: 1/n_players for player in players},
            discounting=0.99,
            players=players,
            state_names=None  # Will be generated automatically
        )
    elif n_players == 4:
        players = ["W", "T", "C", "F"]
        base_config = dict(
            base_temp={"W": 21.5, "T": 14.0, "C": 11.5, "F": 9.0},
            ideal_temp={player: 13. for player in players},
            delta_temp={player: 3. for player in players},
            power={player: 1/n_players for player in players},
            protocol={player: 1/n_players for player in players},
            discounting=0.99,
            players=players,
            state_names=None  # Will be generated automatically
        )
    else:
        raise ValueError(f"Unsupported number of players: {n_players}")

    # Scenario-specific configurations
    scenario_base = args.scenario.replace('_n4', '')  # Remove n4 suffix for lookup

    scenario_configs = {
        'weak_governance': {
            'experiment_name': f'weak_governance_n{n_players}',
            'm_damage': {player: 1. for player in players},
            'power_rule': 'weak_governance',
            'min_power': None,
            'unanimity_required': True
        },
        'power_threshold': {
            'experiment_name': f'power_threshold_n{n_players}',
            'm_damage': {player: 1. for player in players},
            'power_rule': 'power_threshold',
            'min_power': 0.501,
            'unanimity_required': True
        },
        'power_threshold_no_unanimity': {
            'experiment_name': f'power_threshold_no_unanimity_n{n_players}',
            'm_damage': {player: (0.75 if player == 'W' else 1.25 if player == 'T' else 1.)
                        for player in players},
            'power_rule': 'power_threshold',
            'min_power': 0.501,
            'unanimity_required': False
        }
    }

    config = {**base_config, **scenario_configs[scenario_base]}

    # Setup logger with scenario-specific log file
    log_file = Path('./logs') / f"{config['experiment_name']}.log"
    logger = get_logger(log_file=log_file)

    # Solver parameters: only include when provided on CLI so file defaults remain
    solver_params = {}
    if args.max_outer_iter is not None:
        solver_params['max_outer_iter'] = args.max_outer_iter
    if args.max_inner_iter is not None:
        solver_params['max_inner_iter'] = args.max_inner_iter

    logger.info("=" * 80)
    logger.info(f"FINDING EQUILIBRIUM FOR: {args.scenario}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Power rule: {config['power_rule']}")
    logger.info(f"  Minimum power: {config.get('min_power', 'N/A')}")
    logger.info(f"  Unanimity required: {config['unanimity_required']}")
    logger.info(f"  Damage parameters: {config['m_damage']}")
    logger.info(f"  Discounting: {config['discounting']}")
    if args.description:
        logger.info(f"  Description: {args.description}")
    logger.info("")

    # Find equilibrium
    result = find_equilibrium(
        config,
        output_file=args.output,
        solver_params=solver_params,
        verbose=not args.quiet,
        description=args.description,
        load_from_checkpoint=args.load_from_checkpoint,
        logger=logger
    )

    if result['verification_success']:
        logger.info("\n" + "=" * 80)
        logger.success("SUCCESS: Found valid equilibrium!")
        logger.info("=" * 80)
    else:
        logger.warning("\n" + "=" * 80)
        logger.warning("WARNING: Equilibrium verification failed!")
        logger.warning(result['verification_message'])
        logger.warning("=" * 80)


if __name__ == "__main__":
    main()
