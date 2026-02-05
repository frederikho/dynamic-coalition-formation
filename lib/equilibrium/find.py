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
from lib.equilibrium.scenarios import get_scenario, list_scenarios
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
    get_deploying_coalitions,
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
    deploying_coalitions = get_deploying_coalitions(states=states)

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
    'tau_margin': ,
    'project_to_exact': should always be True actually
    """
    
    # Default parameters resembling Jere's implementation
    default_params = {
        'tau_p_init': 1e-6,
        'tau_r_init': 1e-6,
        'tau_decay': 0.6,
        'tau_min': 1e-8,
        'max_outer_iter': 400,
        'max_inner_iter': 100,
        'damping': 0,
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

    # Special parameters for n>=4
    if len(config['players']) >= 4:
        default_params.update({
            'tau_p_init': 1,
            'tau_r_init': 1,
            'tau_decay': 0.90,
            'tau_min': 0.001,
            'damping': 0.9,
            'max_inner_iter': 100,
            'max_outer_iter': 1000,
            'inner_tol': 2e-3,
            'outer_tol': 2e-3,
            'consecutive_tol': 3,
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
        'scenario_name': config.get('scenario_name', config.get('experiment_name', 'N/A')),
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


def _save_to_file(strategy_df, output_file, setup, metadata, V=None, verbose=True, logger=None):
    """
    Save strategy profile to Excel file with metadata.

    Args:
        strategy_df: Strategy DataFrame
        output_file: Output file path
        setup: Setup dictionary
        metadata: Metadata dictionary
        V: Value functions DataFrame (optional)
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
        deploying_coalitions=setup.get('deploying_coalitions', None)
    )

    if verbose and logger:
        logger.info(f"\nEquilibrium strategy saved to: {output_file}")
        runtime_seconds = float(metadata['runtime_seconds'])
        logger.info(f"Runtime: {runtime_seconds//60:.0f}m {runtime_seconds%60:.1f}s")


def find_equilibrium(config, output_file=None, solver_params=None, verbose=True, description=None, load_from_checkpoint=False, random_seed=None, logger=None):
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

    # Check if checkpoint exists and inform user
    checkpoint_dir = './checkpoints'
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{config_hash}.pkl"
    checkpoint_exists = checkpoint_path.exists()

    if verbose and checkpoint_exists:
        if load_from_checkpoint:
            logger.info(f"Found existing checkpoint: {checkpoint_path}")
            logger.info("Will resume from checkpoint. Use --fresh to start from scratch.")
            logger.info("")
        else:
            logger.info(f"Found existing checkpoint: {checkpoint_path}")
            logger.info("Starting fresh (--fresh flag used). Checkpoint will be overwritten.")
            logger.info("")
    elif verbose and not checkpoint_exists and load_from_checkpoint:
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
        verbose=verbose,
        random_seed=random_seed,
        logger=logger
    )

    if verbose:
        logger.info(f"Random seed for initialization: {solver.random_seed}")
        logger.info("")

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
        'solver_result': solver_result,
        'random_seed': solver.random_seed
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
            runtime_seconds, start_timestamp, end_timestamp, description,
            random_seed=solver.random_seed
        )

        # Save to file with value functions and geo levels
        _save_to_file(found_strategy_df, output_file, setup, metadata, V, verbose, logger)

    return result


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
  # Find equilibrium for weak governance scenario (3 players)
  python -m lib.equilibrium.find weak_governance_n3

  # Find equilibrium for power threshold with 4 players
  python -m lib.equilibrium.find power_threshold_n4

  # With custom output file
  python -m lib.equilibrium.find power_threshold_n3 -o my_results.xlsx

  # Start fresh (ignore existing checkpoint)
  python -m lib.equilibrium.find weak_governance_n3 --fresh

Available scenarios (use --list-scenarios to see all):
  {', '.join(available_scenarios[:10])}...
        """
    )
    parser.add_argument(
        'scenario',
        nargs='?',
        help='Scenario to run (see --list-scenarios for options)'
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

    args = parser.parse_args()

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

    # Require scenario if not listing
    if not args.scenario:
        parser.error("scenario is required (use --list-scenarios to see available options)")

    # By default, load from checkpoint unless --fresh is specified
    load_from_checkpoint = not args.fresh

    # Get scenario configuration
    try:
        config = get_scenario(args.scenario)
    except KeyError as e:
        parser.error(str(e))

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
    logger.info(f"  Discounting: {config['discounting']}")
    if args.description:
        logger.info(f"  Description: {args.description}")

    # Print player parameters table
    players = config['players']
    logger.info("")
    logger.info("Player parameters:")

    # Header
    header = f"  {'Parameter':<15}" + "".join(f"{p:>8}" for p in players)
    logger.info(header)
    logger.info("  " + "-" * (15 + 8 * len(players)))

    # Base temperature
    base_temps = config['base_temp']
    logger.info(f"  {'Base Temp':<15}" + "".join(f"{base_temps[p]:>8.1f}" for p in players))

    # Ideal temperature
    ideal_temps = config['ideal_temp']
    logger.info(f"  {'Ideal Temp':<15}" + "".join(f"{ideal_temps[p]:>8.1f}" for p in players))

    # Delta temperature (climate change)
    delta_temps = config['delta_temp']
    logger.info(f"  {'ΔTemp':<15}" + "".join(f"{delta_temps[p]:>8.2f}" for p in players))

    # Damage coefficient
    damages = config['m_damage']
    logger.info(f"  {'Damage Coeff':<15}" + "".join(f"{damages[p]:>8.2f}" for p in players))

    # Power share
    powers = config['power']
    logger.info(f"  {'Power':<15}" + "".join(f"{powers[p]:>8.3f}" for p in players))

    # Protocol probability
    protocols = config['protocol']
    logger.info(f"  {'Protocol':<15}" + "".join(f"{protocols[p]:>8.3f}" for p in players))

    logger.info("")

    # Find equilibrium
    result = find_equilibrium(
        config,
        output_file=args.output,
        solver_params=solver_params,
        verbose=not args.quiet,
        description=args.description,
        load_from_checkpoint=load_from_checkpoint,
        random_seed=args.seed,
        logger=logger
    )

    if result['verification_success']:
        logger.info("\n" + "=" * 80)
        logger.info(f"Scenario: {args.scenario}")
        logger.success("SUCCESS: Found valid equilibrium!")
        logger.info("=" * 80)
    else:
        logger.warning("\n" + "=" * 80)
        logger.warning(f"Scenario: {args.scenario}")
        logger.warning("WARNING: Equilibrium verification failed!")
        logger.warning(result['verification_message'])
        logger.warning("=" * 80)


if __name__ == "__main__":
    main()
