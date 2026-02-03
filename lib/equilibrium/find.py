"""
Find equilibrium strategy profiles using the smoothed fixed-point iteration algorithm.

This script allows you to find equilibria for different game configurations
and save the resulting strategy profiles to Excel files.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.equilibrium.solver import EquilibriumSolver
from lib.equilibrium.excel_writer import write_strategy_table_excel
from lib.utils import (derive_effectivity, get_payoff_matrix,
                       verify_equilibrium, get_geoengineering_levels)
from lib.probabilities import TransitionProbabilities
from lib.mdp import MDP


def setup_experiment(config):
    """Setup experiment configuration."""

    # Initialize countries
    all_countries = []
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

    # Initialize states
    W, T, C = all_countries
    coalition_map = {
        '( )': [Coalition([W]), Coalition([T]), Coalition([C])],
        '(TC)': [Coalition([W]), Coalition([T, C])],
        '(WC)': [Coalition([T]), Coalition([W, C])],
        '(WT)': [Coalition([C]), Coalition([W, T])],
        '(WTC)': [Coalition([W, T, C])]
    }

    states = [State(
        name=name,
        coalitions=coalition_map[name],
        all_countries=all_countries,
        power_rule=config["power_rule"],
        min_power=config["min_power"]
    ) for name in config["state_names"]]

    payoffs = get_payoff_matrix(states=states, columns=config["players"])
    geoengineering = get_geoengineering_levels(states=states)

    # Derive effectivity from template
    excel_file = "./strategy_tables/initial_strategy_profile.xlsx"
    template_df = pd.read_excel(excel_file, header=[0, 1], index_col=[0, 1, 2])

    effectivity = derive_effectivity(
        df=template_df,
        players=config["players"],
        states=config["state_names"]
    )

    return {
        'players': config["players"],
        'state_names': config["state_names"],
        'effectivity': effectivity,
        'protocol': config["protocol"],
        'payoffs': payoffs,
        'geoengineering': geoengineering,
        'discounting': config["discounting"],
        'unanimity_required': config["unanimity_required"],
        'power_rule': config["power_rule"],
        'min_power': config["min_power"]
    }


def find_equilibrium(config, output_file=None, solver_params=None, verbose=True):
    """
    Find equilibrium for a given configuration.

    Args:
        config: Configuration dictionary
        output_file: Path to save the equilibrium strategy profile (optional)
        solver_params: Dictionary of solver parameters (optional)
        verbose: Whether to print progress

    Returns:
        Dictionary with equilibrium results
    """
    if solver_params is None:
        solver_params = {}

    # Default solver parameters
    default_params = {
        'tau_p_init': 1.0,
        'tau_r_init': 1.0,
        'tau_decay': 0.95,
        'tau_min': 0.01,
        'max_outer_iter': 1000,  # Safety valve - rarely hit with convergence criterion
        'max_inner_iter': 100,
        'damping': 0.5,
        'inner_tol': 1e-6,
        'outer_tol': None,  # Defaults to 10*inner_tol
        'consecutive_tol': 2,
        'tau_margin': 0.01,
        'project_to_exact': True
    }

    # overwrite with parameters to resemble Jere's implementation
    default_params = {
        'tau_p_init': 1e-6, # if this is small, the softmax ...
        'tau_r_init': 1e-6, # if this is small, the sigmoid for acceptance becomes step-like
        'tau_decay': 0.9, # if this is close to 1, the annealing is slow
        'tau_min': 1e-8, # the temperature that has to be reached for convergence
        'max_outer_iter': 1000,  # Safety valve - convergence criterion will stop earlier
        'max_inner_iter': 10,
        'damping': 1,  # 1 means no damping
        'inner_tol': 1e-10,
        'outer_tol': 1e-9,  # If this is None, defaults to 10*inner_tol = 1e-9
        'consecutive_tol': 1, 
        'tau_margin': 0.01,
        'project_to_exact': True
    }
        
    default_params.update(solver_params)

    # Setup experiment
    setup = setup_experiment(config)

    # Run equilibrium solver
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

    # Solve for equilibrium
    found_strategy_df, solver_result = solver.solve(**default_params)

    # Fill NaN values for non-committee members
    found_strategy_df_filled = found_strategy_df.copy()
    found_strategy_df_filled.fillna(0., inplace=True)

    # Verify the found equilibrium
    tp = TransitionProbabilities(
        df=found_strategy_df_filled,
        effectivity=setup['effectivity'],
        players=setup['players'],
        states=setup['state_names'],
        protocol=setup['protocol'],
        unanimity_required=setup['unanimity_required']
    )
    P, P_proposals, P_approvals = tp.get_probabilities()

    mdp = MDP(
        n_states=len(setup['state_names']),
        transition_probs=P,
        discounting=setup['discounting']
    )

    V = pd.DataFrame(index=setup['state_names'], columns=setup['players'])
    for player in setup['players']:
        V.loc[:, player] = mdp.solve_value_func(setup['payoffs'].loc[:, player])

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

    if verbose:
        print("\n" + "=" * 80)
        print("EQUILIBRIUM VERIFICATION")
        print("=" * 80)
        print(f"Status: {message}")
        print("\nValue functions:")
        print(V)
        print("\nTransition probabilities:")
        print(P)
        print("\nGeoengineering levels:")
        print(setup['geoengineering'])

    # Save to file if requested
    if output_file is not None:
        # Ensure directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with custom Excel writer to match original format exactly
        write_strategy_table_excel(found_strategy_df, output_file, setup['players'],
                                   setup['effectivity'], setup['state_names'])

        if verbose:
            print(f"\nEquilibrium strategy saved to: {output_file}")

    result['verification_success'] = success
    result['verification_message'] = message

    return result


def main():
    """Command-line interface for finding equilibria."""
    parser = argparse.ArgumentParser(
        description='Find equilibrium strategy profiles for coalition formation games'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['weak_governance', 'power_threshold', 'power_threshold_no_unanimity', 'custom'],
        default='power_threshold',
        help='Predefined scenario to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for equilibrium strategy (e.g., ./strategy_tables/found_equilibrium.xlsx)'
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

    args = parser.parse_args()

    # Setup configuration based on scenario
    players = ["W", "T", "C"]
    n_players = len(players)

    base_config = dict(
        base_temp={"W": 21.5, "T": 14.0, "C": 11.5},
        ideal_temp={player: 13. for player in players},
        delta_temp={player: 3. for player in players},
        power={player: 1/n_players for player in players},
        protocol={player: 1/n_players for player in players},
        discounting=0.99,
        players=players,
        state_names=['( )', '(TC)', '(WC)', '(WT)', '(WTC)']
    )

    scenario_configs = {
        'weak_governance': {
            'experiment_name': 'weak_governance',
            'm_damage': {player: 1. for player in players},
            'power_rule': 'weak_governance',
            'min_power': None,
            'unanimity_required': True
        },
        'power_threshold': {
            'experiment_name': 'power_threshold',
            'm_damage': {player: 1. for player in players},
            'power_rule': 'power_threshold',
            'min_power': 0.5,
            'unanimity_required': True
        },
        'power_threshold_no_unanimity': {
            'experiment_name': 'power_threshold_no_unanimity',
            'm_damage': {"W": 0.75, "T": 1.25, "C": 1.},
            'power_rule': 'power_threshold',
            'min_power': 0.5,
            'unanimity_required': False
        }
    }

    config = {**base_config, **scenario_configs[args.scenario]}

    # Set output file if not specified
    if args.output is None:
        args.output = f"./strategy_tables/found_{args.scenario}.xlsx"

    # Solver parameters: only include when provided on CLI so file defaults remain
    solver_params = {}
    if args.max_outer_iter is not None:
        solver_params['max_outer_iter'] = args.max_outer_iter
    if args.max_inner_iter is not None:
        solver_params['max_inner_iter'] = args.max_inner_iter

    print("=" * 80)
    print(f"FINDING EQUILIBRIUM FOR: {args.scenario}")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Power rule: {config['power_rule']}")
    print(f"  Minimum power: {config.get('min_power', 'N/A')}")
    print(f"  Unanimity required: {config['unanimity_required']}")
    print(f"  Damage parameters: {config['m_damage']}")
    print(f"  Discounting: {config['discounting']}")
    print()

    # Find equilibrium
    result = find_equilibrium(
        config,
        output_file=args.output,
        solver_params=solver_params,
        verbose=not args.quiet
    )

    if result['verification_success']:
        print("\n" + "=" * 80)
        print("SUCCESS: Found valid equilibrium!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("WARNING: Equilibrium verification failed!")
        print(result['verification_message'])
        print("=" * 80)


if __name__ == "__main__":
    main()
