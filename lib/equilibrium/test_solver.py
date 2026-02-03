"""
Test the equilibrium solver on the three known equilibria cases.

This script tests whether the equilibrium solver can recover the
hand-picked equilibria from random initialization.
"""

import pandas as pd
import numpy as np
from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import (derive_effectivity, get_payoff_matrix,
                       verify_equilibrium, get_geoengineering_levels)
from lib.probabilities import TransitionProbabilities
from lib.mdp import MDP


def setup_experiment(config):
    """Setup experiment configuration, similar to main.run_experiment."""

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

    # We need to derive effectivity from a template strategy table
    # Use the initial strategy profile as template
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


def test_equilibrium_case(experiment_name: str, config: dict,
                          true_strategy_file: str):
    """Test equilibrium solver on one case and compare with true equilibrium."""

    print("=" * 80)
    print(f"Testing: {experiment_name}")
    print("=" * 80)

    # Setup experiment
    setup = setup_experiment(config)

    # Load true equilibrium for comparison
    true_strategy_df = pd.read_excel(
        f"./strategy_tables/{true_strategy_file}",
        header=[0, 1], index_col=[0, 1, 2]
    )
    true_strategy_df.fillna(0., inplace=True)

    # Run equilibrium solver
    solver = EquilibriumSolver(
        players=setup['players'],
        states=setup['state_names'],
        effectivity=setup['effectivity'],
        protocol=setup['protocol'],
        payoffs=setup['payoffs'],
        discounting=setup['discounting'],
        unanimity_required=setup['unanimity_required'],
        verbose=True
    )

    # Solve for equilibrium
    found_strategy_df, solver_result = solver.solve(
        tau_p_init=1.0,
        tau_r_init=1.0,
        tau_decay=0.95,
        tau_min=0.01,
        max_outer_iter=50,
        max_inner_iter=100,
        damping=0.5,
        inner_tol=1e-6,
        project_to_exact=True
    )

    # Fill NaN values in found strategy (for non-committee members)
    found_strategy_df_filled = found_strategy_df.copy()
    found_strategy_df_filled.fillna(0., inplace=True)

    print("\n" + "-" * 80)
    print("VERIFICATION OF FOUND EQUILIBRIUM")
    print("-" * 80)

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
        'experiment_name': experiment_name,
        'V': V,
        'P': P,
        'geoengineering': setup['geoengineering'],
        'payoffs': setup['payoffs'],
        'P_proposals': P_proposals,
        'P_approvals': P_approvals,
        'players': setup['players'],
        'state_names': setup['state_names'],
        'effectivity': setup['effectivity'],
        'strategy_df': found_strategy_df_filled
    }

    success, message = verify_equilibrium(result)
    print(f"\nEquilibrium verification: {message}")

    print("\n" + "-" * 80)
    print("COMPARISON WITH TRUE EQUILIBRIUM")
    print("-" * 80)

    # Compare value functions
    tp_true = TransitionProbabilities(
        df=true_strategy_df,
        effectivity=setup['effectivity'],
        players=setup['players'],
        states=setup['state_names'],
        protocol=setup['protocol'],
        unanimity_required=setup['unanimity_required']
    )
    P_true, _, _ = tp_true.get_probabilities()

    mdp_true = MDP(
        n_states=len(setup['state_names']),
        transition_probs=P_true,
        discounting=setup['discounting']
    )

    V_true = pd.DataFrame(index=setup['state_names'], columns=setup['players'])
    for player in setup['players']:
        V_true.loc[:, player] = mdp_true.solve_value_func(
            setup['payoffs'].loc[:, player]
        )

    print("\nValue functions (found):")
    print(V)

    print("\nValue functions (true):")
    print(V_true)

    print("\nValue function difference:")
    V_diff = V.astype(float) - V_true.astype(float)
    print(V_diff)

    max_value_diff = np.max(np.abs(V_diff.values))
    print(f"\nMax value function difference: {max_value_diff:.10f}")

    # Compare transition matrices
    print("\nTransition matrix (found):")
    print(P)

    print("\nTransition matrix (true):")
    print(P_true)

    print("\nTransition matrix difference:")
    P_diff = P.astype(float) - P_true.astype(float)
    print(P_diff)

    max_P_diff = np.max(np.abs(P_diff.values))
    print(f"\nMax transition matrix difference: {max_P_diff:.10f}")

    # Check if value functions are close
    values_close = np.allclose(V.values.astype(float),
                               V_true.values.astype(float),
                               atol=1e-6)
    P_close = np.allclose(P.values.astype(float),
                          P_true.values.astype(float),
                          atol=1e-6)

    print("\n" + "=" * 80)
    if success and values_close and P_close:
        print("SUCCESS: Found equilibrium matches true equilibrium!")
    elif success and values_close and not P_close:
        print("PARTIAL SUCCESS: Found valid equilibrium with same value functions")
        print("                 but different transition probabilities.")
        print("                 This can happen when there are multiple equilibria.")
    elif success:
        print("PARTIAL SUCCESS: Found valid equilibrium but different from true.")
        print("                 Multiple equilibria may exist.")
    else:
        print("FAILURE: Found strategy is not a valid equilibrium.")
    print("=" * 80)

    return success, values_close, P_close


def main():
    """Test all three known equilibrium cases."""

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

    test_cases = [
        {
            'name': 'weak_governance',
            'config': {
                **base_config,
                'm_damage': {player: 1. for player in players},
                'power_rule': 'weak_governance',
                'min_power': None,
                'unanimity_required': True
            },
            'true_strategy': 'weak_governance.xlsx'
        },
        {
            'name': 'power_threshold',
            'config': {
                **base_config,
                'm_damage': {player: 1. for player in players},
                'power_rule': 'power_threshold',
                'min_power': 0.5,
                'unanimity_required': True
            },
            'true_strategy': 'power_threshold.xlsx'
        },
        {
            'name': 'power_threshold_no_unanimity',
            'config': {
                **base_config,
                'm_damage': {"W": 0.75, "T": 1.25, "C": 1.},
                'power_rule': 'power_threshold',
                'min_power': 0.5,
                'unanimity_required': False
            },
            'true_strategy': 'power_threshold_no_unanimity.xlsx'
        }
    ]

    results = []
    for test_case in test_cases:
        success, values_close, P_close = test_equilibrium_case(
            test_case['name'],
            test_case['config'],
            test_case['true_strategy']
        )
        results.append({
            'name': test_case['name'],
            'success': success,
            'values_close': values_close,
            'P_close': P_close
        })
        print("\n\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        status = "PASS" if r['success'] and r['values_close'] else "PARTIAL/FAIL"
        print(f"{r['name']:40s} {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
