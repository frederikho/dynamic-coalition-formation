#!/usr/bin/env python3
"""
Verify that our generated effectivity matches the effectivity from existing strategy tables.

This confirms that generate_effectivity_heyen_lehtomaa() implements the correct rules.
"""

import pandas as pd
import numpy as np
from lib.effectivity import heyen_lehtomaa_2021
from lib.utils import derive_effectivity


def test_effectivity_against_strategy_table(scenario_name, excel_file):
    """
    Compare generated effectivity with effectivity derived from strategy table.

    Args:
        scenario_name: Name of scenario (e.g., 'weak_governance')
        excel_file: Path to strategy table Excel file
    """
    print(f"\n{'='*80}")
    print(f"Testing: {scenario_name}")
    print(f"{'='*80}")

    # Read strategy table
    df = pd.read_excel(excel_file, header=[0, 1], index_col=[0, 1, 2])

    # Get players and states
    players = ['W', 'T', 'C']
    states = ['( )', '(TC)', '(WC)', '(WT)', '(WTC)']

    # Derive effectivity from strategy table (ground truth)
    effectivity_from_table = derive_effectivity(
        df=df,
        players=players,
        states=states
    )

    # Generate effectivity using our function
    effectivity_generated = heyen_lehtomaa_2021(players, states)

    # Compare
    all_keys = set(effectivity_from_table.keys()) | set(effectivity_generated.keys())

    mismatches = []
    for key in sorted(all_keys):
        proposer, current, next_state, responder = key

        val_table = effectivity_from_table.get(key, None)
        val_generated = effectivity_generated.get(key, None)

        if val_table != val_generated:
            mismatches.append({
                'proposer': proposer,
                'current': current,
                'next': next_state,
                'responder': responder,
                'table': val_table,
                'generated': val_generated
            })

    # Report results
    if not mismatches:
        print(f"✓ SUCCESS: All {len(all_keys)} effectivity entries match!")
        print(f"  Generated effectivity is correct for {scenario_name}")
        return True
    else:
        print(f"✗ FAILURE: {len(mismatches)} mismatches found out of {len(all_keys)} entries")
        print(f"\nMismatches:")
        for m in mismatches[:20]:  # Show first 20
            print(f"  {m['proposer']} proposes {m['current']} → {m['next']}, "
                  f"responder {m['responder']}: "
                  f"table={m['table']}, generated={m['generated']}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        return False


def main():
    """Test all available strategy tables."""

    scenarios = [
        ('Weak Governance', './strategy_tables/weak_governance.xlsx'),
        ('Power Threshold', './strategy_tables/power_threshold.xlsx'),
        ('Power Threshold (No Unanimity)', './strategy_tables/power_threshold_no_unanimity.xlsx'),
    ]

    all_passed = True

    for name, file in scenarios:
        try:
            passed = test_effectivity_against_strategy_table(name, file)
            all_passed = all_passed and passed
        except Exception as e:
            print(f"\n✗ ERROR testing {name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if all_passed:
        print("✓ All tests passed!")
        print("  generate_effectivity_heyen_lehtomaa() correctly implements")
        print("  the effectivity rules from Heyen & Lehtomaa (2021)")
    else:
        print("✗ Some tests failed")
        print("  The generated effectivity does not match the strategy tables")

    return 0 if all_passed else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
