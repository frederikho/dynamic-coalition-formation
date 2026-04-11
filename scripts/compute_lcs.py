#!/usr/bin/env python3
"""Compute the Largest Consistent Set (Chwe 1994) for a payoff table.

Usage examples:

    # With a named scenario (reads effectivity rule from config or --effectivity-rule):
    PYTHONPATH=. python3 scripts/compute_lcs.py \\
        payoff_tables/simple_cycle_usachn-1.01-reduced-further.xlsx \\
        --scenario power_threshold_RICE_n3 \\
        --allow-non-canonical-states \\
        --effectivity-rule free_exit

    # Without a named scenario (players and protocol are inferred from the table):
    PYTHONPATH=. python3 scripts/compute_lcs.py \\
        payoff_tables/my_table.xlsx \\
        --effectivity-rule heyen_lehtomaa_2021
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.effectivity import get_effectivity
from lib.equilibrium.find import setup_experiment
from lib.equilibrium.lcs import compute_lcs
from lib.equilibrium.scenarios import fill_players, get_scenario


def _resolve_payoff_file(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = REPO_ROOT / "payoff_tables" / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find payoff file: {value!r}")


def _infer_players(path: Path) -> list[str]:
    df = pd.read_excel(path, sheet_name="Payoffs", header=1, index_col=0)
    excluded_prefixes = ("W_SAI",)
    excluded_names = {"Source file"}
    players = [
        str(col) for col in df.columns
        if not any(str(col).startswith(pfx) for pfx in excluded_prefixes)
        and str(col) not in excluded_names
    ]
    if not players:
        raise ValueError(f"Could not infer players from payoff table columns in {path.name}")
    return players


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the Largest Consistent Set (Chwe 1994) for a payoff table."
    )
    parser.add_argument(
        "payoff_table",
        help="Path to the payoff table .xlsx, or basename under payoff_tables/.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Named scenario to load (e.g. power_threshold_RICE_n3). "
             "If omitted, players and protocol are inferred from the table.",
    )
    parser.add_argument(
        "--effectivity-rule",
        type=str,
        default=None,
        choices=("heyen_lehtomaa_2021", "unanimous_consent", "deployer_exit", "free_exit"),
        help="Effectivity rule. Overrides whatever the scenario config specifies.",
    )
    parser.add_argument(
        "--allow-non-canonical-states",
        action="store_true",
        help="Allow non-canonical state names from reduced payoff tables.",
    )
    args = parser.parse_args()

    payoff_path = _resolve_payoff_file(args.payoff_table)

    # Build config the same way search_ordinal_rankings.py does.
    if args.scenario is not None:
        config = get_scenario(args.scenario)
        config["payoff_table"] = str(payoff_path)
        if config.get("players") is None:
            config = fill_players(config, _infer_players(payoff_path))
        if args.allow_non_canonical_states:
            config["allow_non_canonical_states"] = True
        if args.effectivity_rule is not None:
            config["effectivity_rule"] = args.effectivity_rule
    else:
        players = _infer_players(payoff_path)
        uniform = 1.0 / len(players)
        config = {
            "players": players,
            "protocol": {p: uniform for p in players},
            "payoff_table": str(payoff_path),
            "allow_non_canonical_states": args.allow_non_canonical_states,
        }
        if args.effectivity_rule is not None:
            config["effectivity_rule"] = args.effectivity_rule

    setup = setup_experiment(config)
    players: list[str] = setup["players"]
    state_names: list[str] = setup["state_names"]
    effectivity: dict = setup["effectivity"]
    payoffs: pd.DataFrame = setup["payoffs"]

    effective_rule = args.effectivity_rule or config.get("effectivity_rule", "heyen_lehtomaa_2021")

    print("LCS — Largest Consistent Set (Chwe 1994)")
    print("=" * 60)
    print(f"file:             {payoff_path}")
    print(f"scenario:         {args.scenario or '(inferred)'}")
    print(f"effectivity_rule: {effective_rule}")
    print(f"players:          {players}")
    print(f"states:           {state_names}")
    print()

    print("Short-term Payoffs")
    print("-" * 60)
    print(payoffs[players].to_string(float_format=lambda x: f"{x:.6f}"))
    print()

    lcs, indirect_dom = compute_lcs(players, state_names, payoffs[players], effectivity)

    lcs_sorted = tuple(sorted(lcs))
    print("Result")
    print("-" * 60)
    print(f"LCS:  {lcs_sorted}")
    print(f"size: {len(lcs)}")
    print()

    print("Indirect Dominance  (row a ≪ col b)")
    print("-" * 60)
    col_w = max(len(s) for s in state_names) + 2
    header = f"{'a \\ b':<{col_w}}" + "".join(f"{b:<{col_w}}" for b in state_names)
    print(header)
    for a in state_names:
        cells = "".join(
            f"{'<<':<{col_w}}" if indirect_dom.get((a, b), False) else f"{'.':<{col_w}}"
            for b in state_names
        )
        print(f"{a:<{col_w}}{cells}")


if __name__ == "__main__":
    main()
