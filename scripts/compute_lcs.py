#!/usr/bin/env python3
"""Compute stability sets (LCS and LCCS) for a payoff table.

Stability concepts:
    - LCS (Strict):  Chwe (1994), based on indirect strict dominance.
    - LCS (Weak):    Mauleon & Vannetelbosch (2004), based on indirect weak dominance.
    - LCCS (Strict): Mauleon & Vannetelbosch (2004) Def 5, cautious stability (strict).
    - LCCS (Weak):   Mauleon & Vannetelbosch (2004) Def 6, cautious stability (weak).

Usage examples:
    PYTHONPATH=. python3 scripts/compute_lcs.py \
        payoff_tables/burke_usarusndechn_2035-2060_payoff_2035-2300.xlsx \
        --scenario power_threshold_RICE_n3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.effectivity import get_effectivity
from lib.equilibrium.find import setup_experiment
from lib.equilibrium.lcs import compute_lcs, compute_lccs
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
        description="Compute farsighted stability sets (LCS, LCCS) for a payoff table."
    )
    parser.add_argument(
        "payoff_table",
        help="Path to the payoff table .xlsx, or basename under payoff_tables/.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Named scenario to load (e.g. power_threshold_RICE_n3).",
    )
    parser.add_argument(
        "--effectivity-rule",
        type=str,
        default=None,
        choices=("heyen_lehtomaa_2021", "unanimous_consent", "deployer_exit", "free_exit", "adjacent_step"),
        help="Effectivity rule. Overrides whatever the scenario config specifies.",
    )
    parser.add_argument(
        "--allow-non-canonical-states",
        action="store_true",
        help="Allow non-canonical state names from reduced payoff tables.",
    )
    args = parser.parse_args()

    payoff_path = _resolve_payoff_file(args.payoff_table)

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
    geo: pd.DataFrame = setup["geoengineering"]

    effective_rule = args.effectivity_rule or config.get("effectivity_rule", "heyen_lehtomaa_2021")

    print("Farsighted Stability Analysis")
    print("=" * 60)
    print(f"file:             {payoff_path}")
    print(f"scenario:         {args.scenario or '(inferred)'}")
    print(f"effectivity_rule: {effective_rule}")
    print(f"players:          {players}")
    print(f"states:           {state_names}")
    print()

    print("Stability Sets")
    print("-" * 60)
    
    # Compute all sets
    results = {
        "LCS (Strict)":  compute_lcs(players, state_names, payoffs[players], effectivity, weak=False),
        "LCS (Weak)":    compute_lcs(players, state_names, payoffs[players], effectivity, weak=True),
        "LCCS (Strict)": compute_lccs(players, state_names, payoffs[players], effectivity, weak=False),
        "LCCS (Weak)":   compute_lccs(players, state_names, payoffs[players], effectivity, weak=True),
    }

    # Print summary table of sets
    print(f"{'Stability Concept':<15} | {'Size':>4} | {'Members'}")
    print("-" * 80)
    for name, (members, _) in results.items():
        members_sorted = sorted(list(members))
        print(f"{name:<15} | {len(members):>4} | {tuple(members_sorted)}")
    print()

    # W_SAI Analysis Comparison
    if geo is not None and "G" in geo.columns:
        print("W_SAI Analysis Comparison")
        print("-" * 60)
        total_sai = geo["G"]
        
        metrics = ["Mean", "Min", "Max", "Var", "Std"]
        header = f"{'Set':<15} |" + "".join(f"{m:>12}" for m in metrics)
        print(header)
        print("-" * 80)
        
        # Add 'Total' as reference
        all_vals = [total_sai.mean(), total_sai.min(), total_sai.max(), total_sai.var(), total_sai.std()]
        print(f"{'Total (all)':<15} |" + "".join(f"{v:>12.2f}" for v in all_vals))
        
        for name, (members, _) in results.items():
            if not members:
                print(f"{name:<15} |" + "".join(f"{'N/A':>12}" for _ in metrics))
                continue
            
            s = geo.loc[list(members), "G"]
            vals = [
                s.mean(),
                s.min(),
                s.max(),
                s.var() if len(s) > 1 else 0.0,
                s.std() if len(s) > 1 else 0.0
            ]
            print(f"{name:<15} |" + "".join(f"{v:>12.2f}" for v in vals))
        print()

    # Detailed indirect dominance for the default (LCS Strict)
    _, indirect_dom = results["LCS (Strict)"]
    if len(state_names) <= 5:
        print("Indirect Dominance (Strict)  (row a ≪ col b)")
        print("-" * 60)
        col_w = max(len(s) for s in state_names) + 2
        header_dom = f"{'a \\ b':<{col_w}}" + "".join(f"{b:<{col_w}}" for b in state_names)
        print(header_dom)
        for a in state_names:
            cells = "".join(
                f"{'<<':<{col_w}}" if indirect_dom.get((a, b), False) else f"{'.':<{col_w}}"
                for b in state_names
            )
            print(f"{a:<{col_w}}{cells}")
    else:
        print("(Indirect dominance table skipped for >5 states)")


if __name__ == "__main__":
    main()
