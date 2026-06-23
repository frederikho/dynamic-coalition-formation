#!/usr/bin/env python3
"""
Run Jere's original MIP-VFI on a RICE payoff table.

Usage:
    source .venv/bin/activate
    python scripts/run_jeres_vfi.py payoff_tables/kalkuhl_usachnrusnde_2035-2100.xlsx
    python scripts/run_jeres_vfi.py payoff_tables/kalkuhl_usachnrusnde_2035-2100.xlsx --n-restarts 50
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Suppress module-level print statements in jeres_implementation on import
_saved_stdout = sys.stdout
sys.stdout = io.StringIO()
import lib.equilibrium.mip_vfi.jeres_implementation as jere
sys.stdout = _saved_stdout

import pandas as pd

from lib.equilibrium.find import (
    setup_experiment, _parse_players_from_payoff_table,
    _compute_verification, _build_metadata, _save_to_file,
)
from lib.equilibrium.scenarios import get_scenario, fill_players
from lib.equilibrium.excel_writer import generate_filename
from lib.utils import verify_equilibrium_detailed


def _parse_fw_state_name(name: str, players: list[str]) -> tuple:
    """Convert framework state name → Jere canonical partition of player indices."""
    player_idx = {p: i for i, p in enumerate(players)}

    if name == "( )":
        return jere.canon_partition([frozenset({i}) for i in range(len(players))])

    groups = re.findall(r'\(([^)]+)\)', name)
    used: set[int] = set()
    partitions = []
    for group in groups:
        remaining = group
        members: set[int] = set()
        while remaining:
            matched = False
            for p in sorted(players, key=len, reverse=True):
                if remaining.startswith(p):
                    members.add(player_idx[p])
                    remaining = remaining[len(p):]
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Cannot parse '{remaining}' in state '{name}'")
        partitions.append(frozenset(members))
        used.update(members)

    for i in range(len(players)):
        if i not in used:
            partitions.append(frozenset({i}))

    return jere.canon_partition(partitions)


def load_rice_into_jere(payoff_table: str | Path) -> dict:
    """Load a RICE payoff table and patch Jere's module globals. Returns config."""
    path = Path(payoff_table)
    if not path.exists():
        path = REPO / "payoff_tables" / path.name

    try:
        players = _parse_players_from_payoff_table(path)
    except ValueError:
        import pandas as pd
        df = pd.read_excel(path, sheet_name="Metadata", header=None, index_col=0)
        players = [p.strip() for p in str(df.loc["players"].iloc[0]).split(",")]

    config = get_scenario("power_threshold_RICE_by_GDP")
    config = fill_players(config, players)
    config["payoff_table"] = str(path)
    config["effectivity_rule"] = "heyen_lehtomaa_2021"

    setup = setup_experiment(config)
    state_names = setup["state_names"]
    payoffs_df = setup["payoffs"]

    n = len(players)

    # Patch Jere's module globals
    jere.PLAYERS = players
    jere.N_PLAYERS = n
    jere.PROPOSER_PROBS = [config["protocol"][p] for p in players]

    all_parts = set(jere.canon_partition(p) for p in jere.gen_partitions(range(n)))
    jere.STATES = sorted(all_parts, key=lambda p: (len(p), sorted(len(c) for c in p)))
    jere.N_STATES = len(jere.STATES)
    jere.STATE_IDX = {s: i for i, s in enumerate(jere.STATES)}

    # Map framework state names → Jere partition keys
    fw_to_jere = {name: _parse_fw_state_name(name, players) for name in state_names}
    jere_to_fw = {v: k for k, v in fw_to_jere.items()}

    missing = [s for s in jere.STATES if s not in jere_to_fw]
    if missing:
        raise ValueError(f"Jere states not covered by framework: {missing}")

    payoffs_np = np.zeros((jere.N_STATES, n))
    for j_idx, j_state in enumerate(jere.STATES):
        fw_name = jere_to_fw[j_state]
        payoffs_np[j_idx] = payoffs_df.loc[fw_name, players].values

    jere.PAYOFFS = payoffs_np

    return config


def _build_state_mapping(state_names: list[str], players: list[str]) -> tuple[dict, dict]:
    """
    Build bidirectional index maps between framework state order and Jere state order.

    Returns:
        fw_to_jere_idx: {fw_state_idx: jere_state_idx}
        jere_to_fw_idx: {jere_state_idx: fw_state_idx}
    """
    fw_to_jere_idx: dict[int, int] = {}
    jere_to_fw_idx: dict[int, int] = {}
    for fw_idx, name in enumerate(state_names):
        jere_key = _parse_fw_state_name(name, players)
        j_idx = jere.STATE_IDX[jere_key]
        fw_to_jere_idx[fw_idx] = j_idx
        jere_to_fw_idx[j_idx] = fw_idx
    return fw_to_jere_idx, jere_to_fw_idx


def save_jere_solution(
    eq: dict,
    config: dict,
    output_file: str = "auto",
) -> str:
    """
    Convert a single Jere equilibrium to find_equilibrium.py format and save.

    Args:
        eq: One entry from jere.find_equilibria() — must have 'sigmas', 'alphas'.
        config: Config dict as returned by load_rice_into_jere().
        output_file: Path to write to, or 'auto' to generate automatically.

    Returns:
        Path of the written file.
    """
    from datetime import datetime
    from lib.equilibrium.solver import EquilibriumSolver
    from lib.equilibrium.mip_vfi import _arrays_to_strategy_df

    setup = setup_experiment(config)
    players = setup["players"]
    state_names = setup["state_names"]
    n_players = len(players)
    n_states = len(state_names)

    fw_to_jere_idx, jere_to_fw_idx = _build_state_mapping(state_names, players)

    sigmas = eq["sigmas"]   # sigmas[j_s][(i, j_sp)] = prob
    alphas = eq["alphas"]   # alphas[j_s][(j, j_sp)] = prob

    # Convert to framework arrays: all_sigmas[fw_s, i, fw_sp]
    all_sigmas = np.zeros((n_states, n_players, n_states))
    all_alphas = np.zeros((n_states, n_players, n_states))

    for fw_s in range(n_states):
        j_s = fw_to_jere_idx[fw_s]
        for (i, j_sp), prob in sigmas[j_s].items():
            if j_sp in jere_to_fw_idx:
                all_sigmas[fw_s, i, jere_to_fw_idx[j_sp]] = prob
        for (j_voter, j_sp), prob in alphas[j_s].items():
            if j_sp in jere_to_fw_idx:
                all_alphas[fw_s, j_voter, jere_to_fw_idx[j_sp]] = prob

    solver = EquilibriumSolver(
        players=players,
        states=state_names,
        effectivity=setup["effectivity"],
        protocol=setup["protocol"],
        payoffs=setup["payoffs"],
        discounting=setup["discounting"],
        unanimity_required=setup["unanimity_required"],
        power_rule=setup["power_rule"],
        forbidden_proposals=setup.get("forbidden_proposals", frozenset()),
        effectivity_rule=setup.get("effectivity_rule", "heyen_lehtomaa_2021"),
        verbose=False,
        geo_levels=setup.get("geoengineering"),
    )

    strategy_df = _arrays_to_strategy_df(solver, all_sigmas, all_alphas)
    strategy_df_filled = strategy_df.fillna(0.0)

    V, P, P_proposals, P_approvals = _compute_verification(strategy_df_filled, setup)

    result_for_verify = {
        "V": V, "P": P, "P_proposals": P_proposals, "P_approvals": P_approvals,
        "players": players, "state_names": state_names,
        "effectivity": setup["effectivity"],
        "forbidden_proposals": setup.get("forbidden_proposals", frozenset()),
        "strategy_df": strategy_df_filled,
    }
    success, message, _ = verify_equilibrium_detailed(result_for_verify)
    print(f"Verification: {message}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    solver_result = {
        "converged": True,
        "stopping_reason": "jeres_vfi",
        "outer_iterations": 0,
        "final_tau_p": 0.0,
        "final_tau_r": 0.0,
    }
    metadata = _build_metadata(
        config, setup, {}, solver_result, success,
        runtime_seconds=0,
        start_time=now, end_time=now,
        description="jeres_vfi",
    )

    if output_file == "auto":
        output_file = generate_filename(config, description="jeres_vfi")

    _save_to_file(strategy_df, output_file, setup, metadata, V, P, verbose=True)
    return output_file


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run Jere's original MIP-VFI on a RICE payoff table."
    )
    parser.add_argument("payoff_table", help="Path to RICE payoff table Excel file")
    parser.add_argument("--n-restarts", type=int, default=40,
                        help="Number of random V_init draws for multi-start search (default: 40)")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="Max VFI iterations per run (default: 300)")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="VFI convergence tolerance (default: 1e-6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single", action="store_true",
                        help="Single run from PAYOFFS init instead of multi-start search")
    parser.add_argument("--save", action="store_true",
                        help="Save found equilibria as find_equilibrium.py strategy Excel files")
    parser.add_argument("--output", type=str, default="auto",
                        help="Output file path when --save is used (default: auto-generated)")
    args = parser.parse_args()

    config = load_rice_into_jere(args.payoff_table)

    print(f"Players : {jere.PLAYERS}")
    print(f"N_states: {jere.N_STATES}  (B({jere.N_PLAYERS}) = {jere.N_STATES})")
    print(f"Delta   : {config['discounting']}")
    print(f"Rho     : {[round(p, 4) for p in jere.PROPOSER_PROBS]}")
    print()

    if args.single:
        print("Single VFI run from payoff initialisation...")
        V, sigmas, alphas, qs = jere.vfi(
            delta=config["discounting"],
            max_iter=args.max_iter,
            tol=args.tol,
            proposer_probs=jere.PROPOSER_PROBS,
            verbose=True,
        )
        jere.verify_equilibrium(sigmas, alphas, qs, V)
        if args.save:
            eq = {"sigmas": sigmas, "alphas": alphas, "qs": qs, "V": V}
            print("\nSaving equilibrium...")
            save_jere_solution(eq, config, output_file=args.output)
    else:
        print(f"Multi-start search: {args.n_restarts} restarts, max_iter={args.max_iter}, tol={args.tol}")
        equilibria = jere.find_equilibria(
            delta=config["discounting"],
            proposer_probs=jere.PROPOSER_PROBS,
            n_restarts=args.n_restarts,
            tol=args.tol,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose_each=False,
        )
        print(f"\nFound {len(equilibria)} equilibri{'um' if len(equilibria) == 1 else 'a'}.")
        if equilibria:
            jere.print_equilibrium_summary(equilibria, config["discounting"])
            if args.save:
                for k, eq in enumerate(equilibria):
                    out = args.output if len(equilibria) == 1 else "auto"
                    print(f"\nSaving equilibrium {k + 1}/{len(equilibria)}...")
                    save_jere_solution(eq, config, output_file=out)


if __name__ == "__main__":
    main()
