#!/usr/bin/env python3
"""Batch hardness report for RICE payoff tables."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    script_dir = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == script_dir:
        sys.path[0] = repo_root
    elif repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from lib.equilibrium.find import (
    _parse_players_from_payoff_table,
    find_equilibrium,
    is_valid_rice_payoff_table_filename,
    iter_valid_rice_payoff_tables,
)
from lib.equilibrium.scenarios import fill_players, get_scenario


def _parse_filename_metadata(path: Path) -> Dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    family = parts[0] if parts else ""

    year_token = ""
    payoff_horizon = ""
    for token in reversed(parts):
        if token.isdigit() or (len(token) == 9 and token[:4].isdigit() and token[4] == "-" and token[5:].isdigit()):
            year_token = token
            break
    if "_payoff_" in stem:
        payoff_horizon = stem.split("_payoff_", 1)[1]
    elif "_summed_until_" in stem:
        payoff_horizon = f"summed_until_{stem.split('_summed_until_', 1)[1]}"

    return {
        "family": family,
        "year_window": year_token,
        "payoff_horizon": payoff_horizon,
        "file_stem": stem,
    }


def _score_row(row: Dict[str, Any]) -> tuple:
    failure_first = 0 if row.get("verification_success") else 1
    error_first = 1 if row.get("run_error") else 0
    runtime = float(row.get("runtime_seconds") or 0.0)
    min_margin = row.get("min_nonzero_approval_margin")
    margin_score = float(min_margin) if min_margin is not None else float("inf")
    ambiguous = int(row.get("proposal_ambiguous_rows") or 0)
    return (error_first, failure_first, -runtime, margin_score, -ambiguous, row["payoff_table"])


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return f"{float(value):.6g}"


def _print_terminal_summary(results: List[Dict[str, Any]], invalid_files: List[Path]) -> None:
    print("\nRICE Hardness Report")
    print("=" * 80)
    for row in sorted(results, key=_score_row):
        status = "OK" if row.get("verification_success") else "FAIL"
        if row.get("run_error"):
            status = "ERR"
        print(
            f"{status:4}  "
            f"{row['payoff_table']:<55}  "
            f"runtime={_format_float(row.get('runtime_seconds')):<8}  "
            f"stop={str(row.get('stopping_reason', '-')):<18}  "
            f"min_margin={_format_float(row.get('min_nonzero_approval_margin')):<10}  "
            f"ambig={row.get('proposal_ambiguous_rows', '-')}"
        )

    if invalid_files:
        print("\nRejected payoff tables")
        print("-" * 80)
        for path in invalid_files:
            print(path.name)

    successes = sum(1 for row in results if row.get("verification_success"))
    errors = sum(1 for row in results if row.get("run_error"))
    failures = len(results) - successes - errors
    print("\nSummary")
    print("-" * 80)
    print(f"verified_success: {successes}")
    print(f"verification_failed: {failures}")
    print(f"run_errors: {errors}")
    print(f"invalid_payoff_tables: {len(invalid_files)}")


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "payoff_table",
        "payoff_path",
        "valid_payoff_table",
        "family",
        "year_window",
        "payoff_horizon",
        "players",
        "n_players",
        "runtime_seconds",
        "verification_success",
        "verification_message_first_line",
        "stopping_reason",
        "outer_iterations",
        "converged",
        "final_tau_p",
        "final_tau_r",
        "min_nonzero_approval_margin",
        "num_small_approval_margins",
        "proposal_ambiguous_rows",
        "proposal_min_best_gap",
        "output_written",
        "output_file",
        "random_seed",
        "run_error",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["players"] = ",".join(row.get("players", [])) if isinstance(row.get("players"), list) else row.get("players")
            writer.writerow(out)


def _run_single_case(
    scenario_name: str,
    payoff_table: Path,
    approval_margin_threshold: float,
    verbose: bool,
    solver_approach: str,
) -> Dict[str, Any]:
    config = get_scenario(scenario_name)
    players = _parse_players_from_payoff_table(payoff_table)
    if config.get("players") is None:
        config = fill_players(config, players)
    config["payoff_table"] = str(payoff_table)

    result = find_equilibrium(
        config,
        output_file="auto",
        solver_params=None,
        verbose=verbose,
        description=None,
        load_from_checkpoint=False,
        random_seed=None,
        logger=None,
        save_payoffs=False,
        save_unverified=False,
        diagnostics=True,
        approval_margin_threshold=approval_margin_threshold,
        solver_approach=solver_approach,
    )
    diagnostics = result["diagnostics"].copy()
    diagnostics.update(_parse_filename_metadata(payoff_table))
    diagnostics["payoff_table"] = payoff_table.name
    diagnostics["payoff_path"] = str(payoff_table.resolve())
    diagnostics["valid_payoff_table"] = True
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run power_threshold_RICE_n3 across valid payoff tables and build a hardness report."
    )
    parser.add_argument(
        "--payoff-dir",
        type=str,
        default="payoff_tables",
        help="Directory containing payoff-table xlsx files (default: payoff_tables)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="power_threshold_RICE_n3",
        help="Scenario to run for all valid payoff tables (default: power_threshold_RICE_n3)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="auto",
        help="CSV output path for the report, or 'auto' for reports/rice_hardness_<timestamp>.csv",
    )
    parser.add_argument(
        "--approval-margin-threshold",
        type=float,
        default=1e-3,
        help="Threshold for counting small approval margins in diagnostics (default: 1e-3)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose solver output while still printing the final report",
    )
    parser.add_argument(
        "--solver-approach",
        type=str,
        choices=["annealing", "support_enumeration", "active_set"],
        default="annealing",
        help="Solver approach to use for every attempted payoff table (default: annealing)",
    )

    args = parser.parse_args()

    payoff_dir = Path(args.payoff_dir)
    all_xlsx = sorted(payoff_dir.glob("*.xlsx"))
    valid_files = iter_valid_rice_payoff_tables(payoff_dir)
    valid_set = {path.name for path in valid_files}
    invalid_files = [path for path in all_xlsx if path.name not in valid_set]

    results: List[Dict[str, Any]] = []
    for payoff_table in valid_files:
        try:
            diagnostics = _run_single_case(
                scenario_name=args.scenario,
                payoff_table=payoff_table,
                approval_margin_threshold=args.approval_margin_threshold,
                verbose=not args.quiet,
                solver_approach=args.solver_approach,
            )
        except Exception as exc:
            diagnostics = _parse_filename_metadata(payoff_table)
            diagnostics.update({
                "payoff_table": payoff_table.name,
                "payoff_path": str(payoff_table.resolve()),
                "valid_payoff_table": True,
                "verification_success": False,
                "run_error": str(exc),
            })
        results.append(diagnostics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_csv == "auto":
        output_csv = Path("reports") / f"rice_hardness_{args.solver_approach}_{timestamp}.csv"
    else:
        output_csv = Path(args.output_csv)

    invalid_rows = []
    for path in invalid_files:
        row = _parse_filename_metadata(path)
        row.update({
            "payoff_table": path.name,
            "payoff_path": str(path.resolve()),
            "valid_payoff_table": False,
        })
        invalid_rows.append(row)

    _write_csv(results + invalid_rows, output_csv)
    _print_terminal_summary(results, invalid_files)
    print(f"\nCSV report written to: {output_csv}")


if __name__ == "__main__":
    main()
