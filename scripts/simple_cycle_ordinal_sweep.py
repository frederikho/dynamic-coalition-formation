#!/usr/bin/env python3
"""Sweep status-quo payoffs for reduced simple-cycle tables under ordinal search."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.find import (
    _compute_verification,
    _parse_players_from_payoff_table,
    setup_experiment,
)
from lib.equilibrium.ordinal_ranking import solve_with_ordinal_ranking_n3
from lib.equilibrium.scenarios import fill_players, get_scenario
from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import verify_equilibrium_detailed


class NullLogger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass

    def success(self, _message: str) -> None:
        pass


def _build_config(scenario_name: str, payoff_table: str) -> dict[str, Any]:
    config = get_scenario(scenario_name)
    config["payoff_table"] = payoff_table
    if config.get("players") is None:
        config = fill_players(config, _parse_players_from_payoff_table(Path(payoff_table)))
    return config


def _verification_kind(message: str) -> str:
    if not message:
        return ""
    first = message.splitlines()[0]
    if "All tests passed" in first:
        return "pass"
    if "Proposal strategy error" in message:
        return "proposal"
    if "Approval strategy error" in message:
        return "approval"
    return "other"


def _extract_player(message: str) -> str:
    if not message:
        return ""
    match = re.search(r"with player ([A-Za-z0-9_]+)!", message)
    return match.group(1) if match else ""


def _extract_error_line(message: str) -> str:
    if not message:
        return ""
    for line in message.splitlines():
        line = line.strip()
        if "strategy error with player" in line:
            return line
    for line in message.splitlines():
        line = line.strip()
        if line and not line.startswith("The value functions V are:"):
            return line
    return ""


def _run_once(
    *,
    template_setup: dict[str, Any],
    payoffs: pd.DataFrame,
    random_seed: int,
    max_combinations: int | None,
    ranking_order: str,
    workers: int,
    batch_size: int,
    weak_orders: bool,
) -> dict[str, Any]:
    solver = EquilibriumSolver(
        players=template_setup["players"],
        states=template_setup["state_names"],
        effectivity=template_setup["effectivity"],
        protocol=template_setup["protocol"],
        payoffs=payoffs,
        discounting=template_setup["discounting"],
        unanimity_required=template_setup["unanimity_required"],
        verbose=False,
        random_seed=random_seed,
        initialization_mode="uniform",
        logger=NullLogger(),
    )
    strategy_df, solver_result = solve_with_ordinal_ranking_n3(
        solver,
        max_combinations=max_combinations,
        shuffle=False,
        random_seed=0,
        ranking_order=ranking_order,
        progress_every=0,
        workers=workers,
        batch_size=batch_size,
        weak_orders=weak_orders,
        logger=NullLogger(),
    )
    strategy_df_filled = strategy_df.copy()
    strategy_df_filled.fillna(0.0, inplace=True)
    setup = dict(template_setup)
    setup["payoffs"] = payoffs
    V, P, P_proposals, P_approvals = _compute_verification(strategy_df_filled, setup)
    result = {
        "V": V,
        "P": P,
        "P_proposals": P_proposals,
        "P_approvals": P_approvals,
        "players": setup["players"],
        "state_names": setup["state_names"],
        "effectivity": setup["effectivity"],
        "strategy_df": strategy_df_filled,
    }
    success, message, _detail = verify_equilibrium_detailed(result)
    return {
        "verification_success": bool(success),
        "verification_message": message,
        "stopping_reason": solver_result.get("stopping_reason", ""),
        "tested_combinations": int(solver_result.get("tested_combinations", 0)),
        "total_combinations": int(solver_result.get("total_combinations", 0)),
        "runtime_seconds": float(solver_result.get("runtime_seconds", 0.0)),
    }


def _format_bool(value: bool) -> str:
    return "Y" if value else "N"


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        ("sq", 4),
        ("strict", 6),
        ("weak", 4),
        ("strict_s", 8),
        ("weak_s", 8),
        ("strict_t", 8),
        ("weak_t", 8),
        ("strict_fail", 11),
        ("weak_fail", 9),
    ]
    print(" ".join(label.ljust(width) for label, width in headers))
    print(" ".join("-" * width for _label, width in headers))
    for row in rows:
        values = [
            str(row["status_quo_abs"]).ljust(4),
            _format_bool(row["strict_success"]).ljust(6),
            _format_bool(row["weak_success"]).ljust(4),
            f'{row["strict_tested"]}'.ljust(8),
            f'{row["weak_tested"]}'.ljust(8),
            f'{row["strict_runtime_seconds"]:.2f}'.ljust(8),
            f'{row["weak_runtime_seconds"]:.2f}'.ljust(8),
            row["strict_fail_kind"][:11].ljust(11),
            row["weak_fail_kind"][:9].ljust(9),
        ]
        print(" ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-payoff-table",
        default="simple_cycle_usachnnde-100-reduced.xlsx",
        help="Reduced payoff table used as template.",
    )
    parser.add_argument(
        "--scenario",
        default="power_threshold_RICE_n3",
        help="Scenario name.",
    )
    parser.add_argument("--min-abs", type=int, default=1, help="Minimum absolute value for the status-quo payoff.")
    parser.add_argument("--max-abs", type=int, default=99, help="Maximum absolute value for the status-quo payoff.")
    parser.add_argument("--workers", type=int, default=12, help="Ordinal-ranking worker count.")
    parser.add_argument("--batch-size", type=int, default=20000, help="Ordinal-ranking batch size.")
    parser.add_argument("--ranking-order", default="payoff", choices=["lexicographic", "payoff"])
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="Optional cap per search. Default is full exhaustive search.",
    )
    parser.add_argument(
        "--skip-strict",
        action="store_true",
        help="Skip the strict-order search and only run weak orders.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output CSV path. Default writes to reports/ with a timestamped name.",
    )
    args = parser.parse_args()

    config = _build_config(args.scenario, args.template_payoff_table)
    template_setup = setup_experiment(config)
    base_payoffs = template_setup["payoffs"].copy(deep=True)
    if "( )" not in base_payoffs.index:
        raise ValueError("Template payoff table has no '( )' row.")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for abs_value in range(args.min_abs, args.max_abs + 1):
        payoffs = base_payoffs.copy(deep=True)
        payoffs.loc["( )", template_setup["players"]] = -float(abs_value)

        strict_result = None
        if not args.skip_strict:
            strict_result = _run_once(
                template_setup=template_setup,
                payoffs=payoffs,
                random_seed=abs_value,
                max_combinations=args.max_combinations,
                ranking_order=args.ranking_order,
                workers=args.workers,
                batch_size=args.batch_size,
                weak_orders=False,
            )

        weak_result = _run_once(
            template_setup=template_setup,
            payoffs=payoffs,
            random_seed=abs_value,
            max_combinations=args.max_combinations,
            ranking_order=args.ranking_order,
            workers=args.workers,
            batch_size=args.batch_size,
            weak_orders=True,
        )

        row = {
            "status_quo_abs": abs_value,
            "strict_success": bool(strict_result["verification_success"]) if strict_result else False,
            "strict_stopping_reason": strict_result["stopping_reason"] if strict_result else "",
            "strict_tested": strict_result["tested_combinations"] if strict_result else 0,
            "strict_total": strict_result["total_combinations"] if strict_result else 0,
            "strict_runtime_seconds": strict_result["runtime_seconds"] if strict_result else 0.0,
            "strict_fail_kind": _verification_kind(strict_result["verification_message"]) if strict_result else "",
            "strict_fail_player": _extract_player(strict_result["verification_message"]) if strict_result else "",
            "strict_fail_line": _extract_error_line(strict_result["verification_message"]) if strict_result else "",
            "weak_success": bool(weak_result["verification_success"]),
            "weak_stopping_reason": weak_result["stopping_reason"],
            "weak_tested": weak_result["tested_combinations"],
            "weak_total": weak_result["total_combinations"],
            "weak_runtime_seconds": weak_result["runtime_seconds"],
            "weak_fail_kind": _verification_kind(weak_result["verification_message"]),
            "weak_fail_player": _extract_player(weak_result["verification_message"]),
            "weak_fail_line": _extract_error_line(weak_result["verification_message"]),
        }
        rows.append(row)

    _print_table(rows)

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else (
        reports_dir / f"simple_cycle_ordinal_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started
    strict_successes = sum(1 for row in rows if row["strict_success"])
    weak_successes = sum(1 for row in rows if row["weak_success"])
    print()
    print(f"rows: {len(rows)}")
    print(f"strict_successes: {strict_successes}")
    print(f"weak_successes: {weak_successes}")
    print(f"elapsed_seconds: {elapsed:.2f}")
    print(f"csv: {output_csv}")


if __name__ == "__main__":
    main()
