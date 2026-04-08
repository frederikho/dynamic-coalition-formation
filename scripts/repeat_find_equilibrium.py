#!/usr/bin/env python3
"""Run repeated equilibrium searches in parallel and summarize outcomes."""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.find import (
    _parse_players_from_payoff_table,
    setup_experiment,
    find_equilibrium,
)
from lib.equilibrium.solver import EquilibriumSolver
from lib.equilibrium.active_set_n3 import _capture_exact_cycle, _build_basin_signature
from lib.equilibrium.scenarios import fill_players, get_scenario


class NullLogger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass

    def success(self, _message: str) -> None:
        pass


def _parse_solver_param(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Missing parameter name in {raw!r}")
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = value
    return key, parsed


def _build_config(scenario_name: str, payoff_table: str) -> dict:
    config = get_scenario(scenario_name)
    config["payoff_table"] = payoff_table
    if config.get("players") is None:
        config = fill_players(config, _parse_players_from_payoff_table(Path(payoff_table)))
    return config


def _run_once(
    scenario_name: str,
    payoff_table: str,
    solver_approach: str,
    solver_params: dict[str, Any],
    run_idx: int,
    seed: int | None,
) -> dict[str, Any]:
    config = _build_config(scenario_name, payoff_table)
    solver_params = dict(solver_params)
    if solver_approach == "ordinal_ranking" and "ordinal_ranking_progress_every" not in solver_params:
        solver_params["ordinal_ranking_progress_every"] = 0
    if solver_approach == "ordinal_ranking" and "ordinal_ranking_workers" not in solver_params:
        solver_params["ordinal_ranking_workers"] = 1
    result = find_equilibrium(
        config=config,
        output_file=None,
        solver_params=solver_params,
        verbose=False,
        load_from_checkpoint=False,
        random_seed=seed,
        logger=NullLogger(),
        save_payoffs=False,
        save_unverified=False,
        diagnostics=False,
        solver_approach=solver_approach,
    )
    return {
        "run": run_idx,
        "seed": result["random_seed"],
        "verification_success": result["verification_success"],
        "stopping_reason": result.get("stopping_reason"),
        "runtime_seconds": result.get("runtime_seconds"),
        "probe_seconds": None,
        "solve_seconds": result.get("runtime_seconds"),
        "worker_wall_seconds": result.get("runtime_seconds"),
        "verification_message": result.get("verification_message"),
        "verification_detail": result.get("verification_detail"),
        "active_set_detail": (result.get("solver_result") or {}).get("active_set"),
        "basin_signature": ((result.get("solver_result") or {}).get("active_set") or {}).get("basin_signature"),
    }


def _probe_once(
    scenario_name: str,
    payoff_table: str,
    run_idx: int,
    seed: int | None,
    solver_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import time

    solver_params = solver_params or {}
    config = _build_config(scenario_name, payoff_table)
    setup = setup_experiment(config)
    solver = EquilibriumSolver(
        players=setup["players"],
        states=setup["state_names"],
        effectivity=setup["effectivity"],
        protocol=setup["protocol"],
        payoffs=setup["payoffs"],
        discounting=setup["discounting"],
        unanimity_required=setup["unanimity_required"],
        verbose=False,
        random_seed=seed,
        initialization_mode=solver_params.get("initialization_mode", "uniform"),
        logger=NullLogger(),
    )
    t0 = time.perf_counter()
    basin_signature = _build_basin_signature(_capture_exact_cycle(solver))
    probe_seconds = time.perf_counter() - t0
    return {
        "run": run_idx,
        "seed": solver.random_seed,
        "runtime_seconds": probe_seconds,
        "probe_seconds": probe_seconds,
        "solve_seconds": None,
        "worker_wall_seconds": probe_seconds,
        "basin_signature": basin_signature,
    }


def _probe_basin_signature(
    scenario_name: str,
    payoff_table: str,
    solver_approach: str,
    seed: int | None,
    solver_params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if solver_approach != "active_set":
        return None
    solver_params = solver_params or {}
    config = _build_config(scenario_name, payoff_table)
    setup = setup_experiment(config)
    solver = EquilibriumSolver(
        players=setup["players"],
        states=setup["state_names"],
        effectivity=setup["effectivity"],
        protocol=setup["protocol"],
        payoffs=setup["payoffs"],
        discounting=setup["discounting"],
        unanimity_required=setup["unanimity_required"],
        verbose=False,
        random_seed=seed,
        initialization_mode=solver_params.get("initialization_mode", "uniform"),
        logger=NullLogger(),
    )
    return _build_basin_signature(_capture_exact_cycle(solver))


def _probe_once_subprocess(
    scenario_name: str,
    payoff_table: str,
    run_idx: int,
    seed: int | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-mode",
        "probe",
        "--worker-scenario",
        scenario_name,
        "--worker-payoff-table",
        payoff_table,
        "--worker-run-idx",
        str(run_idx),
    ]
    if seed is not None:
        cmd.extend(["--worker-seed", str(seed)])
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    result["worker_wall_seconds"] = time.perf_counter() - t0
    return result


def _run_once_subprocess(
    scenario_name: str,
    payoff_table: str,
    solver_approach: str,
    solver_params: dict[str, Any],
    run_idx: int,
    seed: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-mode",
        "solve",
        "--worker-scenario",
        scenario_name,
        "--worker-payoff-table",
        payoff_table,
        "--worker-solver-approach",
        solver_approach,
        "--worker-run-idx",
        str(run_idx),
        "--worker-solver-params-json",
        json.dumps(solver_params),
    ]
    if seed is not None:
        cmd.extend(["--worker-seed", str(seed)])

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        basin_signature = _probe_basin_signature(
            scenario_name=scenario_name,
            payoff_table=payoff_table,
            solver_approach=solver_approach,
            seed=seed,
            solver_params=solver_params,
        )
        return {
            "run": run_idx,
            "seed": seed,
            "verification_success": False,
            "stopping_reason": "timeout",
            "runtime_seconds": elapsed,
            "probe_seconds": None,
            "solve_seconds": elapsed,
            "worker_wall_seconds": elapsed,
            "verification_message": f"Timed out after {timeout_seconds:.0f}s",
            "verification_detail": None,
            "active_set_detail": None,
            "basin_signature": basin_signature,
        }
    result = json.loads(completed.stdout)
    elapsed = time.perf_counter() - t0
    result["worker_wall_seconds"] = elapsed
    if result.get("solve_seconds") is None:
        result["solve_seconds"] = result.get("runtime_seconds")
    if result.get("probe_seconds") is None:
        result["probe_seconds"] = None
    return result


def _progress_char(row: dict[str, Any]) -> str:
    if row.get("stopping_reason") == "skipped_known_basin":
        return "."
    if row.get("verification_success"):
        return "S"
    if row.get("stopping_reason") == "timeout":
        return "T"
    return "F"


def _print_progress_char(
    row: dict[str, Any],
    completed: int,
    total_runs: int,
    progress_column: int,
) -> int:
    if progress_column % 100 == 0:
        if progress_column > 0:
            print()
        prefix_width = len(f"({total_runs}/{total_runs}) ")
        print(f"{f'({completed}/{total_runs})':<{prefix_width}}", end="", flush=True)
    print(_progress_char(row), end="", flush=True)
    return progress_column + 1


def _summarize(results: list[dict[str, Any]], total_wall_seconds: float | None = None) -> None:
    detail_limit = 5
    error_width = 140

    def _fmt_seed(value: Any) -> str:
        return f"{value:>10}" if value is not None else f"{'-':>10}"

    def _fmt_runtime(value: Any) -> str:
        return f"{value:.2f}s" if value is not None else "-"

    def _shorten(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        return text[: width - 3].rstrip() + "..."

    success_count = sum(1 for row in results if row["verification_success"])
    skipped_count = sum(1 for row in results if row.get("stopping_reason") == "skipped_known_basin")
    runtimes = [row["runtime_seconds"] for row in results if row.get("runtime_seconds") is not None]
    probe_times = [row["probe_seconds"] for row in results if row.get("probe_seconds") is not None]
    solve_times = [row["solve_seconds"] for row in results if row.get("solve_seconds") is not None]
    worker_wall_times = [row["worker_wall_seconds"] for row in results if row.get("worker_wall_seconds") is not None]
    success_runtimes = [
        row["runtime_seconds"]
        for row in results
        if row["verification_success"] and row.get("runtime_seconds") is not None
    ]
    stop_counter = Counter(row.get("stopping_reason") for row in results)
    timeout_count = stop_counter.get("timeout", 0)
    violation_counter = Counter()
    basin_counter = Counter()
    success_basin_counter = Counter()

    for row in results:
        basin_signature = row.get("basin_signature")
        if basin_signature:
            basin_label = json.dumps(basin_signature, sort_keys=True)
            basin_counter[basin_label] += 1
            if row["verification_success"]:
                success_basin_counter[basin_label] += 1

        detail = row.get("verification_detail")
        if not detail:
            continue
        if detail.get("type") == "proposal":
            label = f"proposal {detail['proposer']} @ {detail['current_state']}"
        elif detail.get("type") == "approval":
            label = (
                f"approval {detail['approver']} on "
                f"{detail['proposer']}:{detail['current_state']}->{detail['next_state']}"
            )
        else:
            label = str(detail)
        violation_counter[label] += 1

    print("\nSummary")
    print("-" * 80)
    print(f"runs: {len(results)}")
    print(f"verified_success: {success_count}")
    print(f"verification_failed: {len(results) - success_count - skipped_count}")
    print(f"skipped_known_basin: {skipped_count}")
    print(f"timeouts: {timeout_count}")

    print("\nTiming")
    print("-" * 80)
    if total_wall_seconds is not None:
        print(f"wall_time: {total_wall_seconds:.2f}s")
    if worker_wall_times:
        print(f"worker_wall_sum: {sum(worker_wall_times):.2f}s")
    if probe_times:
        print(f"probe_sum: {sum(probe_times):.2f}s")
        print(f"probe_mean: {statistics.mean(probe_times):.2f}s")
        print(f"probe_max: {max(probe_times):.2f}s")
    if solve_times:
        print(f"solve_sum: {sum(solve_times):.2f}s")
        print(f"solve_mean: {statistics.mean(solve_times):.2f}s")
        print(f"solve_max: {max(solve_times):.2f}s")
    if runtimes:
        print(f"runtime_mean: {statistics.mean(runtimes):.2f}s")
        print(f"runtime_median: {statistics.median(runtimes):.2f}s")
        print(f"runtime_min: {min(runtimes):.2f}s")
        print(f"runtime_max: {max(runtimes):.2f}s")
    if success_runtimes:
        print(f"success_runtime_mean: {statistics.mean(success_runtimes):.2f}s")
        print(f"success_runtime_median: {statistics.median(success_runtimes):.2f}s")
        print(f"success_runtime_min: {min(success_runtimes):.2f}s")
        print(f"success_runtime_max: {max(success_runtimes):.2f}s")

    print("\nStopping reasons")
    print("-" * 80)
    for reason, count in sorted(stop_counter.items(), key=lambda item: (-item[1], str(item[0]))):
        print(f"{reason}: {count}")

    print("\nFirst Violations")
    print("-" * 80)
    if not violation_counter:
        print("none")
    else:
        for label, count in violation_counter.most_common(10):
            print(f"{label}: {count}")

    print("\nBasin Signatures")
    print("-" * 80)
    if not basin_counter:
        print("none")
    else:
        for label, count in basin_counter.most_common(10):
            success_count_for_basin = success_basin_counter.get(label, 0)
            print(f"{count} total / {success_count_for_basin} success: {label}")

    print("\nSuccessful runs")
    print("-" * 80)
    successes = [row for row in results if row["verification_success"]]
    if not successes:
        print("none")
    else:
        for row in successes[:detail_limit]:
            print(
                f"run={row['run']:>3}  seed={_fmt_seed(row.get('seed'))}  "
                f"runtime={_fmt_runtime(row.get('runtime_seconds'))}  stop={row['stopping_reason']}"
            )
        if len(successes) > detail_limit:
            print(f"... {len(successes) - detail_limit} more")

    print("\nFailed runs")
    print("-" * 80)
    failures = [
        row for row in results
        if not row["verification_success"] and row.get("stopping_reason") != "skipped_known_basin"
    ]
    for row in failures[:detail_limit]:
        first_error = ""
        message = row.get("verification_message") or ""
        for line in message.splitlines():
            if "strategy error" in line:
                first_error = _shorten(line, error_width)
                break
        print(
            f"run={row['run']:>3}  seed={_fmt_seed(row.get('seed'))}  "
            f"runtime={_fmt_runtime(row.get('runtime_seconds'))}  stop={row['stopping_reason']}"
        )
        if first_error:
            print(f"  {first_error}")
    if len(failures) > detail_limit:
        print(f"... {len(failures) - detail_limit} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one equilibrium setup repeatedly in parallel and summarize the outcomes."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=["solve", "probe"], default="solve", help=argparse.SUPPRESS)
    parser.add_argument("--worker-scenario", help=argparse.SUPPRESS)
    parser.add_argument("--worker-payoff-table", help=argparse.SUPPRESS)
    parser.add_argument("--worker-solver-approach", help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-idx", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-solver-params-json", default="{}", help=argparse.SUPPRESS)
    parser.add_argument("scenario", nargs="?", help="Scenario name, e.g. power_threshold_RICE_n3")
    parser.add_argument("--payoff-table", help="Payoff table filename or path")
    parser.add_argument(
        "--solver-approach",
        default="active_set",
        choices=["annealing", "support_enumeration", "active_set", "ordinal_ranking"],
        help="Solver approach to use for every run",
    )
    parser.add_argument("--runs", type=int, default=100, help="Number of repeated runs")
    parser.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1), help="Parallel workers")
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Solve timeout in seconds. With --skip-known-basins, applies only to first visits of a new basin.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Optional base seed; run i uses base_seed + i. If omitted, each run chooses its own seed.",
    )
    parser.add_argument(
        "--solver-param",
        action="append",
        default=[],
        help="Override solver params as KEY=VALUE, e.g. --solver-param active_set_max_candidates=2048",
    )
    parser.add_argument(
        "--skip-known-basins",
        action="store_true",
        help="For active_set, probe a cheap basin signature first and skip repeated basin signatures.",
    )
    parser.add_argument(
        "--compact-progress",
        action="store_true",
        help="Print one character per completed run: S success, T timeout, F failed, . skipped known basin.",
    )
    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="Stop early after the first verified success.",
    )
    args = parser.parse_args()

    if args.worker:
        if args.worker_mode == "probe":
            result = _probe_once(
                scenario_name=args.worker_scenario,
                payoff_table=args.worker_payoff_table,
                run_idx=args.worker_run_idx,
                seed=args.worker_seed,
            )
        else:
            result = _run_once(
                scenario_name=args.worker_scenario,
                payoff_table=args.worker_payoff_table,
                solver_approach=args.worker_solver_approach,
                solver_params=json.loads(args.worker_solver_params_json),
                run_idx=args.worker_run_idx,
                seed=args.worker_seed,
            )
        print(json.dumps(result))
        return

    if not args.scenario:
        parser.error("the following arguments are required: scenario")
    if not args.payoff_table:
        parser.error("the following arguments are required: --payoff-table")

    solver_params: dict[str, Any] = {}
    for raw in args.solver_param:
        key, value = _parse_solver_param(raw)
        solver_params[key] = value

    print("Repeat Equilibrium Runs")
    print("-" * 80)
    print(f"scenario: {args.scenario}")
    print(f"payoff_table: {args.payoff_table}")
    print(f"solver_approach: {args.solver_approach}")
    print(f"runs: {args.runs}")
    print(f"jobs: {args.jobs}")
    print(f"timeout: {args.timeout}")
    print(f"base_seed: {args.base_seed}")
    print(f"skip_known_basins: {args.skip_known_basins}")
    print(f"stop_on_success: {args.stop_on_success}")
    if args.compact_progress:
        print("progress_legend: S=success T=timeout F=failed .=skipped_known_basin")
    if solver_params:
        print(f"solver_params: {solver_params}")

    results: list[dict[str, Any]] = []
    batch_t0 = time.perf_counter()
    interrupted = False
    try:
        if args.skip_known_basins:
            if args.solver_approach != "active_set":
                parser.error("--skip-known-basins is currently only supported with --solver-approach active_set")
            probe_workers = 1
            solve_workers = max(1, args.jobs - probe_workers)
            probe_executor = ThreadPoolExecutor(max_workers=probe_workers)
            solve_executor = ThreadPoolExecutor(max_workers=solve_workers)
            try:
                pending = set()
                future_meta: dict[Any, tuple] = {}
                seen_basins: dict[str, int] = {}
                next_run_idx = 0
                stop_submitting_probes = False

                def submit_next_probe() -> None:
                    nonlocal next_run_idx
                    if stop_submitting_probes or next_run_idx >= args.runs:
                        return
                    run_idx = next_run_idx
                    next_run_idx += 1
                    seed = None if args.base_seed is None else args.base_seed + run_idx
                    future = probe_executor.submit(
                        _probe_once,
                        args.scenario,
                        args.payoff_table,
                        run_idx,
                        seed,
                        solver_params,
                    )
                    pending.add(future)
                    future_meta[future] = ("probe", run_idx, seed, None, None)

                for _ in range(min(probe_workers, args.runs)):
                    submit_next_probe()

                completed = 0
                progress_column = 0
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        kind, run_idx, seed, basin_label, probe_seconds = future_meta.pop(future)
                        if kind == "probe":
                            probe_result = future.result()
                            basin_signature = probe_result.get("basin_signature")
                            basin_label = json.dumps(basin_signature, sort_keys=True)
                            seed = probe_result.get("seed")
                            probe_seconds = probe_result.get("probe_seconds")
                            if basin_label in seen_basins:
                                row = {
                                    "run": run_idx,
                                    "seed": seed,
                                    "verification_success": False,
                                    "stopping_reason": "skipped_known_basin",
                                    "runtime_seconds": probe_seconds,
                                    "probe_seconds": probe_seconds,
                                    "solve_seconds": None,
                                    "worker_wall_seconds": probe_seconds,
                                    "verification_message": (
                                        f"Skipped known basin first seen at run {seen_basins[basin_label]}."
                                    ),
                                    "verification_detail": None,
                                    "active_set_detail": None,
                                    "basin_signature": basin_signature,
                                }
                                results.append(row)
                                completed += 1
                                if args.compact_progress:
                                    progress_column = _print_progress_char(
                                        row,
                                        completed,
                                        args.runs,
                                        progress_column,
                                    )
                                elif completed % max(1, min(10, args.runs // 10 or 1)) == 0 or completed == args.runs:
                                    print(f"completed: {completed}/{args.runs}")
                            else:
                                seen_basins[basin_label] = run_idx
                                solve_future = solve_executor.submit(
                                    _run_once_subprocess,
                                    args.scenario,
                                    args.payoff_table,
                                    args.solver_approach,
                                    solver_params,
                                    run_idx,
                                    seed,
                                    args.timeout,
                                )
                                pending.add(solve_future)
                                future_meta[solve_future] = ("solve", run_idx, seed, basin_label, probe_seconds)
                            submit_next_probe()
                        else:
                            row = future.result()
                            if row.get("basin_signature") is None and basin_label is not None:
                                row["basin_signature"] = json.loads(basin_label)
                            if probe_seconds is not None:
                                row["probe_seconds"] = probe_seconds
                                if row.get("runtime_seconds") is not None:
                                    row["runtime_seconds"] = row["runtime_seconds"] + probe_seconds
                                if row.get("worker_wall_seconds") is not None:
                                    row["worker_wall_seconds"] = row["worker_wall_seconds"] + probe_seconds
                            results.append(row)
                            completed += 1
                            if args.compact_progress:
                                progress_column = _print_progress_char(
                                    row,
                                    completed,
                                    args.runs,
                                    progress_column,
                                )
                            elif completed % max(1, min(10, args.runs // 10 or 1)) == 0 or completed == args.runs:
                                print(f"completed: {completed}/{args.runs}")
                            if args.stop_on_success and row.get("verification_success"):
                                stop_submitting_probes = True
                                for pending_future in list(pending):
                                    pending_future.cancel()
                                pending.clear()
                                future_meta.clear()
                                break
                if args.compact_progress and progress_column % 100 != 0:
                    print()
            finally:
                probe_executor.shutdown(wait=False, cancel_futures=True)
                solve_executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor = ThreadPoolExecutor(max_workers=args.jobs)
            try:
                futures = []
                for run_idx in range(args.runs):
                    seed = None if args.base_seed is None else args.base_seed + run_idx
                    futures.append(
                        executor.submit(
                            _run_once_subprocess,
                            args.scenario,
                            args.payoff_table,
                            args.solver_approach,
                            solver_params,
                            run_idx,
                            seed,
                            args.timeout,
                        )
                    )

                completed = 0
                progress_column = 0
                for future in as_completed(futures):
                    row = future.result()
                    results.append(row)
                    completed += 1
                    if args.compact_progress:
                        progress_column = _print_progress_char(
                            row,
                            completed,
                            args.runs,
                            progress_column,
                        )
                    elif completed % max(1, min(10, args.runs // 10 or 1)) == 0 or completed == args.runs:
                        print(f"completed: {completed}/{args.runs}")
                if args.compact_progress and progress_column % 100 != 0:
                    print()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Cancelling outstanding work.")

    results.sort(key=lambda row: row["run"])
    total_wall_seconds = time.perf_counter() - batch_t0
    _summarize(results, total_wall_seconds=total_wall_seconds)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
