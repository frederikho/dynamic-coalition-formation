from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import multiprocessing as mp
import signal

import optuna
from optuna.samplers import TPESampler

from lib.equilibrium.scenarios import get_scenario, list_scenarios
from lib.hpo.runner import run_single, RunResult


def _run_one_scenario(args: Dict[str, Any]) -> RunResult:
    config = get_scenario(args["scenario_name"])
    return run_single(
        config=config,
        solver_params=args["solver_params"],
        max_time_seconds=args["max_time_seconds"],
        penalty_factor=args["penalty_factor"],
        random_seed=args["random_seed"],
        payoff_table=args.get("payoff_table"),
    )


def _init_worker() -> None:
    # Ignore SIGINT in workers; main process handles KeyboardInterrupt.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_children() -> None:
    for p in mp.active_children():
        try:
            p.terminate()
        except Exception:
            pass
    for p in mp.active_children():
        try:
            p.join(timeout=1)
        except Exception:
            pass


@dataclass
class StudyConfig:
    max_time_seconds: float = 300.0
    penalty_factor: float = 1.5
    max_outer_iter: int = 1_000_000
    n_trials: int = 50
    random_seed: int = 123
    results_dir: Path = Path("results/hpo")
    scenarios: Optional[List[str]] = None
    n_players: int = 3
    scenario_count: Optional[int] = 8
    n_jobs: int = 1
    storage: Optional[str] = None
    study_name: Optional[str] = None
    verbose: bool = True
    log_every_trials: int = 1
    # Payoff-table mode: discover .xlsx files in this folder and run each against `scenario`
    payoff_tables_dir: Optional[Path] = None
    scenario: str = "power_threshold_RICE_n3"


def _ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_best(path: Path, trial: optuna.Trial, value: float) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "value": value,
        "params": trial.params,
    }
    path.write_text(json.dumps(payload, indent=2))


def _append_trial_csv(path: Path, trial: optuna.Trial, value: float, extra: Dict[str, Any]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(extra.keys()) + ["value"] + list(trial.params.keys()))
        if is_new:
            writer.writeheader()
        row = {**extra, "value": value, **trial.params}
        writer.writerow(row)


def _sample_params(trial: optuna.Trial) -> Dict[str, Any]:
    tau_p_init = trial.suggest_float("tau_p_init", 1e-4, 5.0, log=True)
    tau_r_init = trial.suggest_float("tau_r_init", 1e-4, 5.0, log=True)
    tau_decay = trial.suggest_float("tau_decay", 0.5, 0.999)
    tau_ratio = trial.suggest_float("tau_min_ratio", 1e-6, 1e-2, log=True)
    tau_min = min(tau_p_init, tau_r_init) * tau_ratio

    damping = trial.suggest_float("damping", 0.0, 1.0)
    max_inner_iter = trial.suggest_int("max_inner_iter", 50, 400)

    inner_tol = trial.suggest_float("inner_tol", 1e-4, 1e-1, log=True)
    outer_mult = trial.suggest_float("outer_tol_mult", 1.0, 100.0, log=True)
    outer_tol = inner_tol * outer_mult

    consecutive_tol = trial.suggest_int("consecutive_tol", 1, 10)
    verify_every_n = trial.suggest_int("verify_every_n", 1, 50)

    return {
        "tau_p_init": tau_p_init,
        "tau_r_init": tau_r_init,
        "tau_decay": tau_decay,
        "tau_min": tau_min,
        "max_inner_iter": max_inner_iter,
        "damping": damping,
        "inner_tol": inner_tol,
        "outer_tol": outer_tol,
        "consecutive_tol": consecutive_tol,
        "verify_every_n": verify_every_n,
    }


def _objective(trial: optuna.Trial, cfg: StudyConfig, scenarios: List[tuple]) -> float:
    # scenarios is a list of (scenario_name, payoff_table_path_or_None) tuples
    params = _sample_params(trial)
    params["max_outer_iter"] = cfg.max_outer_iter
    params["project_to_exact"] = True

    if cfg.verbose and trial.number % max(cfg.log_every_trials, 1) == 0:
        print(f"[HPO] Trial {trial.number} starting ({len(scenarios)} scenarios)...", flush=True)
        print(f"[HPO] Trial {trial.number} params: {params}", flush=True)

    scores = []
    runtimes = []
    successes = 0
    stopping_reasons: Dict[str, int] = {}

    if cfg.n_jobs and cfg.n_jobs > 1:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.n_jobs, mp_context=ctx, initializer=_init_worker) as executor:
            future_map = {}
            for idx, (scenario_name, payoff_table) in enumerate(scenarios, start=1):
                seed = cfg.random_seed + (trial.number * 1000) + idx
                future = executor.submit(
                    _run_one_scenario,
                    {
                        "scenario_name": scenario_name,
                        "solver_params": params,
                        "max_time_seconds": cfg.max_time_seconds,
                        "penalty_factor": cfg.penalty_factor,
                        "random_seed": seed,
                        "payoff_table": payoff_table,
                    },
                )
                future_map[future] = scenario_name
            try:
                for future in as_completed(future_map):
                    run = future.result()
                    scores.append(run.score)
                    runtimes.append(run.runtime_seconds)
                    if run.verification_success:
                        successes += 1
                    stopping_reasons[run.stopping_reason] = stopping_reasons.get(run.stopping_reason, 0) + 1
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                _terminate_children()
                raise
    else:
        for idx, (scenario_name, payoff_table) in enumerate(scenarios, start=1):
            seed = cfg.random_seed + (trial.number * 1000) + idx
            run = _run_one_scenario(
                {
                    "scenario_name": scenario_name,
                    "solver_params": params,
                    "max_time_seconds": cfg.max_time_seconds,
                    "penalty_factor": cfg.penalty_factor,
                    "random_seed": seed,
                    "payoff_table": payoff_table,
                }
            )
            scores.append(run.score)
            runtimes.append(run.runtime_seconds)
            if run.verification_success:
                successes += 1
            stopping_reasons[run.stopping_reason] = stopping_reasons.get(run.stopping_reason, 0) + 1

    avg_score = sum(scores) / len(scores)
    avg_runtime = sum(runtimes) / len(runtimes)
    success_rate = successes / len(scores)

    extra = {
        "trial": trial.number,
        "avg_runtime": avg_runtime,
        "success_rate": success_rate,
        "stopping_reasons": json.dumps(stopping_reasons),
    }

    _append_trial_csv(cfg.results_dir / "trials.csv", trial, avg_score, extra)

    if cfg.verbose and trial.number % max(cfg.log_every_trials, 1) == 0:
        print(
            f"[HPO] Trial {trial.number} done: avg_score={avg_score:.2f}s "
            f"avg_runtime={avg_runtime:.2f}s success_rate={success_rate:.2%}",
            flush=True
        )

    return avg_score


def run_study(cfg: StudyConfig) -> None:
    _ensure_results_dir(cfg.results_dir)

    # Build list of (scenario_name, payoff_table_or_None) tuples
    if cfg.payoff_tables_dir is not None:
        xlsx_files = sorted(Path(cfg.payoff_tables_dir).glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"No .xlsx files found in {cfg.payoff_tables_dir}")
        base_pairs = [(cfg.scenario, p) for p in xlsx_files]
        scenarios = base_pairs[:cfg.scenario_count] if cfg.scenario_count else base_pairs
    elif cfg.scenarios is None:
        base_scenarios = [
            name for name in list_scenarios(filter_players=cfg.n_players)
            if "RICE" not in name
        ]
        if cfg.scenario_count is None or cfg.scenario_count <= len(base_scenarios):
            names = base_scenarios[:cfg.scenario_count] if cfg.scenario_count else base_scenarios
        else:
            names = list(itertools.islice(itertools.cycle(base_scenarios), cfg.scenario_count))
        scenarios = [(name, None) for name in names]
    else:
        names = cfg.scenarios[:cfg.scenario_count] if cfg.scenario_count else cfg.scenarios
        scenarios = [(name, None) for name in names]

    if cfg.verbose:
        print(
            f"[HPO] Running {cfg.n_trials} trials on {len(scenarios)} scenarios "
            f"(n={cfg.n_players}, jobs={cfg.n_jobs}).",
            flush=True
        )
        print(f"[HPO] Results dir: {cfg.results_dir}", flush=True)
        if cfg.study_name:
            print(f"[HPO] Study name: {cfg.study_name}", flush=True)
        if cfg.storage:
            print(f"[HPO] Storage: {cfg.storage}", flush=True)
        print("[HPO] Dashboard: optuna-dashboard sqlite:///results/hpo/study.db", flush=True)
        print(
            "[HPO] Parallel trials: run multiple processes with the same --study-name.",
            flush=True
        )

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Use a per-process seed so parallel workers explore different regions.
    sampler_seed = (cfg.random_seed + os.getpid()) % (2**31)
    sampler = TPESampler(seed=sampler_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=cfg.storage,
        study_name=cfg.study_name,
        load_if_exists=True if cfg.storage else False,
    )

    best_value = float("inf")

    def _callback(study: optuna.Study, trial: optuna.Trial) -> None:
        nonlocal best_value
        if trial.value is not None and trial.value < best_value:
            best_value = trial.value
            _write_best(cfg.results_dir / "best.json", trial, trial.value)

    try:
        study.optimize(lambda t: _objective(t, cfg, scenarios), n_trials=cfg.n_trials, callbacks=[_callback])
    except KeyboardInterrupt:
        if cfg.verbose:
            print("\n[HPO] Interrupted by user (KeyboardInterrupt).", flush=True)
        try:
            best_trial = study.best_trial
        except ValueError:
            best_trial = None
        if best_trial is not None:
            _write_best(cfg.results_dir / "best.json", best_trial, study.best_value)
        _terminate_children()
        return
