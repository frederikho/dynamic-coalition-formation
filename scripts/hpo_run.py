#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.hpo.study import StudyConfig, run_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for n=4 scenarios.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials to run")
    parser.add_argument("--max-time", type=float, default=300.0, help="Per-run time budget in seconds")
    parser.add_argument("--results-dir", type=str, default="results/hpo", help="Output directory for logs")
    parser.add_argument("--n-players", type=int, default=3, help="Number of players (scenario filter)")
    parser.add_argument("--scenario-count", type=int, default=8, help="Number of scenarios to evaluate per trial")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers per trial")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///results/hpo/study.db",
        help="Optuna storage URL (e.g. sqlite:///results/hpo/study.db)"
    )
    parser.add_argument("--study-name", type=str, default="hpo_n3", help="Optuna study name")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    cfg = StudyConfig(
        n_trials=args.trials,
        max_time_seconds=args.max_time,
        results_dir=Path(args.results_dir),
        n_players=args.n_players,
        scenario_count=args.scenario_count,
        n_jobs=args.n_jobs,
        storage=args.storage,
        study_name=args.study_name,
        verbose=not args.quiet,
    )

    run_study(cfg)


if __name__ == "__main__":
    main()
