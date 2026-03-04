**HPO Overview**
This package runs Optuna/TPE hyperparameter optimisation for the equilibrium solver.
It is designed to be side-effect free during trials (no checkpoints, no Excel output).

**Key Files**
- `lib/hpo/runner.py`: runs one scenario with a time budget and returns a score.
- `lib/hpo/study.py`: Optuna study loop, logging, and persistence.
- `scripts/hpo_run.py`: CLI entrypoint.

**Objective**
- If verification succeeds: score = runtime.
- If verification fails or time budget is hit: score = `max_time_seconds * penalty_factor` (default 1.5).

**Defaults**
- n=3 scenarios, 8 scenarios per trial (non-RICE).
- 300s time budget.
- SQLite storage at `results/hpo/study.db`.
- Study name: `hpo_n3`.

**Run**
```bash
python scripts/hpo_run.py --trials 50 --max-time 300 --n-jobs 6 --study-name hpo_n3_75s
```

**Dashboard**
```bash
optuna-dashboard sqlite:///results/hpo/study.db
```
Select the study name in the UI dropdown.

**Parallel Trials**
Run multiple processes with the same storage and study name:
```bash
python scripts/hpo_run.py --n-jobs 4 --study-name hpo_n3_75s &
python scripts/hpo_run.py --n-jobs 4 --study-name hpo_n3_75s &
```

**Notes**
- For RICE scenarios, you must supply a payoff table; these are excluded by default.
- Time budgets are enforced inside the solver (`max_time_seconds`).
