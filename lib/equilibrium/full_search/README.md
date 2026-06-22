# full_search — exact full-proposal-mixing certified MPE search

Certified-complete search for stationary Markov Perfect Equilibria of the n3
`power_threshold_RICE` / `adjacent_step` game (δ=0.99, unanimity, uniform protocol),
allowing **full mixing** over both acceptance decisions and proposal supports. The goal is
a *waterproof* verdict per label: exact rational arithmetic, no tolerances, no
parameter dependence. (Moved here from `scripts/` on 2026-06-22 to declutter.)

## Layout

```
full_search/
  full_mixing_sweep.py        core solver + CLI (subcommands: trial / run / find)
  certified_label_solver.py   per-label exact solver (used only by full_mixing_sweep)
  analysis/                   measurement & forecasting (the "rigorous measurement" route)
    forecast_fullmix.py         cost-model ETA fit from the live log
    probe_label_timing.py       times find_in_label across the cost spectrum
    dissect_slow_label.py       finds one slow tail label and breaks down where time goes
  routes/                     optimization routes toward the ~100x target (see routes/README.md)
```

## How it works (one label = a weak-order triple)

Enumerate `r` candidate proposal supports. Per support, build the support-fixed system
(acceptance-mixing + proposal-weight variables); interpolate the genuinely-multilinear
`num_s[k]=V_s[k]·det` and `det` exactly via FLINT over the `2^nv` cube vertices; build the
variety equations symbolically; solve with **msolve** (Gröbner + real-root isolation) and
exact-verify any 0-dim rational candidate with `verify_witness` (orders/ties/optimality
checked with `==`). Outcomes: feasible (verified equilibrium) / infeasible / **deferred**
(positive-dimensional `k≥2` variety — needs a real-QE backend, not yet built).

`find` = cheapest-first find-one: scan labels by predicted cost `r·c(m)` ascending, solve
only easy branches (`nv≤max_nv`, 0-dim), stop at the first exactly-verified witness. A HIT
is conclusive; a NULL is INCONCLUSIVE (deferred branches skipped — not a non-existence proof).

## Running

```bash
source .venv/bin/activate
python -m lib.equilibrium.full_search.full_mixing_sweep find \
    --payoff burke_usaruschn_2035-2060 --workers 14 --fraction 0.50 --max-nv 8
```

Phase A scans all 541³ labels for predicted cost, sorts cheapest-first, caches the order.
Phase B solves cheapest-first, checkpointed every 2000 labels (resumes after crash/power-loss).
Progress is throttled to one line / 60 s with a trailing-window (last ~120 s) rate + ETA.

**Data lives at the repo root** (not in this package):
- `strategy_tables/fullmix_<payoff>_order.npy`         — Phase A cheapest-first order (cache)
- `strategy_tables/fullmix_<payoff>_find_progress.txt` — Phase B checkpoint (`k defs`)
- `strategy_tables/fullmix_<payoff>_FOUND.txt`         — written on a verified hit
- `logs/fullmix_<payoff>_find.log` (+ `_console.log`)  — progress / console

## Status (2026-06-22)

Measured forecast for the cheapest-50% sweep on the laptop (14 workers): **~64 days**, with
the expensive tail (frac > 0.8) being both ~85% of the runtime AND mostly deferred/inconclusive.
A single slow tail label (~25 s) breaks down as **~59% sympy `poly_build` + ~39% msolve
subprocess spawn, only ~2% actual FLINT math** — i.e. the cost is tooling overhead, not the
exact arithmetic. Target: **~100× speedup**. Decision: build the measurement harness +
projection first, then a FLINT `poly_build` spike to measure the real ceiling before choosing
Python+FLINT vs a Julia (Nemo + Groebner.jl) rewrite. See `routes/README.md`.

## Cross-package deps (still in `scripts/`, shared with older approaches)

- `scripts.residual_metric_probe.build_setup`
- `scripts._reduced_helpers._generate_weak_orders`
