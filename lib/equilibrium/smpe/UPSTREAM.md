# Vendored: Jere's SMPE solver

**Source:** `coalition-game` repo, `coalition_game/`, commit
`ec55bf2a00fcad6d4f8021e7d39f554e4333c1d0` (2026-09-22).
Method and design record: that repo's `README.md` and `TECHNICAL.md`.

Copied: `smpe.py`, `smpe_hard.py`, `smpe_sweep.py`, `check_equilibrium.py`,
`certify_equilibrium.py`. His `test_smpe.py` lives at
`tests/test_smpe_vendored.py`. Not copied: the benchmark and example-data
scripts and the tests that need his example CSV/JSON files.

## Local changes (grep `[farsighted-coalitions]`)

1. **Package-relative imports** in all five files (`from . import smpe`, ...).
   His CLIs therefore run as modules:
   `python -m lib.equilibrium.smpe.check_equilibrium payoffs.csv profile.json`.
2. **`compute_q(alpha, committees, exclude_j)`** takes a `CoalitionGeometry`
   (a bare voters array still works) and returns q = 0 on `geom.blocked`.
   All 11 call sites pass the geometry. Every derivative of q goes through this
   function, so blocked entries have zero derivative consistently.
3. **`CoalitionGeometry.blocked`** (all False by default) and
   **`CoalitionGeometry.apply_committees(voters, blocked, source)`**, which
   replace his hard-coded committee rule by an external one.
4. **`committees=None` keyword** on `Game.build`, `solve_smpe`, `solve_hard`,
   `solve_point`, `continue_solution`, `_walk`, `sweep`, threaded along the
   existing `rows/rho/rescale` path. With `committees=None` behaviour is
   upstream's exactly.
   Not threaded: `solve_with_mixing_set` (manual tool) and the CLI label listing.

The adapter (`__init__.py`, `solve_with_smpe`) is ours; its docstring explains
the translation of our effectivity to his model and why it is exact.

## Re-syncing with upstream

Copy the five files again, re-apply 1–4, then run
`pytest tests/test_smpe_vendored.py tests/test_smpe_adapter.py` and
`python scripts/smpe_grid.py --out ...` (compare against
`reports/jere_smpe/grid_single.json`).

## Caveat on his exact checker / certifier

They re-derive committees from HIS rule. Their verdicts mean something only
for effectivity rules equivalent to it — in practice `heyen_lehtomaa_2021`.
For any other rule, `lib.utils.verify_equilibrium` is the only valid check.
