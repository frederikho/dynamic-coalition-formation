# routes — optimization routes toward ~100× faster per-label solving

Per-support cost is dominated by tooling, not math (measured 2026-06-22 on a slow tail label):
`~62 ms sympy poly_build + ~42 ms msolve subprocess (≈12 ms pure spawn) + ~2 ms FLINT`.
Reaching the ~100× target means attacking `poly_build` and the msolve plumbing. Nothing is
ever *skipped* — the principle is "use the cheapest *conclusive exact* method per system"
(e.g. linear systems solved directly in FLINT; Gröbner only for genuinely nonlinear ones).

## Decision (2026-06-22): measurement-first, then a FLINT spike

1. **Measurement harness** (in `../analysis/`): frozen stratified benchmark corpus across the
   cost spectrum, clean methodology (quiesced cores, repeats, percentiles, CPU vs wall),
   per-stage attribution, and a big-N projection from measured `(r, nv)` cost. Every
   optimization is gated on identical `verify_witness` verdicts on the corpus.
2. **FLINT spike** (`flint_spike/`): port `poly_build` to FLINT coefficient-vectors /
   `fmpq_mpoly` and solve linear/0-dim systems in-process; re-measure on the frozen corpus.
   The measured ceiling decides between the routes below.

## Routes evaluated

- **Python + FLINT** (incremental, ~20–50× ceiling): fmpq_mpoly polynomials, exact in-process
  linear/0-dim solving, msolve subprocess only for nonlinear systems. Lower risk; capped by
  the subprocess floor + Python loop overhead.
- **Julia (Nemo + Groebner.jl)** (~100× ceiling): removes sympy AND the msolve subprocess in
  one language, compiled inner loops, multithreaded, exact over ℚ. Larger rewrite + exactness
  re-validation.
- **GPU (RTX 4060)** — *not pursued*: bottleneck is symbolic/Gröbner (irregular, sequential)
  and exactness needs bignum; the only GPU-suitable piece is the 2% FLINT linear algebra.
  Revisit only if the pipeline is ever recast as pure multi-modular numeric linear algebra.
- **C/C++** — maximum ceiling, maximum effort; most of its win is reachable more cheaply via
  the routes above.

## Separate from speed: completeness

The deferred `k≥2` (positive-dimensional) branches are *inconclusive today* regardless of
speed. Solving "everything" requires a dimension-reduced real-QE backend (CAD via
QEPCAD/Redlog, or msolve + witness slicing) — an algorithmic addition, not a tooling speedup.
