# Resume-point: optimizing `vertex_eval` (the current bottleneck)

After the Julia in-process Gröbner win, the fixed rand4000 set solves in **46.4 s** (from 135 s
CLI), parity-exact. The dominant stage is now **`vertex_eval` ≈ 27.6 s (60%)**.

## Measured anatomy of `vertex_eval`
Per cube vertex we build the transition matrix `P` (fmpq), form `A = I − δP`, solve `A·V = b`,
and store `num[s][k] = V[s,k]·det` for all S·n entries — repeated over `2^nv` vertices per support.
Measured split (1000 labels, 167 930 vertex solves):
- exact `A.solve(b)`  ~37%  (~15 µs each on the real big rationals; **irreducible-ish**)
- `A.det()`           ~7%   (**negligible** — no det-fusion win; and python-flint `fmpz_mat.solve`
  returns `fmpq` anyway, so no free determinant)
- **Python `fmpq` matrix construction + `V·det` storage  ~56%**  ← the real hog

Theoretical floor if construction were free ≈ **27–30 s** total (solve + poly_build + msolve).

## Done
- **Hoisted constant `b = (1−δ)·U`** out of the per-vertex path into `__init__` (it never changes).
  48.6 → 46.4 s, parity identical. Safe constant-hoisting is now exhausted.

## Options for further gains (each a real rewrite; parity must be re-validated against the
## harness fingerprint `476417fd05ace691` on rand4000)

1. **Gray-code incremental evaluation** (~2–4× on vertex_eval, biggest bang for the buck inside
   Python). Order the `2^nv` vertices in Gray-code so consecutive vertices flip ONE variable.
   Flipping one variable changes only ONE row of `P` (the proposing state's row) → a rank-1 update
   to `A`. Maintain the solve via exact Sherman–Morrison (`O(S²)` update instead of an `O(S³)`
   solve) and update `det` by the rank-1 factor. Avoids both rebuilding the matrix AND re-solving.
   Risk: intricate exact-`fmpq` linear algebra (the Δ-row vector, the det sign/scale, numerical-free
   correctness). Validate hard.

2. **Move vertex-eval + multilinear interpolation into Julia/Nemo** (highest ceiling; architecturally
   natural now that Julia is embedded). Replace the interpreted Python `fmpq` construction loops with
   compiled Nemo `QQ` matrix arithmetic + the Möbius transform in Julia; return the `num`/`det`
   coefficient arrays (or the built eqs) to Python — or do the whole `solve_profile` in Julia.
   Risk: large rewrite of the hot core + parity re-validation + marshaling design.

3. **Precompute the bit-independent part of `P`/`A` per support** (moderate win, moderate risk).
   Forced-accept/reject transitions, single-support diagonals, and protocol weights don't depend on
   the vertex bits — build that base matrix once per support, then per vertex overlay only the
   variable-dependent entries (tied-acceptance products, weight splits). Cuts construction without
   touching the solve. Less powerful than (1)+(2) but lower risk; could combine with Gray-code.

## Notes
- The `2^nv` vertex COUNT is fundamental (multilinear interpolation needs all cube vertices); the
  win is making each vertex cheaper (1,3) or sharing work across vertices (1).
- `det` is needed only to form `num = V·det` (DET itself is dead in the decision); keep it.
- Start by re-confirming the split on the current code, then prototype (1) on the high-`nv` supports
  (where `2^nv` is large and the payoff concentrates).
