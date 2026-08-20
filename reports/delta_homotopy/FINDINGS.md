# Delta homotopy on the n=3 kalkuhl batch — does VFI keep finding equilibria?

> **SUPERSEDED IN PART (2026-08-20).** The readings below of *why* cells fail were
> written before the enumeration work. Two corrections, both material:
>
> 1. A cell marked "no pure equilibrium" rests on an `ordinal_ranking` negative, and
>    OR builds profiles from STRICT value orderings. A pure equilibrium resting on an
>    exact tie in V is outside that image, so a null is *conditional*: it means "no
>    pure equilibrium among strict-ranking profiles". Sound where the game has no
>    exact V tie at a would-be equilibrium; the payoff audit found no exact payoff
>    ties in any of the nine trios, so the 60 cells are believed sound but not proved.
> 2. Sections reading the failure bands as evidence of genuine near-indifference were
>    inference, not measurement. Six cells later turned out to hold pure equilibria
>    VFI simply missed (found by enumeration), so a failure band mixes two causes.
>
> See `reports/mixed_solver/FINDINGS.md` for the current state.


Run 2026-08-18. Driver `scripts/delta_homotopy_n3.py`; raw data in
`sweep1.json` (11-point grid), `refine.json` (8-point refinement),
`seed7.json` / `seed1234.json` (replication).

**Setting.** All ten `kalkuhl_<trio>_2035-2100` tables, scenario
`power_threshold_RICE_n3` (equal power 1/3, `min_power` 0.501, unanimity,
effectivity `heyen_lehtomaa_2021`), payoff normalisation on, solver `jeres_vfi`
with 1 restart (restarts are structurally inert here), `max_iter` 2000, 600 s
cap, `--verify-atol 1e-12`, `--jeres-tol 1e-14`. 19 delta points per trio.

`--discounting` was added to `lib/equilibrium/find.py` for this; it overrides the
scenario's delta and validates that the value lies in (0, 1).

**A verdict of "failed" is a statement about the solver at that setting, never
about the game.** An equilibrium exists at every one of these 190 points
(HL2021 Supplementary, footnote 11 — finite state space).

---

## 1. The grid

```
trio            0.5    0.7    0.8   0.82   0.84   0.86   0.88    0.9   0.92   0.94   0.95   0.96   0.97   0.98   0.99  0.995  0.997  0.999
chneurnde         o      o      o      o      o      o      o      .      .      .      .      .      .      .      .      .      .      .
chneurrus         o      o      o      o      o      o      o      .      .      .      .      .      .      .      .      .      .      o
chneurusa         o      o      o      o      o      o      o      o      .      .      .      .      .      .      .      .      .      .
chnnderus         o      o      o      o      o      o      o      o      o      .      .      .      .      .      .      .      .      .
chnndeusa         o      o      o      o      o      o      .      .      .      .      .      .      .      .      .      .      .      .
chnrususa         o      o      o      o      .      .      .      o      o      .      .      .      .      o      o      o      o      o
eurnderus         o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o      o
eurndeusa         o      o      o      .      .      .      o      o      o      o      o      o      o      o      o      .      .      .
eurrususa         o      o      o      o      o      o      o      .      .      .      .      .      .      .      o      o      o      o
nderususa         o      o      o      o      o      o      .      .      .      .      .      .      o      o      .      .      .      .

solved/10        10     10     10      9      8      8      7      5      4      2      2      2      3      4      4      3      3      4
```

## 2. Four findings

**(a) Solvability degrades smoothly in delta, and the degradation is steep.**
10/10 for every delta ≤ 0.80; then a monotone slide to 2/10 across
delta ∈ [0.94, 0.96]; then a partial recovery to 3–4/10 above 0.97. The paper's
delta = 0.99 gives 4/10, matching the four saved profiles exactly
(`chnrususa`, `eurnderus`, `eurndeusa`, `eurrususa`) — so the sweep reproduces
the known baseline. It sits inside the hard region, not on a cliff edge.

**(b) The failures are deterministic.** Seeds 42, 7 and 1234 give bit-identical
verdicts on all four irregular trios across delta ∈ {0.95, 0.97, 0.98, 0.99,
0.999}. And retrying every failure at the looser `atol` 1e-10 flipped **zero** of
121 verdicts. Neither search randomness nor verification precision explains any
of these failures — consistent with VFI being globally convergent to a unique
fixed point, which makes the verdict a deterministic function of delta.

**(c) For 4 of 10 trios the failure set is not an interval.** `chnrususa` solves
on [0.5, 0.82], fails on [0.84, 0.86], solves on [0.88, 0.92], fails on
[0.94, 0.97], and solves on [0.98, 0.999] — four crossings. `eurndeusa` fails
only on [0.82, 0.86] and again above 0.995. `chneurrus` fails everywhere from
0.9 except a single solved point at 0.999. `nderususa` has an island of success
at [0.97, 0.98]. These bands survive the 0.02-spaced refinement, so they are not
grid artefacts. The remaining 6 trios have one clean crossing each.

**(d) The failure bands bracket a regime change in the prediction itself.** The
absorbing sets, over solved points only:

| trio | low-delta absorbing set | high-delta absorbing set |
|------|------------------------|--------------------------|
| chneurusa | [0.5–0.8] (CHNEUR)+(CHNUSA) | [0.82–0.9] +(EURUSA) |
| chnrususa | [0.5–0.9] (CHNUSA)+(RUSUSA)+(CHNRUSUSA) | [0.98–0.999] (CHNUSA) |
| eurnderus | [0.5–0.92] (EURNDE)+(EURRUS)+(EURNDERUS) | [0.94–0.999] (EURRUS) |
| eurndeusa | [0.5–0.8] (EURNDE)+(EURUSA)+(NDEUSA) | [0.88–0.99] (EURUSA) |
| eurrususa | [0.5–0.86] (EURRUS)+(EURUSA)+(EURRUSUSA) | [0.99–0.999] (EURUSA) |
| nderususa | [0.5–0.86] (NDEUSA)+(RUSUSA) | [0.97–0.98] +(NDERUSUSA) |

In `chnrususa`, `eurndeusa` and `eurrususa` the failure band sits **exactly
between** the two regimes. That is the signature CLAUDE.md predicts for genuine
near-indifference: at the delta where a move's ranking flips, the equilibrium
requires real mixing, and VFI's pure iteration cycles instead of converging.

**The counterexample that keeps this honest:** `eurnderus` makes the same
three-state → one-state transition (between 0.92 and 0.94) and solves at all 19
points. A regime change is therefore *not sufficient* for failure.

## 3. Answer to the question

**No — VFI's success is not preserved under smooth variation of delta.** It is
near-perfect in the myopic region (delta ≤ 0.8, 30/30 solves), and unreliable in
the patient region the paper actually uses (delta ≥ 0.9: 44/140 solves, 31%).
The transition is smooth in aggregate but per-trio it is a sequence of sharp,
reproducible crossings, several of which are non-monotone.

The practical consequence: **delta is a first-order determinant of solvability,
comparable to payoff normalisation in effect size, and nothing in the current
reporting records that.** A "solved / not solved" verdict for a table is only
meaningful paired with the delta it was obtained at.

## 4. An economic result that falls out of this, independent of the solver

In all six trios where both regimes are observed, **the grand coalition is
absorbing at low delta and is not absorbing at high delta** (`chnrususa`,
`eurnderus`, `eurrususa` lose it outright; `chneurusa`, `chnndeusa`,
`eurndeusa` never have it). Farsightedness dissolves the grand coalition — the
mechanism section 2 of `write-up/benchmark_appendix_draft.md` asserts, now
visible as a function of delta rather than argued from V = (I − δP)⁻¹(1 − δ)u.
`nderususa` runs the other way (gains `(NDERUSUSA)` at 0.97–0.98) and is worth a
closer look.

## 5. Caveats

- Only `jeres_vfi` was swept. `merit_descent` solved `eurrususa` at delta = 0.99
  when `jeres_vfi` previously could not; under this run's settings (`max_iter`
  2000, `atol` 1e-12) `jeres_vfi` solves it too, so the settings matter as much
  as the solver choice.
- 1 restart, not 40. Justified by the inertness result in CLAUDE.md and by the
  seed replication above, but it is an assumption, not a re-measurement.
- The 0.02 grid cannot resolve bands narrower than that; `chnrususa`'s
  [0.84, 0.86] failure could be finer-structured than it appears.
- Absorbing sets are only observable at solved points, so every row of the
  table in 2(d) has a gap where the transition actually happens.

## 6. Was the tolerance a constant-strength test across the sweep?

Payoff normalisation was ON for all 190 solves (`--no-normalise-payoffs` is
`store_false` with `default=True`; the driver never passes it). That pins each
player's payoff range to 1, but **not** the V spread, which is what `atol`
actually has to resolve. Measured over the solved profiles (smallest per-player
V spread across states, the binding one):

| delta | min V spread | median | atol 1e-12 as a fraction of the min |
|-------|--------------|--------|--------------------------------------|
| 0.5   | 4.48e-1 | 4.98e-1 | 2.2e-12 |
| 0.8   | 1.93e-1 | 2.15e-1 | 5.2e-12 |
| 0.9   | 1.02e-1 | 1.15e-1 | 9.8e-12 |
| 0.95  | 5.65e-2 | 5.71e-2 | 1.8e-11 |
| 0.99  | 1.26e-2 | 1.45e-2 | 7.9e-11 |
| 0.995 | 6.38e-3 | 8.26e-3 | 1.6e-10 |
| 0.999 | 1.29e-3 | 1.71e-3 | 7.8e-10 |

The V spread collapses ~350x over the sweep, roughly proportional to (1 - delta)
at the top end as states converge on a common limiting distribution. Two
consequences:

- **The test never goes vacuous.** Even at delta = 0.999 the tolerance is 7.8e-10
  of the V spread — nine orders below the signal, inside the safe window.
- **The residual drift cannot manufacture the result.** A fixed `atol` is a
  *weaker* test as delta rises, and a wider tie band is also what stopped VFI
  cycling in the overnight ladder. The hard region was therefore tested slightly
  more leniently and still solved far less often, so the confound works against
  the finding.

One regime is genuinely tight: at delta >= 0.995 the forward error on the V solve
is ~4e-13 (cond(I - delta*P) ~ 2000 x machine epsilon), only ~2.5x below the
1e-12 tolerance, so a valid profile could in principle be rejected on rounding
alone. The 1e-10 retry pass sits 250x above that error floor and flipped zero of
121 failures, which rules it out empirically.

## 7. Would a different normalisation help? No — tested.

Two distinct rescalings, only one of which was ever a candidate:

- **Payoff-side (what `normalise_payoffs` does).** Mathematically incapable of
  changing a verdict: a per-player positive affine map is an exact symmetry of
  the equilibrium concept. Choosing a different map changes the numbers, not
  which profiles verify.
- **V-side / delta-aware.** NOT a symmetry, so it could in principle matter --
  and section 6 shows the V spread moves 350x over the sweep while `atol` and
  `jeres_tol` were held fixed, making the convergence test relatively ~350x
  stricter at delta = 0.999 than at 0.5.

Tested directly (`relative_tol.json`): all 74 failures at delta >= 0.9 rerun with
`atol = 1e-4 x V spread` and `jeres_tol = atol/100`, using the measured scaling
V spread ~ 1.28 x (1 - delta). At delta = 0.99 that is `atol` 1.3e-6, a million
times looser than the sweep's 1e-12 and correctly delta-scaled.

**Flipped: 0 of 74.**

So the delta-dependence in the sweep's tolerances was real but inert. The only
tolerance that ever recovers these cases is the overnight ladder's 1e-1 (10% of
each player's payoff range) -- another 1000x looser again, and an artefact rather
than a solution.

**Consequence.** The remaining failures are structural, not a units or precision
problem: they require mixed strategies that VFI's pure iteration plus 1-D
bisection cannot represent. No rescaling of payoffs, values or tolerances reaches
them. The delta sweep does, however, give a cheap way to LOCATE the mixing --
bracket a failing table between its nearest solved delta on each side and the
absorbing-set change identifies which transition's ranking is flipping.
`eurrususa` is the sharpest case: pure up to 0.86 with three absorbing states,
pure again from 0.99 with one, nothing in between.
