# Continuation in delta: warm starts are NOT inert

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


Run 2026-08-18. Driver `scripts/delta_continuation_n3.py`; data in `cont2.json`,
controls in `cold_controls.json` / `cold_controls2.json`. Baseline is the cold
sweep in `reports/delta_homotopy/`.

## What was tested

Walk delta in steps of 0.005 from inside each trio's solved region into its
failure region, seeding each solve with the value function of the previous
step's verified profile (`--jeres-v-init`, added for this). Everything else is
identical to the cold sweep: same solver, same seed 42, same 1 restart, same
`--verify-atol 1e-12`, same `max_iter` 2000.

Every point the warm walk solved was then re-solved **cold at the same delta**,
so each claimed gain is a controlled comparison, not an inference from a
neighbouring grid point.

## Result

**25 verdict flips attributable to warm starting.**

| trio | points warm solves and cold fails | range |
|------|-----------------------------------|-------|
| `nderususa` | 18 | 0.875 – 0.96 |
| `eurrususa` | 7 | 0.89 – 0.92 |

`nderususa` is the striking one. Cold, it solves at delta <= 0.86, fails across
[0.88, 0.96], solves at 0.97 and 0.98, then fails from 0.99. Warm-started, it
solves **continuously from 0.86 to 0.985 — 26 consecutive points** — and stalls
only at 0.99, where the cold sweep also fails. Its cold failure band was almost
entirely an artefact of initialisation.

## This refutes a standing claim, in a specific and bounded way

CLAUDE.md records: *"The multi-start is inert. Do not add restarts or seeds. VFI
here is globally convergent to a unique fixed point... If a game's unique fixed
point is not an equilibrium, no number of restarts can find one."*

That reasoning has two premises which do not carry to this case:

1. **It was measured on runs that CONVERGE.** The 12-restart experiment found a
   bit-identical V in 3-8 iterations. The failing tables *cycle*; for them no
   fixed point is reached at all, so "the unique fixed point" is not established
   and the conclusion drawn from its uniqueness does not apply.
2. **It concerns RANDOM restarts**, drawn from `game.payoffs + N(0, 10*spread)` —
   a diffuse ball. An equilibrium V of the same game at an adjacent delta is a
   strategically coherent point that such draws hit with probability zero.
   "Random restarts are inert" does not imply "informed warm starts are inert",
   and measurably it does not.

**What still stands:** random restarts remain inert, and the advice not to add
restart ladders or seed sweeps is unchanged — 25 flips came from *structure* in
the initialisation, not from more of it. Every walk here used exactly 1 restart.

## What the stalls now mean

With initialisation controlled for, a stall is much better evidence of a genuine
pure-branch termination than a cold failure was. Two-sided brackets:

| trio | walks up to | walks down to | genuine gap |
|------|-------------|---------------|-------------|
| `eurrususa` | 0.92 (stall 0.925) | 0.99 (stall 0.985) | **[0.925, 0.985]** |
| `chnrususa` | 0.82 (stall 0.825) | 0.98 (stall 0.975) | [0.825, 0.975] |
| `chneurrus` | 0.88 (stall 0.885) | 0.999 (stall 0.994) | [0.885, 0.994] |

Trios that stall within 1-3 steps of their cold boundary (`chneurnde`,
`chneurusa`, `chnnderus`, `chnndeusa`, `chnrususa`, `eurndeusa`) are where the
pure branch plausibly does terminate. `eurrususa` is the sharpest case: the
branch is now bracketed from both sides, and the last verified profile on each
side is the concrete input for constructing the mixed equilibrium in between —
two phases whose convex combination is where the mixing should live.

## Revision to `reports/delta_homotopy/FINDINGS.md`

Section 2(c) of that report read the non-contiguous failure sets as a property of
the games. For `nderususa` that was wrong: its islands were initialisation
artefacts. The delta-dependence of the *aggregate* solve rate stands (the low-delta
region is genuinely easy), but **per-trio failure bands from cold starts overstate
the hard region**, and the corrected boundaries are the stalls above.

## Caveats

- Only two trios gained. Eight walked 1-3 steps and stopped, so warm starting is
  not a general fix — it moved boundaries, it did not remove them.
- Step size 0.005 and span 0.15. A finer step might cross a stall that a coarser
  one cannot; untested.
- Warm starts were taken only from the immediately preceding step. Seeding from a
  further-away solved delta, or from a different trio, is untested.
- The unit convention must match between the seed profile and the run. Now
  enforceable: profiles record `normalise_payoffs` (fixed 2026-08-18), though the
  warm-start path does not yet check it.
