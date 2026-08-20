# Computing mixed equilibria: what works, what does not, and why

Written 2026-08-20. Supersedes the diagnostic sections of
`reports/delta_homotopy/FINDINGS.md` and `reports/delta_continuation/FINDINGS.md`,
which were written before any of this was measured.

---

## 1. The problem, stated precisely

An equilibrium is a **support pattern** plus **M mixing values**.

The support pattern is 69 discrete facts for n=3: for each of the 15
(proposer, state) slots, which targets carry sigma > 0; and for each of the 54
acceptance probabilities, whether it is 0, 1, or interior. For a *pure* slot the
support IS the value -- a singleton proposal support forces sigma = 1, and a
trichotomy label of "0" forces alpha = 0. Only the interior slots leave a number
to determine, and M counts those.

Two facts split the problem cleanly, and both are measured below:

- **Given the support pattern, solving is free.** 40/40 in under 0.2 s.
- **Finding the support pattern is the whole difficulty.**

## 2. The test set

`payoff_tables/mixedcontrol_m{M}_{NN}_chneurusa.xlsx`, 40 tables, 10 each at
M = 1..4, with profiles in `reports/planted_profiles_v3/`.

Built by working *backwards* from a value function: choose V, force M exact ties,
derive the profile that best-responds to it, then choose payoffs making that V the
true value function of the resulting T via `u = (I - delta T) V / (1 - delta)`.
The equilibrium is then correct by construction, not by search.

Three guards, in the order they run:

1. the planted profile must verify **under the framework committee rule**
   (`heyen_lehtomaa_2021`; jeres's own `voters()` disagrees on 9 of 60 transitions);
2. **all 2^M pure variants must fail** -- push every knob to a bound and check. This
   is the guard that matters and it does not route through any ranking
   representation;
3. `ordinal_ranking` must find no pure equilibrium (secondary).

Guard 2 was added after an earlier batch was certified with guard 3 alone: 10 of 38
tables had pure equilibria that `ordinal_ranking` structurally cannot see (section
5). It rejected 4,148 candidates in the rebuild -- 93% of all rejections.

A hardness filter also rejects tables where more than 10% of random theta verify.
Without it the benchmark measured almost nothing: on the first batch theta = 0.5
alone verified for 8 of 10 tables at M=1. After it, a random point verifies ~0% of
the time at M >= 2.

**Known defect, not yet fixed.** `build_one` fixes proposals using an absolute
1e-15 threshold and *then* affinely rescales V, which can push a sub-threshold gap
above it. The planted profiles remain valid (the verifier accepts within `atol`)
but are internally inconsistent. Apply the rescale before choosing proposals.

## 3. Solving given the support pattern: 40/40

`scripts/solve_support_v2.py`, median under 0.05 s, max 0.18 s.

| M | tables | solved | median | max |
|---|--------|--------|--------|-----|
| 1 | 10 | 10 | 0.02 s | 0.04 s |
| 2 | 10 | 10 | 0.01 s | 0.02 s |
| 3 | 10 | 10 | 0.03 s | 0.04 s |
| 4 | 10 | 10 | 0.15 s | 0.18 s |

Three things made it work, and all three are non-obvious:

- **Search the CLOSED box [0,1].** A tie-resting pure equilibrium is a legitimate
  endpoint of the feasible set; demanding strict interiority discards it.
- **Bounded trust-region least-squares, not `hybr`.** The Jacobian has a ZERO
  DIAGONAL -- a mixer's own indifference is insensitive to their own probability --
  which Newton handles badly and worse as M grows.
- **At M = 1, scan rather than root-find.** With one mixer the other players'
  conditions are inequalities, so the solution is an INTERVAL, and the mixer's own
  residual is flat across it. At M >= 2 those conditions become equations and a
  root-find is well posed.

**Caveat on the M labels.** Most planted knobs are inert: alpha reaches V only via
`T[x,y] += rho * sigma * q`, so a tie on a transition nobody proposes changes
nothing. Mean load-bearing knobs are 1.00 / 1.10 / 1.30 / 1.30 for M = 1..4, and
only 3 of 40 tables have more than two. **The effective dimension is ~1 regardless
of the M label**, so "100% at M=4" is weaker than it reads. The generator should
require knobs to be load-bearing.

## 4. Inside jeres_vfi, warm: 0/40 -> 40/40

Enabled by `mixed_solve=True`; the default path is byte-identical and unaffected.
Warm means `V_init` = the planted V. Five defects, in the order they mattered:

1. **The MIP cannot determine a freed alpha.** `solve_state_mip` allocates it as a
   variable but optimises a ZERO objective, so the LP returns it on a bound --
   measured 570/570 at 0. Wrong alpha -> wrong q -> wrong proposal argmax -> a V
   without the tie, so the iteration walks off any V that carries one. This is why
   seeding VFI with a mixed equilibrium's own V recovered nothing.
2. **Both directions of each tie were freed.** Their residuals are negatives of one
   another, so the system is rank-deficient and flat and every root-find stalls.
   The equilibrium needs only ONE direction free; the other takes the sign rule's
   value, which at a tie is 0. Which one is not known a priori, so try both -- 2 per
   tie, and ties are few. This is what let a root-find replace a grid.
3. **The `return` was missing.** The verified profile was assigned and then discarded
   by the continuing loop.
4. **The proposal argmax was recomputed inside the residual**, making it
   discontinuous -- it jumps whenever the argmax switches. Patterns are now
   enumerated once and held fixed.
5. **The argmax used machine epsilon; the verifier uses `atol`.** `verify_proposals`
   treats gains within `atol` as tied and accepts either choice, so a stricter rule
   emits only one of the tied patterns -- and if the equilibrium uses the other it is
   never generated. On `m1_07`, 8 of 15 slots are tied at `q*g = 0`. Comparing at
   `verify_atol` closed the last failure.

**A wrong turn worth not repeating.** Restricting the search to alphas that are
"load-bearing for T" dropped 27/40 to 9/40. An alpha on an unproposed transition
cannot change T, but it still enters the proposer's `q(y)*g(y)` comparison over ALL
targets, so pinning it moves the argmax off the pattern being tested.

## 5. ordinal_ranking cannot find mixed equilibria

Two settled fixes came out of this and are worth keeping regardless:

- a crash in `weak_equality.py`, where `nb_iters` was read on the SUCCESS path but
  only assigned under `if _use_nb` -- a solved pattern was reported as a failure;
- `_USE_BOUNDED_SOLVER`, replacing the logit-space Newton/`hybr` inner solve with
  bounded trust-region least-squares in physical space: 15/40 -> 26/40 on the same
  controls. The legacy path is retained behind the flag.

But the enumeration itself is the barrier, and it is structural. See the entry under
"Suggestive, not established" in CLAUDE.md for the full argument and its caveats.
In short: OR enumerates value RANKINGS and DERIVES a profile from each. Acceptance
is derived faithfully; proposals are derived by an ORDINAL rule ("propose the
approved target in your best tier") while the equilibrium condition is CARDINAL
(argmax of `q(y)*(V(y)-V(x))`). Those agree only when every q = 1. Measured: the
planted support pattern is reachable by SOME weak order in **0 of 40** cases, with a
positive control at 8/8. Repairing it needs proposals enumerated as a second axis --
5^15 ~ 3e10 per weak order -- and the planted target sits near-uniformly across tier
positions, so truncating the branch loses most solutions.

## 6. What is open

**Cold is untested, and there is a specific reason to expect it to fail.** The mixed
path triggers on `_freed_alphas`, which tests for EXACT ties (`|dV| <= 1e-12`). Cold
VFI starts from payoffs and cycles; its iterates have near-ties but essentially never
exact ones, so the path may never fire. Treating near-ties within a tolerance as
candidate knobs is the next concrete step, and it decides whether any of this reaches
production games.

Also open: the 60 real cells (blocked on cold); runtime (3m33 for 40 tables warm,
too slow for a delta sweep); and the generator defects in section 2.

## 7. What this does NOT change

The pure/mixed map stands: 120 cells with a verified pure equilibrium, 60 with none
found. Nothing here touches the default solver path, and the pure solves still
verify. The 60 cells remain conditional in exactly the way section 5 describes --
sound where the game has no exact V tie at a would-be equilibrium, which the payoff
audit supports (no exact payoff ties in any of the nine trios) but does not prove.
