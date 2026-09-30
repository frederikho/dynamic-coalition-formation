# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements a **flexible framework** for analyzing farsighted coalition formation in the context of solar geoengineering governance. The framework can accommodate different numbers of countries, various governance structures, heterogeneous country characteristics, and different climate-economic models.

### Original Publication (2021)

The code was originally developed for "Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation" by Heyen & Lehtomaa (Oxford Open Climate Change, 2021). That paper used a **three-country illustrative example** to demonstrate the framework's capabilities.

### Framework Capabilities

The framework addresses how different governance structures affect international cooperation on solar geoengineering (SG) deployment. A key concern is the "free-driver" problem: a country with strong preferences for cooling might unilaterally deploy SG to its preferred level, imposing damages on others who prefer less or no intervention.

Two main advantages over traditional static coalition formation models:

1. **Dynamic farsightedness**: Countries anticipate future coalition changes. For example, when considering leaving a coalition, a country foresees that its departure might trigger further disintegration among remaining members, which affects the initial decision to leave.

2. **Institutional flexibility**: The framework can model different "rules of the game" by varying:
   - Number of countries (3, 5, 10, or more)
   - Protocol (who proposes changes and when)
   - Approval committees (who must consent to transitions)
   - Voting rules (unanimity vs majority)
   - Treaty characteristics (reversible vs irreversible, open vs exclusive membership)
   - Country characteristics (power, preferences, climate impacts)
   - Climate-economic models (simple to complex)

### The 2021 Paper's Three-Country Example

**Note: This was the illustrative example used in the 2021 publication. Current/future work may use different assumptions.**

The 2021 paper featured three countries with different baseline temperatures:
- **W (Warm)**: Base temperature 21.5°C, ideal SG level 11.5°C cooling
- **T (Temperate)**: Base temperature 14.0°C, ideal SG level 4.0°C cooling
- **C (Cold)**: Base temperature 11.5°C, ideal SG level 1.5°C cooling

Climate change had warmed all countries uniformly by 3°C. All countries had the same ideal temperature (13°C), equal power shares (1/3 each), and faced quadratic climate damages. SG deployment was assumed costless and to provide uniform global cooling (G).

**Payoff function (2021 paper)**: ui(G) = G · (2αi - G), where αi is country i's ideal SG level. This was normalized so zero SG deployment gives zero payoff; positive payoffs indicate benefiting from SG.

## Main Scenarios from the 2021 Paper

**Note: These scenarios are specific to the 2021 publication. New projects may analyze different scenarios with different parameters.**

The 2021 paper analyzed four experiments comparing different governance regimes in the three-country model:

### 1. Weak Governance (Main Text)
Countries are free to deploy SG unilaterally without restriction.

**Key result**: The system converges to the absorbing state ( ) where all countries are singletons. Country W deploys its ideal level (G=11.5°C), acting as a "free-driver" and imposing substantial damages on T and C who prefer much less cooling. This demonstrates the free-driver problem in the absence of governance.

**Why dynamics matter**: Even though T and C could form coalition (TC), it's equivalent to ( ) under weak governance since W deploys unilaterally regardless. The static stability conditions would also predict W always leaves any coalition.

### 2. Power Threshold (Main Text)
SG deployment requires the deploying coalition to possess at least 50% of global power. With equal power shares (1/3 each), only coalitions of 2+ countries can deploy.

**Key result**: The system converges to absorbing state (TC) with moderate SG deployment (G=2.75°C, the midpoint of T's and C's ideal levels). Country W would prefer to join but both T and C prefer excluding W to avoid excessive cooling.

**Why dynamics matter**: The grand coalition (WTC) appears stable under static analysis (no single-step profitable deviations), but farsighted players see through the facade. Country C anticipates that if it breaks away, T will follow, eventually reaching the preferred state (TC). Country T has the same incentive. This cycle of anticipated disintegration makes (WTC) unstable despite no immediate profitable deviations.

### 3. Power Threshold with Heterogeneous Damages (Supplementary)
Same as scenario 2, but with different marginal damage parameters (W: 0.75, T: 1.25, C: 1.0) to explore robustness. Uses unanimity for approval.

### 4. Power Threshold without Unanimity (Supplementary)
Same parameters as scenario 3, but uses majority approval rules instead of unanimity. Under majority rule, only approval from a majority of existing members (plus all new joiners) is required for transitions.

**Key insight**: Majority approval can stabilize the grand coalition (WTC). When C considers breaking away anticipating (TC) as the final state, it realizes that T could later invite W back to form (WTC) without C's approval. Since deviating from (WTC) causes temporary losses, C no longer wants to initiate the disintegration cycle.

## Key Terminology

- **State**: A coalition structure, e.g., '( )' = all singletons, '(TC)' = T and C cooperate, '(WTC)' = grand coalition
- **Protocol**: The mechanism determining which country proposes state changes (uniform = each country has equal probability 1/3)
- **Approval committee**: The set of countries whose consent is required for a proposed transition (derived from effectivity correspondence)
- **Effectivity correspondence**: Mapping from (proposer, current_state, next_state, responder) to whether responder is in approval committee
- **Value function** V(x): Long-run expected payoff for a player in state x, accounting for future state transitions
- **Farsightedness parameter** δ: Discount factor (0.99 in paper) controlling how much future payoffs matter relative to immediate payoffs
- **Absorbing state**: A state from which the system never transitions away in equilibrium
- **Free-driver problem**: Unlike free-riding (not contributing to public good), free-driving means a single actor deploys SG excessively, harming others

## Commands

### Running the main simulation
```bash
python main.py
```
This replicates all results from the paper. Results are written to the `results/` folder as LaTeX tables.

### Interactive Visualizer (NEW)

An interactive transition graph visualizer is available to explore coalition formation dynamics:

```bash
# Start the backend service
python -m service_viz

# In a new terminal, start the frontend
cd viz
npm install  # first time only
npm run dev
```

### Rendering transition-graph images (for AI agents)

To produce an image of the transition graph (the right-hand panel of the visualizer) from the command line:

```bash
source .venv/bin/activate
python viz/render_graph.py eq_n3_power_threshold_RICE_by_GDP_fbbdac -o graph.png
```

Requires Playwright (`pip install playwright`). Auto-starts the backend/frontend if not running; reuses them if they are. See `VIZ_README.md` → "Command-Line Graph Rendering" for options.

See `VIZ_README.md` for complete documentation.

### Equilibrium Solver (NEW)

An equilibrium solver is available to automatically find equilibrium strategy profiles from random initialization:

```bash
# Find equilibrium for a scenario
python3 find_equilibrium.py --scenario power_threshold

# Test solver on known equilibria
python3 test_equilibrium_solver.py

# Or use as a module
python3 -m lib.equilibrium.find --scenario weak_governance
```

The solver uses a **smoothed fixed-point iteration algorithm** with simulated annealing to discover equilibria computationally. This extends the framework beyond hand-picked strategy profiles to allow exploration of new scenarios.

**Key features:**
- Finds equilibria from random initialization
- Validated on all three equilibria from the paper (exact recovery)
- Configurable solver parameters for robustness
- Outputs standard Excel strategy tables compatible with existing code

See `lib/equilibrium/README.md` for complete documentation and algorithmic details.

### Testing
```bash
pytest
```
Runs all tests in the `tests/` directory.

### Testing specific modules
```bash
pytest tests/test_mdp.py
pytest tests/test_state.py
pytest tests/test_coalition.py
```

## Architecture

### Core Game Structure

The model is built on five interconnected classes that represent different aspects of the coalition formation game:

**Country (lib/country.py)** - Individual players with temperature preferences and power.
- Each country has: base temperature, climate-induced temperature change, ideal temperature, marginal damage parameter, and power share
- Calculates payoffs as negative damages from geoengineering deployment
- Key properties: `ideal_geoengineering_level`, `weighted_damage`, `payoff(G)`

**Coalition (lib/coalition.py)** - Groups of cooperating countries.
- Aggregates member country powers and preferences
- Calculates `avg_ideal_G` (average ideal geoengineering level) using weighted damage parameters

**State (lib/state.py)** - Coalition structures representing system configurations.
- Named like '( )' (all singletons), '(TC)' (T and C cooperate), '(WTC)' (grand coalition)
- Determines which coalition can implement geoengineering via `strongest_coalition` based on power rules
- Two power rules: 'power_threshold' (coalition needs min_power share) or 'weak_governance' (free-driver case)
- Calculates static payoffs for all countries given the coalition structure

**TransitionProbabilitiesOptimized (lib/probabilities_optimized.py)** - Maps player strategies to state transition probabilities.
- Reads strategy profiles from Excel files in `strategy_tables/`
- Derives effectivity correspondence (who approves which transitions)
- Supports two approval mechanisms: unanimous or majority-based
- Returns three probability matrices: P (state transitions), P_proposals (proposition strategies), P_approvals (approval probabilities)

**MDP (lib/mdp.py)** - Solves the Markov Decision Process.
- Given static payoffs, transition probabilities, and discount factor
- Solves linear system to find value functions (long-run expected payoffs) for each player in each state
- Implementation: `V = (I - γP)^(-1) * (1-γ) * u` where γ is discounting, P is transition matrix, u is static payoffs

### Workflow

The main simulation follows this sequence (see main.py:run_experiment):

1. **Initialize countries** with parameters (temperatures, damages, power shares)
2. **Create coalition structures** - All 5 possible states with different coalition configurations
3. **Load strategy profiles** from Excel files in `strategy_tables/`
4. **Derive effectivity** - Determines approval committees from strategy table structure
5. **Calculate transition probabilities** based on proposition and approval strategies
6. **Solve MDP** - Compute value functions for each player in each state
7. **Verify equilibrium** - Check that strategies are consistent with value functions (no profitable deviations)
8. **Write results** to LaTeX tables

### Important Assumptions in the 2021 Paper's Example

**Note: These were simplifications made for the 2021 publication's illustrative three-country example. The framework is flexible and can be extended in many directions for new projects.**

1. **No side-payments**: Countries cannot make transfers to each other to incentivize coalition formation. (Framework extension: side-payments can be incorporated.)

2. **Unilateral exit allowed**: Any country can leave a treaty without approval from other members (reflects many real international treaties including Paris Agreement). Remaining members stay together at least temporarily. (Framework extension: can model different exit rules.)

3. **Markovian strategies**: Current strategies only depend on the current state, not on the full history of negotiations. (Framework extension: history-dependent strategies could model reputation effects.)

4. **Uniform protocol**: Each country has equal probability (1/3) of being selected as proposer in each period. (Framework extension: protocol can weight countries by power or other characteristics.)

5. **Equal power shares**: All countries have power = 1/3 in the 2021 example. (Framework capability: supports heterogeneous power shares.)

6. **Disjoint coalitions**: Each country belongs to at most one coalition. In the three-country model, there are 5 possible coalition structures: ( ), (TC), (WC), (WT), (WTC). (Framework capability: with N countries, the number of possible coalition structures grows rapidly.)

7. **Stylized climate model**: The 2021 example assumed uniform temperature changes, quadratic damages, costless SG, and uniform cooling. (Framework capability: can accommodate more realistic climate-economic models with heterogeneous impacts, SG costs, etc.)

### Strategy Tables

Strategy profiles are defined in Excel files (e.g., `strategy_tables/weak_governance.xlsx`). The structure encodes:
- **Proposition probabilities**: For each state and proposer, probability distribution over next states
- **Acceptance probabilities**: For each transition, whether approval committee members approve
- **Effectivity correspondence**: Implicitly defined by which cells are filled (NaN = not in approval committee)

The effectivity correspondence determines who must approve each transition. Empty cells in the acceptance rows indicate a player is not part of that approval committee.

### Equilibrium Verification

The `verify_equilibrium` function (lib/utils.py) checks two conditions:

**Condition 1 (Proposals)**: Players only propose transitions with positive probability if they maximize expected value given approval probabilities.

**Condition 2 (Approvals)**: Approval committee members approve if V(next) > V(current), reject if V(next) < V(current), and can do either if indifferent.

### Model Parameters

Key configurable parameters in main.py:

- `base_temp`, `ideal_temp`, `delta_temp`: Temperature parameters for each country
- `m_damage`: Marginal damage parameter (quadratic loss function coefficient)
- `power`: Each country's share of global power (must sum to 1)
- `protocol`: Probability distribution over who proposes (usually uniform)
- `discounting`: Discount factor γ (typically 0.99) - controls farsightedness
- `power_rule`: 'power_threshold' or 'weak_governance'
- `min_power`: Minimum power share required to implement geoengineering (for power_threshold)
- `unanimity_required`: Boolean for approval committee voting rule

### Results

Output files in `results/` are LaTeX tables containing:
- `V_*.tex`: Value functions (long-run expected payoffs) for each state and player
- `payoffs_*.tex`: Static payoffs for each state and player
- `P_*.tex`: State transition probability matrices
- `geoengineering_*.tex`: Geoengineering deployment levels by state

## Working with Experiments

### Modifying Existing Experiments

To test different model parameterizations:

1. **Change climate parameters**: Edit `base_config` in main.py to modify baseline temperatures, ideal temperatures, climate change magnitude (delta_temp).

2. **Change damage parameters**: Edit `m_damage` in individual experiment configs to vary how much countries care about temperature deviations.

3. **Change power distribution**: Modify `power` dict (must sum to 1) to make countries asymmetric in influence.

4. **Change farsightedness**: Adjust `discounting` (δ). Values closer to 1 make countries more patient and farsighted. Values closer to 0 make them myopic.

5. **Change governance rules**:
   - Set `power_rule` to 'weak_governance' or 'power_threshold'
   - Adjust `min_power` for power threshold scenarios
   - Toggle `unanimity_required` between True and False

### Creating New Strategy Profiles

Strategy profiles in `strategy_tables/*.xlsx` have a specific structure:
- Multi-level column headers: (Proposer W, state), (Proposer T, state), (Proposer C, state)
- Multi-level row headers: (state, "Proposition", NaN) for proposal rows, (state, "Acceptance", country) for approval rows
- **Proposition rows**: Probability distribution over next states for each proposer in each current state
- **Acceptance rows**: Probability of approval by each country in the approval committee (NaN = not in committee)

To create a new strategy profile:
1. Copy an existing Excel file as a template
2. Modify proposition probabilities (must sum to 1 for each proposer in each state)
3. Modify acceptance probabilities (0 = reject, 1 = approve, 0<p<1 = mixed strategy)
4. Leave cells blank (NaN) for countries not in the approval committee for that transition
5. Reference the new file in a new experiment config in main.py

### Equilibrium Existence: an equilibrium ALWAYS exists

**An equilibrium exists for every game we solve. There are no exceptions.** This is
established in Heyen & Lehtomaa (2021), Supplementary Material, Section A:

> "The existence of an equilibrium is guaranteed under the conditions of continuous
> payoff functions u_i(x) and compact state space X (Harris 1985; Ray 2007), but in
> general the equilibria are not unique."

and in its footnote 11:

> "A stationary Markov equilibrium exists for finite X (Hyndman and Ray 2007,
> Supplementary notes)."

Our state space is always finite (5 states for n=3, 15 for n=4), so the existence
guarantee applies unconditionally to every payoff table in `payoff_tables/`.

**The operational consequence — this is the important part:**

When a solver reports "no equilibrium found", that is a statement about *the solver
and its settings*, never about the game. It means one of:

- the iteration budget ran out before convergence (`jeres_vfi_max_iter` defaults to
  300, but VFI contracts at rate delta, so delta=0.999 needs ~20,000 iterations —
  see the delta-dependence note below);
- the verification tolerance was set tighter than the solver's achievable precision;
- the equilibrium requires **mixed** strategies of a kind the solver cannot reach.
  Note `jeres_vfi` *does* search for mixed strategies -- `_resolve_cycle` bisects on
  the interpolated value function, with a mean-strategy fallback -- but only along a
  1-D line between two phases of a detected cycle. The "0/1 acceptance only"
  limitation belongs to `ordinal_ranking`, NOT to `jeres_vfi`;
- the search missed it (initialization, local optimum). Note this is NOT about
  restarts -- see "the multi-start is inert" below.

Never write up, log, or report a solver failure as "this game has no equilibrium",
and never treat a failure rate as an economic finding. Diagnose the setting instead.
Conversely, a *pure*-strategy equilibrium is not guaranteed, so "no pure equilibrium
found here" is a legitimate conclusion once convergence has been ruled out.

### Interpreting Equilibrium Verification

When `verify_equilibrium` fails, the error message indicates:
- **Proposal errors**: A country is proposing a state that doesn't maximize expected value, OR not proposing a state that would maximize expected value
- **Approval errors**: A country is approving/rejecting inconsistently with whether the new state improves their value function

Common reasons for failures:
- Strategy profile doesn't account for farsighted incentives
- Numerical precision issues (value functions very close, use atol=1e-12 tolerance)
- Incorrect effectivity correspondence (wrong approval committees)

### Understanding State Transitions

The transition probability matrix P shows how likely the system moves from one state to another in a single period:
- P[x,y] = probability of transitioning from state x to state y
- Diagonal elements P[x,x] = probability of staying in state x
- Each row sums to 1 (system must be somewhere next period)

High-probability transitions indicate the likely evolution path. Absorbing states have P[x,x] = 1.

### Solver settings: discounting, tolerance, iteration budget

These three interact, and getting them wrong produces failures that look like results.

**`jeres_vfi` is policy iteration, and its iteration budget matters sometimes.**
Despite the name, `vfi()` is not value function iteration. Each pass does a greedy
policy improvement (`_vfi_step`) followed by an *exact* policy evaluation
(`compute_values` solves `(I - delta*T) V = (1-delta) u` directly). That is Newton's
method on the Bellman equation, so a **converging** run finishes in tens of
iterations no matter how close delta is to 1 — the delta^k intuition from true value
iteration does not apply, and reasoning from it gives wrong diagnoses.

But the loop can **cycle** instead of converging, and the cycle-detection and
bisection path needs iterations to find and resolve those cycles. Both of these are
measured facts on the n=3 batch, and they pull in opposite directions:

- Raising `--jeres-max-iter` from 300 to 5,000 (delta=0.99) or 30,000 (delta=0.999)
  changed **no** verdict for chneurusa, nderususa or chnrususa.
- Raising it from 300 to 1,375 flipped **eurrususa** at rtol=1e-2 from failure to a
  verified equilibrium, at an identical verification tolerance (63s -> 284s).

So: never explain a failure by the iteration budget without testing it, and never
leave the budget at the default when it costs real solutions. `--verify-rtol` now
raises it automatically (see `_cycle_resolution_budget()`); when running the solver
directly, pass `--jeres-max-iter` explicitly at high delta.

**`vfi()` returns unconverged values silently** at `max_iter` (see its docstring:
"or at max_iter if not converged"). A run that never converged is currently
indistinguishable from one that did. Check convergence explicitly before believing
any high-delta result.

**The multi-start is inert. Do not add restarts or seeds.** (But see the
warm-start carve-out immediately below -- *random* multi-start is what is inert.) VFI here is globally
convergent to a unique fixed point: 12 restarts from far-apart initialisations
(initial |dV| 13-24, against a payoff range ~0.004) all reach a **bit-identical**
V in 3-8 iterations. Confirmed across all 91 tables -- switching restart noise
scaling changed 0 verdicts, twice. If a game's unique fixed point is not an
equilibrium, no number of restarts can find one, and a restart ladder just burns
hours re-deriving the same answer.

**Carve-out, measured 2026-08-18: INFORMED warm starts are NOT inert.** Seeding VFI
with the equilibrium V of the same table at an adjacent delta (`--jeres-v-init`)
flips **25 verdicts** that cold starts fail, at identical seed, tolerance and
restart count (1). `nderususa` goes from a cold failure band of [0.88, 0.96] to
solving continuously from 0.86 to 0.985. See `reports/delta_continuation/`.

The inertness argument does not reach this case for two reasons: it was measured on
runs that CONVERGE (the failing tables cycle, so no unique fixed point is reached
and uniqueness cannot be invoked), and it concerns RANDOM draws from
`payoffs + N(0, 10*spread)`, a diffuse ball that hits a strategically coherent
point with probability zero. The gain comes from STRUCTURE in the initialisation,
not from more initialisations -- so the advice against restart ladders and seed
sweeps stands unchanged.

Operational consequence: a cold-start failure is now weak evidence of a hard game.
Try continuation from a neighbouring parameter value before concluding anything,
and read a *stall under continuation* -- ideally bracketed from both sides -- as
the real signal that a pure branch has terminated.

### Verification tolerance: `atol` is a NUMERICAL parameter, not an economic one

This caused more wasted effort than anything else in this codebase. Read it before
touching a tolerance.

**Where it enters.** Exactly two places in `lib/utils.py`, both widening what counts
as a *tie*, so a larger `atol` is always a WEAKER test:
- `verify_approvals`: inside `atol` the condition becomes vacuous (any probability in
  [0,1] passes); outside it, exact behaviour is demanded (`p == 1.` / `p == 0.`).
- `verify_proposals`: `atol` widens the argmax set, easing the subset test.

Both use `rtol=0`, so the tolerance is purely absolute and only means anything
relative to the units of V.

**Its only legitimate job is absorbing floating-point error in the V solve.** The
conditions themselves are exact; in exact arithmetic no tolerance would be needed.
Measured on a real game:

| delta | cond(I-delta*P) | forward error on V | V spread |
|-------|-----------------|--------------------|----------|
| 0.9   | 18              | 4e-15              | 3e-1     |
| 0.99  | 194             | 4e-14              | 3e-1     |
| 0.999 | 1964            | 4e-13              | 3e-1     |

Thirteen orders of magnitude separate the noise floor from the signal, so **any**
choice in that window gives identical verdicts. Use a fixed `--verify-atol 1e-12`
(1e-10 at delta=0.999). One number works for every table because normalisation puts
every game's V on a comparable scale and the error bound depends on conditioning and
machine epsilon, not on the payoffs.

**Anything above ~1e-10 is not tolerance, it is redefining the game** -- declaring
real preferences to be ties. Tuning `atol` upward until a run passes produces
"equilibria" that are artefacts; at `atol` >= the V spread the verifier accepts any
strategy profile whatsoever.

**Compare `atol` to the V SPREAD, not the payoff spread.** The `(1-delta)` factor
compresses V far below the payoff range, so "1% of what is at stake" can be 100% of
what the verifier actually has to resolve.

**`--verify-rtol` and `payoff_scale()` are superseded.** They were built to make a
tolerance comparable across raw-unit tables, which normalisation already achieves,
and they scale to the payoff spread rather than the V spread. Prefer a fixed
`--verify-atol`.

### Payoff normalisation (on by default)

`find_equilibrium` rescales each player's payoffs to [0,1] before solving.
`--no-normalise-payoffs` opts out, for reproducing historical runs only.

**Why it is safe.** Every equilibrium condition compares a player against
*themselves*, so a per-player positive affine map `u_i -> a_i*u_i + b_i` leaves the
equilibrium set exactly unchanged (V carries the map through; argmaxes and value-gap
signs are preserved). Proven analytically and guarded by
`scripts/check_solver_invariance.py`, which tests arbitrary maps, not just [0,1].

**Why it matters.** RICE payoffs sit near -13 with spreads near 1e-4, so the whole
strategic content lives in the fifth significant digit. Measured over 91 n=3 tables:
solve rate 42/91 -> 67/91, timeouts and failures both halved, 32% less wall time.

**What it BREAKS if ignored: cross-player quantities.** Normalisation is exact per
player but destroys the common unit *between* players. Utilitarian welfare sums must
read `setup['payoffs_raw']`.

**The unit convention is now recorded and honoured (fixed 2026-08-18).** Profiles
carry a `normalise_payoffs` metadata field, and `run_verification` reconstructs V in
the convention the profile was solved in, while keeping `details['payoffs']` and
`details['V_raw']` in raw units so cross-player sums stay meaningful. Guarded by
`tests/test_verification_units.py`.

**An earlier note here claimed `compare.py`'s `mpe_expected_welfare` was meaningless
for normalised profiles. That was wrong** -- `run_verification` loads payoffs fresh
from the payoff table in raw units and never reads the profile's normalised V, so the
welfare numbers were always in raw units. The real defect was narrower: `atol` is
ABSOLUTE, and raw V spreads run ~5100x smaller than normalised ones on RICE tables
(measured on `kalkuhl_chnrususa` at delta 0.99: 3.17e-6 raw vs 1.63e-2 normalised).
So a tolerance chosen as strict during solving became a near-vacuous test on reload.
Profiles written before the fix cannot have their convention recovered;
`run_verification` treats them as raw (the historical behaviour) and says so.

### Benchmarking a solver change

`scripts/bench_solver.py run --label X` then `compare before after`. Runs every n=3
payoff table (~6-17 min), prints newly-solved AND regressions by name. Use it for any
solver change; several plausible-sounding fixes in this codebase changed exactly
nothing, and only the benchmark revealed that.

Caveats when quoting its numbers: **22 of the 91 tables are synthetic `simple_cycle`
fixtures**, so 91-denominators are not production scenarios; and a timeout is not a
failure -- report them separately.

**Current honest ceiling: 44/91 tables at `atol` 1e-10.**

### Known numerical traps

**MIP probabilities can fall outside [0,1].** HiGHS reports within its own
feasibility tolerance, so a variable bounded to [0,1] can return `-1.0011e-14`
(observed). The verifier tests probabilities by EXACT equality, so such a value fails
both the indifference branch (not >= 0) and the strict branch (not == 0), rejecting a
good profile over a rounding artefact. Handled by `_clean_probability` in
`jeres_vfi/solver.py`, which must both clip AND snap onto the bounds -- clipping
alone leaves `1 - 1e-14` failing `p == 1.`.

### Suggestive, not established

Flagged so they are not mistaken for findings:

- **Richer cycle mixing probably would not help.** Searching the full simplex spanned
  by the cycle phases (rather than `_resolve_cycle`'s 1-D line) finds no verifying
  point on any of the six unsolved n=3 trios; best residual 5e-3, seven orders above
  tolerance. But the search is numerical, an equilibrium is generically an isolated
  point, and the objective is discontinuous (strategies come from a MIP). Only the
  exact algebraic route (`lib/equilibrium/full_search`, msolve) can *prove* absence.
- **Table families differ sharply in conditioning** (burke spread/level 1.3e-2 vs
  kalkuhl 3.2e-5, measured). Whether that makes kalkuhl harder *after* normalisation
  is NOT established.
- **The remaining failures may need supports the VFI dynamics never visit.** An
  inference from the above plus msolve branch statistics, not a demonstrated fact.
- **`ordinal_ranking` may be structurally unable to find MIXED equilibria, and the
  gap may not be closable at feasible cost.** Treat as a warning, not a verdict --
  it rests on 40 synthetic control tables, not on production games.

  OR enumerates value RANKINGS and *derives* a full profile from each. The
  acceptance half of that derivation looks faithful: in equilibrium alpha must
  follow sign(dV), so the weak order pins it, and 4-12 weak orders per player passed
  the acceptance constraints on every control table. The PROPOSAL half is derived by
  an ordinal rule -- "propose the approved target in your best tier" -- while the
  equilibrium condition is cardinal: argmax over q(y) * (V(y) - V(x)). Those agree
  when every q = 1, which is why strict mode is fine for pure equilibria. Once
  acceptance mixes, q < 1, and a lower-ranked target with high approval probability
  can beat a higher-ranked one with low approval probability.

  Measured on the 40 controls (`reports/planted_profiles_v3`,
  `scripts/or_reachability_test.py`): the planted support pattern was reachable by
  SOME weak order in **0 of 40** cases, with acceptance satisfiable every time and
  proposals never. A positive control -- patterns derived from a weak order and fed
  back in -- returned 8/8 reachable, so the test itself discriminates.

  Why the obvious repair looks unaffordable: all five targets were approved in all
  600 slots, so branching over proposals costs 5^15 ~ 3e10 per weak order, ~5e18
  overall. And the planted target sat almost uniformly across tier positions
  (100/124/138/133/105 for positions 0-4), so truncating to the top few loses most
  solutions -- OR's current "position 0 only" is right about 17% of the time, near
  chance.

  Caveats before anyone acts on this: the controls are synthetic games built by
  planting exact ties in V, which is not how RICE payoffs behave; the reachability
  test encodes OR's derivation as this author reads it, and that reading has been
  wrong twice before in this session; and none of it touches strict mode, where
  q = 1 makes the ordinal rule exact -- so the pure-equilibrium results are
  unaffected.

  Two real fixes came out of the same investigation and ARE settled: a crash in
  `weak_equality.py` where `nb_iters` was read on the success path but only assigned
  under `if _use_nb` (a solved pattern was reported as a failure), and
  `_USE_BOUNDED_SOLVER`, which replaces the logit-space Newton/hybr inner solve with
  bounded trust-region least-squares in physical space -- 15/40 -> 26/40 on the same
  controls. The legacy path is retained behind that flag.

### Mixed equilibria: solving vs finding the support pattern

An equilibrium is a **support pattern** (69 discrete facts at n=3: which targets each
proposer plays, and whether each of the 54 acceptances is 0 / 1 / interior) plus the
**M interior values**. For a pure slot the support IS the value, so only the interior
slots leave a number to determine.

**Given the support pattern, solving is free.** 40/40 on the synthetic controls in
under 0.2 s (`scripts/solve_support_v2.py`), and 40/40 inside `jeres_vfi` with
`mixed_solve=True` when seeded with the right V. Three non-obvious requirements:
search the CLOSED box [0,1] (a tie-resting pure equilibrium is a legitimate
endpoint); use bounded trust-region least-squares, not `hybr`, because the Jacobian
has a ZERO DIAGONAL (a mixer's own indifference is insensitive to their own
probability); and at M=1 scan rather than root-find, because with a single mixer the
other players' conditions are inequalities, so the solution is an INTERVAL.

**Finding the support pattern is the entire remaining difficulty.** Full write-up,
including the five defects fixed in `vfi()` and one wrong turn that cost 27/40 -> 9/40,
is in `reports/mixed_solver/FINDINGS.md`.

Three things there that will otherwise be rediscovered the hard way:

- `solve_state_mip` optimises a ZERO objective, so a freed alpha comes back on a
  bound (measured 570/570 at 0). Nothing in the MIP determines it.
- An exact tie frees BOTH directions, whose residuals are negatives of one another;
  keeping both makes every root-find stall on a rank-deficient, flat system. Free one
  direction, let the sign rule pin the other, try both choices.
- Generate proposals at the VERIFIER's tolerance, not machine epsilon.
  `verify_proposals` accepts gains tied within `atol`, so a stricter rule emits only
  one of the tied patterns and can miss the equilibrium entirely.

**Test set:** `payoff_tables/mixedcontrol_m{M}_*.xlsx`, 40 tables, profiles in
`reports/planted_profiles_v3/`. Every table has a planted mixed equilibrium AND no
pure equilibrium, so a solver cannot pass without mixing. Caveat when quoting the M
labels: most planted knobs are INERT (alpha reaches V only through
`T[x,y] += rho*sigma*q`), so the effective dimension is ~1 regardless of M.

**Untested: cold.** The mixed path triggers on EXACT ties (`|dV| <= 1e-12`), and cold
VFI's iterates have near-ties but essentially never exact ones, so it may never fire.
Do not assume warm success carries over.

### Notes
- Important: Never use fallbacks or placeholders! Better fail early than that it seems it works while it actually doesnt. 
- `jeres_vfi` is the working solver. `lib/equilibrium/full_search` (msolve) is exact
  but not a practical general tool: positive-dimensional branches are deferred
  (a null is inconclusive, never a non-existence proof) and it costs on the order of
  a day per table. Do not propose it as a routine alternative.
- A general solver should normalise payoffs internally rather than requiring
  preprocessed tables. Solving is unit-invariant; reporting is not.
- The n=3 setting is equal power (1/3 each) with `min_power` 0.501 -- deliberately
  closer to the 2021 paper than GDP-weighted power, which does not add much at n=3.
- Never use mutating git commands like git add or git commit, that's entirely controlled by the user. 
- I am running /viz using npm run dev, so no rebuild is necessary after changes, is done automatically. When do you changes to the viz/service_viz.py, you will need to restart though. 
- Activate the environment .venv before running code. If not you will get errors such as ModuleNotFound.
