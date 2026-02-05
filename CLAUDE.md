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
python -m viz_service

# In a new terminal, start the frontend
cd viz
npm install  # first time only
npm run dev
```

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

**TransitionProbabilities (lib/probabilities.py)** - Maps player strategies to state transition probabilities.
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

### Notes
- Never use fallbacks. Better fail early than that it seems it works while it actually doesnt. 
- Never use git commands, that's entirely controlled by the user. 
- I am running /viz using npm run dev, so no rebuild is necessary after changes, is done automatically. When do you changes to the viz/viz_service.py, you will need to restart though. 
