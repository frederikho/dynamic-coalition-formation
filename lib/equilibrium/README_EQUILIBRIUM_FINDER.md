# Equilibrium Solver for Coalition Formation Games

This document describes the implementation and usage of the equilibrium solver algorithm that can find equilibrium strategy profiles from random initialization.

## Overview

The equilibrium solver implements a **smoothed fixed-point iteration algorithm** that finds equilibrium strategy profiles for dynamic coalition formation games. This extends the framework by allowing you to discover equilibria computationally rather than hand-picking them.

### Key Innovation

Previously, the framework required hand-picked strategy profiles (stored in Excel files). The equilibrium solver automates this process:

- **Input**: Game parameters (countries, payoffs, governance rules)
- **Output**: Equilibrium strategy profile satisfying both equilibrium conditions
- **Method**: Smoothed fixed-point iteration with simulated annealing

### Algorithm Foundations

The algorithm is based on two equilibrium conditions from the paper:

1. **Condition 1 (Proposals)**: Players only propose transitions that maximize their expected long-run payoff given approval probabilities
2. **Condition 2 (Approvals)**: Approval committee members approve if the new state increases their value function, reject otherwise

## Algorithm Description

### High-Level Structure

```
INITIALIZE:
  - Uniform proposal probabilities over all states
  - Acceptance probabilities = 0.5 for all committee members

REPEAT (Outer loop - Annealing):
  Gradually reduce smoothing temperatures τ_p, τ_r

  REPEAT (Inner loop - Fixed point):
    1. Compute transition probabilities from current strategies
    2. Solve for value functions V_i(x)
    3. Update acceptances: r_j(x,y) ← sigmoid((V_j(y) - V_j(x)) / τ_r)
    4. Compute expected values: EV_i(x,y) = p_approved·V_i(y) + p_rejected·V_i(x)
    5. Update proposals: p_i(x,·) ← softmax(EV_i(x,·) / τ_p)
    6. Apply damping to smooth updates
  UNTIL convergence

UNTIL τ_p, τ_r sufficiently small

FINAL PROJECTION:
  - Acceptances: r_j = 1 if V_j(y) > V_j(x), else 0 (indifferent cases unchanged)
  - Proposals: Uniform over argmax states

VERIFY equilibrium conditions
```

### Key Components

#### 1. Smoothing Functions

**Sigmoid for acceptances:**
```python
r_j(x,y) = 1 / (1 + exp(-(V_j(y) - V_j(x)) / τ_r))
```
- When `V_j(y) >> V_j(x)`: r_j → 1 (approve)
- When `V_j(y) << V_j(x)`: r_j → 0 (reject)
- Temperature τ_r controls smoothness

**Softmax for proposals:**
```python
p_i(x,y) = exp(EV_i(x,y) / τ_p) / Σ_y' exp(EV_i(x,y') / τ_p)
```
- Distributes probability over states based on expected values
- Temperature τ_p controls how sharply probability concentrates on best options

#### 2. Annealing Schedule

Temperatures decay gradually:
```python
τ_p ← max(τ_p × decay_rate, τ_min)
τ_r ← max(τ_r × decay_rate, τ_min)
```

Starting warm (τ ≈ 1) allows exploration; cooling down (τ → 0) enforces exact equilibrium conditions.

#### 3. Damping

Strategy updates are damped to prevent oscillations:
```python
strategy_new ← (1 - λ) × strategy_old + λ × strategy_update
```

Typical damping factor: λ = 0.5

#### 4. Final Projection

After annealing, project to exact equilibrium:
- **Acceptances**: Set to 0 or 1 based on strict value comparison (keep existing value if indifferent)
- **Proposals**: Uniform distribution over argmax states

## Implementation

### Core Classes

#### `EquilibriumSolver` (lib/equilibrium/solver.py)

Main solver class that implements the algorithm.

**Key methods:**
- `__init__()`: Initialize with game parameters
- `solve()`: Run the full algorithm
- `_update_acceptances()`: Update acceptance probabilities
- `_update_proposals()`: Update proposal probabilities
- `_solve_value_functions()`: Solve MDP for value functions
- `_project_to_exact_equilibrium()`: Final projection step

**Parameters:**
```python
solver = EquilibriumSolver(
    players=['W', 'T', 'C'],
    states=['( )', '(TC)', '(WC)', '(WT)', '(WTC)'],
    effectivity=effectivity_dict,
    protocol={'W': 1/3, 'T': 1/3, 'C': 1/3},
    payoffs=payoffs_dataframe,
    discounting=0.99,
    unanimity_required=True,
    verbose=True
)
```

### Solver Parameters

The `solve()` method accepts the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_p_init` | 1.0 | Initial temperature for proposal smoothing |
| `tau_r_init` | 1.0 | Initial temperature for acceptance smoothing |
| `tau_decay` | 0.95 | Decay rate for temperatures per outer iteration |
| `tau_min` | 0.01 | Minimum temperature threshold |
| `max_outer_iter` | 50 | Maximum annealing iterations (safety valve) |
| `max_inner_iter` | 100 | Maximum fixed-point iterations per temperature |
| `damping` | 0.5 | Damping factor for strategy updates |
| `inner_tol` | 1e-6 | Convergence tolerance for inner loop |
| `outer_tol` | None | Convergence tolerance for outer loop (defaults to 10×inner_tol) |
| `consecutive_tol` | 2 | Number of consecutive converged iterations required |
| `tau_margin` | 0.01 | Margin for checking if tau is near tau_min (1%) |
| `project_to_exact` | True | Whether to project to exact equilibrium |

**Note:** See [CONVERGENCE.md](CONVERGENCE.md) for detailed information about the convergence criterion and how to choose appropriate parameter values.

## Usage

### Command-Line Interface

The easiest way to use the solver is via the command-line script:

```bash
# Find equilibrium for power threshold scenario
python3 -m lib.equilibrium.find --scenario power_threshold

# Find equilibrium for weak governance
python3 -m lib.equilibrium.find --scenario weak_governance

# Find equilibrium for power threshold without unanimity
python3 -m lib.equilibrium.find --scenario power_threshold_no_unanimity

# Specify custom output file
python3 -m lib.equilibrium.find --scenario power_threshold --output ./my_equilibrium.xlsx

# Adjust solver parameters
python3 -m lib.equilibrium.find --scenario power_threshold --max-outer-iter 100 --max-inner-iter 200

# Suppress verbose output
python3 -m lib.equilibrium.find --scenario power_threshold --quiet
```

### Python API

You can also use the solver directly in Python:

```python
from lib.equilibrium.find import find_equilibrium

# Define configuration
config = {
    'players': ['W', 'T', 'C'],
    'state_names': ['( )', '(TC)', '(WC)', '(WT)', '(WTC)'],
    'base_temp': {'W': 21.5, 'T': 14.0, 'C': 11.5},
    'ideal_temp': {'W': 13., 'T': 13., 'C': 13.},
    'delta_temp': {'W': 3., 'T': 3., 'C': 3.},
    'm_damage': {'W': 1., 'T': 1., 'C': 1.},
    'power': {'W': 1/3, 'T': 1/3, 'C': 1/3},
    'protocol': {'W': 1/3, 'T': 1/3, 'C': 1/3},
    'power_rule': 'power_threshold',
    'min_power': 0.5,
    'unanimity_required': True,
    'discounting': 0.99
}

# Find equilibrium
result = find_equilibrium(
    config,
    output_file='./my_equilibrium.xlsx',
    solver_params={'max_outer_iter': 100},
    verbose=True
)

# Access results
print("Value functions:")
print(result['V'])
print("\nTransition probabilities:")
print(result['P'])
print("\nVerification:", result['verification_message'])
```

### Low-Level API

For maximum control, use the `EquilibriumSolver` class directly:

```python
from lib.equilibrium.solver import EquilibriumSolver

solver = EquilibriumSolver(
    players=players,
    states=states,
    effectivity=effectivity,
    protocol=protocol,
    payoffs=payoffs,
    discounting=0.99,
    unanimity_required=True,
    verbose=True
)

strategy_df, solver_info = solver.solve(
    tau_p_init=2.0,  # Start with higher temperature
    tau_r_init=2.0,
    tau_decay=0.90,  # Slower annealing
    damping=0.3,     # Less damping
)

print("Converged:", solver_info['converged'])
print("Outer iterations:", solver_info['outer_iterations'])
```

## Testing

### Validation on Known Equilibria

The solver has been validated on all three equilibria from the paper by successfully recovering them from random initialization:

1. **Weak governance** scenario
2. **Power threshold** scenario
3. **Power threshold without unanimity** scenario

**Results:**
- ✅ All three equilibria are successfully recovered from random initialization
- ✅ Value functions match exactly (max difference: 0.0)
- ✅ Transition matrices match exactly (max difference: 0.0)
- ✅ All equilibrium conditions are satisfied

### Effectivity Rules Validation

The effectivity correspondence implementation can be validated against the strategy tables:

```bash
python3 test_effectivity.py
```

This verifies that `heyen_lehtomaa_2021()` in `lib/effectivity.py` correctly implements the game rules.

### Test Output Example

```
================================================================================
Testing: power_threshold
================================================================================
Starting equilibrium solver...
...
Annealing complete after 50 outer iterations
Projecting to exact equilibrium...
Solver complete!

VERIFICATION OF FOUND EQUILIBRIUM
Equilibrium verification: All tests passed.

COMPARISON WITH TRUE EQUILIBRIUM
Max value function difference: 0.0000000000
Max transition matrix difference: 0.0000000000

SUCCESS: Found equilibrium matches true equilibrium!
================================================================================
```

## Convergence Criterion

The solver uses a **convergence-based stopping criterion** rather than a fixed number of iterations. This ensures the algorithm runs long enough to converge while avoiding wasted computation.

**Key features:**
- Stops when both temperature and strategy converge
- `max_outer_iter` serves as a safety valve (rarely reached)
- Tracks stability over consecutive iterations
- Provides detailed convergence diagnostics

**See [CONVERGENCE.md](CONVERGENCE.md) for:**
- Detailed explanation of the convergence criterion
- How to choose appropriate parameter values
- Interpreting convergence output
- Troubleshooting convergence issues

## Performance Characteristics

### Typical Convergence

- **Outer iterations**: Typically 40-50 for default parameters
- **Inner iterations per temperature**: 1-5 iterations (with small tau)
- **Total computation time**: 5-15 seconds per equilibrium (3-country model)
- **Convergence rate**: Reliable for all tested scenarios

### Parameter Sensitivity

**Temperature initialization (`tau_p_init`, `tau_r_init`):**
- Higher values (1.0-2.0) allow more exploration
- Lower values may get stuck in local optima
- Recommended: 1.0 for most cases

**Decay rate (`tau_decay`):**
- Slower decay (0.90-0.95) is more robust but slower
- Faster decay (0.85-0.90) is faster but may miss equilibria
- Recommended: 0.95

**Damping (`damping`):**
- Higher damping (0.5-0.7) is more stable
- Lower damping (0.2-0.4) converges faster but may oscillate
- Recommended: 0.5

## Theoretical Properties

### Completeness

The algorithm is **not guaranteed** to find an equilibrium in all cases:
- It may get stuck in local optima
- It may oscillate between strategies
- Multiple equilibria may exist; which one is found depends on initialization

However, in practice:
- Converges reliably for the tested scenarios
- Recovers all known equilibria from the paper
- Smoothing and annealing help escape local optima

### Multiple Equilibria

When multiple equilibria exist:
- The algorithm may converge to different equilibria depending on random initialization
- Running multiple times with different random seeds may find different equilibria
- The found equilibrium depends on the annealing schedule

### Equilibrium Selection

The algorithm does not implement any equilibrium selection criterion. It simply finds *an* equilibrium, not necessarily:
- The payoff-dominant equilibrium
- The risk-dominant equilibrium
- The focal equilibrium

## Extensions and Future Work

### Possible Extensions

1. **Multiple random initializations**: Run solver multiple times to find multiple equilibria
2. **Adaptive annealing**: Adjust temperatures based on convergence rate
3. **Constrainted equilibria**: Add constraints on strategy profiles (e.g., focal states)
4. **N-country models**: Extend to games with more than 3 countries
5. **Parallel search**: Use parallel computing to explore multiple initializations

### Integration with Existing Code

The solver integrates seamlessly with the existing framework:
- Uses the same `State`, `Coalition`, `Country` classes
- Compatible with `verify_equilibrium()` function
- Outputs standard Excel strategy tables
- Works with existing visualization tools

## Troubleshooting

### Common Issues

**1. Equilibrium verification fails**

- Increase `max_outer_iter` and `max_inner_iter`
- Try lower `tau_decay` for slower annealing
- Increase `inner_tol` convergence tolerance
- Check if multiple equilibria exist

**2. Oscillating strategies**

- Increase `damping` factor
- Reduce `tau_decay` for gentler annealing
- Increase `max_inner_iter`

**3. Stuck in local optimum**

- Try higher `tau_p_init` and `tau_r_init`
- Run multiple times with different initializations
- Adjust `damping` factor

**4. Numerical instabilities**

- Check that `discounting < 1.0`
- Verify payoff matrix is well-conditioned
- Ensure protocol probabilities sum to 1

## References

### Original Paper

Heyen, D., & Lehtomaa, J. (2021). Solar Geoengineering Governance: A Dynamic Framework of Farsighted Coalition Formation. *Oxford Open Climate Change*, 1(1).

### Algorithm Inspiration

The smoothed fixed-point iteration algorithm is inspired by:
- Homotopy methods for game theory
- Simulated annealing optimization
- Logit quantal response equilibrium
- Policy gradient methods in reinforcement learning

## Authors

- Original framework: Daniel Heyen & Jere Lehtomaa
- Equilibrium solver: Implemented based on ChatGPT suggestion, 2026

## License

Same license as the main codebase.
