# Convergence Criterion for Equilibrium Solver

## Overview

The equilibrium solver now uses a **convergence-based stopping criterion** instead of a fixed number of outer iterations. This ensures the solver runs long enough to converge while avoiding wasted computation.

## Stopping Criteria

The solver can stop in three ways (checked in order):

### 1. Early Termination (Equilibrium Verified)

**Preferred stopping mode**: If strategies have stabilized and projection yields a verified equilibrium:

1. **Strategy stability**: `max_change < outer_tol` for `consecutive_tol` consecutive outer iterations
2. **Projection**: Temporarily project strategies to exact equilibrium
3. **Verification**: Check if projected equilibrium satisfies all equilibrium conditions
4. **If verification passes**: Stop immediately (equilibrium found!)
5. **If verification fails**: Restore strategies and continue annealing

This decouples "annealing completeness" from "equilibrium correctness" - if we've found a valid equilibrium, we stop, regardless of temperature.

### 2. Regular Convergence (Annealing Complete)

If early termination hasn't occurred, stop when **both** conditions are met:

1. **Temperature convergence**: Both `tau_p` and `tau_r` are near their minimum value:
   ```
   tau_p <= tau_min * (1 + tau_margin)
   tau_r <= tau_min * (1 + tau_margin)
   ```

2. **Strategy stability**: The strategy has been stable for `consecutive_tol` consecutive outer iterations:
   ```
   max_change < outer_tol  (for last k iterations)
   ```
   where `k = consecutive_tol`

### 3. Safety Valve (Max Iterations)

If neither convergence criterion is met:
- Stop when `outer_iter >= max_outer_iter`
- This indicates the solver didn't converge properly
- Review parameters or problem formulation

## Parameters

### New Parameters

- **`outer_tol`** (float, optional): Convergence tolerance for outer loop
  - Default: `10 * inner_tol`
  - Typical value: `1e-9` (when `inner_tol = 1e-10`)
  - A larger value means less strict convergence requirements

- **`consecutive_tol`** (int): Number of consecutive converged outer iterations required
  - Default: `2`
  - Typical range: 2-5
  - Higher values = more conservative (slower but more robust)

- **`tau_margin`** (float): Margin for checking if tau is near tau_min
  - Default: `0.01` (i.e., 1%)
  - The solver considers tau "at minimum" when `tau <= tau_min * (1 + tau_margin)`

### Modified Parameters

- **`max_outer_iter`**: Now serves as a **safety valve** rather than the primary stopping criterion
  - Default: `1000` (previously `5`)
  - If this is reached, the solver will warn that convergence wasn't achieved
  - Set this large enough that you rarely hit it in practice

## How It Works

### Algorithm Flow

1. **Initialization**: Set `tau_p = tau_p_init`, `tau_r = tau_r_init`

2. **Outer loop** (annealing):
   - Run inner fixed-point iteration until `max_change < inner_tol`
   - Record the final `max_change` from the inner loop

   - **Early termination check** (if strategies are stable):
     - Have the last k outer iterations all converged? (stability check)
     - If yes: temporarily project to exact equilibrium and verify
     - If verification passes: **STOP** (equilibrium found!)
     - If verification fails: restore strategies and continue

   - **Regular convergence check**:
     - Are both tau values near tau_min? (temperature check)
     - Have the last k outer iterations all converged? (stability check)
     - If both conditions met: **STOP** (converged)

   - Otherwise: decay temperatures and continue
   - If `outer_iter >= max_outer_iter`: **STOP** (safety valve)

3. **Projection**: Project to exact equilibrium (if `project_to_exact = True` and not already projected)

### Why This Works

The algorithm uses **simulated annealing** to smooth the equilibrium conditions:
- High temperatures (`tau_p`, `tau_r`) make the softmax/sigmoid functions smooth
- As temperatures decrease, the functions become sharper (closer to argmax/step function)
- Eventually, at `tau_min`, the smoothed problem is very close to the exact equilibrium

**Early termination is safe** because:
- **Projection is diagnostic**: It's a mathematical mapping, not part of the iteration
- **Verification is ground truth**: If the projected profile passes verification, it's a valid Markov perfect equilibrium
- **Annealing is a means, not the goal**: The purpose is to find an equilibrium, not to complete the annealing schedule
- **No commitment without verification**: If projection fails verification, we restore and continue

The convergence criteria ensure:
- **Early termination**: Stop as soon as we have a verified equilibrium (efficient!)
- **Strategy stability**: The strategy has stopped changing (within tolerance) for k consecutive iterations
- **Consecutive checks**: We're not stopping on a transient fluctuation
- **Temperature convergence** (regular mode): We've fully annealed to minimum temperature

## Example Usage

### Default Parameters (Recommended)

```python
from lib.equilibrium.find import find_equilibrium

result = find_equilibrium(
    config,
    output_file='output.xlsx',
    solver_params={
        'tau_p_init': 1e-6,
        'tau_r_init': 1e-6,
        'tau_decay': 0.95,
        'tau_min': 1e-7,
        'max_outer_iter': 1000,  # Safety valve
        'max_inner_iter': 2,
        'damping': 1.0,
        'inner_tol': 1e-10,
        'outer_tol': None,  # Defaults to 10*inner_tol = 1e-9
        'consecutive_tol': 2,
        'tau_margin': 0.01,
    }
)
```

### Custom Parameters for Faster (Less Strict) Convergence

```python
solver_params = {
    'outer_tol': 1e-8,  # Looser than default (1e-9)
    'consecutive_tol': 2,  # Only 2 consecutive iterations
}
```

### Custom Parameters for Slower (More Robust) Convergence

```python
solver_params = {
    'outer_tol': 1e-10,  # Stricter than default
    'consecutive_tol': 3,  # Require 3 consecutive iterations
}
```

## Interpreting Results

### Early Termination (Best Case)

```
  Strategies stable for 2 iterations, attempting early verification...
  ✓ Equilibrium verification PASSED

  Early termination (equilibrium found):
    - Strategies stable for 2 iterations
    - Projected equilibrium verified
    - tau_p=3.5849e-07, tau_r=3.5849e-07 (not yet at tau_min=1.0000e-07)

Stopped early after 20 outer iterations (equilibrium verified)

Skipping final projection (already projected and verified)
```

This indicates:
- **Optimal outcome**: Found a verified equilibrium early
- Stopped before completing full annealing (saved ~25 iterations in this example)
- The equilibrium is guaranteed correct (passed verification)
- No need for final projection (already done)

### Regular Convergence

```
  Convergence criterion met (annealing complete):
    - tau_p=1.0000e-07 <= 1.0100e-07
    - tau_r=1.0000e-07 <= 1.0100e-07
    - Last 2 max_changes < 1.00e-09

Annealing converged after 46 outer iterations

Projecting to exact equilibrium...
```

This indicates:
- The solver converged after completing the annealing schedule
- It took 46 outer iterations to reach convergence
- Both temperature and stability conditions were met
- Final projection will be applied

### Hit Safety Valve (Needs Attention)

```
Annealing stopped at max_outer_iter=1000 (safety valve)
  Final max_change: 2.345e-08 (outer_tol: 1.00e-09)
```

This indicates:
- The solver didn't converge within `max_outer_iter` iterations
- The final `max_change` is shown (compare to `outer_tol`)
- Consider:
  - Increasing `outer_tol` if `max_change` is close
  - Increasing `max_outer_iter` if you want to wait longer
  - Checking if there's a fundamental issue with the problem

## Result Dictionary

The solver returns additional information in the result dictionary:

```python
strategy_df, result = solver.solve(...)

result = {
    'converged': True,  # Whether convergence criterion was met
    'stopping_reason': 'early_verification',  # 'early_verification', 'converged', or 'max_iter'
    'outer_iterations': 20,  # Number of outer iterations
    'final_tau_p': 3.58e-7,  # Final tau_p value
    'final_tau_r': 3.58e-7,  # Final tau_r value
    'final_max_change': 0.0,  # Final max_change from inner loop
    'outer_tol': 1e-9,  # Outer tolerance used
    'recent_max_changes': [0.0, 0.0],  # Last k max_changes
}
```

**Stopping reasons:**
- `'early_verification'`: Found and verified equilibrium before annealing completed (best case)
- `'converged'`: Regular convergence (temperature + stability conditions met)
- `'max_iter'`: Hit safety valve (may need parameter adjustment)

## Choosing Parameters

### Rule of Thumb

**Don't choose `max_outer_iter` by a fixed number; choose it by a stopping criterion.**

A suitable `max_outer_iter` is "large enough that you rarely hit it before stopping naturally."

For most problems:
- `consecutive_tol = 2` is sufficient (checks stability over 2 iterations)
- `outer_tol = 10 * inner_tol` provides appropriate strictness
- `tau_margin = 0.01` allows 1% tolerance on temperature convergence
- `max_outer_iter = 1000` is a safe default (convergence usually happens in 30-100 iterations)

### Problem-Specific Tuning

If you find that:
- **Convergence is too slow**: Increase `outer_tol` or decrease `consecutive_tol`
- **Results are unstable**: Increase `consecutive_tol` or decrease `outer_tol`
- **Hitting safety valve**: Increase `max_outer_iter` or relax `outer_tol`
- **Stopping too early**: Decrease `outer_tol` or increase `consecutive_tol`

## Mathematical Details

### Temperature Decay Schedule

The temperatures decay geometrically:
```
tau_{t+1} = max(tau_t * tau_decay, tau_min)
```

The number of iterations to reach `tau_min` from `tau_init` is approximately:
```
n ≈ log(tau_min / tau_init) / log(tau_decay)
```

For the default parameters:
```
n ≈ log(1e-7 / 1e-6) / log(0.95)
  ≈ log(0.1) / log(0.95)
  ≈ 45 iterations
```

This explains why convergence typically happens around iteration 45-50.

### Convergence Metrics

- **Inner convergence**: `max_change = max(|new_p - old_p|, |new_r - old_r|)`
  - Measures maximum change in any strategy component
  - Stops inner loop when `max_change < inner_tol`

- **Outer convergence**: Last `k` values of `max_change < outer_tol`
  - Ensures strategy has stabilized over multiple outer iterations
  - Prevents stopping on transient fluctuations

## Benefits of Early Termination

Early termination via equilibrium verification provides several advantages:

1. **Computational Efficiency**:
   - Stop as soon as a valid equilibrium is found
   - Can save 30-50% of iterations (e.g., stop at iteration 20 instead of 46)
   - Particularly beneficial for large-scale problems

2. **Correctness Guarantee**:
   - Equilibrium is verified before stopping
   - No need to trust that "low temperature = good equilibrium"
   - Verification is the ground truth, not annealing schedule

3. **Decoupled Concerns**:
   - Annealing schedule is a heuristic to find equilibrium
   - Equilibrium correctness is determined by mathematical verification
   - These are logically independent - early termination respects this

4. **Robustness**:
   - Less sensitive to annealing parameter choices
   - If equilibrium is found early, we stop regardless of `tau_min` setting
   - Fails gracefully: if verification fails, we continue annealing

5. **Practical Implications**:
   - Faster iteration during research
   - Can use more conservative (slower) annealing schedules without time penalty
   - System automatically finds the "right" stopping point

## Backward Compatibility

The new features are **optional** and have sensible defaults:
- Existing code will work without modification
- Default behavior: convergence-based stopping with early termination and `max_outer_iter = 1000` safety valve
- To reproduce old behavior: set `max_outer_iter` to the old value and `outer_tol = 0` (will always hit safety valve)
