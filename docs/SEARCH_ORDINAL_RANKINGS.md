# Search Ordinal Rankings: A Comprehensive Explanation

## Overview

`search_ordinal_rankings.py` is a wrapper script that performs an **exhaustive or sampled search over ordinal value rankings** to discover equilibria in farsighted coalition formation games. It works by:

1. **Enumerating ordinal rankings**: Generate all possible ways players can rank states by value
2. **Inducing strategies**: Automatically derive proposal and approval strategies from each ranking
3. **Computing value functions**: Solve the Markov decision process (MDP) to find long-run payoffs
4. **Verifying equilibrium**: Check if strategies are incentive-compatible with computed values
5. **Parallel search**: Use multiprocessing to search through millions of ranking combinations efficiently

The key insight is that in a **farsighted equilibrium**, each player must have a consistent preference ordering over states, and their strategies must optimize outcomes given that ordering and others' orderings.

---

## Core Concepts

### 1. Ordinal Rankings

An **ordinal ranking** is a player's preference ordering of states by value. There are two types:

#### Strict Rankings (Total Orders)
A strict ranking assigns each state to a unique position (rank 0 is best, rank N-1 is worst):

```
Player W's strict ranking: [State_WTC at rank 0, State_TC at rank 1, State_WC at rank 2, ...]
Player T's strict ranking: [State_TC at rank 0, State_WTC at rank 1, State_C at rank 2, ...]
...
```

For N states, there are **N! possible strict rankings per player**.

#### Weak Rankings (Weak Orders)
A weak ranking allows ties (indifference) between states. Multiple states can occupy the same tier:

```
Player W's weak ranking: [States {WTC, WC} in tier 0 (tied for best),
                          State TC in tier 1,
                          State W in tier 2 (worst)]
```

Weak orders are useful when value functions result in equalities (V(WTC) = V(WC)), which arises from problem symmetry or numerical precision. The number of weak orders grows as Stirling numbers of the second kind, which is dramatically smaller than N! for large N.

### 2. Induced Strategies from Rankings

Given a ranking tuple `(rank_W, rank_T, rank_C, ...)` for N players across K states:

**Proposal Strategy (Pure Case):**
- When Player P is the proposer in state S, they propose the highest-ranked state (lowest rank number) among states **approved by all committees**.
- If proposing state S' requires approval from committee C, then C will approve if S' has **lower rank** than current state S in **each committee member's ranking**.
- The proposer compares all approved transitions and picks the best one.

**Approval Strategy:**
- When Player P is asked to approve transition S→S', they approve if **rank_P(S') < rank_P(S)**.
- If indifferent (rank_P(S') = rank_P(S)), they can approve or reject; both are equilibrium-consistent.

**Transition Probabilities:**
- The proposal probabilities multiply with approval probabilities to determine full transition matrix P.
- In multiproposer setting: If each player has equal protocol weight 1/N, the full transition probability is: P[S, S'] = Σ_p (protocol_weight_p) × (proposal_prob_p[S,S']) × (approval_prob[S,S']).

### 3. Induced Strategy Example

Consider 3 states: (singletons), (TC), (WTC) and three players W, T, C:

**Ranking Tuple:** 
- W: [WTC:0, TC:1, ∅:2]  (W prefers grand coalition)
- T: [TC:0, WTC:1, ∅:2]   (T prefers excluding W)
- C: [TC:0, WTC:1, ∅:2]   (C prefers excluding W)

**Induced Proposal Strategy:**

When state is **singletons (∅)** and **W is proposer**:
- W wants to transition to WTC (its rank 0)
- For WTC to be approved by all committees, we need to check: would T and C approve WTC?
  - T's rank(WTC)=1 < rank(∅)=2 → T would approve ✓
  - C's rank(WTC)=1 < rank(∅)=2 → C would approve ✓
- So W proposes WTC (approved)
- W compares: WTC (best rank 0), TC (not approved), ∅ (current)
- Proposal: W proposes WTC with probability 1

When state is **singletons** and **T is proposer**:
- T wants TC (rank 0)
- Would W approve TC? W has rank(TC)=1 < rank(∅)=2 → Yes ✓
- Would C approve TC? C has rank(TC)=0 < rank(∅)=2 → Yes ✓
- Proposal: T proposes TC

**Approval Strategy:**
When W proposes ∅ → WTC:
- T checks: rank_T(WTC)=1 < rank_T(∅)=2? Yes → Approve
- C checks: rank_C(WTC)=1 < rank_C(∅)=2? Yes → Approve

---

## Algorithm: Ordinal Ranking Search

### High-Level Flow

```
Input: 
  - Payoff matrix (states × players)
  - Protocol (proposer probabilities for each player)
  - Effectivity (which players approve each transition)
  - Discount factor γ

Output:
  - All equilibria (ranking tuples that satisfy equilibrium conditions)

1. Generate all ordinal rankings
   - Strict: all N! permutations per player
   - Weak: all weak orders (Bell number ~B_N)
   Result: state_perms array of all rankings

2. For each ranking tuple (rank_W, rank_T, ..., rank_C):
   a) Induce strategy profile
      - Compute proposal_choice[player, state]
      - Compute approval_action[proposer, approver, current, next]
      - Compute approval_pass [proposer, current, next] (probability approved)
      - Compute transition matrix P[state, next_state]
   
   b) Solve MDP
      - Solve linear system: V = (I - γP)^(-1) × (1-γ) × u
      - u are static payoffs in each state
      - Result: V_array[state, player] = expected long-run payoff
   
   c) Verify equilibrium conditions
      - Condition 1: Proposal check
        For each proposer in each state, does they propose states that maximize their expected value?
      - Condition 2: Approval check
        For each approval committee member and each transition:
        - If V(next) > V(current): Must approve (probability = 1.0)
        - If V(next) < V(current): Must reject (probability = 0.0)
        - If V(next) = V(current): Can do either (0.0 ≤ prob ≤ 1.0)

3. Collect all ranking tuples that pass equilibrium verification
```

### State Permutation Index Encoding

The search space is too large to enumerate explicitly, so rankings are indexed:

**Strict Rankings:**
- Each ranking is assigned index 0 to N!-1
- `state_perms[perm_idx]` = rank permutation array
- `pos[perm_idx, state] = rank` = rank of state in permutation perm_idx

**Weak Orders:**
- Each weak order is assigned index 0 to B_N-1 (Bell number)
- Similar indexing: `state_perms[order_idx]` = tier assignment array

**Ranking Tuple Encoding:**
- A ranking tuple (rank_W, rank_T, ...) is represented as `(w_idx, t_idx, c_idx, ...)` 
- where each component is a perm/order index
- Total combinations = (N!)^P for strict, (B_N)^P for weak, where P = # players

### Multiprocessing Architecture

Given that combinations can reach billions (e.g., (13!)^3 ≈ 10^12 for 13-state, 3-player), search is parallelized:

**Worker Initialization** (`_init_worker_ctx`):
- Each worker process initializes shared context once
- Precomputes committee indices (which players approve each transition)
- Stores payoff matrix, protocol, discount factor

**Batch Processing** (`_search_chunk`):
- Master process generates batches of ranking tuples
- Each batch sent to worker
- Worker tests all tuples in batch independently
- Worker returns: successes, timing stats, verification results

**Streaming Output**:
- When hits are found, written to disk immediately (doesn't wait for batch to finish)
- Deduplication options: by transition matrix P, or by full strategy profile
- Solves memory issues when millions of equilibria exist

---

## Weak Orders and Weak Equality Solving

### Weak Orders: Handling Indifference

When a player is indifferent between states (V(S) = V(S')), strict rankings don't apply. **Weak orders** allow encoding indifference by placing indifferent states in the same tier.

For weak order rankings, instead of a unique best proposal:
- Proposer identifies all **approved states with best tier**
- Randomizes uniformly among tied states
- This generates mixed strategies (probabilities between 0 and 1)

**When to use weak orders:**
- Problem has symmetry (e.g., identical players) → Equal value function entries
- Reduced payoff tables (some states genuinely have equal values)
- Searching for mixed strategy equilibria

### Weak Equality Solving

In rare cases, **indifference constraints** (V(S) = V(S')) are themselves equilibrium conditions. Example:

> If a player is indifferent between two states, they can support mixed strategies with any split between them, as long as all constraints remain satisfied.

The solver includes an optional **weak equality solver** that:

1. Detects indifference constraints from weak orders
2. Sets up a **system of linear equalities** (V(S) = V(S') as equality, not inequality)
3. Solves the system to find values consistent with indifference
4. Verifies if strategies remain incentive-compatible

**When enabled:**
- `--weak-equality-solve`: Activate weak equality solving
- `--weak-equality-max-vars`: Cap number of variables in system (default: unlimited)

This is most valuable when searching weak orders yields candidates where indifference is crucial.

---

## Performance Optimizations

### 1. Large Mode

For very large state spaces (e.g., 10+ states), computing all N! permutations causes memory overflow.

**Large mode** switches to:
- On-the-fly generation of random ranking combinations
- No precomputed `state_perms` array
- Slower individual ranking testing but manageable memory
- Activates when `N! > LARGE_PERM_THRESHOLD` (set in constants.py)

### 2. Payoff-Based Ranking Ordering

Instead of testing rankings in lexicographic order, **payoff ordering** prioritizes rankings that align with players' true payoff preferences:

**Payoff Ordering Algorithm:**
1. For each player, compute baseline ranking = sort states by payoff (descending)
2. For each ranking candidate, compute distance to baseline:
   - **Kendall distance**: # of pairwise order swaps needed
   - **Footrule distance**: Σ |rank_candidate(s) - rank_baseline(s)|
3. Sort rankings by (Kendall, Footrule) lexicographically
4. Search in this order: closer-to-payoff rankings tested first

**Benefit**: Equilibria often involve rankings close to payoff orderings, so this heuristic finds solutions faster.

**Usage:**
```bash
--ranking-order payoff    # Default: lexicographic
```

### 3. Numeric Optimizations (Numba JIT)

For strict strict rankings (dominant branch), critical loops are compiled with **Numba** for speed:

- `_build_arrays_weak_nb`: Strategy induction from weak orders
- `_verify_fast_nb`: Equilibrium verification
- `_solve_V_nb`: Value function solving

Achieves 10-100x speedup on tight loops.

---

## Command-Line Usage

### Basic Syntax

```bash
python search_ordinal_rankings.py <payoff_table> [OPTIONS]
```

### Key Arguments

**Input:**
- `<payoff_table>`: Path to Excel file with state × player payoff matrix
- `--scenario`: Scenario name (e.g., "power_threshold") - loads config; if omitted, infers from payoff table

**Search Strategy:**
- `--weak-orders`: Search weak orders instead of strict permutations
- `--weak-equality-solve`: Enable weak equality solving (slow, use with caution)
- `--ranking-order payoff`: Prioritize rankings close to payoff orderings (much faster)
- `--shuffle`: Randomize order of ranking tuples (useful for partial searches)
- `--max-combinations`: Test at most N combinations (for sampling: reduces runtime)
- `--random-seed`: Seed for shuffling randomness

**Governance:**
- `--effectivity-rule`: Rule for determining approval committees
  - `unanimous_consent` (default): All committee members must approve
  - `heyen_lehtomaa_2021`: Custom rule from 2021 paper
  - `deployer_exit`, `free_exit`: Alternative rules
- Other effectivity details inferred from payoff table or scenario

**Parallelization:**
- `--workers 8`: Number of worker processes (default: 8)
- `--progress-every 1000`: Print progress every N rankings tested

**Output:**
- `--write-all-output-dir <DIR>`: Write all found equilibria to directory (with strategy tables)
- `--write-all`: Write all with auto-generated directory
- `--dedup-by transition`: Deduplicate by transition matrix (same P, different strategies)
- `--dedup-by strategy`: Deduplicate by full strategy profile
- `--stop-on-success`: Return after finding first equilibrium (default: exhaustive search)

### Examples

**Find one equilibrium fast (payoff ordering + stop-on-success):**
```bash
python search_ordinal_rankings.py payoffs.xlsx \
  --scenario power_threshold \
  --ranking-order payoff \
  --stop-on-success
```

**Exhaustive search with weak-equality solving:**
```bash
python search_ordinal_rankings.py payoffs.xlsx \
  --weak-orders \
  --weak-equality-solve \
  --weak-equality-max-vars 100 \
  --write-all-output-dir results/
```

**Partial search (1M combinations) for quick exploration:**
```bash
python search_ordinal_rankings.py payoffs.xlsx \
  --max-combinations 1000000 \
  --shuffle \
  --random-seed 42
```

---

## Output Interpretation

### Console Output

```
Ordinal Ranking Search: payoffs.xlsx
Scenario: power_threshold
Players: ['W', 'T', 'C']
Found 3 Equilibrium/ia!
Results written to: strategy_tables/all_mydata
```

### Strategy Tables (Excel Files)

When `--write-all-output-dir` is used, outputs include:

1. **strategy_table_XXXXX.xlsx**: Each unique strategy profile
   - Column: (State→NextState, Proposer) pairs
   - Row: (Player, Action) - either "Proposes" or "Approves"
   - Entry: Probability (0.0, 1.0, or mixed)

2. **summary.csv**: Metadata for all equilibria
   - Columns: ranking_tuple, perms, transition_matrix_id, success_details

### CSV Reports (for aggregation)

When running with reporting options:
- **rice_hardness_*.csv**: Ranking difficulty metrics (Kendall, footrule distances)
- Helps understand which types of rankings equilibrate

---

## Interpretation: What the Rankings Mean

Once equilibria are found, the ordinal rankings reveal strategic reasoning:

### Example: Three-Country Coalition Game

Suppose we find equilibrium with rankings:

```
W's ranking: [ WTC:0, TC:1, WC:2, WT:3, W:4, T:5, C:6 ]
T's ranking: [ TC:0, WTC:1, C:2, T:3, WC:4, WT:5, W:6 ]
C's ranking: [ TC:0, WTC:1, C:2, T:3, WC:4, WT:5, W:6 ]
```

**Interpretation:**
- **W's perspective**: Prefers grand coalition most (gets cooling at 11.5°C ideal). But accepts TC (T+C coalition without W). Would rather be in some coalition than alone.
- **T's perspective**: Prefers TC most (cooling at 4°C with like-minded C). Ranks WTC second because W's ideal is too high. Avoids being with just W.
- **C's perspective**: Mirrors T's preferences.

**Strategic Dynamics:**
Because W's ranking places TC high, it signals "I accept being outside if T-C form". This credible commitment allows T and C to exclude W, which they prefer, while W rationally doesn't force entry.

### Finding the Absorbing State

Look at the **first-ranked state** across all players' equilibrium rankings. The **intersection of first choices** often identifies the absorbing state:

- If all players rank one state first → That's the convergence point
- If players have conflicting first choices → Convergent cycle or mixing across states

---

## Potential Issues and Debugging

### No Equilibrium Found

**Causes:**
1. **Problem is genuinely hard**: Many coalition games have no ordinal ranking equilibria
2. **Search incomplete**: Use `--max-combinations` that's too small; increase or remove limit
3. **Weak equality solving needed**: If many states have equal payoffs, try `--weak-equality-solve`

**Remedies:**
- Increase `--max-combinations` or run without limit (exhaustive)
- Try `--weak-orders` if payoff table has ties
- Check payoff matrix: are values too symmetric?

### Search Too Slow

**Causes:**
1. Large state space (10+ states): Combinatorial explosion
2. High player count: (N!)^P grows extremely fast

**Remedies:**
- Use `--max-combinations` for sampling
- Enable `--ranking-order payoff`: Much faster heuristic
- Reduce `--workers` if I/O bottleneck; increase if CPU-bound
- Try `--stop-on-success` if one equilibrium suffices

### Strategy Table Looks Wrong

**Check:**
1. Proposal probabilities **sum to 1** for each (proposer, state)
2. Approval entries **between 0 and 1**
3. Transition matrix P **rows sum to 1**

If not, likely numerical error in induction. Check payoff table formatting.

---

## Mathematical Background

### Bellman Equation for Value Function

Given transition matrix P and static payoffs u:

$$ V = (I - \gamma P)^{-1} (1-\gamma) u $$

where:
- V = [V_1, ..., V_n] = expected long-run payoff for each player
- P = transition matrix, P[s, s'] = Pr(s'|s)
- u = static payoff matrix, u[s, player]
- γ = discount factor (0.99 in framework)

### Equilibrium Conditions

A ranking tuple is an equilibrium iff:

1. **Proposal IC**: For each proposer p in state s, p proposes state s* with positive probability only if s* maximizes:
   $$ E[V(s', p) | s' \text{ approved}] = \sum_{s'} 1_{s' \text{ approved}} \frac{P_p(s'|s)}{P_p(\text{approved}|s)} V(s', p) $$

2. **Approval IC**: For each approver a and proposal s→s', a approves iff:
   $$ V(s', a) \geq V(s, a) $$
   (or mixed if indifferent)

### Markov Perfect Equilibrium

This is a **stationary Markov Perfect Equilibrium** (MPE):
- Strategies depend only on current state (not history)
- Each player chooses actions maximizing discounted future payoffs given others' strategies
- Strategies are **mutually best-responses** in the infinitely-repeated game

---

## Connection to Broader Framework

This search complements `find_equilibrium.py` (smoothed fixed-point solver):

| Aspect | Ordinal Ranking | Fixed-Point Solver |
|--------|-----------------|-------------------|
| **Approach** | Exhaustive/sampled enumeration | Gradient-based optimization |
| **Guarantees** | Finds ALL equilibria (if exhaustive) | Finds LOCAL optima |
| **Speed** | Slower for small spaces; fast with heuristics | Fast even for medium spaces |
| **Interpretability** | Clear rankings explain reasoning | Opaque continuous values |
| **Robustness** | Finds edge cases, mixed strategies | Finds typical equilibria |

---

## References

See `lib/equilibrium/README.md` for algorithmic details, complexity analysis, and broader equilibrium-finding methods.



## GUESS STRATEGY (per 50,000 combos)
════════════════════════════════════════

Every guess follows this tree:

                    Newton
                       |
        _______________+_______________
       |               |               |
    Converged       Progress      Stalled
       |            (>1.0e-6)         |
       ✓             |           (nb_skip)
    SUCCESS       Seed Scipy      |
    [11 cases]        |       Cold start
                      |       Scipy anyway
                      |           |
                      |___________|
                          |
                      Scipy Root
                          |
        __________________+__________________
       |                  |                  |
    Success           Bad Progress        Bad Residual
    [0 cases]         [~24k/guess]         [~45/guess]
                      (~170k total)        (~2.1k total)