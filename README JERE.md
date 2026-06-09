# Farsighted Coalition Formation Solver

A Python implementation of the **Stationary Markov Perfect Equilibrium (SMPE)** solver for N-player farsighted coalition formation games in the spirit of [Ray & Vohra (2015)](https://doi.org/10.3982/ECTA11016). Players form and dissolve coalitions over time, discount future payoffs, and reason about long-run consequences when evaluating proposals — not just immediate gains.

---

## Table of Contents

1. [Economic Model](#1-economic-model)
2. [Equilibrium Concept](#2-equilibrium-concept)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Per-State MIP](#4-per-state-mip)
5. [Value Function Iteration](#5-value-function-iteration)
6. [Cycle Detection and Mixed-Strategy Resolution](#6-cycle-detection-and-mixed-strategy-resolution)
7. [Equilibrium Verification](#7-equilibrium-verification)
8. [Multiple Equilibria Search](#8-multiple-equilibria-search)
9. [Configuration Reference](#9-configuration-reference)
10. [Approximations and Limitations](#10-approximations-and-limitations)
11. [Dependencies](#11-dependencies)
12. [Quick-Start Examples](#12-quick-start-examples)

---

## 1. Economic Model

### Players and States

There are $N$ players indexed $i \in \mathcal{I} = \{0, 1, \ldots, N-1\}$. The **state** of the world at any point in time is a **partition** of $\mathcal{I}$ into coalitions — a collection of non-overlapping, exhaustive groups. For example, with three players $\{A, B, C\}$, the possible states include the grand coalition $\{A,B,C\}$, any pair-plus-singleton structure such as $\{A,B\}+\{C\}$, and full dissolution $\{A\}+\{B\}+\{C\}$.

The total number of states equals the **Bell number** $B_N$ (the number of set partitions of an $N$-element set):

| $N$ | $B_N$ |
|-----|-------|
| 2   | 2     |
| 3   | 5     |
| 4   | 15    |
| 5   | 52    |
| 6   | 203   |

The code enumerates all $B_N$ states automatically, or you can supply a restricted custom state list via `STATES_OVERRIDE` to model games where certain coalition structures are infeasible.

### Payoffs

Each player $i$ receives a **flow payoff** $\pi(s, i)$ in each period they occupy state $s$. These are collected into the matrix `PAYOFFS` of shape $(|\mathcal{S}|, N)$. Payoffs can reflect anything the modeller wants — value created by coalition size, asymmetric bargaining power, externalities imposed on non-members, etc.

### Timing Within a Period

Each period proceeds as follows:

1. The current state $s \in \mathcal{S}$ is observed by all players.
2. Nature selects a **proposer** $i$ with probability $\rho_i$ (default: uniform, $\rho_i = 1/N$).
3. Player $i$ proposes a next state $s' \in \mathcal{S}$.
4. The **committee** of required voters casts ballots simultaneously.
5. If all required voters accept, the state transitions to $s'$; otherwise it remains at $s$.
6. All players collect flow payoff $\pi(s_{\text{realized}}, \cdot)$ for this period.

This within-period structure repeats each period forever.

### Committee Rule

The set of players whose consent is required for a proposal $s \to s'$ by proposer $i$ is:

$$\text{voters}(s, s', i) = \text{changed}(s, s') \setminus \{i\}$$

where $\text{changed}(s, s') = \{j \in \mathcal{I} : \text{block}(j, s) \neq \text{block}(j, s')\}$ is the set of players whose coalition membership changes. Only players who are directly affected need to agree; uninvolved players are not consulted.

**Unilateral exit** is an important special case: if player $i$ proposes a state that results from $i$ simply leaving their current coalition alone (with all remaining members staying together), then $\text{voters} = \emptyset$ — no consent is required. This represents a player's inalienable right to exit.

---

## 2. Equilibrium Concept

### Stationary Markov Perfect Equilibrium (SMPE)

The code computes a **Stationary Markov Perfect Equilibrium** — a profile of strategies that depend only on the current state (Markov), are stationary (time-invariant), and form a Nash equilibrium at every state (perfect).

Each player's **strategy** has two components:

- **Proposal strategy** $\sigma_i(s, s') \in [0,1]$: the probability that player $i$, when selected as proposer at state $s$, proposes moving to state $s'$. Must satisfy $\sum_{s'} \sigma_i(s, s') = 1$.

- **Acceptance strategy** $\alpha_j(s, s') \in [0,1]$: the probability that voter $j$ accepts a proposal to move from $s$ to $s'$, regardless of who proposed it (acceptance depends only on the transition, not on the identity of the proposer).

### Continuation Values

Given a strategy profile $(\sigma, \alpha)$, the **transition probability** from state $s$ to $s'$ is:

$$T(s, s') = \sum_{i \in \mathcal{I}} \rho_i \cdot \sigma_i(s, s') \cdot q_i(s, s')$$

where the **acceptance probability** is the product of all required voters' acceptance strategies:

$$q_i(s, s') = \prod_{j \in \text{voters}(s, s', i)} \alpha_j(s, s')$$

The diagonal entry $T(s, s)$ absorbs all probability mass not directed elsewhere (status quo, blocked proposals, etc.).

Given the Markov chain $T$, each player $i$'s **continuation value** at each state satisfies the Bellman equation:

$$V_i(s) = (1 - \delta)\,\pi(s, i) + \delta \sum_{s'} T(s, s')\, V_i(s')$$

In matrix form (one system per player $i$):

$$\mathbf{V}_i = (1 - \delta)\,\boldsymbol{\pi}_i + \delta\, T\, \mathbf{V}_i$$

$$\Rightarrow \quad \mathbf{V}_i = (I - \delta T)^{-1}(1 - \delta)\,\boldsymbol{\pi}_i$$

This linear system is solved exactly using `scipy.linalg.solve` at each VFI iteration.

### Equilibrium Conditions

A strategy profile is an SMPE if and only if, **at every state $s$**:

**Acceptance (cutoff rule):** Voter $j$ accepts a transition to $s'$ if and only if $V_j(s') \geq V_j(s)$, i.e., accepting weakly raises their continuation value. Formally:

$$\alpha_j(s, s') = \begin{cases} 1 & \text{if } V_j(s') > V_j(s) \\ 0 & \text{if } V_j(s') < V_j(s) \\ \alpha_j^* \in [0,1] & \text{if } V_j(s') = V_j(s) \end{cases}$$

**Proposal (best-response):** Proposer $i$ places positive weight only on proposals that maximise their expected continuation value. The expected value of proposing $s'$ is:

$$\mathbb{E}_i[s \to s'] = q_i(s, s') \cdot V_i(s') + (1 - q_i(s, s')) \cdot V_i(s)$$

(With probability $q_i$ the proposal is accepted and the state moves; with probability $1 - q_i$ it is rejected and stays.) Player $i$ mixes only between proposals achieving the maximum of this over all $s'$.

---

## 3. Algorithm Overview

The solver proceeds in three nested layers:

```
┌─────────────────────────────────────────────────────────┐
│  Outer loop: Value Function Iteration (VFI)             │
│    Initialize V = PAYOFFS (myopic values)               │
│    Repeat:                                              │
│      ┌─────────────────────────────────────────────┐    │
│      │  Middle: Per-State MIP (one per state s)    │    │
│      │    Given V, find σ, α, q satisfying         │    │
│      │    equilibrium conditions at s              │    │
│      └─────────────────────────────────────────────┘    │
│      Build T from (σ, q, ρ)                             │
│      Update V = (I − δT)⁻¹(1−δ)π                       │
│    Until ‖ΔV‖ < tol  OR  cycle detected                │
│      ┌─────────────────────────────────────────────┐    │
│      │  Cycle resolver: V-interpolation bisect     │    │
│      │    Find V* where oscillating gain = 0       │    │
│      │    Run MIP at V* to recover mixed strategy  │    │
│      └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Per-State MIP

At each VFI iteration, for each state $s$, the code solves a **Mixed Integer Program** to find the strategy profile that is a Nash equilibrium at $s$ given the current continuation values $V$.

### Variables

| Symbol | Type | Range | Meaning |
|--------|------|-------|---------|
| $\sigma_{i,s'}$ | continuous | $[0,1]$ | Prob. proposer $i$ proposes $s'$ |
| $z_{i,s'}$ | binary | $\{0,1\}$ | Best-response indicator for $(i, s')$ |
| $\alpha_{j,s'}$ | continuous | $[0,1]$ | Acceptance prob. of voter $j$ for $s'$ (free voters only) |
| $q_{i,s'}$ | continuous | $[0,1]$ | Product of acceptances for proposal $(i, s')$ |
| $w_{i,s',m}$ | continuous | $[0,1]$ | McCormick auxiliaries for sequential products |

### Pre-solving Acceptance Decisions

Before constructing the MIP, each voter's acceptance is resolved analytically using the cutoff rule. Let $\varepsilon = 10^{-8}$ (the indifference threshold):

$$\alpha_j(s, s') = \begin{cases} 1 & \text{if } V_{s',j} - V_{s,j} > \varepsilon \\ 0 & \text{if } V_{s',j} - V_{s,j} < -\varepsilon \\ \text{free variable} & \text{if } |V_{s',j} - V_{s,j}| \leq \varepsilon \end{cases}$$

This eliminates most $\alpha$ variables before the MIP is built, dramatically reducing its size.

### Constraints

**C1 — Proposals are a probability distribution:**
$$\sum_{s'} \sigma_{i,s'} = 1 \qquad \forall i$$

**C2 — Proposal weight requires best-response indicator:**
$$\sigma_{i,s'} \leq z_{i,s'} \qquad \forall i, s'$$

**C3 — Exactly one best-response per proposer:**
$$\sum_{s'} z_{i,s'} = 1 \qquad \forall i$$

**C4 — Best-response (big-M formulation, gain form):**

For each proposer $i$ and any two proposals $s', s''$, if $z_{i,s'} = 1$ (player $i$ puts weight on $s'$), then the expected gain from $s'$ must be at least as large as the expected gain from $s''$:

$$q_{i,s''} \cdot g_{i,s''} - q_{i,s'} \cdot g_{i,s'} \leq M(1 - z_{i,s'}) \qquad \forall i, s' \neq s''$$

where $g_{i,s'} = V_{s',i} - V_{s,i}$ is the gain from transitioning to $s'$ and $M = 2 \max_{s,i} |V_{s,i}|$ is a data-driven big-M constant. Writing the objective in gain form (relative to the status quo) rather than in raw value form is essential for correctness when payoffs can be negative — a blocked proposal ($q = 0$) should produce zero gain, not a spuriously large or small value.

**C5 — Linearisation of the acceptance product:**

The acceptance probability $q_{i,s'} = \prod_{j \in \text{voters}(s,s',i)} \alpha_{j,s'}$ is a product of continuous variables — nonlinear and not directly expressible as a linear constraint. For voters who are fixed (strictly accept or reject), the product collapses to a constant or single variable. For voters who are free (indifferent), a **sequential McCormick chain** is used.

For $k$ free voters $j_1, j_2, \ldots, j_k$, introduce auxiliary variables $w_0, w_1, \ldots, w_{k-3}$ and define the chain:

$$w_0 = \alpha_{j_1} \cdot \alpha_{j_2}, \quad w_1 = w_0 \cdot \alpha_{j_3}, \quad \ldots \quad q = w_{k-3} \cdot \alpha_{j_k}$$

Each product $z = a \cdot b$ with $a, b \in [0,1]$ is linearised via the four **McCormick inequalities**:

$$z \leq a, \quad z \leq b, \quad z \geq a + b - 1, \quad z \geq 0$$

These inequalities define the tightest linear relaxation of the bilinear constraint $z = ab$ over the unit square, and together with the integer constraints on $z_{i,s'}$ they recover the exact product at an optimal solution.

**MIP objective:** Zero (feasibility MIP — we seek any equilibrium-consistent strategy, not an optimum).

The MIP is solved using `scipy.milp` (which calls HiGHS internally). Since the objective is zero, the solver finds any feasible point satisfying all constraints — which corresponds to a valid SMPE strategy at state $s$ given the current $V$.

---

## 5. Value Function Iteration

VFI exploits the **contraction mapping** property of the Bellman operator. Starting from an initial $V^{(0)}$ (default: the myopic payoff matrix `PAYOFFS`), each iteration applies:

$$V^{(t+1)} = (I - \delta T^{(t)})^{-1}(1-\delta)\,\Pi$$

where $T^{(t)}$ is the transition matrix derived from the MIP strategies at $V^{(t)}$.

If strategies were fixed, the operator would be a strict contraction with modulus $\delta < 1$, guaranteeing convergence at rate $O(\delta^t)$. In practice, strategies are piecewise constant in $V$ (they jump at indifference thresholds), so the effective operator is **piecewise affine** rather than smooth. Within each piece (i.e., away from thresholds) the contraction still holds, and VFI converges rapidly. Convergence is declared when:

$$\max_{s, i} |V^{(t+1)}_{s,i} - V^{(t)}_{s,i}| < \texttt{tol} \quad (\text{default } 10^{-6})$$

---

## 6. Cycle Detection and Mixed-Strategy Resolution

### Why Cycles Occur

When the true equilibrium requires a player to **mix** (place positive probability on multiple proposals or acceptance decisions), VFI can enter a limit cycle. The mechanism is:

1. At $V^{(t)}$, player $i$'s gain $g_i(s \to s') = V^{(t)}_{s',i} - V^{(t)}_{s,i}$ is slightly positive → the MIP assigns $\sigma_i(s, s') = 1$ (pure exit).
2. This changes $T$, which changes $V^{(t+1)}$.
3. At $V^{(t+1)}$, the same gain is slightly negative → the MIP assigns $\sigma_i(s, s') = 0$ (pure stay).
4. This restores a $T$ close to the original, so $V^{(t+2)} \approx V^{(t)}$.

The true equilibrium lies exactly at the gain-zero threshold, where the player is indifferent and must mix. VFI never settles there because it always snaps to a pure strategy.

### Detection

Every $V$ iterate is stored. A cycle of period $k$ is detected when:

$$\max_{s,i} |V^{(t)}_{s,i} - V^{(t-k)}_{s,i}| < 500 \cdot \texttt{tol}$$

for some $k \in \{2, 3, \ldots, \texttt{cycle\_window}\}$ (default window: 8).

### Resolution via V-Interpolation Bisection

Once a cycle of period $k$ is detected, the cycle window contains $k$ distinct $V$ matrices $\{V_0, V_1, \ldots, V_{k-1}\}$. The resolver:

**Step 1 — Identify the oscillating gain.** For every pair of consecutive $V$ matrices and every $(i, s, s')$ triple, compute whether the gain $g_i(s \to s') = V_{s',i} - V_{s,i}$ changes sign:

$$g_i^{(a)}(s \to s') \cdot g_i^{(b)}(s \to s') < 0$$

Sign changes can arise from either proposal oscillation (player $i$'s $\sigma$ flips) or acceptance oscillation (player $i$'s $\alpha$ as a voter flips). Both are detected. The sign change with the widest bracket $|g^{(a)} - g^{(b)}|$ is selected as the dominant oscillation, since it is most numerically stable to bisect.

**Step 2 — Find the widest-bracket V pair.** Among all pairs $(V_a, V_b)$ in the cycle window that bracket the dominant gain (i.e., $g^{(a)} \cdot g^{(b)} < 0$), the pair with the largest $|g^{(a)} - g^{(b)}|$ is chosen.

**Step 3 — V-interpolation bisect.** Because $V(t) = (1-t) V_a + t V_b$ is linear in $t$, the gain function

$$\text{gain}(t) = V(t)_{s',i} - V(t)_{s,i}$$

is also linear in $t$ — so it crosses zero exactly once. The crossing point $t^*$ is found via Brent's method (`scipy.optimize.brentq`) to tolerance $10^{-10}$:

$$t^* = \text{brentq}(\text{gain}, 0, 1) \quad \Rightarrow \quad V^* = (1-t^*)\,V_a + t^*\,V_b$$

At $V^*$, the dominant gain is exactly zero, placing the oscillating player precisely at their indifference threshold.

**Step 4 — Full MIP at $V^*$.** The per-state MIP is solved at $V^*$ for all states simultaneously. Because the gain is zero at the indifference threshold, the oscillating player's $\alpha$ or $\sigma$ is classified as a **free variable** by the pre-solve step. The MIP then finds the mixing probability that is consistent with the equilibrium conditions — typically the unique value that makes all constraints feasible.

**Why not T-interpolation?** An earlier implementation bisected on the transition matrix $T(p) = (1-p) T_a + p T_b$ rather than on $V$ directly. This fails in a subtle case: when the oscillating player is a *voter* (not a proposer), their sigma is zero in both iterates, so the T-contribution of the transition they control is zero in both — $T(p)$ is constant in $p$ and no bisection is possible. The V-interpolation approach is immune to this because gains are computed directly from $V$ regardless of who controls the transition.

### Secondary Near-Indifferences

After cycle resolution, the recovered strategies are passed through the equilibrium verifier. If secondary violations remain (other players who are also near-indifferent under $V^*$), VFI is restarted from $V^*$ rather than from the original initialisation. The cycle-detection and resolution logic applies again if needed. In practice, at most one or two restart rounds are required.

---

## 7. Equilibrium Verification

After convergence, the code runs two independent checks:

**Acceptance verification** (`verify_responses`): For every voter $j$, state $s$, and transition $s'$: if $V_{s',j} - V_{s,j} > \varepsilon$ then $\alpha_j(s,s')$ must equal 1; if $< -\varepsilon$ then it must equal 0. Violations are printed with the magnitude of the discrepancy.

**Proposal verification** (`verify_proposals`): For every proposer $i$ at every state $s$: compute the expected payoff of every proposed transition $s'$,

$$\mathbb{E}_i[s \to s'] = q_i(s,s') \cdot V_{s',i} + (1 - q_i(s,s')) \cdot V_{s,i}$$

and check that all transitions receiving positive probability $\sigma_i(s,s') > 0$ achieve the maximum expected payoff. Violations are printed with the actual vs. best expected value.

Both checks use the tolerance $\varepsilon = 10^{-8}$. A valid equilibrium passes both.

---

## 8. Multiple Equilibria Search

Coalition formation games typically admit multiple SMPE. The `find_equilibria()` function implements a **multi-start heuristic** to discover as many as possible.

### Sources of Variation

**1. Random $V^{(0)}$ initialisation.** VFI is a fixed-point iteration whose basin of attraction depends on the starting point. Different starting values can converge to different fixed points. Starting values are drawn as:

$$V^{(0)} = \texttt{PAYOFFS} + \varepsilon, \quad \varepsilon_{s,i} \sim \mathcal{N}(0,\, \sigma_\pi^2)$$

where $\sigma_\pi = \max_{s,i} |\pi(s,i)|$ scales the noise to the payoff magnitude.

**2. Proposer probability variation.** The vector $\rho$ governs effective bargaining power. Changing $\rho$ can select different equilibria because it changes which player's proposal drives transitions. By default the code tries:
- Uniform: $\rho_i = 1/N$ for all $i$
- Player-$i$-favoured: $\rho_i = 0.5$, $\rho_j = 0.5/(N-1)$ for $j \neq i$ — one for each player

**Deduplication.** Two runs are considered to have found the same equilibrium if:

$$\max_{s,i} |V^{(1)}_{s,i} - V^{(2)}_{s,i}| \leq \texttt{atol} \quad (\text{default } 0.01)$$

Deduplication is on $V$ rather than on strategies, because two strategy profiles that differ only off the equilibrium path (on transitions that are never proposed) are payoff-equivalent and represent the same equilibrium for all practical purposes.

### Completeness Caveat

Multi-start is a **heuristic**, not an exhaustive algorithm. There is no guarantee that all equilibria are found. The set of all SMPE is generally unknown for $N \geq 3$, and characterising it analytically is an open research problem. The practical recommendation is to increase `n_restarts` and diversify `rho_variants` if completeness is a concern, while acknowledging that some equilibria in narrow basins of attraction may still be missed.

---

## 9. Configuration Reference

All user-facing settings live at the top of `coalition_mip_n.py`:

```python
PLAYERS = [0, 1, 2]              # Player indices
PNAMES  = {0: "A", 1: "B", 2: "C"}  # Display names

PAYOFFS_OVERRIDE = np.array([...])   # (N_STATES × N_PLAYERS) payoff matrix
                                     # Set to None to use default proportional payoffs

DELTA = 0.95                     # Common discount factor δ ∈ (0,1)

STATES_OVERRIDE = None           # Custom state list (see below) or None for all partitions

SEARCH_MODE = False              # False: single solve; True: multi-start search
```

### Custom State Spaces (`STATES_OVERRIDE`)

To restrict the state space, set `STATES_OVERRIDE` to a list of partitions. Each partition is a tuple of `frozenset`s. The code validates that every state covers each player exactly once.

```python
STATES_OVERRIDE = [
    (frozenset({0,1,2}),),                              # grand coalition
    (frozenset({0,2}), frozenset({1})),                 # {A,C} + {B}
    (frozenset({0}), frozenset({1}), frozenset({2})),   # singletons
]
```

If a player's natural unilateral exit would land them in an excluded state, that exit is treated as a regular proposal requiring consent (rather than being free).

**Important:** when using `STATES_OVERRIDE`, your `PAYOFFS_OVERRIDE` must have exactly `len(STATES_OVERRIDE)` rows, in the same order as your state list.

### `find_equilibria()` Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `delta` | — | Discount factor (required) |
| `n_restarts` | 40 | Random $V^{(0)}$ draws per $\rho$ variant |
| `rho_variants` | None | List of proposer probability vectors; if None, uses uniform + one per player |
| `tol` | $10^{-6}$ | VFI convergence tolerance |
| `max_iter` | 200 | Maximum VFI iterations per run |
| `seed` | 0 | NumPy random seed for reproducibility |
| `verbose_each` | False | Print VFI progress for every run |

---

## 10. Approximations and Limitations

### What is exact

- The equilibrium conditions (cutoff acceptance, best-response proposals) are enforced exactly as hard constraints in the MIP.
- The value function solve $(I - \delta T)^{-1}(1-\delta)\pi$ is a direct linear solve, not an approximation.
- The McCormick relaxation of the acceptance product is tight at integer solutions — when $z_{i,s'}$ are pinned to $\{0,1\}$ by the integrality constraint, the linearised $q$ recovers the true product exactly.

### Approximations and modelling choices

**Stationary strategies only.** The code restricts to Markov strategies that depend only on the current state — not on history. History-dependent ("non-stationary") equilibria, which may exist and may achieve different payoffs, are not computed.

**Proposer selection is exogenous and random.** The model assumes Nature selects a proposer each period with fixed probabilities $\rho$. In reality, the protocol for who gets to propose may be endogenous, strategic, or determined by an auction. The `rho_variants` feature lets you probe sensitivity to this assumption.

**Simultaneous voting.** All required voters cast ballots simultaneously without observing each other's votes. Sequential or open-vote protocols would change the equilibrium.

**Indifference threshold $\varepsilon = 10^{-8}$.** A voter is treated as strictly accepting or rejecting only if the gain exceeds $10^{-8}$ in absolute value. Values inside this band are treated as indifference and enter the MIP as free variables. If two states have continuation values very close but not equal, there is a small risk of misclassification. In practice this threshold is well below typical payoff differences.

**Big-M tightness.** The constraint C4 uses $M = 2 \max_{s,i} |V_{s,i}|$. This is data-driven and much tighter than a fixed large constant, but it still introduces numerical slack in the LP relaxation. If payoffs are very heterogeneous across states, some LP relaxation variables may not snap cleanly to integer values. The integrality constraint on $z$ handles this, but it means the LP bound may be weak and the MIP solver may take longer.

**No guarantee of equilibrium uniqueness or completeness.** The MIP finds *an* equilibrium-consistent strategy at each state given $V$, not all of them. Multiple equilibria may exist at the same $V$, and the MIP's internal tie-breaking (based on solver internals of HiGHS) determines which one is returned. The multi-start search in `find_equilibria()` probes for different equilibria but is not exhaustive.

**Scalability.** The number of states grows as the Bell number $B_N$, which grows super-exponentially. The MIP at each state has $O(N \cdot B_N)$ variables and constraints, and is solved $B_N$ times per VFI iteration. For $N = 5$ ($B_5 = 52$ states) this is tractable; for $N = 6$ ($B_6 = 203$) it becomes slow; for $N \geq 7$ the full state space is likely infeasible without custom pruning via `STATES_OVERRIDE`.

---

## 11. Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, linear algebra |
| `scipy.linalg.solve` | Exact solve of $(I - \delta T)^{-1}(1-\delta)\pi$ |
| `scipy.optimize.milp` | Per-state MIP solver (calls HiGHS internally) |
| `scipy.optimize.brentq` | Bracketing bisection for cycle resolution |
| `scipy.optimize.LinearConstraint`, `Bounds` | MIP constraint specification |

All are part of the standard SciPy/NumPy scientific Python stack. No external solvers or licences are required.

---

## 12. Quick-Start Examples

### Single solve (default)

```python
# In coalition_mip_n.py, set:
PLAYERS = [0, 1, 2]
PNAMES  = {0: "W", 1: "T", 2: "C"}
PAYOFFS_OVERRIDE = np.array([
    [ 98.222,  13.222, -15.111],
    [107.25,    9.75,  -22.75 ],
    [ 55.688,  14.438,   0.688],
    [118.188,   1.938, -36.812],
    [  0.,      0.,      0.   ],
])
DELTA = 0.95
SEARCH_MODE = False

# Then:
# python coalition_mip_n.py
```

Output includes equilibrium values, strategies, transition matrix, and stationary distribution, followed by verification.

### Multi-start equilibrium search

```python
SEARCH_MODE = True
DELTA = 0.80
# Optionally increase n_restarts in find_equilibria() call for broader coverage
```

### Restricted state space

```python
# Four-player game but only grand coalition and full dissolution allowed:
PLAYERS = [0, 1, 2, 3]
PNAMES  = {0:"A", 1:"B", 2:"C", 3:"D"}
STATES_OVERRIDE = [
    (frozenset({0,1,2,3}),),                                      # grand
    (frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3})),  # singletons
]
PAYOFFS_OVERRIDE = np.array([
    [10., 8., 6., 4.],   # grand coalition
    [ 0., 0., 0., 0.],   # singletons
])
```

### Programmatic use

```python
from coalition_mip_n import vfi, find_equilibria, print_results, verify_equilibrium

# Single solve with custom proposer weights
V, sigmas, alphas, qs = vfi(delta=0.90, proposer_probs=[0.5, 0.3, 0.2])
print_results(V, sigmas, alphas, qs)
verify_equilibrium(sigmas, alphas, qs, V)

# Search
equilibria = find_equilibria(delta=0.80, n_restarts=60, seed=123)
print(f"Found {len(equilibria)} distinct equilibria")
```
