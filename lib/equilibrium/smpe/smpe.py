"""
smpe.py
=======

Stationary Markov Perfect Equilibria (SMPE) in farsighted coalition-formation
games, following the dynamic-bargaining protocol of Ray & Vohra.

The game
--------
N players are organised into a coalition structure (a set partition).  Each
period a proposer i is drawn with probability rho_i, proposes a new partition
s', and the *affected* players other than i vote.  Unanimity among those voters
implements s'; otherwise the state stays at s.  A unilateral exit by i (i
becomes a singleton, the rest of i's block stays intact) requires no votes.
Flow payoffs pi(s,i) accrue at the current state; players discount at delta.

Equilibrium
-----------
V_i(s) = (1-delta) pi(s,i) + delta sum_{s'} T(s,s') V_i(s')

  (C1) alpha_j(s,s') = 1 if V_j(s') > V_j(s), 0 if V_j(s') < V_j(s), free if =.
  (C2) supp sigma_i(s,.) subset of argmax_{s'} q_i(s,s') (V_i(s') - V_i(s)).

Both conditions are imposed at *every* (s,s') pair, including off path and at
never-pivotal votes (the no-weakly-dominated-voting refinement).  This is
required for (C2) to be well defined and is the default here; it can be
relaxed at never-pivotal nodes with strict_offpath=False in verify().

Algorithm
---------
Layer A  exact policy iteration (linear solve, never approximate VFI), with
         cycle detection on the *discrete* policy key.
Layer B  mixing solver: unknown mixing probabilities are pinned by an exactly
         square system of indifference equations, solved by damped Newton with
         an analytic Jacobian obtained from
             dV/dx = delta (I - delta T)^{-1} (dT/dx) V.
Layer C  homotopy in delta from delta ~ 0 (where V ~ pi and ties are
         non-generic) up to the target, with adaptive stepping.
Layer D  verification, always against V_induced = (I-delta T)^{-1}(1-delta)pi
         recomputed from the returned strategies alone.
Layer E  honest failure reporting.

Numerics
--------
Per-player affine rescaling of payoffs leaves the SMPE set *exactly* invariant:
T is row-stochastic so (I-delta T)^{-1}(1-delta) 1 = 1, hence pi_i -> a_i pi_i +
b_i implies V_i -> a_i V_i + b_i for any fixed strategy profile, and every
equilibrium condition is a within-player comparison of V across states.
Rescaling is therefore free, and is on by default (it is what makes Example 2
tractable).

Only numpy is required.

Author's note on the free-exit predicate
----------------------------------------
It is tempting to identify free exit with |Delta(s,s')| == 1.  That is wrong:
when i leaves a block of size >= 2 the members left behind also change blocks,
so |Delta| >= 2.  Indeed |Delta(s,s')| == 1 is impossible for s' != s.  The
predicate below is the literal definition from the specification; the
impossibility above is asserted as a consistency check.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CoalitionGeometry",
    "Game",
    "Solution",
    "VerificationReport",
    "solve_smpe",
    "solve_with_mixing_set",
    "projected_dynamics",
    "verify",
    "build_T",
    "compute_q",
    "evaluate",
    "evaluate_refined",
]


# =============================================================================
# 1.  Coalition structures
# =============================================================================


def restricted_growth_strings(n: int) -> Iterable[List[int]]:
    """Enumerate set partitions of {0..n-1} in restricted-growth-string order."""
    if n <= 0:
        yield []
        return
    a = [0] * n

    def rec(i: int, mx: int):
        if i == n:
            yield list(a)
            return
        for v in range(mx + 2):
            a[i] = v
            yield from rec(i + 1, max(mx, v))

    yield from rec(1, 0)


def rgs_to_partition(a: Sequence[int]) -> Tuple[frozenset, ...]:
    k = max(a) + 1 if len(a) else 0
    blocks = [frozenset(i for i, v in enumerate(a) if v == b) for b in range(k)]
    blocks.sort(key=min)
    return tuple(blocks)


def parse_partition(spec: Any, players: Sequence[str]) -> Tuple[frozenset, ...]:
    """Parse a partition given as a string or as nested sequences of names.

    Accepted string forms:  "{A,B}+{C}",  "{A,B} + {C}",  "AB|C",  "AB,C"
    Accepted sequence forms: [["A","B"],["C"]],  ["AB","C"] (single-char names)
    """
    idx = {p: i for i, p in enumerate(players)}
    single_char = all(len(p) == 1 for p in players)

    def to_block(tok: Any) -> frozenset:
        if isinstance(tok, (list, tuple, set, frozenset)):
            names = list(tok)
        else:
            t = str(tok).strip().strip("{}").strip()
            if "," in t:
                names = [u.strip() for u in t.split(",") if u.strip()]
            elif t in idx:
                names = [t]
            elif single_char:
                names = list(t)
            else:
                raise ValueError(f"cannot parse block {tok!r}")
        out = []
        for nm in names:
            nm = str(nm).strip()
            if nm not in idx:
                raise ValueError(f"unknown player {nm!r} in {spec!r}")
            out.append(idx[nm])
        return frozenset(out)

    if isinstance(spec, str):
        s = spec.strip()
        for sep in ("+", "|", ";"):
            if sep in s:
                toks = [u for u in s.split(sep) if u.strip()]
                break
        else:
            toks = [s]
        blocks = [to_block(t) for t in toks]
    else:
        blocks = [to_block(t) for t in spec]

    cover = set()
    for b in blocks:
        if cover & b:
            raise ValueError(f"overlapping blocks in {spec!r}")
        cover |= b
    if cover != set(range(len(players))):
        raise ValueError(f"partition {spec!r} does not cover all players")
    return tuple(sorted(blocks, key=min))


class CoalitionGeometry:
    """All partition-level combinatorics: states, movers, free exit, voters."""

    def __init__(self, players: Sequence[str]):
        self.players = list(players)
        self.N = len(self.players)
        if self.N < 2:
            raise ValueError("need at least 2 players")
        self.states = [rgs_to_partition(a) for a in restricted_growth_strings(self.N)]
        self.S = len(self.states)
        self.index = {p: i for i, p in enumerate(self.states)}
        self._build()
        self._sanity()

    # -- construction ------------------------------------------------------
    def _build(self):
        N, S = self.N, self.S
        self.block: List[List[frozenset]] = []
        for s in range(S):
            row = [None] * N
            for b in self.states[s]:
                for i in b:
                    row[i] = b
            self.block.append(row)

        movers = np.zeros((S, S, N), dtype=bool)
        free_exit = np.zeros((S, S, N), dtype=bool)
        for s in range(S):
            for t in range(S):
                for j in range(N):
                    movers[s, t, j] = self.block[s][j] != self.block[t][j]
                for i in range(N):
                    if self.block[t][i] != frozenset([i]):
                        continue
                    ok = True
                    for j in range(N):
                        if j == i:
                            continue
                        if self.block[t][j] != (self.block[s][j] - {i}):
                            ok = False
                            break
                    free_exit[s, t, i] = ok

        # voters[s,t,i,j] : j must vote on i's proposal s -> t
        voters = movers[:, :, None, :] & np.ones((1, 1, N, 1), dtype=bool)
        eye = np.eye(N, dtype=bool)
        voters = voters & (~eye)[None, None, :, :]        # proposer never votes
        voters = voters & (~free_exit)[:, :, :, None]     # free exit: no votes

        self.movers = movers
        self.free_exit = free_exit
        self.voters = voters
        # relevant[j,s,t] : j is a required voter for at least one proposer
        self.relevant = np.transpose(voters.any(axis=2), (2, 0, 1))
        # exit_target[s,i] : the unique state reached by i's unilateral exit
        self.exit_target = np.full((S, N), -1, dtype=int)
        for s in range(S):
            for i in range(N):
                tt = np.nonzero(free_exit[s, :, i])[0]
                self.exit_target[s, i] = int(tt[0])
        # [farsighted-coalitions] blocked[s,t,i]: i may not propose s -> t.
        # Empty under Jere's own rule; set by apply_committees().
        self.blocked = np.zeros((S, S, N), dtype=bool)
        self.committee_source = "jere"

    def apply_committees(self, voters: np.ndarray, blocked: np.ndarray,
                         source: str) -> None:
        """[farsighted-coalitions] Replace Jere's committee rule by an external one.

        voters[s,t,i,j]: j must approve i's proposal s -> t.  The proposer must
        NOT be included; see lib/equilibrium/smpe/__init__.py for why a proposer
        on its own committee is equivalent to leaving it off.
        blocked[s,t,i]: the proposal is forbidden (treated as q = 0).

        free_exit is recomputed as "i's unilateral exit, needing no vote and not
        blocked"; it is used only for reporting.  exit_target stays geometric.
        """
        S, N = self.S, self.N
        voters = np.asarray(voters, dtype=bool)
        blocked = np.asarray(blocked, dtype=bool)
        if voters.shape != (S, S, N, N) or blocked.shape != (S, S, N):
            raise ValueError(f"committee arrays have wrong shape: {voters.shape}, "
                             f"{blocked.shape}; expected {(S, S, N, N)}, {(S, S, N)}")
        for i in range(N):
            if voters[:, :, i, i].any():
                raise ValueError("proposer on its own committee; drop it before "
                                 "apply_committees (see smpe/__init__.py)")
        for s in range(S):
            if voters[s, s].any() or blocked[s, s].any():
                raise ValueError("the status quo can neither need votes nor be blocked")
        self.voters = voters
        self.blocked = blocked
        self.relevant = np.transpose(voters.any(axis=2), (2, 0, 1))
        geometric_exit = np.zeros((S, S, N), dtype=bool)
        for s in range(S):
            for i in range(N):
                geometric_exit[s, self.exit_target[s, i], i] = True
        self.free_exit = geometric_exit & ~voters.any(axis=3) & ~blocked
        self.committee_source = source

    def _sanity(self):
        S, N = self.S, self.N
        # (a) |Delta(s,t)| == 1 is impossible for t != s
        for s in range(S):
            for t in range(S):
                k = int(self.movers[s, t].sum())
                if t == s:
                    assert k == 0, "a state must not differ from itself"
                else:
                    assert k >= 2, (
                        f"|Delta|=={k} for distinct states {s},{t}: impossible"
                    )
        # (b) free exit implies no voters
        fe = self.free_exit
        assert not (fe[:, :, :, None] & self.voters).any(), "free exit with voters"
        # (c) each (s,i) has exactly one exit target; it is s iff i is a singleton
        for s in range(S):
            for i in range(N):
                assert fe[s, :, i].sum() == 1
                is_single = self.block[s][i] == frozenset([i])
                assert (self.exit_target[s, i] == s) == is_single
        # (d) the exit target is what it should be
        for s in range(S):
            for i in range(N):
                t = self.exit_target[s, i]
                assert self.block[t][i] == frozenset([i])
        # (e) proposer is never a voter
        for i in range(N):
            assert not self.voters[:, :, i, i].any()

    # -- presentation ------------------------------------------------------
    def label(self, s: int) -> str:
        return "+".join(
            "{" + ",".join(self.players[i] for i in sorted(b)) + "}"
            for b in self.states[s]
        )

    def labels(self) -> List[str]:
        return [self.label(s) for s in range(self.S)]

    def permutation_from(self, rows: Sequence[Any]) -> np.ndarray:
        """perm[s] = index in `rows` of internal state s."""
        if len(rows) != self.S:
            raise ValueError(f"expected {self.S} rows, got {len(rows)}")
        pos = {}
        for r, spec in enumerate(rows):
            p = parse_partition(spec, self.players)
            if p in pos:
                raise ValueError(f"duplicate partition {spec!r}")
            pos[p] = r
        if set(pos) != set(self.states):
            missing = [self.label(s) for s in range(self.S) if self.states[s] not in pos]
            raise ValueError(f"row list is not a complete state list; missing {missing}")
        return np.array([pos[self.states[s]] for s in range(self.S)], dtype=int)


# =============================================================================
# 2.  Game container (with affine rescaling)
# =============================================================================


@dataclass
class Game:
    geom: CoalitionGeometry
    payoffs: np.ndarray            # (S,N) in solver units
    delta: float
    rho: np.ndarray                # (N,)
    raw_payoffs: np.ndarray        # (S,N) in original units, internal ordering
    scale: np.ndarray              # (N,)  raw = solver*scale + shift
    shift: np.ndarray              # (N,)
    row_order: Optional[List[str]] = None

    @property
    def S(self) -> int:
        return self.geom.S

    @property
    def N(self) -> int:
        return self.geom.N

    def to_raw(self, V: np.ndarray) -> np.ndarray:
        return V * self.scale[None, :] + self.shift[None, :]

    @staticmethod
    def build(players, payoffs, delta, rho=None, rows=None, rescale=True,
              target_spread=1.0, committees=None) -> "Game":
        geom = CoalitionGeometry(players)
        if committees is not None:          # [farsighted-coalitions]
            geom.apply_committees(committees.voters, committees.blocked,
                                  committees.source)
        P = np.asarray(payoffs, dtype=float)
        if P.shape != (geom.S, geom.N):
            raise ValueError(f"payoffs must have shape {(geom.S, geom.N)}, got {P.shape}")
        row_order = None
        if rows is not None:
            perm = geom.permutation_from(rows)
            P = P[perm, :]
            row_order = [str(r) for r in rows]
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must lie in (0,1)")
        if rho is None:
            r = np.full(geom.N, 1.0 / geom.N)
        else:
            r = np.asarray(rho, dtype=float)
            if r.shape != (geom.N,) or (r < 0).any() or not np.isclose(r.sum(), 1.0):
                raise ValueError("rho must be a probability vector of length N")
        raw = P.copy()
        if rescale:
            shift = P.mean(axis=0)
            dev = np.abs(P - shift[None, :]).max(axis=0)
            scale = np.where(dev > 0, dev / target_spread, 1.0)
            Q = (P - shift[None, :]) / scale[None, :]
        else:
            shift = np.zeros(geom.N)
            scale = np.ones(geom.N)
            Q = P.copy()
        return Game(geom=geom, payoffs=Q, delta=float(delta), rho=r,
                    raw_payoffs=raw, scale=scale, shift=shift, row_order=row_order)

    def with_delta(self, d: float) -> "Game":
        return Game(self.geom, self.payoffs, float(d), self.rho, self.raw_payoffs,
                    self.scale, self.shift, self.row_order)


# =============================================================================
# 3.  Primitives: q, T, policy evaluation
# =============================================================================


def compute_q(alpha: np.ndarray, committees,
              exclude_j: Optional[int] = None) -> np.ndarray:
    """q[i,s,t] = prod_{j in voters(s,t,i)} alpha[j,s,t].  Empty product = 1.

    [farsighted-coalitions] `committees` is a CoalitionGeometry (preferred) or a
    bare voters array.  With a geometry, q is forced to 0 on geom.blocked: a
    proposal the effectivity rule forbids is, like one certain to be rejected,
    identical to staying put, which the rest of the solver already handles.
    Every derivative of q goes through this function (exclude_j), so the
    blocked entries have zero derivative consistently.
    """
    if isinstance(committees, CoalitionGeometry):
        voters, blocked = committees.voters, committees.blocked
    else:
        voters, blocked = committees, None
    A = np.transpose(alpha, (1, 2, 0))[:, :, None, :]      # (S,S,1,N) -> [s,t,.,j]
    V = voters
    if exclude_j is not None:
        V = voters.copy()
        V[:, :, :, exclude_j] = False
    vals = np.where(V, A, 1.0)
    q = vals.prod(axis=3)                                   # (S,S,N) -> [s,t,i]
    if blocked is not None:
        q = np.where(blocked, 0.0, q)
    return np.transpose(q, (2, 0, 1))                       # (N,S,S)


def build_T(sigma: np.ndarray, q: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """T[s,t] = sum_i rho_i sigma_i(s,t) q_i(s,t) off-diagonal; diagonal absorbs."""
    off = np.einsum("i,ist,ist->st", rho, sigma, q)
    np.fill_diagonal(off, 0.0)
    T = off.copy()
    np.fill_diagonal(T, 1.0 - off.sum(axis=1))
    return T


def evaluate(T: np.ndarray, payoffs: np.ndarray, delta: float) -> np.ndarray:
    """Exact policy evaluation: V = (I - delta T)^{-1} (1-delta) pi."""
    S = T.shape[0]
    A = np.eye(S) - delta * T
    return np.linalg.solve(A, (1.0 - delta) * payoffs)


def evaluate_refined(T: np.ndarray, payoffs: np.ndarray, delta: float):
    """Policy evaluation with iterative refinement in extended precision.

    Returns (V_float64, V_longdouble).  I - delta T is strictly diagonally
    dominant with row sums 1-delta, so kappa ~ O(1/(1-delta)); refinement is
    cheap insurance for the near-tie checks in verification.
    """
    S = T.shape[0]
    A = np.eye(S) - delta * T
    b = (1.0 - delta) * payoffs
    V = np.linalg.solve(A, b)
    Al = A.astype(np.longdouble)
    bl = b.astype(np.longdouble)
    Vl = V.astype(np.longdouble)
    for _ in range(3):
        r = bl - Al @ Vl
        dx = np.linalg.solve(A, np.asarray(r, dtype=np.float64))
        Vl = Vl + dx.astype(np.longdouble)
    return np.asarray(Vl, dtype=np.float64), Vl


def gain_matrix(V: np.ndarray) -> np.ndarray:
    """gain[i,s,t] = V[t,i] - V[s,i]."""
    Vt = V.T                                    # (N,S)
    return Vt[:, None, :] - Vt[:, :, None]


# =============================================================================
# 4.  Pure best responses and policy iteration
# =============================================================================


@dataclass
class PurePolicy:
    alpha: np.ndarray      # (N,S,S) in {0,1}
    choice: np.ndarray     # (N,S) int: the state each proposer picks
    sigma: np.ndarray      # (N,S,S) degenerate
    q: np.ndarray          # (N,S,S)

    def key(self) -> Tuple[bytes, bytes]:
        return (self.alpha.astype(np.int8).tobytes(),
                np.ascontiguousarray(self.choice).tobytes())


def pure_best_response(V: np.ndarray, geom: CoalitionGeometry, *,
                       tie_accept: bool = True, tie_break: str = "stay",
                       tol: float = 1e-11,
                       rng: Optional[np.random.Generator] = None) -> PurePolicy:
    N, S = geom.N, geom.S
    g = gain_matrix(V)
    tie_val = 1.0 if tie_accept else 0.0
    alpha = np.where(g > tol, 1.0, np.where(g < -tol, 0.0, tie_val))

    q = compute_q(alpha, geom)
    G = q * g                                   # (N,S,S) proposer gain
    best = G.max(axis=2)                        # (N,S)
    isbest = G >= best[:, :, None] - tol

    choice = np.empty((N, S), dtype=int)
    for i in range(N):
        for s in range(S):
            if isbest[i, s, s]:                 # staying is optimal
                choice[i, s] = s
                continue
            cand = np.nonzero(isbest[i, s])[0]
            if tie_break == "stay" or tie_break == "first":
                choice[i, s] = int(cand[0])
            elif tie_break == "last":
                choice[i, s] = int(cand[-1])
            elif tie_break == "random":
                assert rng is not None
                choice[i, s] = int(rng.choice(cand))
            else:
                raise ValueError(f"unknown tie_break {tie_break!r}")

    sigma = np.zeros((N, S, S))
    for i in range(N):
        sigma[i, np.arange(S), choice[i]] = 1.0
    return PurePolicy(alpha=alpha, choice=choice, sigma=sigma, q=q)


@dataclass
class PIResult:
    status: str                       # 'fixed' | 'cycle' | 'maxiter'
    policies: List[PurePolicy]        # the cycle (length 1 if fixed point)
    V_seq: List[np.ndarray]           # V that generated each cycle policy
    iterations: int


def policy_iteration(game: Game, V0: np.ndarray, *, tie_accept=True,
                     tie_break="stay", max_iter=400, tol=1e-11,
                     rng=None) -> PIResult:
    """Exact policy iteration with cycle detection on the discrete policy key."""
    geom = game.geom
    V = np.array(V0, dtype=float, copy=True)
    seen: Dict[Tuple[bytes, bytes], int] = {}
    pols: List[PurePolicy] = []
    Vs: List[np.ndarray] = []
    for k in range(max_iter):
        pol = pure_best_response(V, geom, tie_accept=tie_accept,
                                 tie_break=tie_break, tol=tol, rng=rng)
        kk = pol.key()
        if kk in seen:
            st = seen[kk]
            cyc, cycV = pols[st:], Vs[st:]
            return PIResult("fixed" if len(cyc) == 1 else "cycle", cyc, cycV, k)
        seen[kk] = len(pols)
        pols.append(pol)
        Vs.append(V)
        T = build_T(pol.sigma, pol.q, game.rho)
        V = evaluate(T, game.payoffs, game.delta)
    return PIResult("maxiter", pols[-1:], Vs[-1:], max_iter)


# =============================================================================
# 5.  Mixing regimes and the indifference system
# =============================================================================


@dataclass
class Regime:
    """A candidate equilibrium regime: which decisions mix, and the rest fixed."""
    accept_vars: List[Tuple[int, int, int]]                 # (j,s,t)
    prop_vars: List[Tuple[int, int, Tuple[int, ...]]]       # (i,s,support)
    base_alpha: np.ndarray                                  # (N,S,S)
    base_choice: np.ndarray                                 # (N,S)
    origin: str = ""

    @property
    def accept_eqs(self) -> List[Tuple[int, int, int]]:
        """Acceptance indifference equations, deduplicated.

        V_j(t) - V_j(s) = 0 and V_j(s) - V_j(t) = 0 are the *same* condition, but
        alpha_j(s,t) and alpha_j(t,s) are two *different* strategic variables.
        When a cycle flips both directions we therefore get two unknowns and one
        equation: the indifference set is a manifold and any feasible point on it
        is an equilibrium.  The system is solved in least-norm form for exactly
        this reason.
        """
        seen = set()
        out = []
        for (j, s, t) in self.accept_vars:
            key = (j, min(s, t), max(s, t))
            if key in seen:
                continue
            seen.add(key)
            out.append((j, s, t))
        return out

    @property
    def unknowns(self) -> List[Tuple]:
        u: List[Tuple] = [("accept", j, s, t) for (j, s, t) in self.accept_vars]
        for (i, s, sup) in self.prop_vars:
            for r in range(len(sup) - 1):
                u.append(("prop", i, s, sup, r))
        return u

    @property
    def k(self) -> int:
        return len(self.unknowns)

    def key(self) -> Tuple:
        return (tuple(sorted(self.accept_vars)),
                tuple(sorted((i, s, sup) for (i, s, sup) in self.prop_vars)),
                self.base_alpha.astype(np.int8).tobytes(),
                np.ascontiguousarray(self.base_choice).tobytes())

    def describe(self, geom: CoalitionGeometry) -> List[str]:
        out = []
        for (j, s, t) in self.accept_vars:
            out.append(f"  accept  {geom.players[j]:>4s} on "
                       f"{geom.label(s)} -> {geom.label(t)}")
        for (i, s, sup) in self.prop_vars:
            opts = " / ".join(geom.label(a) for a in sup)
            out.append(f"  propose {geom.players[i]:>4s} at {geom.label(s)}"
                       f" over [{opts}]")
        return out


def regime_from_cycle(cycle: List[PurePolicy], geom: CoalitionGeometry) -> Regime:
    alphas = np.stack([p.alpha for p in cycle])        # (m,N,S,S)
    choices = np.stack([p.choice for p in cycle])      # (m,N,S)
    a_flip = (alphas.max(0) != alphas.min(0)) & geom.relevant
    accept_vars = [tuple(int(v) for v in idx) for idx in zip(*np.nonzero(a_flip))]
    c_flip = choices.max(0) != choices.min(0)
    prop_vars = []
    for i, s in zip(*np.nonzero(c_flip)):
        sup = tuple(sorted(set(int(v) for v in choices[:, i, s])))
        prop_vars.append((int(i), int(s), sup))
    return Regime(accept_vars, prop_vars, cycle[0].alpha.copy(),
                  cycle[0].choice.copy(), origin="cycle")


def regime_from_V(V: np.ndarray, geom: CoalitionGeometry, *, tie_tol: float,
                  tie_accept: bool = True, origin: str = "V") -> Regime:
    """Build a regime by declaring every near-indifference a mixing decision."""
    N, S = geom.N, geom.S
    g = gain_matrix(V)
    alpha = np.where(g > tie_tol, 1.0, np.where(g < -tie_tol, 0.0,
                                                1.0 if tie_accept else 0.0))
    near_a = (np.abs(g) <= tie_tol) & geom.relevant
    accept_vars = [tuple(int(v) for v in idx) for idx in zip(*np.nonzero(near_a))]

    q = compute_q(alpha, geom)
    G = q * g
    best = G.max(axis=2)
    isbest = G >= best[:, :, None] - tie_tol
    choice = np.empty((N, S), dtype=int)
    prop_vars = []
    for i in range(N):
        for s in range(S):
            if isbest[i, s, s]:
                # staying is optimal; every t with q_i(s,t)==0 or V_i(t)==V_i(s)
                # ties with it but is payoff- and transition-equivalent to
                # staying, so there is nothing to mix over.
                choice[i, s] = s
                continue
            cand = np.nonzero(isbest[i, s])[0]
            choice[i, s] = int(cand[0])
            if len(cand) > 1:
                prop_vars.append((i, s, tuple(int(a) for a in cand)))
    return Regime(accept_vars, prop_vars, alpha, choice, origin=origin)


class RegimeSystem:
    """Residual and analytic Jacobian of the indifference system of a regime."""

    def __init__(self, game: Game, regime: Regime):
        self.game = game
        self.geom = game.geom
        self.regime = regime
        self.unknowns = regime.unknowns
        self.k = len(self.unknowns)
        self.m = len(regime.accept_eqs) + sum(len(sup) - 1
                                              for (_, _, sup) in regime.prop_vars)

    # -- policy from x -----------------------------------------------------
    def unpack(self, x: np.ndarray):
        geom, R = self.geom, self.regime
        N, S = geom.N, geom.S
        alpha = R.base_alpha.copy()
        p = 0
        for (j, s, t) in R.accept_vars:
            alpha[j, s, t] = x[p]
            p += 1
        sigma = np.zeros((N, S, S))
        for i in range(N):
            sigma[i, np.arange(S), R.base_choice[i]] = 1.0
        for (i, s, sup) in R.prop_vars:
            sigma[i, s, :] = 0.0
            L = len(sup)
            w = np.empty(L)
            w[: L - 1] = x[p: p + L - 1]
            p += L - 1
            w[L - 1] = 1.0 - w[: L - 1].sum()
            for r, t in enumerate(sup):
                sigma[i, s, t] = w[r]
        return sigma, alpha

    def state(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        sigma, alpha = self.unpack(x)
        q = compute_q(alpha, self.geom)
        T = build_T(sigma, q, self.game.rho)
        V = evaluate(T, self.game.payoffs, self.game.delta)
        return dict(sigma=sigma, alpha=alpha, q=q, T=T, V=V)

    # -- residual ----------------------------------------------------------
    def residual_from_state(self, st: Dict[str, np.ndarray]) -> np.ndarray:
        V, q = st["V"], st["q"]
        F = []
        for (j, s, t) in self.regime.accept_eqs:
            F.append(V[t, j] - V[s, j])
        for (i, s, sup) in self.regime.prop_vars:
            a1 = sup[0]
            g1 = q[i, s, a1] * (V[a1, i] - V[s, i])
            for a in sup[1:]:
                F.append(g1 - q[i, s, a] * (V[a, i] - V[s, i]))
        return np.array(F, dtype=float)

    def residual(self, x: np.ndarray) -> np.ndarray:
        return self.residual_from_state(self.state(x))

    # -- analytic Jacobian -------------------------------------------------
    def jacobian(self, x: np.ndarray, st: Optional[Dict] = None) -> np.ndarray:
        if st is None:
            st = self.state(x)
        geom, game = self.geom, self.game
        S, N = geom.S, geom.N
        V, q, T, sigma, alpha = st["V"], st["q"], st["T"], st["sigma"], st["alpha"]
        M = np.eye(S) - game.delta * T

        dVs, dqs = [], []
        for u in self.unknowns:
            dsigma = np.zeros_like(sigma)
            dq = np.zeros_like(q)
            if u[0] == "accept":
                _, j, s, t = u
                partial = compute_q(alpha, geom, exclude_j=j)   # (N,S,S)
                dq[:, s, t] = geom.voters[s, t, :, j] * partial[:, s, t]
            else:
                _, i, s, sup, r = u
                dsigma[i, s, sup[r]] += 1.0
                dsigma[i, s, sup[-1]] -= 1.0
            doff = (np.einsum("i,ist,ist->st", game.rho, dsigma, q)
                    + np.einsum("i,ist,ist->st", game.rho, sigma, dq))
            np.fill_diagonal(doff, 0.0)
            dT = doff.copy()
            np.fill_diagonal(dT, -doff.sum(axis=1))
            dV = np.linalg.solve(M, game.delta * (dT @ V))
            dVs.append(dV)
            dqs.append(dq)

        J = np.zeros((self.m, self.k))
        row = 0
        for (j, s, t) in self.regime.accept_eqs:
            for p in range(self.k):
                J[row, p] = dVs[p][t, j] - dVs[p][s, j]
            row += 1
        for (i, s, sup) in self.regime.prop_vars:
            a1 = sup[0]
            for a in sup[1:]:
                for p in range(self.k):
                    d1 = (dqs[p][i, s, a1] * (V[a1, i] - V[s, i])
                          + q[i, s, a1] * (dVs[p][a1, i] - dVs[p][s, i]))
                    d2 = (dqs[p][i, s, a] * (V[a, i] - V[s, i])
                          + q[i, s, a] * (dVs[p][a, i] - dVs[p][s, i]))
                    J[row, p] = d1 - d2
                row += 1
        return J

    # -- feasibility -------------------------------------------------------
    def feasible(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        if (x < -tol).any() or (x > 1 + tol).any():
            return False
        p = len(self.regime.accept_vars)
        for (i, s, sup) in self.regime.prop_vars:
            L = len(sup)
            w = x[p: p + L - 1]
            p += L - 1
            if w.sum() > 1 + tol:
                return False
        return True

    def clip(self, x: np.ndarray) -> np.ndarray:
        y = np.clip(x, 0.0, 1.0)
        p = len(self.regime.accept_vars)
        for (i, s, sup) in self.regime.prop_vars:
            L = len(sup)
            w = y[p: p + L - 1]
            tot = w.sum()
            if tot > 1.0:
                y[p: p + L - 1] = w / tot
            p += L - 1
        return y


# =============================================================================
# 6.  Newton solve for a regime
# =============================================================================


def _newton(sysm: RegimeSystem, x0: np.ndarray, *, ftol=1e-12, xtol=1e-14,
            max_iter=120) -> Tuple[bool, np.ndarray, float]:
    """Least-norm Gauss-Newton with Levenberg damping and a box.

    The system need not be square: duplicated acceptance indifferences leave
    more unknowns than equations, and any feasible point on the solution
    manifold is an equilibrium.  The min-norm step handles both cases.
    """
    x = sysm.clip(np.array(x0, dtype=float))
    F = sysm.residual(x)
    nF = np.linalg.norm(F, np.inf)
    lam = 0.0
    for _ in range(max_iter):
        if nF < ftol:
            return True, x, nF
        st = sysm.state(x)
        J = sysm.jacobian(x, st)
        if lam > 0:
            A = np.vstack([J, np.sqrt(lam) * np.eye(sysm.k)])
            b = np.concatenate([F, np.zeros(sysm.k)])
            dx = -np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            dx = -np.linalg.lstsq(J, F, rcond=None)[0]
        if not np.all(np.isfinite(dx)):
            return False, x, nF
        step, improved = 1.0, False
        for _ in range(40):
            xn = sysm.clip(x + step * dx)
            Fn = sysm.residual(xn)
            nFn = np.linalg.norm(Fn, np.inf)
            if nFn < nF:
                improved = True
                break
            step *= 0.5
        if not improved:
            if lam == 0.0:
                lam = 1e-8
                continue
            if lam < 1e4:
                lam *= 30.0
                continue
            return nF < ftol, x, nF
        lam = max(lam * 0.1, 0.0) if lam > 1e-12 else 0.0
        if np.linalg.norm(xn - x, np.inf) < xtol and nFn >= ftol:
            return nFn < ftol, xn, nFn
        x, F, nF = xn, Fn, nFn
    return nF < ftol, x, nF


def _bisect_1d(sysm: RegimeSystem, lo: float, hi: float, *, ftol=1e-12,
               iters=200) -> Optional[float]:
    flo = sysm.residual(np.array([lo]))[0]
    fhi = sysm.residual(np.array([hi]))[0]
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = sysm.residual(np.array([mid]))[0]
        if abs(fm) < ftol or hi - lo < 1e-15:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def solve_regime(game: Game, regime: Regime, *, x_hints: Sequence[np.ndarray] = (),
                 rng: Optional[np.random.Generator] = None, n_random: int = 40,
                 ftol=1e-12, max_iter: int = 120,
                 deadline: Optional[float] = None) -> Dict[str, Any]:
    """Find x in [0,1]^k solving the regime's indifference system."""
    sysm = RegimeSystem(game, regime)
    k = sysm.k
    if k == 0:
        return dict(ok=False, reason="degenerate regime: no mixing unknowns",
                    system=sysm)

    starts: List[np.ndarray] = [np.asarray(h, dtype=float) for h in x_hints]
    starts.append(np.full(k, 0.5))
    if k == 1:
        starts += [np.array([v]) for v in (0.1, 0.25, 0.75, 0.9)]
    if rng is None:
        rng = np.random.default_rng(0)
    for _ in range(n_random):
        starts.append(rng.uniform(0.02, 0.98, size=k))

    # k == 1: bracket first (the residual generically changes sign between the
    # two pure regimes that the VFI cycle alternates between).
    if k == 1:
        r = _bisect_1d(sysm, 0.0, 1.0, ftol=ftol)
        if r is not None:
            starts.insert(0, np.array([r]))

    best = None
    for x0 in starts:
        if deadline is not None and time.time() > deadline and best is not None:
            break
        ok, x, nF = _newton(sysm, x0, ftol=ftol, max_iter=max_iter)
        if ok and sysm.feasible(x):
            J = sysm.jacobian(x)
            sv = np.linalg.svd(J, compute_uv=False) if J.size else np.array([1.0])
            rank = int((sv > 1e-9 * max(sv[0], 1e-300)).sum())
            rank_def = bool(rank < sysm.k)
            return dict(ok=True, x=x, resid=nF, system=sysm,
                        rank_deficient=rank_def, singvals=sv)
        if best is None or nF < best[1]:
            best = (x, nF)
    xb, nb = best
    reason = ("Newton converged outside [0,1]^k (x = "
              f"{np.array2string(xb, precision=4)})" if nb < ftol
              else f"no root of the indifference system found (|F|inf = {nb:.3e})")
    return dict(ok=False, reason=reason, x=xb, resid=nb, system=sysm)


# =============================================================================
# 7.  Regime validation (are the *non*-mixing decisions still best responses?)
# =============================================================================


def validate_regime(game: Game, regime: Regime, st: Dict[str, np.ndarray],
                    tol: float = 1e-9) -> Tuple[bool, List[str], List[Tuple]]:
    """Are the *non*-mixing decisions still best responses at the solved V?

    Also returns the violating decisions as candidate keys: these are precisely
    the decisions that the mixing set is missing, which lets the search grow the
    set in a guided way instead of enumerating subsets blindly.
    """
    geom = game.geom
    V, q, sigma, alpha = st["V"], st["q"], st["sigma"], st["alpha"]
    g = gain_matrix(V)
    bad: List[str] = []

    keys: List[Tuple] = []
    mixing_a = set(regime.accept_vars)
    rel = geom.relevant
    for j in range(geom.N):
        for s in range(geom.S):
            for t in range(geom.S):
                if not rel[j, s, t] or (j, s, t) in mixing_a:
                    continue
                if g[j, s, t] > tol and alpha[j, s, t] < 1 - tol:
                    bad.append(f"alpha({geom.players[j]}|{s}->{t}) should be 1")
                    keys.append(("acc", j, min(s, t), max(s, t)))
                elif g[j, s, t] < -tol and alpha[j, s, t] > tol:
                    bad.append(f"alpha({geom.players[j]}|{s}->{t}) should be 0")
                    keys.append(("acc", j, min(s, t), max(s, t)))

    G = q * g
    best = G.max(axis=2)
    for i in range(geom.N):
        for s in range(geom.S):
            sup = np.nonzero(sigma[i, s] > tol)[0]
            for t in sup:
                if best[i, s] - G[i, s, t] > tol:
                    bad.append(f"sigma({geom.players[i]}@{s}) puts mass on {t}, "
                               f"regret {best[i, s] - G[i, s, t]:.3e}")
                    keys.append(("prop", i, s, None))
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    return (len(bad) == 0), bad, keys


# =============================================================================
# 8.  Verification -- ALWAYS against V_induced, never against an internal V
# =============================================================================


@dataclass
class VerificationReport:
    ok: bool
    V_induced: np.ndarray
    T: np.ndarray
    max_accept_violation: float
    max_proposal_regret: float
    bellman_residual: float
    sigma_simplex_error: float
    alpha_range_error: float
    row_stochastic_error: float
    strict_offpath: bool
    tol: float
    violations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        s = (f"verification [{head}] (tol={self.tol:.1e}, "
             f"off-path={'strict' if self.strict_offpath else 'relaxed'})\n"
             f"    max acceptance violation : {self.max_accept_violation:.3e}\n"
             f"    max proposal regret      : {self.max_proposal_regret:.3e}\n"
             f"    Bellman residual         : {self.bellman_residual:.3e}\n"
             f"    sigma simplex error      : {self.sigma_simplex_error:.3e}\n"
             f"    alpha range error        : {self.alpha_range_error:.3e}\n"
             f"    T row-stochastic error   : {self.row_stochastic_error:.3e}")
        if self.violations:
            s += "\n    violations:\n" + "\n".join(
                "      - " + v for v in self.violations[:20])
            if len(self.violations) > 20:
                s += f"\n      ... and {len(self.violations)-20} more"
        return s


def verify(game: Game, sigma: np.ndarray, alpha: np.ndarray, *,
           tol: float = 1e-7, strict_offpath: bool = True) -> VerificationReport:
    """Check that (sigma, alpha) is an SMPE, using only (sigma, alpha, pi, delta, rho).

    Everything is rebuilt from scratch and every equilibrium condition is
    checked against V_induced = (I - delta T)^{-1}(1-delta) pi.  No internal
    value function from the solver is used or trusted.

    Sufficiency: payoffs are bounded and discounted, so the one-shot deviation
    principle applies; proposing and voting are the only decision nodes, and
    conditions (C1)-(C2) are exactly the one-shot deviation conditions there.
    """
    geom = game.geom
    N, S = geom.N, geom.S
    viol: List[str] = []

    # --- structural ------------------------------------------------------
    sig_err = float(np.abs(sigma.sum(axis=2) - 1.0).max())
    if sig_err > tol:
        viol.append(f"proposal distributions do not sum to 1 (err {sig_err:.3e})")
    neg = float(max(0.0, -sigma.min()))
    if neg > tol:
        viol.append(f"negative proposal probability ({neg:.3e})")
    a_err = float(max(0.0, -alpha.min(), alpha.max() - 1.0))
    if a_err > tol:
        viol.append(f"acceptance probability outside [0,1] ({a_err:.3e})")

    q = compute_q(alpha, geom)
    T = build_T(sigma, q, game.rho)
    rs_err = float(max(np.abs(T.sum(axis=1) - 1.0).max(), max(0.0, -T.min())))
    if rs_err > tol:
        viol.append(f"T is not row-stochastic (err {rs_err:.3e})")

    # --- induced value function (extended-precision refinement) ----------
    V, Vl = evaluate_refined(T, game.payoffs, game.delta)
    d = np.longdouble(game.delta)
    bell = float(np.abs(Vl - ((1 - d) * game.payoffs.astype(np.longdouble)
                              + d * (T.astype(np.longdouble) @ Vl))).max())

    gl = Vl.T[:, None, :] - Vl.T[:, :, None]          # gain[j,s,t], longdouble

    # --- (C1) acceptance cutoffs ----------------------------------------
    max_acc = 0.0
    if strict_offpath:
        mask = geom.relevant.copy()
    else:
        # relaxed: only votes where j is pivotal with positive probability
        piv = np.zeros((N, S, S), dtype=bool)
        for j in range(N):
            other = compute_q(alpha, geom, exclude_j=j)     # (N,S,S)
            reach = np.einsum("i,ist->st", game.rho, sigma * other)
            piv[j] = geom.relevant[j] & (reach > tol)
        mask = piv
    for j in range(N):
        for s in range(S):
            for t in range(S):
                if not mask[j, s, t]:
                    continue
                gv = float(gl[j, s, t])
                if gv > tol:
                    v = float(1.0 - alpha[j, s, t])
                elif gv < -tol:
                    v = float(alpha[j, s, t])
                else:
                    v = 0.0
                if v > max_acc:
                    max_acc = v
                if v > tol:
                    viol.append(
                        f"(C1) {geom.players[j]} at {geom.label(s)} -> "
                        f"{geom.label(t)}: gain {gv:+.3e} but alpha="
                        f"{alpha[j,s,t]:.6f}")

    # --- (C2) proposals are best responses -------------------------------
    ql = q.astype(np.longdouble)
    G = ql * gl
    best = G.max(axis=2)
    max_reg = 0.0
    for i in range(N):
        for s in range(S):
            for t in np.nonzero(sigma[i, s] > tol)[0]:
                reg = float(best[i, s] - G[i, s, t])
                if reg > max_reg:
                    max_reg = reg
                if reg > tol:
                    viol.append(
                        f"(C2) {geom.players[i]} at {geom.label(s)} proposes "
                        f"{geom.label(int(t))} with prob {sigma[i,s,t]:.4f}; "
                        f"regret {reg:.3e}")

    ok = (not viol) and max_acc <= tol and max_reg <= tol and bell <= 1e-9
    return VerificationReport(
        ok=ok, V_induced=V, T=T, max_accept_violation=max_acc,
        max_proposal_regret=max_reg, bellman_residual=bell,
        sigma_simplex_error=sig_err, alpha_range_error=a_err,
        row_stochastic_error=rs_err, strict_offpath=strict_offpath,
        tol=tol, violations=viol)


# =============================================================================
# 9.  Markov chain analysis
# =============================================================================


def chain_analysis(T: np.ndarray, eps: float = 1e-12) -> Dict[str, Any]:
    """Recurrent classes, their stationary laws, absorption and long-run law."""
    S = T.shape[0]
    A = T > eps
    R = A | np.eye(S, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(S, 2)))) + 1):   # transitive closure
        R = R | (R @ R)
    recurrent = np.array([all(R[t, s] for t in range(S) if R[s, t])
                          for s in range(S)])
    classes: List[List[int]] = []
    assigned = -np.ones(S, dtype=int)
    for s in range(S):
        if not recurrent[s] or assigned[s] >= 0:
            continue
        cl = [t for t in range(S) if recurrent[t] and R[s, t] and R[t, s]]
        for t in cl:
            assigned[t] = len(classes)
        classes.append(sorted(cl))

    stationaries = []
    for cl in classes:
        sub = T[np.ix_(cl, cl)]
        M = np.vstack([sub.T - np.eye(len(cl)), np.ones(len(cl))])
        b = np.zeros(len(cl) + 1)
        b[-1] = 1.0
        mu, *_ = np.linalg.lstsq(M, b, rcond=None)
        mu = np.clip(mu, 0, None)
        mu = mu / mu.sum()
        stationaries.append(mu)

    longrun = np.zeros((S, S))
    trans = [s for s in range(S) if assigned[s] < 0]
    absorb = np.zeros((S, len(classes)))
    for c, cl in enumerate(classes):
        for t in cl:
            absorb[t, c] = 1.0
    if trans and classes:
        Q = T[np.ix_(trans, trans)]
        Rm = np.column_stack([T[np.ix_(trans, cl)].sum(axis=1) for cl in classes])
        X = np.linalg.solve(np.eye(len(trans)) - Q, Rm)
        for r, s in enumerate(trans):
            absorb[s, :] = X[r, :]
    for s in range(S):
        for c, cl in enumerate(classes):
            longrun[s, np.array(cl)] += absorb[s, c] * stationaries[c]
    return dict(classes=classes, stationary=stationaries, absorption=absorb,
                longrun=longrun, recurrent=recurrent)


def exit_stability(game: Game, V: np.ndarray, tol: float = 1e-10):
    """For each state, which players strictly gain by exiting unilaterally."""
    geom = game.geom
    out = {}
    for s in range(geom.S):
        movers = []
        for i in range(geom.N):
            t = geom.exit_target[s, i]
            if t != s and V[t, i] - V[s, i] > tol:
                movers.append((geom.players[i], float(V[t, i] - V[s, i])))
        out[s] = movers
    return out


# =============================================================================
# 10.  Solution container
# =============================================================================


@dataclass
class Solution:
    game: Game
    sigma: np.ndarray
    alpha: np.ndarray
    T: np.ndarray
    V: np.ndarray                 # in solver (rescaled) units
    V_raw: np.ndarray             # in original payoff units
    status: str                   # PURE | MIXED | MIXED_NONUNIQUE
    report: VerificationReport
    chain: Dict[str, Any]
    route: str = ""
    regime: Optional[Regime] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def dedup_key(self, nd: int = 6) -> Tuple:
        return tuple(np.round(self.V, nd).ravel()) + tuple(np.round(self.T, nd).ravel())

    # -- printing ---------------------------------------------------------
    def describe(self, show_strategies: bool = True) -> str:
        g, geom = self.game, self.game.geom
        L = geom.labels()
        w = max(len(x) for x in L)
        out = []
        out.append(f"status: {self.status}   route: {self.route}   "
                   f"delta = {g.delta}")
        out.append("")
        out.append("Equilibrium value function V (original payoff units)")
        out.append("  " + " " * w + "  " + "".join(f"{p:>12s}" for p in geom.players))
        for s in range(geom.S):
            out.append("  " + L[s].ljust(w) + "  "
                       + "".join(f"{self.V_raw[s,i]:12.6f}" for i in range(geom.N)))
        out.append("")
        out.append("Transition matrix T")
        out.append("  " + " " * w + "  " + "".join(f"{x:>10s}" for x in
                                                   [f"[{i}]" for i in range(geom.S)]))
        for s in range(geom.S):
            out.append("  " + L[s].ljust(w) + "  "
                       + "".join(f"{self.T[s,t]:10.4f}" for t in range(geom.S)))
        out.append("")
        cls = self.chain["classes"]
        out.append(f"Recurrent classes: {len(cls)}")
        for c, cl in enumerate(cls):
            mem = ", ".join(L[t] for t in cl)
            mu = ", ".join(f"{L[t]}:{m:.4f}" for t, m in zip(cl, self.chain['stationary'][c]))
            kind = "absorbing state" if len(cl) == 1 else f"cycle of {len(cl)}"
            out.append(f"  class {c} ({kind}): {mem}")
            if len(cl) > 1:
                out.append(f"      stationary law: {mu}")
        out.append("")
        out.append("Long-run distribution by starting state")
        out.append("  " + " " * w + "  " + "".join(f"{x:>10s}" for x in
                                                   [f"[{i}]" for i in range(geom.S)]))
        for s in range(geom.S):
            out.append("  " + L[s].ljust(w) + "  "
                       + "".join(f"{self.chain['longrun'][s,t]:10.4f}"
                                 for t in range(geom.S)))
        if show_strategies:
            out.append("")
            out.append("Proposal strategies sigma_i(s, .)  (non-zero entries)")
            for i in range(geom.N):
                out.append(f"  proposer {geom.players[i]}:")
                for s in range(geom.S):
                    parts = [f"{self.sigma[i,s,t]:.4f} -> {L[t]}"
                             for t in range(geom.S) if self.sigma[i, s, t] > 1e-9]
                    tag = "  (stay)" if self.sigma[i, s, s] > 1 - 1e-9 else ""
                    out.append(f"    at {L[s].ljust(w)} : " + ";  ".join(parts) + tag)
            out.append("")
            out.append("Acceptance strategies alpha_j(s,s') at pairs where j votes")
            for j in range(geom.N):
                lines = []
                for s in range(geom.S):
                    for t in range(geom.S):
                        if geom.relevant[j, s, t]:
                            lines.append(f"    {L[s].ljust(w)} -> {L[t].ljust(w)}"
                                         f" : {self.alpha[j,s,t]:.6f}")
                if lines:
                    out.append(f"  voter {geom.players[j]}:")
                    out.extend(lines)
        out.append("")
        out.append(self.report.summary())
        return "\n".join(out)


# =============================================================================
# 11.  Top-level solver
# =============================================================================


@dataclass
class SolveResult:
    game: Game
    status: str                 # PURE | MIXED | MIXED_NONUNIQUE | NO_SMPE_FOUND
    solution: Optional[Solution]
    solutions: List[Solution]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.solution is not None

    def describe(self, show_strategies: bool = True) -> str:
        if self.solution is None:
            return failure_report(self)
        s = self.solution.describe(show_strategies=show_strategies)
        extra = []
        n = len(self.solutions)
        if n > 1:
            extra.append(f"\n{n} distinct verified equilibria found; the one above "
                         f"is the canonical branch. Value functions of the others:")
            L = self.game.geom.labels()
            for k, alt in enumerate(self.solutions[1:], start=1):
                extra.append(f"  [{k}] {alt.status} via {alt.route}")
                for st in range(self.game.S):
                    extra.append("      " + L[st].ljust(max(len(x) for x in L))
                                 + "  " + "".join(f"{alt.V_raw[st,i]:12.6f}"
                                                  for i in range(self.game.N)))
        if "delta_path" in self.diagnostics and self.diagnostics["delta_path"]:
            extra.append("\nRegime map along the delta-homotopy:")
            last = None
            for (d, stat, nmix) in self.diagnostics["delta_path"]:
                tag = (stat, nmix)
                if tag != last:
                    extra.append(f"  delta ~ {d:.4f}: {stat}"
                                 + (f" ({nmix} mixing decisions)" if nmix else ""))
                    last = tag
        return s + "\n" + "\n".join(extra)


def _classify(game: Game, sigma: np.ndarray, alpha: np.ndarray,
              rank_deficient: bool) -> str:
    pure_sigma = bool((sigma.max(axis=2) > 1 - 1e-9).all())
    rel = game.geom.relevant
    a = alpha[rel]
    pure_alpha = bool(np.all((a < 1e-9) | (a > 1 - 1e-9))) if a.size else True
    if pure_sigma and pure_alpha:
        return "PURE"
    return "MIXED_NONUNIQUE" if rank_deficient else "MIXED"


def _finalize(game: Game, sigma: np.ndarray, alpha: np.ndarray, route: str,
              regime: Optional[Regime], *, verify_tol: float,
              rank_deficient: bool = False,
              diagnostics: Optional[Dict] = None) -> Optional[Solution]:
    sigma = np.asarray(sigma, dtype=float).copy()
    alpha = np.asarray(alpha, dtype=float).copy()
    sigma = np.clip(sigma, 0.0, None)
    ssum = sigma.sum(axis=2, keepdims=True)
    sigma = sigma / np.where(ssum > 0, ssum, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    rep = verify(game, sigma, alpha, tol=verify_tol, strict_offpath=True)
    if not rep.ok:
        return None
    # pin the (payoff-irrelevant) non-voting entries of alpha by the cutoff rule
    g = gain_matrix(rep.V_induced)
    tidy = np.where(g > 0, 1.0, np.where(g < 0, 0.0, alpha))
    alpha = np.where(game.geom.relevant, alpha, tidy)

    ch = chain_analysis(rep.T)
    return Solution(game=game, sigma=sigma, alpha=alpha, T=rep.T,
                    V=rep.V_induced, V_raw=game.to_raw(rep.V_induced),
                    status=_classify(game, sigma, alpha, rank_deficient),
                    report=rep, chain=ch, route=route, regime=regime,
                    diagnostics=diagnostics or {})


def _cycle_hint(regime: Regime, cycle: List[PurePolicy]) -> np.ndarray:
    alphas = np.stack([p.alpha for p in cycle])
    h: List[float] = []
    for (j, s, t) in regime.accept_vars:
        h.append(float(alphas[:, j, s, t].mean()))
    for (i, s, sup) in regime.prop_vars:
        h.extend([1.0 / len(sup)] * (len(sup) - 1))
    return np.array(h, dtype=float)


def _proj_simplex(Y: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean projection onto the probability simplex."""
    S = Y.shape[-1]
    U = -np.sort(-Y, axis=-1)
    css = np.cumsum(U, axis=-1) - 1.0
    ind = np.arange(1, S + 1)
    cond = U - css / ind > 0
    rho = S - 1 - np.argmax(cond[..., ::-1], axis=-1)
    theta = np.take_along_axis(css, rho[..., None], axis=-1) / (rho[..., None] + 1)
    return np.maximum(Y - theta, 0.0)


def projected_dynamics(game: Game, *, iters: int = 6000, c0: float = 0.5,
                       power: float = 0.55):
    """Projected better-response dynamics with a vanishing step.

    alpha <- clip(alpha + c * gain, 0, 1);  sigma <- proj_simplex(sigma + c * G).

    This is a *regime identifier*, not a solver: because the update is
    continuous it does not lock into the pure-strategy oscillation that traps
    policy iteration, so the decisions that end up strictly interior are
    exactly the ones that need to mix.  The probabilities it returns are only
    approximate and are never used directly -- they seed an exact solve.
    """
    geom = game.geom
    N, S = game.N, game.S
    alpha = np.full((N, S, S), 0.5)
    sigma = np.full((N, S, S), 1.0 / S)
    a_acc = np.zeros_like(alpha)
    s_acc = np.zeros_like(sigma)
    n = 0
    start = iters // 2
    V = game.payoffs.copy()
    for k in range(iters):
        q = compute_q(alpha, geom)
        T = build_T(sigma, q, game.rho)
        V = evaluate(T, game.payoffs, game.delta)
        g = gain_matrix(V)
        G = q * g
        sc = max(float(np.abs(g).max()), 1e-12)
        c = c0 / (1 + k) ** power
        alpha = np.clip(alpha + c * g / sc, 0.0, 1.0)
        sigma = _proj_simplex(sigma + c * G / sc)
        if k >= start:
            a_acc += alpha
            s_acc += sigma
            n += 1
    return a_acc / n, s_acc / n, V


def _effective_options(geom: CoalitionGeometry, q: np.ndarray, i: int, s: int,
                       qtol: float = 1e-9) -> List[int]:
    """Proposals that are not transition-equivalent to staying put."""
    return [s] + [t for t in range(geom.S) if t != s and q[i, s, t] > qtol]


def regime_with_mixing(game: Game, V: np.ndarray, cands: Sequence[Tuple],
                       tol: float = 1e-11) -> Optional[Regime]:
    """Regime in which `cands` mix and everything else best-responds to V."""
    geom = game.geom
    pol = pure_best_response(V, geom, tie_accept=True, tie_break="stay", tol=tol)
    acc: List[Tuple[int, int, int]] = []
    props: List[Tuple[int, int, Tuple[int, ...]]] = []
    for c in cands:
        if c[0] == "acc":
            _, j, s, t = c
            # both directions of the same indifference are separate variables
            if geom.relevant[j, s, t]:
                acc.append((j, s, t))
            if geom.relevant[j, t, s]:
                acc.append((j, t, s))
        else:
            _, i, s, sup = c
            if sup is None:
                opts = _effective_options(geom, pol.q, i, s)
                if len(opts) < 2:
                    continue
                G = np.array([pol.q[i, s, t] * (V[t, i] - V[s, i]) for t in opts])
                order = np.argsort(-G)
                sup = tuple(sorted([opts[order[0]], opts[order[1]]]))
            else:
                sup = tuple(sorted(sup))
            if len(sup) >= 2:
                props.append((i, s, sup))
    if not acc and not props:
        return None
    return Regime(acc, props, pol.alpha.copy(), pol.choice.copy(),
                  origin="constrained")


def _hint_from_dynamics(regime: Regime, abar: np.ndarray,
                        sbar: np.ndarray) -> np.ndarray:
    """Seed the indifference solve with the dynamics' own mixed strategies."""
    h: List[float] = []
    for (j, s, t) in regime.accept_vars:
        h.append(float(np.clip(abar[j, s, t], 0.02, 0.98)))
    for (i, s, sup) in regime.prop_vars:
        w = np.array([max(sbar[i, s, t], 1e-6) for t in sup])
        w = w / w.sum()
        h.extend(float(v) for v in w[:-1])
    return np.array(h, dtype=float)


def _constrained_pi(game: Game, cands: Sequence[Tuple], V0: np.ndarray,
                    rng: np.random.Generator, *, max_outer: int = 12,
                    n_random: int = 6, max_iter: int = 120, tol: float = 1e-9,
                    dyn: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    deadline: Optional[float] = None):
    """Policy iteration with a fixed set of decisions held mixed.

    Each round: every *non*-mixing decision is set by pure best response to the
    current V, then the mixing probabilities are solved exactly from the
    indifference system.  Iterating resolves the chicken-and-egg between the
    mixing probabilities and the surrounding pure decisions.
    """
    V = np.array(V0, dtype=float, copy=True)
    xprev = None
    lastkey = None
    last_viols: List[Tuple] = []
    best_resid = np.inf
    for _ in range(max_outer):
        if deadline is not None and time.time() > deadline:
            return None, best_resid, last_viols
        R = regime_with_mixing(game, V, cands)
        if R is None or R.k == 0:
            return None, best_resid, last_viols
        hints = [xprev] if (xprev is not None and len(xprev) == R.k) else []
        if dyn is not None:
            hints.append(_hint_from_dynamics(R, dyn[0], dyn[1]))
        res = solve_regime(game, R, x_hints=hints, rng=rng, n_random=n_random,
                           max_iter=max_iter, deadline=deadline)
        best_resid = min(best_resid, float(res.get("resid", np.inf)))
        if not res["ok"]:
            return None, best_resid, last_viols
        sysm, x = res["system"], res["x"]
        xprev = x
        st = sysm.state(x)
        ok, _bad, vkeys = validate_regime(game, R, st, tol=tol)
        if ok:
            return (st, R, res), 0.0, []
        key = R.key()
        if key == lastkey:
            return None, best_resid, vkeys
        lastkey = key
        last_viols = vkeys
        V = st["V"]
    return None, best_resid, last_viols


def _candidates(game: Game, pi_res: PIResult, abar: np.ndarray,
                sbar: np.ndarray) -> List[Tuple]:
    """Decisions that might need to mix, scored by how much evidence there is."""
    geom = game.geom
    score: Dict[Tuple, int] = {}
    if len(pi_res.policies) > 1:
        al = np.stack([p.alpha for p in pi_res.policies])
        ch = np.stack([p.choice for p in pi_res.policies])
        flip = (al.max(0) != al.min(0)) & geom.relevant
        for j, s, t in zip(*np.nonzero(flip)):
            key = ("acc", int(j), int(min(s, t)), int(max(s, t)))
            score[key] = score.get(key, 0) + 2
        for i, s in zip(*np.nonzero(ch.max(0) != ch.min(0))):
            sup = tuple(sorted(set(int(v) for v in ch[:, i, s])))
            score[("prop", int(i), int(s), sup)] = \
                score.get(("prop", int(i), int(s), sup), 0) + 2
    qd = compute_q(abar, geom)
    for j in range(game.N):
        for s in range(game.S):
            for t in range(game.S):
                if geom.relevant[j, s, t] and 5e-3 < abar[j, s, t] < 1 - 5e-3:
                    key = ("acc", j, min(s, t), max(s, t))
                    score[key] = score.get(key, 0) + 3
    for i in range(game.N):
        for s in range(game.S):
            opts = _effective_options(geom, qd, i, s)
            if sum(1 for t in opts if sbar[i, s, t] > 5e-3) > 1:
                key = ("prop", i, s, None)
                score[key] = score.get(key, 0) + 3
    return sorted(score, key=lambda k: (-score[k], str(k)))


def _cycle_candidates(game: Game, pi_res: PIResult) -> List[Tuple]:
    """Just the decisions that actually oscillate along the cycle."""
    geom = game.geom
    if len(pi_res.policies) <= 1:
        return []
    al = np.stack([p.alpha for p in pi_res.policies])
    ch = np.stack([p.choice for p in pi_res.policies])
    out: List[Tuple] = []
    seen = set()
    flip = (al.max(0) != al.min(0)) & geom.relevant
    for j, st, t in zip(*np.nonzero(flip)):
        key = ("acc", int(j), int(min(st, t)), int(max(st, t)))
        if key not in seen:
            seen.add(key)
            out.append(key)
    for i, st in zip(*np.nonzero(ch.max(0) != ch.min(0))):
        sup = tuple(sorted(set(int(v) for v in ch[:, i, st])))
        out.append(("prop", int(i), int(st), sup))
    return out


def _mixing_search(game: Game, pi_res: PIResult, route: str, *, verify_tol: float,
                   tie_tol: float, rng: np.random.Generator,
                   max_subset: int = 4, max_candidates: int = 24,
                   dyn_iters: Optional[int] = None,
                   time_budget: float = 90.0):
    """Resolve a policy-iteration cycle by finding which decisions mix.

    1. projected dynamics  -> which decisions are interior (regime candidates)
    2. subset enumeration  -> constrained policy iteration on each candidate set
    3. exact Newton solve  -> the mixing probabilities
    4. independent verify  -> against V_induced
    """
    attempts: List[Dict[str, Any]] = []
    deadline = time.time() + time_budget
    if dyn_iters is None:
        dyn_iters = 6000 + 800 * max(0, game.S - 5)
    abar, sbar, Vd = projected_dynamics(game, iters=dyn_iters)
    C = _candidates(game, pi_res, abar, sbar)[:max_candidates]
    V_cyc = np.mean(np.stack(pi_res.V_seq), axis=0)
    attempts.append(dict(origin="projected dynamics", k=len(C),
                         outcome=f"{len(C)} candidate mixing decisions",
                         decisions=_describe_candidates(game.geom, C)))
    if not C:
        return None, attempts

    starts = [Vd, V_cyc]
    cyc = _cycle_candidates(game, pi_res)
    shortlist: List[Tuple[float, Tuple, np.ndarray]] = []

    # pass 0: guided growth.  Seed the mixing set with one well-evidenced
    # candidate, solve, and let the decisions that fail validation say what to
    # add next.  This usually lands the right mixing set in a few rounds, and
    # only falls through to enumeration when it does not.
    seeds: List[List[Tuple]] = []
    if cyc:
        seeds.append(list(cyc))                       # everything that flips
        seeds.append([c for c in cyc if c[0] == "acc"])
        seeds.extend([c] for c in cyc[:6])
    seeds.append([])
    seeds.extend([c] for c in C[:8])
    for seed in seeds:
        if time.time() > deadline:
            break
        M = list(seed)
        for _round in range(7):
            if time.time() > deadline:
                break
            if not M:
                M = [C[0]]
            got, _resid, viols = _constrained_pi(
                game, M, Vd, rng, max_outer=8, n_random=4, max_iter=100,
                tol=max(tie_tol, 1e-9), dyn=(abar, sbar), deadline=deadline)
            if got is not None:
                st, R, res = got
                sol = _finalize(game, st["sigma"], st["alpha"], route + "/mix",
                                R, verify_tol=verify_tol,
                                rank_deficient=res.get("rank_deficient", False),
                                diagnostics=dict(x=res["x"],
                                                 residual=res["resid"],
                                                 mixing_set=list(M)))
                if sol is not None:
                    attempts.append(dict(
                        origin="guided growth of the mixing set", k=R.k,
                        outcome="verified SMPE",
                        decisions=R.describe(game.geom)))
                    return sol, attempts
            add = [v for v in viols if v not in M][:2]
            if not add:
                break
            M = M + add


    # pass 1: cheap sweep.  Small mixing sets are searched over all candidates;
    # larger ones only over the best-evidenced candidates, since the number of
    # subsets grows fast and high-scoring decisions are far likelier to mix.
    # Two orderings of the candidate list are searched, interleaved by subset
    # size: one ranked by the dynamics, one that puts the decisions which
    # actually oscillate along the cycle first.  Neither ordering dominates --
    # the dynamics are the better guide when they converge (near-flat games),
    # the cycle is the better guide when they do not (larger state spaces) --
    # and interleaving by size means a small mixing set is found early under
    # either.
    orderings = [C]
    if cyc:
        C_cyc = list(cyc) + [c for c in C if c not in cyc]
        C_cyc = C_cyc[:max_candidates]
        if C_cyc != C:
            orderings.append(C_cyc)
    tried: set = set()
    for size in range(1, min(max_subset, len(C)) + 1):
        pool_n = {1: len(C), 2: len(C), 3: 20, 4: 10}.get(size, 8)
        for order in orderings:
            if time.time() > deadline:
                break
            for sub in itertools.combinations(order[:min(pool_n, len(order))],
                                              size):
                if time.time() > deadline:
                    break
                key = frozenset(sub)
                if key in tried:
                    continue
                tried.add(key)
                got, resid, _vk = _constrained_pi(
                    game, list(sub), Vd, rng, max_outer=5, n_random=1,
                    max_iter=50, tol=max(tie_tol, 1e-9), dyn=(abar, sbar))
                if got is not None:
                    st, R, res = got
                    sol = _finalize(game, st["sigma"], st["alpha"],
                                    route + "/mix", R, verify_tol=verify_tol,
                                    rank_deficient=res.get("rank_deficient",
                                                           False),
                                    diagnostics=dict(x=res["x"],
                                                     residual=res["resid"],
                                                     mixing_set=list(sub)))
                    if sol is not None:
                        attempts.append(dict(
                            origin="constrained policy iteration", k=R.k,
                            outcome="verified SMPE",
                            decisions=R.describe(game.geom)))
                        return sol, attempts
                elif np.isfinite(resid):
                    shortlist.append((resid, sub, Vd))
                    shortlist.append((resid, sub, V_cyc))

    # pass 2: full budget on the most promising subsets
    shortlist.sort(key=lambda z: z[0])
    for resid, sub, V0 in shortlist[:30]:
        if time.time() > deadline:
            break
        got, _, _vk = _constrained_pi(game, list(sub), V0, rng, max_outer=12,
                                      n_random=25, max_iter=140,
                                      tol=max(tie_tol, 1e-9), dyn=(abar, sbar),
                                      deadline=deadline)
        if got is None:
            continue
        st, R, res = got
        sol = _finalize(game, st["sigma"], st["alpha"], route + "/mix", R,
                        verify_tol=verify_tol,
                        rank_deficient=res.get("rank_deficient", False),
                        diagnostics=dict(x=res["x"], residual=res["resid"],
                                         mixing_set=list(sub)))
        if sol is not None:
            attempts.append(dict(origin="constrained policy iteration (retry)",
                                 k=R.k, outcome="verified SMPE",
                                 decisions=R.describe(game.geom)))
            return sol, attempts

    # last resort: let everything in the candidate set mix at once
    for V0 in starts + [game.payoffs.copy()]:
        got, _, _vk = _constrained_pi(game, C, V0, rng, max_outer=12, n_random=30,
                                      max_iter=140, tol=max(tie_tol, 1e-9),
                                      dyn=(abar, sbar), deadline=deadline)
        if got is None:
            continue
        st, R, res = got
        sol = _finalize(game, st["sigma"], st["alpha"], route + "/mix", R,
                        verify_tol=verify_tol,
                        rank_deficient=res.get("rank_deficient", False),
                        diagnostics=dict(x=res["x"], residual=res["resid"],
                                         mixing_set=list(C)))
        if sol is not None:
            attempts.append(dict(origin="all candidates mixing", k=R.k,
                                 outcome="verified SMPE",
                                 decisions=R.describe(game.geom)))
            return sol, attempts

    attempts.append(dict(
        origin="subset enumeration", k=len(C),
        outcome=(f"no verified SMPE from any mixing set of size <= "
                 f"{min(max_subset, len(C))} drawn from {len(C)} candidates"
                 + ("; SEARCH WAS CUT SHORT BY THE TIME BUDGET"
                    if time.time() > deadline else "")),
        decisions=_describe_candidates(game.geom, C)))
    return None, attempts


def _describe_candidates(geom: CoalitionGeometry, C: Sequence[Tuple]) -> List[str]:
    out = []
    for c in C:
        if c[0] == "acc":
            _, j, s, t = c
            out.append(f"  accept  {geom.players[j]:>4s} between "
                       f"{geom.label(s)} and {geom.label(t)}")
        else:
            _, i, s, sup = c
            tail = ("" if sup is None else
                    " over [" + " / ".join(geom.label(a) for a in sup) + "]")
            out.append(f"  propose {geom.players[i]:>4s} at {geom.label(s)}{tail}")
    return out


def _attempt(game: Game, V0: np.ndarray, *, tie_accept: bool, tie_break: str,
             route: str, verify_tol: float, tie_tol: float,
             rng: np.random.Generator, allow_mixing: bool = True,
             time_budget: float = 90.0):
    pi_res = policy_iteration(game, V0, tie_accept=tie_accept,
                              tie_break=tie_break, rng=rng)
    diag: Dict[str, Any] = dict(pi_status=pi_res.status,
                                pi_iterations=pi_res.iterations,
                                cycle_period=len(pi_res.policies))
    if pi_res.status == "fixed":
        p = pi_res.policies[0]
        sol = _finalize(game, p.sigma, p.alpha, route + "/pure", None,
                        verify_tol=verify_tol)
        if sol is not None:
            sol.diagnostics.update(diag)
            return sol, diag, pi_res
        diag["pure_fixed_point_failed_verification"] = True
    if allow_mixing and pi_res.status != "fixed":
        sol, attempts = _mixing_search(game, pi_res, route,
                                       verify_tol=verify_tol, tie_tol=tie_tol,
                                       rng=rng, time_budget=time_budget)
        diag["mixing_attempts"] = attempts
        if sol is not None:
            sol.diagnostics.update(diag)
            return sol, diag, pi_res
    return None, diag, pi_res


def _homotopy(game: Game, *, verify_tol: float, tie_tol: float,
              rng: np.random.Generator, d_start: float = 0.01,
              step0: float = 0.05, min_step: float = 5e-5):
    """Track a branch of equilibria in delta from ~0 up to the target."""
    d_t = game.delta
    d = min(d_start, 0.5 * d_t)
    V = game.payoffs.copy()
    path: List[Tuple[float, str, int]] = []
    step = step0
    sol = None
    guard = 0
    while guard < 4000:
        guard += 1
        g = game.with_delta(d)
        s, _diag, _pi = _attempt(g, V, tie_accept=True, tie_break="stay",
                                 route=f"homotopy(delta={d:.5f})",
                                 verify_tol=verify_tol, tie_tol=tie_tol, rng=rng)
        if s is None:
            if step <= min_step:
                return None, path, dict(failed_at_delta=d, step=step)
            d = d - step + step / 2.0
            step = step / 2.0
            if d <= 0.0:
                return None, path, dict(failed_at_delta=d, step=step)
            continue
        nmix = 0 if s.regime is None else s.regime.k
        path.append((d, s.status, nmix))
        V = s.V
        sol = s
        if d >= d_t - 1e-15:
            return sol, path, dict()
        step = min(step * 1.6, step0)
        d = min(d_t, d + step)
    return None, path, dict(reason="homotopy step guard exceeded")


def solve_smpe(players: Sequence[str], payoffs, delta: float, *, rho=None,
               rows=None, rescale: bool = True, homotopy: bool = False,
               multistart: int = 12, seed: int = 0, verify_tol: float = 1e-7,
               tie_tol: float = 1e-8, failure_multistart: int = 400,
               collect_all: bool = True, verbose: bool = False,
               time_budget: float = 90.0, committees=None) -> SolveResult:
    """Find and verify an SMPE of the coalition-formation game.

    Parameters
    ----------
    players   : list of N player names.
    payoffs   : (B_N, N) flow payoff matrix.
    delta     : discount factor in (0,1).
    rho       : proposer probabilities (default uniform).
    time_budget : seconds allowed for each mixing search before it gives up and
                reports honestly that the search was truncated.
    rows      : optional list of partition labels giving the row order of
                `payoffs` (e.g. ["{A,B,C}", "{A,C}+{B}", ...]).  If omitted the
                canonical restricted-growth order is assumed.
    rescale   : per-player affine rescaling of payoffs (exactly SMPE-invariant).
    """
    t0 = time.time()
    game = Game.build(players, payoffs, delta, rho=rho, rows=rows, rescale=rescale,
                      committees=committees)
    rng = np.random.default_rng(seed)
    sols: List[Solution] = []
    seen_keys = set()
    diag: Dict[str, Any] = dict(routes=[])

    def add(s: Optional[Solution]):
        if s is None:
            return False
        k = s.dedup_key()
        if k in seen_keys:
            return False
        seen_keys.add(k)
        sols.append(s)
        return True

    # --- canonical direct attempts ---------------------------------------
    canonical = None
    for idx, (ta, tb) in enumerate([(True, "stay"), (False, "stay"),
                                    (True, "first"), (True, "last"),
                                    (False, "first")]):
        route = f"direct(tie_accept={ta},tie_break={tb})"
        # the mixing search is the expensive part; the alternative tie-breaks
        # are run only to pick up *additional pure* equilibria cheaply.
        s, d, _ = _attempt(game, game.payoffs.copy(), tie_accept=ta, tie_break=tb,
                           route=route, verify_tol=verify_tol, tie_tol=tie_tol,
                           rng=rng, allow_mixing=(idx == 0 or canonical is None),
                           time_budget=time_budget)
        diag["routes"].append((route, d))
        if add(s) and canonical is None:
            canonical = s
        if s is not None and not collect_all:
            break

    # --- delta-homotopy ---------------------------------------------------
    if homotopy and (collect_all or canonical is None):
        hs, path, hdiag = _homotopy(game, verify_tol=verify_tol, tie_tol=tie_tol,
                                    rng=rng)
        diag["delta_path"] = path
        diag["homotopy"] = hdiag
        if add(hs) and canonical is None:
            canonical = hs

    # --- random multistart ------------------------------------------------
    if collect_all or canonical is None:
        lo = game.payoffs.min(axis=0)
        hi = game.payoffs.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        n = multistart if canonical is not None else max(multistart,
                                                         failure_multistart)
        for it in range(n):
            V0 = lo[None, :] + span[None, :] * rng.uniform(-0.25, 1.25,
                                                           size=game.payoffs.shape)
            tb = rng.choice(["stay", "first", "last", "random"])
            ta = bool(rng.integers(0, 2))
            s, d, pi = _attempt(game, V0, tie_accept=ta, tie_break=str(tb),
                                route=f"multistart#{it}", verify_tol=verify_tol,
                                tie_tol=tie_tol, rng=rng,
                                allow_mixing=(canonical is None and it < 2),
                                time_budget=time_budget)
            if add(s) and canonical is None:
                canonical = s
            diag.setdefault("multistart_status", []).append(d["pi_status"])

    diag["elapsed_sec"] = time.time() - t0
    if canonical is None:
        diag.update(_failure_diagnostics(game, rng, verify_tol=verify_tol,
                                         tie_tol=tie_tol,
                                         n=max(64, failure_multistart // 4)))
        return SolveResult(game=game, status="NO_SMPE_FOUND", solution=None,
                           solutions=[], diagnostics=diag)
    # canonical first
    sols = [canonical] + [s for s in sols if s is not canonical]
    return SolveResult(game=game, status=canonical.status, solution=canonical,
                       solutions=sols, diagnostics=diag)


# =============================================================================
# 12.  Failure diagnostics
# =============================================================================


def _failure_diagnostics(game: Game, rng, *, verify_tol: float, tie_tol: float,
                         n: int = 100) -> Dict[str, Any]:
    geom = game.geom
    lo, hi = game.payoffs.min(axis=0), game.payoffs.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    periods: Dict[int, int] = {}
    example = None
    for it in range(n):
        V0 = (game.payoffs.copy() if it == 0 else
              lo[None, :] + span[None, :] * rng.uniform(-0.25, 1.25,
                                                        size=game.payoffs.shape))
        pi = policy_iteration(game, V0, tie_accept=bool(rng.integers(0, 2)),
                              tie_break=str(rng.choice(["stay", "first", "last"])),
                              rng=rng)
        p = len(pi.policies) if pi.status == "cycle" else (
            1 if pi.status == "fixed" else -1)
        periods[p] = periods.get(p, 0) + 1
        if example is None and pi.status == "cycle":
            example = pi

    out: Dict[str, Any] = dict(restart_cycle_periods=periods, restarts=n)
    if example is not None:
        R = regime_from_cycle(example.policies, geom)
        Vs = np.stack(example.V_seq)
        out["cycle_period"] = len(example.policies)
        out["cycle_amplitude"] = float(np.abs(Vs.max(0) - Vs.min(0)).max())
        out["oscillating_decisions"] = R.describe(geom)
        out["n_mixing_unknowns"] = R.k
        stab = {}
        for m, V in enumerate(example.V_seq):
            es = exit_stability(game, V)
            stab[m] = {geom.label(s): [f"{p} (+{g:.3e})" for p, g in v]
                       for s, v in es.items() if v}
        out["exit_stability_along_cycle"] = stab
        out["states_stable_under_exit"] = {
            m: [geom.label(s) for s, v in exit_stability(game, V).items() if not v]
            for m, V in enumerate(example.V_seq)}
    return out


def failure_report(res: SolveResult) -> str:
    d = res.diagnostics
    geom = res.game.geom
    out = [f"NO SMPE FOUND  (delta = {res.game.delta}, N = {geom.N}, "
           f"{geom.S} states)", ""]
    out.append("This is a report of search failure, NOT a proof of non-existence.")
    out.append("The search covers SMPE in *independent* mixed strategies with")
    out.append("proposer-independent acceptance.  Correlated strategies, public")
    out.append("randomisation and proposer-dependent acceptance are outside it.")
    out.append("")
    if "restart_cycle_periods" in d:
        pp = ", ".join(f"period {k if k>0 else '?'}: {v}"
                       for k, v in sorted(d["restart_cycle_periods"].items()))
        out.append(f"Policy iteration from {d.get('restarts','?')} randomised "
                   f"restarts (random V0, tie-breaks, proposer orderings):")
        out.append(f"  outcome distribution -> {pp}")
        out.append("  (period 1 would mean a pure fixed point; none verified)")
    if "cycle_amplitude" in d:
        out.append("")
        out.append(f"Representative cycle: period {d['cycle_period']}, "
                   f"max amplitude in V of {d['cycle_amplitude']:.4e} "
                   f"(rescaled units, payoff spread 1.0)")
        out.append(f"Oscillating decisions ({d['n_mixing_unknowns']} mixing "
                   f"unknowns):")
        out.extend(d["oscillating_decisions"])
    if "states_stable_under_exit" in d:
        out.append("")
        out.append("Exit stability along the cycle (states where no player")
        out.append("strictly gains by unilateral exit):")
        for m, sts in d["states_stable_under_exit"].items():
            out.append(f"  cycle phase {m}: "
                       + (", ".join(sts) if sts else "NONE"))
        if all(not v for v in d["states_stable_under_exit"].values()):
            out.append("  -> no coalition structure is stable under unilateral")
            out.append("     exit at any phase; the free-exit rule alone rules out")
            out.append("     every absorbing state, which is the structural reason")
            out.append("     the value function cannot settle.")
    tried = []
    for route, rd in d.get("routes", []):
        for a in rd.get("mixing_attempts", []) or []:
            tried.append((route, a))
    if tried:
        out.append("")
        out.append("Mixing systems attempted:")
        for route, a in tried[:8]:
            out.append(f"  [{a['origin']}] {a['k']} unknown(s) -> {a['outcome']}")
            if a.get("reason"):
                out.append(f"      {a['reason']}")
            for v in a.get("violations", [])[:4]:
                out.append(f"      violation: {v}")
            for line in a["decisions"][:6]:
                out.append("    " + line)
    out.append("")
    out.append("Conclusion: no SMPE with independent mixing was found by exact")
    out.append("policy iteration, Newton solution of the indifference systems,")
    out.append("delta-homotopy, or randomised multistart.  A non-existence")
    out.append("*certificate* would require exhausting the regime space, which")
    out.append("is not tractable here, and is not claimed.")
    return "\n".join(out)


# =============================================================================
# 13.  Convenience: scan over delta
# =============================================================================


def solve_with_mixing_set(players, payoffs, delta, mixing_set, *, rho=None,
                          rows=None, rescale=True, v_starts=None, seed=0,
                          verify_tol=1e-7, n_random=15, dyn_iters=200000,
                          dyn_c0=1.0, dyn_power=0.5) -> Optional[Solution]:
    """Solve with the set of mixing decisions supplied explicitly.

    `solve_smpe` identifies the mixing set heuristically and can miss large
    ones -- in particular *indifference classes*, where one player's value is
    equal across three or more states at once, which need several acceptance
    pairs plus that player's proposals all mixing together.  When the automated
    search fails but the structure is known (from `projected_dynamics`, or from
    a bifurcation analysis in delta), this solves for the mixing probabilities
    directly and verifies the result the same way.

    `mixing_set` is a list of decisions, each either

        ("acc",  voter_index, state_a, state_b)   -- j indifferent between them
        ("prop", proposer_index, state, None)     -- i mixes its proposal there

    State indices are into `rows` when `rows` is given, else into the solver's
    canonical order.  Returns a verified `Solution`, or None.

    Worked example -- Example 4 at delta = 0.87, where A is indifferent across
    {A,B,C}, {A,C}+{B} and {A}+{B,C} simultaneously::

        A, C = 0, 2
        GRAND, AC_B, A_BC = 0, 1, 2          # indices into ROWS
        sol = solve_with_mixing_set(
            ["A","B","C"], EX4, 0.87, rows=ROWS, mixing_set=[
                ("acc",  A, GRAND, AC_B), ("acc",  A, GRAND, A_BC),
                ("acc",  A, AC_B,  A_BC),
                ("prop", A, GRAND, None), ("prop", A, AC_B, None),
                ("prop", A, A_BC,  None),
                ("prop", C, AC_B,  None), ("prop", C, A_BC, None)])
    """
    game = Game.build(players, payoffs, delta, rho=rho, rows=rows,
                      rescale=rescale)
    if rows is not None:
        perm = game.geom.permutation_from(rows)
        back = {int(perm[s]): s for s in range(game.S)}
        conv = []
        for c in mixing_set:
            if c[0] == "acc":
                conv.append(("acc", c[1], back[c[2]], back[c[3]]))
            else:
                conv.append(("prop", c[1], back[c[2]], None))
        mixing_set = conv
    rng = np.random.default_rng(seed)
    abar, sbar, Vd = projected_dynamics(game, iters=dyn_iters, c0=dyn_c0,
                                        power=dyn_power)
    pi = policy_iteration(game, game.payoffs.copy())
    starts = list(v_starts) if v_starts is not None else (
        [Vd, np.mean(np.stack(pi.V_seq), axis=0)] + list(pi.V_seq)
        + [game.payoffs.copy()])
    for V0 in starts:
        got, _r, _v = _constrained_pi(game, list(mixing_set), V0, rng,
                                      max_outer=14, n_random=n_random,
                                      max_iter=160, dyn=(abar, sbar))
        if got is None:
            continue
        st, R, res = got
        sol = _finalize(game, st["sigma"], st["alpha"], "manual mixing set", R,
                        verify_tol=verify_tol,
                        rank_deficient=res.get("rank_deficient", False),
                        diagnostics=dict(x=res["x"], residual=res["resid"],
                                         mixing_set=list(mixing_set)))
        if sol is not None:
            return sol
    return None


def delta_scan(players, payoffs, deltas, *, rows=None, rho=None, rescale=True,
               **kw) -> List[Tuple[float, SolveResult]]:
    return [(d, solve_smpe(players, payoffs, d, rows=rows, rho=rho,
                           rescale=rescale, **kw)) for d in deltas]
