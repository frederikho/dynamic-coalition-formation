"""
Markov equilibrium as a mixed complementarity problem.

Every previous attempt in this repo decomposed the equilibrium conditions --
froze a baseline profile, isolated one player's indifference, guessed a mixed
support -- and each decomposition failed for the same reason: the mixing
probabilities are not determined by any one condition in isolation.  On a control
game whose answer is known, the indifference gap of the mixing player is
machine-zero across a whole interval of theta while only one theta is an
equilibrium; what selects it are *other* players' conditions.

So solve all of them at once.  The equilibrium is exactly a complementarity
system:

  ACCEPTANCE, per voter j and transition x -> y, with g = V_j(y) - V_j(x):
      alpha = 0        =>  g <= 0     (reject)
      0 < alpha < 1    =>  g == 0     (indifferent -- THIS is mixing)
      alpha = 1        =>  g >= 0     (accept)

  PROPOSALS, per proposer i in state x, with gain h(y) = q_i(x->y) * (V_i(y) - V_i(x))
  and a multiplier lambda:
      0 <= sigma(y)  _|_  (lambda - h(y)) >= 0,     sum_y sigma(y) = 1

  BELLMAN:
      V = (1 - delta) u + delta T(sigma, alpha) V

Mixing is not a special case to be detected here; it is simply what a variable
does when its bound is not active.  And sigma lives on a simplex, so MIXED
PROPOSALS are representable -- the per-state MIP cannot express them at all, since
its constraints force exactly one proposal target.

The complementarity conditions are turned into equations with the natural
residual (min/mid map) and the square system is solved with least_squares.  A
solution is only ever accepted after the framework's own verifier passes; the
solver's residual is never the acceptance test.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from lib.equilibrium.jeres_vfi.game import Game, voters


def _committee(game: Game, x: int, y: int, i: int) -> frozenset:
    if game.approval_committees is not None:
        return game.approval_committees.get((i, x, y), frozenset())
    return voters(game, game.states[x], game.states[y], i)


class _Layout:
    """Index bookkeeping for the packed unknown vector."""

    def __init__(self, game: Game):
        self.game = game
        n_s, n_p = game.n_states, game.n_players
        forbidden = game.forbidden_transitions or frozenset()

        self.sigma_keys = [
            (i, x, y)
            for i in range(n_p)
            for x in range(n_s)
            for y in range(n_s)
            if y == x or (i, x, y) not in forbidden
        ]
        alpha_keys = set()
        self.comm = {}
        for (i, x, y) in self.sigma_keys:
            if y == x:
                self.comm[(i, x, y)] = frozenset()
                continue
            c = _committee(game, x, y, i)
            self.comm[(i, x, y)] = c
            for j in c:
                alpha_keys.add((x, j, y))
        self.alpha_keys = sorted(alpha_keys)
        self.lam_keys = [(i, x) for i in range(n_p) for x in range(n_s)]

        self.nV = n_s * n_p
        self.nA = len(self.alpha_keys)
        self.nS = len(self.sigma_keys)
        self.nL = len(self.lam_keys)
        self.n = self.nV + self.nA + self.nS + self.nL

        self.a_of = {k: m for m, k in enumerate(self.alpha_keys)}
        self.s_of = {k: m for m, k in enumerate(self.sigma_keys)}
        self.l_of = {k: m for m, k in enumerate(self.lam_keys)}

    def unpack(self, z):
        n_s, n_p = self.game.n_states, self.game.n_players
        V = z[: self.nV].reshape(n_s, n_p)
        a = z[self.nV: self.nV + self.nA]
        s = z[self.nV + self.nA: self.nV + self.nA + self.nS]
        lam = z[self.nV + self.nA + self.nS:]
        return V, a, s, lam

    def pack(self, V, a, s, lam):
        return np.concatenate([V.ravel(), a, s, lam])


def _transition(lay: _Layout, a, s, rho):
    """T from the packed strategy variables, with rejected proposals staying put."""
    game = lay.game
    n_s = game.n_states
    T = np.zeros((n_s, n_s))
    q = np.empty(lay.nS)
    for m, (i, x, y) in enumerate(lay.sigma_keys):
        qi = 1.0
        for j in lay.comm[(i, x, y)]:
            qi *= a[lay.a_of[(x, j, y)]]
        q[m] = qi
        sig = s[m]
        if y == x:
            T[x, x] += rho[i] * sig
        else:
            # A proposal that is not approved leaves the state where it was.
            T[x, y] += rho[i] * sig * qi
            T[x, x] += rho[i] * sig * (1.0 - qi)
    return T, q


def _fb(a, b, mu):
    """
    Smoothed Fischer-Burmeister:  phi(a,b) = sqrt(a^2 + b^2 + 2 mu^2) - a - b.

    phi(a,b) = 0 <=> a >= 0, b >= 0, a*b = 0 in the limit mu -> 0.  The smoothing
    matters: the natural residual min(a,b) encodes the same conditions but is
    kinked, and least_squares assumes smoothness, so it stalls on the kinks --
    measured at |r| = 1.1e-01 on the control, nowhere near a solution.  Solving a
    sequence of smoothed problems with mu -> 0, each warm-started from the last,
    keeps every subproblem differentiable.
    """
    return np.sqrt(a * a + b * b + 2.0 * mu * mu) - a - b


def _fb_box(a, F, lo, hi, mu):
    """Two-sided FB for a variable in [lo, hi] (Billups' reformulation)."""
    return _fb(a - lo, _fb(hi - a, -F, mu), mu)


def _residual(z, lay: _Layout, delta: float, rho, u, mu: float):
    game = lay.game
    n_s, n_p = game.n_states, game.n_players
    V, a, s, lam = lay.unpack(z)
    T, q = _transition(lay, a, s, rho)

    out = []

    # Bellman: V = (1-delta) u + delta T V
    out.append((V - (1.0 - delta) * u - delta * (T @ V)).ravel())

    # Acceptance: alpha in [0,1] complementary to F = -(V_j(y) - V_j(x)).
    # An interior alpha forces g == 0 -- that is mixing, and it needs no special
    # case here; it is simply what happens when neither bound is active.
    ra = np.empty(lay.nA)
    for m, (x, j, y) in enumerate(lay.alpha_keys):
        F = -(V[y, j] - V[x, j])
        ra[m] = _fb_box(a[m], F, 0.0, 1.0, mu)
    out.append(ra)

    # Proposals: 0 <= sigma _|_ (lambda - h) >= 0, on the simplex.
    rs = np.empty(lay.nS)
    for m, (i, x, y) in enumerate(lay.sigma_keys):
        h = 0.0 if y == x else q[m] * (V[y, i] - V[x, i])
        rs[m] = _fb(s[m], lam[lay.l_of[(i, x)]] - h, mu)
    out.append(rs)

    rl = np.empty(lay.nL)
    for m, (i, x) in enumerate(lay.lam_keys):
        tot = sum(s[lay.s_of[(i, x, y)]]
                  for y in range(n_s) if (i, x, y) in lay.s_of)
        rl[m] = tot - 1.0
    out.append(rl)

    return np.concatenate(out)


def _to_profile(lay: _Layout, z):
    """Pack the solution into the (sigmas, alphas, qs) dicts the verifier expects."""
    game = lay.game
    n_s = game.n_states
    V, a, s, lam = lay.unpack(z)
    _, q = _transition(lay, a, s, np.ones(game.n_players) / game.n_players)

    sigmas = [dict() for _ in range(n_s)]
    alphas = [dict() for _ in range(n_s)]
    qs = [dict() for _ in range(n_s)]
    for m, (i, x, y) in enumerate(lay.sigma_keys):
        sigmas[x][(i, y)] = float(np.clip(s[m], 0.0, 1.0))
        qs[x][(i, y)] = float(np.clip(q[m], 0.0, 1.0))
    for m, (x, j, y) in enumerate(lay.alpha_keys):
        alphas[x][(j, y)] = float(np.clip(a[m], 0.0, 1.0))

    # Renormalise each proposer's distribution; least_squares satisfies the simplex
    # to its own tolerance, and the verifier tests probabilities exactly.
    for x in range(n_s):
        for i in range(game.n_players):
            keys = [(i2, y) for (i2, y) in sigmas[x] if i2 == i]
            tot = sum(sigmas[x][k] for k in keys)
            if tot > 0:
                for k in keys:
                    sigmas[x][k] /= tot
    return V, sigmas, alphas, qs


def solve_mcp(game: Game, delta: float, proposer_probs=None, n_starts: int = 24,
              seed: int = 0, verify_atol: float = 1e-12, verbose: bool = False):
    """
    Solve the equilibrium complementarity system.

    Returns (V, sigmas, alphas, qs, info) with info['verified'] set by the
    framework's own conditions, or (None, ..., info) if nothing verified.

    Multi-start here is NOT the inert kind documented for VFI.  VFI converges to a
    unique fixed point, so its random restarts re-derive one answer; this system is
    nonsmooth and genuinely admits several solutions, so distinct starts can land on
    distinct equilibria.
    """
    from lib.equilibrium.jeres_vfi.solver import verify_proposals, verify_responses

    rho = (np.ones(game.n_players) / game.n_players
           if proposer_probs is None else np.asarray(proposer_probs, dtype=float))
    lay = _Layout(game)
    # Solve in per-player unit-range payoffs.  A positive affine map per player
    # leaves the equilibrium set exactly unchanged, and it keeps V, the Bellman
    # residual and the complementarity residuals on one scale, which a single
    # least_squares tolerance has to serve.
    raw = np.asarray(game.payoffs, dtype=float)
    span = raw.max(axis=0) - raw.min(axis=0)
    span = np.where(span > 0, span, 1.0)
    u = (raw - raw.min(axis=0)) / span
    game = Game(players=game.players, payoffs=u, states=game.states,
                state_idx=game.state_idx, n_players=game.n_players,
                n_states=game.n_states,
                approval_committees=game.approval_committees,
                forbidden_transitions=game.forbidden_transitions)
    lay.game = game
    rng = np.random.default_rng(seed)

    lo = np.concatenate([
        np.full(lay.nV, -np.inf), np.zeros(lay.nA), np.zeros(lay.nS),
        np.full(lay.nL, -np.inf)])
    hi = np.concatenate([
        np.full(lay.nV, np.inf), np.ones(lay.nA), np.ones(lay.nS),
        np.full(lay.nL, np.inf)])

    best = dict(verified=False, residual=np.inf, starts=n_starts, solved=0)
    for k in range(n_starts):
        V0 = u + (0.0 if k == 0 else rng.normal(0, 0.1, size=u.shape))
        a0 = (np.full(lay.nA, 0.5) if k == 0 else rng.uniform(0, 1, lay.nA))
        s0 = rng.uniform(0.2, 1.0, lay.nS) if k else np.full(lay.nS, 0.5)
        for i in range(game.n_players):
            for x in range(game.n_states):
                keys = [lay.s_of[(i, x, y)] for y in range(game.n_states)
                        if (i, x, y) in lay.s_of]
                s0[keys] /= s0[keys].sum()
        l0 = np.zeros(lay.nL)
        z0 = np.clip(lay.pack(V0, a0, s0, l0), lo, hi)

        # Homotopy: solve a sequence of smoothed problems, each warm-started from
        # the previous, driving the smoothing to zero.
        z = z0
        try:
            for mu in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 0.0):
                sol = least_squares(_residual, z, bounds=(lo, hi),
                                    args=(lay, delta, rho, u, mu),
                                    xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                    max_nfev=3000)
                z = sol.x
        except Exception:
            continue

        rnorm = float(np.max(np.abs(_residual(z, lay, delta, rho, u, 0.0))))
        if rnorm < best["residual"]:
            best["residual"] = rnorm
        if rnorm > 1e-9:
            continue
        best["solved"] += 1

        V, sigmas, alphas, qs = _to_profile(lay, z)
        r_ok, _ = verify_responses(game, sigmas, alphas, qs, V, atol=verify_atol)
        p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V, atol=verify_atol)
        if verbose:
            interior = [round(v, 6) for d in alphas for v in d.values()
                        if 1e-9 < v < 1 - 1e-9]
            print(f"  start {k:3d}: |r|={rnorm:.2e} responses={r_ok} "
                  f"proposals={p_ok} interior_alpha={interior}")
        if r_ok and p_ok:
            best["verified"] = True
            best["found_at_start"] = k
            return V, sigmas, alphas, qs, best

    return None, None, None, None, best
