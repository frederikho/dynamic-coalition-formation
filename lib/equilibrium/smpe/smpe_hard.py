"""smpe_hard.py -- QRE (logit) homotopy for SMPE in coalition-formation games.

Method A of the two-track plan.  `smpe.py` handles games where policy iteration
converges or the mixing set is small; this handles the rest.

Why a homotopy
--------------
The numerical half of the problem is solved: given a regime -- who mixes, over
what -- the exact Newton solve in `smpe.py` finds the probabilities to machine
precision.  Every failure is the combinatorial half, identifying the regime.
Searching supports does not scale: the benchmark's unsolved instances have a
median of 25 oscillating decisions, and an indifference class of size 3 for one
player already couples 6 of them.

A homotopy never commits to a support.  Smooth the best responses,

    alpha = logistic(lambda * (V_j(t) - V_j(s))),   sigma = softmax(lambda * G),

and track the fixed point of

    Phi_lambda(V) = (I - delta T(V,lambda))^{-1} (1-delta) pi

from lambda ~ 0 (where Phi is nearly constant and the fixed point is trivial)
up to large lambda.  Indifference classes of any size fall out for free.

The path exists: Phi_lambda is continuous and maps the compact convex box
prod_[min_s pi(s,i), max_s pi(s,i)] into itself, so Brouwer applies for every
lambda; as lambda -> infinity the sigmoid becomes the strict cutoff and the
softmax concentrates on the argmax, so limit points are SMPE.  (Fink/Takahashi
existence for finite discounted stochastic games.)

Two things sank my earlier attempts, and both are addressed here:

  * naive lambda-stepping dies at folds -- the path genuinely turns back in
    lambda.  Fixed by arclength continuation in (V, log lambda).

  * a finite-difference Jacobian is useless at large lambda: the sigmoid
    derivative lambda*a*(1-a) spans twenty orders of magnitude across
    components, saturated entries reading exactly zero while entries at a
    switch are O(lambda).  No single step size resolves both.  The Jacobian
    here is analytic.

And rather than tracking to lambda = 1e10, where the path stiffens and steps
collapse, the track hands off as soon as the interior/boundary pattern is
stable: the homotopy supplies the *structure*, and the exact solve in `smpe.py`
supplies exactness and verifiability.

References: Herings & Peeters, "Stationary equilibria in stochastic games:
structure, selection, and computation", JET 2004; Turocy's logit tracer in
Gambit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import smpe
from .smpe import (CoalitionGeometry, Game, Solution, build_T, compute_q,
                  evaluate, gain_matrix, _constrained_pi, _finalize)

__all__ = ["logit_state", "track", "regime_from_logit", "solve_hard",
           "check_jacobian"]


# ---------------------------------------------------------------------------
# smoothed best responses
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    p = x >= 0
    out[p] = 1.0 / (1.0 + np.exp(-x[p]))
    e = np.exp(x[~p])
    out[~p] = e / (1.0 + e)
    return out


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def logit_state(game: Game, V: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    """alpha, q, sigma, T and Phi(V) at precision lambda."""
    geom = game.geom
    g = gain_matrix(V)                       # g[j,s,t] = V[t,j] - V[s,j]
    alpha = _sigmoid(lam * g)
    q = compute_q(alpha, geom)
    G = q * g
    sigma = _softmax(lam * G, axis=2)
    T = build_T(sigma, q, game.rho)
    W = evaluate(T, game.payoffs, game.delta)
    return dict(g=g, alpha=alpha, q=q, G=G, sigma=sigma, T=T, W=W)


U_CLIP = (-20.0, 35.0)          # log lambda in [2e-9, 1.6e15]


def _lam(u: float) -> float:
    return float(np.exp(np.clip(u, *U_CLIP)))


def H(game: Game, y: np.ndarray) -> np.ndarray:
    """Residual V - Phi_lambda(V), with y = (V.ravel(), log lambda)."""
    S, N = game.S, game.N
    V = y[: S * N].reshape(S, N)
    st = logit_state(game, V, _lam(y[-1]))
    return (V - st["W"]).ravel()


# ---------------------------------------------------------------------------
# analytic Jacobian
# ---------------------------------------------------------------------------
def DH(game: Game, y: np.ndarray,
       st: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """d(V - Phi)/d(V, log lambda), shape (S*N, S*N + 1).

    Chain: alpha -> q -> G -> sigma -> T -> Phi, with
        dPhi/dx = delta (I - delta T)^{-1} (dT/dx) Phi.
    """
    geom = game.geom
    S, N = game.S, game.N
    n = S * N
    lam = _lam(y[-1])
    V = y[: n].reshape(S, N)
    if st is None:
        st = logit_state(game, V, lam)
    g, alpha, q, G, sigma, T, W = (st["g"], st["alpha"], st["q"], st["G"],
                                   st["sigma"], st["T"], st["W"])
    M = np.eye(S) - game.delta * T
    ap = alpha * (1.0 - alpha)                       # sigmoid derivative factor
    # partial products excluding one voter, for dq
    excl = [compute_q(alpha, geom, exclude_j=k) for k in range(N)]

    def dT_from(dsigma: np.ndarray, dq: np.ndarray) -> np.ndarray:
        doff = (np.einsum("i,ist,ist->st", game.rho, dsigma, q)
                + np.einsum("i,ist,ist->st", game.rho, sigma, dq))
        np.fill_diagonal(doff, 0.0)
        dT = doff.copy()
        np.fill_diagonal(dT, -doff.sum(axis=1))
        return dT

    J = np.zeros((n, n + 1))

    # --- columns for V ----------------------------------------------------
    for u in range(S):
        for m in range(N):
            col = u * N + m
            # d alpha[m,s,t] = lam * a(1-a) * ((t==u) - (s==u))
            dalpha = np.zeros_like(alpha)
            dalpha[m, :, u] += lam * ap[m, :, u]
            dalpha[m, u, :] -= lam * ap[m, u, :]
            dalpha[m, u, u] = 0.0
            # d q: only voter m varies
            dq = geom.voters[:, :, :, m].transpose(2, 0, 1) * excl[m] \
                * dalpha[m][None, :, :]
            # d G[i,s,t] = dq*g_i + q*dg_i, and g_i varies only for i == m
            dgi = np.zeros_like(g)
            dgi[m, :, u] += 1.0
            dgi[m, u, :] -= 1.0
            dgi[m, u, u] = 0.0
            dG = dq * g + q * dgi
            # d sigma from the softmax
            dz = lam * dG
            dsigma = sigma * (dz - (sigma * dz).sum(axis=2, keepdims=True))
            dW = np.linalg.solve(M, game.delta * (dT_from(dsigma, dq) @ W))
            e = np.zeros((S, N))
            e[u, m] = 1.0
            J[:, col] = (e - dW).ravel()

    # --- column for log lambda -------------------------------------------
    dalpha_u = lam * g * ap                    # dalpha/d(log lambda)
    dq_u = np.zeros_like(q)
    for k in range(N):
        dq_u += geom.voters[:, :, :, k].transpose(2, 0, 1) * excl[k] \
            * dalpha_u[k][None, :, :]
    dG_u = dq_u * g
    dz_u = lam * G + lam * dG_u                # d(lambda*G)/d(log lambda)
    dsigma_u = sigma * (dz_u - (sigma * dz_u).sum(axis=2, keepdims=True))
    dW_u = np.linalg.solve(M, game.delta * (dT_from(dsigma_u, dq_u) @ W))
    J[:, n] = (-dW_u).ravel()
    return J


def check_jacobian(game: Game, y: np.ndarray, h: float = 1e-6
                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Analytic vs central differences.  Meaningful only at modest lambda,
    where finite differences are still trustworthy."""
    n = game.S * game.N
    Ja = DH(game, y)
    Jn = np.zeros_like(Ja)
    for k in range(n + 1):
        yp, ym = y.copy(), y.copy()
        yp[k] += h
        ym[k] -= h
        Jn[:, k] = (H(game, yp) - H(game, ym)) / (2 * h)
    scale = max(float(np.abs(Jn).max()), 1.0)
    return float(np.abs(Ja - Jn).max() / scale), Ja, Jn


# ---------------------------------------------------------------------------
# arclength continuation in (V, log lambda)
# ---------------------------------------------------------------------------
def _structure(game: Game, V: np.ndarray, lam: float,
               sat: float = 8.0) -> Tuple:
    """Discrete signature of the smoothed profile: who is still undecided.

    A decision is 'interior' when its smoothed probability is not saturated,
    i.e. |lambda * margin| < sat.  The signature is what the track is really
    after -- once it stops changing, the regime is determined and the exact
    solver can take over.
    """
    geom = game.geom
    st = logit_state(game, V, lam)
    g, q, G = st["g"], st["q"], st["G"]
    acc = tuple(sorted(
        (int(j), int(min(s, t)), int(max(s, t)))
        for j in range(game.N) for s in range(game.S) for t in range(game.S)
        if geom.relevant[j, s, t] and abs(lam * g[j, s, t]) < sat))
    props = []
    for i in range(game.N):
        for s in range(game.S):
            opts = smpe._effective_options(geom, q, i, s)
            if len(opts) < 2:
                continue
            gm = max(G[i, s, t] for t in opts)
            sup = tuple(sorted(t for t in opts if lam * (gm - G[i, s, t]) < sat))
            if len(sup) > 1:
                props.append((int(i), int(s), sup))
    return (acc, tuple(sorted(props)))


def _corrector(game: Game, y0: np.ndarray, tang: Optional[np.ndarray],
               anchor: Optional[np.ndarray], *, ftol=1e-11,
               max_iter=40) -> Tuple[bool, np.ndarray]:
    """Newton on H (plus an arclength row when tracking), with a line search."""
    n = game.S * game.N
    y = y0.copy()

    def resid(z):
        r = H(game, z)
        if tang is None:
            return r
        return np.concatenate([r, [float(np.dot(tang, z - anchor))]])

    F = resid(y)
    nF = float(np.abs(F).max())
    for _ in range(max_iter):
        if nF < ftol:
            return True, y
        J = DH(game, y)
        if tang is None:
            A, b = J[:, :n], F
            try:
                dy_v = -np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dy_v = -np.linalg.lstsq(A, b, rcond=None)[0]
            dy = np.concatenate([dy_v, [0.0]])
        else:
            A = np.vstack([J, tang[None, :]])
            try:
                dy = -np.linalg.solve(A, F)
            except np.linalg.LinAlgError:
                dy = -np.linalg.lstsq(A, F, rcond=None)[0]
        if not np.all(np.isfinite(dy)):
            return False, y
        step, improved = 1.0, False
        for _ in range(25):
            yn = y + step * dy
            Fn = resid(yn)
            nFn = float(np.abs(Fn).max())
            if nFn < nF:
                improved = True
                break
            step *= 0.5
        if not improved:
            return nF < ftol, y
        y, F, nF = yn, Fn, nFn
    return nF < ftol, y


@dataclass
class Track:
    ok: bool
    V: Optional[np.ndarray]
    lam: float
    steps: int
    reason: str
    signature: Optional[Tuple] = None
    history: List[Tuple[float, int]] = field(default_factory=list)
    snapshots: List[Tuple[float, np.ndarray]] = field(default_factory=list)


def track(game: Game, *, lam0: float = 1e-3, lam_max: float = 1e6,
          h0: float = 0.2, hmin: float = 1e-7, max_steps: int = 3000,
          stable_decades: float = 1.0, sat: float = 8.0,
          lam_floor: float = 100.0,
          deadline: Optional[float] = None) -> Track:
    """Follow the logit path until the discrete structure settles."""
    S, N = game.S, game.N
    n = S * N
    u_max = float(np.log(lam_max))

    # start where the fixed point is easy: lambda ~ 0 makes Phi nearly constant
    y = np.concatenate([game.payoffs.ravel(), [float(np.log(lam0))]])
    ok, y = _corrector(game, y, None, None)
    if not ok:
        return Track(False, None, lam0, 0, "initial corrector failed")

    prev_tan = None
    det_sign = None
    h = h0
    sig = _structure(game, y[:n].reshape(S, N), lam0, sat)
    sig_u = y[-1]
    hist: List[Tuple[float, int]] = []
    snaps: List[Tuple[float, np.ndarray]] = []
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            return Track(False, y[:n].reshape(S, N), float(np.exp(y[-1])), step,
                         "deadline", sig, hist, snaps)
        J = DH(game, y)
        _u, _s, Vt = np.linalg.svd(J)
        tan = Vt[-1]
        # Orientation.  Keeping det([DH; tangent]) at a fixed sign is the robust
        # rule for passing folds (Allgower-Georg): it is invariant along a
        # regular path.  The dot product with the previous tangent -- the rule
        # used before -- flips wrongly when a step overshoots a sharp fold, and
        # the tracker then walks back down toward lambda = 0.  The principal
        # logit branch cannot return there (the lambda = 0 solution is unique
        # and regular), so doing so is always an orientation error.
        d = float(np.linalg.det(np.vstack([J, tan[None, :]])))
        if det_sign is None:
            if tan[-1] < 0:          # start by moving toward larger lambda
                tan, d = -tan, -d
            det_sign = np.sign(d) if d != 0 else 1.0
        elif d != 0 and np.sign(d) != det_sign:
            tan = -tan
        elif d == 0 and prev_tan is not None and np.dot(tan, prev_tan) < 0:
            tan = -tan
        prev_tan = tan

        yp = y + h * tan
        good, z = _corrector(game, yp, tan, yp)
        if not good:
            h *= 0.5
            if h < hmin:
                return Track(False, y[:n].reshape(S, N), float(np.exp(y[-1])),
                             step, "corrector failed at fold", sig, hist, snaps)
            continue
        y = z
        h = min(h * 1.3, h0 * 5)
        lam = _lam(y[-1])
        hist.append((lam, step))

        new = _structure(game, y[:n].reshape(S, N), lam, sat)
        if new != sig:
            sig, sig_u = new, y[-1]
            if lam >= lam_floor / 10:
                snaps.append((lam, y[:n].reshape(S, N).copy()))
        # A signature that is constant only because lambda is still too small
        # to saturate anything ("everything interior") is not information.
        # Stability counts only once lambda is large enough for the profile to
        # have started discriminating.
        elif (lam >= lam_floor
              and y[-1] - max(sig_u, np.log(lam_floor)) >=
              stable_decades * np.log(10)):
            return Track(True, y[:n].reshape(S, N), lam, step,
                         "structure stable", sig, hist, snaps)
        if y[-1] >= u_max:
            return Track(True, y[:n].reshape(S, N), lam, step, "lambda_max",
                         sig, hist, snaps)
    return Track(False, y[:n].reshape(S, N), float(np.exp(y[-1])), max_steps,
                 "max steps", sig, hist, snaps)


# ---------------------------------------------------------------------------
# handoff to the exact solver
# ---------------------------------------------------------------------------
def regime_from_logit(game: Game, V: np.ndarray, lam: float,
                      sat: float = 8.0) -> List[Tuple]:
    """Candidate mixing decisions read off the smoothed profile."""
    acc, props = _structure(game, V, lam, sat)
    cands: List[Tuple] = [("acc", j, s, t) for (j, s, t) in acc]
    cands += [("prop", i, s, sup) for (i, s, sup) in props]
    return cands


def regime_from_profile(game: Game, V: np.ndarray, lam: float,
                        sat: float = 8.0, qtol: float = 1e-9) -> Optional[smpe.Regime]:
    """Build the whole regime from the smoothed profile, not just the mixing set.

    `regime_with_mixing` pins every non-mixing decision by pure best response
    to V.  Near a tie that can pin a decision the wrong way -- and on the hard
    instances the path runs close to many ties.  The logit profile already
    records which way each settled decision fell, so take the pure parts from
    it directly: round the saturated probabilities, keep the unsaturated ones
    as unknowns.  Proposals to a state the voters certainly reject are folded
    into staying put, since they are identical in payoff and in transition.
    """
    geom = game.geom
    st = logit_state(game, V, lam)
    g, alpha, q, G, sigma = st["g"], st["alpha"], st["q"], st["G"], st["sigma"]
    base_alpha = np.where(alpha >= 0.5, 1.0, 0.0)
    acc = [(int(j), int(s_), int(t))
           for j in range(game.N) for s_ in range(game.S) for t in range(game.S)
           if geom.relevant[j, s_, t] and abs(lam * g[j, s_, t]) < sat]
    base_choice = np.zeros((game.N, game.S), dtype=int)
    props = []
    for i in range(game.N):
        for s_ in range(game.S):
            w = sigma[i, s_].copy()
            for t in range(game.S):          # fold certain rejections into stay
                if t != s_ and q[i, s_, t] < qtol:
                    w[s_] += w[t]
                    w[t] = 0.0
            gm = max(G[i, s_, t] for t in range(game.S) if w[t] > 0 or t == s_)
            sup = sorted(t for t in range(game.S)
                         if (w[t] > 0 or t == s_) and lam * (gm - G[i, s_, t]) < sat)
            base_choice[i, s_] = int(max(range(game.S), key=lambda t: w[t]))
            if len(sup) > 1:
                props.append((i, s_, tuple(sup)))
    if not acc and not props:
        return None
    return smpe.Regime(acc, props, base_alpha, base_choice, origin="logit profile")


def profile_hint(game: Game, R: smpe.Regime, ls: Dict[str, np.ndarray],
                 qtol: float = 1e-9) -> np.ndarray:
    """Mixing probabilities read off the smoothed profile, as a start for x.

    A proposal the voters certainly reject (q = 0) is identical to staying put
    in payoff and in transition, so its softmax mass belongs to 'stay'.  Taking
    only the in-support weights and renormalising -- which is what the generic
    hint does -- can badly misstate the mix: a proposer spreading 0.162 over
    each of three rejected proposals and staying, and 0.351 on exit, really
    stays with probability 0.649, not 0.316.
    """
    alpha, sigma, q = ls["alpha"], ls["sigma"], ls["q"]
    h: List[float] = []
    for (j, s_, t) in R.accept_vars:
        h.append(float(np.clip(alpha[j, s_, t], 1e-4, 1 - 1e-4)))
    for (i, s_, sup) in R.prop_vars:
        w = sigma[i, s_].copy()
        for t in range(game.S):
            if t != s_ and q[i, s_, t] < qtol:
                w[s_] += w[t]
                w[t] = 0.0
        ws = np.array([max(w[t], 1e-9) for t in sup])
        ws = ws / ws.sum()
        h.extend(float(v) for v in ws[:-1])
    return np.array(h, dtype=float)


def _attempt_profile(game, V, lam, sat, rng, t_end, verify_tol):
    R = regime_from_profile(game, V, lam, sat)
    if R is None:
        st = logit_state(game, V, lam)
        return _finalize(game, np.where(st["sigma"] >= st["sigma"].max(
            axis=2, keepdims=True), 1.0, 0.0), np.round(st["alpha"]),
            "homotopy/profile-pure", None, verify_tol=verify_tol)
    ls = logit_state(game, V, lam)
    hint = profile_hint(game, R, ls)
    res = smpe.solve_regime(game, R, x_hints=[hint], rng=rng, n_random=6,
                            max_iter=160, deadline=t_end)
    if "x" not in res:
        return None
    sysm = res["system"]
    x = sysm.clip(res["x"])
    st = sysm.state(x)
    ok_v, _b, _k = smpe.validate_regime(game, R, st, tol=1e-9)
    if not ok_v or res.get("resid", np.inf) > 1e-10:
        # The indifference system is usually underdetermined, so its roots form
        # a manifold and least-norm Newton picks the point nearest the start --
        # which knows nothing about the inequalities the non-mixing decisions
        # need.  Use the manifold's free directions to satisfy them too.
        for x_start in (x, hint):
            okr, xr, fr, mr = manifold_repair(game, R, x_start, deadline=t_end)
            if okr:
                x = xr
                break
        else:
            return None
        st = sysm.state(x)
    return _finalize(game, st["sigma"], st["alpha"], "homotopy/profile", R,
                     verify_tol=verify_tol,
                     rank_deficient=res.get("rank_deficient", False),
                     diagnostics=dict(x=x, residual=res.get("resid"),
                                      mixing_set=[("acc", j, s_, t) for (j, s_, t)
                                                  in R.accept_vars]
                                      + [("prop", i, s_, sp) for (i, s_, sp)
                                         in R.prop_vars], lam=lam, sat=sat))


def _attempt_cands(game, cands, V0, rng, t_end, verify_tol, tag, extra,
                   lam=None):
    if not cands:
        st = logit_state(game, V0, 1e8)
        return _finalize(game, st["sigma"], st["alpha"], tag + "/pure", None,
                         verify_tol=verify_tol)
    # Seed the exact solve with the logit profile's own mixing probabilities.
    # On the path they are already within O(1/lambda) of the answer, which is
    # a far better start than anything random -- and without it the Newton
    # solve regularly fails to find a root the homotopy has proved exists.
    dyn = None
    if lam is not None:
        ls = logit_state(game, V0, lam)
        dyn = (ls["alpha"], ls["sigma"])
    got, _r, _v = _constrained_pi(game, cands, V0, rng, max_outer=12,
                                  n_random=8, max_iter=140, deadline=t_end,
                                  dyn=dyn)
    if got is None:
        return None
    st, R, res = got
    return _finalize(game, st["sigma"], st["alpha"], tag + "/mix", R,
                     verify_tol=verify_tol,
                     rank_deficient=res.get("rank_deficient", False),
                     diagnostics=dict(x=res["x"], residual=res["resid"],
                                      mixing_set=list(cands), **extra))


def solve_hard(players, payoffs, delta: float, *, rho=None, rows=None,
               rescale: bool = True, seed: int = 0, verify_tol: float = 1e-7,
               time_budget: float = 60.0, sats: Sequence[float] = (8.0, 4.0, 15.0),
               lam_max: float = 1e7, max_snapshots: int = 8,
               verbose: bool = False, committees=None) -> Optional[Solution]:
    """Track the logit path, read off the regime, solve exactly, verify.

    The regime is read at several points along the path, newest first: the
    last structure change is usually right, but a structure can settle and
    then shift again, and each exact attempt costs well under a second, so
    trying several is cheap insurance.  If none work, leave-one-out on the
    final candidate set handles a single misread decision.
    """
    t_end = time.time() + time_budget
    game = Game.build(players, payoffs, delta, rho=rho, rows=rows,
                      rescale=rescale, committees=committees)
    rng = np.random.default_rng(seed)
    tr = track(game, lam_max=lam_max, deadline=t_end)
    if verbose:
        print(f"   track: ok={tr.ok} lam={tr.lam:.3g} steps={tr.steps} "
              f"({tr.reason}), {len(tr.snapshots)} snapshots")
    if tr.V is None:
        return None
    points = [(tr.lam, tr.V)] + list(reversed(tr.snapshots))[:max_snapshots]
    seen = set()
    final_cands = None
    for lam, V in points:
        for sat in sats:
            if time.time() > t_end:
                return None
            sol = _attempt_profile(game, V, lam, sat, rng, t_end, verify_tol)
            if sol is not None:
                return sol
            sol = _attempt_profile_grow(game, V, lam, sat, rng, t_end,
                                        verify_tol)
            if sol is not None:
                return sol
            cands = regime_from_logit(game, V, lam, sat)
            key = tuple(sorted(map(str, cands)))
            if key in seen:
                continue
            seen.add(key)
            if final_cands is None:
                final_cands = cands
            sol = _attempt_cands(game, cands, V, rng, t_end, verify_tol,
                                 "homotopy", dict(lam=lam, sat=sat), lam=lam)
            if verbose:
                print(f"   lam={lam:.3g} sat={sat}: {len(cands)} cands -> "
                      + ("OK" if sol else "no"))
            if sol is not None:
                return sol
    # leave-one-out on the final structure
    if final_cands and len(final_cands) <= 30:
        for k in range(len(final_cands)):
            if time.time() > t_end:
                break
            sub = final_cands[:k] + final_cands[k + 1:]
            sol = _attempt_cands(game, sub, tr.V, rng, t_end, verify_tol,
                                 "homotopy-loo", dict(lam=tr.lam), lam=tr.lam)
            if sol is not None:
                if verbose:
                    print(f"   leave-one-out: solved dropping {final_cands[k]}")
                return sol
    return None


# ---------------------------------------------------------------------------
# manifold repair: use the free directions of an underdetermined system
# ---------------------------------------------------------------------------
def _margins(game: Game, R: smpe.Regime, st: Dict[str, np.ndarray]) -> np.ndarray:
    """Every inequality the non-mixing decisions need, as margins that must be
    >= 0.  Voters pinned to accept need V_j(t) >= V_j(s) (and the reverse for
    reject); a proposer's chosen set must be at least as good as every option.
    """
    geom = game.geom
    V, q = st["V"], st["q"]
    g = gain_matrix(V)
    G = q * g
    mix_a = set(R.accept_vars)
    out = []
    for j in range(game.N):
        for s_ in range(game.S):
            for t in range(game.S):
                if not geom.relevant[j, s_, t] or (j, s_, t) in mix_a:
                    continue
                out.append(g[j, s_, t] if R.base_alpha[j, s_, t] >= 0.5
                           else -g[j, s_, t])
    mix_p = {(i, s_): sup for (i, s_, sup) in R.prop_vars}
    for i in range(game.N):
        for s_ in range(game.S):
            chosen = mix_p.get((i, s_), (int(R.base_choice[i, s_]),))
            for c in chosen:
                for t in range(game.S):
                    if t not in chosen:
                        out.append(G[i, s_, c] - G[i, s_, t])
    return np.array(out, dtype=float)


def manifold_repair(game: Game, R: smpe.Regime, x0: np.ndarray, *,
                    weight: float = 10.0, target: float = 1e-9,
                    max_iter: int = 80, h: float = 1e-7,
                    deadline: Optional[float] = None
                    ) -> Tuple[bool, np.ndarray, float, float]:
    """Find x in [0,1]^k with F(x) = 0 AND every non-mixing margin >= 0.

    Gauss-Newton on r(x) = [F(x); weight * min(0, margin(x) - target)].  The
    Jacobian is by finite differences: x is low-dimensional and the map is
    smooth and well-scaled in x, unlike the lambda-direction where finite
    differences failed.  Returns (ok, x, |F|, worst margin).
    """
    sysm = smpe.RegimeSystem(game, R)

    def r(x):
        st = sysm.state(x)
        F = sysm.residual_from_state(st)
        m = _margins(game, R, st)
        return np.concatenate([F, weight * np.minimum(0.0, m - target)]), F, m

    x = sysm.clip(np.asarray(x0, dtype=float))
    rv, F, m = r(x)
    cost = float(rv @ rv)
    lam = 1e-6
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if np.abs(F).max() < 1e-11 and (m.size == 0 or m.min() >= -1e-10):
            return True, x, float(np.abs(F).max()), float(m.min() if m.size else 0)
        J = np.zeros((rv.size, x.size))
        for k in range(x.size):
            xp = x.copy()
            xp[k] += h
            J[:, k] = (r(sysm.clip(xp))[0] - rv) / h
        improved = False
        for _ls in range(12):
            A = np.vstack([J, np.sqrt(lam) * np.eye(x.size)])
            b = np.concatenate([rv, np.zeros(x.size)])
            dx = -np.linalg.lstsq(A, b, rcond=None)[0]
            xn = sysm.clip(x + dx)
            rn, Fn, mn = r(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, rv, F, m, cost = xn, rn, Fn, mn, cn
                lam = max(lam / 5.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
    return (bool(np.abs(F).max() < 1e-11 and (m.size == 0 or m.min() >= -1e-10)),
            x, float(np.abs(F).max()), float(m.min() if m.size else 0.0))


def grow_regime(game: Game, R: smpe.Regime, vkeys: Sequence[Tuple],
                st: Dict[str, np.ndarray]) -> Optional[smpe.Regime]:
    """Add the decisions validation flagged to the mixing set.

    A decision that flips when the exact solve moves V by O(1/lambda) was
    sitting at an indifference the path had not yet resolved -- it belongs to
    the same indifference class, so it has to mix too.
    """
    geom = game.geom
    acc = list(R.accept_vars)
    props = {(i, s_): list(sup) for (i, s_, sup) in R.prop_vars}
    V, q = st["V"], st["q"]
    G = q * gain_matrix(V)
    changed = False
    for key in vkeys:
        if key[0] == "acc":
            _, j, a, b = key
            for (s_, t) in ((a, b), (b, a)):
                if geom.relevant[j, s_, t] and (j, s_, t) not in acc:
                    acc.append((j, s_, t))
                    changed = True
        else:
            _, i, s_, _sup = key
            cur = props.get((i, s_), [int(R.base_choice[i, s_])])
            opts = smpe._effective_options(geom, q, i, s_)
            best = max(opts, key=lambda t: G[i, s_, t])
            if best not in cur:
                props[(i, s_)] = sorted(set(cur) | {best})
                changed = True
    if not changed:
        return None
    return smpe.Regime(acc, [(i, s_, tuple(sp)) for (i, s_), sp in props.items()
                             if len(sp) > 1],
                       R.base_alpha.copy(), R.base_choice.copy(),
                       origin="profile+grown")


def _solve_regime_repaired(game, R, hints, rng, t_end):
    """Solve a regime; repair along the manifold if the root breaks a margin.
    Returns (state, x, R, ok, vkeys)."""
    res = smpe.solve_regime(game, R, x_hints=hints, rng=rng, n_random=6,
                            max_iter=160, deadline=t_end)
    if "x" not in res:
        return None, None, False, []
    sysm = res["system"]
    x = sysm.clip(res["x"])
    st = sysm.state(x)
    ok, _b, vk = smpe.validate_regime(game, R, st, tol=1e-9)
    if ok and res.get("resid", 1.0) <= 1e-10:
        return st, x, True, []
    for xs in [x] + list(hints):
        okr, xr, _f, _m = manifold_repair(game, R, xs, deadline=t_end)
        if okr:
            st = sysm.state(xr)
            return st, xr, True, []
    return st, x, False, vk


def _attempt_profile_grow(game, V, lam, sat, rng, t_end, verify_tol,
                          rounds: int = 6):
    R = regime_from_profile(game, V, lam, sat)
    if R is None:
        return None
    ls = logit_state(game, V, lam)
    for _r in range(rounds):
        if time.time() > t_end:
            return None
        hints = [profile_hint(game, R, ls), np.full(R.k, 0.5)]
        st, x, ok, vk = _solve_regime_repaired(game, R, hints, rng, t_end)
        if ok:
            return _finalize(game, st["sigma"], st["alpha"],
                             "homotopy/profile-grown", R,
                             verify_tol=verify_tol,
                             diagnostics=dict(x=x, lam=lam, sat=sat,
                                              mixing_set=[("acc",) + a for a in
                                                          R.accept_vars]
                                              + [("prop",) + p for p in
                                                 R.prop_vars]))
        if st is None or not vk:
            return None
        R2 = grow_regime(game, R, vk, st)
        if R2 is None:
            return None
        R = R2
    return None
