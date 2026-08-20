#!/usr/bin/env python3
"""
Solve the synthetic control game exactly, in the M = 1 case.

M = 1 means the whole profile is pure except for a single mixing probability
theta.  The naive way to solve for theta -- pair it with its own indifference
equation -- cannot work, and not for numerical reasons:

    changing theta moves probability mass only between the two states the mixing
    voter is indifferent between, so dV_j/dtheta is proportional to that voter's
    own gap f(theta).  That is a linear homogeneous ODE df/dtheta = c(theta) f, so
    f(theta_0) = 0 at any single point forces f == 0 everywhere.

The mixing voter's own condition is therefore a constraint on the REST of the
profile, and carries no information about theta at all.  What pins theta is some
OTHER player's condition, whose value does move with theta.  So the pairing is
off-diagonal, and the support pattern has to name two things: which alpha is
interior, and which other condition is the binding equality.

Given both, the system is one equation in one unknown -- univariate -- and can be
solved and checked exhaustively.  This script does that: enumerate every candidate
binding condition, root-find theta for each, and verify the resulting profile with
the framework's own conditions.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

from lib.equilibrium.jeres_vfi.solver import (  # noqa: E402
    compute_values,
    full_transition_matrix,
    verify_proposals,
    verify_responses,
)

spec = importlib.util.spec_from_file_location(
    "mcg", REPO / "scripts" / "mixed_control_game.py")
mcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcg)

DELTA = mcg.DELTA
ATOL = 1e-12
GRID = np.linspace(1e-6, 1 - 1e-6, 401)


def main():
    ctrl = mcg.build_control(["A", "B", "C"], seed=0)
    if ctrl is None:
        print("could not build the control game")
        return 1
    game = ctrl["game"]
    tx, ty, tj = ctrl["tie"]
    comm = mcg.committees(game)
    n_s, n_p = game.n_states, game.n_players
    sigmas = ctrl["sigmas"]                      # the pure part, held fixed

    print(f"control game: {n_p} players, {n_s} states, delta={DELTA}")
    print(f"support pattern: interior alpha_{tj}({tx}->{ty}); everything else pure")
    print(f"(the planted answer is theta={mcg.THETA}; the solve does not use it)\n")

    def state_at(theta):
        alphas = [dict(d) for d in ctrl["alphas"]]
        alphas[tx][(tj, ty)] = float(theta)
        qs = mcg.qs_from_alphas(game, comm, alphas)
        T = full_transition_matrix(game, sigmas, qs, None)
        return compute_values(game, T, DELTA), alphas, qs

    # Candidate binding conditions: every acceptance gap, and every proposal
    # comparison against the target actually proposed.
    cands = []
    for (x, j, y) in sorted({(x, j, y) for (i, x, y) in
                             [(i, x, y) for i in range(n_p)
                              for x in range(n_s) for y in range(n_s) if x != y]
                             for j in comm[(i, x, y)]}):
        cands.append(("acc", x, j, y))
    for i in range(n_p):
        for x in range(n_s):
            chosen = next((y for y in range(n_s)
                           if sigmas[x].get((i, y), 0.0) > 0.5), x)
            for y in range(n_s):
                if y != chosen:
                    cands.append(("prop", i, x, chosen, y))

    def residual(cand, theta):
        V, alphas, qs = state_at(theta)
        if cand[0] == "acc":
            _, x, j, y = cand
            return V[y, j] - V[x, j]
        _, i, x, c, y = cand
        hc = 0.0 if c == x else qs[x][(i, c)] * (V[c, i] - V[x, i])
        hy = 0.0 if y == x else qs[x][(i, y)] * (V[y, i] - V[x, i])
        return hc - hy

    flat = live = 0
    solutions = []
    for cand in cands:
        vals = np.array([residual(cand, t) for t in GRID])
        if np.max(np.abs(vals - vals[0])) < 1e-14:
            flat += 1          # carries no information about theta
            continue
        live += 1
        sign = np.sign(vals)
        for k in range(len(GRID) - 1):
            if sign[k] == 0 or sign[k] * sign[k + 1] >= 0:
                continue
            try:
                root = brentq(lambda t: residual(cand, t),
                              GRID[k], GRID[k + 1], xtol=1e-15, rtol=8.9e-16)
            except Exception:
                continue
            V, alphas, qs = state_at(root)
            r_ok, _ = verify_responses(game, sigmas, alphas, qs, V, atol=ATOL)
            p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V, atol=ATOL)
            solutions.append((cand, root, r_ok and p_ok))

    print(f"candidate binding conditions : {len(cands)}")
    print(f"  flat in theta (no information): {flat}")
    print(f"  genuinely varying with theta  : {live}")
    print(f"  roots found in (0,1)          : {len(solutions)}\n")

    own = ("acc", tx, tj, ty)
    own_flat = all(np.max(np.abs(np.array([residual(c, t) for t in GRID[:20]])
                                 - residual(c, GRID[0]))) < 1e-14
                   for c in [own] if c in cands)
    print(f"the mixing voter's OWN condition {own} is flat in theta: {own_flat}")
    print("  (predicted by the ODE argument -- it cannot determine theta)\n")

    good = [s for s in solutions if s[2]]
    for cand, root, ok in sorted(solutions, key=lambda s: -s[2])[:12]:
        tag = "  <== EQUILIBRIUM" if ok else ""
        print(f"  {str(cand):<34} theta={root:.12f}  verifies={ok}{tag}")

    print(f"\nverified equilibria found: {len(good)}")
    if good:
        best = good[0]
        print(f"  theta = {best[1]:.12f}   (planted {mcg.THETA})")
        print(f"  pinned by: {best[0]}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
