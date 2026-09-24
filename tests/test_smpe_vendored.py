"""test_smpe.py -- self-tests for the SMPE solver.

Run:  python test_smpe.py
"""
from __future__ import annotations

import sys
import numpy as np

from lib.equilibrium.smpe import smpe
from lib.equilibrium.smpe.smpe import (CoalitionGeometry, Game, Regime, RegimeSystem, build_T,
                  compute_q, evaluate, gain_matrix, pure_best_response,
                  solve_smpe, verify)

FAILURES = []


def check(name, cond, info=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {info}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def test_geometry():
    print("\n[1] geometry: partitions, free exit, voters")
    for n, bell in [(2, 2), (3, 5), (4, 15), (5, 52)]:
        g = CoalitionGeometry([chr(65 + i) for i in range(n)])
        check(f"Bell number N={n}", g.S == bell, f"got {g.S}")

    g = CoalitionGeometry(["A", "B", "C"])
    idx = {g.label(s): s for s in range(g.S)}
    grand = idx["{A,B,C}"]
    singles = idx["{A}+{B}+{C}"]
    ab_c = idx["{A,B}+{C}"]

    # A exiting the grand coalition leaves {B,C}
    tgt = g.exit_target[grand, 0]
    check("A exits {A,B,C} -> {A}+{B,C}", g.label(tgt) == "{A}+{B,C}",
          g.label(tgt))
    check("that exit needs no votes", not g.voters[grand, tgt, 0].any())
    # ... yet all three players change block, so |Delta| == 3, not 1
    check("|Delta| == 3 for that exit", int(g.movers[grand, tgt].sum()) == 3)

    # merging singletons into {A,B} requires B's consent (proposer A)
    vs = np.nonzero(g.voters[singles, ab_c, 0])[0]
    check("A proposing {A,B}+{C} from singletons needs exactly B",
          list(vs) == [1], str(vs))
    # C is unaffected and does not vote
    check("C does not vote on {A,B} forming", not g.voters[singles, ab_c, 0, 2])

    # going from {A,B}+{C} to {A,C}+{B} is not anyone's free exit
    ac_b = idx["{A,C}+{B}"]
    check("no free exit between {A,B}+{C} and {A,C}+{B}",
          not g.free_exit[ab_c, ac_b].any())

    # already-singleton player: exit target is the state itself
    check("singleton's exit target is itself",
          all(g.exit_target[singles, i] == singles for i in range(3)))

    # N=4: exhaustive re-derivation of the free-exit predicate
    g4 = CoalitionGeometry(["A", "B", "C", "D"])
    bad = 0
    for s in range(g4.S):
        for t in range(g4.S):
            for i in range(4):
                want = (g4.block[t][i] == frozenset([i])) and all(
                    g4.block[t][j] == (g4.block[s][j] - {i})
                    for j in range(4) if j != i)
                if bool(g4.free_exit[s, t, i]) != want:
                    bad += 1
    check("N=4 free-exit predicate exhaustive", bad == 0, f"{bad} mismatches")
    check("N=4: |Delta(s,t)|==1 never happens for s!=t",
          all(g4.movers[s, t].sum() != 1 for s in range(g4.S)
              for t in range(g4.S) if s != t))


# ---------------------------------------------------------------------------
def test_primitives():
    print("\n[2] primitives: q, T row-stochasticity, exact evaluation")
    rng = np.random.default_rng(3)
    for n in (3, 4):
        g = CoalitionGeometry([chr(65 + i) for i in range(n)])
        for _ in range(20):
            alpha = rng.uniform(size=(g.N, g.S, g.S))
            sigma = rng.uniform(size=(g.N, g.S, g.S))
            sigma /= sigma.sum(axis=2, keepdims=True)
            rho = rng.uniform(size=g.N)
            rho /= rho.sum()
            q = compute_q(alpha, g.voters)
            T = build_T(sigma, q, rho)
            assert np.all(T >= -1e-14) and np.allclose(T.sum(axis=1), 1.0)
        # free exit always has q == 1
        fe = g.free_exit                                   # (S,S,N)
        qq = np.transpose(q, (1, 2, 0))                    # (S,S,N)
        check(f"N={n}: q==1 wherever there are no voters",
              np.allclose(qq[fe], 1.0))
        check(f"N={n}: T row-stochastic for random strategies", True)

    # Bellman identity of the exact solve
    g = CoalitionGeometry(["A", "B", "C"])
    pi = rng.normal(size=(g.S, g.N))
    alpha = rng.uniform(size=(g.N, g.S, g.S))
    sigma = rng.uniform(size=(g.N, g.S, g.S)); sigma /= sigma.sum(2, keepdims=True)
    T = build_T(sigma, compute_q(alpha, g.voters), np.full(g.N, 1 / g.N))
    d = 0.97
    V = evaluate(T, pi, d)
    res = np.abs(V - ((1 - d) * pi + d * T @ V)).max()
    check("exact policy evaluation satisfies Bellman", res < 1e-12, f"{res:.2e}")

    # V of a constant payoff equals that constant (the identity behind rescaling)
    Vc = evaluate(T, np.ones((g.S, g.N)) * 4.2, d)
    check("(I-dT)^-1 (1-d) 1 = 1", np.allclose(Vc, 4.2))


# ---------------------------------------------------------------------------
def test_jacobian():
    print("\n[3] analytic Jacobian of the indifference system vs finite differences")
    rng = np.random.default_rng(11)
    g = CoalitionGeometry(["A", "B", "C"])
    pi = rng.normal(size=(g.S, g.N))
    game = Game.build(["A", "B", "C"], pi, 0.9, rescale=False)

    rel = list(zip(*np.nonzero(g.relevant)))
    worst = 0.0
    for trial in range(8):
        picks = [tuple(int(v) for v in rel[k])
                 for k in rng.choice(len(rel), size=2, replace=False)]
        i0, s0 = int(rng.integers(g.N)), int(rng.integers(g.S))
        sup = tuple(sorted(rng.choice(g.S, size=3, replace=False).tolist()))
        base_alpha = rng.uniform(0.2, 0.8, size=(g.N, g.S, g.S))
        base_choice = rng.integers(0, g.S, size=(g.N, g.S))
        R = Regime(list(picks), [(i0, s0, sup)], base_alpha, base_choice)
        sysm = RegimeSystem(game, R)
        x = rng.uniform(0.15, 0.4, size=sysm.k)
        J = sysm.jacobian(x)
        Jn = np.zeros_like(J)
        h = 1e-6
        for p in range(sysm.k):
            xp, xm = x.copy(), x.copy()
            xp[p] += h; xm[p] -= h
            Jn[:, p] = (sysm.residual(xp) - sysm.residual(xm)) / (2 * h)
        err = np.abs(J - Jn).max() / max(1.0, np.abs(Jn).max())
        worst = max(worst, err)
    check("analytic Jacobian matches central differences", worst < 1e-6,
          f"max rel err {worst:.2e}")


# ---------------------------------------------------------------------------
def test_two_players():
    print("\n[4] N=2 closed forms")
    P = ["A", "B"]
    # (a) both prefer the grand coalition -> it is absorbing, V = pi(grand)
    pay = np.array([[1.0, 1.0], [0.0, 0.0]])
    for d in (0.5, 0.9, 0.99):
        r = solve_smpe(P, pay, d, rescale=False, multistart=4)
        ok = (r.ok and np.allclose(r.solution.V_raw[0], [1.0, 1.0])
              and np.isclose(r.solution.T[0, 0], 1.0))
        check(f"N=2 grand absorbing (delta={d})", ok,
              "" if r.ok else "no solution")

    # (b) A strictly prefers to be alone -> free exit makes the split absorbing
    pay = np.array([[0.0, 1.0], [1.0, 0.0]])
    for d in (0.5, 0.9, 0.99):
        r = solve_smpe(P, pay, d, rescale=False, multistart=4)
        ok = (r.ok and np.isclose(r.solution.T[1, 1], 1.0)
              and np.allclose(r.solution.V_raw[1], [1.0, 0.0]))
        # analytic V at the grand coalition
        rhoA = 0.5
        VA = ((1 - d) * 0.0 + d * rhoA * 1.0) / (1 - d * (1 - rhoA))
        ok = ok and np.isclose(r.solution.V_raw[0, 0], VA, atol=1e-9)
        check(f"N=2 free exit absorbs to split (delta={d})", ok)


# ---------------------------------------------------------------------------
def test_affine_invariance():
    print("\n[5] affine invariance of the equilibrium set")
    P = ["A", "B", "C"]
    pay = np.array([
        [98.222, 13.222, -15.111],
        [107.250, 9.750, -22.750],
        [55.688, 14.438, 0.688],
        [118.188, 1.938, -36.812],
        [0.0, 0.0, 0.0]])
    rows = ["{A,B,C}", "{A,C}+{B}", "{A}+{B,C}", "{A,B}+{C}", "{A}+{B}+{C}"]
    a = np.array([3.0, 0.01, 7.5])
    b = np.array([-100.0, 4.0, 0.0])
    r1 = solve_smpe(P, pay, 0.9, rows=rows, multistart=4)
    r2 = solve_smpe(P, pay * a + b, 0.9, rows=rows, multistart=4)
    if r1.ok and r2.ok:
        V2 = r2.solution.V_raw
        V1 = r1.solution.V_raw * a[None, :] + b[None, :]
        check("rescaled game gives affinely equivalent V",
              np.allclose(V1, V2, atol=1e-6), f"{np.abs(V1-V2).max():.2e}")
        check("rescaled game gives the same transition matrix",
              np.allclose(r1.solution.T, r2.solution.T, atol=1e-6))
    else:
        check("affine invariance (both solved)", False)

    # rescale=True vs rescale=False need not select the *same* equilibrium
    # (the game has several), but a profile that is an equilibrium of one
    # scaling must be an equilibrium of the other -- that is the invariance.
    r3 = solve_smpe(P, pay, 0.9, rows=rows, rescale=False, multistart=4)
    check("unscaled run also finds a verified equilibrium", r3.ok)
    if r1.ok and r3.ok:
        rep = verify(r1.game, r3.solution.sigma, r3.solution.alpha, tol=1e-7)
        check("a profile verified unscaled also verifies rescaled", rep.ok)
        rep2 = verify(r3.game, r1.solution.sigma, r1.solution.alpha, tol=1e-5)
        check("a profile verified rescaled also verifies unscaled", rep2.ok)
        if not np.allclose(r1.solution.V_raw, r3.solution.V_raw, atol=1e-6):
            print("        (note: the two runs selected different equilibria, "
                  "which this game admits)")


# ---------------------------------------------------------------------------
def test_verifier_can_fail():
    print("\n[6] the verifier actually rejects non-equilibria")
    P = ["A", "B", "C"]
    pay = np.array([
        [98.222, 13.222, -15.111],
        [107.250, 9.750, -22.750],
        [55.688, 14.438, 0.688],
        [118.188, 1.938, -36.812],
        [0.0, 0.0, 0.0]])
    rows = ["{A,B,C}", "{A,C}+{B}", "{A}+{B,C}", "{A,B}+{C}", "{A}+{B}+{C}"]
    r = solve_smpe(P, pay, 0.99, rows=rows, multistart=4)
    check("baseline solves at delta=0.99", r.ok)
    if not r.ok:
        return
    sol = r.solution
    game = r.game

    # (a) perturb a proposal off the argmax
    found = False
    for i in range(game.N):
        for s in range(game.S):
            for t in range(game.S):
                if sol.sigma[i, s, t] > 1e-9:
                    continue
                sig = sol.sigma.copy()
                sig[i, s] = 0.0
                sig[i, s, t] = 1.0
                rep = verify(game, sig, sol.alpha, tol=1e-7)
                if not rep.ok:
                    found = True
                    break
            if found:
                break
        if found:
            break
    check("a deviated proposal is rejected by verify()", found)

    # (b) flip an acceptance decision that matters
    flipped = False
    for j in range(game.N):
        for s in range(game.S):
            for t in range(game.S):
                if not game.geom.relevant[j, s, t]:
                    continue
                al = sol.alpha.copy()
                al[j, s, t] = 1.0 - al[j, s, t]
                rep = verify(game, sol.sigma, al, tol=1e-7)
                if not rep.ok:
                    flipped = True
                    break
            if flipped:
                break
        if flipped:
            break
    check("a flipped acceptance is rejected by verify()", flipped)

    # (c) a non-stochastic sigma is rejected
    sig = sol.sigma.copy()
    sig[0, 0] *= 0.5
    rep = verify(game, sig, sol.alpha, tol=1e-7)
    check("non-normalised sigma is rejected", not rep.ok)


# ---------------------------------------------------------------------------
def test_state_order_permutation():
    print("\n[7] row-order permutation")
    g = CoalitionGeometry(["A", "B", "C"])
    rows = ["{A,B,C}", "{A,C}+{B}", "{A}+{B,C}", "{A,B}+{C}", "{A}+{B}+{C}"]
    perm = g.permutation_from(rows)
    labs = [rows[perm[s]] for s in range(g.S)]
    check("permutation maps internal order to the brief's order",
          labs == g.labels(), f"{labs} vs {g.labels()}")
    pay = np.arange(15.0).reshape(5, 3)
    game = Game.build(["A", "B", "C"], pay, 0.9, rows=rows, rescale=False)
    ok = all(np.allclose(game.raw_payoffs[s], pay[perm[s]]) for s in range(5))
    check("payoff rows permuted consistently", ok)
    try:
        g.permutation_from(rows[:-1] + ["{A,B,C}"])
        check("duplicate/incomplete row list rejected", False)
    except ValueError:
        check("duplicate/incomplete row list rejected", True)


# ---------------------------------------------------------------------------
def test_mixed_is_really_mixed():
    print("\n[8] mixed solutions satisfy indifference exactly")
    P = ["A", "B", "C"]
    pay = np.array([
        [98.222, 13.222, -15.111],
        [107.250, 9.750, -22.750],
        [55.688, 14.438, 0.688],
        [118.188, 1.938, -36.812],
        [0.0, 0.0, 0.0]])
    rows = ["{A,B,C}", "{A,C}+{B}", "{A}+{B,C}", "{A,B}+{C}", "{A}+{B}+{C}"]
    for d in (0.90, 0.95):
        r = solve_smpe(P, pay, d, rows=rows, multistart=4)
        check(f"example 1 solves at delta={d}", r.ok)
        if not r.ok:
            continue
        sol = r.solution
        V = sol.report.V_induced
        g = gain_matrix(V)
        worst = 0.0
        for j in range(r.game.N):
            for s in range(r.game.S):
                for t in range(r.game.S):
                    a = sol.alpha[j, s, t]
                    if r.game.geom.relevant[j, s, t] and 1e-9 < a < 1 - 1e-9:
                        worst = max(worst, abs(g[j, s, t]))
        check(f"  interior alpha implies indifference (delta={d})", worst < 1e-8,
              f"{worst:.2e}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_geometry()
    test_primitives()
    test_jacobian()
    test_two_players()
    test_affine_invariance()
    test_verifier_can_fail()
    test_state_order_permutation()
    test_mixed_is_really_mixed()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILING: " + ", ".join(FAILURES))
        sys.exit(1)
    print("all tests passed")
