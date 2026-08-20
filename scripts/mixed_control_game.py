#!/usr/bin/env python3
"""
A positive control for the mixed-equilibrium solver.

Without a game whose mixed equilibrium is known, "the solver found nothing" is
uninterpretable: it could mean the equilibrium needs a support the search never
visits, or it could mean the search is broken.  This builds a game where the
answer is known by CONSTRUCTION.

The construction, which needs no search and no LP.  Work backwards from the value
function instead of forwards from a guessed profile:

  1. Choose a value function V freely, except make one voter EXACTLY indifferent
     about one transition -- V_j(y) = V_j(x).
  2. Derive the profile that best-responds to that V.  Every acceptance is pinned
     by the sign of its value gap, except the tied one, which is free: set it to
     theta.  Every proposer takes the argmax.
  3. That profile fixes a transition matrix T.  Choose the payoffs that make the V
     from step 1 the true value function of T:  u = (I - delta*T) V / (1 - delta).

Now V is simultaneously the value function of the profile AND the thing the profile
best-responds to, with one player exactly indifferent and mixing at theta.  That is
the definition of a mixed equilibrium, so the game is certified without solving
anything.

An earlier attempt fixed the profile first and solved an LP for the payoffs.  That
is infeasible for structural reasons, not incidental ones: a tie contradicts the
chain of ordering constraints that connects the two tied states, and the proposal
conditions over-determine V on their own.  Working backwards from V avoids both.

Usage:
    python scripts/mixed_control_game.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium.jeres_vfi import Game, all_partitions, voters  # noqa: E402
from lib.equilibrium.jeres_vfi.solver import (  # noqa: E402
    compute_values,
    full_transition_matrix,
    verify_proposals,
    verify_responses,
    vfi,
)

DELTA = 0.9
THETA = 0.37          # deliberately not 1/2, so an averaged profile cannot fake it
ATOL = 1e-12


def committees(game):
    out = {}
    for i in range(game.n_players):
        for x in range(game.n_states):
            for y in range(game.n_states):
                out[(i, x, y)] = voters(game, game.states[x], game.states[y], i)
    return out


def qs_from_alphas(game, comm, alphas):
    qs = []
    for x in range(game.n_states):
        q = {}
        for i in range(game.n_players):
            for y in range(game.n_states):
                v = 1.0
                for j in comm[(i, x, y)]:
                    v *= alphas[x].get((j, y), 1.0)
                q[(i, y)] = v
        qs.append(q)
    return qs


def profile_at(game, comm, V, tie, theta):
    """Best responses to V, with the one tied acceptance set to theta."""
    n_s, n_p = game.n_states, game.n_players
    alphas = []
    for x in range(n_s):
        alp = {}
        for i in range(n_p):
            for y in range(n_s):
                for j in comm[(i, x, y)]:
                    gap = V[y, j] - V[x, j]
                    alp[(j, y)] = theta if (x, y, j) == tie else (
                        1.0 if gap > 0 else 0.0)
        alphas.append(alp)

    qs = qs_from_alphas(game, comm, alphas)
    sigmas = []
    for x in range(n_s):
        sig = {}
        for i in range(n_p):
            best, best_val = x, 0.0
            for y in range(n_s):
                if y == x:
                    continue
                val = qs[x][(i, y)] * (V[y, i] - V[x, i])
                if val > best_val + 1e-15:
                    best, best_val = y, val
            for y in range(n_s):
                sig[(i, y)] = 1.0 if y == best else 0.0
        sigmas.append(sig)
    return sigmas, alphas, qs


def pure_equilibrium_exists(game, u):
    """
    Does ANY pure-strategy equilibrium of this game exist?

    Delegates to `ordinal_ranking`, which searches 0/1 acceptance only.  That
    limitation is exactly what makes it usable as a proof of absence: an exhausted
    run is a statement about the whole pure space rather than about a search path.
    Validated separately at 20/20 recovery on games whose pure equilibria are known.
    """
    import subprocess
    import tempfile

    import pandas as pd

    codes = ["CHN", "EUR", "USA"][: game.n_players]
    a, b, c = (codes + ["", ""])[:3]
    names = ["( )", f"({a}{b})", f"({a}{c})", f"({b}{c})", f"({a}{b}{c})"]
    from lib.equilibrium.jeres_vfi import fw_state_name_to_partition
    rows = {n: u[game.state_idx[fw_state_name_to_partition(n, codes)]] for n in names}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=codes)
    df.index.name = "state"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "probe_chneurusa.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            df.to_excel(xl, sheet_name="Payoffs", startrow=1)
        out = subprocess.run(
            [sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
             "--payoff-table", str(path), "--discounting", str(DELTA),
             "--effectivity-rule", "heyen_lehtomaa_2021",
             "--solver-approach", "ordinal_ranking", "--verify-atol", "1e-12",
             "--fresh", "--output", "/dev/null", "--oneline", "--quiet"],
            cwd=REPO, capture_output=True, text=True, timeout=1800)
        return "VERIFIED" in (out.stdout + out.stderr)


def build_control(players, seed):
    """
    A game with a known mixed equilibrium AND no pure equilibrium at all.

    Both halves matter.  A known mixed equilibrium makes the answer checkable; the
    absence of any pure equilibrium makes the check discriminating, because
    otherwise the solver passes by finding a pure equilibrium of the same game and
    the mixed path is never exercised.
    """
    rng = np.random.default_rng(seed)
    n_p = len(players)
    n_s = len(all_partitions(n_p))
    probe = Game.from_payoffs(players, np.zeros((n_s, n_p)))
    comm = committees(probe)
    ties = sorted({(x, y, j) for (i, x, y), c in comm.items() for j in c if x != y})

    for _ in range(120):
        V = rng.uniform(-1.0, 1.0, size=(n_s, n_p))
        for tie in ties:
            tx, ty, tj = tie
            Vt = V.copy()
            Vt[ty, tj] = Vt[tx, tj]          # the exact indifference
            sigmas, alphas, qs = profile_at(probe, comm, Vt, tie, THETA)
            T = full_transition_matrix(probe, sigmas, qs, None)
            # payoffs that make Vt the true value function of this profile
            u = (np.eye(n_s) - DELTA * T) @ Vt / (1.0 - DELTA)
            game = Game.from_payoffs(players, u)
            V_check = compute_values(game, T, DELTA)
            if np.max(np.abs(V_check - Vt)) > 1e-9:
                continue
            r_ok, _ = verify_responses(game, sigmas, alphas, qs, V_check, atol=ATOL)
            p_ok, _ = verify_proposals(game, sigmas, alphas, qs, V_check, atol=ATOL)
            if not (r_ok and p_ok):
                continue

            # Reject degenerate games: if a player's payoffs barely move across
            # states, every transition is a trivial tie for them and the control
            # tests nothing.
            spread = u.max(axis=0) - u.min(axis=0)
            if spread.min() < 0.1:
                continue

            # The control is only DISCRIMINATING if NO PURE equilibrium exists.
            #
            # An earlier version required only that plain VFI fail cold.  That is
            # not the same thing and does not bite: it tests VFI's search dynamics,
            # so a game where a pure equilibrium exists but VFI simply misses it
            # passes the guard while requiring no mixing at all.  Measured on the
            # first control built this way, alpha = 1.0 verified -- a pure
            # equilibrium -- so "VFI failed cold" proved nothing.
            #
            # Cheap filter first: if pushing the tie to either bound still yields an
            # equilibrium, mixing is not needed here.
            pure_ok = False
            for bound in (0.0, 1.0):
                sg2, al2, qs2 = profile_at(probe, comm, V_check, tie, bound)
                T2 = full_transition_matrix(game, sg2, qs2, None)
                V2 = compute_values(game, T2, DELTA)
                b_r, _ = verify_responses(game, sg2, al2, qs2, V2, atol=ATOL)
                b_p, _ = verify_proposals(game, sg2, al2, qs2, V2, atol=ATOL)
                if b_r and b_p:
                    pure_ok = True
                    break
            if pure_ok:
                continue

            # Then the real test: exhaust the pure strategy space.
            if pure_equilibrium_exists(game, u):
                continue

            return dict(game=game, tie=tie, sigmas=sigmas, alphas=alphas,
                        qs=qs, V=V_check, u=u)
    return None


def main():
    ctrl = None
    for players in (["A", "B"], ["A", "B", "C"]):
        ctrl = build_control(players, seed=0)
        print(f"n={len(players)}: {'FOUND' if ctrl else 'none'}")
        if ctrl:
            break
    if ctrl is None:
        print("\nNo control game could be constructed.")
        return 1

    game = ctrl["game"]
    tx, ty, tj = ctrl["tie"]
    print("\nCONTROL GAME — mixed equilibrium known by construction")
    print(f"  players     : {game.players}    states: {game.n_states}    delta: {DELTA}")
    print(f"  mixing      : voter {game.players[tj]} on state {tx} -> {ty}, "
          f"alpha = {THETA}")
    print(f"  its value gap: {ctrl['V'][ty, tj] - ctrl['V'][tx, tj]:.3e}  (exact tie)")
    print(f"  payoffs:\n{np.round(ctrl['u'], 4)}")
    np.save(REPO / "reports" / "mixed_control_payoffs.npy", ctrl["u"])

    print("\nCan vfi() recover it?")
    for mixed in (False, True):
        Vf, sg, al, q = vfi(game, delta=DELTA, max_iter=300, tol=1e-14,
                            verbose=False, verify_atol=ATOL, mixed_solve=mixed)
        r_ok, _ = verify_responses(game, sg, al, q, Vf, atol=ATOL)
        p_ok, _ = verify_proposals(game, sg, al, q, Vf, atol=ATOL)
        interior = sorted({round(v, 6) for a in al for v in a.values()
                           if 1e-9 < v < 1 - 1e-9})
        tag = "mixed_solve=ON " if mixed else "mixed_solve=OFF"
        print(f"  {tag}: verified={r_ok and p_ok}   interior alphas={interior}")
    print(f"\n  Recovering the control means verified=True with {THETA} among the"
          f"\n  interior alphas. A verified PURE profile is a different equilibrium"
          f"\n  of the same game, not a recovery of this one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
