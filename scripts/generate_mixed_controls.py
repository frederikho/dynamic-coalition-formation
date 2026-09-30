#!/usr/bin/env python3
"""
Generate synthetic payoff tables whose mixed equilibria are known by construction.

Ten tables each for M = 1, 2, 3, 4, where M counts the interior mixing
probabilities in the planted equilibrium.

Construction (same idea as scripts/mixed_control_game.py, generalised to M ties):
work backwards from the value function rather than forwards from a profile.

  1. Draw a value function V, then force M exact ties V_j(y) = V_j(x).
  2. Derive the profile that best-responds to that V.  Acceptances are pinned by
     the sign of their value gap except on the ties, which are free: set them to
     chosen probabilities.  Proposers take the argmax.
  3. Choose payoffs making that V the true value function of the resulting T:
         u = (I - delta*T) V / (1 - delta)

V is then both the value function of the profile and what the profile
best-responds to, with M players' probabilities strictly interior -- a mixed
equilibrium, certified without solving anything.

Two properties are recorded per table because they change what a solver faces:

  isolated / positive-dimensional.  A mixer's own indifference condition is FLAT
  in their own probability -- the standard indifference principle: a player who
  mixes is indifferent, so their own payoff cannot depend on their own weights.
  Their probability is pinned by OTHER players' conditions.  When only one player
  mixes, those conditions are inequalities and the solution is an INTERVAL; when
  several mix, they become equations and the solution is a POINT.

  theta interval width, for the M = 1 tables, measured by root-finding the other
  players' binding conditions.

Payoff spread is varied by a per-player affine rescale.  That is an exact symmetry
of the equilibrium set, so it changes nothing mathematically -- which is the point:
a solver that is not invariant to it is broken, and these tables will show it.

Usage:
    python scripts/generate_mixed_controls.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium.jeres_vfi import (  # noqa: E402
    Game,
    all_partitions,
    fw_state_name_to_partition,
    voters,
)
from lib.equilibrium.jeres_vfi.solver import (  # noqa: E402
    compute_values,
    full_transition_matrix,
    verify_proposals,
    verify_responses,
)

DELTA = 0.9
ATOL = 1e-12
PLAYERS = ["CHN", "EUR", "USA"]
OUT = Path("/home/frederik/Code/farsighted-coalitions/payoff_tables")
PER_M = 10
EPS = 1e-9


def committees(game):
    """Approval committees under the FRAMEWORK effectivity rule.

    jeres_vfi's own voters() disagrees with heyen_lehtomaa_2021 on 9 of the 60
    transitions -- it gives unilateral exits an EMPTY committee where the framework
    puts the proposer in it -- so building controls against voters() produces games
    that differ from the ones the CLI actually solves.
    """
    from lib.effectivity import get_effectivity
    names = state_names()
    eff = get_effectivity("heyen_lehtomaa_2021", PLAYERS, names)
    idx = {game.state_idx[fw_state_name_to_partition(n, PLAYERS)]: n for n in names}
    return {(i, x, y): frozenset(
                k for k, pk in enumerate(PLAYERS)
                if eff.get((PLAYERS[i], idx[x], idx[y], pk), 0) == 1)
            for i in range(game.n_players)
            for x in range(game.n_states)
            for y in range(game.n_states)}


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


def profile_at(game, comm, V, thetas):
    """Best responses to V, with the tied acceptances set to the given values."""
    n_s, n_p = game.n_states, game.n_players
    alphas = []
    for x in range(n_s):
        alp = {}
        for i in range(n_p):
            for y in range(n_s):
                for j in comm[(i, x, y)]:
                    if (x, y, j) in thetas:
                        alp[(j, y)] = float(thetas[(x, y, j)])
                    else:
                        alp[(j, y)] = 1.0 if V[y, j] - V[x, j] > 0 else 0.0
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


def n_interior(alphas):
    return sum(1 for d in alphas for v in d.values() if EPS < v < 1 - EPS)


def is_isolated(game, comm, V, thetas, sigmas, step=1e-4):
    """Perturb each mixing probability; isolated if every perturbation breaks it."""
    for key in thetas:
        for d in (+step, -step):
            t2 = dict(thetas)
            t2[key] = float(np.clip(thetas[key] + d, EPS, 1 - EPS))
            _, al2, qs2 = profile_at(game, comm, V, t2)
            T2 = full_transition_matrix(game, sigmas, qs2, None)
            V2 = compute_values(game, T2, DELTA)
            r, _ = verify_responses(game, sigmas, al2, qs2, V2, atol=ATOL)
            p, _ = verify_proposals(game, sigmas, al2, qs2, V2, atol=ATOL)
            if r and p:
                return False
    return True


def theta_interval(game, comm, V, thetas, sigmas):
    """For a single mixing probability, the interval of values that still verify."""
    if len(thetas) != 1:
        return None
    key = next(iter(thetas))

    def ok(t):
        t2 = {key: float(t)}
        _, al, qs = profile_at(game, comm, V, t2)
        T = full_transition_matrix(game, sigmas, qs, None)
        Vv = compute_values(game, T, DELTA)
        r, _ = verify_responses(game, sigmas, al, qs, Vv, atol=ATOL)
        p, _ = verify_proposals(game, sigmas, al, qs, Vv, atol=ATOL)
        return r and p

    grid = np.linspace(1e-6, 1 - 1e-6, 601)
    good = [t for t in grid if ok(t)]
    if not good:
        return None
    lo, hi = min(good), max(good)
    for a, b, want_lo in ((max(1e-6, lo - 2e-3), lo, True), (hi, min(1 - 1e-6, hi + 2e-3), False)):
        try:
            edge = brentq(lambda t: 1.0 if ok(t) else -1.0, a, b, xtol=1e-12)
            if want_lo:
                lo = edge
            else:
                hi = edge
        except Exception:
            pass
    return float(lo), float(hi)


def build_one(rng, n_ties):
    """One candidate table, or None."""
    n_p = len(PLAYERS)
    n_s = len(all_partitions(n_p))
    probe = Game.from_payoffs(PLAYERS, np.zeros((n_s, n_p)))
    comm = committees(probe)
    cands = sorted({(x, y, j) for (i, x, y), c in comm.items() for j in c if x != y})

    V = rng.uniform(-1.0, 1.0, size=(n_s, n_p))

    # Spread the ties over DISTINCT players where possible.
    #
    # This is what makes the control discriminating.  A player who mixes is
    # indifferent, so their own indifference condition is flat in their own
    # probability: put several knobs on one player and those equations are
    # degenerate, the solution set is a positive-dimensional region, and a solver
    # "succeeds" by landing anywhere inside it -- on the previous batch theta=0.5
    # already verified for 8/10 tables at M=1 and a random point verified 61% of
    # the time.  Knobs on DIFFERENT players couple through the off-diagonal terms,
    # the Jacobian is generically invertible, and the solution is an isolated
    # point that has to actually be found.
    by_player = {}
    for c in cands:
        by_player.setdefault(c[2], []).append(c)
    order = list(by_player)
    rng.shuffle(order)
    ties = []
    while len(ties) < n_ties:
        progressed = False
        for j in order:
            if len(ties) >= n_ties:
                break
            pool = [c for c in by_player[j] if c not in ties]
            if not pool:
                continue
            ties.append(pool[int(rng.integers(len(pool)))])
            progressed = True
        if not progressed:
            return None
    for (x, y, j) in ties:
        V[y, j] = V[x, j]
    thetas = {t: float(rng.uniform(0.25, 0.75)) for t in ties}

    # Rescale BEFORE deriving the profile.  profile_at picks proposals with an
    # ABSOLUTE 1e-15 threshold, so applying the affine map afterwards can push a
    # sub-threshold gain across it and leave the stored proposals inconsistent with
    # the payoffs actually written out.  A positive affine map per player leaves the
    # equilibrium set unchanged, so doing it first costs nothing.
    scale = np.exp(rng.uniform(-2.5, 2.5, size=n_p))
    shift = rng.uniform(-5.0, 5.0, size=n_p)
    V = V * scale + shift

    sigmas, alphas, qs = profile_at(probe, comm, V, thetas)
    T = full_transition_matrix(probe, sigmas, qs, None)
    u = (np.eye(n_s) - DELTA * T) @ V / (1.0 - DELTA)

    game = Game.from_payoffs(PLAYERS, u)
    V_check = compute_values(game, T, DELTA)
    if np.max(np.abs(V_check - V)) > 1e-7 * max(1.0, np.abs(V).max()):
        return None
    r, _ = verify_responses(game, sigmas, alphas, qs, V_check, atol=ATOL)
    p, _ = verify_proposals(game, sigmas, alphas, qs, V_check, atol=ATOL)
    if not (r and p):
        return None
    m = n_interior(alphas)
    if m == 0:
        return None
    spread = float((u.max(axis=0) - u.min(axis=0)).min())
    if spread < 1e-6:
        return None
    return dict(game=game, u=u, V=V_check, thetas=thetas, ties=ties, M=m,
                sigmas=sigmas, alphas=alphas, qs=qs, comm=comm, spread=spread,
                isolated=is_isolated(game, comm, V_check, thetas, sigmas))


def state_names():
    a, b, c = PLAYERS
    return ["( )", f"({a}{b})", f"({a}{c})", f"({b}{c})", f"({a}{b}{c})"]


def write_table(rec, path, note):
    game = rec["game"]
    rows = {n: rec["u"][game.state_idx[fw_state_name_to_partition(n, PLAYERS)]]
            for n in state_names()}
    df = pd.DataFrame.from_dict(rows, orient="index", columns=PLAYERS)
    df.index.name = "state"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Payoffs", startrow=1)
        xl.sheets["Payoffs"].cell(row=1, column=1).value = note
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260819)
    buckets = {1: [], 2: [], 3: [], 4: []}
    tries = 0
    while any(len(v) < PER_M for v in buckets.values()) and tries < 200000:
        tries += 1
        rec = build_one(rng, int(rng.integers(1, 5)))
        if rec is None:
            continue
        m = rec["M"]
        if m in buckets and len(buckets[m]) < PER_M:
            buckets[m].append(rec)

    manifest = []
    names = state_names()
    for m, recs in sorted(buckets.items()):
        for k, rec in enumerate(recs):
            iv = theta_interval(rec["game"], rec["comm"], rec["V"],
                                rec["thetas"], rec["sigmas"])
            fname = f"mixedcontrol_m{m}_{k:02d}_chneurusa.xlsx"
            planted = {f"{names[x]}->{names[y]}|{PLAYERS[j]}": round(v, 12)
                       for (x, y, j), v in rec["thetas"].items()}
            note = (f"Synthetic control, M={m} interior mixing probabilities, "
                    f"delta={DELTA}, "
                    f"{'isolated solution' if rec['isolated'] else 'positive-dimensional'}; "
                    f"planted {planted}")
            write_table(rec, OUT / fname, note)
            manifest.append(dict(
                file=fname, M=m, delta=DELTA, isolated=bool(rec["isolated"]),
                min_payoff_spread=round(rec["spread"], 6),
                planted_thetas=planted,
                theta_interval=(None if iv is None else
                                [round(iv[0], 12), round(iv[1], 12)]),
                theta_interval_width=(None if iv is None else round(iv[1] - iv[0], 12)),
            ))
            print(f"  {fname}  M={m}  spread={rec['spread']:.3g}  "
                  f"{'isolated' if rec['isolated'] else 'continuum'}"
                  f"{'' if iv is None else f'  theta in [{iv[0]:.6f}, {iv[1]:.6f}] '
                                           f'(width {iv[1]-iv[0]:.2e})'}")

    (REPO / "reports" / "mixed_controls_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} tables written to {OUT}")
    print(f"manifest: reports/mixed_controls_manifest.json  ({tries} draws)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
