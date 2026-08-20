#!/usr/bin/env python3
"""
Enumerate the 54 KNOBS for M=1 -- which is NOT enumerating the support pattern.

NOTE: this never completed a run and is not an approach.  Enumerating which alpha is
interior covers 1 of the 69 support-pattern slots; the other 68 are left to a clamped
iteration, which is the part that fails.  Kept for the measurement it produced.

The existing `--jeres-mixed-solve` path nominates candidate mixing transitions from
sign changes in the value gaps across a VFI cycle.  Measured on ten synthetic
tables whose answer is known, that proxy nominated the correct transition in 1 of
10 cases.  Every other refinement -- rank tests, root-finder guards, alternation --
was improving how well the wrong candidates got solved.

At M=1 the candidate set is small enough that no proxy is needed: there are only
as many acceptance knobs as there are (state, voter, target) triples with the voter
on the committee -- 54 for n=3.  Enumerate all of them.

For each candidate knob and each seed value of theta, run a CLAMPED value
iteration: everyone best-responds to the current V except the knob, which is held
at theta.  Then verify with the framework's own conditions.  A verified profile is
an equilibrium regardless of how it was reached.

Recall that theta is generically NOT pinned to a point here: with a single mixer,
the other players' conditions are inequalities, so the solution is an interval.
The scan therefore reports the interval it finds, not a single root.

Usage:
    python scripts/enumerate_knobs_m1.py [--tables mixedcontrol_m1]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium.jeres_vfi import Game, voters  # noqa: E402
from lib.equilibrium.jeres_vfi.solver import (  # noqa: E402
    compute_values,
    full_transition_matrix,
    solve_state_mip,
    verify_proposals,
    verify_responses,
)

TAB = Path("/home/frederik/Code/farsighted-coalitions/payoff_tables")
DELTA = 0.9
ATOL = 1e-12
EPS = 1e-9


def committees(game):
    return {(i, x, y): (game.approval_committees.get((i, x, y), frozenset())
                        if game.approval_committees is not None
                        else voters(game, game.states[x], game.states[y], i))
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


def clamped_vfi(game, comm, knob, theta, delta, iters=60):
    """Best responses to V everywhere except `knob`, which is held at theta."""
    kx, ky, kj = knob
    V = np.array(game.payoffs, dtype=float)
    sigmas = alphas = qs = None
    for _ in range(iters):
        sigmas = [None] * game.n_states
        alphas = [None] * game.n_states
        for x in range(game.n_states):
            a, b, _c, ok = solve_state_mip(game, x, V)
            if not ok:
                a = {(i, x): 1.0 for i in range(game.n_players)}
                b = {}
            sigmas[x], alphas[x] = dict(a), dict(b)
        alphas[kx][(kj, ky)] = float(theta)
        qs = qs_from_alphas(game, comm, alphas)
        V_new = compute_values(
            game, full_transition_matrix(game, sigmas, qs, None), delta)
        if np.max(np.abs(V_new - V)) < 1e-14:
            V = V_new
            break
        V = V_new
    return V, sigmas, alphas, qs


def solve_table(game, delta, seeds):
    comm = committees(game)
    knobs = sorted({(x, y, j) for (i, x, y), c in comm.items()
                    for j in c if x != y})
    for knob in knobs:
        hits = []
        for th in seeds:
            V, sg, al, qs = clamped_vfi(game, comm, knob, th, delta)
            if not (EPS < al[knob[0]].get((knob[2], knob[1]), 0.0) < 1 - EPS):
                continue
            r, _ = verify_responses(game, sg, al, qs, V, atol=ATOL)
            p, _ = verify_proposals(game, sg, al, qs, V, atol=ATOL)
            if r and p:
                hits.append(th)
        if hits:
            return knob, hits, len(knobs)
    return None, [], len(knobs)


def load_game(path, players):
    import pandas as pd
    from lib.equilibrium.jeres_vfi import fw_state_name_to_partition
    df = pd.read_excel(path, sheet_name="Payoffs", header=1, index_col=0)
    probe = Game.from_payoffs(players, np.zeros((5, len(players))))
    u = np.zeros((5, len(players)))
    for name in df.index:
        u[probe.state_idx[fw_state_name_to_partition(str(name).strip(), players)]] = \
            df.loc[name, players].values
    return Game.from_payoffs(players, u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="mixedcontrol_m1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    man = json.loads((REPO / "reports/mixed_controls_manifest.json").read_text())
    recs = [r for r in man["tables"] if r["file"].startswith(args.prefix)]
    seeds = list(np.linspace(0.1, 0.9, args.seeds))
    players = ["CHN", "EUR", "USA"]

    from concurrent.futures import ThreadPoolExecutor

    def one(rec):
        game = load_game(TAB / rec["file"], players)
        t = time.time()
        knob, hits, nk = solve_table(game, rec.get("delta", DELTA), seeds)
        return rec, knob, hits, nk, time.time() - t

    print(f"{'table':<36}{'solved':>8}{'knob found':>26}{'theta hits':>12}{'s':>7}")
    n_ok = 0
    nk = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for rec, knob, hits, nk, dt in pool.map(one, recs):
            if knob:
                n_ok += 1
                span = f"{min(hits):.2f}-{max(hits):.2f}"
                print(f"{rec['file']:<36}{'YES':>8}{str(knob):>26}{span:>12}{dt:>7.1f}",
                      flush=True)
            else:
                print(f"{rec['file']:<36}{'no':>8}{'-':>26}{'-':>12}{dt:>7.1f}",
                      flush=True)
    print(f"\nsolved {n_ok}/{len(recs)}   (support enumerated over {nk} candidate knobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
