#!/usr/bin/env python3
"""
Given only the KNOB IDENTITY, can we solve the game?

This supplies ONE slot of the support pattern -- which alpha is interior -- and
leaves the other 68 (53 alpha bounds, 15 proposal choices) to be discovered by a
clamped iteration.  It is NOT the "given the support pattern" case the theory
predicts to be easy; see solve_given_support_pattern.py for that, which scores
10/10 on the same tables.

Scoring 0/10 here therefore locates the difficulty in the rest of the support
pattern, not in the scalar solve.
A knob is one mixing degree of freedom: ('acc', x, y, j) means voter j's
acceptance probability for x -> y is free in (0,1).  M = number of knobs.

For each table: clamp the knobs at a grid of values, let everyone else
best-respond (clamped value iteration), and verify with the framework's own
conditions.  Grid cost is ~points^M, which is the wall this is meant to measure.
"""
import json, sys, time, itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
spec = importlib.util.spec_from_file_location("enu", REPO/"scripts/enumerate_knobs_m1.py")
enu = importlib.util.module_from_spec(spec); spec.loader.exec_module(enu)
from lib.equilibrium.jeres_vfi import fw_state_name_to_partition
from lib.equilibrium.jeres_vfi.solver import (compute_values, full_transition_matrix,
    solve_state_mip, verify_responses, verify_proposals)

PL = ["CHN","EUR","USA"]; ATOL = 1e-12; EPS = 1e-9

def knobs_of(rec, game):
    out = []
    for key in rec["planted_thetas"]:
        mv, pl = key.split("|"); a, b = mv.split("->")
        out.append((game.state_idx[fw_state_name_to_partition(a, PL)],
                    game.state_idx[fw_state_name_to_partition(b, PL)], PL.index(pl)))
    return out

def clamped(game, comm, knobs, thetas, delta, iters=60):
    V = np.array(game.payoffs, dtype=float)
    for _ in range(iters):
        sg = [None]*game.n_states; al = [None]*game.n_states
        for x in range(game.n_states):
            a, b, _c, ok = solve_state_mip(game, x, V)
            if not ok:
                a = {(i, x): 1.0 for i in range(game.n_players)}; b = {}
            sg[x], al[x] = dict(a), dict(b)
        for (kx, ky, kj), th in zip(knobs, thetas):
            al[kx][(kj, ky)] = float(th)
        qs = enu.qs_from_alphas(game, comm, al)
        Vn = compute_values(game, full_transition_matrix(game, sg, qs, None), delta)
        if np.max(np.abs(Vn - V)) < 1e-14:
            V = Vn; break
        V = Vn
    return V, sg, al, qs

def main():
    man = json.loads((REPO/"reports/mixed_controls_manifest.json").read_text())["tables"]
    want = sys.argv[1] if len(sys.argv) > 1 else "mixedcontrol_m1"
    pts = int(sys.argv[2]) if len(sys.argv) > 2 else 21
    recs = [r for r in man if r["file"].startswith(want)]
    grid = np.linspace(0.05, 0.95, pts)
    print(f"{'table':<36}{'M':>2}{'evals':>8}{'solved':>8}{'theta hits':>22}{'s':>7}")
    ok_n = 0
    for rec in recs:
        game = enu.load_game(enu.TAB/rec["file"], PL); comm = enu.committees(game)
        kn = knobs_of(rec, game); t = time.time(); hits = []
        combos = list(itertools.product(grid, repeat=len(kn)))
        for th in combos:
            V, sg, al, qs = clamped(game, comm, kn, th, rec.get("delta", 0.9))
            if not all(EPS < al[a].get((c, b), 0.0) < 1-EPS for (a, b, c) in kn):
                continue
            r, _ = verify_responses(game, sg, al, qs, V, atol=ATOL)
            p, _ = verify_proposals(game, sg, al, qs, V, atol=ATOL)
            if r and p:
                hits.append(th)
        dt = time.time()-t; ok_n += bool(hits)
        span = (f"{min(h[0] for h in hits):.2f}-{max(h[0] for h in hits):.2f}"
                f" ({len(hits)}/{len(combos)})") if hits else "-"
        print(f"{rec['file']:<36}{rec['M']:>2}{len(combos):>8}"
              f"{('YES' if hits else 'no'):>8}{span:>22}{dt:>7.1f}", flush=True)
    print(f"\nsolved given the planted support: {ok_n}/{len(recs)}")

if __name__ == "__main__":
    main()
