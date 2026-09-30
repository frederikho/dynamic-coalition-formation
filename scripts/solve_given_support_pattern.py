#!/usr/bin/env python3
"""
Given the SUPPORT PATTERN, solve for the mixing probabilities.

Terminology, kept precise because conflating these cost real time:

  knob identity    which single alpha is interior -- ONE of 69 slots.
  SUPPORT PATTERN  the full active set: for each of the 15 (proposer, state) pairs,
                   which targets carry sigma > 0; and for each of the 54 alphas,
                   whether it is 0, 1, or interior.  Everything except the M
                   interior VALUES follows from this.
  profile          the support pattern PLUS those M values.

This script supplies the support pattern and solves for the values -- the case the
theory predicts to be easy at M=1.  It is 10/10 in milliseconds, confirming that.

Contrast scripts/solve_given_knob.py, which supplies only the knob identity (1 of
69 slots) and scores 0/10.  The gap between the two is where the difficulty lives.
"""
import json, sys, time, itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium import mixed_controls as enu  # noqa: E402
import importlib.util
from lib.equilibrium.jeres_vfi import Game
from lib.equilibrium.jeres_vfi.solver import (compute_values, full_transition_matrix,
    verify_responses, verify_proposals)

PL=["CHN","EUR","USA"]; ATOL=1e-12; EPS=1e-9
PROF = REPO/"reports"/"planted_profiles"

def load(p):
    d = json.loads(p.read_text())
    game = Game.from_payoffs(PL, np.array(d["payoffs"], dtype=float))
    sig = [{tuple(int(v) for v in k.split(",")): val for k, val in dd.items()}
           for dd in d["sigmas"]]
    alp = [{tuple(int(v) for v in k.split(",")): val for k, val in dd.items()}
           for dd in d["alphas"]]
    return d, game, sig, alp

def main():
    pts = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    files = sorted(PROF.glob("*.json"))
    grid = np.linspace(0.02, 0.98, pts)
    print(f"{'table':<36}{'M':>2}{'evals':>7}{'solved':>8}{'theta hits':>26}{'s':>7}")
    ok = 0
    for f in files:
        d, game, sig, alp = load(f)
        comm = enu.committees(game)
        knobs = [tuple(k) for k in d["knobs"]]
        t = time.time(); hits = []
        for th in itertools.product(grid, repeat=len(knobs)):
            a2 = [dict(x) for x in alp]
            for (kx, ky, kj), v in zip(knobs, th):
                a2[kx][(kj, ky)] = float(v)
            qs = enu.qs_from_alphas(game, comm, a2)
            V = compute_values(game, full_transition_matrix(game, sig, qs, None),
                               d["delta"])
            r, _ = verify_responses(game, sig, a2, qs, V, atol=ATOL)
            p, _ = verify_proposals(game, sig, a2, qs, V, atol=ATOL)
            if r and p:
                hits.append(th)
        dt = time.time()-t; ok += bool(hits)
        span = (f"{min(h[0] for h in hits):.2f}-{max(h[0] for h in hits):.2f}"
                f"  ({len(hits)} of {pts**len(knobs)})") if hits else "-"
        print(f"{d['file'][:35]:<36}{d['M']:>2}{pts**len(knobs):>7}"
              f"{('YES' if hits else 'no'):>8}{span:>26}{dt:>7.1f}", flush=True)
    print(f"\nsolved given the SUPPORT PATTERN: {ok}/{len(files)}")
    print("(the planted interior values are loaded but overwritten by the scan,\n  so only the support pattern -- pinned alpha bounds and pure sigma targets --\n  is actually used; no profile is written out)")

if __name__ == "__main__":
    main()
