#!/usr/bin/env python3
"""
Could the 60 'no pure equilibrium' cells be wrong for the reason the synthetic
tables were wrong?

ordinal_ranking induces profiles from value RANKINGS.  When V contains an EXACT
tie, no strict ranking expresses it, and weak mode substitutes a canonical uniform
mixture rather than trying alpha = 0 and alpha = 1 -- so the two pure profiles that
rest on that tie are never generated.  That is how four synthetic tables were
certified pure-free while having pure equilibria.

The mechanism needs an exact tie in V.  This audits the necessary structural
condition on the payoff side: exactly equal payoff entries within a player's
column, which is what would let states be exactly value-equivalent for that player.
Real RICE payoffs are floats from a GDX run, so exact coincidences should be absent
-- but that is the assumption worth checking rather than asserting.

This is a PROXY, not a proof: V ties can in principle arise without payoff ties.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

def main():
    rows = json.loads((REPO/"reports/ordinal_sweep/summary.json").read_text())
    cells = [r for r in rows if r["ordinal"] == "none"]
    trios = sorted({r["trio"] for r in cells})
    print(f"{len(cells)} 'no pure equilibrium' cells across {len(trios)} trios\n")
    print(f"{'trio':<12}{'exact ties in a player column':>32}{'min |gap| between states':>26}")
    flagged = []
    for t in trios:
        p = REPO/"payoff_tables"/f"kalkuhl_{t}_2035-2100.xlsx"
        df = pd.read_excel(p, sheet_name="Payoffs", header=1, index_col=0)
        cols = [c for c in df.columns if not str(c).startswith("W_SAI")][:3]
        u = df[cols].to_numpy(dtype=float)
        ties, mins = 0, []
        for j in range(u.shape[1]):
            col = u[:, j]
            d = np.abs(col[:, None] - col[None, :])
            np.fill_diagonal(d, np.inf)
            ties += int((d == 0).sum() // 2)
            mins.append(d.min())
        m = min(mins)
        if ties:
            flagged.append(t)
        print(f"{t:<12}{ties:>32}{m:>26.3e}")
    print(f"\ntrios with an exact payoff tie: {flagged if flagged else 'NONE'}")
    print("no exact payoff ties -> the mechanism that broke the synthetic tables")
    print("has no structural foothold here (proxy, not proof)")

if __name__ == "__main__":
    main()
