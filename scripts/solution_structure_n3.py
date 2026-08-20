#!/usr/bin/env python3
"""
Classify every solved profile as pure or mixed, and track how the solution
changes along delta within a trio.

Pure  : every non-NaN strategy entry is exactly 0 or 1.
Mixed : some entry lies strictly inside (0,1).  Reported separately for
        PROPOSITION rows (mixing over which state to propose) and ACCEPTANCE
        rows (mixing over whether to consent), because the solver reaches these
        two by different routes.

Solution identity is the rounded strategy matrix: two deltas share an id when
their equilibrium strategies agree entrywise.  This shows whether the solution
drifts continuously in delta or is piecewise constant with jumps.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EPS = 1e-9


def solved_points() -> dict:
    """(trio, delta) -> profile path, for every VERIFIED profile we produced."""
    out = {}
    for label, d in (("sweep1", "delta_homotopy"), ("refine", "delta_homotopy")):
        p = REPO / f"reports/{d}/{label}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            if r["atol"] == "1e-12" and r["status"] == "solved":
                for name in (f"{r['trio']}_d{r['delta']}_atol1e-12_s42.xlsx",
                             f"{r['trio']}_d{r['delta']}_atol1e-12.xlsx"):
                    f = REPO / "reports/delta_homotopy/profiles" / name
                    if f.exists():
                        out[(r["trio"], r["delta"])] = f
                        break
    cont = REPO / "reports/delta_continuation/cont2.json"
    if cont.exists():
        for w in json.loads(cont.read_text())["walks"]:
            for s in w["steps"]:
                if not s["solved"]:
                    continue
                f = REPO / "reports/delta_continuation/profiles" / f"{w['trio']}_d{s['delta']:g}.xlsx"
                if f.exists():
                    out.setdefault((w["trio"], s["delta"]), f)
    return out


def classify(path: Path) -> dict:
    df = pd.read_excel(path, sheet_name="Strategy", header=[0, 1], index_col=[0, 1, 2])
    vals = df.to_numpy(dtype=float)
    kinds = np.array([str(i[1]) for i in df.index])

    def frac(mask):
        v = vals[mask]
        v = v[~np.isnan(v)]
        if v.size == 0:
            return 0, 0
        interior = ((v > EPS) & (v < 1 - EPS)).sum()
        return int(interior), int(v.size)

    prop_mix, prop_n = frac(kinds == "Proposition")
    acc_mix, acc_n = frac(kinds == "Acceptance")
    digest = hashlib.sha1(
        np.nan_to_num(np.round(vals, 6), nan=-1).tobytes()
    ).hexdigest()[:6]
    return {"prop_mixed": prop_mix, "prop_n": prop_n,
            "acc_mixed": acc_mix, "acc_n": acc_n,
            "pure": (prop_mix == 0 and acc_mix == 0), "id": digest}


def main():
    pts = solved_points()
    rows = []
    for (trio, delta), path in sorted(pts.items()):
        try:
            rows.append({"trio": trio, "delta": delta, **classify(path)})
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
    df = pd.DataFrame(rows).sort_values(["trio", "delta"])

    ids = {}
    for trio, g in df.groupby("trio"):
        for i, d in enumerate(dict.fromkeys(g["id"])):
            ids[(trio, d)] = chr(ord("A") + i)
    df["sol"] = [ids[(r.trio, r.id)] for r in df.itertuples()]

    out = REPO / "reports/solution_structure.csv"
    df.to_csv(out, index=False)

    print("SOLUTION IDENTITY AND PURITY ALONG DELTA")
    print("  letter = distinct strategy profile within that trio")
    print("  *      = MIXED (some strategy strictly between 0 and 1)\n")
    for trio, g in df.groupby("trio"):
        cells = " ".join(
            f"{r.delta:g}:{r.sol}{'*' if not r.pure else ''}" for r in g.itertuples()
        )
        print(f"  {trio:<12} {cells}")

    print(f"\nprofiles classified: {len(df)}")
    print(f"pure  : {int(df['pure'].sum())}")
    print(f"mixed : {int((~df['pure']).sum())}")
    if (~df["pure"]).any():
        m = df[~df["pure"]]
        print(f"  of which mixing in PROPOSITIONS: {int((m['prop_mixed']>0).sum())}")
        print(f"  of which mixing in ACCEPTANCES : {int((m['acc_mixed']>0).sum())}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
