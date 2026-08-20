#!/usr/bin/env python3
"""
Fill the M=1 bucket to 10 with a relaxed hardness threshold.

Hard M=1 games are rare by construction: one knob on one player, and that player's
own indifference is flat in their own probability, so the feasible set is an
interval carved out by other players' INEQUALITIES -- usually a wide one.  At the
0.10 random-hit threshold only 3 survived 6,000 draws.

The M>=2 buckets carry the discriminating cases (0.0% random hit rate).  These
seats are filled with easier tables so the bucket is complete; the manifest records
each table's measured hit rate so nobody mistakes an easy one for a hard one.

All the correctness guards are unchanged: the planted profile must verify under the
FRAMEWORK committee rule, all 2^M pure variants must fail, and ordinal_ranking must
find no pure equilibrium.
"""
import json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
sp = importlib.util.spec_from_file_location("rb", REPO/"scripts/rebuild_mixing_controls.py")
rb = importlib.util.module_from_spec(sp); sp.loader.exec_module(rb)
gen, enu = rb.gen, rb.enu

TARGET = 10
PROF = REPO/"reports"/"planted_profiles_v3"
OUT = gen.OUT

def hit_rate(rec, samples=120):
    u = np.asarray(rec["game"].payoffs, float)
    g, _ = rb.framework_game(u); comm = enu.committees(g)
    M = len(rec["ties"]); rng = np.random.default_rng(0)
    pts = [np.full(M, 0.5)] + [rng.uniform(0, 1, M) for _ in range(samples)]
    hits = 0
    for th in pts:
        a2 = [dict(x) for x in rec["alphas"]]
        for (kx, ky, kj), t in zip(rec["ties"], th):
            a2[kx][(kj, ky)] = float(t)
        qs = enu.qs_from_alphas(g, comm, a2)
        V = rb.compute_values(g, rb.full_transition_matrix(g, rec["sigmas"], qs, None), rb.DELTA)
        r, _ = rb.verify_responses(g, rec["sigmas"], a2, qs, V, atol=rb.EPS)
        p, _ = rb.verify_proposals(g, rec["sigmas"], a2, qs, V, atol=rb.EPS)
        hits += bool(r and p)
    return hits/len(pts)

def main():
    have = sorted(PROF.glob("mixedcontrol_m1_*.json"))
    need = TARGET - len(have)
    print(f"M=1 has {len(have)}, need {need} more (relaxed hardness)")
    if need <= 0:
        return 0
    tmp = REPO/"reports"/"_topup_candidate.xlsx"
    rng = np.random.default_rng(777)
    kept, drawn = [], 0
    t0 = time.time()
    while len(kept) < need and drawn < 20000:
        rec = gen.build_one(rng, 1)
        drawn += 1
        if rec is None or rec["M"] != 1:
            continue
        if not rb.planted_verifies(rec):      continue
        if rb.has_pure_variant(rec):          continue
        if not rb.or_says_no_pure(rec, tmp):  continue
        hr = hit_rate(rec)
        kept.append((rec, hr))
        print(f"  kept {len(kept)}/{need}  hit_rate={hr:.1%}  "
              f"(draw {drawn}, {time.time()-t0:.0f}s)", flush=True)

    man = json.loads((REPO/"reports"/"mixed_controls_manifest.json").read_text())
    names = gen.state_names()
    start = len(have)
    for k, (rec, hr) in enumerate(kept):
        fname = f"mixedcontrol_m1_{start+k:02d}_chneurusa.xlsx"
        planted = {f"{names[x]}->{names[y]}|{gen.PLAYERS[j]}": round(v, 12)
                   for (x, y, j), v in rec["thetas"].items()}
        gen.write_table(rec, OUT/fname,
            f"Synthetic control: M=1, delta={rb.DELTA}, mixing REQUIRED; "
            f"random-guess hit rate {hr:.1%}; planted {planted}")
        (PROF/f"{fname.replace('.xlsx','')}.json").write_text(json.dumps(dict(
            file=fname, M=1, delta=rb.DELTA,
            payoffs=np.asarray(rec["game"].payoffs).tolist(),
            V=np.asarray(rec["V"]).tolist(),
            knobs=[[int(a), int(b), int(c)] for a, b, c in rec["ties"]],
            thetas=[float(rec["thetas"][t]) for t in rec["ties"]],
            sigmas=[{f"{i},{y}": float(v) for (i, y), v in d.items()} for d in rec["sigmas"]],
            alphas=[{f"{j},{y}": float(v) for (j, y), v in d.items()} for d in rec["alphas"]],
        ), indent=2))
        man["tables"].append(dict(file=fname, M=1, delta=rb.DELTA, mixing_required=True,
                                  planted_thetas=planted, random_hit_rate=round(hr, 4),
                                  min_payoff_spread=round(rec["spread"], 6)))
    (REPO/"reports"/"mixed_controls_manifest.json").write_text(json.dumps(man, indent=2))
    if tmp.exists(): tmp.unlink()
    print(f"\nadded {len(kept)} tables ({drawn} draws, {time.time()-t0:.0f}s)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
