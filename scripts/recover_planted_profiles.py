#!/usr/bin/env python3
"""
Recover the planted PROFILES for the synthetic control tables.

The generator computes each planted profile (sigma, alpha, V) and then throws it
away, keeping only the payoff table and the mixing probabilities.  That is enough
to state the answer but not enough to test the solver "given the planted profile",
because the pure part -- every other proposer's target and every other voter's 0/1
-- is exactly the piece that turned out to be binding.

Recovering it needs no re-certification.  The generator's RNG is consumed only by
build_one; the pure-free filters consume none.  So replaying the draw sequence with
the same seed reproduces the identical candidate stream, and each kept table can be
identified by matching its payoff matrix against what is on disk.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
spec = importlib.util.spec_from_file_location("gen", REPO/"scripts/generate_mixed_controls.py")
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
sp2 = importlib.util.spec_from_file_location("enu", REPO/"scripts/enumerate_knobs_m1.py")
enu = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(enu)

PL = ["CHN","EUR","USA"]
OUT = REPO/"reports"/"planted_profiles"

def main():
    targets = [int(a) for a in sys.argv[1:]] or [1]
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((REPO/"reports/mixed_controls_manifest.json").read_text())["tables"]
    on_disk = {}
    for rec in man:
        pass
        g = enu.load_game(enu.TAB/rec["file"], PL)
        on_disk[rec["file"]] = np.asarray(g.payoffs, dtype=float)
    print(f"matching {len(on_disk)} tables for M in {targets}")

    # The generator consumed RNG for EVERY target in order, so the replay has to
    # walk the same sequence even for buckets we are not collecting -- skipping
    # target 1 leaves the stream misaligned for target 2 onward.
    per_m = {m: sum(1 for r in man if r["M"] == m) for m in (1, 2, 3, 4)}
    rng = np.random.default_rng(20260819)
    found, draws = {}, 0
    for target in (1, 2, 3, 4):
        matched_here = 0
        for _ in range(4000):
            if matched_here >= per_m.get(target, 0):
                break
            rec = gen.build_one(rng, target)
            draws += 1
            if rec is None or rec["M"] != target:
                continue
            u = np.asarray(rec["game"].payoffs, dtype=float)
            for fname, ref in on_disk.items():
                if fname in found or ref.shape != u.shape:
                    continue
                if np.allclose(u, ref, rtol=0, atol=1e-9):
                    found[fname] = rec
                    matched_here += 1
                    print(f"  matched {fname} after {draws} draws", flush=True)
                    break

    for fname, rec in found.items():
        prof = dict(
            file=fname, M=rec["M"], delta=gen.DELTA,
            V=np.asarray(rec["V"]).tolist(),
            payoffs=np.asarray(rec["game"].payoffs).tolist(),
            knobs=[[int(a), int(b), int(c)] for (a, b, c) in rec["ties"]],
            thetas=[float(rec["thetas"][t]) for t in rec["ties"]],
            sigmas=[{f"{i},{y}": float(v) for (i, y), v in d.items()} for d in rec["sigmas"]],
            alphas=[{f"{j},{y}": float(v) for (j, y), v in d.items()} for d in rec["alphas"]],
        )
        (OUT/f"{fname.replace('.xlsx','')}.json").write_text(json.dumps(prof, indent=2))
    print(f"\nrecovered {len(found)}/{len(on_disk)} profiles -> {OUT}  ({draws} draws)")
    missing = [f for f in on_disk if f not in found]
    if missing:
        print("NOT recovered:", missing)

if __name__ == "__main__":
    main()
