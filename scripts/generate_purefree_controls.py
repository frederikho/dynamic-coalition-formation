#!/usr/bin/env python3
"""
Generate synthetic tables that have a planted MIXED equilibrium and NO pure one.

Why the second half matters.  A table with a planted mixed equilibrium is not a
test of anything if the same game also has a pure equilibrium: a solver passes it
without ever mixing.  An earlier batch of 40 such tables was 40/40 non-discriminating
for exactly this reason.

There is no local construction that rules pure equilibria out -- "no pure
equilibrium" is a global property (every self-consistent value ordering must fail),
so it has to be tested, not imposed.  Hence rejection sampling, with a two-stage
filter to keep it affordable:

  stage 1 (cheap, in-process): run cold VFI.  If it verifies, the game HAS a pure
          equilibrium and the candidate is discarded immediately.  Sound as a
          reject: VFI without --jeres-mixed-solve returns pure profiles.
  stage 2 (expensive, subprocess): ordinal_ranking, which searches 0/1 acceptance
          only, so exhausting it proves no pure equilibrium exists.

Only candidates surviving stage 1 reach stage 2, and stage 1 kills the large
majority, so the cost is dominated by the few genuinely interesting games.

Usage:
    python scripts/generate_purefree_controls.py [--draws 1000] [--per-m 10]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

from lib.equilibrium.jeres_vfi.solver import (  # noqa: E402
    compute_values,
    full_transition_matrix,
    verify_proposals,
    verify_responses,
    vfi,
)

spec = importlib.util.spec_from_file_location(
    "gen", REPO / "scripts" / "generate_mixed_controls.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

OUT = gen.OUT
DELTA = gen.DELTA
ATOL = gen.ATOL


def has_pure_via_vfi(rec) -> bool:
    """Cheap sound reject: if cold VFI verifies, a pure equilibrium exists."""
    game = rec["game"]
    try:
        V, sg, al, q = vfi(game, delta=DELTA, max_iter=300, tol=1e-14,
                           verbose=False, verify_atol=ATOL, mixed_solve=False)
    except Exception:
        return False
    r, _ = verify_responses(game, sg, al, q, V, atol=ATOL)
    p, _ = verify_proposals(game, sg, al, q, V, atol=ATOL)
    return bool(r and p)


def has_pure_exhaustive(rec, tmp: Path) -> bool | None:
    """Exhaust the pure space with ordinal_ranking. None on timeout."""
    gen.write_table(rec, tmp, "candidate")
    cmd = [sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
           "--payoff-table", str(tmp), "--discounting", str(DELTA),
           "--effectivity-rule", "heyen_lehtomaa_2021",
           "--solver-approach", "ordinal_ranking", "--verify-atol", "1e-12",
           "--fresh", "--output", "/dev/null", "--oneline", "--quiet"]
    try:
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                             timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    return "VERIFIED" in (out.stdout + out.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000,
                    help="candidate draws per M")
    ap.add_argument("--per-m", type=int, default=10)
    ap.add_argument("--only-m", type=int, default=None,
                    help="top up a single M bucket instead of regenerating all")
    ap.add_argument("--no-clean", action="store_true",
                    help="keep tables already on disk and append to them")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        for old in OUT.glob("mixedcontrol_m*_chneurusa.xlsx"):
            old.unlink()
        print("removed the previous batch\n")

    tmp = REPO / "reports" / "_purefree_candidate.xlsx"
    rng = np.random.default_rng(20260819)
    kept = {1: [], 2: [], 3: [], 4: []}
    stats = {m: dict(drawn=0, vfi_pure=0, exhaust_pure=0, timeout=0, kept=0)
             for m in kept}
    manifest_path = REPO / "reports" / "mixed_controls_manifest.json"
    t0 = time.time()

    targets = (args.only_m,) if args.only_m else (1, 2, 3, 4)
    # Existing tables keep their names; new ones are numbered after them.
    existing = {m: len(list(OUT.glob(f"mixedcontrol_m{m}_*_chneurusa.xlsx")))
                for m in (1, 2, 3, 4)}
    for target in targets:
        for _ in range(args.draws):
            if len(kept[target]) + existing[target] >= args.per_m:
                break
            rec = gen.build_one(rng, target)
            if rec is None or rec["M"] != target:
                continue
            stats[target]["drawn"] += 1
            if has_pure_via_vfi(rec):
                stats[target]["vfi_pure"] += 1
                continue
            res = has_pure_exhaustive(rec, tmp)
            if res is None:
                stats[target]["timeout"] += 1
                continue
            if res:
                stats[target]["exhaust_pure"] += 1
                continue
            kept[target].append(rec)
            stats[target]["kept"] += 1
            print(f"  M={target}  kept {len(kept[target])}/{args.per_m}  "
                  f"(after {stats[target]['drawn']} draws, "
                  f"{time.time()-t0:.0f}s)", flush=True)
        print(f"M={target}: {stats[target]}", flush=True)

    names = gen.state_names()
    manifest = []
    for m, recs in sorted(kept.items()):
        for k, rec in enumerate(recs):
            iv = gen.theta_interval(rec["game"], rec["comm"], rec["V"],
                                    rec["thetas"], rec["sigmas"])
            fname = f"mixedcontrol_m{m}_{k + existing[m]:02d}_chneurusa.xlsx"
            planted = {f"{names[x]}->{names[y]}|{gen.PLAYERS[j]}": round(v, 12)
                       for (x, y, j), v in rec["thetas"].items()}
            note = (f"Synthetic control: M={m} interior mixing probabilities, "
                    f"delta={DELTA}, NO pure equilibrium exists "
                    f"(ordinal_ranking exhausted), "
                    f"{'isolated' if rec['isolated'] else 'positive-dimensional'}; "
                    f"planted {planted}")
            gen.write_table(rec, OUT / fname, note)
            manifest.append(dict(
                file=fname, M=m, delta=DELTA, pure_free=True,
                isolated=bool(rec["isolated"]),
                min_payoff_spread=round(rec["spread"], 6),
                planted_thetas=planted,
                theta_interval=(None if iv is None else
                                [round(iv[0], 12), round(iv[1], 12)]),
                theta_interval_width=(None if iv is None else
                                      round(iv[1] - iv[0], 12)),
            ))
    if args.no_clean and manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        manifest = prior.get("tables", []) + manifest
        stats = {**{int(k): v for k, v in prior.get("stats", {}).items()}, **stats}
    manifest_path.write_text(json.dumps(
        dict(stats={str(k): v for k, v in stats.items()}, tables=manifest), indent=2))
    if tmp.exists():
        tmp.unlink()

    print(f"\n{len(manifest)} pure-free tables written ({time.time()-t0:.0f}s)")
    for m in (1, 2, 3, 4):
        s = stats[m]
        rate = (s["kept"] / s["drawn"] * 100) if s["drawn"] else 0.0
        print(f"  M={m}: kept {s['kept']:>2}  from {s['drawn']:>4} draws "
              f"({rate:.1f}%)   rejected: vfi-pure {s['vfi_pure']}, "
              f"exhaust-pure {s['exhaust_pure']}, timeout {s['timeout']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
