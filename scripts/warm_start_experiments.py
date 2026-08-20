#!/usr/bin/env python3
"""
Warm-start experiments: how many more (trio, delta) points can we solve?

Baseline is the cold sweep in `reports/delta_homotopy/` plus the cold controls in
`reports/delta_continuation/`. Every point claimed as a gain here is one where a
COLD solve at the identical delta, seed, tolerance and restart count fails.

Four strategies, run in order of expected yield:

  walk      Chained continuation from EVERY boundary of EVERY maximal solved
            region, in BOTH directions. Supersedes delta_continuation_n3.py,
            whose anchor heuristic only walked the outermost boundaries and so
            never explored the edges of interior solved regions.
  anchor    Same targets, but every step seeds from the SAME fixed anchor profile
            rather than from the previous step. Tests whether the gain comes from
            proximity in delta or from chaining.
  cross     Seed an unsolved point from a DIFFERENT trio solved at the same delta,
            matching players by position. What transfers is the shape of the value
            function across coalition structures, not any player's identity.
  bank      Seed an unsolved point from every available solved profile of the SAME
            trio, at any delta, nearest first. Tests long-range transfer.

Usage:
    python scripts/warm_start_experiments.py --label exp1 --strategies walk,anchor,cross,bank
    python scripts/warm_start_experiments.py --report exp1
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "warm_start_experiments"
PROFILE_DIR = OUT_DIR / "profiles"
VERIFIED = re.compile(r"\bVERIFIED\b")

GRID = [0.5, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94,
        0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]


# ---------------------------------------------------------------- known verdicts

def load_cold() -> dict:
    """(trio, delta) -> bool, every COLD verdict measured so far."""
    cold = {}
    for label in ("sweep1", "refine"):
        path = REPO / "reports/delta_homotopy" / f"{label}.json"
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                if r["atol"] == "1e-12":
                    cold[(r["trio"], r["delta"])] = (r["status"] == "solved")
    for name in ("cold_controls.json", "cold_controls2.json"):
        path = REPO / "reports/delta_continuation" / name
        if path.exists():
            for r in json.loads(path.read_text()):
                cold[(r["trio"], r["delta"])] = r["cold_solved"]
    if not cold:
        raise FileNotFoundError(
            "no cold verdicts found; run scripts/delta_homotopy_n3.py first"
        )
    return cold


def known_profiles() -> dict:
    """(trio, delta) -> path, every verified profile already on disk."""
    out = {}
    for d in (REPO / "reports/delta_continuation/profiles",
              REPO / "reports/delta_homotopy/profiles",
              PROFILE_DIR):
        if not d.exists():
            continue
        for f in d.glob("*.xlsx"):
            parts = f.stem.split("_d")
            if len(parts) < 2:
                continue
            trio = parts[0]
            try:
                delta = float(parts[1].split("_")[0])
            except ValueError:
                continue
            out.setdefault((trio, delta), f)
    return out


# ---------------------------------------------------------------------- solving

def solve(trio: str, delta: float, seed_profile: Path | None, args,
          positional: bool = False, tag: str = "") -> tuple[bool, Path, float]:
    out = PROFILE_DIR / f"{trio}_d{delta:g}{tag}.xlsx"
    cmd = [
        sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
        "--payoff-table", str(REPO / f"payoff_tables/kalkuhl_{trio}_2035-2100.xlsx"),
        "--discounting", f"{delta:g}",
        "--effectivity-rule", "heyen_lehtomaa_2021",
        "--solver-approach", "jeres_vfi",
        "--jeres-n-restarts", "1", "--jeres-seed", "42",
        "--jeres-max-iter", str(args.max_iter),
        "--verify-atol", args.atol, "--jeres-tol", f"{float(args.atol)/100:g}",
        "--fresh", "--output", str(out), "--oneline",
    ]
    if seed_profile is not None:
        cmd += ["--jeres-v-init", str(seed_profile)]
        if positional:
            cmd += ["--jeres-v-init-map", "positional"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=args.timeout)
        ok = bool(VERIFIED.search(proc.stdout + proc.stderr))
    except subprocess.TimeoutExpired:
        ok = False
    return ok, out, round(time.time() - start, 1)


# ------------------------------------------------------------------- strategies

def boundaries(trio: str, solved: dict) -> list[tuple[float, int]]:
    """Every edge of every maximal solved run on GRID, with a direction to walk."""
    ds = [d for d in GRID if (trio, d) in solved]
    edges = []
    for i, d in enumerate(ds):
        if not solved[(trio, d)]:
            continue
        prev_unsolved = i == 0 or not solved[(trio, ds[i - 1])]
        next_unsolved = i == len(ds) - 1 or not solved[(trio, ds[i + 1])]
        if next_unsolved:
            edges.append((d, +1))
        if prev_unsolved and i > 0:
            edges.append((d, -1))
    return edges


def run_walk(trio: str, anchor: float, direction: int, args, chained: bool) -> dict:
    ok, profile, secs = solve(trio, anchor, None, args, tag="_anchor")
    steps = [{"delta": anchor, "solved": ok, "seconds": secs, "seed": "cold-anchor"}]
    if not ok:
        return {"trio": trio, "anchor": anchor, "direction": direction,
                "chained": chained, "steps": steps, "stalled_at": anchor}
    fixed = profile
    seed = profile
    delta = anchor
    while True:
        delta = round(delta + args.step * direction, 6)
        if not 0.0 < delta < 1.0 or abs(delta - anchor) > args.max_span:
            return {"trio": trio, "anchor": anchor, "direction": direction,
                    "chained": chained, "steps": steps, "stalled_at": None}
        ok, profile, secs = solve(trio, delta, seed, args,
                                  tag="_w" if chained else "_a")
        steps.append({"delta": delta, "solved": ok, "seconds": secs,
                      "seed": str((seed if chained else fixed).name)})
        if not ok:
            return {"trio": trio, "anchor": anchor, "direction": direction,
                    "chained": chained, "steps": steps, "stalled_at": delta}
        seed = profile if chained else fixed


def run_cross(target: tuple[str, float], sources: list[tuple[str, Path]], args) -> dict:
    """Seed one unsolved point from other trios solved at the same delta."""
    trio, delta = target
    tried = []
    for src_trio, src_path in sources:
        ok, _, secs = solve(trio, delta, src_path, args, positional=True,
                            tag=f"_x{src_trio}")
        tried.append({"source": src_trio, "solved": ok, "seconds": secs})
        if ok:
            break
    return {"trio": trio, "delta": delta, "attempts": tried,
            "solved": any(t["solved"] for t in tried)}


def run_bank(target: tuple[str, float], bank: list[tuple[float, Path]], args) -> dict:
    """Seed one unsolved point from the same trio solved at other deltas."""
    trio, delta = target
    ordered = sorted(bank, key=lambda b: abs(b[0] - delta))[: args.bank_depth]
    tried = []
    for src_delta, src_path in ordered:
        ok, _, secs = solve(trio, delta, src_path, args, tag=f"_b{src_delta:g}")
        tried.append({"source_delta": src_delta, "solved": ok, "seconds": secs})
        if ok:
            break
    return {"trio": trio, "delta": delta, "attempts": tried,
            "solved": any(t["solved"] for t in tried)}


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label")
    ap.add_argument("--report")
    ap.add_argument("--strategies", default="walk,anchor,cross,bank")
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--max-span", type=float, default=0.2)
    ap.add_argument("--atol", default="1e-12")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bank-depth", type=int, default=4)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        print((OUT_DIR / f"{args.report}.txt").read_text())
        return
    if not args.label:
        ap.error("pass --label to run, or --report to re-print")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    # Append every finished result immediately.  A long run that is interrupted
    # must not lose everything it measured -- this happened once.
    jsonl = OUT_DIR / f"{args.label}.jsonl"

    def checkpoint(kind, payload):
        with jsonl.open("a") as fh:
            fh.write(json.dumps({"kind": kind, **payload}) + "\n")

    cold = load_cold()
    profiles = known_profiles()
    trios = sorted({t for t, _ in cold})
    strategies = args.strategies.split(",")
    t0 = time.time()
    result = {"strategies": strategies, "step": args.step, "atol": args.atol}

    if "walk" in strategies or "anchor" in strategies:
        jobs = [(t, d, dirn) for t in trios for d, dirn in boundaries(t, cold)]
        print(f"walk/anchor: {len(jobs)} boundaries", flush=True)
        for chained, key in ((True, "walk"), (False, "anchor")):
            if key not in strategies:
                continue
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(run_walk, t, d, dirn, args, chained)
                        for t, d, dirn in jobs]
                result[key] = []
                for f in futs:
                    r = f.result()
                    result[key].append(r)
                    checkpoint(key, r)
            gained = sum(1 for w in result[key] for s in w["steps"]
                         if s["solved"] and not cold.get((w["trio"], s["delta"]), True))
            print(f"  {key}: {gained} points where cold is known to fail", flush=True)

    unsolved = [(t, d) for t in trios for d in GRID
                if (t, d) in cold and not cold[(t, d)]]

    if "cross" in strategies:
        print(f"cross: {len(unsolved)} unsolved targets", flush=True)
        jobs = []
        for t, d in unsolved:
            srcs = [(o, profiles[(o, d)]) for o in trios
                    if o != t and (o, d) in profiles and cold.get((o, d))]
            if srcs:
                jobs.append(((t, d), srcs))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_cross, tgt, srcs, args) for tgt, srcs in jobs]
            result["cross"] = []
            for f in futs:
                r = f.result()
                result["cross"].append(r)
                checkpoint("cross", r)
        print(f"  cross: {sum(r['solved'] for r in result['cross'])} solved",
              flush=True)

    if "bank" in strategies:
        print(f"bank: {len(unsolved)} unsolved targets", flush=True)
        jobs = []
        for t, d in unsolved:
            bank = [(dd, p) for (tt, dd), p in profiles.items()
                    if tt == t and dd != d and cold.get((tt, dd), True)]
            if bank:
                jobs.append(((t, d), bank))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_bank, tgt, bank, args) for tgt, bank in jobs]
            result["bank"] = []
            for f in futs:
                r = f.result()
                result["bank"].append(r)
                checkpoint("bank", r)
        print(f"  bank: {sum(r['solved'] for r in result['bank'])} solved",
              flush=True)

    result["wall_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / f"{args.label}.json").write_text(json.dumps(result, indent=2))
    print(f"\nwritten: {OUT_DIR / (args.label + '.json')}  "
          f"({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
