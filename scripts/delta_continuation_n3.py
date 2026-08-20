#!/usr/bin/env python3
"""
Continuation in delta: warm-start each step from the previous step's solution.

The delta sweep (`scripts/delta_homotopy_n3.py`) solves every point cold. This
walks delta in small steps from inside each trio's solved region into its failure
region, seeding each solve with the value function of the previous step's
verified profile.

Why this is not already answered by "the multi-start is inert". That result was
measured on runs that CONVERGE, and it concerns RANDOM restarts, which draw from
a diffuse ball around game.payoffs. The failing tables cycle rather than
converge, and an equilibrium V of the same game at an adjacent delta is a
strategically coherent point that random draws never hit. Neither the uniqueness
argument nor the inertness measurement covers this case.

Both outcomes are informative:
  * the solved region extends -> warm starts matter, and we gain equilibria;
  * it stalls at the same delta as the cold sweep -> evidence that the PURE
    branch genuinely terminates there, and the last verified profile before the
    stall is the concrete starting point for constructing the mixed equilibrium.

Usage:
    python scripts/delta_continuation_n3.py --label cont1
    python scripts/delta_continuation_n3.py --report cont1
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
OUT_DIR = REPO / "reports" / "delta_continuation"
PROFILE_DIR = OUT_DIR / "profiles"
SWEEP = REPO / "reports" / "delta_homotopy"

VERIFIED = re.compile(r"\bVERIFIED\b")


def cold_verdicts(atol: str = "1e-12") -> dict:
    """(trio, delta) -> status, from the cold sweep, as the comparison baseline."""
    rows = []
    for label in ("sweep1", "refine"):
        path = SWEEP / f"{label}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run scripts/delta_homotopy_n3.py first — the "
                "continuation is defined relative to the cold sweep's verdicts."
            )
        rows += json.loads(path.read_text())["rows"]
    return {(r["trio"], r["delta"]): r["status"]
            for r in rows if r["atol"] == atol}


def solve(trio: str, delta: float, v_init: Path | None, args) -> tuple[bool, Path, float]:
    out = PROFILE_DIR / f"{trio}_d{delta:g}.xlsx"
    cmd = [
        sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
        "--payoff-table", str(REPO / f"payoff_tables/kalkuhl_{trio}_2035-2100.xlsx"),
        "--discounting", f"{delta:g}",
        "--effectivity-rule", "heyen_lehtomaa_2021",
        "--solver-approach", "jeres_vfi",
        "--jeres-n-restarts", str(args.restarts),
        "--jeres-seed", "42",
        "--jeres-max-iter", str(args.max_iter),
        "--verify-atol", args.atol,
        "--jeres-tol", f"{float(args.atol) / 100:g}",
        "--fresh", "--output", str(out), "--oneline",
    ]
    if v_init is not None:
        cmd += ["--jeres-v-init", str(v_init)]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=args.timeout)
        ok = bool(VERIFIED.search(proc.stdout + proc.stderr))
    except subprocess.TimeoutExpired:
        ok = False
    return ok, out, round(time.time() - start, 1)


def walk(trio: str, start_delta: float, direction: int, args, cold: dict) -> dict:
    """Step from start_delta in `direction` until a step fails to verify."""
    step = args.step * direction
    v_init: Path | None = None
    steps = []
    delta = start_delta

    # Seed the walk by re-solving the known-good anchor cold.
    ok, profile, secs = solve(trio, delta, None, args)
    steps.append({"delta": delta, "warm": False, "solved": ok, "seconds": secs})
    if not ok:
        return {"trio": trio, "direction": direction, "anchor": start_delta,
                "steps": steps, "stalled_at": delta,
                "note": "anchor did not re-solve cold; walk not attempted"}
    v_init = profile

    while True:
        delta = round(delta + step, 6)
        if not 0.0 < delta < 1.0 or abs(delta - start_delta) > args.max_span:
            return {"trio": trio, "direction": direction, "anchor": start_delta,
                    "steps": steps, "stalled_at": None,
                    "note": "walked to the end of the span without stalling"}
        ok, profile, secs = solve(trio, delta, v_init, args)
        steps.append({"delta": delta, "warm": True, "solved": ok,
                      "seconds": secs,
                      "cold_status": cold.get((trio, delta), "not-in-sweep")})
        if not ok:
            return {"trio": trio, "direction": direction, "anchor": start_delta,
                    "steps": steps, "stalled_at": delta,
                    "note": "warm start failed here"}
        v_init = profile


def anchors(cold: dict, args) -> list[tuple[str, float, int]]:
    """Highest solved delta below each trio's failure region, and vice versa."""
    trios = sorted({t for t, _ in cold})
    jobs = []
    for t in trios:
        ds = sorted(d for (tt, d) in cold if tt == t)
        solved = [d for d in ds if cold[(t, d)] == "solved"]
        failed = [d for d in ds if cold[(t, d)] != "solved"]
        if not failed or not solved:
            continue
        below = [d for d in solved if d < min(failed)]
        if below:
            jobs.append((t, max(below), +1))
        if args.both_directions:
            above = [d for d in solved if d > max(failed)]
            if above:
                jobs.append((t, min(above), -1))
    return jobs


def report(blob: dict) -> str:
    lines = ["CONTINUATION vs COLD SWEEP", ""]
    gained = 0
    for w in blob["walks"]:
        arrow = "up" if w["direction"] > 0 else "down"
        lines.append(f"{w['trio']}  ({arrow} from delta={w['anchor']})  — {w['note']}")
        for s in w["steps"]:
            if not s["warm"]:
                lines.append(f"    {s['delta']:<8} anchor      "
                             f"{'ok' if s['solved'] else 'FAILED':<8} {s['seconds']}s")
                continue
            cold_s = s.get("cold_status", "?")
            flag = ""
            if s["solved"] and cold_s != "solved":
                flag = "   <-- WARM START GAINED THIS POINT"
                gained += 1
            lines.append(f"    {s['delta']:<8} warm        "
                         f"{'ok' if s['solved'] else 'stall':<8} "
                         f"{s['seconds']}s   cold={cold_s}{flag}")
        lines.append("")
    lines.append(f"points gained over the cold sweep: {gained}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label")
    ap.add_argument("--report")
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--max-span", type=float, default=0.06,
                    help="stop a walk after this much total movement in delta")
    ap.add_argument("--atol", default="1e-12")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--both-directions", action="store_true", default=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        print(report(json.loads((OUT_DIR / f"{args.report}.json").read_text())))
        return
    if not args.label:
        ap.error("pass --label to run, or --report to re-print")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    cold = cold_verdicts(args.atol)
    jobs = anchors(cold, args)
    print(f"{len(jobs)} walks, step {args.step}, span {args.max_span}", flush=True)

    t0 = time.time()
    walks = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(walk, t, d, dirn, args, cold) for t, d, dirn in jobs]
        for f in futures:
            w = f.result()
            walks.append(w)
            print(f"  {w['trio']:<10} dir={w['direction']:+d} "
                  f"anchor={w['anchor']} stalled_at={w['stalled_at']}", flush=True)

    blob = {"step": args.step, "max_span": args.max_span, "atol": args.atol,
            "wall_seconds": round(time.time() - t0, 1), "walks": walks}
    (OUT_DIR / f"{args.label}.json").write_text(json.dumps(blob, indent=2))
    text = report(blob)
    (OUT_DIR / f"{args.label}.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
