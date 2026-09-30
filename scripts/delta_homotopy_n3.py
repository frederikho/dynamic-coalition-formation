#!/usr/bin/env python3
"""
Model-parameter homotopy in delta for the n=3 kalkuhl benchmark batch.

Question: as the discount factor is varied smoothly, does jeres_vfi keep finding
equilibria? A contiguous band of failures says the hard trios are genuinely
mixed-equilibrium regions the VFI dynamics cannot land on; failures scattered
across delta would instead point at solver flakiness.

Each (trio, delta) is one cold solve. Warm-starting from the previous delta
would be pointless here: VFI in this repo is globally convergent to a unique
fixed point, so initialisation changes speed, not the verdict (CLAUDE.md,
"the multi-start is inert"). The homotopy is therefore in the model parameter,
not in the initialisation.

Usage:
    python scripts/delta_homotopy_n3.py --label run1
    python scripts/delta_homotopy_n3.py --report run1
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
OUT_DIR = REPO / "reports" / "delta_homotopy"
PROFILE_DIR = OUT_DIR / "profiles"

TRIOS = [
    "chneurnde", "chneurrus", "chneurusa", "chnnderus", "chnndeusa",
    "chnrususa", "eurnderus", "eurndeusa", "eurrususa", "nderususa",
]
DELTAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]

VERIFIED = re.compile(r"\bVERIFIED\b")
ITERS = re.compile(r"converged.*?(\d+) iteration", re.IGNORECASE)


def table_for(trio: str) -> Path:
    return REPO / "payoff_tables" / f"kalkuhl_{trio}_2035-2100.xlsx"


def absorbing_states(profile: Path) -> list[str] | None:
    """States the process never leaves: diagonal of P equal to one."""
    if not profile.exists():
        return None
    import pandas as pd
    P = pd.read_excel(profile, sheet_name="Transition Matrix",
                      index_col=0, skiprows=1)
    P.columns = [str(c).strip() for c in P.columns]
    P.index = [str(i).strip() for i in P.index]
    return [s for s in P.index if s in P.columns and float(P.loc[s, s]) > 1 - 1e-9]


def run_one(trio: str, delta: float, atol: str, args) -> dict:
    profile = PROFILE_DIR / (
        f"{trio}_d{delta}_atol{atol}_s{args.seed}"
        + ("" if args.effectivity_rule == "heyen_lehtomaa_2021"
           else f"_{args.effectivity_rule}")
        + ".xlsx"
    )
    cmd = [
        sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
        "--payoff-table", str(table_for(trio)),
        "--discounting", str(delta),
        "--effectivity-rule", args.effectivity_rule,
        "--solver-approach", "jeres_vfi",
        "--jeres-n-restarts", str(args.restarts),
        "--jeres-seed", str(args.seed),
        "--jeres-max-iter", str(args.max_iter),
        "--verify-atol", atol,
        "--jeres-tol", f"{float(atol) / 100:g}",
        "--fresh",
        "--output", str(profile),
        "--oneline",
    ]
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=args.timeout)
        text = proc.stdout + proc.stderr
        solved = bool(VERIFIED.search(text))
        status = "solved" if solved else "failed"
    except subprocess.TimeoutExpired:
        text, solved, status = "", False, "timeout"

    row = {
        "trio": trio, "delta": delta, "atol": atol, "status": status,
        "seconds": round(time.time() - start, 1),
    }
    if solved:
        row["absorbing"] = absorbing_states(profile)
        m = ITERS.search(text)
        if m:
            row["iterations"] = int(m.group(1))
    return row


def sweep(args) -> list[dict]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(t, d) for t in TRIOS for d in DELTAS]
    print(f"pass 1: {len(jobs)} solves at atol {args.atol}, "
          f"{args.workers} workers", flush=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, t, d, args.atol, args) for t, d in jobs]
        for i, f in enumerate(futures, 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['trio']:<10} d={r['delta']:<6} "
                  f"{r['status']:<8} {r['seconds']}s", flush=True)

    # Second pass: every failure retried at the looser tolerance CLAUDE.md
    # allows at high delta, reported separately so the primary verdict stays
    # at the strict tolerance.
    retry = [(r["trio"], r["delta"]) for r in rows if r["status"] != "solved"]
    print(f"\npass 2: {len(retry)} retries at atol {args.retry_atol}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, t, d, args.retry_atol, args)
                   for t, d in retry]
        for i, f in enumerate(futures, 1):
            r = f.result()
            rows.append(r)
            print(f"  [{i}/{len(retry)}] {r['trio']:<10} d={r['delta']:<6} "
                  f"{r['status']:<8} {r['seconds']}s", flush=True)
    return rows


def report(rows: list[dict], atol: str, retry_atol: str) -> str:
    by = {(r["trio"], r["delta"], r["atol"]): r for r in rows}
    mark = {"solved": "o", "failed": ".", "timeout": "T"}

    lines = []
    head = "trio        " + " ".join(f"{d:>6}" for d in DELTAS)
    lines.append(f"VERDICT GRID   (o solved / . failed / T timeout)   atol={atol}")
    lines.append(head)
    for t in TRIOS:
        cells = []
        for d in DELTAS:
            r = by.get((t, d, atol))
            cells.append(f"{mark.get(r['status'], '?') if r else '?':>6}")
        lines.append(f"{t:<12}" + " ".join(cells))

    lines.append("")
    lines.append(f"SAME, RETRIED AT atol={retry_atol} (only where the strict pass failed)")
    lines.append(head)
    for t in TRIOS:
        cells = []
        for d in DELTAS:
            r = by.get((t, d, retry_atol))
            cells.append(f"{(mark.get(r['status'], '?') if r else '-'):>6}")
        lines.append(f"{t:<12}" + " ".join(cells))

    lines.append("")
    lines.append(f"SOLVE RATE BY DELTA (atol={atol})")
    for d in DELTAS:
        n = sum(1 for t in TRIOS
                if (by.get((t, d, atol)) or {}).get("status") == "solved")
        lines.append(f"  delta={d:<7} {n}/{len(TRIOS)}  {'#' * n}")

    lines.append("")
    lines.append("ABSORBING STATES ALONG DELTA (solved points only)")
    for t in TRIOS:
        seq = []
        for d in DELTAS:
            r = by.get((t, d, atol))
            if r and r.get("absorbing") is not None:
                seq.append(f"{d}:{'+'.join(r['absorbing']) or 'none'}")
        lines.append(f"  {t:<12} " + ("  ".join(seq) if seq else "(none solved)"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", help="run the sweep and store it under this name")
    ap.add_argument("--report", help="re-print the report for a stored run")
    ap.add_argument("--atol", default="1e-12")
    ap.add_argument("--retry-atol", default="1e-10")
    ap.add_argument("--restarts", type=int, default=1,
                    help="restarts are structurally inert here; 1 by default")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--effectivity-rule", default="heyen_lehtomaa_2021",
                    help="effectivity rule passed to the solver "
                         "(e.g. adjacent_step)")
    ap.add_argument("--deltas", help="comma-separated delta grid, overrides the default")
    ap.add_argument("--trios", help="comma-separated trio subset, overrides the default")
    args = ap.parse_args()

    global DELTAS, TRIOS
    if args.deltas:
        DELTAS = [float(x) for x in args.deltas.split(",")]
    if args.trios:
        TRIOS = args.trios.split(",")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.report:
        path = OUT_DIR / f"{args.report}.json"
        blob = json.loads(path.read_text())
        print(report(blob["rows"], blob["atol"], blob["retry_atol"]))
        return
    if not args.label:
        ap.error("pass --label to run a sweep, or --report to re-print one")

    t0 = time.time()
    rows = sweep(args)
    blob = {"atol": args.atol, "retry_atol": args.retry_atol,
            "deltas": DELTAS, "trios": TRIOS,
            "max_iter": args.max_iter, "restarts": args.restarts,
            "seed": args.seed,
            "effectivity_rule": args.effectivity_rule,
            "timeout": args.timeout,
            "wall_seconds": round(time.time() - t0, 1), "rows": rows}
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(blob, indent=2))
    text = report(rows, args.atol, args.retry_atol)
    (OUT_DIR / f"{args.label}.txt").write_text(text)
    print("\n" + text)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
