#!/usr/bin/env python3
"""
Head-to-head comparison of solver approaches on a fixed set of payoff tables.

Unlike `bench_solver.py` (which hardcodes jeres_vfi and tracks whether a *change*
to that one solver helps), this runs SEVERAL solvers over the SAME tables at the
SAME settings, so the resulting numbers are comparable across solvers.

Defaults target the n=3 kalkuhl benchmark batch: the ten CHN/EUR/NDE/RUS/USA
trios generated 2026-08-14, at delta = 0.99 with per-player payoff normalisation
and a fixed --verify-atol (see CLAUDE.md: atol is a numerical parameter; 1e-12 is
far above the V-solve noise floor and far below the V spread, so it is a strict
test that is identical for every table).

A timeout is NOT a failure. It is reported as its own status and kept out of the
failure count.

Usage:
    python scripts/compare_solvers.py run --label kalkuhl_d99
    python scripts/compare_solvers.py report kalkuhl_d99
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
OUT_DIR = REPO / "reports" / "compare_solvers"

# The ten C(5,3) trios of the n=3 benchmark batch (tables generated 2026-08-14).
KALKUHL_N3 = [
    "kalkuhl_chneurnde_2035-2100.xlsx",
    "kalkuhl_chneurrus_2035-2100.xlsx",
    "kalkuhl_chneurusa_2035-2100.xlsx",
    "kalkuhl_chnnderus_2035-2100.xlsx",
    "kalkuhl_chnndeusa_2035-2100.xlsx",
    "kalkuhl_chnrususa_2035-2100.xlsx",
    "kalkuhl_eurnderus_2035-2100.xlsx",
    "kalkuhl_eurndeusa_2035-2100.xlsx",
    "kalkuhl_eurrususa_2035-2100.xlsx",
    "kalkuhl_nderususa_2035-2100.xlsx",
]

DEFAULT_SOLVERS = ["support_enumeration", "active_set", "jeres_vfi", "merit_descent"]

ONELINE = re.compile(r"\|\s*(VERIFIED|FAILED)\s*\|\s*([0-9.]+)s(?:\s*\|\s*([^|]+))?")


def solver_flags(solver: str, args) -> list[str]:
    """Per-solver budget flags. Kept explicit so the report can record them."""
    if solver == "jeres_vfi":
        # CLAUDE.md: the multi-start is inert (bit-identical fixed point from far
        # apart inits), so a small restart budget costs nothing; max_iter matters
        # because cycle detection + bisection needs room.
        return ["--jeres-n-restarts", str(args.jeres_restarts),
                "--jeres-max-iter", str(args.jeres_max_iter)]
    if solver == "merit_descent":
        return ["--merit-restarts", str(args.merit_restarts),
                "--merit-walk", str(args.merit_walk)]
    return []


def run_one(table: Path, solver: str, args) -> dict:
    cmd = [
        sys.executable, "-m", "lib.equilibrium.find", args.scenario,
        "--payoff-table", str(table),
        "--solver-approach", solver,
        "--verify-atol", args.atol,
        "--quiet", "--oneline",
    ] + solver_flags(solver, args)

    start = time.time()
    reason = None
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=args.timeout)
        text = proc.stdout + proc.stderr
        m = ONELINE.search(text)
        if m:
            status = "solved" if m.group(1) == "VERIFIED" else "failed"
            reason = (m.group(3) or "").strip() or None
        else:
            # No oneline marker at all: the run died before reporting a verdict.
            status = "error"
            reason = (text.strip().splitlines() or ["no output"])[-1][:200]
    except subprocess.TimeoutExpired:
        status, reason = "timeout", f">{args.timeout}s"

    return {"table": table.name, "solver": solver, "status": status,
            "solved": status == "solved", "reason": reason,
            "seconds": round(time.time() - start, 1)}


def cmd_run(args) -> None:
    names = args.tables or KALKUHL_N3
    tables = []
    for n in names:
        p = REPO / "payoff_tables" / n
        if not p.exists():
            raise SystemExit(f"Missing payoff table: {p}")
        tables.append(p)

    jobs = [(t, s) for s in args.solvers for t in tables]
    print(f"{len(args.solvers)} solvers x {len(tables)} tables = {len(jobs)} runs "
          f"({args.scenario}, atol {args.atol}, {args.timeout}s cap, "
          f"{args.workers} workers)")

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda j: run_one(j[0], j[1], args), jobs))

    payload = {
        "settings": {
            "scenario": args.scenario, "atol": args.atol,
            "timeout": args.timeout, "solvers": args.solvers,
            "tables": [t.name for t in tables],
            "jeres_restarts": args.jeres_restarts,
            "jeres_max_iter": args.jeres_max_iter,
            "merit_restarts": args.merit_restarts,
            "merit_walk": args.merit_walk,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wall {payload['settings']['elapsed_seconds']}s -> {path}\n")
    render(payload)


def render(payload: dict) -> None:
    rows, st = payload["rows"], payload["settings"]
    solvers, tables = st["solvers"], st["tables"]
    by = {(r["table"], r["solver"]): r for r in rows}
    mark = {"solved": "OK", "failed": "--", "timeout": "TO", "error": "ER"}

    width = max(len(t) for t in tables) + 2
    print(f"{'table':{width}s}" + "".join(f"{s[:18]:>20s}" for s in solvers))
    for t in tables:
        line = f"{t.replace('kalkuhl_', '').replace('_2035-2100.xlsx', ''):{width}s}"
        for s in solvers:
            r = by.get((t, s))
            line += f"{mark.get(r['status'], '?') + ' ' + str(r['seconds']) + 's':>20s}" if r else f"{'-':>20s}"
        print(line)

    print()
    print(f"{'solver':22s}{'solved':>8s}{'failed':>8s}{'timeout':>9s}{'error':>7s}{'wall_s':>9s}")
    for s in solvers:
        rs = [r for r in rows if r["solver"] == s]
        print(f"{s:22s}"
              f"{sum(r['status'] == 'solved' for r in rs):>8d}"
              f"{sum(r['status'] == 'failed' for r in rs):>8d}"
              f"{sum(r['status'] == 'timeout' for r in rs):>9d}"
              f"{sum(r['status'] == 'error' for r in rs):>7d}"
              f"{round(sum(r['seconds'] for r in rs), 1):>9}")

    solved_by = {s: {r["table"] for r in rows
                     if r["solver"] == s and r["solved"]} for s in solvers}
    union = set().union(*solved_by.values()) if solved_by else set()
    print(f"\nunion solved by any solver: {len(union)}/{len(tables)}")
    for s in solvers:
        only = solved_by[s] - set().union(
            *[solved_by[o] for o in solvers if o != s]) if len(solvers) > 1 else solved_by[s]
        if only:
            print(f"  only {s}: {', '.join(sorted(only))}")
    unsolved = [t for t in tables if t not in union]
    if unsolved:
        print(f"  solved by none: {', '.join(sorted(unsolved))}")


def cmd_report(args) -> None:
    path = OUT_DIR / f"{args.label}.json"
    if not path.exists():
        raise SystemExit(f"No run named {args.label!r} ({path})")
    render(json.loads(path.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run the solver x table grid")
    run.add_argument("--label", required=True)
    run.add_argument("--solvers", nargs="+", default=DEFAULT_SOLVERS)
    run.add_argument("--tables", nargs="+", default=None,
                     help="payoff-table filenames (default: the 10 kalkuhl trios)")
    run.add_argument("--scenario", default="power_threshold_RICE_n3")
    run.add_argument("--atol", default="1e-12")
    run.add_argument("--timeout", type=int, default=180)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--jeres-restarts", type=int, default=8)
    run.add_argument("--jeres-max-iter", type=int, default=2000)
    run.add_argument("--merit-restarts", type=int, default=400)
    run.add_argument("--merit-walk", type=int, default=4000)
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="re-render a saved run")
    rep.add_argument("label")
    rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
