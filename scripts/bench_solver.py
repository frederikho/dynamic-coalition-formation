#!/usr/bin/env python3
"""
Fast solvability benchmark across many n=3 payoff tables.

Purpose: track whether a change to the solver makes it solve MORE tables, in
minutes rather than hours. Run it before a change, run it after, diff the two.

The design choice that makes it both fast and sensitive: a small restart budget
(default 8). Solver failures are what cost time, and their cost is proportional
to the restart count, so a small budget bounds the runtime. It also makes the
benchmark sensitive to exactly the thing usually worth measuring -- whether the
multi-start search is actually exploring. A solver whose random restarts are
useless degenerates to a single deterministic start and solves only the tables
that start lucky; one whose restarts explore well solves many more per restart.

Deliberately NOT a correctness test. It reports how many tables yield a verified
equilibrium under a fixed budget, not whether the equilibria are right. Keep the
pytest suite for correctness.

Usage:
    python scripts/bench_solver.py --label before
    # ... change the solver ...
    python scripts/bench_solver.py --label after
    python scripts/bench_solver.py --compare before after
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
OUT_DIR = REPO / "reports" / "bench_solver"
SUCCESS = re.compile(r"SUCCESS: Found valid", re.IGNORECASE)


def n3_tables(pattern: str | None) -> list[Path]:
    """Every payoff table whose filename resolves to exactly three players."""
    sys.path.insert(0, str(REPO))
    from lib.equilibrium.find import _infer_or_parse_players_from_payoff_table

    paths = sorted((REPO / "payoff_tables").glob(pattern or "*.xlsx"))
    out = []
    for p in paths:
        # Synthetic control fixtures are not production scenarios and would shift
        # the denominator every count in this repo is quoted against.
        if p.name.startswith("mixedcontrol_"):
            continue
        try:
            if len(_infer_or_parse_players_from_payoff_table(p)) == 3:
                out.append(p)
        except Exception:
            continue  # not a player-encoding filename; not a benchmark target
    return out


def run_one(table: Path, args) -> dict:
    cmd = [
        sys.executable, "-m", "lib.equilibrium.find", args.scenario,
        "--payoff-table", str(table),
        "--solver-approach", "jeres_vfi",
        "--jeres-n-restarts", str(args.restarts),
        "--jeres-max-iter", str(args.max_iter),
        "--jeres-restart-scaling", args.restart_scaling,
        "--quiet",
    ]
    if args.normalise:
        cmd += ["--verify-atol", args.atol,
                "--jeres-tol", f"{float(args.atol) / 100:g}"]
    else:
        cmd += ["--no-normalise-payoffs", "--verify-rtol", args.rtol]

    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=args.timeout)
        text = proc.stdout + proc.stderr
        solved = bool(SUCCESS.search(text))
        status = "solved" if solved else "failed"
    except subprocess.TimeoutExpired:
        solved, status = False, "timeout"
    return {"table": table.name, "solved": solved, "status": status,
            "seconds": round(time.time() - start, 1)}


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    solved = sum(r["solved"] for r in rows)
    return {
        "tables": n,
        "solved": solved,
        "failed": sum(r["status"] == "failed" for r in rows),
        "timeout": sum(r["status"] == "timeout" for r in rows),
        "solve_rate": round(solved / n, 4) if n else 0.0,
        "wall_seconds": round(sum(r["seconds"] for r in rows), 1),
    }


def cmd_run(args) -> None:
    tables = n3_tables(args.pattern)
    if not tables:
        raise SystemExit(f"No n=3 payoff tables matched {args.pattern!r}")
    if args.limit:
        tables = tables[: args.limit]
    print(f"Benchmarking {len(tables)} tables "
          f"({args.restarts} restarts, {args.timeout}s cap, {args.workers} workers)")

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda t: run_one(t, args), tables))

    summary = summarise(rows)
    summary["elapsed_seconds"] = round(time.time() - started, 1)
    summary["settings"] = {
        "restarts": args.restarts, "max_iter": args.max_iter,
        "timeout": args.timeout, "normalise": args.normalise,
        "atol": args.atol if args.normalise else None,
        "rtol": None if args.normalise else args.rtol,
        "scenario": args.scenario,
        "restart_scaling": args.restart_scaling,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"\n  solved {summary['solved']}/{summary['tables']}"
          f"  ({summary['solve_rate']:.1%})"
          f"   failed {summary['failed']}   timeout {summary['timeout']}")
    print(f"  wall {summary['elapsed_seconds']}s -> {path}")


def cmd_compare(args) -> None:
    def load(label):
        p = OUT_DIR / f"{label}.json"
        if not p.exists():
            raise SystemExit(f"No benchmark run named {label!r} ({p})")
        return json.loads(p.read_text())

    a, b = load(args.before), load(args.after)
    sa, sb = a["summary"], b["summary"]
    print(f"{'':32s} {args.before:>12s} {args.after:>12s}")
    for key in ("tables", "solved", "failed", "timeout", "elapsed_seconds"):
        print(f"{key:32s} {sa[key]:>12} {sb[key]:>12}")
    print(f"{'solve_rate':32s} {sa['solve_rate']:>11.1%} {sb['solve_rate']:>11.1%}")

    before = {r["table"]: r["solved"] for r in a["rows"]}
    after = {r["table"]: r["solved"] for r in b["rows"]}
    gained = sorted(t for t in after if after[t] and not before.get(t, False))
    lost = sorted(t for t in after if not after[t] and before.get(t, False))

    print(f"\nnewly solved ({len(gained)}):")
    for t in gained:
        print("  +", t)
    print(f"\nregressions ({len(lost)}):")
    for t in lost:
        print("  -", t)
    if not lost:
        print("  none")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    run = sub.add_parser("run", help="benchmark the current solver")
    run.add_argument("--label", required=True, help="name for this run, e.g. 'before'")
    run.add_argument("--pattern", default=None, help="glob within payoff_tables/")
    run.add_argument("--scenario", default="power_threshold_RICE_n3")
    run.add_argument("--restarts", type=int, default=8,
                     help="small on purpose: bounds runtime AND makes the benchmark "
                          "sensitive to whether restarts explore usefully (default 8)")
    run.add_argument("--max-iter", type=int, default=2000)
    run.add_argument("--timeout", type=int, default=45, help="seconds per table")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--restart-scaling", choices=["spread", "level"], default="spread",
                     help="multi-start noise scaling to benchmark (default spread)")
    run.add_argument("--no-normalise", dest="normalise", action="store_false",
                     default=True,
                     help="benchmark raw-unit solving (historical behaviour) instead "
                          "of the default per-player normalised payoffs")
    run.add_argument("--atol", default="1e-2", help="used only with --normalise")
    run.add_argument("--rtol", default="1e-2", help="used only without --normalise")
    run.set_defaults(func=cmd_run)

    cmp_ = sub.add_parser("compare", help="diff two labelled runs")
    cmp_.add_argument("before")
    cmp_.add_argument("after")
    cmp_.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        raise SystemExit(1)
    args.func(args)


if __name__ == "__main__":
    main()
