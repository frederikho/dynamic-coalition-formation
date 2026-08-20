#!/usr/bin/env python3
"""
Exhaustive pure-strategy enumeration on every still-unsolved (trio, delta) cell.

Validated first by scripts/ordinal_crosscheck.py: ordinal_ranking recovers 20/20
points where jeres_vfi already found a pure profile, so a `none` here is a real
exhaustion of the pure space, and a `found` is a pure equilibrium VFI could not
reach.  Results append to a .jsonl as they finish so an interrupted run keeps
everything it measured.
"""
import json, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "ordinal_sweep"
GRID = [0.5, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94,
        0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]
VER = re.compile(r"\bVERIFIED\b")


def run(job):
    trio, delta = job
    out = OUT / "profiles" / f"{trio}_d{delta:g}_ordinal.xlsx"
    cmd = [sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
           "--payoff-table", str(REPO / f"payoff_tables/kalkuhl_{trio}_2035-2100.xlsx"),
           "--discounting", f"{delta:g}", "--effectivity-rule", "heyen_lehtomaa_2021",
           "--solver-approach", "ordinal_ranking", "--verify-atol", "1e-12",
           "--fresh", "--output", str(out), "--oneline", "--quiet"]
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        status = "found" if VER.search(p.stdout + p.stderr) else "none"
    except subprocess.TimeoutExpired:
        status = "timeout"
    return {"trio": trio, "delta": delta, "ordinal": status,
            "seconds": round(time.time() - t, 1)}


def main():
    (OUT / "profiles").mkdir(parents=True, exist_ok=True)
    cold = {}
    for lbl in ("sweep1", "refine"):
        for r in json.loads((REPO / f"reports/delta_homotopy/{lbl}.json").read_text())["rows"]:
            if r["atol"] == "1e-12":
                cold[(r["trio"], r["delta"])] = (r["status"] == "solved")
    for f in ("cold_controls.json", "cold_controls2.json"):
        p = REPO / "reports/delta_continuation" / f
        if p.exists():
            for r in json.loads(p.read_text()):
                cold[(r["trio"], r["delta"])] = r["cold_solved"]

    have = set()
    for d in ("reports/delta_homotopy/profiles", "reports/delta_continuation/profiles",
              "reports/warm_start_experiments/profiles"):
        for f in (REPO / d).glob("*.xlsx"):
            m = re.match(r"^([a-z]+)_d([0-9.]+)", f.stem)
            if m:
                have.add((m[1], float(m[2])))

    done = {}
    cc = REPO / "reports/ordinal_crosscheck.json"
    if cc.exists():
        for r in json.loads(cc.read_text()):
            if r["group"] == "unsolved":
                done[(r["trio"], r["delta"])] = r
    jsonl = OUT / "results.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            done[(r["trio"], r["delta"])] = r

    trios = sorted({t for t, _ in cold})
    todo = [(t, d) for t in trios for d in GRID
            if (t, d) not in have and (t, d) not in done]
    print(f"{len(todo)} unsolved cells to enumerate "
          f"({len(done)} already done)\n", flush=True)

    rows = list(done.values())
    with ThreadPoolExecutor(max_workers=14) as pool:
        for r in pool.map(run, todo):
            rows.append(r)
            with jsonl.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            flag = "   <-- NEW PURE EQUILIBRIUM" if r["ordinal"] == "found" else ""
            print(f"  {r['trio']:<11} d={r['delta']:<7} {r['ordinal']:<8} "
                  f"{r['seconds']:>7}s{flag}", flush=True)

    found = [r for r in rows if r["ordinal"] == "found"]
    print(f"\nenumerated {len(rows)} cells: {len(found)} found, "
          f"{sum(1 for r in rows if r['ordinal']=='none')} exhausted-none, "
          f"{sum(1 for r in rows if r['ordinal']=='timeout')} timeout")
    print(f"\nNEW PURE EQUILIBRIA ({len(found)}):")
    for r in sorted(found, key=lambda r: (r["trio"], r["delta"])):
        print(f"   {r['trio']:<11} d={r['delta']}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
