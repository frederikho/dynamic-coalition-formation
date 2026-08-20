#!/usr/bin/env python3
"""
Cross-check jeres_vfi's verdicts against exhaustive pure-strategy enumeration.

Two samples:
  SOLVED   points where we already hold a verified (pure) profile.  ordinal_ranking
           searches only 0/1 acceptance, so it should recover ALL of them.  Any miss
           means the enumeration is not actually exhaustive over pure profiles.
  UNSOLVED points no warm-start strategy has cracked.  If enumeration also finds
           nothing, no pure equilibrium exists there and the true one is mixed.
           If it DOES find one, VFI missed a pure equilibrium -- a real gain.
"""
import json, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GRID = [0.5, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94,
        0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]
VER = re.compile(r"\bVERIFIED\b")


def run(job):
    trio, delta, group = job
    cmd = [sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
           "--payoff-table", str(REPO / f"payoff_tables/kalkuhl_{trio}_2035-2100.xlsx"),
           "--discounting", f"{delta:g}", "--effectivity-rule", "heyen_lehtomaa_2021",
           "--solver-approach", "ordinal_ranking", "--verify-atol", "1e-12",
           "--fresh", "--output", "/dev/null", "--oneline", "--quiet"]
    t = time.time()
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=900)
        ok = bool(VER.search(p.stdout + p.stderr))
        status = "found" if ok else "none"
    except subprocess.TimeoutExpired:
        status = "timeout"
    return {"trio": trio, "delta": delta, "group": group,
            "ordinal": status, "seconds": round(time.time() - t, 1)}


def main():
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

    trios = sorted({t for t, _ in cold})
    solved = [(t, d) for t in trios for d in GRID if (t, d) in have]
    unsolved = [(t, d) for t in trios for d in GRID
                if (t, d) not in have and cold.get((t, d)) is False]

    # spread the sample across trios and across the delta range
    def sample(pts, per_trio):
        out = []
        for t in trios:
            ds = sorted(d for tt, d in pts if tt == t)
            if not ds:
                continue
            step = max(1, len(ds) // per_trio)
            out += [(t, d) for d in ds[::step][:per_trio]]
        return out

    jobs = ([(t, d, "solved") for t, d in sample(solved, 2)]
            + [(t, d, "unsolved") for t, d in sample(unsolved, 2)])
    print(f"{len(jobs)} enumeration runs "
          f"({sum(1 for j in jobs if j[2]=='solved')} solved / "
          f"{sum(1 for j in jobs if j[2]=='unsolved')} unsolved)\n", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for r in pool.map(run, jobs):
            rows.append(r)
            print(f"  {r['group']:<9} {r['trio']:<11} d={r['delta']:<7} "
                  f"ordinal={r['ordinal']:<8} {r['seconds']}s", flush=True)

    out = REPO / "reports/ordinal_crosscheck.json"
    out.write_text(json.dumps(rows, indent=2))
    for g in ("solved", "unsolved"):
        sub = [r for r in rows if r["group"] == g]
        if not sub:
            continue
        f = sum(1 for r in sub if r["ordinal"] == "found")
        print(f"\n{g:<9}: ordinal_ranking found an equilibrium in "
              f"{f}/{len(sub)} = {100*f/len(sub):.0f}%")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
