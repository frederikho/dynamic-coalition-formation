"""
Run Jere's SMPE solver (`smpe_sweep.sweep`, coalition-game repo) on OUR
full-precision kalkuhl payoff tables, so the result can be verified with our
own tools (`scripts/verify_jere_smpe_profiles.py --sweep-dir ...`).

Jere's own examples use our payoffs rounded to 6 decimals, which is enough to
move exact ties in V and so to change which mixed profiles are equilibria.
This script feeds the unrounded numbers straight from `setup['payoffs_raw']`.

Per trio it writes, in his formats:
    <out>/kalkuhl_<trio>.csv              payoff matrix with players/rows header
    <out>/kalkuhl_<trio>.json             full sweep export
    <out>/kalkuhl_<trio>_strategies.csv   strategy records
so `check_equilibrium.py` and `certify_equilibrium.py` can also be run on them.

Usage
-----
    python scripts/run_jere_smpe_sweep.py
    python scripts/run_jere_smpe_sweep.py --trios chneurrus --deltas 0.95 0.99
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import verify_jere_smpe_profiles as VJ  # noqa: E402

TRIOS = ["chneurnde", "chneurrus", "chneurusa", "chnnderus", "chnndeusa",
         "chnrususa", "eurnderus", "eurndeusa", "eurrususa", "nderususa"]
# The grid of the "Pure-Mixed Boundary" phase map.
DELTAS = [0.50, 0.70, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95,
          0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]


def trio_matrix(trio: str):
    """(players, rows, payoffs) for Jere's sweep, from our unrounded table."""
    setup = VJ._build_setup(trio, 0.9, "heyen_lehtomaa_2021")
    players = list(setup["players"])
    raw = setup["payoffs_raw"]
    rows, mat = [], []
    for name in setup["state_names"]:
        part = VJ._fw_partition(name, players)
        blocks = sorted((sorted(b, key=players.index) for b in part),
                        key=lambda b: players.index(b[0]))
        rows.append("+".join("{" + ",".join(b) + "}" for b in blocks))
        mat.append([float(raw.loc[name, p]) for p in players])
    return players, rows, np.array(mat)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trios", nargs="*", default=TRIOS)
    ap.add_argument("--deltas", nargs="*", type=float, default=DELTAS)
    ap.add_argument("--out", default=str(ROOT / "reports" / "jere_smpe"))
    ap.add_argument("--jere-dir", default=str(VJ.JERE_DIR))
    ap.add_argument("--budget", type=float, default=15.0,
                    help="seconds per homotopy solve (Jere's default 15)")
    ap.add_argument("--no-follow", action="store_true",
                    help="solve each delta independently (no branch following)")
    args = ap.parse_args()

    sys.path.insert(0, args.jere_dir)
    from smpe_sweep import sweep  # Jere's code, imported unmodified

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for trio in args.trios:
        players, rows, pay = trio_matrix(trio)
        stem = out / f"kalkuhl_{trio}"
        with open(f"{stem}.csv", "w") as fh:
            fh.write(f"# kalkuhl_{trio}_2035-2100: full precision, from payoffs_raw\n")
            fh.write(f"# players: {','.join(players)}\n")
            fh.write(f"# rows: {'|'.join(rows)}\n")
            np.savetxt(fh, pay, delimiter=",", fmt="%.17g")
        t = time.time()
        res = sweep(players, pay, deltas=args.deltas, rows=rows,
                    follow_branch=not args.no_follow, budget=args.budget)
        dt = time.time() - t
        res.to_json(f"{stem}.json")
        res.to_csv(f"{stem}_strategies.csv")
        tally = {}
        for p in res.points:
            k = p.solution.status if p.solution is not None else "UNSOLVED"
            tally[k] = tally.get(k, 0) + 1
        line = " ".join(f"{p.delta:g}:{(p.solution.status[0] if p.solution else '-')}"
                        for p in res.points)
        print(f"{trio:10s} {res.n_solved}/{len(res.points)} solved [{dt:.0f}s] {tally}\n    {line}",
              flush=True)


if __name__ == "__main__":
    main()
