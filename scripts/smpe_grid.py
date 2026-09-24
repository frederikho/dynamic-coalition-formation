"""
Run solver_approach='smpe' through find_equilibrium on the n=3 kalkuhl grid
(10 trios x 18 deltas, the "Pure-Mixed Boundary" phase map) and record our
verifier's verdict for every cell.

Each cell is solved on its own (no continuation) unless --anchors is given,
in which case every cell is solved with the whole delta grid as anchors,
i.e. with Jere's branch following and gap filling.

    python scripts/smpe_grid.py --out reports/jere_smpe/grid_single.json
    python scripts/smpe_grid.py --anchors --out reports/jere_smpe/grid_anchored.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lib.equilibrium.find import find_equilibrium  # noqa: E402
from run_jere_smpe_sweep import TRIOS, DELTAS  # noqa: E402
from lib.equilibrium.scenarios import get_scenario, fill_players  # noqa: E402
from lib.equilibrium.find import _infer_or_parse_players_from_payoff_table  # noqa: E402


def config_for(trio, delta, rule):
    table = ROOT / "payoff_tables" / f"kalkuhl_{trio}_2035-2100.xlsx"
    c = get_scenario("power_threshold_RICE_n3")
    c.update(payoff_table=str(table), discounting=delta, effectivity_rule=rule,
             normalise_payoffs=True)
    return fill_players(c, _infer_or_parse_players_from_payoff_table(table))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trios", nargs="*", default=TRIOS)
    ap.add_argument("--deltas", nargs="*", type=float, default=DELTAS)
    ap.add_argument("--rule", default="heyen_lehtomaa_2021")
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--budget", type=float, default=15.0)
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for trio in args.trios:
        line = []
        for d in args.deltas:
            params = {"smpe_budget": args.budget}
            if args.anchors:
                params["smpe_anchor_deltas"] = list(args.deltas)
            t = time.time()
            res = find_equilibrium(config_for(trio, d, args.rule), verbose=False,
                                   solver_approach="smpe", verify_atol=args.atol,
                                   solver_params=params)
            sr = res.get("solver_result") or {}
            ok = bool(res.get("verification_success"))
            status = sr.get("smpe_status", "UNSOLVED")
            rows.append(dict(trio=trio, delta=d, verified=ok, status=status,
                             method=sr.get("smpe_method"), seconds=time.time() - t,
                             message=None if ok else res.get("verification_message")))
            line.append(f"{d:g}:{'-' if status == 'UNSOLVED' else status[0]}"
                        f"{'' if ok or status == 'UNSOLVED' else '!'}")
        print(f"{trio:10s} {' '.join(line)}", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    n = len(rows)
    solved = [r for r in rows if r["status"] != "UNSOLVED"]
    print(f"\nsolved {len(solved)}/{n}; verified by our verifier at atol {args.atol:g}: "
          f"{sum(r['verified'] for r in rows)}/{n}  "
          f"(solved but NOT verified: {sum(1 for r in solved if not r['verified'])})")
    print("legend: P pure, M mixed, - unsolved, ! solver returned a profile our verifier rejects")


if __name__ == "__main__":
    main()
