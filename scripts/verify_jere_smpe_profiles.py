"""
Independently verify the SMPE profiles exported by Jere's `smpe_sweep.py`
(coalition-game repo) with THIS framework's verifier.

Nothing from Jere's solver is trusted except the exported strategy records
(proposal and vote probabilities).  Payoffs are loaded from our own payoff
tables, effectivity from our own `heyen_lehtomaa_2021` rule, and V is
recomputed by our MDP -- exactly the path `find_equilibrium` uses.

Committee mismatch, and how it is handled
-----------------------------------------
Jere's committees agree with `heyen_lehtomaa_2021` on 51 of 60 (proposer,
transition) pairs at n=3.  The 9 differences are all unilateral exits: Jere
requires no vote (q = 1), we put the proposer alone on his own exit committee.

Jere's votes do not depend on the proposer, ours may.  So the exported vote
alpha_j(s,t) is used for every proposer EXCEPT j's own exit, where Jere's
exported value belongs to a different proposer's committee and must not be
copied: at an exact tie Jere can legally have j vote 0.5 when another player
proposes s->t while j's own exit to t still passes with certainty.  Copying the
0.5 halves that transition and changes the game (observed: example6, delta
0.95, USA leaving the grand coalition).  The own-exit vote is instead set by
the cutoff rule on Jere's reported V: 1 if V_j(t) >= V_j(s) (tie within
TIE_REL of j's payoff spread -> 1, reproducing Jere's q = 1), else 0.  A
proposer never proposes an exit with a strict loss, so the 0 branch only
touches off-path votes, which our verifier still checks.

Usage
-----
    python scripts/verify_jere_smpe_profiles.py
    python scripts/verify_jere_smpe_profiles.py --examples example6 --atol 1e-10
    python scripts/verify_jere_smpe_profiles.py --payoffs jere

Which payoffs
-------------
Jere's example matrices are OUR kalkuhl tables rounded to 6 decimals.  That
moves payoffs by up to ~5e-7, i.e. up to 0.6% of a player's spread (RUS in
chneurrus has spread 7.8e-5).  Mixed equilibria live on exact ties in V, so a
profile exact for the rounded game is generically NOT an equilibrium of the
unrounded one.  `--payoffs jere` verifies against the game Jere actually
solved (tests his method and our conventions against each other);
`--payoffs ours` (default) against our full-precision tables (tests whether
his published numbers are equilibria of OUR game).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.equilibrium.find import (  # noqa: E402
    setup_experiment, normalise_payoffs, _compute_verification, _infer_or_parse_players_from_payoff_table,
)
from lib.equilibrium.scenarios import get_scenario, fill_players  # noqa: E402
from lib.equilibrium.solver import EquilibriumSolver  # noqa: E402
from lib.equilibrium.mip_vfi import _arrays_to_strategy_df  # noqa: E402
from lib.equilibrium.jeres_vfi.game import fw_state_name_to_partition  # noqa: E402
from lib.utils import verify_equilibrium_detailed, get_approval_committee  # noqa: E402

JERE_DIR = Path("/mnt/data1/Code/coalition-game/coalition_game")

# Own-exit tie threshold, relative to the player's raw payoff spread.  Matches
# Jere's EPS_MAX_RELATIVE; exact ties in his V come out as 0.0 or ~1e-16.
TIE_REL = 1e-9

# Jere's examples -> our kalkuhl trios.  Identified by matching payoffs
# (checked again at runtime against our raw table in main()).
# example3 and example5 are the same matrix; results_wide.json duplicates example5.
EXAMPLES = {
    "example4": ("chnndeusa", {"A": "CHN", "B": "NDE", "C": "USA"}),
    "example5": ("chnnderus", {"A": "CHN", "B": "NDE", "C": "RUS"}),
    "example6": ("chneurusa", {"A": "CHN", "B": "EUR", "C": "USA"}),
    "example7": ("chneurrus", {"A": "CHN", "B": "EUR", "C": "RUS"}),
}


def _parse_label(label: str, letter_map: dict) -> frozenset:
    """'{A,C}+{B}' -> frozenset of frozensets of OUR player names."""
    blocks = []
    for blk in label.split("+"):
        names = blk.strip().strip("{}").split(",")
        blocks.append(frozenset(letter_map[n.strip()] for n in names))
    return frozenset(blocks)


def _fw_partition(name: str, players: list) -> frozenset:
    return frozenset(frozenset(players[i] if isinstance(i, int) else i for i in b)
                     for b in fw_state_name_to_partition(name, players))


def _build_setup(trio: str, delta: float, effectivity_rule: str):
    table = ROOT / "payoff_tables" / f"kalkuhl_{trio}_2035-2100.xlsx"
    config = get_scenario("power_threshold_RICE_n3")
    config["payoff_table"] = str(table)
    config["discounting"] = float(delta)
    config["effectivity_rule"] = effectivity_rule
    config["normalise_payoffs"] = True
    if config.get("players") is None:
        config = fill_players(config, _infer_or_parse_players_from_payoff_table(table))
    return setup_experiment(config)


def _jere_example_payoffs(jere_dir, ex, setup, state_map, letter_map):
    """Jere's 6-decimal matrix for `ex`, as a copy of our raw payoff frame.
    Raises if it differs from our table by more than 6-decimal rounding, which
    would mean the example-to-trio mapping in EXAMPLES is wrong."""
    sys.path.insert(0, str(jere_dir))
    from examples import EXAMPLES as JX, ROWS
    ours = setup["payoffs_raw"]
    his = ours.copy()
    for r, lab in enumerate(ROWS):
        for L, p in letter_map.items():
            v = float(JX[ex]["payoffs"][r, "ABC".index(L)])
            if abs(float(ours.loc[state_map[lab], p]) - v) > 5e-7 + 1e-12:
                raise SystemExit(f"payoff mismatch {ex} {lab} {p}: "
                                 f"{ours.loc[state_map[lab], p]} vs {v}")
            his.loc[state_map[lab], p] = v
    return his


def verify_point(point, setup, state_map, letter_map, atol):
    players = setup["players"]
    states = setup["state_names"]
    S, N = len(states), len(players)
    pidx = {p: k for k, p in enumerate(players)}
    sidx = {s: k for k, s in enumerate(states)}

    sig = np.zeros((S, N, S))
    alp = np.zeros((S, N, S))
    for r in point["strategies"]:
        s = sidx[state_map[r["state"]]]
        t = sidx[state_map[r["target"]]]
        i = pidx[letter_map[r["player"]]]
        if r["kind"] == "propose":
            sig[s, i, t] = float(r["prob"])
        elif r["kind"] == "accept":
            alp[s, i, t] = float(r["prob"])

    solver = EquilibriumSolver(
        players=players, states=states, effectivity=setup["effectivity"],
        protocol=setup["protocol"], payoffs=setup["payoffs"],
        discounting=setup["discounting"], unanimity_required=setup["unanimity_required"],
        power_rule=setup["power_rule"],
        forbidden_proposals=setup.get("forbidden_proposals", frozenset()),
        effectivity_rule=setup.get("effectivity_rule", "heyen_lehtomaa_2021"),
        verbose=False, random_seed=0, geo_levels=setup.get("geoengineering"),
    )
    df = _arrays_to_strategy_df(solver, sig, alp)

    # Own-exit votes: see module docstring.
    Vj = point["V"]  # raw units, keyed by Jere label then letter
    inv_state = {v: k for k, v in state_map.items()}
    inv_letter = {v: k for k, v in letter_map.items()}
    raw = setup["payoffs_raw"]
    overridden = []
    for p in players:
        tie = TIE_REL * float(raw[p].max() - raw[p].min())
        L = inv_letter[p]
        for s_name in states:
            for t_name in states:
                if s_name == t_name:
                    continue
                if p not in get_approval_committee(setup["effectivity"], players,
                                                   p, s_name, t_name):
                    continue
                gain = Vj[inv_state[t_name]][L] - Vj[inv_state[s_name]][L]
                a = 1.0 if gain >= -tie else 0.0
                cell = ((s_name, "Acceptance", p), (f"Proposer {p}", t_name))
                if df.loc[cell] != a:
                    overridden.append((p, s_name, t_name, float(df.loc[cell]), a, gain))
                df.loc[cell] = a
    df = df.fillna(0.0)
    V, P, P_prop, P_app = _compute_verification(df, setup)
    result = {
        "V": V, "P": P, "P_proposals": P_prop, "P_approvals": P_app,
        "players": players, "state_names": states, "effectivity": setup["effectivity"],
        "forbidden_proposals": setup.get("forbidden_proposals", frozenset()),
        "strategy_df": df, "payoffs": setup["payoffs"],
        "geoengineering": setup["geoengineering"],
    }
    ok, msg, _ = verify_equilibrium_detailed(result, atol=atol)

    # Cross-check: our P against Jere's reported transition matrix.
    Pj = np.zeros((S, S))
    for a, row in point["transition"].items():
        for b, pr in row.items():
            Pj[sidx[state_map[a]], sidx[state_map[b]]] = pr
    dP = float(np.abs(P.values.astype(float) - Pj).max())

    # Our V vs Jere's V, both raw: map ours back via the per-player affine map.
    dV = None
    if raw is not None:
        norm = setup["payoffs"]
        dV = 0.0
        for p in players:
            u, v = raw[p].values.astype(float), norm[p].values.astype(float)
            a = (u.max() - u.min()) / (v.max() - v.min())
            b = u.min() - a * v.min()
            ours_raw = a * V[p].values.astype(float) + b
            jv = np.array([Vj[inv_state[s]][inv_letter[p]] for s in states])
            dV = max(dV, float(np.abs(ours_raw - jv).max()))
    vspread = float((V.max() - V.min()).min())
    return ok, msg, dP, dV, vspread, overridden


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", nargs="*", default=list(EXAMPLES))
    ap.add_argument("--atol", type=float, nargs="*", default=[1e-12, 1e-10])
    ap.add_argument("--effectivity-rule", default="heyen_lehtomaa_2021")
    ap.add_argument("--jere-dir", default=str(JERE_DIR))
    ap.add_argument("--payoffs", choices=["ours", "jere"], default="ours",
                    help="verify against our full-precision table or Jere's 6-decimal matrix")
    ap.add_argument("--sweep-dir", default=None,
                    help="verify kalkuhl_<trio>.json sweeps written by run_jere_smpe_sweep.py "
                         "(full-precision payoffs, real player names) instead of Jere's examples")
    ap.add_argument("--out", default=None, help="write per-point results as JSON")
    args = ap.parse_args()

    # (label, trio, letter_map, json path, is one of Jere's rounded examples)
    sources = []
    if args.sweep_dir:
        if args.payoffs != "ours":
            ap.error("--sweep-dir runs were solved on our payoffs; use --payoffs ours")
        for f in sorted(Path(args.sweep_dir).glob("kalkuhl_*.json")):
            trio = f.stem.split("_")[1]
            players = json.loads(f.read_text())["players"]
            sources.append((f.stem, trio, {p: p for p in players}, f, False))
    else:
        for ex in args.examples:
            trio, letter_map = EXAMPLES[ex]
            sources.append((ex, trio, letter_map, Path(args.jere_dir) / f"{ex}.json", True))

    rows = []
    for ex, trio, letter_map, path, is_example in sources:
        data = json.loads(path.read_text())
        assert sorted(data["players"]) == sorted(letter_map), data["players"]
        print(f"\n=== {ex} -> kalkuhl_{trio}  ({len(data['points'])} points) ===")
        state_map = None
        for pt in data["points"]:
            if not pt.get("solved"):
                print(f"  delta {pt['delta']:.3f}  Jere: UNSOLVED")
                continue
            setup = _build_setup(trio, pt["delta"], args.effectivity_rule)
            if state_map is None:
                fw = {_fw_partition(s, setup["players"]): s for s in setup["state_names"]}
                state_map = {lab: fw[_parse_label(lab, letter_map)] for lab in data["states"]}
                if is_example:
                    jere_raw = _jere_example_payoffs(args.jere_dir, ex, setup, state_map,
                                                     letter_map)
            if args.payoffs == "jere":
                setup["payoffs_raw"] = jere_raw
                setup["payoffs"] = normalise_payoffs(jere_raw)
            verdicts = []
            for atol in args.atol:
                ok, msg, dP, dV, vspread, filled = verify_point(pt, setup, state_map,
                                                                letter_map, atol)
                verdicts.append((atol, ok, msg))
            first_fail = next((m for a, o, m in verdicts if not o), "")
            vstr = "  ".join(f"atol {a:.0e}: {'PASS' if o else 'FAIL'}" for a, o, _ in verdicts)
            print(f"  delta {pt['delta']:.3f}  {pt['status']:<16s} {vstr}   "
                  f"|dP| {dP:.1e}  |dV|raw {dV:.1e}  Vspread {vspread:.1e}  "
                  f"own-exit overrides {len(filled)}")
            if first_fail:
                print("      " + first_fail.replace("\n", "\n      ")[:1500])
            rows.append(dict(example=ex, trio=trio, payoffs=args.payoffs, delta=pt["delta"], status=pt["status"],
                             method=pt.get("method"), dP=dP, dV=dV, vspread=vspread,
                             verdicts={f"{a:.0e}": o for a, o, _ in verdicts},
                             first_fail=first_fail))

    print("\n=== summary ===")
    for atol in args.atol:
        k = f"{atol:.0e}"
        for status in ("PURE", "MIXED", "MIXED_NONUNIQUE"):
            sub = [r for r in rows if r["status"] == status]
            if sub:
                print(f"  atol {k}  {status:<16s} {sum(r['verdicts'][k] for r in sub)}/{len(sub)} pass")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
