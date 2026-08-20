#!/usr/bin/env python3
"""
Rebuild the synthetic control batch so every table genuinely REQUIRES mixing.

Two changes from the previous generator:

1. PRIMARY GUARD is now direct: push every knob to a bound and test all 2^M pure
   variants.  If any verifies, the table has a pure equilibrium and is rejected.
   This needs no ranking representation, which is exactly why it catches what
   ordinal_ranking structurally cannot: at an exact tie a pure profile can reject a
   move in both directions, or reject a move yet decline to propose its reverse,
   and no strict ranking rationalises either.  Ten of the previous 38 tables were
   mis-certified for that reason.
   Secondary guard remains ordinal_ranking, which still rules out pure equilibria
   that do NOT rest on a tie, on support patterns other than the planted one.

2. Profiles are SAVED alongside each table.  The previous batch discarded them and
   they had to be recovered by replaying the generator's RNG, which is fragile and
   failed outright for the tables written by a separate top-up run.

Usage:
    python scripts/rebuild_mixing_controls.py [--per-m 10] [--draws 4000]
"""
import argparse, itertools, json, subprocess, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
g_sp = importlib.util.spec_from_file_location("gen", REPO/"scripts/generate_mixed_controls.py")
gen = importlib.util.module_from_spec(g_sp); g_sp.loader.exec_module(gen)
e_sp = importlib.util.spec_from_file_location("enu", REPO/"scripts/enumerate_knobs_m1.py")
enu = importlib.util.module_from_spec(e_sp); e_sp.loader.exec_module(enu)
from lib.equilibrium.jeres_vfi import Game, fw_state_name_to_partition
from lib.equilibrium.jeres_vfi.solver import (compute_values, full_transition_matrix,
    verify_proposals, verify_responses)
from lib.effectivity import get_effectivity

PL = gen.PLAYERS; ST = gen.state_names(); OUT = gen.OUT
DELTA = gen.DELTA; EPS = 1e-12
PROF = REPO/"reports"/"planted_profiles_v3"

def framework_game(u):
    eff = get_effectivity("heyen_lehtomaa_2021", PL, ST)
    g0 = Game.from_payoffs(PL, u)
    j_of = {n: g0.state_idx[fw_state_name_to_partition(n, PL)] for n in ST}
    ac = {(i, j_of[x], j_of[y]): frozenset(k for k, pk in enumerate(PL)
          if eff.get((pi, x, y, pk), 0) == 1)
          for i, pi in enumerate(PL) for x in ST for y in ST}
    return Game(players=PL, payoffs=u, states=g0.states, state_idx=g0.state_idx,
                n_players=3, n_states=5, approval_committees=ac), j_of

def has_pure_variant(rec):
    """Any of the 2^M bound assignments an equilibrium?  Uses framework committees."""
    u = np.asarray(rec["game"].payoffs, float)
    g, _ = framework_game(u); comm = enu.committees(g)
    for bounds in itertools.product((0.0, 1.0), repeat=len(rec["ties"])):
        a2 = [dict(x) for x in rec["alphas"]]
        for (kx, ky, kj), b in zip(rec["ties"], bounds):
            a2[kx][(kj, ky)] = b
        qs = enu.qs_from_alphas(g, comm, a2)
        V = compute_values(g, full_transition_matrix(g, rec["sigmas"], qs, None), DELTA)
        r, _ = verify_responses(g, rec["sigmas"], a2, qs, V, atol=EPS)
        p, _ = verify_proposals(g, rec["sigmas"], a2, qs, V, atol=EPS)
        if r and p:
            return True
    return False

def planted_verifies(rec):
    """The planted mixed profile must verify under the FRAMEWORK committee rule."""
    u = np.asarray(rec["game"].payoffs, float)
    g, _ = framework_game(u); comm = enu.committees(g)
    qs = enu.qs_from_alphas(g, comm, rec["alphas"])
    V = compute_values(g, full_transition_matrix(g, rec["sigmas"], qs, None), DELTA)
    r, _ = verify_responses(g, rec["sigmas"], rec["alphas"], qs, V, atol=EPS)
    p, _ = verify_proposals(g, rec["sigmas"], rec["alphas"], qs, V, atol=EPS)
    return r and p

def too_easy(rec, samples=120, max_hit_rate=0.10):
    """Reject tables whose feasible set is so large that any guess verifies.

    A control only tests a solver if finding the mixing probabilities is actually
    hard.  Measured on the previous batch, theta = 0.5 alone verified for 8/10
    tables at M=1 and a uniformly random point verified 61% of the time -- so a
    100% solve rate there measured almost nothing.
    """
    u = np.asarray(rec["game"].payoffs, float)
    g, _ = framework_game(u); comm = enu.committees(g)
    M = len(rec["ties"])
    rng = np.random.default_rng(0)
    pts = [np.full(M, 0.5)] + [rng.uniform(0, 1, M) for _ in range(samples)]
    hits = 0
    for th in pts:
        a2 = [dict(x) for x in rec["alphas"]]
        for (kx, ky, kj), t in zip(rec["ties"], th):
            a2[kx][(kj, ky)] = float(t)
        qs = enu.qs_from_alphas(g, comm, a2)
        V = compute_values(g, full_transition_matrix(g, rec["sigmas"], qs, None), DELTA)
        r, _ = verify_responses(g, rec["sigmas"], a2, qs, V, atol=EPS)
        p, _ = verify_proposals(g, rec["sigmas"], a2, qs, V, atol=EPS)
        hits += bool(r and p)
    return hits / len(pts) > max_hit_rate


def or_says_no_pure(rec, tmp):
    gen.write_table(rec, tmp, "candidate")
    out = subprocess.run(
        [sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
         "--payoff-table", str(tmp), "--discounting", str(DELTA),
         "--effectivity-rule", "heyen_lehtomaa_2021",
         "--solver-approach", "ordinal_ranking", "--verify-atol", "1e-12",
         "--fresh", "--output", "/dev/null", "--oneline", "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    return "VERIFIED" not in (out.stdout + out.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-m", type=int, default=10)
    ap.add_argument("--draws", type=int, default=4000)
    args = ap.parse_args()
    PROF.mkdir(parents=True, exist_ok=True)
    tmp = REPO/"reports"/"_rebuild_candidate.xlsx"
    rng = np.random.default_rng(20260820)
    kept = {m: [] for m in (1, 2, 3, 4)}
    stats = {m: dict(drawn=0, planted_fail=0, pure_variant=0, too_easy=0,
                     or_pure=0, kept=0)
             for m in kept}
    t0 = time.time()
    for target in (1, 2, 3, 4):
        for _ in range(args.draws):
            if len(kept[target]) >= args.per_m:
                break
            rec = gen.build_one(rng, target)
            if rec is None or rec["M"] != target:
                continue
            stats[target]["drawn"] += 1
            if not planted_verifies(rec):
                stats[target]["planted_fail"] += 1; continue
            if has_pure_variant(rec):
                stats[target]["pure_variant"] += 1; continue
            if too_easy(rec):
                stats[target]["too_easy"] = stats[target].get("too_easy", 0) + 1
                continue
            if not or_says_no_pure(rec, tmp):
                stats[target]["or_pure"] += 1; continue
            kept[target].append(rec); stats[target]["kept"] += 1
            print(f"  M={target} kept {len(kept[target])}/{args.per_m} "
                  f"(draw {stats[target]['drawn']}, {time.time()-t0:.0f}s)", flush=True)
        print(f"M={target}: {stats[target]}", flush=True)

    for old in OUT.glob("mixedcontrol_m*_chneurusa.xlsx"):
        old.unlink()
    for old in PROF.glob("*.json"):
        old.unlink()
    manifest = []
    for m, recs in sorted(kept.items()):
        for k, rec in enumerate(recs):
            fname = f"mixedcontrol_m{m}_{k:02d}_chneurusa.xlsx"
            planted = {f"{ST[x]}->{ST[y]}|{PL[j]}": round(v, 12)
                       for (x, y, j), v in rec["thetas"].items()}
            gen.write_table(rec, OUT/fname,
                f"Synthetic control: M={m}, delta={DELTA}, mixing REQUIRED "
                f"(all 2^M pure variants fail; ordinal_ranking finds none); "
                f"planted {planted}")
            (PROF/f"{fname.replace('.xlsx','')}.json").write_text(json.dumps(dict(
                file=fname, M=m, delta=DELTA,
                payoffs=np.asarray(rec["game"].payoffs).tolist(),
                V=np.asarray(rec["V"]).tolist(),
                knobs=[[int(a), int(b), int(c)] for a, b, c in rec["ties"]],
                thetas=[float(rec["thetas"][t]) for t in rec["ties"]],
                sigmas=[{f"{i},{y}": float(v) for (i, y), v in d.items()} for d in rec["sigmas"]],
                alphas=[{f"{j},{y}": float(v) for (j, y), v in d.items()} for d in rec["alphas"]],
            ), indent=2))
            manifest.append(dict(file=fname, M=m, delta=DELTA, mixing_required=True,
                                 planted_thetas=planted,
                                 min_payoff_spread=round(rec["spread"], 6)))
    (REPO/"reports"/"mixed_controls_manifest.json").write_text(json.dumps(
        dict(stats={str(k): v for k, v in stats.items()}, tables=manifest), indent=2))
    if tmp.exists(): tmp.unlink()
    print(f"\n{len(manifest)} tables written ({time.time()-t0:.0f}s); profiles -> {PROF}")
    for m in (1, 2, 3, 4):
        print(f"  M={m}: {stats[m]}")

if __name__ == "__main__":
    main()
