#!/usr/bin/env python3
"""Empirical per-label cost measurement for the certified search (Section 7).

Running instance: burke_usaruschn_2035-2060, scenario power_threshold_RICE_n3,
effectivity adjacent_step.

Two phases, written for replication:

  Phase 1 (cheap, no solving) over N1 random weak-order triples R:
    - r(R) : the proposal-support multiplier
             r(R) = prod over the 15 (proposer,state) cells of (2^t - 1),
             where t = number of feasible targets tied at that proposer's BEST
             feasible tier in that state. t=1 -> pure proposal (factor 1).
             This is the R-dependent count of realisable proposal supports
             (Section 7.5 / sec:rdep): how many proposal active-sets a full solve
             must consider for triple R.
    - m(R) : number of acceptance mixing variables (distinct (k,x,y) tied
             committee contexts on the proposed transitions) -- the unknowns of
             the per-label acceptance system; drives the cost of ONE sub-solve.
    - n_eq(R): number of indifference equations.

  Phase 2 (timed, bounded) over a random sample:
    - run the EXACT certified per-label solve (CertifiedLabelSolver, acceptance
      mixing with fixed proposals, sympy backend) with a per-solve SIGALRM
      timeout, recording (m, r, wall_time, status).

Outputs (this directory): counts.csv, timings.csv, and PNG plots; prints an
expected-time computation. NOTE the timed solver is the acceptance-class solver
(the only certified solver built); the FULL per-label time is modelled as
r(R) x c, one acceptance-style sub-solve per realisable support (Section 7).
"""
import sys, os, time, signal, csv, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from scripts.residual_metric_probe import build_setup
from scripts._reduced_helpers import _generate_weak_orders
from scripts.certified_label_solver import CertifiedLabelSolver

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIO = "power_threshold_RICE_n3"
TABLE = "burke_usaruschn_2035-2060"
RULE = "adjacent_step"

setup = build_setup(SCENARIO, ROOT / "payoff_tables" / f"{TABLE}.xlsx", RULE)
cls = CertifiedLabelSolver(setup)
fg = cls.fg
n, S = fg.n, fg.S
orders = _generate_weak_orders(S)
NO = len(orders)


def proposal_support_count(tiers):
    """r(R): product over cells of (2^t - 1), t = #feasible at proposer's best tier."""
    r = 1
    for ii in range(n):
        ti = tiers[ii]
        for xi in range(S):
            feas = fg.feasible[(ii, xi)]
            best = min(int(ti[y]) for y in feas)
            t = sum(1 for y in feas if int(ti[y]) == best)
            r *= (2 ** t - 1)
    return r


def mixing_var_count(tiers):
    """m(R): distinct (k,x,y) tied committee contexts on the proposed transitions."""
    prop = cls._fixed_proposals(tiers)
    keys = set()
    for (ii, xi), yi in prop.items():
        if yi == xi:
            continue
        for ki in fg.committee[(ii, xi, yi)]:
            if int(tiers[ki][yi]) == int(tiers[ki][xi]):
                keys.add((ki, xi, yi))
    return len(keys)


def indiff_eq_count(tiers):
    c = 0
    for ki in range(n):
        by = {}
        for si in range(S):
            by.setdefault(int(tiers[ki][si]), []).append(si)
        for g in by.values():
            if len(g) > 1:
                c += len(g) - 1
    return c


def random_triple(rng):
    return tuple(orders[rng.randint(NO)] for _ in range(n))


# ---------------- Phase 1: cheap counts over N1 triples ----------------
def phase1(N1=10000, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    t0 = time.time()
    for _ in range(N1):
        tiers = random_triple(rng)
        rows.append((proposal_support_count(tiers), mixing_var_count(tiers), indiff_eq_count(tiers)))
    dt = time.time() - t0
    with open(HERE / "counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["r", "m", "n_eq"]); w.writerows(rows)
    r = np.array([x[0] for x in rows], float)
    m = np.array([x[1] for x in rows], int)
    print(f"[phase1] {N1} triples classified in {dt:.1f}s")
    print(f"  r (proposal-support count): mean={r.mean():.3g} median={np.median(r):.0f} "
          f"max={r.max():.3g}  frac r==1 (all pure): {(r==1).mean():.3%}")
    print(f"  m (acceptance mixing vars): mean={m.mean():.2f} median={np.median(m):.0f} "
          f"max={m.max()}  frac m==0: {(m==0).mean():.3%}")
    return r, m


# ---------------- Phase 2: timed exact solves (bounded) ----------------
class _Timeout(Exception):
    pass

def _alarm(sig, frm):
    raise _Timeout()

def phase2(budget_s=300, per_solve_timeout=12, seed=1, max_solves=400):
    rng = np.random.RandomState(seed)
    rows = []
    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.time()
    while time.time() - t0 < budget_s and len(rows) < max_solves:
        tiers = random_triple(rng)
        m = mixing_var_count(tiers); r = proposal_support_count(tiers)
        ts = time.time(); status = "ok"
        signal.setitimer(signal.ITIMER_REAL, per_solve_timeout)
        try:
            cls.solve(tiers)
        except _Timeout:
            status = "timeout"
        except Exception:
            status = "error"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        rows.append((m, r, time.time() - ts, status))
    with open(HERE / "timings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["m", "r", "time_s", "status"]); w.writerows(rows)
    ok = [x for x in rows if x[3] == "ok"]
    print(f"[phase2] {len(rows)} solves in {time.time()-t0:.0f}s "
          f"({len(ok)} ok, {sum(x[3]=='timeout' for x in rows)} timeout, "
          f"{sum(x[3]=='error' for x in rows)} error)")
    return rows


# ---------------- plots ----------------
def plots(r, m, timings):
    # r distribution (log10, since heavy-tailed)
    plt.figure(figsize=(6, 4))
    rr = np.log10(np.clip(r, 1, None))
    plt.hist(rr, bins=40, color="#3b6", edgecolor="k", linewidth=.3)
    plt.xlabel(r"$\log_{10}\, r(R)$  (proposal-support count)")
    plt.ylabel("triples"); plt.title(f"Proposal-support count over 10k triples ({TABLE})")
    plt.tight_layout(); plt.savefig(HERE / "plot_r_distribution.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(m, bins=range(int(m.max()) + 2), color="#36b", edgecolor="k", linewidth=.3, align="left")
    plt.xlabel("m(R)  (acceptance mixing variables)"); plt.ylabel("triples")
    plt.title(f"Acceptance mixing-variable count ({TABLE})")
    plt.tight_layout(); plt.savefig(HERE / "plot_m_distribution.png", dpi=130); plt.close()

    ok = [(mm, rr_, tt) for (mm, rr_, tt, st) in timings if st == "ok"]
    to = [(mm, rr_, tt) for (mm, rr_, tt, st) in timings if st == "timeout"]
    if ok:
        mm = np.array([x[0] for x in ok]); tt = np.array([x[2] for x in ok])
        plt.figure(figsize=(6, 4))
        plt.scatter(mm, tt, s=14, alpha=.6, label="solved", color="#36b")
        if to:
            mmt = np.array([x[0] for x in to]); ttt = np.array([x[2] for x in to])
            plt.scatter(mmt, ttt, s=14, alpha=.6, marker="x", color="r", label="timeout")
        # per-m mean (trend)
        for mv in sorted(set(mm)):
            sel = mm == mv
            if sel.sum() >= 2:
                plt.plot(mv, tt[sel].mean(), "ks", ms=6)
        plt.xlabel("m(R)  (acceptance mixing variables)")
        plt.ylabel("exact certified solve time (s, sympy)")
        plt.title("Per-label solve time vs mixing-variable count")
        plt.legend(); plt.tight_layout()
        plt.savefig(HERE / "plot_time_vs_m.png", dpi=130); plt.close()


# ---------------- expected-time computation ----------------
def expected_time(r, timings):
    ok = [(m, rr, t) for (m, rr, t, st) in timings if st == "ok"]
    if not ok:
        print("[expected] no completed solves; cannot estimate c."); return
    c = np.median([t for (_, _, t) in ok])      # median sub-solve cost (robust to outliers)
    c_mean = np.mean([t for (_, _, t) in ok])
    Er = r.mean()
    total_labels = NO ** 3
    print(f"\n[expected time]  (sympy backend; per-support sub-solve median c={c:.2f}s, mean={c_mean:.2f}s)")
    print(f"  E[r] (proposal-support multiplier) = {Er:.3g}")
    for cores in (14, 134):
        # acceptance-only reference (r=1): one sub-solve per label
        acc_days = total_labels * c / cores / 86400
        # full model: r sub-solves per label
        full_days = total_labels * Er * c / cores / 86400
        print(f"  cores={cores:>3}: acceptance-only ~{acc_days:,.0f} d  |  "
              f"full (x E[r]) ~{full_days:,.0f} d   [sympy]")
    print("  NB: a fast certified backend (msolve / HomotopyContinuation.jl) is expected")
    print("      to cut c by 2-3 orders of magnitude; divide the above accordingly.")


if __name__ == "__main__":
    print(f"instance={TABLE}  states={fg.states}  weak-order triples={NO}^3={NO**3:,}")
    r, m = phase1(N1=10000, seed=0)
    timings = phase2(budget_s=300, per_solve_timeout=12, seed=1)
    plots(r, m, timings)
    expected_time(r, timings)
    print(f"\nwrote: counts.csv, timings.csv, plot_r_distribution.png, "
          f"plot_m_distribution.png, plot_time_vs_m.png  (in {HERE})")
