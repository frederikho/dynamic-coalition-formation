#!/usr/bin/env python3
"""Cost concentration & a-priori scheduling analysis (Section 7.7).

Premise: the per-label cost drivers m(R) (acceptance mixing vars) and r(R)
(proposal-support multiplier) are CHEAP structural counts -- computed in ~50 us
per triple WITHOUT solving (see measure_label_cost.py, Phase 1). So we can predict
each label's cost a priori and schedule cheapest-first. This script quantifies:

  (a) a time model t(m) fit from the timed sample (timings.csv);
  (b) predictor validation: does t(m) track the actual measured solve time?
  (c) cost-concentration ("Lorenz") curves for the acceptance-only cost (~t(m))
      and the full cost (~ r * t(m)), over a fresh large sample of triples;
  (d) the headline schedule numbers: cheapest p% of labels -> what % of total work,
      and how much of the work sits in the expensive tail.

Outputs: cost_model.txt, plot_lorenz.png, and printed tables. Read-only w.r.t. the
game (no equilibrium solving here beyond reusing timings.csv).
"""
import sys, csv, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from scripts.residual_metric_probe import build_setup
from scripts._reduced_helpers import _generate_weak_orders
from scripts.certified_label_solver import CertifiedLabelSolver
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

setup = build_setup("power_threshold_RICE_n3",
                    ROOT / "payoff_tables" / "burke_usaruschn_2035-2060.xlsx", "adjacent_step")
cls = CertifiedLabelSolver(setup); fg = cls.fg
n, S = fg.n, fg.S
orders = _generate_weak_orders(S); NO = len(orders); TOTAL = NO ** 3


def counts(tiers):
    r = 1
    for ii in range(n):
        ti = tiers[ii]
        for xi in range(S):
            feas = fg.feasible[(ii, xi)]
            best = min(int(ti[y]) for y in feas)
            t = sum(1 for y in feas if int(ti[y]) == best)
            r *= (2 ** t - 1)
    prop = cls._fixed_proposals(tiers)
    keys = set()
    for (ii, xi), yi in prop.items():
        if yi != xi:
            for ki in fg.committee[(ii, xi, yi)]:
                if int(tiers[ki][yi]) == int(tiers[ki][xi]):
                    keys.add((ki, xi, yi))
    return r, len(keys)


# ---- (a) time model t(m) from timings.csv (median per m; geometric extrapolation) ----
def build_time_model():
    rows = list(csv.DictReader(open(HERE / "timings.csv")))
    bym = {}
    for r in rows:
        bym.setdefault(int(r["m"]), []).append((float(r["time_s"]), r["status"]))
    med = {}
    for m, lst in bym.items():
        med[m] = float(np.median([t for t, _ in lst]))
    # geometric fit log t = A + B m on the uncensored medians (m with no timeouts)
    clean = sorted(m for m in med if all(s != "timeout" for _, s in bym[m]) and med[m] > 0)
    xs = np.array(clean, float); ys = np.log(np.array([med[m] for m in clean]))
    B, A = np.polyfit(xs, ys, 1)
    def t_model(m):
        if m in med and all(s != "timeout" for _, s in bym[m]):
            return med[m]
        return float(math.exp(A + B * m))   # extrapolate (censored / unseen m)
    return t_model, med, (A, B), bym


def main():
    t_model, med, (A, B), bym = build_time_model()
    with open(HERE / "cost_model.txt", "w") as f:
        f.write(f"geometric time model: t(m) = exp({A:.3f} + {B:.3f} m)  "
                f"(per-step factor e^B = {math.exp(B):.2f})\n")
        f.write("measured medians: " + ", ".join(f"m{m}:{med[m]:.3f}s" for m in sorted(med)) + "\n")
    print(f"time model t(m)=exp({A:.3f}+{B:.3f} m), per-step factor {math.exp(B):.2f}x")

    # (b) predictor validation on the 'ok' timed rows
    ok = [(int(r["m"]), float(r["time_s"])) for r in csv.DictReader(open(HERE / "timings.csv")) if r["status"] == "ok"]
    am = np.array([m for m, _ in ok]); at = np.array([t for _, t in ok]); pt = np.array([t_model(m) for m in am])
    # Spearman rank corr between predicted (=m, monotone) and actual
    from scipy.stats import spearmanr
    rho = spearmanr(am, at).statistic
    print(f"predictor: Spearman(m, actual_time) = {rho:.3f}  (m alone strongly orders cost)")

    # (c) fresh large sample -> per-label predicted costs
    rng = np.random.RandomState(7); N = 100000
    ms = np.empty(N, int); rs = np.empty(N, float)
    for i in range(N):
        tiers = tuple(orders[rng.randint(NO)] for _ in range(n))
        rr, mm = counts(tiers); rs[i] = rr; ms[i] = mm
    cost_acc = np.array([t_model(m) for m in ms])          # acceptance-only: one solve
    cost_full = rs * cost_acc                               # full: r sub-solves
    print(f"\nsample N={N}: mean m={ms.mean():.2f}, mean r={rs.mean():.3g}, "
          f"frac pure (r==1)={ (rs==1).mean():.1%}")

    def concentration(cost, name):
        order = np.argsort(cost); c = cost[order]; cum = np.cumsum(c); tot = cum[-1]
        frac_labels = np.arange(1, len(c) + 1) / len(c)
        frac_work = cum / tot
        print(f"\n[{name}]  cheapest p% of labels  ->  % of total work")
        for p in [20, 50, 80, 90, 95, 99, 99.9]:
            idx = min(len(c) - 1, int(p / 100 * len(c)))
            print(f"   cheapest {p:>5}% labels : {100*frac_work[idx]:7.4f}% of work")
        # inverse: work in the most expensive tail (c is already sorted ascending)
        for q in [1, 0.1, 0.01]:
            k = max(1, int(q / 100 * len(c)))
            print(f"   most expensive {q:>4}% labels : {100*c[-k:].sum()/tot:6.2f}% of work")
        return frac_labels, frac_work

    fl_a, fw_a = concentration(cost_acc, "ACCEPTANCE-only cost ~ t(m)")
    fl_f, fw_f = concentration(cost_full, "FULL cost ~ r*t(m)")

    # (d) Lorenz plot
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(fl_a, fw_a, label="acceptance-only ($\\sim t(m)$)", color="#36b")
    plt.plot(fl_f, fw_f, label="full ($\\sim r\\,t(m)$)", color="#b33")
    plt.plot([0, 1], [0, 1], "k--", lw=.7, label="uniform (reference)")
    plt.xlabel("fraction of labels (cheapest-first)")
    plt.ylabel("fraction of total work cleared")
    plt.title("Cost concentration: cheapest-first scheduling (burke\\_usaruschn)")
    plt.legend(loc="upper left"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(HERE / "plot_lorenz.png", dpi=130); plt.close()
    print(f"\nwrote cost_model.txt, plot_lorenz.png (in {HERE})")


if __name__ == "__main__":
    main()
