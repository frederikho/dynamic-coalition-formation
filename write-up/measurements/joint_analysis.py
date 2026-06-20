#!/usr/bin/env python3
"""Joint distribution of the two cost components r(R) and m(R), and how their costs
interact (Section 7). Both are cheap structural counts (no solving).

Components:
  m(R) : acceptance mixing dimension  -> drives the per-(sub)solve cost c(m).
  r(R) : number of candidate proposal supports = prod_cells (2^t - 1),
         t = #feasible at the proposer's best tier -> number of sub-solves a complete
         full search must do for label R.
Full per-label cost model:  C_full(R) ~ r(R) * c(m(R))   (a lower bound: a support
with mixed proposals has > m variables, so costs >= c(m)).
Acceptance-class per-label cost:  C_acc(R) = c(m(R))   (r=1, proposals fixed).

We measure: the r- and m-distributions, their rank correlation, r-by-m, the
concentration of total cost on the tie-rich tail, and the statistical stability
(confidence) of the totals.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "write-up" / "measurements"))
from bench_build_interp import cls, fg, n, S, orders, NO, m_of
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
TOTAL = NO ** 3

# measured acceptance per-label cost c(m) (build+solve, Section 7.9; high m censored at 60s)
CM = {0:0.0001,1:0.0009,2:0.0017,3:0.0111,4:0.0436,5:0.6312,6:0.6482,
      7:9.926,8:18.61,9:25.06,10:58.25,11:60.0,12:60.0,13:60.0,14:60.0}
def c(m): return CM.get(m, 60.0)

def r_of(tiers):
    r = 1
    for ii in range(n):
        ti = tiers[ii]
        for xi in range(S):
            feas = fg.feasible[(ii, xi)]; best = min(int(ti[y]) for y in feas)
            t = sum(1 for y in feas if int(ti[y]) == best)
            r *= (2 ** t - 1)
    return r

def main(N=200000, seed=42):
    rng = np.random.RandomState(seed)
    R = np.empty(N); M = np.empty(N, int)
    for i in range(N):
        t = tuple(orders[rng.randint(NO)] for _ in range(n))
        R[i] = r_of(t); M[i] = m_of(t)
    cm = np.array([c(m) for m in M])
    Cacc = cm                 # acceptance per-label cost
    Cfull = R * cm            # full per-label cost (lower-bound model)

    print(f"N={N}")
    print(f"\n-- m(R): acceptance mixing dim --")
    print(f"  median {np.median(M):.0f}  mean {M.mean():.2f}  p99 {np.percentile(M,99):.0f}  max {M.max()}")
    print(f"-- r(R): candidate proposal supports --")
    for p in [50,90,99,99.9]:
        print(f"  p{p}: {np.percentile(R,p):,.0f}")
    print(f"  mean {R.mean():,.0f}   max {R.max():,.0f}")

    print(f"\n-- interaction: are high-r labels also high-m? --")
    print(f"  Spearman rho(r, m) = {spearmanr(R, M).statistic:.3f}")
    print(f"  median r by m:")
    for mv in range(0, M.max()+1):
        sel = M == mv
        if sel.sum() >= 20:
            print(f"    m={mv:>2}: median r={np.median(R[sel]):>12,.0f}  mean r={R[sel].mean():>14,.0f}  (n={sel.sum()})")

    def conc(cost, name):
        s = np.sort(cost); tot = s.sum()
        print(f"\n-- cost concentration [{name}] --")
        for q in [1,0.1,0.01]:
            k = max(1, int(q/100*len(s)))
            print(f"  top {q:>4}% of labels: {100*s[-k:].sum()/tot:6.2f}% of total cost")
        print(f"  single most expensive label: {100*s[-1]/tot:.2f}% of total cost")
    conc(Cacc, "acceptance ~ c(m)")
    conc(Cfull, "full ~ r*c(m)")

    print(f"\n-- totals & confidence (10-way split, mean per chunk) --")
    for cost, name, scale in [(Cacc,"acceptance",1), (Cfull,"full",1)]:
        chunks = np.array_split(cost, 10)
        means = np.array([ch.mean() for ch in chunks])
        tot_core_h = cost.mean()*TOTAL/3600
        rel = means.std()/means.mean()
        print(f"  {name:<11}: E[cost/label]={cost.mean():.4g}s  total={tot_core_h:,.3g} core-h "
              f"= {tot_core_h/14/24:,.3g} d/14w ; chunk-mean rel.scatter={rel:.1%} "
              f"({'STABLE' if rel<0.25 else 'UNSTABLE (heavy tail)'})")

    # Lorenz (cost-concentration) plot, consistent with the measured c(m)
    plt.figure(figsize=(6.2,4.2))
    for cost,lab,col in [(Cacc,"acceptance $\\sim c(m)$","#36b"),(Cfull,"full $\\sim r\\,c(m)$","#b33")]:
        s=np.sort(cost); cum=np.cumsum(s)/s.sum(); fl=np.arange(1,len(s)+1)/len(s)
        plt.plot(fl,cum,label=lab,color=col)
    plt.plot([0,1],[0,1],"k--",lw=.6,label="uniform"); plt.xlabel("fraction of labels (cheapest first)")
    plt.ylabel("fraction of total work"); plt.title("Cost concentration (measured $c(m)$)")
    plt.legend(loc="upper left"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(HERE/"plot_lorenz.png",dpi=130); plt.close()

    # joint hist plot (log r vs m)
    plt.figure(figsize=(6,4.2))
    hb=plt.hexbin(M, np.log10(np.clip(R,1,None)), gridsize=25, bins='log', cmap='viridis')
    plt.colorbar(hb,label='log10 count'); plt.xlabel('m(R) (acceptance mixing dim)')
    plt.ylabel('log10 r(R) (proposal supports)'); plt.title('Joint distribution of the two cost components')
    plt.tight_layout(); plt.savefig(HERE/"plot_joint_r_m.png",dpi=130); plt.close()
    print(f"\nwrote plot_joint_r_m.png")

if __name__ == "__main__":
    main()
