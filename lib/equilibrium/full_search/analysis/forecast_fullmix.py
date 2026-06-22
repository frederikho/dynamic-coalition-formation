#!/usr/bin/env python3
"""Cost-model ETA for the full-mixing `find` run.

The cheapest-first order in fullmix_<payoff>_order.npy is sorted ascending by the
structural predicted cost r*c(m) (the SAME quantity the solver spends time on). So
wall-seconds-per-label should be ~ affine in that cost: t_label = eps + kappa*cost
(eps = fixed per-label dispatch/overhead, kappa = seconds per unit predicted cost).

We (1) sample cost across the whole order, (2) read observed (position, recent lab/s)
pairs from logs/fullmix_<payoff>_find.log, (3) fit eps,kappa by least squares on
1/lab_s vs cost, (4) integrate t_label over the remaining labels for an ETA.
"""
import sys, re, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]   # full_search/analysis/ -> repo root
sys.path.insert(0, str(ROOT))
from lib.equilibrium.full_search.full_mixing_sweep import FullMixingSolver, c_model, DATA

PAYOFF = sys.argv[1] if len(sys.argv) > 1 else "burke_usaruschn_2035-2060"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 60000   # cost-sample positions

s = FullMixingSolver(PAYOFF)
NO = s.NO
order = np.load(DATA / f"fullmix_{PAYOFF}_order.npy")
N = len(order)

def cost_of(packed):
    a = packed // (NO*NO); b = (packed // NO) % NO; c = packed % NO
    r, m = s.r_and_m((s.orders[a], s.orders[b], s.orders[c]))
    return r * c_model(m)

# (1) sample cost across the whole (monotone-by-construction) order
pos = np.linspace(0, N-1, M).astype(np.int64)
cost = np.array([cost_of(int(order[p])) for p in pos])
# cumulative mean cost up to each sampled position -> integral approximation
# mean cost over a position range == mean of samples in that range (uniform sampling)
total_cost = cost.mean() * N

# (2) parse the log for (k, recent lab/s)
log = (ROOT / "logs" / f"fullmix_{PAYOFF}_find.log").read_text().splitlines()
pat = re.compile(r"\[find\] ([\d,]+)/[\d,]+ .*recent ([\d.]+) lab/s")
obs_k, obs_rate = [], []
for ln in log:
    mobj = pat.search(ln)
    if mobj:
        obs_k.append(int(mobj.group(1).replace(",", "")))
        obs_rate.append(float(mobj.group(2)))
obs_k = np.array(obs_k); obs_rate = np.array(obs_rate)
if len(obs_k) < 3:
    sys.exit(f"only {len(obs_k)} log points; need a few more progress lines first")

# cost at each observed position via interpolation on the (monotone) sampled curve
obs_cost = np.interp(obs_k, pos, cost)
obs_tlabel = 1.0 / obs_rate                       # aggregate wall-sec per label

# (3) non-negative least-squares fit  t_label = eps + kappa*cost  (eps,kappa >= 0,
#     both physically required: eps = fixed overhead floor, kappa = sec per cost unit)
from scipy.optimize import nnls
A = np.vstack([np.ones_like(obs_cost), obs_cost]).T
(eps, kappa), _ = nnls(A, obs_tlabel)
pred = A @ np.array([eps, kappa])
ss_res = float(((obs_tlabel - pred) ** 2).sum())
ss_tot = float(((obs_tlabel - obs_tlabel.mean()) ** 2).sum())
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
print("observed vs fitted at log points (cost, lab/s obs, lab/s fit):")
for c_, ro_, pf_ in sorted(zip(obs_cost, obs_rate, 1.0/np.maximum(pred,1e-12))):
    print(f"  cost={c_:.4g}  obs={ro_:8.0f}  fit={pf_:8.0f}")
print(f"fit R^2={r2:.3f}\n")

# current position = last logged k (or checkpoint)
cur = int(obs_k.max())
ckpt = DATA / f"fullmix_{PAYOFF}_find_progress.txt"
if ckpt.exists():
    cur = max(cur, int(ckpt.read_text().split()[0]))

# (4) integrate predicted t_label over remaining labels
rem_mask = pos >= cur
rem_mean_cost = cost[rem_mask].mean() if rem_mask.any() else cost[-1]
rem_n = N - cur
eta_rem = rem_n * (eps + kappa * rem_mean_cost)
done_mean_cost = cost[~rem_mask].mean() if (~rem_mask).any() else cost[0]

print(f"payoff={PAYOFF}  labels N={N:,}  position={cur:,} ({100*cur/N:.1f}%)")
print(f"cost samples M={M:,}")
print(f"fit  t_label = {eps*1e6:.2f}us + {kappa:.3e} * cost   (aggregate, 14 workers)")
print(f"  implied floor rate (cost->0): {1/eps:,.0f} lab/s")
print()
print(f"predicted cost   total = {total_cost:.4g}")
print(f"  mean cost done   (first {100*cur/N:.1f}%): {done_mean_cost:.4g}")
print(f"  mean cost remain (last {100*(N-cur)/N:.1f}%): {rem_mean_cost:.4g}"
      f"   ({rem_mean_cost/max(done_mean_cost,1e-30):.1f}x the cheap part)")
print()
print(f"observed lab/s range in log: {obs_rate.min():.0f} .. {obs_rate.max():.0f}")
print(f"--> remaining {rem_n:,} labels: ETA {eta_rem/3600:.1f} h ({eta_rem/86400:.2f} days)")
print(f"    effective remaining rate: {rem_n/eta_rem:,.0f} lab/s")

# cost percentiles across the order (how back-loaded is the work?)
qs = [50, 80, 90, 95, 99, 100]
print("\ncost percentiles over the order (cheapest-first):")
for q in qs:
    print(f"  p{q:>3}: cost={np.percentile(cost, q):.4g}")
# fraction of TOTAL predicted cost in the last 10% / 1% of labels
csum = np.cumsum(cost) / cost.sum()
for tail in (0.10, 0.05, 0.01):
    idx = int((1-tail)*M)
    print(f"  last {tail*100:.0f}% of labels hold "
          f"{100*(1-csum[idx]):.1f}% of total predicted cost")
