#!/usr/bin/env python3
"""Empirically time find_in_label across the cost spectrum of the cheapest-first order.

We can't fit t_label(cost) from the live log yet (all solved labels so far are in the
cheapest tier). So sample labels at representative positions of the sorted order, time
the EXACT worker path (find_in_label with max_nv=8), and report measured wall-sec per
label vs predicted cost r*c(m). Runs alongside the live job, so per-label wall time here
~ the per-worker time under the same contention; aggregate run rate ~ 14 / t_label.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]   # full_search/analysis/ -> repo root
sys.path.insert(0, str(ROOT))
from lib.equilibrium.full_search.full_mixing_sweep import FullMixingSolver, c_model, DATA

PAYOFF = sys.argv[1] if len(sys.argv) > 1 else "burke_usaruschn_2035-2060"
WORKERS = 14
s = FullMixingSolver(PAYOFF); s.max_nv = 8
NO = s.NO
order = np.load(DATA / f"fullmix_{PAYOFF}_order.npy")
N = len(order)

def unpack(packed):
    a = packed // (NO*NO); b = (packed // NO) % NO; c = packed % NO
    return (s.orders[a], s.orders[b], s.orders[c])

# (frac-of-order, #labels to time there). Denser/fewer in the expensive tail.
plan = [(0.10,40),(0.30,40),(0.50,40),(0.65,30),(0.80,30),(0.88,25),
        (0.93,20),(0.96,15),(0.98,12),(0.99,10),(0.995,8),(0.999,6),(1.0,6)]

rng = np.random.RandomState(0)
rows = []   # (frac, mean_cost, mean_t, n, n_defer)
print(f"{'frac':>6} {'pos':>12} {'mean_cost':>11} {'sec/label':>10} {'lab/s/wkr':>10} {'defer%':>7}", flush=True)
for frac, n in plan:
    p0 = min(int(frac * (N-1)), N-1)
    # take n labels around that position
    lo = max(0, p0 - n//2); idxs = list(range(lo, min(lo+n, N)))
    costs = []; ts = []; ndef = 0
    for p in idxs:
        t = unpack(int(order[p]))
        r, m = s.r_and_m(t); costs.append(r * c_model(m))
        t0 = time.time()
        res, _ = s.find_in_label(t)
        ts.append(time.time() - t0)
        if res == "deferred": ndef += 1
    mc = float(np.mean(costs)); mt = float(np.mean(ts))
    rows.append((frac, mc, mt, len(idxs), ndef))
    print(f"{frac:6.3f} {p0:12,} {mc:11.4g} {mt:10.4f} {WORKERS/mt:10.1f} "
          f"{100*ndef/len(idxs):7.0f}", flush=True)

# Forecast the remaining run by integrating measured t_label over remaining labels.
# Map every label's predicted cost -> measured t via the (frac->t) samples, but it's
# cleaner to integrate per position-segment using the sampled mean t in each segment.
ckpt = DATA / f"fullmix_{PAYOFF}_find_progress.txt"
cur = int(ckpt.read_text().split()[0]) if ckpt.exists() else 0
fr = np.array([r[0] for r in rows]); mt = np.array([r[2] for r in rows])
# piecewise-constant t_label across order fractions: integrate from cur/N to 1.0
edges = np.concatenate([[0.0], (fr[:-1]+fr[1:])/2, [1.0]])   # segment boundaries in frac
seg_lo, seg_hi = edges[:-1], edges[1:]
cur_frac = cur / N
sec = 0.0
for lo, hi, t in zip(seg_lo, seg_hi, mt):
    lo = max(lo, cur_frac)
    if hi <= lo: continue
    sec += (hi - lo) * N * t          # labels in segment * sec/label (one worker)
sec_wall = sec / WORKERS              # spread over workers
print(f"\nposition {cur:,} ({100*cur_frac:.1f}%)")
print(f"measured-cost ETA for remaining: {sec_wall/3600:.1f} h ({sec_wall/86400:.2f} days) "
      f"on {WORKERS} workers")
print("(per-label times measured under live-job contention ~ per-worker run time)")
