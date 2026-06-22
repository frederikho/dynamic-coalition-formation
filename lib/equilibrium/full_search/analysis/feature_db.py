#!/usr/bin/env python3
"""Cheap structural feature DB + runtime calibration + cheapest-50% forecast.

Features (computable WITHOUT solving): per label r (=#candidate supports), m, n_prof, and
sum_2nv = sum over solved (nv<=max_nv) supports of 2^nv (the vertex-eval cost proxy).
Runtime model is calibrated on a sample we actually solved with the Julia solver
(reports/fullmix_bench/<cal_tag>.json) and applied to the whole cheap-50% (order.npy).

  calibrate  -- fit wall ~ a*sum_2nv + b*n_prof + c on the solved sample; print R^2, save coeffs.
  build      -- parallel-compute (r,m,n_prof,sum_2nv) for every cheap-50% label -> features.npz.
  forecast   -- apply the calibrated model to features.npz; total cheapest-50% runtime + CI-free sum.
"""
import sys, json, time, argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from lib.equilibrium.full_search.full_mixing_sweep import FullMixingSolver, DATA

REPORTS = ROOT / "reports" / "fullmix_bench"
MODEL = REPORTS / "runtime_model.json"

_S = None
def _winit(payoff, max_nv):
    global _S
    _S = FullMixingSolver(payoff); _S.max_nv = max_nv

def _label_feats(s, pk):
    NO = s.NO
    a = pk // (NO * NO); b = (pk // NO) % NO; c = pk % NO
    t = (s.orders[a], s.orders[b], s.orders[c])
    r, m = s.r_and_m(t); s2 = 0; npf = 0
    for prof in s.candidate_profiles(t):
        acc, wk = s._vars_for_profile(t, prof); nv = len(acc) + len(wk); npf += 1
        if nv <= s.max_nv:
            s2 += (1 << nv)
    return (float(r), float(m), float(npf), float(s2))

def _feats_chunk(pks):
    s = _S
    return np.array([_label_feats(s, int(pk)) for pk in pks], dtype=np.float64)


def calibrate(args):
    s = FullMixingSolver(args.payoff); s.max_nv = args.max_nv
    rows = json.loads((REPORTS / f"{args.cal_tag}.json").read_text())["rows"]
    X = np.array([_label_feats(s, r["packed"])[2:4] for r in rows])   # [n_prof, sum_2nv]
    y = np.array([r["wall_med"] for r in rows])
    A = np.column_stack([X[:, 1], X[:, 0], np.ones(len(y))])          # sum_2nv, n_prof, 1
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_text(json.dumps({"payoff": args.payoff, "cal_tag": args.cal_tag,
                                 "a_sum2nv": coef[0], "b_nprof": coef[1], "c": coef[2],
                                 "r2": r2, "n_cal": len(y)}, indent=1))
    print(f"[calibrate] wall ~ {coef[0]:.4e}*sum_2nv + {coef[1]:.4e}*n_prof + {coef[2]:.4e}  "
          f"R^2={r2:.4f}  (n={len(y)})  -> {MODEL.name}")


def build(args):
    order = np.load(DATA / f"fullmix_{args.payoff}_order.npy"); N = len(order)
    chunks = [c.tolist() for c in np.array_split(order, args.workers * 50)]
    print(f"[build] {args.payoff}: {N:,} cheap-50% labels in {len(chunks)} chunks on {args.workers} workers", flush=True)
    parts = []; done = 0; t0 = time.time()
    with Pool(args.workers, initializer=_winit, initargs=(args.payoff, args.max_nv)) as pool:
        for arr in pool.imap(_feats_chunk, chunks):       # imap preserves order -> aligned
            parts.append(arr); done += len(arr)
            el = time.time() - t0
            print(f"[build] {done:,}/{N:,} ({100*done/N:.0f}%), {el:.0f}s, ETA {el/done*(N-done):.0f}s", flush=True)
    F = np.vstack(parts)
    out = DATA / f"fullmix_{args.payoff}_features.npz"
    np.savez(out, packed=order, r=F[:, 0], m=F[:, 1], n_prof=F[:, 2], sum2nv=F[:, 3])
    print(f"[build] saved {out.name} in {(time.time()-t0)/60:.1f} min")


def forecast(args):
    mdl = json.loads(MODEL.read_text())
    f = np.load(DATA / f"fullmix_{args.payoff}_features.npz")
    sum2nv = f["sum2nv"]; nprof = f["n_prof"]; N = len(sum2nv)
    pred = mdl["a_sum2nv"] * sum2nv + mdl["b_nprof"] * nprof + mdl["c"]
    pred = np.clip(pred, 0, None)                      # per-label single-core seconds
    tot_core = pred.sum(); W = args.workers
    print(f"[forecast] {args.payoff}: cheapest-50% = {N:,} labels, model R^2={mdl['r2']:.3f}")
    print(f"[forecast] total single-core: {tot_core/3600:.1f} h ({tot_core/86400:.2f} core-days)")
    print(f"[forecast] on {W} workers: {tot_core/W/86400:.2f} wall-days "
          f"({tot_core/W/3600:.1f} wall-hours)")
    print(f"[forecast] mean {1000*pred.mean():.2f} ms/label; "
          f"top 1% of labels hold {100*np.sort(pred)[int(0.99*N):].sum()/tot_core:.0f}% of time")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "build", "forecast"):
        p = sub.add_parser(name); p.add_argument("--payoff", default="burke_usaruschn_2035-2060")
        p.add_argument("--workers", type=int, default=14); p.add_argument("--max-nv", type=int, default=8)
        p.add_argument("--cal-tag", default="julia_bhoist")
    a = ap.parse_args()
    {"calibrate": calibrate, "build": build, "forecast": forecast}[a.cmd](a)
