#!/usr/bin/env python3
"""Measurement harness for the full-mixing solver.

A frozen, stratified benchmark corpus + clean per-stage measurement + big-N projection,
so every optimization (FLINT spike, etc.) is judged against the SAME labels with an
exactness-parity gate. Two subcommands:

  build  -- pick a frozen stratified corpus from the cheapest-first order, save it.
  bench  -- measure the current solver on the frozen corpus: per-stage time (wall+cpu),
            structural (r, nv), msolve outcome mix, and a parity signature per label.
            Writes a JSON baseline; a later run (e.g. FLINT) is diffed against it.

Methodology notes (for trustworthy numbers):
  * run single-process, ideally on a QUIESCED machine (pause the big run first); use
    --pin to taskset onto an isolated core.
  * repeats>1, report median + spread; we record CPU time too (subprocess msolve is not
    counted in CPU, so wall-vs-cpu gap exposes spawn/IO overhead under contention).
  * the corpus is frozen on disk (packed label indices) so results are comparable forever.
"""
import sys, os, time, json, argparse, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]   # full_search/analysis/ -> repo root
sys.path.insert(0, str(ROOT))
import lib.equilibrium.full_search.full_mixing_sweep as fm
from lib.equilibrium.full_search.full_mixing_sweep import (
    FullMixingSolver, c_model, PROF, MSTAT, DATA)

BENCH = DATA          # order.npy + bench corpora live in full_search/data/ (gitignored)
REPORTS = ROOT / "reports" / "fullmix_bench"


def _unpack(s, packed):
    NO = s.NO
    a = packed // (NO * NO); b = (packed // NO) % NO; c = packed % NO
    return (s.orders[a], s.orders[b], s.orders[c])


def _structural(s, tiers):
    """Cheap structural descriptors of a label: r (candidate supports), m, and the
    nv distribution over candidate profiles (max + count by value)."""
    r, m = s.r_and_m(tiers)
    nvs = []
    for profile in s.candidate_profiles(tiers):
        acc, wk = s._vars_for_profile(tiers, profile)
        nvs.append(len(acc) + len(wk))
    nvs = np.array(nvs, dtype=int)
    return dict(r=int(r), m=int(m), n_profiles=int(len(nvs)),
                nv_max=int(nvs.max()) if len(nvs) else 0,
                nv_mean=float(nvs.mean()) if len(nvs) else 0.0)


def _sig(result):
    """Exactness-parity signature of a find_in_label result; stable across implementations."""
    profile, payload = result
    if profile is None:
        return "none"
    if profile == "deferred":
        return f"deferred:{payload}"
    # verified equilibrium: canonicalize support + witness exactly (as strings)
    supp = sorted((tuple(k), tuple(sorted(v))) for k, v in profile.items())
    wit = sorted((str(k), str(v)) for k, v in payload.items())
    blob = repr((supp, wit)).encode()
    return "found:" + hashlib.sha1(blob).hexdigest()[:16]


def build(args):
    s = FullMixingSolver(args.payoff)
    order = np.load(BENCH / f"fullmix_{args.payoff}_order.npy")
    N = len(order)
    rng = np.random.RandomState(args.seed)
    if args.mode == "random":
        # UNIFORM random over the cheapest-50% order: captures the true cost distribution
        # (incl. the rare expensive tail) in proportion -> unbiased total-time estimator.
        picked = sorted(set(int(order[i]) for i in rng.randint(0, N, args.n)))
        desc = f"uniform-random n={args.n}"
    else:
        # stratified by position quantile (== predicted-cost strata): good for the cost CURVE
        picked = []
        for q in np.linspace(0.0, 1.0, args.strata):
            center = int(q * (N - 1)); lo = max(0, center - args.per // 2)
            picked.extend(int(order[p]) for p in range(lo, min(lo + args.per, N)))
        picked = sorted(set(picked))
        desc = f"{args.strata} strata x ~{args.per}"
    out = BENCH / f"fullmix_{args.payoff}_{args.name}.npy"
    np.save(out, np.array(picked, dtype=np.int64))
    print(f"[build] {args.payoff}: corpus '{args.name}' of {len(picked)} labels ({desc}) -> {out.name}")


def _measure_one(s, packed, repeats):
    tiers = _unpack(s, packed)
    struct = _structural(s, tiers)
    walls, cpus = [], []
    sig = None; stages = None; mstat = None
    for _ in range(repeats):
        PROF.clear(); MSTAT.clear()
        w0, c0 = time.perf_counter(), time.process_time()
        res = s.find_in_label(tiers)
        walls.append(time.perf_counter() - w0)
        cpus.append(time.process_time() - c0)
        sig = _sig(res)
        stages = {k: [round(v[0], 6), v[1]] for k, v in PROF.items()}
        mstat = dict(MSTAT)
    return dict(packed=int(packed), **struct,
                wall_med=float(np.median(walls)), wall_min=float(np.min(walls)),
                cpu_med=float(np.median(cpus)),
                stages=stages, mstat=mstat, sig=sig)


def _make_solver(which, payoff):
    if which == "flint":
        from lib.equilibrium.full_search.routes.flint_spike.flint_solver import FlintMixingSolver
        return FlintMixingSolver(payoff)
    if which == "julia":
        from lib.equilibrium.full_search.routes.julia_spike.julia_solver import FlintJuliaSolver
        s = FlintJuliaSolver(payoff)
        # warm up the Julia JIT (build+solve a tiny degree-2 system) so it doesn't pollute timing
        import flint
        ctx = flint.fmpq_mpoly_ctx.get(["x0", "x1"], flint.Ordering.lex)
        g = ctx.gens()
        s._msolve_flint(2, [g[0] * g[0] - 1, g[1] - g[0]], ["a_0_0_0", "w_0_0_0"])
        return s
    return FullMixingSolver(payoff)


def bench(args):
    s = _make_solver(args.solver, args.payoff)
    s.max_nv = args.max_nv; s.msolve_timeout = args.msolve_timeout
    if args.pin is not None:
        os.sched_setaffinity(0, {args.pin})
        print(f"[bench] pinned to CPU {args.pin}")
    corpus = np.load(BENCH / f"fullmix_{args.payoff}_{args.corpus}.npy")
    if args.limit:
        corpus = corpus[:args.limit]
    print(f"[bench] {args.payoff}: {len(corpus)} labels, repeats={args.repeats}, "
          f"max_nv={args.max_nv}, msolve_timeout={args.msolve_timeout}s", flush=True)
    rows = []; t0 = time.time()
    for i, packed in enumerate(corpus.tolist(), 1):
        rows.append(_measure_one(s, packed, args.repeats))
        if i % 25 == 0 or i == len(corpus):
            print(f"[bench] {i}/{len(corpus)}  ({time.time()-t0:.0f}s)", flush=True)

    # aggregate
    tot_wall = sum(r["wall_med"] for r in rows)
    tot_cpu = sum(r["cpu_med"] for r in rows)
    stage_tot = {}
    for r in rows:
        for k, (sec, n) in r["stages"].items():
            e = stage_tot.setdefault(k, [0.0, 0]); e[0] += sec; e[1] += n
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{args.tag}.json"
    payload = dict(payoff=args.payoff, tag=args.tag, n_labels=len(rows),
                   repeats=args.repeats, max_nv=args.max_nv,
                   msolve_timeout=args.msolve_timeout,
                   total_wall_med=tot_wall, total_cpu_med=tot_cpu,
                   stage_totals={k: v for k, v in stage_tot.items()}, rows=rows)
    out.write_text(json.dumps(payload, indent=1))

    print(f"\n[bench] corpus total: wall {tot_wall:.2f}s  cpu {tot_cpu:.2f}s  "
          f"(gap {tot_wall-tot_cpu:.2f}s = subprocess/IO/contention)")
    print("[bench] stage breakdown (corpus-summed):")
    for k, (sec, n) in sorted(stage_tot.items(), key=lambda kv: -kv[1][0]):
        print(f"   {k:<12} {sec:8.2f}s  {100*sec/max(tot_wall,1e-9):5.1f}%  {n:>7} calls")
    print(f"[bench] wrote {out}")
    print(f"[bench] parity fingerprint: "
          f"{hashlib.sha1(repr([(r['packed'], r['sig']) for r in rows]).encode()).hexdigest()[:16]}")


def project(args):
    """Bootstrap the total cheapest-50% run time from a bench JSON measured on a
    UNIFORM-random corpus (unbiased). Reports a confidence interval in wall-days."""
    d = json.loads((REPORTS / f"{args.tag}.json").read_text())
    w = np.array([r["wall_med"] for r in d["rows"]])
    order = np.load(BENCH / f"fullmix_{d['payoff']}_order.npy"); N = len(order)
    W = args.workers
    rng = np.random.RandomState(0)
    B = 5000
    means = np.array([w[rng.randint(0, len(w), len(w))].mean() for _ in range(B)])
    to_days = lambda m: m * N / W / 86400.0
    lo, mid, hi = np.percentile(means, [2.5, 50, 97.5])
    order_idx = np.argsort(w)
    tail1 = w[order_idx[int(0.99 * len(w)):]].sum() / w.sum()
    print(f"[project] {d['payoff']} tag={args.tag}: n={len(w)} labels, "
          f"mean {1000*w.mean():.2f} ms/label, on {W} workers")
    print(f"[project] cheapest-50% ({N:,} labels): "
          f"{to_days(mid):.2f} wall-days  (95% CI {to_days(lo):.2f}–{to_days(hi):.2f})")
    print(f"[project] heavy-tail: top 1% of sampled labels hold {100*tail1:.0f}% of total time; "
          f"max single label {w.max():.2f}s")
    print(f"[project] -> CI width {100*(to_days(hi)-to_days(lo))/to_days(mid):.0f}% of estimate "
          f"({'tight - high confidence' if (hi-lo)/mid < 0.25 else 'wide - tail under-sampled, need more labels'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("build")
    bp.add_argument("--payoff", default="burke_usaruschn_2035-2060")
    bp.add_argument("--name", default="bench_corpus")    # corpus file suffix
    bp.add_argument("--mode", choices=["strata", "random"], default="strata")
    bp.add_argument("--n", type=int, default=4000)        # random-mode sample size
    bp.add_argument("--strata", type=int, default=21)   # 0%,5%,...,100% of the order
    bp.add_argument("--per", type=int, default=15)       # labels per stratum
    bp.add_argument("--seed", type=int, default=0)
    mp = sub.add_parser("bench")
    mp.add_argument("--payoff", default="burke_usaruschn_2035-2060")
    mp.add_argument("--tag", default="baseline")
    mp.add_argument("--solver", choices=["baseline", "flint", "julia"], default="baseline")
    mp.add_argument("--repeats", type=int, default=3)
    mp.add_argument("--max-nv", type=int, default=8)
    mp.add_argument("--msolve-timeout", type=float, default=10.0)
    mp.add_argument("--pin", type=int, default=None)     # taskset to one CPU id
    mp.add_argument("--limit", type=int, default=0)      # measure only first N (validation)
    mp.add_argument("--corpus", default="bench_corpus")  # which corpus file to load
    pp = sub.add_parser("project")
    pp.add_argument("--tag", required=True)               # bench JSON to project from
    pp.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"build": build, "bench": bench, "project": project}[a.cmd](a)
