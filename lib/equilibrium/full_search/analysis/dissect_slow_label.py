#!/usr/bin/env python3
"""Find one >10s label in the right tail and explain WHERE the time goes.

Scans the expensive end of the cheapest-first order, times find_in_label, and on the
first label that exceeds THRESH seconds dumps a full structural breakdown: the weak-order
label, the candidate proposal supports r, and per-support the variable count nv, the
vertex-evaluation count 2^nv (each an exact SxS fmpq solve), the msolve outcome, and the
wall time. Then the PROF stage breakdown for that single label.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]   # full_search/analysis/ -> repo root
sys.path.insert(0, str(ROOT))
import lib.equilibrium.full_search.full_mixing_sweep as fm
from lib.equilibrium.full_search.full_mixing_sweep import FullMixingSolver, c_model, PROF, prof_report

PAYOFF = sys.argv[1] if len(sys.argv) > 1 else "burke_usaruschn_2035-2060"
START_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.88
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

s = FullMixingSolver(PAYOFF); s.max_nv = 8; s.msolve_timeout = 10.0
NO = s.NO
order = np.load(ROOT / "strategy_tables" / f"fullmix_{PAYOFF}_order.npy")
N = len(order)

def unpack(packed):
    a = packed // (NO*NO); b = (packed // NO) % NO; c = packed % NO
    return (s.orders[a], s.orders[b], s.orders[c])

p = int(START_FRAC * (N-1))
print(f"scanning from position {p:,} (frac {START_FRAC}) for a >{THRESH}s label...", flush=True)
chosen = None
while p < N:
    t = unpack(int(order[p]))
    t0 = time.time(); s.find_in_label(t); dt = time.time() - t0
    if dt >= THRESH:
        chosen = (p, t, dt); break
    p += 1
if chosen is None:
    sys.exit("no >THRESH label found in scan range")

pos, tiers, dt = chosen
r, m = s.r_and_m(tiers)
print(f"\n=== SLOW LABEL at order position {pos:,} ({100*pos/N:.2f}%) ===")
print(f"find_in_label wall time: {dt:.1f}s   (under live-job contention)")
print(f"predicted cost r*c(m) = {r} * {c_model(m)} = {r*c_model(m):.4g}   (m={m} tied contexts)")
print("weak-order tiers (player -> rank of each state; lower=preferred):")
for ii in range(s.n):
    print(f"  {s.fg.players[ii]:>6}: {list(map(int, tiers[ii]))}   states={list(s.states)}")

# Re-walk the candidate profiles with per-support instrumentation.
PROF.clear()
print(f"\ncandidate proposal supports (r should be {r}):")
print(f"{'#':>4} {'nv':>3} {'2^nv':>6} {'status':>11} {'sec':>8}  supports")
n_prof=0; n_def=0; n_solved=0; n_timeout=0; tot_vertex=0; t_solved=0.0
for profile in s.candidate_profiles(tiers):
    n_prof += 1
    acc, wk = s._vars_for_profile(tiers, profile)
    nv = len(acc) + len(wk)
    tp = time.time(); status, payload = s.solve_profile(tiers, profile); st = time.time()-tp
    if status == "deferred" and nv > s.max_nv:
        tag = "defer(nv>8)"; n_def += 1
    elif status == "deferred":
        tag = "defer(posdim/to)"; n_def += 1; n_timeout += 1
    else:
        tag = status; n_solved += 1; tot_vertex += (1 << nv); t_solved += st
    # compact support description
    supp = {f"{s.fg.players[i]}@{s.states[x]}": "".join(s.states[y] or '.' for y in v)
            for (i, x), v in profile.items() if len(v) > 1 or any(yy!=x for yy in v)}
    if n_prof <= 40 or st > 0.5:   # don't spam: show first 40 + any slow ones
        print(f"{n_prof:>4} {nv:>3} {1<<nv:>6} {tag:>11} {st:>8.3f}  "
              f"{ {k:v for k,v in list(supp.items())[:3]} }")
print(f"\nTOTALS: r={n_prof} candidate supports; {n_solved} solved, {n_def} deferred "
      f"({n_timeout} via posdim/timeout, {n_def-n_timeout} via nv>{s.max_nv})")
print(f"vertex evaluations (exact {s.S}x{s.S} fmpq solves) in solved supports: {tot_vertex:,}")
print(f"time in solved supports: {t_solved:.1f}s; deferred(nv>8) are near-instant skips")
print("\nstage breakdown for this one label:\n" + prof_report())
