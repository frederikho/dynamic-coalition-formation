#!/usr/bin/env python3
"""Profile the msolve component on the fixed rand4000 set: per call to _msolve_flint
record (nv, max eq total-degree, n_eqs, wall, status). Aggregate to see how much
subprocess time sits in LINEAR (max-deg<=1) systems -- solvable in-process with exact
FLINT linear algebra (parity-preserving) -- vs genuinely nonlinear ones needing msolve.
"""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from lib.equilibrium.full_search.full_mixing_sweep import DATA
from lib.equilibrium.full_search.routes.flint_spike.flint_solver import FlintMixingSolver
from lib.equilibrium.full_search.analysis.harness import _unpack

P = "burke_usaruschn_2035-2060"
STATS = []   # (nv, maxdeg, n_eqs, dt, status)

class ProfFlint(FlintMixingSolver):
    def _msolve_flint(self, nv, eqs, names):
        maxdeg = max((e.total_degree() for e in eqs), default=0)
        t0 = time.perf_counter()
        status, sols = super()._msolve_flint(nv, eqs, names)
        STATS.append((nv, int(maxdeg), len(eqs), time.perf_counter() - t0, status))
        return status, sols

s = ProfFlint(P); s.max_nv = 8
corpus = np.load(DATA / f"fullmix_{P}_rand4000.npy")
LIM = int(sys.argv[1]) if len(sys.argv) > 1 else len(corpus)
corpus = corpus[:LIM]
t0 = time.time()
for i, pk in enumerate(corpus.tolist(), 1):
    s.find_in_label(_unpack(s, pk))
    if i % 1000 == 0:
        print(f"  {i}/{len(corpus)}  ({time.time()-t0:.0f}s)", flush=True)
A = np.array([(nv, md, ne, dt) for nv, md, ne, dt, st in STATS], dtype=float)
status = [st for *_, st in STATS]
nv, md, ne, dt = A[:,0], A[:,1], A[:,2], A[:,3]
SPAWN = 0.004   # dt above this => almost certainly a real subprocess spawn

print(f"\n_msolve_flint calls: {len(STATS)}   total wall in stage: {dt.sum():.1f}s")
print(f"  early-return (dt<{SPAWN*1000:.0f}ms, no spawn): {int((dt<SPAWN).sum())} calls, {dt[dt<SPAWN].sum():.1f}s")
print(f"  likely spawns (dt>={SPAWN*1000:.0f}ms):          {int((dt>=SPAWN).sum())} calls, {dt[dt>=SPAWN].sum():.1f}s")
print("\nby max eq total-degree (deg 1 = LINEAR, in-process-able):")
for d in sorted(set(md.astype(int))):
    m = md == d
    print(f"  deg={d}: {int(m.sum()):6d} calls  {dt[m].sum():6.1f}s  "
          f"(spawns {int((m&(dt>=SPAWN)).sum())}, {dt[m&(dt>=SPAWN)].sum():.1f}s)")
lin = (md <= 1)
print(f"\nLINEAR (deg<=1) share of stage time: {100*dt[lin].sum()/dt.sum():.0f}%  "
      f"({dt[lin].sum():.1f}s of {dt.sum():.1f}s)")
print(f"  of which is spawn time recoverable in-process: {dt[lin&(dt>=SPAWN)].sum():.1f}s")
from collections import Counter
print("status mix:", dict(Counter(status)))

# spawns cross-tabbed by (degree, nv): feasibility of in-process resultant elimination
print("\nSPAWNS by (max-deg, nv)  [time s]:")
sp = dt >= SPAWN
for d in sorted(set(md[sp].astype(int))):
    row = []
    for v in sorted(set(nv[sp & (md == d)].astype(int))):
        msk = sp & (md == d) & (nv == v)
        row.append(f"nv={v}:{int(msk.sum())}({dt[msk].sum():.1f}s)")
    print(f"  deg={d}: " + "  ".join(row))
# the key number: spawn time in systems solvable in-process by linear(deg1) + bivariate-quadratic(deg2,nv<=2)
inproc = sp & (((md <= 1)) | ((md == 2) & (nv <= 2)))
print(f"\nin-process-able spawn time (deg<=1 OR deg2&nv<=2): {dt[inproc].sum():.1f}s "
      f"of {dt[sp].sum():.1f}s spawn  ({100*dt[inproc].sum()/dt[sp].sum():.0f}%)")
inproc3 = sp & (((md <= 1)) | ((md == 2) & (nv <= 3)))
print(f"  extending to deg2&nv<=3: {dt[inproc3].sum():.1f}s ({100*dt[inproc3].sum()/dt[sp].sum():.0f}%)")
