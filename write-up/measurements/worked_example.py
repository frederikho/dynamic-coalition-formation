#!/usr/bin/env python3
"""Worked combinatorics example: a maximum-work label vs a minimum-work label, the
full-model budget-vs-cutoff table, and the precise (non-rounded) cost concentration.

Clarifies what is estimable: r(R) and m(R) are deterministic structural functions
(no solving); c(m) is measured; so per-label cost r(R)*c(m(R)) is computable for ANY
label, including the single worst one. The only thing with high sampling variance is
the COMPLETE total (heavy tail), not the per-label or cheapest-fraction costs.
"""
import sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "write-up" / "measurements"))
from bench_build_interp import cls, fg, n, S, orders, NO, m_of
TOTAL = NO ** 3
CM = {0:0.0001,1:0.0009,2:0.0017,3:0.0111,4:0.0436,5:0.6312,6:0.6482,
      7:9.926,8:18.61,9:25.06,10:58.25,11:60.0,12:60.0,13:60.0,14:60.0}
def c(m): return CM.get(m, 60.0)
states = fg.states

def r_breakdown(tiers):
    rows=[]; r=1
    for ii in range(n):
        ti=tiers[ii]
        for xi in range(S):
            feas=fg.feasible[(ii,xi)]; best=min(int(ti[y]) for y in feas)
            t=sum(1 for y in feas if int(ti[y])==best)
            fac=2**t-1; r*=fac
            rows.append((fg.players[ii],states[xi],t,fac))
    return r, rows

def m_count(tiers):
    return m_of(tiers)

def describe(name, tiers):
    r, rows=r_breakdown(tiers); m=m_count(tiers)
    print(f"\n=== {name} ===")
    for ii in range(n):
        order=sorted(range(S), key=lambda s:int(tiers[ii][s]))
        groups=[];
        import itertools as it
        for tier,grp in it.groupby(order, key=lambda s:int(tiers[ii][s])):
            groups.append("=".join(states[s] for s in grp))
        print(f"  {fg.players[ii]} ranking: "+"  >  ".join(groups))
    print(f"  per-cell (proposer,state): t=top-tier multiplicity, factor=2^t-1")
    # show only cells with factor>1 (the ones that create work), summarise others
    nontrivial=[x for x in rows if x[3]>1]
    for (p,xs,t,fac) in nontrivial[:6]:
        print(f"    {p} @ {xs}: t={t} factor={fac}")
    if len(nontrivial)>6: print(f"    ... ({len(nontrivial)} cells with factor>1)")
    print(f"  r(R)=prod factors = {r:,}")
    print(f"  m(R)={m}   c(m)~{c(m)}s")
    print(f"  acceptance cost C_acc=c(m)={c(m):.4g}s")
    print(f"  full cost  C_full~r*c(m)={r*c(m):.4g}s  (= {r*c(m)/3600/24/365:.3g} years for THIS ONE label)")
    return r,m

# HIGH: fully tied (every player indifferent among all 5 states)
hi = tuple(np.zeros(S, dtype=np.int64) for _ in range(n))
# LOW: strict distinct order for every player
lo = tuple(np.arange(S, dtype=np.int64) for _ in range(n))
describe("MAX-work label: every player indifferent among all states", hi)
describe("MIN-work label: every player strictly orders all states", lo)

# Full-model budget table + precise concentration from a sample
rng=np.random.RandomState(7); Nn=100000
R=np.empty(Nn); M=np.empty(Nn,int)
for i in range(Nn):
    t=tuple(orders[rng.randint(NO)] for _ in range(n)); R[i]=r_breakdown(t)[0]; M[i]=m_of(t)
Cfull=R*np.array([c(m) for m in M]); Cacc=np.array([c(m) for m in M])
scale=TOTAL/Nn
print("\n=== FULL-MODEL budget table: cheapest X% of labels ===")
print(f"{'cheapest':>10} {'core-hours':>14} {'14 workers':>14} {'134 workers':>14}")
sf=np.sort(Cfull); cf=np.cumsum(sf)
for p in [0.5,0.9,0.99,0.999,0.9999]:
    k=max(1,int(p*Nn)); ch=cf[k-1]*scale/3600
    print(f"{p*100:>9.2f}% {ch:>14,.1f} {ch/14/24:>12,.2f}d {ch/134/24:>12,.2f}d")
print(f"   complete (incl. unsampled all-tied label r={r_breakdown(hi)[0]:,}): dominated by that single label")
print("\n=== precise concentration (full cost), no rounding ===")
tot=sf.sum()
for q in [1,0.1,0.01]:
    k=max(1,int(q/100*Nn)); print(f"  top {q}% of labels: {100*sf[-k:].sum()/tot:.4f}% of sample full-cost")
print(f"  single most expensive SAMPLED label: {100*sf[-1]/tot:.4f}%  (r={R.max():,.0f})")
print(f"  cheapest 99% of labels: {100*cf[int(0.99*Nn)-1]/tot:.4f}% of sample full-cost")
