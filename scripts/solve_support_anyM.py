#!/usr/bin/env python3
"""
Given the support pattern, solve for the mixing probabilities -- any M.

The structure of the problem changes with M, and the reason is the indifference
principle:

  Each interior alpha contributes one equation -- its holder must be exactly
  indifferent -- and one unknown, its own probability.  But a player who mixes is
  by definition indifferent, so their own payoff cannot depend on their own
  weights: d(equation k)/d(theta k) = 0.  The Jacobian has a ZERO DIAGONAL.

  M = 1: that Jacobian is the 1x1 zero matrix.  Singular, underdetermined, so the
         solution is an INTERVAL bounded by other players' inequalities.  Scan it.
  M >= 2: the OFF-diagonal entries are nonzero -- theta_2 moves player 1's values --
         so the Jacobian is generically invertible and the solution is an ISOLATED
         POINT.  Root-find it; no grid, no pts^M blow-up.

Every candidate is accepted only after the framework's own verifier passes.

Usage:
    python scripts/solve_support_anyM.py [--profiles planted_profiles]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import root
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
sp = importlib.util.spec_from_file_location("enu", REPO/"scripts/enumerate_knobs_m1.py")
enu = importlib.util.module_from_spec(sp); sp.loader.exec_module(enu)
from lib.equilibrium.jeres_vfi import Game, fw_state_name_to_partition
from lib.equilibrium.jeres_vfi.solver import (compute_values, full_transition_matrix,
    verify_proposals, verify_responses)
from lib.effectivity import get_effectivity

PL=["CHN","EUR","USA"]; ST=["( )","(CHNEUR)","(CHNUSA)","(EURUSA)","(CHNEURUSA)"]
EPS=1e-12

def load(p):
    d=json.loads(p.read_text()); u=np.array(d["payoffs"],float)
    g0=Game.from_payoffs(PL,u)
    eff=get_effectivity("heyen_lehtomaa_2021",PL,ST)
    j_of={n:g0.state_idx[fw_state_name_to_partition(n,PL)] for n in ST}
    ac={(i,j_of[x],j_of[y]):frozenset(k for k,pk in enumerate(PL)
        if eff.get((pi,x,y,pk),0)==1)
        for i,pi in enumerate(PL) for x in ST for y in ST}
    g=Game(players=PL,payoffs=u,states=g0.states,state_idx=g0.state_idx,
           n_players=3,n_states=5,approval_committees=ac)
    sig=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["sigmas"]]
    alp=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["alphas"]]
    return d,g,enu.committees(g),sig,alp,[tuple(k) for k in d["knobs"]]

def state_at(g,comm,sig,alp,knobs,theta,delta):
    a2=[dict(x) for x in alp]
    for (kx,ky,kj),t in zip(knobs,theta): a2[kx][(kj,ky)]=float(t)
    qs=enu.qs_from_alphas(g,comm,a2)
    V=compute_values(g,full_transition_matrix(g,sig,qs,None),delta)
    return a2,qs,V

def verify(g,sig,a2,qs,V):
    r,_=verify_responses(g,sig,a2,qs,V,atol=EPS)
    p,_=verify_proposals(g,sig,a2,qs,V,atol=EPS)
    return r and p

def solve_one(d,g,comm,sig,alp,knobs,pts=41,starts=12):
    M=len(knobs); delta=d["delta"]
    if M==1:
        hits=[t for t in np.linspace(0.02,0.98,pts)
              if verify(g,sig,*state_at(g,comm,sig,alp,knobs,[t],delta)[:2],
                        state_at(g,comm,sig,alp,knobs,[t],delta)[2])]
        return ("interval", (min(hits),max(hits)) if hits else None, pts)
    def resid(th):
        a2,qs,V=state_at(g,comm,sig,alp,knobs,np.clip(th,1e-9,1-1e-9),delta)
        return np.array([V[y,j]-V[x,j] for (x,y,j) in knobs]) \
               + np.sum(np.abs(th-np.clip(th,1e-9,1-1e-9)))
    rng=np.random.default_rng(0); evals=0
    for s in range(starts):
        th0=np.full(M,0.5) if s==0 else rng.uniform(0.05,0.95,M)
        try: sol=root(resid,th0,method="hybr")
        except Exception: continue
        evals+=int(getattr(sol,"nfev",0))
        th=np.clip(sol.x,0.0,1.0)
        if not sol.success or np.max(np.abs(resid(th)))>1e-9: continue
        if not all(1e-9<t<1-1e-9 for t in th): continue
        a2,qs,V=state_at(g,comm,sig,alp,knobs,th,delta)
        if verify(g,sig,a2,qs,V): return ("point",th,evals)
    return ("point",None,evals)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--profiles",default="planted_profiles")
    a=ap.parse_args()
    PROF=REPO/"reports"/a.profiles
    print(f"{'table':<36}{'M':>2}{'kind':>10}{'solved':>8}{'result':>34}{'s':>8}")
    agg={}
    for f in sorted(PROF.glob("*.json")):
        d,g,comm,sig,alp,knobs=load(f)
        t0=time.time(); kind,res,ev=solve_one(d,g,comm,sig,alp,knobs); dt=time.time()-t0
        ok=res is not None
        if kind=="interval":
            txt=f"[{res[0]:.3f}, {res[1]:.3f}]" if ok else "-"
        else:
            txt=np.array2string(np.round(res,6)) if ok else "-"
        agg.setdefault(d["M"],[]).append((ok,dt))
        print(f"{d['file'][:35]:<36}{d['M']:>2}{kind:>10}{('YES' if ok else 'no'):>8}"
              f"{txt:>34}{dt:>8.2f}", flush=True)
    print(f"\n{'M':>2}{'tables':>8}{'solved':>8}{'median s':>11}{'max s':>9}")
    for m in sorted(agg):
        v=agg[m]; ts=sorted(x[1] for x in v)
        print(f"{m:>2}{len(v):>8}{sum(1 for x in v if x[0]):>8}"
              f"{ts[len(ts)//2]:>11.2f}{max(ts):>9.2f}")

if __name__=="__main__":
    main()
