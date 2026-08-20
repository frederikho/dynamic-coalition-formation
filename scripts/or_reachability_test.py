#!/usr/bin/env python3
"""
Is the planted support pattern inside the space OR's enumeration spans?

OR enumerates weak orders and DERIVES a full profile from each:

    acceptance   alpha = 1  iff tier_j[next] <  tier_j[cur]
                 alpha = 0  iff tier_j[next] >  tier_j[cur]
                 alpha free iff tier_j[next] == tier_j[cur]
    proposal     the approved targets in the proposer's BEST tier

So a support pattern is REACHABLE only if some weak order induces it.  Acceptance
constraints involve one player's own tiers, so they filter per player (541 each);
the proposal constraint couples players through the approval structure, so it is
checked on the surviving combinations.

If no weak order induces the planted pattern, the solution lies outside the space
OR searches, and no amount of enumeration or solver work reaches it.
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lib.equilibrium.ordinal_ranking.ranking_orders import _generate_weak_orders
from lib.equilibrium.jeres_vfi import Game, fw_state_name_to_partition
from lib.effectivity import get_effectivity

PL=["CHN","EUR","USA"]; ST=["( )","(CHNEUR)","(CHNUSA)","(EURUSA)","(CHNEURUSA)"]
EPS=1e-9
eff=get_effectivity("heyen_lehtomaa_2021",PL,ST)
CI=[[[tuple(sorted(k for k,pk in enumerate(PL) if eff.get((pi,x,y,pk),0)==1))
      for y in ST] for x in ST] for pi in PL]
WO=[np.asarray(t) for t in _generate_weak_orders(5)]

def load(f):
    d=json.loads(f.read_text()); u=np.array(d["payoffs"],float)
    g0=Game.from_payoffs(PL,u)
    jm={n:g0.state_idx[fw_state_name_to_partition(n,PL)] for n in ST}
    sig=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["sigmas"]]
    alp=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["alphas"]]
    # framework-indexed views
    A={}      # (approver, cur, nxt) -> planted alpha
    for ci_,cn in enumerate(ST):
        for aj in range(3):
            for ni,nn in enumerate(ST):
                v=alp[jm[cn]].get((aj,jm[nn]))
                if v is not None: A[(aj,ci_,ni)]=float(v)
    S={}      # (proposer, cur) -> planted target support
    for ci_,cn in enumerate(ST):
        for p in range(3):
            sup={ni for ni,nn in enumerate(ST) if sig[jm[cn]].get((p,jm[nn]),0.0)>EPS}
            S[(p,ci_)]=sup
    return d,A,S

def acc_ok(tier, j, A):
    for (aj,c,n),v in A.items():
        if aj!=j or c==n: continue
        if not any(j in CI[p][c][n] for p in range(3)): continue
        d=int(tier[n])-int(tier[c])
        # OR FREES a tied acceptance, so at a tie any planted value is compatible;
        # only a strict tier difference pins it.  Requiring alpha=0 to mean
        # "strictly worse" is wrong: the reverse direction of a planted tie carries
        # alpha=0 while the states are tied, and that is exactly the free case.
        if v>=1-EPS and d>0:  return False   # derived 0, planted 1
        if v<=EPS   and d<0:  return False   # derived 1, planted 0
        if EPS<v<1-EPS and d!=0: return False # interior needs a genuine tie
    return True

def prop_ok(tiers, S):
    for p in range(3):
        for c in range(5):
            approved=[]
            for n in range(5):
                ok=True
                for aj in CI[p][c][n]:
                    if n!=c and int(tiers[aj][n])>int(tiers[aj][c]): ok=False
                if ok: approved.append(n)
            if not approved: approved=[c]
            best=min(int(tiers[p][n]) for n in approved)
            winners={n for n in approved if int(tiers[p][n])==best}
            if winners!=S[(p,c)]: return False
    return True

def main():
    PROF=REPO/"reports"/"planted_profiles_v3"
    print(f"{'table':<20}{'M':>2}{'per-player weak orders passing acceptance':>44}{'reachable':>11}")
    reach=0; tot=0
    for f in sorted(PROF.glob("*.json")):
        d,A,S=load(f)
        cands=[[t for t in WO if acc_ok(t,j,A)] for j in range(3)]
        counts="x".join(str(len(c)) for c in cands)
        ok=False
        if all(cands):
            n=1
            for c in cands: n*=len(c)
            if n<=2_000_000:
                for combo in itertools.product(*cands):
                    if prop_ok(combo,S): ok=True; break
            else:
                counts+=" (too many)"
        tot+=1; reach+=ok
        print(f"{d['file'][14:20]:<20}{d['M']:>2}{counts:>44}{('YES' if ok else 'NO'):>11}")
    print(f"\nplanted pattern reachable by SOME weak order: {reach}/{tot}")

if __name__=="__main__":
    main()
