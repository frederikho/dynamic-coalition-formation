#!/usr/bin/env python3
"""
Why does ordinal_ranking miss pure equilibria that demonstrably verify?

OR's induced profile is a deterministic function of a strict ranking:

    alpha_j(x->y) = 1  iff  rank_j[y] < rank_j[x]
    sigma_i(x)    = argmin rank_i over  {x} U {approvable targets}

So "can OR ever produce profile P*" is a constraint problem, not a search: P*
is reachable iff there EXISTS a strict ranking satisfying, for every player,

    acceptance:  alpha = 1  ->  rank[y] < rank[x]
                 alpha = 0  ->  rank[y] > rank[x]
    proposal:    rank[chosen] < rank[y]  for every other y in {x} U approvable

Each constraint is an edge in a precedence graph.  A strict ranking exists iff
that graph is acyclic.  A cycle is a proof that NO ranking induces P*, hence that
OR cannot find it however many rankings it enumerates.
"""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium import mixed_controls as enu  # noqa: E402
import importlib.util
from lib.equilibrium.jeres_vfi import Game, fw_state_name_to_partition
from lib.equilibrium.jeres_vfi.solver import (compute_values, full_transition_matrix,
    verify_responses, verify_proposals)
from lib.effectivity import get_effectivity

PL=["CHN","EUR","USA"]; ST=["( )","(CHNEUR)","(CHNUSA)","(EURUSA)","(CHNEURUSA)"]
EPS=1e-12

def framework_game(u_j, g0):
    eff=get_effectivity("heyen_lehtomaa_2021",PL,ST)
    j_of={n:g0.state_idx[fw_state_name_to_partition(n,PL)] for n in ST}
    ac={(i,j_of[x],j_of[y]):frozenset(k for k,pk in enumerate(PL)
        if eff.get((pi,x,y,pk),0)==1)
        for i,pi in enumerate(PL) for x in ST for y in ST}
    return Game(players=PL,payoffs=u_j,states=g0.states,state_idx=g0.state_idx,
                n_players=3,n_states=5,approval_committees=ac), ac, j_of

def cyclic(edges, n):
    """Is the precedence graph cyclic?  Returns a witness cycle or None."""
    adj={i:set() for i in range(n)}
    for a,b in edges: adj[a].add(b)
    colour={}; stack=[]
    def dfs(u):
        colour[u]=1; stack.append(u)
        for v in adj[u]:
            if colour.get(v)==1:
                return stack[stack.index(v):]+[v]
            if colour.get(v) is None:
                r=dfs(v)
                if r: return r
        colour[u]=2; stack.pop(); return None
    for u in range(n):
        if colour.get(u) is None:
            r=dfs(u)
            if r: return r
    return None

def main():
    PROF=REPO/"reports"/"planted_profiles"
    for name in ("m1_03","m1_04","m1_07","m1_08"):
        f=next(PROF.glob(f"*{name}*.json")); d=json.loads(f.read_text())
        u=np.array(d["payoffs"],float); g0=Game.from_payoffs(PL,u)
        g,ac,j_of=framework_game(u,g0); fw_of={v:i for i,(n,v) in enumerate(j_of.items())}
        sig=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["sigmas"]]
        alp=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["alphas"]]
        kx,ky,kj=d["knobs"][0]
        # the pure profile: knob at whichever bound verifies
        found=None
        for th in (0.0,1.0):
            a2=[dict(x) for x in alp]; a2[kx][(kj,ky)]=th
            comm=enu.committees(g); qs=enu.qs_from_alphas(g,comm,a2)
            V=compute_values(g,full_transition_matrix(g,sig,qs,None),d["delta"])
            r,_=verify_responses(g,sig,a2,qs,V,atol=EPS)
            p,_=verify_proposals(g,sig,a2,qs,V,atol=EPS)
            if r and p: found=(th,a2,qs,V); break
        if not found:
            print(f"{name}: no verifying pure profile"); continue
        th,a2,qs,V=found
        print(f"\n=== {d['file']}   pure profile at knob={th}")
        for p in range(3):
            edges=[]; why={}
            for x in range(5):
                for y in range(5):
                    if x==y: continue
                    if any(p in ac.get((i,x,y),()) for i in range(3)):
                        av=a2[x].get((p,y))
                        if av is None: continue
                        if av>=1-EPS: edges.append((y,x)); why[(y,x)]=f"accept {ST[fw_of[x]]}->{ST[fw_of[y]]}"
                        elif av<=EPS:  edges.append((x,y)); why[(x,y)]=f"reject {ST[fw_of[x]]}->{ST[fw_of[y]]}"
                chosen=next((y for y in range(5) if sig[x].get((p,y),0.0)>0.5), x)
                appr=[y for y in range(5) if y!=x and qs[x].get((p,y),0.0)>1-EPS]
                for y in set(appr+[x]):
                    if y!=chosen:
                        edges.append((chosen,y))
                        why[(chosen,y)]=f"{PL[p]} in {ST[fw_of[x]]} picks {ST[fw_of[chosen]]} over {ST[fw_of[y]]}"
            cyc=cyclic(edges,5)
            if cyc:
                print(f"  player {PL[p]}: NO strict ranking can induce this profile")
                for a,b in zip(cyc,cyc[1:]):
                    print(f"      needs rank[{ST[fw_of[a]]}] < rank[{ST[fw_of[b]]}]   ({why.get((a,b),'')})")
            else:
                print(f"  player {PL[p]}: constraints satisfiable")
if __name__=="__main__":
    main()
