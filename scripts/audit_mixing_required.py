#!/usr/bin/env python3
"""
Which synthetic tables genuinely REQUIRE mixing?

The pure-free guard used ordinal_ranking, which represents behaviour as a strict
value ranking.  At an exact tie a pure profile can do things no strict order
rationalises (reject a move in both directions; reject a move yet decline to
propose its reverse), so OR cannot express those profiles and wrongly reported
"no pure equilibrium".  Four of ten M=1 tables were mis-certified that way.

This test needs no ranking representation at all: push every knob to a bound and
check all 2^M pure variants.  If any verifies, the table has a pure equilibrium
and does not require mixing.
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import importlib.util
sp = importlib.util.spec_from_file_location("enu", REPO/"scripts/enumerate_knobs_m1.py")
enu = importlib.util.module_from_spec(sp); sp.loader.exec_module(enu)
from lib.equilibrium.jeres_vfi import Game, fw_state_name_to_partition
from lib.equilibrium.jeres_vfi.solver import (compute_values, full_transition_matrix,
    verify_responses, verify_proposals)
from lib.effectivity import get_effectivity

PL=["CHN","EUR","USA"]; ST=["( )","(CHNEUR)","(CHNUSA)","(EURUSA)","(CHNEURUSA)"]
EPS=1e-12

def main():
    PROF=REPO/"reports"/"planted_profiles"
    eff=get_effectivity("heyen_lehtomaa_2021",PL,ST)
    rows=[]
    for f in sorted(PROF.glob("*.json")):
        d=json.loads(f.read_text()); u=np.array(d["payoffs"],float)
        g0=Game.from_payoffs(PL,u)
        j_of={n:g0.state_idx[fw_state_name_to_partition(n,PL)] for n in ST}
        ac={(i,j_of[x],j_of[y]):frozenset(k for k,pk in enumerate(PL)
            if eff.get((pi,x,y,pk),0)==1)
            for i,pi in enumerate(PL) for x in ST for y in ST}
        g=Game(players=PL,payoffs=u,states=g0.states,state_idx=g0.state_idx,
               n_players=3,n_states=5,approval_committees=ac)
        comm=enu.committees(g)
        sig=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["sigmas"]]
        alp=[{tuple(int(v) for v in k.split(",")):val for k,val in dd.items()} for dd in d["alphas"]]
        knobs=[tuple(k) for k in d["knobs"]]
        pure=None
        for bounds in itertools.product((0.0,1.0),repeat=len(knobs)):
            a2=[dict(x) for x in alp]
            for (kx,ky,kj),b in zip(knobs,bounds): a2[kx][(kj,ky)]=b
            qs=enu.qs_from_alphas(g,comm,a2)
            V=compute_values(g,full_transition_matrix(g,sig,qs,None),d["delta"])
            r,_=verify_responses(g,sig,a2,qs,V,atol=EPS)
            p,_=verify_proposals(g,sig,a2,qs,V,atol=EPS)
            if r and p: pure=bounds; break
        rows.append((d["file"],d["M"],pure))
    print(f"{'M':>2}{'tables':>8}{'requires mixing':>17}{'has a pure variant':>21}")
    for m in (1,2,3,4):
        sub=[r for r in rows if r[1]==m]
        good=[r for r in sub if r[2] is None]
        print(f"{m:>2}{len(sub):>8}{len(good):>17}{len(sub)-len(good):>21}")
    print(f"\n{'TOTAL':>2}{len(rows):>8}{sum(1 for r in rows if r[2] is None):>17}"
          f"{sum(1 for r in rows if r[2] is not None):>21}")
    bad=[r for r in rows if r[2] is not None]
    print("\ntables with a pure variant (NOT usable as mixing controls):")
    for f,m,b in bad: print(f"   {f:<38} M={m}  bounds={b}")
    keep=[r[0] for r in rows if r[2] is None]
    (REPO/"reports"/"mixing_required_tables.json").write_text(json.dumps(keep,indent=2))
    print(f"\nusable list -> reports/mixing_required_tables.json ({len(keep)} tables)")

if __name__=="__main__":
    main()
