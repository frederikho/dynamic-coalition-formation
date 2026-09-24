"""certify_equilibrium.py -- prove that an EXACT equilibrium lies next to a
reported mixed one (computer-assisted proof, Krawczyk interval test).

`check_equilibrium.py` proves pure profiles exact, but a mixed profile can only
be shown eps-consistent: mixing needs exact indifference, which a floating-
point number cannot hit.  This script closes that gap.  For a reported mixed
profile it proves:

    There is an exact SMPE with the same support -- the same pure decisions and
    the same set of randomisations -- whose mixing probabilities lie within
    distance rho of the reported ones (rho is printed for each delta).

How
---
An equilibrium with a given support solves a polynomial system in the unknowns
w = (V, z), where z are the mixing probabilities:

    (A.2)        (I - delta P(z)) V_i - (1 - delta) u_i = 0        every i
    indifference V_j(y) - V_j(x) = 0      for every vote that mixes
                 G_i(x,y) = G_i(x,y')     for proposals a proposer mixes over,
                 with G_i(x,y) = Psi(x,y,A) (V_i(y) - V_i(x)).

The Krawczyk operator  K(Z) = c - Y F(c) + (I - Y J(Z)) (Z - c)  is evaluated
over a box Z around the reported point c.  If K(Z) lies in the interior of Z,
the system has exactly one solution in Z (Krawczyk 1969; Moore 1977).  Then
every remaining condition -- pure votes on the right side of indifference,
unused proposals no better than used ones, probabilities inside (0,1) -- is
checked over the WHOLE box, so it holds at the exact solution too.

Rigour
------
  * All interval arithmetic uses exact rationals, so there is no rounding
    to control: interval enclosures are exact.
  * Payoffs, delta and rho are exact rationals, as in check_equilibrium.py,
    whose primitives (committee rule, free exit) this builds on -- it does not
    use the solver.
  * NumPy is used only to choose the preconditioner Y and which variables to
    solve for.  Krawczyk's theorem holds for ANY Y, so a poor numerical choice
    can make the proof fail, never make it wrong.

Structure that must be handled exactly
--------------------------------------
Indifference is transitive: a player indifferent between x,y and y,z is
indifferent between x,z.  Mixing votes are therefore grouped into indifference
classes per player (union-find), and each class contributes a spanning set of
equations; every pairwise indifference inside a class then holds EXACTLY at the
root, not merely within the box.  This is what lets ties between used and
unused proposals be certified: a tie that follows from the equations is exact.

When there are more mixing probabilities than equations (a continuum of
equilibria), the surplus probabilities are fixed at their reported values and
the proof is for the equilibrium through those values.

    python certify_equilibrium.py example4.csv example4.json
    python certify_equilibrium.py example4.csv example4.json --delta 0.87 -v
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Fr
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .check_equilibrium import (committee, parse_state, read_payoffs,
                               check_point, verdict)


# ---------------------------------------------------------------------------
# exact interval arithmetic
# ---------------------------------------------------------------------------
class Iv:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = Fr(lo)
        self.hi = Fr(lo) if hi is None else Fr(hi)

    def __add__(self, o):
        o = _iv(o)
        return Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, o):
        o = _iv(o)
        return Iv(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o):
        return _iv(o) - self

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __mul__(self, o):
        o = _iv(o)
        if self.lo == self.hi and o.lo == o.hi:
            v = self.lo * o.lo
            return Iv(v, v)
        c = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Iv(min(c), max(c))

    __rmul__ = __mul__

    def is_zero(self):
        return self.lo == 0 and self.hi == 0

    def mid(self):
        return (self.lo + self.hi) / 2

    def __repr__(self):
        return f"[{float(self.lo):.17g}, {float(self.hi):.17g}]"


def _iv(x):
    return x if isinstance(x, Iv) else Iv(x)


ZERO = Iv(0)


# ---------------------------------------------------------------------------
# forward-mode automatic differentiation over intervals
# ---------------------------------------------------------------------------
class D:
    """Interval value with an interval gradient (sparse), for polynomials."""
    __slots__ = ("v", "g")

    def __init__(self, v, g=None):
        self.v = _iv(v)
        self.g = g if g is not None else {}

    def __add__(self, o):
        if not isinstance(o, D):
            return D(self.v + o, dict(self.g))
        g = dict(self.g)
        for k, x in o.g.items():
            g[k] = g[k] + x if k in g else x
        return D(self.v + o.v, g)

    __radd__ = __add__

    def __neg__(self):
        return D(-self.v, {k: -x for k, x in self.g.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, D) else -_iv(o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, D):
            o = _iv(o)
            if o.is_zero():
                return D(ZERO)
            return D(self.v * o, {k: x * o for k, x in self.g.items()})
        g = {k: x * o.v for k, x in self.g.items()}
        for k, x in o.g.items():
            t = x * self.v
            g[k] = g[k] + t if k in g else t
        return D(self.v * o.v, g)

    __rmul__ = __mul__


def const(x) -> D:
    return D(_iv(x))


# ---------------------------------------------------------------------------
# union-find for indifference classes
# ---------------------------------------------------------------------------
class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
            return True
        return False

    def same(self, a, b):
        return self.find(a) == self.find(b)


# ---------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------
def certify_point(players: Sequence[str], rows: Sequence[str],
                  u_rows: List[List[Fr]], delta: Fr, strategies: List[dict],
                  rho_p: Optional[Dict[str, Fr]] = None,
                  promote: bool = False, promote_rel: float = 1e-10,
                  verbose: bool = False) -> dict:
    """Certify one profile.

    promote=True: pure votes cast at a numerical indifference (|gain| below
    promote_rel x the voter's payoff spread) are hypothesised to be at EXACT
    indifference, and those equalities are added to the system.  Condition 2
    allows any vote at exact indifference, so a pure vote there is legal.  The
    hypothesis costs no rigour: if it is false the system has no nearby root
    and the Krawczyk test fails.
    """
    n = len(players)
    X = [parse_state(r) for r in rows]
    label = {x: r for x, r in zip(X, rows)}
    S = len(X)
    idx = {x: k for k, x in enumerate(X)}
    u = {x: dict(zip(players, u_rows[k])) for k, x in enumerate(X)}
    if rho_p is None:
        rho_p = {i: Fr(1, n) for i in players}

    # --- read the profile exactly, normalise proposal distributions --------
    p: Dict[Tuple, Fr] = {}
    r: Dict[Tuple, Fr] = {}
    for rec in strategies:
        key = (rec["player"], parse_state(rec["state"]), parse_state(rec["target"]))
        (p if rec["kind"] == "propose" else r)[key] = Fr(rec["prob"])
    for i in players:
        for x in X:
            tot = sum((p.get((i, x, y), Fr(0)) for y in X), Fr(0))
            for y in X:
                if (i, x, y) in p:
                    p[(i, x, y)] /= tot
    sup = {(i, x): [y for y in X if p.get((i, x, y), Fr(0)) > 0]
           for i in players for x in X}
    mix_votes = [k for k, v in r.items() if 0 < v < 1]
    mix_props = [(i, x) for (i, x), s in sup.items() if len(s) >= 2]
    if not mix_votes and not mix_props:
        return dict(status="PURE", reason="no randomisation: use "
                    "check_equilibrium.py, which proves pure profiles exact")

    def cm(x, y, i):
        return committee(x, y, i, players)

    missing = sorted({(j, label[x], label[y]) for i in players for x in X
                      for y in X for j in cm(x, y, i) if (j, x, y) not in r})
    if missing:
        return dict(status="NOT CERTIFIED",
                    reason=f"{len(missing)} required votes missing from the "
                    f"profile, e.g. {missing[:2]}")

    # float V at the reported profile, for the centre and the preconditioner
    res0 = check_point(players, rows, u_rows, delta, strategies, rho_p)
    V0 = {("V", i, x): Fr(float(res0["V"][label[x]][i]))
          for i in players for x in X}

    # --- indifference classes ----------------------------------------------
    uf = UF()
    for (j, x, y) in mix_votes:
        uf.union((j, x), (j, y))
    promoted: List[Tuple] = []
    if promote:
        for (j, x, y), v in r.items():
            if v not in (0, 1) or uf.same((j, x), (j, y)):
                continue
            spread = max(u[z][j] for z in X) - min(u[z][j] for z in X)
            gain = V0[("V", j, y)] - V0[("V", j, x)]
            if abs(gain) <= Fr(promote_rel) * spread:
                uf.union((j, x), (j, y))
                promoted.append((j, label[x], label[y]))

    def rpure(j, x, y):
        v = r.get((j, x, y))
        return v if v in (0, 1) else None

    def psi_struct_zero(x, y, i):
        return any(rpure(j, x, y) == 0 for j in cm(x, y, i))

    def g_struct_zero(i, x, y):
        return y == x or psi_struct_zero(x, y, i) or uf.same((i, x), (i, y))

    # A proposer mixing between a structurally-zero proposal and y must be
    # indifferent: G(y) = 0 with Psi(y) > 0 means V_i(y) = V_i(x), a class
    # relation.  Iterate, since new unions change what is structural.
    prop_eq_pairs: List[Tuple] = []
    changed = True
    while changed:
        changed = False
        for (i, x) in mix_props:
            s = sup[(i, x)]
            if any(g_struct_zero(i, x, y) for y in s):
                for y in s:
                    if not g_struct_zero(i, x, y):
                        if uf.union((i, x), (i, y)):
                            changed = True
    for (i, x) in mix_props:
        s = sup[(i, x)]
        if not any(g_struct_zero(i, x, y) for y in s):
            for y in s[1:]:
                prop_eq_pairs.append((i, x, s[0], y))

    # --- unknowns ------------------------------------------------------------
    var_names: List[Tuple] = []
    for i in players:
        for x in X:
            var_names.append(("V", i, x))
    z_names: List[Tuple] = []
    z0: Dict[Tuple, Fr] = {}
    for (j, x, y) in mix_votes:
        z_names.append(("r", j, x, y))
        z0[("r", j, x, y)] = r[(j, x, y)]
    for (i, x) in mix_props:
        for y in sup[(i, x)][:-1]:          # last one = 1 - sum of the others
            z_names.append(("p", i, x, y))
            z0[("p", i, x, y)] = p[(i, x, y)]


    # --- the system, as a function of which z are free ------------------------
    def build(free_z: Sequence[Tuple], box: Optional[Dict[Tuple, Iv]]):
        """Return (equations as D, name->index).  With box=None, variables are
        evaluated at the centre; otherwise over the given intervals."""
        names = var_names + list(free_z)
        index = {nm: k for k, nm in enumerate(names)}

        def var(nm):
            if nm in index:
                k = index[nm]
                val = box[nm] if box is not None else Iv(center[nm])
                return D(val, {k: Iv(1)})
            return const(z0[nm])                    # fixed surplus variable

        def rexp(j, x, y):
            if 0 < r[(j, x, y)] < 1:
                return var(("r", j, x, y))
            return const(r[(j, x, y)])

        def psi(x, y, i):
            out = const(1)
            for j in cm(x, y, i):
                out = out * rexp(j, x, y)
            return out

        def pexp(i, x, y):
            s = sup[(i, x)]
            if y not in s:
                return None
            if len(s) == 1:
                return const(1)
            if y != s[-1]:
                return var(("p", i, x, y))
            out = const(1)
            for yy in s[:-1]:
                out = out - var(("p", i, x, yy))
            return out

        Vv = {(i, x): var(("V", i, x)) for i in players for x in X}
        eqs: List[D] = []
        # (A.2)
        P = {}
        for x in X:
            off = const(0)
            for y in X:
                if y == x:
                    continue
                acc = const(0)
                for i in players:
                    pe = pexp(i, x, y)
                    if pe is None:
                        continue
                    acc = acc + pe * psi(x, y, i) * rho_p[i]
                P[(x, y)] = acc
                off = off + acc
            P[(x, x)] = const(1) - off
        for i in players:
            for x in X:
                e = Vv[(i, x)] - (1 - delta) * u[x][i]
                for y in X:
                    e = e - P[(x, y)] * Vv[(i, y)] * delta
                eqs.append(e)
        # indifference classes: spanning equations
        classes: Dict[Tuple, List] = {}
        for i in players:
            for x in X:
                classes.setdefault(uf.find((i, x)), []).append((i, x))
        for members in classes.values():
            if len(members) < 2:
                continue
            (i0, x0) = members[0]
            for (i1, x1) in members[1:]:
                eqs.append(Vv[(i1, x1)] - Vv[(i0, x0)])
        # proposer indifference where nothing is structurally zero
        def G(i, x, y):
            return psi(x, y, i) * (Vv[(i, y)] - Vv[(i, x)])
        for (i, x, y1, y2) in prop_eq_pairs:
            eqs.append(G(i, x, y1) - G(i, x, y2))
        return eqs, index, Vv, rexp, pexp, psi

    center: Dict[Tuple, Fr] = dict(V0)
    center.update(z0)

    # count equations, choose which z to solve for (square, well conditioned)
    eqs_all, index_all, *_ = build(z_names, None)
    n_eq = len(eqs_all)
    need = n_eq - len(var_names)
    if need > len(z_names):
        return dict(status="NOT CERTIFIED",
                    reason=f"{need} indifference equations but only "
                    f"{len(z_names)} mixing probabilities (overdetermined)")
    Jf = np.zeros((n_eq, len(var_names) + len(z_names)))
    for a, e in enumerate(eqs_all):
        for k, gv in e.g.items():
            Jf[a, k] = float(gv.mid())
    chosen: List[int] = []
    base = list(range(len(var_names)))
    for _ in range(need):
        best, best_s = None, -1.0
        for c in range(len(var_names), len(var_names) + len(z_names)):
            if c in chosen:
                continue
            cols = base + chosen + [c]
            sv = np.linalg.svd(Jf[:, cols], compute_uv=False)
            s_min = sv[min(len(cols), n_eq) - 1]
            if s_min > best_s:
                best, best_s = c, s_min
        chosen.append(best)
    free_z = [z_names[c - len(var_names)] for c in chosen]
    fixed_z = [nm for nm in z_names if nm not in free_z]

    # --- Krawczyk --------------------------------------------------------------
    eqs_c, index, *_ = build(free_z, None)
    m = len(eqs_c)
    Fc = [e.v.lo for e in eqs_c]                     # exact at the centre
    Jc = np.zeros((m, m))
    for a, e in enumerate(eqs_c):
        for k, gv in e.g.items():
            Jc[a, k] = float(gv.mid())
    if abs(np.linalg.det(Jc)) == 0 or np.linalg.cond(Jc) > 1e14:
        return dict(status="NOT CERTIFIED",
                    reason=f"Jacobian singular (cond {np.linalg.cond(Jc):.1e})")
    Yf = np.linalg.inv(Jc)
    Y = [[Fr(float(Yf[a, b])) for b in range(m)] for a in range(m)]
    names = var_names + free_z
    cvec = [center[nm] for nm in names]
    YF = [sum((Y[a][b] * Fc[b] for b in range(m)), Fr(0)) for a in range(m)]
    base_rad = max(abs(v) for v in YF)
    # start just above the Newton correction and widen only if needed: the
    # smaller the certified box, the stronger the statement
    rad = max(base_rad * 4, Fr(1, 10**30))

    for attempt in range(14):
        box = {nm: Iv(cvec[k] - rad, cvec[k] + rad) for k, nm in enumerate(names)}
        eqs_b, _ix, Vv, rexp, pexp, psi = build(free_z, box)
        # (I - Y J(Z)) (Z - c): each component of Z - c is [-rad, rad]
        JZ = [[eqs_b[a].g.get(b, ZERO) for b in range(m)] for a in range(m)]
        ok = True
        for a in range(m):
            acc_lo = Fr(0)
            acc_hi = Fr(0)
            for b in range(m):
                t = Iv(1 if a == b else 0)
                for k in range(m):
                    if Y[a][k] != 0:
                        t = t - JZ[k][b] * Y[a][k]
                mag = max(abs(t.lo), abs(t.hi))
                acc_hi += mag * rad
            acc_lo = -acc_hi
            k_lo = cvec[a] - YF[a] + acc_lo
            k_hi = cvec[a] - YF[a] + acc_hi
            if not (cvec[a] - rad < k_lo and k_hi < cvec[a] + rad):
                ok = False
                break
        if ok:
            break
        rad *= 8
    if not ok:
        return dict(status="NOT CERTIFIED",
                    reason="Krawczyk containment failed for every box tried")

    # --- every other condition, over the whole certified box -------------------
    fails: List[str] = []
    # probabilities strictly inside (0,1)
    for nm in free_z:
        b = box[nm]
        if not (b.lo > 0 and b.hi < 1):
            fails.append(f"{nm} not inside (0,1) over the box")
    for (i, x) in mix_props:
        last = pexp(i, x, sup[(i, x)][-1]).v
        if not (last.lo > 0 and last.hi < 1):
            fails.append(f"p_{i}({label[x]}) last component not inside (0,1)")
    for nm in fixed_z:
        if not (0 < z0[nm] < 1):
            fails.append(f"fixed {nm} not inside (0,1)")
    # Condition 2: pure votes; mixing votes are exact by class construction
    for (j, x, y), v in r.items():
        if uf.same((j, x), (j, y)):
            continue                          # exactly indifferent at the root
        if 0 < v < 1:
            fails.append(f"mixing vote {j} {label[x]}->{label[y]} not in a class")
            continue
        gain = Vv[(j, y)].v - Vv[(j, x)].v
        if v == 1 and not gain.lo > 0:
            fails.append(f"{j} accepts {label[x]}->{label[y]} but gain {gain}")
        if v == 0 and not gain.hi < 0:
            fails.append(f"{j} rejects {label[x]}->{label[y]} but gain {gain}")
    # Condition 1: used proposals are optimal.  Differences are formed
    # symbolically before intervals are taken: two proposals leading to states
    # in the same indifference class of the proposer have identical V, so
    # G(y) - G(k) = (Psi_y - Psi_k)(V_i(y) - V_i(x)), and an exact tie
    # (Psi_y = Psi_k) is recognised as exact instead of being lost to the
    # dependency problem of interval arithmetic.
    def gdiff(i, x, y, k):
        """(exactly_zero, interval) for G_i(x,y) - G_i(x,k) at the root."""
        zy, zk = g_struct_zero(i, x, y), g_struct_zero(i, x, k)
        if zy and zk:
            return True, ZERO
        if not zy and not zk and uf.same((i, y), (i, k)):
            dpsi = psi(x, y, i) - psi(x, k, i)
            if not dpsi.g and dpsi.v.is_zero():
                return True, ZERO
            return False, (dpsi * (Vv[(i, y)] - Vv[(i, x)])).v
        gy = ZERO if zy else (psi(x, y, i) * (Vv[(i, y)] - Vv[(i, x)])).v
        gk = ZERO if zk else (psi(x, k, i) * (Vv[(i, k)] - Vv[(i, x)])).v
        return False, gy - gk

    for i in players:
        for x in X:
            s = sup[(i, x)]
            y_ref = s[0]
            for k in X:
                if k in s:
                    continue
                exact_zero, diff = gdiff(i, x, y_ref, k)
                if exact_zero:
                    continue                  # exact tie: k is also optimal
                if not diff.lo > 0:
                    fails.append(f"{i} at {label[x]}: unused {label[k]} not "
                                 f"certified worse than used {label[y_ref]} "
                                 f"(G difference {diff})")
    if fails:
        return dict(status="NOT CERTIFIED", reason="; ".join(fails[:3])
                    + (f" (+{len(fails)-3} more)" if len(fails) > 3 else ""),
                    radius=rad)
    def refine(iters: int = 6, bits: int = 220) -> List[dict]:
        """Newton's method in exact rationals on the certified system, from
        the reported point; returns strategy records with the refined mixing
        probabilities as exact Fractions.  Used to cross-check the certificate
        with the independent checker -- not needed for the proof itself."""
        from .check_equilibrium import solve_exact
        grid = Fr(1, 2 ** bits)
        for _ in range(iters):
            eqs_n, ix_n, *_r = build(free_z, None)
            Fn = [e.v.lo for e in eqs_n]
            Jn = [[e.g.get(b, ZERO).lo for b in range(m)] for e in eqs_n]
            step = solve_exact(Jn, [-f for f in Fn])
            for k, nm in enumerate(names):
                v = center[nm] + step[k]
                center[nm] = Fr(round(v / grid)) * grid
        zval = dict(z0)
        for nm in free_z:
            zval[nm] = center[nm]
        out = []
        lastp = {}
        for (i, x) in mix_props:
            ss = sup[(i, x)]
            lastp[(i, x, ss[-1])] = 1 - sum(zval[("p", i, x, y)] for y in ss[:-1])
        for rec in strategies:
            rec2 = dict(rec)
            x, y = parse_state(rec["state"]), parse_state(rec["target"])
            if rec["kind"] == "accept" and ("r", rec["player"], x, y) in zval:
                rec2["prob"] = zval[("r", rec["player"], x, y)]
            elif rec["kind"] == "propose":
                key = ("p", rec["player"], x, y)
                if key in zval:
                    rec2["prob"] = zval[key]
                elif (rec["player"], x, y) in lastp:
                    rec2["prob"] = lastp[(rec["player"], x, y)]
                else:
                    rec2["prob"] = p[(rec["player"], x, y)]   # normalised, exact
            out.append(rec2)
        return out, {nm: center[nm] for nm in names}, box

    return dict(status="CERTIFIED", radius=rad, n_unknowns=m,
                n_mixing=len(z_names), n_fixed=len(fixed_z),
                fixed=[_nm(nm, label) for nm in fixed_z], promoted=promoted,
                refine=refine)


def _nm(nm, label):
    if nm[0] == "r":
        return f"r_{nm[1]}({label[nm[2]]} -> {label[nm[3]]})"
    return f"p_{nm[1]}({label[nm[2]]} -> {label[nm[3]]})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("payoffs")
    ap.add_argument("profile")
    ap.add_argument("--delta", type=float)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    players, rows, u = read_payoffs(a.payoffs)
    prof = json.load(open(a.profile))
    print(f"{'delta':>6s}  {'result':<15s} {'radius':>9s}  detail")
    tally = {}
    for pt in prof["points"]:
        if not pt.get("solved"):
            continue
        if a.delta is not None and abs(pt["delta"] - a.delta) > 1e-12:
            continue
        d = Fr(repr(pt["delta"]))
        res = certify_point(players, rows, u, d, pt["strategies"])
        if res["status"] == "PURE":
            v = verdict(check_point(players, rows, u, d, pt["strategies"]))
            res = dict(status="EXACT (pure)" if v == "EXACT EQUILIBRIUM"
                       else "FAIL (pure)")
        if res["status"] == "NOT CERTIFIED":
            res2 = certify_point(players, rows, u, d, pt["strategies"],
                                 promote=True)
            if res2["status"] == "CERTIFIED" or not res2.get("reason"):
                res = res2
            else:
                res["reason"] += f" | with exact-indifference hypothesis: {res2['reason']}"
        st = res["status"]
        tally[st] = tally.get(st, 0) + 1
        rad = f"{float(res['radius']):.1e}" if "radius" in res else "-"
        det = ""
        if st == "CERTIFIED":
            det = (f"{res['n_mixing']} mixing probabilities"
                   + (f", {res['n_fixed']} fixed (continuum)" if res["n_fixed"]
                      else ""))
            if res.get("promoted"):
                det += (f"; {len(res['promoted'])} pure vote(s) proved to be "
                        "cast at exact indifference")
        elif "reason" in res:
            det = res["reason"]
        print(f"{float(d):6.3f}  {st:<15s} {rad:>9s}  {det}")
        if a.verbose and res.get("fixed"):
            print("        fixed at reported values: " + ", ".join(res["fixed"]))
        if a.verbose and res.get("promoted"):
            print("        exactly indifferent, voting pure: " + ", ".join(
                f"{j} on {x} vs {y}" for j, x, y in res["promoted"]))
    print("\n" + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
    print("CERTIFIED: proved that an exact equilibrium with the same support "
          "lies within 'radius'\nof the reported mixing probabilities and "
          "values.  EXACT (pure): proved exact by\ncheck_equilibrium.py.")


if __name__ == "__main__":
    main()
