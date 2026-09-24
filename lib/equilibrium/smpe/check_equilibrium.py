"""check_equilibrium.py -- independent, exact check of the equilibrium conditions.

This is a second implementation of the equilibrium conditions, written to mirror
the paper's appendix line by line and sharing NO code with the solver.  It reads
a strategy profile from the solver's exported JSON and the payoff matrix from
its CSV, and checks, in exact rational arithmetic with no tolerances:

  (A.2)  V_i(x) = (1-delta) u_i(x) + delta * sum_y P(x,y) V_i(y),
         with P built from p, r and the protocol rho, and solved exactly;

  Condition 1 (proposer consistency)
         p_i(x,y,A) > 0  only if  y is in F_A(x)  and
         (y,A) in argmax_{(k,Q): k in F_Q(x)} Psi(x,k,Q) V_i(k) + (1-Psi) V_i(x);

  Condition 2 (responder consistency)
         r_i(x,y,A) = 1 if V_i(y) > V_i(x),  = 0 if V_i(y) < V_i(x),
         anything in [0,1] only if V_i(y) = V_i(x).

Why a separate implementation
-----------------------------
The solver's own `verify` shares code with the solver and uses a tolerance
(1e-7).  An error in the shared code would be invisible to it, and a voter
mixing while 5e-8 from indifference would pass it.  This checker re-derives
the committee rule, free exit, Psi and P from their definitions, and does all
arithmetic exactly, so a pass here does not rest on either of those.

What a pass proves
------------------
Inputs are taken exactly: payoffs as the decimals written in the CSV, delta as
its decimal (0.87 = 87/100), and each probability as the exact value of the
binary float the solver produced.  Then:

  * PURE profiles (all p, r in {0,1}): if every violation is exactly zero, the
    profile IS an exact equilibrium of the stated game.  This is a proof,
    not a numerical approximation.

  * MIXED profiles: mixing requires exact indifference, V_i(y) = V_i(x), which a
    floating-point solution can only approximate.  The checker reports the
    exact largest violation eps.  The profile is then an exact
    eps-consistent profile: every condition holds up to eps, measured in the
    same units as u.  Report eps alongside the result.

Model conventions (must match the paper's definitions of F_A(x) and Q)
-----------------------------------------------------------------------
  * The approval committee is determined by the proposal: for proposer i
    moving x -> y, A = { j != i : j's coalition differs between x and y },
    except that i's own unilateral exit (i leaves, the rest of i's coalition
    stays together) needs no approval, A = {}.  The proposer does not choose A.
  * Every state is feasible to propose, including the status quo y = x.
  * Psi(x,y,A) = prod_{j in A} r_j(x,y): unanimity, independent votes.
  * r_j(x,y) does not depend on who proposed (the solver's model).

    python check_equilibrium.py example4.csv example4.json
    python check_equilibrium.py example4.csv example4.json --delta 0.87 --detail
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction as Fr
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

Partition = FrozenSet[FrozenSet[str]]


# ---------------------------------------------------------------------------
# coalition structures, from first principles
# ---------------------------------------------------------------------------
def parse_state(label: str) -> Partition:
    """'{A,C}+{B}' -> frozenset({frozenset({'A','C'}), frozenset({'B'})})"""
    blocks = []
    for part in label.split("+"):
        names = [x.strip() for x in part.strip().strip("{}").split(",") if x.strip()]
        blocks.append(frozenset(names))
    return frozenset(blocks)


def block_of(x: Partition, i: str) -> FrozenSet[str]:
    for b in x:
        if i in b:
            return b
    raise ValueError(f"{i} not in {x}")


def unilateral_exit(x: Partition, i: str) -> Partition:
    """The state reached when i leaves its coalition, the rest staying together."""
    b = block_of(x, i)
    if len(b) == 1:
        return x
    rest = b - {i}
    return frozenset((x - {b}) | {rest, frozenset({i})})


def committee(x: Partition, y: Partition, i: str, players: Sequence[str]) -> FrozenSet[str]:
    """Approval committee for proposer i moving x -> y."""
    if y != x and y == unilateral_exit(x, i):
        return frozenset()
    return frozenset(j for j in players
                     if j != i and block_of(x, j) != block_of(y, j))


# ---------------------------------------------------------------------------
# exact linear algebra
# ---------------------------------------------------------------------------
def solve_exact(M: List[List[Fr]], b: List[Fr]) -> List[Fr]:
    """Gaussian elimination with exact rationals.  M is square, nonsingular."""
    n = len(M)
    A = [row[:] + [b[k]] for k, row in enumerate(M)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * p for a, p in zip(A[r], A[c])]
    return [A[k][n] / A[k][k] for k in range(n)]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def read_payoffs(path: str) -> Tuple[List[str], List[str], List[List[Fr]]]:
    """Payoff CSV with '# players:' and '# rows:' headers; values read exactly
    as the decimals written (98.222 -> 98222/1000), not via binary floats."""
    players = rows = None
    data: List[List[Fr]] = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.lower().startswith("players:"):
                players = [x.strip() for x in body.split(":", 1)[1].split(",")]
            elif body.lower().startswith("rows:"):
                rows = [x.strip() for x in body.split(":", 1)[1].split("|")]
            continue
        data.append([Fr(tok.strip()) for tok in line.split(",")])
    if players is None or rows is None:
        raise ValueError("payoff CSV needs '# players:' and '# rows:' headers")
    return players, rows, data


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def check_point(players: Sequence[str], rows: Sequence[str],
                u_rows: List[List[Fr]], delta: Fr,
                strategies: List[dict], rho: Optional[Dict[str, Fr]] = None
                ) -> dict:
    """Check Conditions 1, 2 and (A.2) exactly for one delta."""
    n = len(players)
    X = [parse_state(r) for r in rows]
    label = {x: r for x, r in zip(X, rows)}
    idx = {x: k for k, x in enumerate(X)}
    S = len(X)
    u = {x: dict(zip(players, u_rows[k])) for k, x in enumerate(X)}
    if rho is None:
        rho = {i: Fr(1, n) for i in players}

    # p_i(x, y) and r_j(x, y), read exactly from the exported floats
    p: Dict[Tuple[str, Partition, Partition], Fr] = {}
    r: Dict[Tuple[str, Partition, Partition], Fr] = {}
    for rec in strategies:
        x, y = parse_state(rec["state"]), parse_state(rec["target"])
        if x not in idx or y not in idx:
            raise ValueError(f"unknown state in record {rec}")
        key = (rec["player"], x, y)
        (p if rec["kind"] == "propose" else r)[key] = Fr(rec["prob"])

    problems: List[str] = []

    # --- well-formedness -------------------------------------------------
    # Binary floats that are meant to sum to 1 usually miss by ~1e-16.  Each
    # proposal distribution is divided by its exact sum, so the profile that is
    # checked is an exact probability distribution with the same support; the
    # size of that correction is reported.
    worst_sum = Fr(0)
    for i in players:
        for x in X:
            tot = sum((p.get((i, x, y), Fr(0)) for y in X), Fr(0))
            worst_sum = max(worst_sum, abs(tot - 1))
            if tot <= 0:
                problems.append(f"p_{i}({label[x]}, .) is empty")
                continue
            for y in X:
                if (i, x, y) in p:
                    p[(i, x, y)] = p[(i, x, y)] / tot
            for y in X:
                v = p.get((i, x, y), Fr(0))
                if v < 0 or v > 1:
                    problems.append(f"p_{i}({label[x]},{label[y]}) = {float(v)} "
                                    "outside [0,1]")
    for key, v in r.items():
        if v < 0 or v > 1:
            problems.append(f"r out of [0,1]: {key}")

    # every vote that some committee could require must be specified
    missing = []
    for i in players:
        for x in X:
            for y in X:
                for j in committee(x, y, i, players):
                    if (j, x, y) not in r:
                        missing.append((j, label[x], label[y]))
    if missing:
        problems.append(f"{len(set(missing))} required votes missing from the "
                        f"profile, e.g. {sorted(set(missing))[:3]}")

    def Psi(x, y, Q) -> Fr:
        out = Fr(1)
        for j in Q:
            out *= r.get((j, x, y), Fr(0))
        return out

    # --- P and (A.2) -------------------------------------------------------
    P = [[Fr(0)] * S for _ in range(S)]
    for a, x in enumerate(X):
        off = Fr(0)
        for b, y in enumerate(X):
            if y == x:
                continue
            pr = sum((rho[i] * p.get((i, x, y), Fr(0))
                      * Psi(x, y, committee(x, y, i, players)) for i in players),
                     Fr(0))
            P[a][b] = pr
            off += pr
        P[a][a] = 1 - off       # proposals to stay, and every rejection
        if P[a][a] < 0:
            problems.append(f"P row {label[x]} has negative diagonal")
    V: Dict[str, List[Fr]] = {}
    M = [[(Fr(1) if a == b else Fr(0)) - delta * P[a][b] for b in range(S)]
         for a in range(S)]
    for i in players:
        V[i] = solve_exact(M, [(1 - delta) * u[x][i] for x in X])

    def Vi(i, x):
        return V[i][idx[x]]

    # (A.2) residual: exactly zero by construction, re-checked anyway
    a2 = max(abs(Vi(i, x) - ((1 - delta) * u[x][i]
                             + delta * sum(P[idx[x]][idx[y]] * Vi(i, y) for y in X)))
             for i in players for x in X)

    # --- Condition 1: proposer consistency ---------------------------------
    c1_worst = Fr(0)
    c1_where = None
    c1_margin = None       # smallest gap between the best proposal and a worse one
    for i in players:
        for x in X:
            val = {}
            for k in X:
                Q = committee(x, k, i, players)
                ps = Psi(x, k, Q)
                val[k] = ps * Vi(i, k) + (1 - ps) * Vi(i, x)
            best = max(val.values())
            for y in X:
                if p.get((i, x, y), Fr(0)) > 0:
                    reg = best - val[y]
                    if reg > c1_worst:
                        c1_worst, c1_where = reg, (i, label[x], label[y],
                                                   float(p[(i, x, y)]))
            # robustness: how far the proposals NOT made are from optimal
            unused = [best - val[k] for k in X
                      if p.get((i, x, k), Fr(0)) == 0 and val[k] < best]
            if unused:
                m = min(unused)
                c1_margin = m if c1_margin is None else min(c1_margin, m)

    # --- Condition 2: responder consistency --------------------------------
    c2_worst = Fr(0)
    c2_where = None
    c2_margin = None       # smallest |V(y)-V(x)| at a vote that is pure
    n_mixing = 0
    mix_resid = Fr(0)
    for (j, x, y), rv in r.items():
        g = Vi(j, y) - Vi(j, x)
        if rv == 1:
            v = max(Fr(0), -g)
        elif rv == 0:
            v = max(Fr(0), g)
        else:
            v = abs(g)                    # mixing requires exact indifference
            n_mixing += 1
            mix_resid = max(mix_resid, abs(g))
        if v > c2_worst:
            c2_worst, c2_where = v, (j, label[x], label[y], float(rv), float(g))
        if rv in (0, 1) and g != 0:
            c2_margin = abs(g) if c2_margin is None else min(c2_margin, abs(g))

    eps = max(c1_worst, c2_worst)
    pure = all(v in (0, 1) for v in list(p.values()) + list(r.values()))
    exact = (eps == 0 and not problems and a2 == 0)
    # pure votes cast at (numerically) an indifference: legal under Condition 2
    # -- any r is allowed when V(y) = V(x) -- but their sign is not resolved
    # beyond eps, so they are listed rather than silently counted as strict
    near_indiff = []
    if eps > 0:
        for (j, x, y), rv in r.items():
            if rv in (0, 1):
                g = Vi(j, y) - Vi(j, x)
                if g != 0 and abs(g) <= eps:
                    near_indiff.append((j, label[x], label[y], float(rv), float(g)))
    spread = max(max(u[x][i] for x in X) - min(u[x][i] for x in X)
                 for i in players)
    return dict(
        delta=delta, pure=pure, exact=exact, problems=problems,
        eps=eps, eps_relative=(eps / spread if spread else eps),
        condition1_worst=c1_worst, condition1_where=c1_where,
        condition2_worst=c2_worst, condition2_where=c2_where,
        n_mixing_votes=n_mixing, mixing_indifference_residual=mix_resid,
        proposal_sum_error=worst_sum, a2_residual=a2,
        strictness_condition1=c1_margin, strictness_condition2=c2_margin,
        near_indifferent_pure_votes=near_indiff,
        V={label[x]: {i: Vi(i, x) for i in players} for x in X},
    )


EPS_MAX_RELATIVE = Fr(1, 10**9)


def verdict(res: dict, eps_max_relative: Fr = EPS_MAX_RELATIVE) -> str:
    """EXACT: every condition holds exactly.  eps-EQUILIBRIUM: a mixed profile
    whose largest violation, relative to the largest payoff spread, is at most
    eps_max_relative (default 1e-9).  Anything else FAILS -- including a pure
    profile with any violation at all, since it has no rounding to excuse it."""
    if res["problems"]:
        return "FAIL"
    if res["exact"]:
        return "EXACT EQUILIBRIUM"
    if res["pure"]:
        return "FAIL"
    if res["eps_relative"] <= eps_max_relative:
        return "eps-EQUILIBRIUM"
    return "FAIL"


def fmt(x) -> str:
    return "0" if x == 0 else f"{float(x):.2e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("payoffs", help="payoff CSV with '# players:' and '# rows:'")
    ap.add_argument("profile", help="JSON exported by smpe_sweep (res.to_json)")
    ap.add_argument("--delta", type=float, help="check only this delta")
    ap.add_argument("--detail", action="store_true",
                    help="print the worst case of each condition")
    ap.add_argument("--eps-max", type=float, default=1e-9,
                    help="largest violation accepted for a MIXED profile, "
                    "relative to the largest payoff spread (default 1e-9)")
    a = ap.parse_args()
    players, rows, u = read_payoffs(a.payoffs)
    prof = json.load(open(a.profile))
    if list(prof["players"]) != list(players):
        raise SystemExit(f"player mismatch: CSV {players} vs JSON {prof['players']}")
    print(f"{'delta':>6s}  {'verdict':<18s} {'eps':>9s} {'cond 1':>9s} "
          f"{'cond 2':>9s} {'mixing':>6s}  {'slack 1':>9s} {'slack 2':>9s}")
    n_ok = 0
    n = 0
    worst_eps = Fr(0)
    for pt in prof["points"]:
        if not pt.get("solved"):
            continue
        if a.delta is not None and abs(pt["delta"] - a.delta) > 1e-12:
            continue
        n += 1
        d = Fr(repr(pt["delta"]))          # 0.87 -> 87/100 exactly
        res = check_point(players, rows, u, d, pt["strategies"])
        v = verdict(res, Fr(repr(a.eps_max)))
        n_ok += v != "FAIL"
        worst_eps = max(worst_eps, res["eps"])
        print(f"{float(d):6.3f}  {v:<18s} {fmt(res['eps']):>9s} "
              f"{fmt(res['condition1_worst']):>9s} "
              f"{fmt(res['condition2_worst']):>9s} {res['n_mixing_votes']:>6d}  "
              f"{fmt(res['strictness_condition1'] or 0):>9s} "
              f"{fmt(res['strictness_condition2'] or 0):>9s}")
        for pr in res["problems"]:
            print(f"        PROBLEM: {pr}")
        for (j, x, y, rv, g) in res["near_indifferent_pure_votes"]:
            print(f"        note: {j} votes {rv:.0f} on {x} -> {y} with "
                  f"V(y)-V(x) = {g:.1e}, within eps of indifference "
                  "(allowed: any vote is optimal at indifference)")
        if a.detail:
            if res["condition1_where"]:
                i, x, y, pv = res["condition1_where"]
                print(f"        worst cond 1: {i} at {x} proposes {y} (p={pv:.6g})")
            if res["condition2_where"]:
                j, x, y, rv, g = res["condition2_where"]
                print(f"        worst cond 2: {j} votes {rv:.6g} on {x} -> {y}, "
                      f"V(y)-V(x) = {g:.3e}")
    print(f"\n{n_ok}/{n} deltas pass; largest eps = {fmt(worst_eps)} "
          f"(in units of u); mixed profiles accepted up to {a.eps_max:g} x "
          "payoff spread")
    print("eps: largest violation of Condition 1 or 2, computed exactly (units of u).\n"
          "slack 1: smallest shortfall from optimal among proposals NOT made.\n"
          "slack 2: smallest |V(y)-V(x)| at a vote that is not mixed.\n"
          "Slack far above eps means the pure decisions are robust to the "
          "eps-sized approximation\nin the mixing probabilities.")


if __name__ == "__main__":
    main()
