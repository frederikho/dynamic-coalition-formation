"""smpe_sweep.py -- SMPE across a sweep of farsightedness delta.

The entry point for the finished tool: give it a payoff matrix, get verified
equilibria from low to high delta.

    from smpe_sweep import sweep
    res = sweep(["A", "B", "C"], PAYOFFS, rows=ROWS, lo=0.80, hi=0.99, step=0.01)
    print(res.table())

    python smpe_sweep.py payoffs.csv --players A,B,C --lo 0.8 --hi 0.99 --step 0.01

How each delta is solved
------------------------
1. Anchors.  Every grid point is solved independently: policy iteration (a
   fixed point that verifies is a pure SMPE), then the QRE homotopy in
   `smpe_hard`, then the original support search in `smpe` as a last resort.
   The homotopy needs no warm start -- it begins at lambda ~ 0 where the problem
   is trivial -- so it is a reliable anchor at every point.

2. Branch following (on by default).  Games have several equilibria, and
   independent solves at neighbouring deltas may land on different ones, which
   makes V(delta) jump for reasons that have nothing to do with the economics.
   So the sweep also carries the equilibrium from each delta to the next --
   same mixing structure, previous V as the start, in small steps -- and uses
   that continued solution when it verifies.  Where continuation breaks, the
   anchor is used and the point is flagged as a branch change: either a
   genuine bifurcation or a switch to a different equilibrium.

3. Gap filling.  Any delta still unsolved is approached by continuation from
   the nearest solved delta, below first, then above.  On 600 held-out games
   this closed the only case the independent solves missed.

Every returned solution has passed `smpe.verify` against the value function
induced by its own strategies.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import smpe
from . import smpe_hard as SH
from .smpe import Game, Solution, policy_iteration, _constrained_pi, _finalize

__all__ = ["sweep", "solve_point", "continue_solution", "SweepResult"]


# ---------------------------------------------------------------------------
# single-delta pipeline
# ---------------------------------------------------------------------------
def solve_point(players, payoffs, delta: float, *, rows=None, rho=None,
                rescale: bool = True, budget: float = 15.0,
                fallback_budget: float = 5.0, committees=None
                ) -> Tuple[Optional[Solution], Optional[str]]:
    """Solve one delta from scratch.  Returns (solution, method)."""
    g = Game.build(players, payoffs, delta, rho=rho, rows=rows, rescale=rescale,
                   committees=committees)
    pi = policy_iteration(g, g.payoffs.copy())
    if pi.status == "fixed":
        p = pi.policies[0]
        sol = _finalize(g, p.sigma, p.alpha, "pure", None, verify_tol=1e-7)
        if sol is not None:
            return sol, "pure"
    sol = SH.solve_hard(players, payoffs, delta, rho=rho, rows=rows,
                        rescale=rescale, time_budget=budget, committees=committees)
    if sol is not None:
        return sol, "homotopy"
    sol, _d, _p = smpe._attempt(g, g.payoffs.copy(), tie_accept=True,
                                tie_break="stay", route="support",
                                verify_tol=1e-7, tie_tol=1e-8,
                                rng=np.random.default_rng(0),
                                time_budget=fallback_budget)
    return sol, ("support" if sol is not None else None)


def continue_solution(players, payoffs, d_target: float, sol_prev: Solution, *,
                      rows=None, rho=None, rescale: bool = True, seed: int = 0,
                      committees=None) -> Optional[Solution]:
    """Carry an equilibrium to a nearby delta, keeping its mixing structure.

    The previous V is an exact equilibrium value function, so it pins the
    non-mixing decisions correctly by construction.  Rescaling depends only on
    the payoff matrix, so V transfers between deltas without conversion.
    """
    g = Game.build(players, payoffs, d_target, rho=rho, rows=rows,
                   rescale=rescale, committees=committees)
    ms = list(sol_prev.diagnostics.get("mixing_set") or [])
    if not ms:
        pi = policy_iteration(g, sol_prev.V)
        if pi.status != "fixed":
            return None
        p = pi.policies[0]
        return _finalize(g, p.sigma, p.alpha, "continued", None,
                         verify_tol=1e-7)
    # mixing sets from the homotopy carry explicit supports; accept entries
    # recorded as (j, s, t) triples are normalised to the ("acc", ...) form
    cands, seen = [], set()
    for c in ms:
        if c[0] == "acc":           # one entry per unordered pair: both
            key = ("acc", c[1], min(c[2], c[3]), max(c[2], c[3]))  # directions
        elif c[0] == "prop":        # are added by regime_with_mixing
            key = ("prop", c[1], c[2], tuple(c[3]) if c[3] is not None else None)
        else:
            continue
        if key not in seen:
            seen.add(key)
            cands.append(key)
    if not cands:
        return None
    got, _r, _v = _constrained_pi(g, cands, sol_prev.V,
                                  np.random.default_rng(seed), max_outer=14,
                                  n_random=10, max_iter=160)
    if got is None:
        return None
    st, R, res = got
    return _finalize(g, st["sigma"], st["alpha"], "continued", R,
                     verify_tol=1e-7,
                     rank_deficient=res.get("rank_deficient", False),
                     diagnostics=dict(mixing_set=cands, x=res["x"]))


def _walk(players, payoffs, sol: Solution, d_from: float, d_to: float, *,
          step: float, rows, rho, rescale, budget, committees=None
          ) -> Tuple[Optional[Solution], Optional[Tuple[float, float]]]:
    """Continue from d_from to d_to in small steps.

    If the mixing structure breaks at an intermediate point, re-anchor there
    with the homotopy and carry on.  Returns (solution, delta at which the walk
    first re-anchored, or None if it followed one branch throughout).  The
    re-anchor is reported as the interval (last step on the old branch, first
    step on the new one): the branch changed somewhere inside it, and hiding
    that would misreport a bifurcation as a smooth continuation.
    """
    # tolerance guards against 0.01/0.005 == 2.0000000000000004 -> 3 steps
    n = max(1, int(np.ceil(abs(d_to - d_from) / step - 1e-9)))
    cur = sol
    prev_d = d_from
    reanchored: Optional[Tuple[float, float]] = None
    for dn in np.linspace(d_from, d_to, n + 1)[1:]:
        nxt = continue_solution(players, payoffs, float(dn), cur, rows=rows,
                                rho=rho, rescale=rescale, committees=committees)
        if nxt is None:
            nxt = SH.solve_hard(players, payoffs, float(dn), rows=rows, rho=rho,
                                rescale=rescale, time_budget=budget,
                                committees=committees)
            if nxt is not None and reanchored is None:
                reanchored = (float(prev_d), float(dn))
        if nxt is None:
            return None, reanchored
        cur = nxt
        prev_d = float(dn)
    return cur, reanchored


# ---------------------------------------------------------------------------
# strategy profiles
# ---------------------------------------------------------------------------
PROB_TOL = 1e-9


def strategy_records(sol: Solution, delta: float) -> List[Dict[str, Any]]:
    """The full equilibrium strategy profile as flat records.

    Proposal records (kind='propose'), one per proposer, state and proposal made
    with positive probability:
        prob          sigma_i(s, s'): probability of proposing s' at s
        accept_prob   q_i(s, s'): probability the required voters all accept
        effective     prob * accept_prob: probability this proposal is made AND
                      implemented; summing over proposers weighted by rho gives
                      the transition matrix
        free_exit     the proposal is the proposer's own unilateral exit (no vote)
        voters        who must approve
    Acceptance records (kind='accept'), one per voter and every move that voter
    can be asked to approve -- including moves nobody proposes, because the
    equilibrium requires those votes too (strict off-path rule):
        prob          alpha_j(s, s'): probability of voting yes
        gain          V_j(s') - V_j(s), original payoff units; the vote follows
                      its sign, and mixing happens only where it is zero
        proposed_by   proposers who actually put this move to a vote at s
    All states are partition labels, so nothing depends on row order.
    """
    g = sol.game
    geom = g.geom
    L = geom.labels()
    P = geom.players
    q = smpe.compute_q(sol.alpha, geom)
    Vr = sol.V_raw
    nonunique = sol.status == "MIXED_NONUNIQUE"
    recs: List[Dict[str, Any]] = []
    for i in range(g.N):
        for s in range(g.S):
            for t in range(g.S):
                pr = float(sol.sigma[i, s, t])
                # Export every strictly positive probability, however small:
                # the exact checker (check_equilibrium.py) must see the whole
                # support, since even a tiny weight on a suboptimal proposal
                # violates proposer consistency.  Display code filters instead.
                if pr <= 0.0:
                    continue
                voters = [P[j] for j in range(g.N) if geom.voters[s, t, i, j]]
                recs.append(dict(
                    delta=delta, status=sol.status, nonunique=nonunique,
                    kind="propose", player=P[i], state=L[s], target=L[t],
                    stay=(t == s), prob=pr, accept_prob=float(q[i, s, t]),
                    effective=pr * float(q[i, s, t]),
                    free_exit=bool(t != s and geom.free_exit[s, t, i]),
                    voters=",".join(voters)))
    for j in range(g.N):
        for s in range(g.S):
            for t in range(g.S):
                if not geom.relevant[j, s, t]:
                    continue
                by = [P[i] for i in range(g.N)
                      if geom.voters[s, t, i, j] and sol.sigma[i, s, t] > 0.0]
                recs.append(dict(
                    delta=delta, status=sol.status, nonunique=nonunique,
                    kind="accept", player=P[j], state=L[s], target=L[t],
                    prob=float(sol.alpha[j, s, t]),
                    gain=float(Vr[t, j] - Vr[s, j]),
                    proposed_by=",".join(by)))
    return recs


def _fmt_p(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") if 0 < x < 1 else f"{x:.0f}"


def strategy_text(sol: Solution, delta: float) -> str:
    """Readable strategy profile for one delta."""
    g = sol.game
    L = g.geom.labels()
    w = max(len(x) for x in L)
    recs = strategy_records(sol, delta)
    out = [f"delta = {delta:.4f}   {sol.status}"]
    if sol.status == "MIXED_NONUNIQUE":
        out.append("  NOTE: the equilibria here form a continuum.  The mixing "
                   "probabilities below are\n  one verified point on it; other "
                   "values are equilibria too.")
    out.append("")
    out.append("  Proposals  (prob of proposing -> prob it passes)")
    for pl in g.geom.players:
        for st in L:
            rs = [r for r in recs if r["kind"] == "propose"
                  and r["player"] == pl and r["state"] == st]
            parts = []
            for r in sorted(rs, key=lambda r: -r["prob"]):
                if r["stay"]:
                    parts.append(f"stay {_fmt_p(r['prob'])}")
                else:
                    tag = (" [free exit]" if r["free_exit"] else
                           " [rejected for sure]" if r["accept_prob"] <= PROB_TOL
                           else f" -> passes {_fmt_p(r['accept_prob'])}")
                    parts.append(f"{r['target']} {_fmt_p(r['prob'])}{tag}")
            out.append(f"    {pl} at {st.ljust(w)} : " + ";  ".join(parts))
    out.append("")
    out.append("  Votes  (* = actually proposed at that state)")
    for pl in g.geom.players:
        out.append(f"    voter {pl}")
        for st in L:
            rs = [r for r in recs if r["kind"] == "accept"
                  and r["player"] == pl and r["state"] == st]
            if not rs:
                continue
            acc = [r for r in rs if r["prob"] >= 1 - PROB_TOL]
            rej = [r for r in rs if r["prob"] <= PROB_TOL]
            mix = [r for r in rs if PROB_TOL < r["prob"] < 1 - PROB_TOL]
            star = lambda r: r["target"] + ("*" if r["proposed_by"] else "")
            bits = []
            if acc:
                bits.append("accepts " + ", ".join(star(r) for r in acc))
            if rej:
                bits.append("rejects " + ", ".join(star(r) for r in rej))
            if mix:
                bits.append("MIXES " + ", ".join(
                    f"{star(r)} ({_fmt_p(r['prob'])})" for r in mix))
            out.append(f"      at {st.ljust(w)} : " + ";  ".join(bits))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
@dataclass
class SweepPoint:
    delta: float
    solution: Optional[Solution]
    method: Optional[str]            # pure | homotopy | support | continued | gap-filled
    branch_change: bool = False      # continuation from the previous delta broke
    change_at: Optional[Tuple[float, float]] = None  # bracketing interval
    anchor_differs: bool = False     # independent solve found another equilibrium
    seconds: float = 0.0


@dataclass
class SweepResult:
    players: List[str]
    points: List[SweepPoint]
    labels: List[str]
    rows: Optional[List[str]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_solved(self) -> int:
        return sum(p.solution is not None for p in self.points)

    def V(self) -> np.ndarray:
        """Value functions in original payoff units, shape (n_delta, S, N);
        NaN where a delta is unsolved.  States in `self.labels` order."""
        S, N = len(self.labels), len(self.players)
        out = np.full((len(self.points), S, N), np.nan)
        for k, p in enumerate(self.points):
            if p.solution is not None:
                out[k] = p.solution.V_raw
        return out

    def table(self) -> str:
        L = self.labels
        w = max(len(x) for x in L)
        lines = [f"{self.n_solved}/{len(self.points)} deltas solved and verified",
                 "",
                 f"{'delta':>7s}  {'status':<16s} {'method':<11s} "
                 f"{'regret':>8s}  notes"]
        for p in self.points:
            if p.solution is None:
                lines.append(f"{p.delta:7.4f}  {'UNSOLVED':<16s}")
                continue
            notes = []
            if p.branch_change:
                notes.append("branch change" + (
                    f" in ({p.change_at[0]:.4f}, {p.change_at[1]:.4f}]"
                    if p.change_at else ""))
            if p.anchor_differs:
                notes.append("other equilibria exist")
            r = p.solution.report
            lines.append(f"{p.delta:7.4f}  {p.solution.status:<16s} "
                         f"{p.method:<11s} {r.max_proposal_regret:8.1e}  "
                         + ", ".join(notes))
        lines += ["", self.longrun_table()]
        return "\n".join(lines)

    @staticmethod
    def _class_name(cl, L) -> str:
        if len(cl) == 1:
            return L[cl[0]]
        return "cycle[" + " <-> ".join(L[t] for t in cl) + "]"

    def longrun_table(self) -> str:
        """Where play ends up, from each starting coalition structure.

        Consecutive deltas with the same long-run behaviour are merged into one
        range, since the typical picture is a few regimes separated by
        bifurcations.  Probabilities are absorption probabilities: the chance
        that play starting at that structure is eventually absorbed into each
        absorbing state (or recurrent cycle).
        """
        L = self.labels
        w = max(len(x) for x in L)
        pts = [p for p in self.points if p.solution is not None]
        if not pts:
            return "no solved deltas"

        def sig(p):
            ch = p.solution.chain
            names = [self._class_name(c, L) for c in ch["classes"]]
            return (tuple(names),
                    tuple(tuple(np.round(ch["absorption"][s_], 3))
                          for s_ in range(len(L))))

        groups = []
        for p in pts:
            if groups and groups[-1][1] == sig(p):
                groups[-1][0].append(p.delta)
            else:
                groups.append(([p.delta], sig(p)))
        out = ["Long-run outcome: probability of ending in each absorbing state,",
               "by starting coalition structure"]
        for ds, (names, absb) in groups:
            rng = (f"delta {ds[0]:.4f}" if len(ds) == 1
                   else f"delta {ds[0]:.4f} - {ds[-1]:.4f}  ({len(ds)} points)")
            n_abs = len(names)
            head = ("UNIQUE absorbing state: " + names[0] if n_abs == 1
                    else f"{n_abs} absorbing states: " + ",  ".join(names))
            out += ["", f"  {rng}", f"    {head}"]
            if n_abs == 1:
                out.append("    reached with probability 1 from every start")
                continue
            cw = max(12, max(len(n) for n in names) + 2)
            out.append("    " + "start".ljust(w + 2) +
                       "".join(n.rjust(cw) for n in names))
            for s_, row in enumerate(absb):
                out.append("    " + L[s_].ljust(w + 2) +
                           "".join(f"{v:{cw}.3f}" for v in row))
        return "\n".join(out)

    def strategies(self, delta: Optional[float] = None) -> str:
        """Readable strategy profiles, for one delta or all of them."""
        pts = [p for p in self.points if p.solution is not None
               and (delta is None or abs(p.delta - delta) < 1e-9)]
        if delta is not None and not pts:
            raise ValueError(f"no solved point at delta={delta}")
        return "\n\n".join(strategy_text(p.solution, p.delta) for p in pts)

    def strategy_records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in self.points:
            if p.solution is not None:
                out.extend(strategy_records(p.solution, p.delta))
        return out

    def to_csv(self, path: str) -> None:
        """Every strategy at every delta, one row per proposal or vote.
        Columns are described in `strategy_records`."""
        import csv
        cols = ["delta", "status", "nonunique", "kind", "player", "state",
                "target", "prob", "stay", "accept_prob", "effective",
                "free_exit", "voters", "gain", "proposed_by"]
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for r in self.strategy_records():
                wr.writerow({c: r.get(c, "") for c in cols})

    def to_json(self, path: str) -> None:
        """Everything per delta: strategies, V, transition matrix, long-run
        classes, verification margins."""
        import json
        out = dict(players=self.players, states=self.labels, points=[])
        for p in self.points:
            e: Dict[str, Any] = dict(delta=p.delta, solved=p.solution is not None)
            if p.solution is not None:
                sol = p.solution
                e.update(
                    status=sol.status, method=p.method,
                    branch_change=p.branch_change,
                    change_interval=list(p.change_at) if p.change_at else None,
                    V={st: dict(zip(self.players, map(float, sol.V_raw[k])))
                       for k, st in enumerate(self.labels)},
                    transition={a: {b: float(sol.T[i, j])
                                    for j, b in enumerate(self.labels)
                                    if sol.T[i, j] > PROB_TOL}
                                for i, a in enumerate(self.labels)},
                    recurrent_classes=[[self.labels[t] for t in c]
                                       for c in sol.chain["classes"]],
                    absorption={self.labels[s_]: {
                        self._class_name(c, self.labels):
                            float(sol.chain["absorption"][s_, k])
                        for k, c in enumerate(sol.chain["classes"])}
                        for s_ in range(len(self.labels))},
                    max_proposal_regret=float(sol.report.max_proposal_regret),
                    max_accept_violation=float(sol.report.max_accept_violation),
                    strategies=strategy_records(sol, p.delta))
            out["points"].append(e)
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)

    def value_table(self, player: int) -> str:
        L = self.labels
        w = max(len(x) for x in L)
        head = f"{'delta':>7s}  " + "".join(f"{x:>14s}" for x in L)
        lines = [f"V for player {self.players[player]} (original units)", head]
        for p in self.points:
            if p.solution is None:
                lines.append(f"{p.delta:7.4f}  " + "".join(f"{'--':>14s}" for _ in L))
            else:
                lines.append(f"{p.delta:7.4f}  " + "".join(
                    f"{p.solution.V_raw[s, player]:14.6f}" for s in range(len(L))))
        return "\n".join(lines)


def sweep(players: Sequence[str], payoffs, deltas: Optional[Sequence[float]] = None,
          *, lo: float = 0.80, hi: float = 0.99, step: float = 0.01,
          rows=None, rho=None, rescale: bool = True,
          follow_branch: bool = True, walk_step: float = 0.005,
          budget: float = 15.0, verbose: bool = False,
          committees=None) -> SweepResult:
    """Solve for SMPE at every delta in the sweep, with warm starts between.

    Returns a SweepResult whose points are in increasing delta.  Every returned
    solution is verified; unsolved deltas are reported as such, never guessed.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if deltas is None:
        deltas = np.round(np.arange(lo, hi + 1e-12, step), 10)
    deltas = sorted(float(d) for d in deltas)
    kw = dict(rows=rows, rho=rho, rescale=rescale, committees=committees)
    t0 = time.time()

    # 1. anchors -------------------------------------------------------------
    anchors: List[Tuple[Optional[Solution], Optional[str], float]] = []
    for d in deltas:
        t = time.time()
        sol, m = solve_point(players, payoffs, d, budget=budget, **kw)
        anchors.append((sol, m, time.time() - t))
        if verbose:
            print(f"  anchor delta={d:.4f}: {m or 'UNSOLVED'}", flush=True)

    points = [SweepPoint(d, s, m, seconds=t) for d, (s, m, t) in
              zip(deltas, anchors)]

    # 2. branch following ---------------------------------------------------
    if follow_branch:
        for k in range(1, len(points)):
            prev = points[k - 1]
            if prev.solution is None:
                continue
            t = time.time()
            cont, re_at = _walk(players, payoffs, prev.solution, prev.delta,
                                points[k].delta, step=walk_step, budget=budget,
                                **kw)
            anchor = points[k].solution
            if re_at is not None:
                points[k].branch_change = True
                points[k].change_at = re_at
            if cont is not None:
                if anchor is not None and np.abs(anchor.V - cont.V).max() > 1e-6:
                    points[k].anchor_differs = True
                points[k].solution = cont
                points[k].method = (points[k].method + "+followed"
                                    if points[k].method else "gap-filled")
            elif anchor is not None:
                points[k].branch_change = True
                if points[k].change_at is None:   # walk broke: bracket by grid
                    points[k].change_at = (prev.delta, points[k].delta)
            points[k].seconds += time.time() - t

    # 3. gap filling (both directions) ---------------------------------------
    for k, p in enumerate(points):
        if p.solution is not None:
            continue
        order = ([j for j in range(k - 1, -1, -1)]
                 + [j for j in range(k + 1, len(points))])
        for j in order:
            src = points[j]
            if src.solution is None:
                continue
            sol, _re = _walk(players, payoffs, src.solution, src.delta,
                             p.delta, step=walk_step, budget=budget, **kw)
            if sol is not None:
                p.solution, p.method = sol, "gap-filled"
                break

    g = Game.build(players, payoffs, deltas[0], rows=rows, rho=rho,
                   committees=committees, rescale=rescale)
    return SweepResult(players=list(players), points=points,
                       labels=g.geom.labels(),
                       rows=[str(r) for r in rows] if rows is not None else None,
                       diagnostics=dict(seconds=time.time() - t0))


# ---------------------------------------------------------------------------
def _load_matrix(path: str):
    """Returns (payoffs, players or None, rows or None).  A text file may carry
    '# players: A,B,C' and '# rows: {A,B,C}|{A,C}+{B}|...' header lines, so the
    row order travels with the data instead of depending on a flag."""
    if path.endswith(".npy"):
        return np.load(path), None, None
    players = rows = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                continue
            body = line[1:].strip()
            if body.lower().startswith("players:"):
                players = [x.strip() for x in body.split(":", 1)[1].split(",")]
            elif body.lower().startswith("rows:"):
                rows = [x.strip() for x in body.split(":", 1)[1].split("|")]
    pay = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None,
                     comments="#")
    return pay, players, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("payoffs", help=".csv, .txt or .npy, shape (B_N, N)")
    ap.add_argument("--players", help="comma-separated names (or a "
                    "'# players:' header in the file)")
    ap.add_argument("--rows", help="'|'-separated partition labels giving the "
                    "row order, e.g. '{A,B,C}|{A,C}+{B}|...'; default canonical")
    ap.add_argument("--lo", type=float, default=0.80)
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--no-follow", action="store_true",
                    help="independent solves only, no branch following")
    ap.add_argument("--values", action="store_true",
                    help="also print each player's value table")
    ap.add_argument("--strategies", action="store_true",
                    help="also print the full strategy profile at every delta")
    ap.add_argument("--export", metavar="PREFIX",
                    help="write PREFIX_strategies.csv and PREFIX.json")
    ap.add_argument("--check", action="store_true",
                    help="after solving, run the independent exact check "
                    "(check_equilibrium.py) on every profile")
    ap.add_argument("--certify", action="store_true",
                    help="after solving, prove each mixed profile has an exact "
                    "equilibrium next to it (certify_equilibrium.py); "
                    "implies --check")
    a = ap.parse_args()
    pay, f_players, f_rows = _load_matrix(a.payoffs)
    players = ([p.strip() for p in a.players.split(",")] if a.players
               else f_players)
    if players is None:
        ap.error("give --players or a '# players:' header in the file")
    rows = a.rows.split("|") if a.rows else f_rows
    if rows is None:
        print("note: no row order given; assuming the canonical order:")
        for lab in Game.build(players, pay, 0.9).geom.labels():
            print("   ", lab)
    res = sweep(players, pay, lo=a.lo, hi=a.hi,
                step=a.step, rows=rows, follow_branch=not a.no_follow,
                verbose=True)
    print()
    print(res.table())
    if a.values:
        for i in range(len(players)):
            print()
            print(res.value_table(i))
    if a.strategies:
        print()
        print(res.strategies())
    if a.check or a.certify:
        # both post-processors read the exported profile, so export first
        import os
        import subprocess
        import tempfile
        prefix = a.export or os.path.join(tempfile.mkdtemp(), "profile")
        res.to_csv(prefix + "_strategies.csv")
        res.to_json(prefix + ".json")
        here = os.path.dirname(os.path.abspath(__file__))
        for script, why in (("check_equilibrium.py", "exact check"),
                            ("certify_equilibrium.py", "certificate")):
            if script.startswith("certify") and not a.certify:
                continue
            print(f"\n--- {why} ({script}) " + "-" * 30)
            subprocess.run([sys.executable, os.path.join(here, script),
                            a.payoffs, prefix + ".json"])
        if not a.export:
            print(f"\n(profile written to {prefix}.json; pass --export to keep "
                  "it somewhere permanent)")
    elif a.export:
        res.to_csv(a.export + "_strategies.csv")
        res.to_json(a.export + ".json")
        print(f"\nwrote {a.export}_strategies.csv and {a.export}.json")
    if not (a.strategies or a.export or a.check or a.certify):
        print("\n(strategy profiles were computed but not shown or saved: add "
              "--strategies to print them, --export PREFIX to write "
              "PREFIX_strategies.csv and PREFIX.json, --check or --certify to "
              "prove them equilibria)")
    print(f"\n[{res.diagnostics['seconds']:.1f}s]")


if __name__ == "__main__":
    main()
