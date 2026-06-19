"""Context-dependent mixed-equilibrium solver (per-context acceptance mixing).

Generalises the shared-parameter ansatz (scripts/reduced_weak_exact.py): instead of
one mixing parameter per player-indifference-GROUP reused across all contexts, every
indifferent acceptance CONTEXT (proposer i, current x, next y, approver k) gets its
OWN free variable. Proposals are best-response (pure) given V -- this targets the
pure-proposal + mixed-acceptance class (e.g. the manifold equilibria).

Crucially, unlike scripts/apply_ansatz_hard.py (which minimised the merit, flat at a
tie so it cannot pin the mixing), this SOLVES the explicit indifference equations:

    residuals = { V_k(s) - V_k(base)  for each tied group of player k }     (pins mixing)
              + { min(0, V_k(x) - V_k(y) - margin)  for strict tier x<y }   (ranking holds)

via least_squares over the free per-context acceptance variables (sigmoid-mapped).
Built on the forbidden-aware FastGame, so adjacent_step is handled correctly.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from lib.equilibrium.merit_descent import _FastGame


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class ContextSolver:
    def __init__(self, setup, margin: float = 1e-6, br_sweeps: int = 8):
        self.fg = _FastGame(
            players=setup['players'], states=setup['state_names'],
            effectivity=setup['effectivity'], protocol=setup['protocol'],
            payoffs=setup['payoffs'], discounting=setup['discounting'],
            forbidden_proposals=setup.get('forbidden_proposals', frozenset()))
        self.kpos = self.fg._kpos()
        self.margin = margin
        self.br_sweeps = br_sweeps

    # ---- per-ranking free/fixed acceptance classification ----
    def _classify(self, tiers):
        fg = self.fg
        fixed = {}          # pos -> 0/1 value (strict / self-loop / forbidden-irrelevant)
        free = []           # positions of tied, feasible, non-self acceptance contexts
        for k in fg.appr_keys:
            ii, xi, yi, ki = k
            pos = self.kpos[k]
            if xi == yi:
                fixed[pos] = 1.0
                continue
            if yi not in fg.feasible[(ii, xi)]:
                fixed[pos] = 0.0    # forbidden transition: acceptance irrelevant (never proposed)
                continue
            tk = tiers[ki]
            if tk[yi] < tk[xi]:
                fixed[pos] = 1.0
            elif tk[yi] > tk[xi]:
                fixed[pos] = 0.0
            else:
                free.append(pos)
        return fixed, free

    def _avec(self, fixed, free, params):
        a = np.zeros(len(self.fg.appr_keys))
        for pos, v in fixed.items():
            a[pos] = v
        for j, pos in enumerate(free):
            a[pos] = _sigmoid(params[j])
        return a

    def _value_with_br(self, avec):
        """Best-response proposals given acceptances; return (V, proposals)."""
        fg = self.fg
        prop = np.tile(np.arange(fg.S), (fg.n, 1))  # self-loop seed
        V = fg.solve_V(fg.build_P_cont(avec, self.kpos, prop))
        for _ in range(self.br_sweeps):
            new = fg.best_response_proposals(avec, self.kpos, V)
            if np.array_equal(new, prop):
                break
            prop = new
            V = fg.solve_V(fg.build_P_cont(avec, self.kpos, prop))
        return V, prop

    def _equality_groups(self, tiers):
        fg = self.fg
        groups = []  # (player, [state idxs]) for tier-groups of size>1
        for ki in range(fg.n):
            by_tier = {}
            for si in range(fg.S):
                by_tier.setdefault(int(tiers[ki][si]), []).append(si)
            for g in by_tier.values():
                if len(g) > 1:
                    groups.append((ki, g))
        return groups

    def solve(self, tiers, n_starts: int = 4, max_nfev: int = 300, tol: float = 1e-7):
        fg = self.fg
        fixed, free = self._classify(tiers)
        groups = self._equality_groups(tiers)

        def residuals(params):
            avec = self._avec(fixed, free, params)
            V, prop = self._value_with_br(avec)
            res = []
            # (a) indifference equations: tied states must be exactly equal in V
            for ki, g in groups:
                base = V[g[0], ki]
                for si in g[1:]:
                    res.append(V[si, ki] - base)
            # (b) strict orderings must hold (one-sided penalty)
            for ki in range(fg.n):
                t = tiers[ki]
                for x in range(fg.S):
                    for y in range(fg.S):
                        if t[x] < t[y]:
                            res.append(min(0.0, V[x, ki] - V[y, ki] - self.margin))
            return np.asarray(res, dtype=float)

        if not free:
            # no mixing freedom -> pure; just check residuals are ~0
            r = residuals(np.zeros(0))
            if r.size == 0 or np.max(np.abs(r)) <= tol:
                return self._finalize(fixed, free, np.zeros(0), tiers)
            return None

        m = len(free)
        rng = np.random.RandomState(0)
        guesses = [np.zeros(m), np.full(m, 2.0), np.full(m, -2.0)]
        guesses += [rng.uniform(-2, 2, m) for _ in range(max(0, n_starts - 3))]
        for g0 in guesses:
            opt = least_squares(residuals, g0, max_nfev=max_nfev)
            if np.max(np.abs(opt.fun)) <= tol:
                return self._finalize(fixed, free, opt.x, tiers)
        return None

    def _finalize(self, fixed, free, params, tiers):
        avec = self._avec(fixed, free, params)
        V, prop = self._value_with_br(avec)
        return {"avec": avec, "kpos": self.kpos, "proposals": prop, "V": V,
                "params": params, "free": free}
