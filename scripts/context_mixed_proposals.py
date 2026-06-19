"""Context-dependent mixed-equilibrium solver WITH mixed proposals.

Generalises ContextSolver (scripts/context_mixed.py): in addition to per-context
acceptance mixing, the PROPOSER may mix. Each (proposer i, state x) gets a free
softmax distribution over its feasible next states. The system solved by
least_squares is:

  (a) acceptance indifference : V_k(s) = V_k(base)  for each tied group of player k
  (b) strict order            : V_k(x) > V_k(y) + margin   when tier_k[x] < tier_k[y]
  (c) proposal optimality     : for each (i,x),  sum_y p(i,x,y)*(maxEV - EV(i,x,y)) = 0
                                (mass only on EV-argmax; free mixing among EV-ties)

Free variables: tied acceptance contexts (sigmoid) + proposal logits (softmax per
(i,x)). At a root, indifferences hold (pinning BOTH acceptance and proposal mixing),
the ranking is respected, and proposals are rational. Targets the cyclic near-flat
RICE tables where a stationary equilibrium needs the proposer to mix.

Built on the forbidden-aware _FastGame.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

from lib.equilibrium.merit_descent import _FastGame
from scripts.context_mixed import ContextSolver


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


class ContextSolverMP:
    def __init__(self, setup, margin: float = 1e-6):
        self.fg = _FastGame(
            players=setup['players'], states=setup['state_names'],
            effectivity=setup['effectivity'], protocol=setup['protocol'],
            payoffs=setup['payoffs'], discounting=setup['discounting'],
            forbidden_proposals=setup.get('forbidden_proposals', frozenset()))
        self.kpos = self.fg._kpos()
        self.margin = margin
        self._br = ContextSolver(setup, margin=margin)  # pure-proposal base case
        fg = self.fg
        # proposal logit layout: for each (ii,xi) with >1 feasible target, free
        # logits for feas[1:] (first feasible pinned at logit 0 -> softmax identifiability)
        self.prop_layout = []   # (ii, xi, feas_list, start_idx, n_free)
        idx = 0
        for ii in range(fg.n):
            for xi in range(fg.S):
                feas = fg.feasible[(ii, xi)]
                nf = max(0, len(feas) - 1)
                self.prop_layout.append((ii, xi, feas, idx, nf))
                idx += nf
        self.n_prop_params = idx

    # ---- acceptance classification (same as ContextSolver) ----
    def _classify(self, tiers):
        fg = self.fg
        fixed, free = {}, []
        for k in fg.appr_keys:
            ii, xi, yi, ki = k
            pos = self.kpos[k]
            if xi == yi:
                fixed[pos] = 1.0
                continue
            if yi not in fg.feasible[(ii, xi)]:
                fixed[pos] = 0.0
                continue
            tk = tiers[ki]
            if tk[yi] < tk[xi]:
                fixed[pos] = 1.0
            elif tk[yi] > tk[xi]:
                fixed[pos] = 0.0
            else:
                free.append(pos)
        return fixed, free

    def _avec(self, fixed, free, aparams):
        a = np.zeros(len(self.fg.appr_keys))
        for pos, v in fixed.items():
            a[pos] = v
        for j, pos in enumerate(free):
            a[pos] = _sigmoid(aparams[j])
        return a

    def _prop_probs(self, pparams):
        fg = self.fg
        P = np.zeros((fg.n, fg.S, fg.S))
        for (ii, xi, feas, start, nf) in self.prop_layout:
            if nf == 0:
                P[ii, xi, feas[0]] = 1.0
            else:
                logits = np.concatenate([[0.0], pparams[start:start + nf]])
                w = _softmax(logits)
                for j, yi in enumerate(feas):
                    P[ii, xi, yi] = w[j]
        return P

    def _build_P(self, avec, prop_probs):
        fg = self.fg
        P = np.zeros((fg.S, fg.S))
        for ii in range(fg.n):
            pr = fg.protocol[ii]
            for xi in range(fg.S):
                for yi in fg.feasible[(ii, xi)]:
                    w = prop_probs[ii, xi, yi]
                    if w <= 0:
                        continue
                    pa = 1.0 if yi == xi else fg.p_app_cont(avec, self.kpos, ii, xi, yi)
                    P[xi, yi] += pr * w * pa
                    if yi != xi:
                        P[xi, xi] += pr * w * (1.0 - pa)
        return P

    def _equality_groups(self, tiers):
        fg = self.fg
        groups = []
        for ki in range(fg.n):
            by_tier = {}
            for si in range(fg.S):
                by_tier.setdefault(int(tiers[ki][si]), []).append(si)
            for g in by_tier.values():
                if len(g) > 1:
                    groups.append((ki, g))
        return groups

    def solve(self, tiers, n_starts: int = 80, max_nfev: int = 1500,
              tol: float = 1e-7, prop_tol: float = 1e-4):
        fg = self.fg
        fixed, free = self._classify(tiers)
        groups = self._equality_groups(tiers)
        na = len(free)
        npar = na + self.n_prop_params
        n_prop_res = len(self.prop_layout)  # proposal-opt residuals are the last block

        def residuals(params):
            avec = self._avec(fixed, free, params[:na])
            prop = self._prop_probs(params[na:])
            V = fg.solve_V(self._build_P(avec, prop))
            res = []
            # (a) acceptance indifference
            for ki, g in groups:
                base = V[g[0], ki]
                for si in g[1:]:
                    res.append(V[si, ki] - base)
            # (b) strict order
            for ki in range(fg.n):
                t = tiers[ki]
                for x in range(fg.S):
                    for y in range(fg.S):
                        if t[x] < t[y]:
                            res.append(min(0.0, V[x, ki] - V[y, ki] - self.margin))
            # (c) proposal optimality (one smooth residual per (i,x))
            for (ii, xi, feas, start, nf) in self.prop_layout:
                evs = []
                for yi in feas:
                    pa = 1.0 if yi == xi else fg.p_app_cont(avec, self.kpos, ii, xi, yi)
                    evs.append(pa * V[yi, ii] + (1.0 - pa) * V[xi, ii])
                evs = np.array(evs)
                best = evs.max()
                w = prop[ii, xi, feas]
                res.append(float(np.dot(w, best - evs)))
            return np.asarray(res, dtype=float)

        rng = np.random.RandomState(0)
        guesses = [np.zeros(npar)]
        # acceptance part varied; proposal logits start near 0 (uniform) then random
        for _ in range(n_starts - 1):
            g = np.concatenate([rng.uniform(-2, 2, na), rng.uniform(-3, 3, self.n_prop_params)])
            guesses.append(g)
        # --- Step 1: pure-proposal (best-response) base case ---
        br = self._br.solve(tiers)
        if br is not None:
            prop = np.zeros((fg.n, fg.S, fg.S))
            for ii in range(fg.n):
                for xi in range(fg.S):
                    prop[ii, xi, int(br['proposals'][ii, xi])] = 1.0
            return {"avec": br['avec'], "kpos": self.kpos, "prop_probs": prop,
                    "V": br['V'], "params": None, "free": br['free'], "mode": "br"}

        # --- Step 2: proposal-mixing least_squares ---
        def _accept(fun):
            # tight on indifference/strict-order; loose on proposal-opt (snap fixes it)
            if n_prop_res and fun.size:
                core, prop = fun[:-n_prop_res], fun[-n_prop_res:]
            else:
                core, prop = fun, np.zeros(0)
            core_ok = (core.size == 0) or (np.max(np.abs(core)) <= tol)
            prop_ok = (prop.size == 0) or (np.max(np.abs(prop)) <= prop_tol)
            return core_ok and prop_ok

        for g0 in guesses:
            if npar == 0:
                r = residuals(np.zeros(0))
                return self._finalize(fixed, free, np.zeros(0), tiers) if (r.size == 0 or _accept(r)) else None
            opt = least_squares(residuals, g0, max_nfev=max_nfev)
            if _accept(opt.fun):
                return self._finalize(fixed, free, opt.x, tiers)
        return None

    def _finalize(self, fixed, free, params, tiers, ev_tol: float = 1e-6):
        """Snap proposals to their EV-argmax support (so the discrete strategy is
        exactly rational), renormalise, and return arrays for verification."""
        fg = self.fg
        na = len(free)
        avec = self._avec(fixed, free, params[:na])
        prop = self._prop_probs(params[na:])
        V = fg.solve_V(self._build_P(avec, prop))
        # snap each (i,x): keep only EV-argmax feasible targets, renormalise mass
        prop_snap = np.zeros_like(prop)
        for (ii, xi, feas, start, nf) in self.prop_layout:
            evs = []
            for yi in feas:
                pa = 1.0 if yi == xi else fg.p_app_cont(avec, self.kpos, ii, xi, yi)
                evs.append(pa * V[yi, ii] + (1.0 - pa) * V[xi, ii])
            evs = np.array(evs)
            best = evs.max()
            keep = [feas[j] for j in range(len(feas)) if evs[j] >= best - ev_tol]
            mass = sum(prop[ii, xi, y] for y in keep)
            for y in keep:
                prop_snap[ii, xi, y] = prop[ii, xi, y] / mass if mass > 0 else 1.0 / len(keep)
        V2 = fg.solve_V(self._build_P(avec, prop_snap))
        return {"avec": avec, "kpos": self.kpos, "prop_probs": prop_snap,
                "V": V2, "params": params, "free": free}
