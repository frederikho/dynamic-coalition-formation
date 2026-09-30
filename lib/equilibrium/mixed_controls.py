"""
Shared helpers for the synthetic mixed-equilibrium control games.

These were previously carried in a throwaway script that several tools imported.
They are small, they are used by the generator, the solvers and the audits alike,
and every one of them encodes a decision that is easy to get wrong:

  committees()      uses the FRAMEWORK effectivity rule, not jeres_vfi's own
                    voters().  The two disagree on 9 of the 60 transitions at n=3 --
                    voters() gives a unilateral exit an EMPTY committee where
                    heyen_lehtomaa_2021 puts the proposer in it -- so a control built
                    against voters() is a different game from the one the CLI solves.

  qs_from_alphas()  q_i(x->y) is the product of the committee's acceptance
                    probabilities.  This is the only route by which an alpha reaches
                    the value function, via T[x,y] += rho * sigma * q.

  load_game()       reads a payoff table into jeres_vfi's partition ordering.  The
                    framework's state order and jeres's differ, and permuting them
                    silently yields a different game.

See `reports/mixed_solver/PIPELINE.md` for how these fit together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lib.effectivity import get_effectivity
from lib.equilibrium.jeres_vfi import Game, fw_state_name_to_partition

PLAYERS = ["CHN", "EUR", "USA"]
STATES = ["( )", "(CHNEUR)", "(CHNUSA)", "(EURUSA)", "(CHNEURUSA)"]
EFFECTIVITY_RULE = "heyen_lehtomaa_2021"


def state_names(players: list[str] | None = None) -> list[str]:
    """Canonical n=3 coalition-structure names, in the framework's own order."""
    a, b, c = players or PLAYERS
    return ["( )", f"({a}{b})", f"({a}{c})", f"({b}{c})", f"({a}{b}{c})"]


def committees(game: Game, players: list[str] | None = None,
               rule: str = EFFECTIVITY_RULE) -> dict:
    """(proposer_idx, from_idx, to_idx) -> frozenset of voter indices.

    Built from the FRAMEWORK effectivity rule.  Do not substitute jeres_vfi's
    voters(): it disagrees on unilateral exits, and a control game built against it
    is not the game the CLI solves.
    """
    players = players or PLAYERS
    names = state_names(players)
    eff = get_effectivity(rule, players, names)
    idx_of = {game.state_idx[fw_state_name_to_partition(n, players)]: n for n in names}
    return {
        (i, x, y): frozenset(
            k for k, pk in enumerate(players)
            if eff.get((players[i], idx_of[x], idx_of[y], pk), 0) == 1)
        for i in range(game.n_players)
        for x in range(game.n_states)
        for y in range(game.n_states)
    }


def qs_from_alphas(game: Game, comm: dict, alphas: list) -> list:
    """q_i(x->y) = product of the committee's acceptance probabilities."""
    qs = []
    for x in range(game.n_states):
        q = {}
        for i in range(game.n_players):
            for y in range(game.n_states):
                v = 1.0
                for j in comm[(i, x, y)]:
                    v *= alphas[x].get((j, y), 1.0)
                q[(i, y)] = v
        qs.append(q)
    return qs


def load_game(path, players: list[str] | None = None) -> Game:
    """Read a payoff table into jeres_vfi's partition ordering."""
    players = players or PLAYERS
    df = pd.read_excel(path, sheet_name="Payoffs", header=1, index_col=0)
    probe = Game.from_payoffs(players, np.zeros((5, len(players))))
    u = np.zeros((5, len(players)))
    for name in df.index:
        key = fw_state_name_to_partition(str(name).strip(), players)
        u[probe.state_idx[key]] = df.loc[name, players].values
    return Game.from_payoffs(players, u)


def is_prop_knob(k):
    """Proposal knob ("P", x, i, y, z): theta = sigma_i(x->y), 1-theta = sigma_i(x->z).

    Acceptance knobs stay 3-tuples (x, y, j), so old profiles load unchanged.
    """
    return len(k) == 5 and k[0] == "P"


def merit_vector(game, comm, sigmas, alphas, qs, V, knobs):
    """
    Full equilibrium residual: indifference at the knobs PLUS inequality violations.

    Why the inequalities have to be in here.  The bare indifference residual
    (V_j(y) - V_j(x) at each knob) encodes only the MIXING players' own conditions,
    and by the indifference principle those are FLAT in their own probability -- a
    player who mixes is indifferent, so their payoff cannot depend on their own
    weights.  What actually pins theta are the OTHER players' conditions: every pinned
    acceptance must agree with sign(dV), and every proposer's chosen target must be the
    argmax of q(y)*(V(y) - V(x)).  A root-finder on the bare residual is blind to all
    of that, so it wanders among roots that fail verification.

    Measured on the 86-table control set: the bare residual solves 85/86, and on the
    one failure (m6_07) the verifying set is an island a random point hits 0 times in
    300, so no multi-start budget finds it.  With the violations appended -- giving the
    optimiser a gradient toward the feasible region -- it is 86/86 at a mean of 0.06 s.

    Components are signed so that zero means satisfied:
      knob k          V_j(y) - V_j(x)                  (must be exactly 0)
      pinned alpha=1  min(0, dV)                       (needs dV >= 0)
      pinned alpha=0  max(0, dV)                       (needs dV <= 0)
      proposal        max(0, gain(y) - gain(support))  (support must be the argmax)

    PROPOSAL KNOBS.  A knob ("P", x, i, y, z) frees sigma_i(x->y); its residual is the
    proposer's own indifference gain(y) - gain(z), NOT a bare V-gap.  The zero diagonal
    applies to it exactly as to an acceptance knob: shifting weight between two tied
    targets moves (dT.V)[x,i] by rho_i * r_i, so dr_i/dtheta = K * r_i and a proposer
    cannot satisfy their own indifference with their own weight.

    MULTI-TARGET SUPPORTS.  The old code took `chosen` as the single target with
    sigma > 0.5.  A genuinely mixed row has no such target: a 0.5/0.5 row fell through
    to the status-quo default and then emitted a spurious violation against its own
    co-supported target.  The support is now read as {y : sigma > 0}, its gains are
    constrained EQUAL, and only targets outside it get inequalities.  For a pure row
    the support is a singleton and this reduces to the previous behaviour exactly.
    """
    ns, npl = game.n_states, game.n_players
    acc_knobs = [k for k in knobs if not is_prop_knob(k)]
    prop_knobs = [k for k in knobs if is_prop_knob(k)]
    knobset = set(acc_knobs)
    # (i, x) whose indifference is already supplied by a knob residual, so the
    # support-equality block below must not emit it a second time.
    knobbed_rows = {(k[2], k[1]) for k in prop_knobs}

    def gain(x, i, y):
        return 0.0 if y == x else qs[x][(i, y)] * (V[y, i] - V[x, i])

    out = [V[y, j] - V[x, j] for (x, y, j) in acc_knobs]
    out += [gain(x, i, y) - gain(x, i, z) for (_, x, i, y, z) in prop_knobs]
    for x in range(ns):
        for i in range(npl):
            for y in range(ns):
                if y == x:
                    continue
                for j in comm[(i, x, y)]:
                    if (x, y, j) in knobset:
                        continue
                    a = alphas[x].get((j, y))
                    if a is None:
                        continue
                    gap = V[y, j] - V[x, j]
                    if a >= 1 - 1e-12:
                        out.append(min(0.0, gap))
                    elif a <= 1e-12:
                        out.append(max(0.0, gap))
            supp = [y for y in range(ns) if sigmas[x].get((i, y), 0.0) > 0.0]
            if not supp:
                supp = [x]
            g0 = gain(x, i, supp[0])
            if (i, x) not in knobbed_rows:
                for y in supp[1:]:
                    out.append(gain(x, i, y) - g0)      # equal EV across the support
            for y in range(ns):
                if y in supp:
                    continue
                out.append(max(0.0, gain(x, i, y) - g0))
    return np.array(out, dtype=float)
