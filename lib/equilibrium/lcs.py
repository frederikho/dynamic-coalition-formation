"""
Largest Consistent Set (LCS) from Chwe (1994).

Reference:
    Chwe, M. S.-Y. (1994). Farsighted coalitional stability.
    Journal of Economic Theory, 63(2), 299–325.

The LCS is the unique largest set of outcomes that can possibly be stable
when players are fully farsighted: every coalition anticipates that once it
acts, other coalitions may react, and so on without limit.  It improves on
the von Neumann–Morgenstern stable set by not assuming that deviations stop
at the immediate next outcome.

Key definitions (Chwe pp. 301–303)
------------------------------------
Effectiveness  a →_S b : coalition S can move the status quo from a to b.
Direct dom.    a < b    : ∃S s.t. a →_S b and a ≺_S b (all S prefer b > a).
Indirect dom.  a ≪ b    : ∃ chain a=a_0,…,a_m=b and coalitions S_0,…,S_{m-1}
                          s.t. a_i →_{S_i} a_{i+1}  AND  a_i ≺_{S_i} b
                          (at each step all deviators prefer the FINAL
                          outcome b over their CURRENT position a_i).
Consistent set Y ⊆ Z    : a ∈ Y  iff  ∀ deviation a →_S d,
                          ∃ e ∈ Y with (d=e or d ≪ e)
                          s.t.  a ⊁_S e  (NOT all S strictly prefer e > a).
LCS                      : unique largest consistent set; found by iterating
                          f(X) = {a ∈ Z : all deviations from a deterred in X}
                          from X = Z until fixed point (Tarski).

Mapping to our model
--------------------
The effectiveness relation is derived from the effectivity correspondence:
for each (a→b, proposer), the effective coalition is
    S = {proposer} ∪ committee(proposer, a, b).
Different proposers may yield different S for the same (a, b) pair; we
consider all of them — any could initiate the deviation.

Preferences are strict: u_i(b) > u_i(a)  (short-term payoffs).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from lib.utils import get_approval_committee


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_effective_transitions(
    players: list[str],
    states: list[str],
    effectivity: dict,
) -> dict[tuple[str, str], list[frozenset[str]]]:
    """For each ordered (a, b) pair with a ≠ b, return the list of distinct
    effective coalitions S that can move from a to b.

    In the Heyen-Lehtomaa model S = {proposer} ∪ committee(proposer, a, b).
    We enumerate all proposers and deduplicate the resulting coalitions.
    """
    result: dict[tuple[str, str], list[frozenset[str]]] = {}
    for a in states:
        for b in states:
            if a == b:
                continue
            seen: set[frozenset[str]] = set()
            coalitions: list[frozenset[str]] = []
            for proposer in players:
                committee = get_approval_committee(effectivity, players, proposer, a, b)
                s = frozenset([proposer] + list(committee))
                if s not in seen:
                    seen.add(s)
                    coalitions.append(s)
            result[(a, b)] = coalitions
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_indirect_dominance(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    tol: float = 1e-12,
) -> dict[tuple[str, str], bool]:
    """Compute the full indirect dominance relation ≪ over Z × Z.

    ``indirect_dom[(a, b)] = True``  means  a ≪ b ("a is indirectly
    dominated by b"): there exists a chain a = a_0, a_1, …, a_m = b and
    coalitions S_0, …, S_{m-1} such that

        a_i →_{S_i} a_{i+1}    (effective transition)
        u_j(b) > u_j(a_i)  ∀j ∈ S_i   (all deviators prefer final outcome b
                                         over their current position a_i)

    Computed via backward BFS for each target b: start from {b}, expand to
    any predecessor a for which some coalition S can move a → a′ ∈ ancestors
    with all S strictly preferring b over a.

    Parameters
    ----------
    players     : list of player names
    states      : list of state names
    u           : DataFrame(states × players) of short-term payoffs
    effectivity : effectivity correspondence (from heyen_lehtomaa_2021)
    tol         : strict-preference tolerance (u_i(b) > u_i(a) + tol)

    Returns
    -------
    dict[(a, b)] → bool  for all a ≠ b
    """
    transitions = _build_effective_transitions(players, states, effectivity)

    indirect_dom: dict[tuple[str, str], bool] = {
        (a, b): False
        for a in states for b in states if a != b
    }

    for b in states:
        # ancestors: states a with a ≪ b (plus b itself as base)
        ancestors: set[str] = {b}
        changed = True
        while changed:
            changed = False
            for a in states:
                if a in ancestors:
                    continue
                # Try to step a → a_prime ∈ ancestors via a coalition
                # whose members all strictly prefer b over a.
                for a_prime in list(ancestors):
                    key = (a, a_prime)
                    if key not in transitions:
                        continue
                    for coalition in transitions[key]:
                        if all(
                            float(u.loc[b, player]) > float(u.loc[a, player]) + tol
                            for player in coalition
                        ):
                            ancestors.add(a)
                            indirect_dom[(a, b)] = True
                            changed = True
                            break
                    if a in ancestors:
                        break

    return indirect_dom


def compute_lcs(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    tol: float = 1e-12,
) -> tuple[frozenset[str], dict[tuple[str, str], bool]]:
    """Compute the Largest Consistent Set (Chwe 1994, Proposition 1).

    A set Y ⊆ Z is *consistent* iff for every a ∈ Y and every possible
    deviation  a →_S d  there exists e ∈ Y with (d = e  or  d ≪ e) such
    that  a ⊁_S e  (NOT all S strictly prefer e over a).

    Interpretation: a deviation is *deterred* as long as some stable
    endpoint reachable from the deviation target is no better (for some
    member of the deviating coalition) than staying.  Y is consistent iff
    every deviation from every member is deterred within Y.

    The LCS is the unique largest consistent set — found by applying the
    isotonic operator

        f(X) = { a ∈ Z :  for all (a →_S d),  ∃ e ∈ X  with (d=e or d≪e)
                           s.t.  a ⊁_S e }

    iteratively from X = Z until convergence (Tarski fixed-point theorem).

    Parameters
    ----------
    players     : list of player names
    states      : list of state names
    u           : DataFrame(states × players) of short-term payoffs
    effectivity : effectivity correspondence (from heyen_lehtomaa_2021)
    tol         : strict-preference tolerance used throughout

    Returns
    -------
    lcs          : frozenset of states in the LCS
    indirect_dom : dict[(a,b)] = bool — the full indirect dominance relation
                   (exposed for debugging / inspection)
    """
    transitions = _build_effective_transitions(players, states, effectivity)
    indirect_dom = compute_indirect_dominance(players, states, u, effectivity, tol=tol)

    def _f(X: frozenset[str]) -> frozenset[str]:
        """Single application of Chwe's operator f."""
        result: list[str] = []
        for a in states:
            if a not in X:
                continue
            all_deterred = True
            for b in states:
                if b == a:
                    continue
                key = (a, b)
                if key not in transitions:
                    continue
                for coalition in transitions[key]:
                    # Possible deviation: a →_coalition b.
                    # It is deterred iff ∃ e ∈ X with (b=e or b ≪ e)
                    # such that NOT all coalition strictly prefer e over a.
                    deterred = False
                    for e in X:
                        if e != b and not indirect_dom.get((b, e), False):
                            continue  # e is not reachable from b
                        # Check: a ⊁_S e  ⟺  some member of S does not
                        # strictly prefer e over a
                        if not all(
                            float(u.loc[e, player]) > float(u.loc[a, player]) + tol
                            for player in coalition
                        ):
                            deterred = True
                            break
                    if not deterred:
                        all_deterred = False
                        break
                if not all_deterred:
                    break
            if all_deterred:
                result.append(a)
        return frozenset(result)

    # Iterate f from Z until fixed point.
    # f is isotonic (X ⊆ Y ⟹ f(X) ⊆ f(Y)) and f(Z) ⊆ Z, so the sequence
    # Z ⊇ f(Z) ⊇ f²(Z) ⊇ … converges to the LCS in at most |Z| steps.
    X: frozenset[str] = frozenset(states)
    while True:
        X_new = _f(X)
        if X_new == X:
            break
        X = X_new

    return X, indirect_dom
