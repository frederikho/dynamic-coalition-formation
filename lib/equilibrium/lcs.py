"""
Coalitional stability concepts: LCS (Chwe 1994) and LCCS (Mauleon & Vannetelbosch 2004).

References:
    Chwe, M. S.-Y. (1994). Farsighted coalitional stability.
    Journal of Economic Theory, 63(2), 299–325.

    Mauleon, A., & Vannetelbosch, V. (2004). Farsightedness and cautiousness 
    in coalition formation games with positive spillovers.
    Theory and Decision, 56, 291–324.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tqdm import tqdm

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
# Indirect Dominance
# ---------------------------------------------------------------------------

def compute_indirect_strict_dominance(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    tol: float = 1e-12,
    pbar: tqdm | None = None,
) -> dict[tuple[str, str], bool]:
    """Compute the full indirect strict dominance relation ≪ (Chwe 1994).

    a ≪ b iff ∃ chain a=a_0, a_1, ..., a_m=b s.t. a_i →_{S_i} a_{i+1}
    and u_j(b) > u_j(a_i) for all j in S_i.
    """
    transitions = _build_effective_transitions(players, states, effectivity)
    indirect_dom: dict[tuple[str, str], bool] = {(a, b): False for a in states for b in states if a != b}

    for b in states:
        ancestors: set[str] = {b}
        changed = True
        while changed:
            changed = False
            for a in states:
                if a in ancestors: continue
                for a_prime in list(ancestors):
                    key = (a, a_prime)
                    if key not in transitions: continue
                    for coalition in transitions[key]:
                        if all(float(u.loc[b, p]) > float(u.loc[a, p]) + tol for p in coalition):
                            ancestors.add(a)
                            indirect_dom[(a, b)] = True
                            changed = True
                            break
                    if a in ancestors: break
                if pbar is not None:
                    pbar.update(1)
    return indirect_dom


def compute_indirect_weak_dominance(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    tol: float = 1e-12,
    pbar: tqdm | None = None,
) -> dict[tuple[str, str], bool]:
    """Compute the full indirect weak dominance relation ≾ (Mauleon & Vannetelbosch 2004).

    a ≾ b iff ∃ chain a=a_0, a_1, ..., a_m=b s.t. a_i →_{S_i} a_{i+1}
    and (u_j(b) >= u_j(a_i) for all j in S_i AND u_j(b) > u_j(a_i) for some j in S_i).
    """
    transitions = _build_effective_transitions(players, states, effectivity)
    indirect_dom: dict[tuple[str, str], bool] = {(a, b): False for a in states for b in states if a != b}

    for b in states:
        ancestors: set[str] = {b}
        changed = True
        while changed:
            changed = False
            for a in states:
                if a in ancestors: continue
                for a_prime in list(ancestors):
                    key = (a, a_prime)
                    if key not in transitions: continue
                    for coalition in transitions[key]:
                        payoffs_b = [float(u.loc[b, p]) for p in coalition]
                        payoffs_a = [float(u.loc[a, p]) for p in coalition]
                        no_loss = all(pb >= pa - tol for pb, pa in zip(payoffs_b, payoffs_a))
                        some_gain = any(pb > pa + tol for pb, pa in zip(payoffs_b, payoffs_a))
                        if no_loss and some_gain:
                            ancestors.add(a)
                            indirect_dom[(a, b)] = True
                            changed = True
                            break
                    if a in ancestors: break
                if pbar is not None:
                    pbar.update(1)
    return indirect_dom


# ---------------------------------------------------------------------------
# Largest Consistent Set (LCS)
# ---------------------------------------------------------------------------

def compute_lcs(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    weak: bool = False,
    tol: float = 1e-12,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[frozenset[str], dict[tuple[str, str], bool]]:
    """Compute the Largest Consistent Set (LCS).

    If weak=False (default): Chwe (1994) LCS based on indirect strict dominance.
    If weak=True: Mauleon & Vannetelbosch (2004) LCS based on indirect weak dominance.
    """
    transitions = _build_effective_transitions(players, states, effectivity)

    progress_label = progress_desc or ("LCS (Weak)" if weak else "LCS (Strict)")
    pbar = tqdm(total=len(states), desc=progress_label, unit="step", leave=True) if show_progress else None

    if weak:
        indirect_dom = compute_indirect_weak_dominance(players, states, u, effectivity, tol, pbar=pbar)
    else:
        indirect_dom = compute_indirect_strict_dominance(players, states, u, effectivity, tol, pbar=pbar)

    def _f(X: frozenset[str]) -> frozenset[str]:
        result: list[str] = []
        for a in states:
            if pbar is not None:
                pbar.update(1)
            if a not in X: continue
            all_deterred = True
            for b in states:
                if b == a: continue
                if (a, b) not in transitions: continue
                for coalition in transitions[(a, b)]:
                    deterred = False
                    for e in X:
                        if e != b and not indirect_dom.get((b, e), False): continue
                        
                        # Deviation a -> b deterred if endpoint e in X is 'no better' than a for S.
                        payoffs_e = [float(u.loc[e, p]) for p in coalition]
                        payoffs_a = [float(u.loc[a, p]) for p in coalition]
                        
                        if weak:
                            # Deterred if NOT weakly preferred:
                            # NOT (all j in S: u_j(e) >= u_j(a) AND some j in S: u_j(e) > u_j(a))
                            # <=> (some j in S: u_j(e) < u_j(a)) OR (all j in S: u_j(e) == u_j(a))
                            some_worse = any(pe < pa - tol for pe, pa in zip(payoffs_e, payoffs_a))
                            all_equal = all(abs(pe - pa) < tol for pe, pa in zip(payoffs_e, payoffs_a))
                            if some_worse or all_equal:
                                deterred = True
                                break
                        else:
                            # Deterred if NOT strictly preferred:
                            # NOT (all j in S: u_j(e) > u_j(a))
                            # <=> (some j in S: u_j(e) <= u_j(a))
                            if not all(pe > pa + tol for pe, pa in zip(payoffs_e, payoffs_a)):
                                deterred = True
                                break
                    if not deterred:
                        all_deterred = False
                        break
                if not all_deterred: break
            if all_deterred: result.append(a)
        return frozenset(result)

    X: frozenset[str] = frozenset(states)
    iter_count = 0
    while True:
        iter_count += 1
        if pbar is not None:
            pbar.total += len(states)
            pbar.set_postfix_str(f"fixed-point {iter_count}")
            pbar.refresh()
        X_new = _f(X)
        if X_new == X: break
        X = X_new
    if pbar is not None:
        pbar.close()
    return X, indirect_dom


# ---------------------------------------------------------------------------
# Largest Cautious Consistent Set (LCCS)
# ---------------------------------------------------------------------------

def compute_lccs(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    weak: bool = False,
    tol: float = 1e-12,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[frozenset[str], dict[tuple[str, str], bool]]:
    """Compute the Largest Cautious Consistent Set (LCCS).

    If weak=False (default): Def 5 from M&V (2004) based on indirect strict dominance.
    If weak=True: Def 6 from M&V (2004) based on indirect weak dominance.
    """
    transitions = _build_effective_transitions(players, states, effectivity)
    progress_label = progress_desc or ("LCCS (Weak)" if weak else "LCCS (Strict)")
    pbar = tqdm(total=len(states), desc=progress_label, unit="step", leave=True) if show_progress else None
    if weak:
        indirect_dom = compute_indirect_weak_dominance(players, states, u, effectivity, tol, pbar=pbar)
    else:
        indirect_dom = compute_indirect_strict_dominance(players, states, u, effectivity, tol, pbar=pbar)

    def _f(X: frozenset[str]) -> frozenset[str]:
        result: list[str] = []
        for a in states:
            if pbar is not None:
                pbar.update(1)
            if a not in X: continue
            all_deterred = True
            for b in states:
                if b == a: continue
                if (a, b) not in transitions: continue
                for coalition in transitions[(a, b)]:
                    # Reachable stable outcomes from deviation target b
                    E = [e for e in X if e == b or indirect_dom.get((b, e), False)]
                    
                    # Deviation deterred if it's NOT the case that they want to move under ALL cautious distributions.
                    # They move if:
                    # - Strict: for ALL i in S: player i has (No Risk AND Opportunity)
                    # - Weak: [ALL i in S have No Risk] AND [SOME i in S have Opportunity]
                    
                    move = True
                    has_any_opportunity = False
                    for p in coalition:
                        pa = float(u.loc[a, p])
                        pe_vals = [float(u.loc[e, p]) for e in E]
                        no_risk = all(pe >= pa - tol for pe in pe_vals)
                        opportunity = any(pe > pa + tol for pe in pe_vals)
                        
                        if weak:
                            if not no_risk:
                                move = False
                                break
                            if opportunity:
                                has_any_opportunity = True
                        else:
                            if not (no_risk and opportunity):
                                move = False
                                break
                    
                    if weak and move and not has_any_opportunity:
                        move = False
                        
                    if move: # Not deterred
                        all_deterred = False
                        break
                if not all_deterred: break
            if all_deterred: result.append(a)
        return frozenset(result)

    X: frozenset[str] = frozenset(states)
    iter_count = 0
    while True:
        iter_count += 1
        if pbar is not None:
            pbar.total += len(states)
            pbar.set_postfix_str(f"fixed-point {iter_count}")
            pbar.refresh()
        X_new = _f(X)
        if X_new == X: break
        X = X_new
    if pbar is not None:
        pbar.close()
    return X, indirect_dom

# ---------------------------------------------------------------------------
# History-dependent Rational Expectation Farsighted Stable Set (HREFS)
# Dutta & Vartiainen (2020)
# ---------------------------------------------------------------------------

def compute_largest_hrefs(
    players: list[str],
    states: list[str],
    u: pd.DataFrame,
    effectivity: dict,
    strong: bool = False,
    tol: float = 1e-12,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> frozenset[str]:
    """Compute the Largest HREFS (or HSREFS if strong=True).
    
    Based on the UUD (Ultimate Undominated set) algorithm from 
    Dutta & Vartiainen (2020), Section 9.1.
    """
    transitions = _build_effective_transitions(players, states, effectivity)
    progress_label = progress_desc or ("HSREFS (L)" if strong else "HREFS (L)")
    # Unknown work size: use an indeterminate bar to avoid misleading percentages.
    pbar = tqdm(total=None, desc=progress_label, unit="step", leave=True) if show_progress else None
    
    # 1. Find all acyclic objection paths. 
    # An objection path is (x0, S1, x1, ..., xm) s.t. Sk in E(xk-1, xk) and u_Sk(xm) > u_Sk(xk-1).
    
    all_paths: set[tuple] = set()
    if pbar is not None:
        pbar.set_postfix_str("seed paths")
    for x in states:
        all_paths.add((x,)) # Trivial paths
        if pbar is not None:
            pbar.update(1)
        
    queue = [(x,) for x in states]
    if pbar is not None:
        pbar.set_postfix_str("expand paths")
    while queue:
        p = queue.pop(0)
        curr = p[-1]
        for next_s in states:
            if next_s == curr: continue
            if next_s in p: continue # keep it acyclic
            if (curr, next_s) not in transitions: continue
            
            for S in transitions[(curr, next_s)]:
                new_p = p + (S, next_s)
                all_paths.add(new_p)
                queue.append(new_p)
        if pbar is not None:
            pbar.update(1)

    # Filter paths to keep only valid objection paths
    objection_paths = set()
    if pbar is not None:
        pbar.set_postfix_str("filtering paths")
    for p in all_paths:
        if len(p) == 1:
            objection_paths.add(p)
            if pbar is not None:
                pbar.update(1)
            continue
        xm = p[-1]
        valid = True
        for i in range(0, len(p) - 1, 2):
            xi = p[i]
            S = p[i+1]
            if not all(float(u.loc[xm, player]) > float(u.loc[xi, player]) + tol for player in S):
                valid = False
                break
        if valid:
            objection_paths.add(p)
        if pbar is not None:
            pbar.update(1)

    # 2. UUD algorithm: recursively eliminate dominated paths
    P = objection_paths
    if pbar is not None:
        pbar.set_postfix_str("UUD elimination")
    while True:
        paths_by_start = {x: [p for p in P if p[0] == x] for x in states}
        new_P = set()
        for p in P:
            if len(p) == 1:
                x = p[0]
                dominated = False
                for y in states:
                    if y == x: continue
                    if (x, y) not in transitions: continue
                    for S in transitions[(x, y)]:
                        if not paths_by_start[y]: continue
                        if all(all(float(u.loc[path_y[-1], player]) > float(u.loc[x, player]) + tol 
                                   for player in S) 
                               for path_y in paths_by_start[y]):
                            dominated = True
                            break
                    if dominated: break
                if not dominated:
                    new_P.add(p)
            else:
                subpath = p[2:]
                if subpath not in P: continue
                
                x0, S1, x1 = p[0], p[1], p[2]
                term_p = p[-1]
                
                dominated = False
                for y in states:
                    if y == x0: continue
                    if (x0, y) not in transitions: continue
                    for S in transitions[(x0, y)]:
                        if not paths_by_start[y]: continue
                        if strong and not S1.isdisjoint(S): continue
                        if not strong and S != S1: continue
                        if all(all(float(u.loc[path_y[-1], player]) > float(u.loc[term_p, player]) + tol 
                                   for player in S) 
                               for path_y in paths_by_start[y]):
                            dominated = True
                            break
                    if dominated: break
                if not dominated:
                    new_P.add(p)
            if pbar is not None:
                pbar.update(1)
        
        if new_P == P: break
        P = new_P
    if pbar is not None:
        pbar.close()

    return frozenset(p[-1] for p in P)
