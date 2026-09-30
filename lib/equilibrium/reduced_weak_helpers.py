"""Reconstructed helpers for scripts/reduced_weak_exact.py.

These five names were deleted from scripts/search_ordinal_rankings.py (working-tree
edit) which broke reduced_weak_exact.py's import. Reconstructed here, kept faithful
to how reduced_weak_exact uses them.

Array conventions (matching induce_shared_parameter_profile):
  proposal_probs   : (n_players, n_states, n_states)  [proposer, current, next]
  approval_action  : (n_players, n_players, n_states, n_states) [proposer, approver, cur, nxt]
  approval_pass    : (n_players, n_states, n_states)  product of committee approvals
  V_array          : (n_states, n_players)
  committee_idxs   : committee_idxs[proposer_i][current_i][next_i] -> tuple of approver idxs
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

# Re-export existing implementations that still live in the codebase.
from scripts.search_ordinal_rankings import _build_payoff_config  # noqa: F401
from lib.equilibrium.ordinal_ranking.ranking_orders import _generate_weak_orders  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_payoff_file(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = REPO_ROOT / "payoff_tables" / value
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find payoff file: {value!r}")


def _solve_values_fast_array(P_array: np.ndarray, payoff_array: np.ndarray,
                             discounting: float) -> np.ndarray:
    """V = (I - delta P)^-1 (1 - delta) u, solved for all players at once.

    payoff_array: (n_states, n_players); returns V (n_states, n_players)."""
    n = P_array.shape[0]
    A = np.eye(n) - discounting * P_array
    b = (1.0 - discounting) * payoff_array
    return np.linalg.solve(A, b)


def _weak_equality_groups(states: list[str],
                          tiers: tuple[np.ndarray, ...]) -> list[list[list[str]]]:
    """Per player, the list of tier-groups (states sharing a tier). Singletons
    included (they contribute no equality residual)."""
    out: list[list[list[str]]] = []
    for tier in tiers:
        groups: dict[int, list[str]] = {}
        for idx, state in enumerate(states):
            groups.setdefault(int(tier[idx]), []).append(state)
        out.append([groups[t] for t in sorted(groups)])
    return out


def _verify_equilibrium_fast(
    *,
    players: list[str],
    states: list[str],
    effectivity: dict,
    P_proposals=None,
    P_approvals=None,
    V_df=None,
    proposal_probs: np.ndarray,
    approval_action: np.ndarray,
    approval_pass: np.ndarray,
    V_array: np.ndarray,
    committee_idxs,
    forbidden_proposals: frozenset = frozenset(),
    atol: float = 1e-7,
    prob_tol: float = 1e-9,
) -> tuple[bool, str, dict | None]:
    """Fast array verifier of the two MPE conditions. Skips forbidden proposals."""
    n_players, n_states = len(players), len(states)

    # Condition 2: approval rationality.
    for pi in range(n_players):
        for ci in range(n_states):
            for ni in range(n_states):
                if ci == ni:
                    continue
                if (players[pi], states[ci], states[ni]) in forbidden_proposals:
                    continue
                committee = committee_idxs[pi][ci][ni]
                for ki in committee:
                    a = approval_action[pi, ki, ci, ni]
                    vn, vc = V_array[ni, ki], V_array[ci, ki]
                    if abs(vn - vc) <= atol:
                        continue  # indifferent: any prob ok
                    if vn > vc and not (a >= 1.0 - 1e-9):
                        return False, f"approval: {players[ki]} should accept {states[ci]}->{states[ni]}", {
                            "type": "approval", "approver": players[ki],
                            "current": states[ci], "next": states[ni], "a": float(a)}
                    if vn < vc and not (a <= 1e-9):
                        return False, f"approval: {players[ki]} should reject {states[ci]}->{states[ni]}", {
                            "type": "approval", "approver": players[ki],
                            "current": states[ci], "next": states[ni], "a": float(a)}

    # Condition 1: proposal rationality.
    for pi in range(n_players):
        for ci in range(n_states):
            evs = np.full(n_states, -np.inf)
            for ni in range(n_states):
                if ni != ci and (players[pi], states[ci], states[ni]) in forbidden_proposals:
                    continue
                pa = approval_pass[pi, ci, ni] if ni != ci else 1.0
                evs[ni] = pa * V_array[ni, pi] + (1.0 - pa) * V_array[ci, pi]
            best = np.max(evs)
            for ni in range(n_states):
                if proposal_probs[pi, ci, ni] > prob_tol and evs[ni] < best - atol:
                    return False, f"proposal: {players[pi]} in {states[ci]} puts mass on non-argmax {states[ni]}", {
                        "type": "proposal", "proposer": players[pi],
                        "current": states[ci], "next": states[ni],
                        "ev": float(evs[ni]), "best": float(best)}
    return True, "All tests passed (fast).", None
