"""
Multi-start equilibrium search.

Calls run_vfi() repeatedly with randomised V^(0) and varied proposer-probability
vectors rho.  Deduplicates results by max-abs distance between value functions.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from lib.equilibrium.mip_vfi.vfi import run_vfi


def _make_rho_variants(n_players: int, rho_favour: float = 0.5) -> List[np.ndarray]:
    """
    Return: [uniform] + [player-i-favoured for each i].
    """
    variants = [np.full(n_players, 1.0 / n_players)]
    rest = (1.0 - rho_favour) / max(n_players - 1, 1)
    for i in range(n_players):
        rho = np.full(n_players, rest)
        rho[i] = rho_favour
        variants.append(rho)
    return variants


def _is_duplicate(
    V_new: np.ndarray,
    found: List[np.ndarray],
    atol: float,
) -> bool:
    for V_prev in found:
        if np.max(np.abs(V_new - V_prev)) <= atol:
            return True
    return False


def find_equilibria(
    payoffs: np.ndarray,
    approval_structure,
    forbidden: np.ndarray,
    rho_base: np.ndarray,
    delta: float,
    n_restarts: int = 40,
    rho_variants: Optional[List[np.ndarray]] = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    cycle_window: int = 8,
    mip_eps: float = 1e-8,
    mip_time_limit: float = 30.0,
    atol_dedup: float = 0.01,
    seed: int = 0,
    logger=None,
) -> List[Dict]:
    """
    Multi-start search for multiple SMPE.

    For each rho variant and each random V^(0) draw, runs VFI and collects
    distinct equilibria (deduplicated by max|V^a - V^b| <= atol_dedup).

    Args:
        payoffs:       (n_states, n_players) static payoff matrix.
        approval_structure: approval committee structure.
        forbidden:     (n_players, n_states) forbidden proposal mask.
        rho_base:      base proposer probability vector (used as first variant
                       if rho_variants is None).
        delta:         discount factor.
        n_restarts:    random V^(0) draws per rho variant.
        rho_variants:  explicit list of rho vectors; if None builds uniform +
                       player-favoured variants automatically.
        tol, max_iter, cycle_window, mip_eps, mip_time_limit: VFI parameters.
        atol_dedup:    deduplication tolerance on V.
        seed:          numpy random seed.
        logger:        optional logger.

    Returns:
        List of VFI result dicts (each passing verification by caller).
    """

    def _log(msg):
        if logger:
            logger.info(msg)

    n_states, n_players = payoffs.shape
    rng = np.random.RandomState(seed)
    sigma_pi = float(np.max(np.abs(payoffs))) or 1.0

    if rho_variants is None:
        rho_variants = _make_rho_variants(n_players)

    found_Vs: List[np.ndarray] = []
    found_results: List[Dict] = []

    total_runs = len(rho_variants) * (1 + n_restarts)
    run_idx = 0

    for rho in rho_variants:
        # First run: initialise from raw payoffs (deterministic)
        V_inits = [payoffs.copy()]
        # Then n_restarts noisy variants
        for _ in range(n_restarts):
            noise = rng.normal(0.0, sigma_pi, size=payoffs.shape)
            V_inits.append(payoffs + noise)

        for V_init in V_inits:
            run_idx += 1
            _log(f"  Multi-start run {run_idx}/{total_runs}")
            result = run_vfi(
                payoffs=payoffs,
                approval_structure=approval_structure,
                forbidden=forbidden,
                rho=rho,
                delta=delta,
                V_init=V_init,
                tol=tol,
                max_iter=max_iter,
                cycle_window=cycle_window,
                mip_eps=mip_eps,
                mip_time_limit=mip_time_limit,
                logger=logger,
            )

            if not result["success"]:
                continue

            V_found = result["V"]
            if _is_duplicate(V_found, found_Vs, atol_dedup):
                _log(f"    Duplicate equilibrium (skipped).")
                continue

            found_Vs.append(V_found)
            result["rho"] = rho.copy()
            found_results.append(result)
            _log(f"    New equilibrium found! Total: {len(found_results)}")

    _log(f"Multi-start complete: {len(found_results)} distinct equilibria found.")
    return found_results
