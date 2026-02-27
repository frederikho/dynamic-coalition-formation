from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from lib.equilibrium.find import setup_experiment, _compute_verification
from lib.equilibrium.solver import EquilibriumSolver
from lib.utils import verify_equilibrium


@dataclass
class RunResult:
    scenario_name: str
    runtime_seconds: float
    verification_success: bool
    verification_message: str
    stopping_reason: str
    score: float
    random_seed: int
    solver_result: Dict[str, Any]


def _score_run(runtime_seconds: float,
               verification_success: bool,
               max_time_seconds: Optional[float],
               penalty_factor: float) -> float:
    if verification_success:
        return runtime_seconds
    if max_time_seconds is None:
        return runtime_seconds * penalty_factor
    return max_time_seconds * penalty_factor


def run_single(config: Dict[str, Any],
               solver_params: Dict[str, Any],
               max_time_seconds: Optional[float],
               penalty_factor: float = 1.5,
               random_seed: Optional[int] = None) -> RunResult:
    """
    Run a single scenario with given solver params and compute verification.

    This is side-effect free: no checkpoints, no file writing, no logging.
    """
    setup = setup_experiment(config)

    solver = EquilibriumSolver(
        players=setup['players'],
        states=setup['state_names'],
        effectivity=setup['effectivity'],
        protocol=setup['protocol'],
        payoffs=setup['payoffs'],
        discounting=setup['discounting'],
        unanimity_required=setup['unanimity_required'],
        verbose=False,
        random_seed=random_seed,
        logger=None
    )

    t0 = time.time()
    strategy_df, solver_result = solver.solve(
        **solver_params,
        checkpoint_dir='./checkpoints',
        load_from_checkpoint=False,
        config_hash=None,
        max_time_seconds=max_time_seconds,
        disable_checkpoints=True,
        disable_timing_report=True
    )
    runtime_seconds = time.time() - t0

    stopping_reason = solver_result.get('stopping_reason', 'unknown')

    verification_success = False
    verification_message = "Skipped verification"

    if stopping_reason != 'time_budget':
        strategy_df_filled = strategy_df.copy()
        strategy_df_filled.fillna(0.0, inplace=True)

        V, P, P_proposals, P_approvals = _compute_verification(strategy_df_filled, setup)
        result = {
            'V': V,
            'P': P,
            'P_proposals': P_proposals,
            'P_approvals': P_approvals,
            'players': setup['players'],
            'state_names': setup['state_names'],
            'effectivity': setup['effectivity'],
            'strategy_df': strategy_df_filled,
        }
        verification_success, verification_message = verify_equilibrium(result)

    score = _score_run(runtime_seconds, verification_success, max_time_seconds, penalty_factor)

    return RunResult(
        scenario_name=config.get('scenario_name', 'unknown'),
        runtime_seconds=runtime_seconds,
        verification_success=verification_success,
        verification_message=verification_message,
        stopping_reason=stopping_reason,
        score=score,
        random_seed=solver.random_seed,
        solver_result=solver_result
    )
