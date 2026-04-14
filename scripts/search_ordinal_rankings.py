#!/usr/bin/env python3
"""Thin wrapper for searching ordinal rankings using the library implementation."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.solver import EquilibriumSolver
from lib.equilibrium.ordinal_ranking import solve_with_ordinal_ranking_n3
from lib.equilibrium.ordinal_ranking.ranking_orders import (
    _generate_weak_orders,
    _compute_absorbing_pruning_masks,
)
from lib.equilibrium.ordinal_ranking.weak_equality import _NEWTON_GUESS_LIMIT
from lib.equilibrium.ordinal_ranking.numba_loops import _NEWTON_MAX_ITERS
from lib.equilibrium.lcs import compute_lccs
from lib.utils import get_approval_committee
from lib.equilibrium.find import (
    setup_experiment,
    _infer_or_parse_players_from_payoff_table,
    _infer_players_from_payoff_table,
)
from lib.equilibrium.scenarios import get_scenario, fill_players
from lib.equilibrium.excel_writer import write_strategy_table_excel
from lib.verify_cli import _run_verification


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_payoff_config(
    scenario_name: str,
    payoff_table: str,
    effectivity_rule: str | None = None,
    allow_non_canonical_states: bool = False,
) -> dict:
    config = get_scenario(scenario_name)
    config["payoff_table"] = payoff_table
    if config.get("players") is None:
        players = _infer_or_parse_players_from_payoff_table(Path(payoff_table))
        config = fill_players(config, players)
    if effectivity_rule:
        config["effectivity_rule"] = effectivity_rule
    if allow_non_canonical_states:
        config["allow_non_canonical_states"] = True
    return config


def _build_inferred_payoff_config(
    payoff_path: Path,
    effectivity_rule: str | None = None,
    allow_non_canonical_states: bool = False,
) -> dict:
    players = _infer_players_from_payoff_table(payoff_path)
    uniform = 1.0 / len(players)
    zeros = {p: 0.0 for p in players}
    config = {
        "scenario_name": f"ordinal_search_{payoff_path.stem}",
        "players": players,
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "discounting": 0.99,
        "unanimity_required": True,
        "protocol": {p: uniform for p in players},
        "base_temp": zeros.copy(),
        "ideal_temp": zeros.copy(),
        "delta_temp": zeros.copy(),
        "m_damage": {p: 1.0 for p in players},
        "power": {p: uniform for p in players},
        "payoff_table": str(payoff_path),
    }
    if effectivity_rule:
        config["effectivity_rule"] = effectivity_rule
    if allow_non_canonical_states:
        config["allow_non_canonical_states"] = True
    return config


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _format_ranking(states: list[str], perm: np.ndarray) -> str:
    return " > ".join(states[int(idx)] for idx in perm)


def _format_weak_order(states: list[str], tiers: np.ndarray) -> str:
    groups: dict[int, list[str]] = {}
    for state_idx, tier in enumerate(tiers):
        groups.setdefault(int(tier), []).append(states[int(state_idx)])
    ordered = [" = ".join(groups[t]) for t in sorted(groups)]
    return " > ".join(ordered)


def _verify_via_cli(
    config: dict,
    strategy_df: Any,
    solver: EquilibriumSolver,
    effectivity_rule: str,
) -> tuple[bool, str]:
    """Write strategy to a temp file and run the full CLI verifier on it."""
    with tempfile.NamedTemporaryFile(prefix="ordinal_candidate_", suffix=".xlsx", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        meta: dict = {
            "payoff_source": "precomputed_table",
            "payoff_table": str(config.get("payoff_table", "")),
            "power_rule": config.get("power_rule", "power_threshold"),
            "effectivity_rule": effectivity_rule,
            "unanimity_required": config.get("unanimity_required", True),
            "discounting": config.get("discounting", 0.99),
        }
        if config.get("power_rule") == "power_threshold":
            meta["min_power"] = config.get("min_power", 0.501)
        for player in solver.players:
            for field in ("base_temp", "ideal_temp", "delta_temp", "m_damage", "power", "protocol"):
                val = config.get(field, {})
                meta[f"{field}_{player}"] = val.get(player, 0.0) if isinstance(val, dict) else 0.0
        write_strategy_table_excel(
            df=strategy_df,
            excel_file_path=str(temp_path),
            players=solver.players,
            effectivity=solver.effectivity,
            states=solver.states,
            metadata=meta,
            value_functions=solver.value_functions,
            static_payoffs=solver.payoffs,
            transition_matrix=solver.transition_matrix,
        )
        ok, msg, _ = _run_verification(temp_path, effectivity_rule=effectivity_rule)
        return ok, msg
    except Exception as exc:
        return False, f"verifier unavailable: {exc}"
    finally:
        temp_path.unlink(missing_ok=True)


def _resolve_output_path(payoff_path: Path, write_output: str | None) -> Path:
    if write_output:
        return Path(write_output)
    return REPO_ROOT / "strategy_tables" / f"ordinal_{payoff_path.stem}.xlsx"


def _print_solver_execution_summary(
    *,
    result: dict,
    all_successes: list[dict],
    n_hits: int,
    tested: int,
    n_solver_calls: int,
    output_dir: str | None,
    dedup_by: str,
):
    flow = result.get("weak_solver_flow_stats", {}) or {}
    ec = result.get("exit_stats_counts")

    # Terminal exit outcomes (exclude nb_skip, which is a Newton side-event).
    terminal_outcomes = {
        "nb_newton_hit": 0,
        "success_scipy": 0,
        "conv_badresid": 0,
        "maxfev": 0,
        "xtol": 0,
        "bad_progress": 0,
        "exception": 0,
        "nb_skip": 0,
    }
    if ec is not None and np.any(ec):
        totals = ec.sum(axis=0)
        terminal_outcomes = {
            "nb_newton_hit": int(totals[0]),
            "success_scipy": int(totals[1]),
            "conv_badresid": int(totals[2]),
            "maxfev": int(totals[3]),
            "xtol": int(totals[4]),
            "bad_progress": int(totals[5]),
            "exception": int(totals[6]),
            "nb_skip": int(totals[7]),
        }

    newton_attempted = int(flow.get("newton_attempted", 0))
    newton_converged = int(flow.get("newton_converged", 0))
    newton_iters_on_converged = int(flow.get("newton_iters_on_converged", 0))
    newton_progress_seeded = int(flow.get("newton_progress_seeded", 0))
    newton_no_progress = int(flow.get("newton_no_progress", 0))

    scipy_attempted = int(flow.get("scipy_attempted", 0))
    scipy_success_flag = int(flow.get("scipy_success_flag", 0))
    scipy_nfev_total = int(flow.get("scipy_nfev_total", 0))
    scipy_unsuccessful = int(flow.get("scipy_unsuccessful", 0))
    scipy_exception = int(flow.get("scipy_exception", 0))

    success_newton_iters = int(flow.get("success_newton_iters", 0))
    success_newton_exit_res = float(flow.get("success_newton_exit_res_e9", 0)) / 1e9
    success_n_free = int(flow.get("success_n_free", 0))

    final_residual_success = int(flow.get("final_residual_success", 0))
    final_residual_fail = int(flow.get("final_residual_fail", 0))
    final_valid_from_newton = int(flow.get("final_valid_from_newton", 0))
    final_valid_from_scipy = int(flow.get("final_valid_from_scipy", 0))
    final_valid_from_scipy_seeded = int(flow.get("final_valid_from_scipy_seeded", 0))
    final_valid_from_scipy_unseeded = int(flow.get("final_valid_from_scipy_unseeded", 0))
    final_invalid_from_newton = int(flow.get("final_invalid_from_newton", 0))
    final_invalid_from_scipy = int(flow.get("final_invalid_from_scipy", 0))
    final_invalid_from_scipy_seeded = int(flow.get("final_invalid_from_scipy_seeded", 0))
    final_invalid_from_scipy_unseeded = int(flow.get("final_invalid_from_scipy_unseeded", 0))
    weak_payload_returned = int(result.get("weak_payload_returned", 0))
    weak_payload_verified_true = int(result.get("weak_payload_verified_true", 0))
    weak_payload_verified_false = int(result.get("weak_payload_verified_false", 0))
    
    # Split finalize verify results by solver path
    finalize_verify_true_from_newton = int(flow.get("finalize_verify_true_from_newton", 0))
    finalize_verify_true_from_scipy = int(flow.get("finalize_verify_true_from_scipy", 0))
    finalize_verify_true_from_scipy_seeded = int(flow.get("finalize_verify_true_from_scipy_seeded", 0))
    finalize_verify_true_from_scipy_unseeded = int(flow.get("finalize_verify_true_from_scipy_unseeded", 0))
    finalize_verify_false_from_newton = int(flow.get("finalize_verify_false_from_newton", 0))
    finalize_verify_false_from_scipy = int(flow.get("finalize_verify_false_from_scipy", 0))
    finalize_verify_false_from_scipy_seeded = int(flow.get("finalize_verify_false_from_scipy_seeded", 0))
    finalize_verify_false_from_scipy_unseeded = int(flow.get("finalize_verify_false_from_scipy_unseeded", 0))

    source_counts = {"canonical": 0, "weak_equality_solve": 0, "unknown": 0}
    for success in all_successes:
        src = str(success.get("source", "unknown"))
        if src in source_counts:
            source_counts[src] += 1
        else:
            source_counts["unknown"] += 1

    manifest_rows = result.get("manifest", []) if output_dir else []
    written_unique = len(manifest_rows)
    dedup_dropped = max(0, n_hits - written_unique) if output_dir else 0

    solver_call_pct = (100.0 * n_solver_calls / tested) if tested > 0 else 0.0
    t_solver = result.get("t_solver", 0.0)
    t_nb = result.get("t_solver_nb_newton", 0.0)
    t_root = result.get("t_solver_root", 0.0)
    nb_pct = (100.0 * t_nb / t_solver) if t_solver > 0 else 0.0
    root_pct = (100.0 * t_root / t_solver) if t_solver > 0 else 0.0

    print()
    print("SOLVER EXECUTION SUMMARY")
    print("=" * 80)
    print("Input:")
    print(f"  - Combinations tested:        {tested:>12,d}")
    print(f"  - Solver calls triggered:     {n_solver_calls:>12,d} ({solver_call_pct:4.1f}%)")
    print()
    print("Process Flow:")
    print(f"  - Newton's Method ({nb_pct:4.1f}% of solver time)")
    print(f"    - Attempted guesses:        {newton_attempted:>12,d}")
    if newton_converged > 0:
        avg_iters = newton_iters_on_converged / newton_converged
        print(f"    - Converged (Newton):       {newton_converged:>12,d}  (avg iters: {avg_iters:.1f})")
    else:
        print(f"    - Converged (Newton):       {newton_converged:>12,d}")
    print(f"    - No progress (nb_skip):    {newton_no_progress:>12,d}")
    print(f"    - Made progress (seed):     {newton_progress_seeded:>12,d}")
    print()
    print(f"  - Scipy Root Search ({root_pct:4.1f}% of solver time)")
    print(f"    - Attempted:                {scipy_attempted:>12,d}")
    avg_nfev = (scipy_nfev_total / scipy_success_flag) if scipy_success_flag > 0 else 0.0
    print(f"    - success flag=True:        {scipy_success_flag:>12,d}  (avg nfev on success: {avg_nfev:.0f})")
    print(f"    - success flag=False:       {scipy_unsuccessful:>12,d}")
    print("      - Failed outcomes (subset of success flag=False):")
    print(f"        - bad_progress:         {terminal_outcomes['bad_progress']:>12,d}")
    print(f"        - maxfev:               {terminal_outcomes['maxfev']:>12,d}")
    print(f"        - xtol:                 {terminal_outcomes['xtol']:>12,d}")
    print(f"    - Exceptions (try/except):  {scipy_exception:>12,d}")
    print(f"      - exception (exit table): {terminal_outcomes['exception']:>12,d}")
    print()
    print("Residual Gate (r < 1e-7):")
    print(f"  - Passed residual gate:       {final_residual_success:>12,d}")
    print(f"    - via Newton path:          {final_valid_from_newton:>12,d}")
    print(f"    - via Scipy path:           {final_valid_from_scipy:>12,d}")
    print(f"      - Scipy seeded by Newton: {final_valid_from_scipy_seeded:>12,d}")
    print(f"      - Scipy unseeded:         {final_valid_from_scipy_unseeded:>12,d}")
    print(f"  - Failed residual gate:       {final_residual_fail:>12,d}")
    print(f"    - Newton-converged invalid: {final_invalid_from_newton:>12,d}")
    print(f"    - Scipy-converged invalid:  {final_invalid_from_scipy:>12,d}")
    print(f"      - Scipy seeded invalid:   {final_invalid_from_scipy_seeded:>12,d}")
    print(f"      - Scipy unseeded invalid: {final_invalid_from_scipy_unseeded:>12,d}")
    print(f"    - conv_badresid (table):    {terminal_outcomes['conv_badresid']:>12,d}")
    print()
    print("Verification and Dedup Pipeline:")
    print(f"  - Residual-pass candidates:   {final_residual_success:>12,d}")
    print(f"  - Entered finalize step:      {weak_payload_returned:>12,d}")
    print(f"    - finalize verify=True:     {weak_payload_verified_true:>12,d}")
    print(f"      - from Newton path:       {finalize_verify_true_from_newton:>12,d}")
    print(f"      - from Scipy path:        {finalize_verify_true_from_scipy:>12,d}")
    print(f"        - Scipy seeded:         {finalize_verify_true_from_scipy_seeded:>12,d}")
    print(f"        - Scipy unseeded:       {finalize_verify_true_from_scipy_unseeded:>12,d}")
    print(f"    - finalize verify=False:    {weak_payload_verified_false:>12,d}")
    print(f"      - from Newton path:       {finalize_verify_false_from_newton:>12,d}")
    print(f"      - from Scipy path:        {finalize_verify_false_from_scipy:>12,d}")
    print(f"        - Scipy seeded:         {finalize_verify_false_from_scipy_seeded:>12,d}")
    print(f"        - Scipy unseeded:       {finalize_verify_false_from_scipy_unseeded:>12,d}")
    print(f"  - Verified successes:         {n_hits:>12,d}")
    print(f"    - Canonical verify path:    {source_counts['canonical']:>12,d}")
    print(f"    - Weak-equality path:       {source_counts['weak_equality_solve']:>12,d}")
    print(f"    - Unknown source label:     {source_counts['unknown']:>12,d}")
    if final_residual_success > 0:
        print()
        print("Newton Diagnostics at Successful Residual Gate (GPU sizing data):")
        print(f"  - n_free at success:          {success_n_free:>12,d}  (total across {final_residual_success} hits)")
        avg_free = success_n_free / final_residual_success if final_residual_success > 0 else 0
        print(f"  - avg n_free per success:     {avg_free:>12.1f}")
        print(f"  - Newton iters (total):       {success_newton_iters:>12,d}")
        avg_nb = success_newton_iters / final_residual_success if final_residual_success > 0 else 0
        print(f"  - avg Newton iters/success:   {avg_nb:>12.1f}  ← safe K for fixed-iter Newton")
        print(f"  - Newton exit residual (sum): {success_newton_exit_res:>12.6f}")
        avg_res = success_newton_exit_res / final_residual_success if final_residual_success > 0 else 0
        print(f"  - avg Newton exit residual:   {avg_res:>12.6f}  ← residual at SciPy handoff")
    if output_dir:
        print(f"  - Pre-dedup successes:        {n_hits:>12,d}")
        print(f"  - Written unique files:       {written_unique:>12,d}")
        print(f"  - Dedup dropped:              {dedup_dropped:>12,d} (dedup_by={dedup_by})")
    else:
        print("  - Dedup/write step:           not enabled (use --write-all)")


# ---------------------------------------------------------------------------
# LCCS pruning report printer
# ---------------------------------------------------------------------------

def _print_pruning_report(
    report: dict,
    players: list,
    states: list,
    payoff_array,
    total_original: int,
) -> None:
    print()
    print("LCCS Absorbing-State Pruning Analysis")
    print("=" * 80)
    absorbing = report["absorbing_state"]
    n_orig = report["n_orders_original"]
    total_pruned = report["total_after_pruning"]
    factor = report["reduction_factor"]
    pct_kept = 100.0 * total_pruned / max(total_original, 1)
    print(f"  Absorbing state (trusted from LCCS): {absorbing}")
    print(f"  Weak orders per player (original):   {n_orig:,d}")
    print()
    print(f"  {'Player':<10}  {'Valid':>8}  {'Pruned':>8}  {'% kept':>8}  Constraints")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  -----------")
    for pi, player in enumerate(players):
        pp = report["per_player"][pi]
        n_valid = pp["n_valid"]
        n_pruned = pp["n_pruned"]
        pct = 100.0 * n_valid / n_orig
        cs = pp["constraints"]
        if cs:
            c_strs = [
                f"{states[c['bi']]} ({c['rule']})" for c in cs
            ]
            c_label = "; ".join(c_strs)
        else:
            c_label = "(none)"
        print(f"  {player:<10}  {n_valid:>8,d}  {n_pruned:>8,d}  {pct:>7.1f}%  {c_label}")
    print()
    print(f"  Full search space:  {total_original:>15,d}")
    print(f"  After pruning:      {total_pruned:>15,d}")
    print(f"  Reduction factor:   {factor:>15.1f}×  ({pct_kept:.2f}% of original)")
    print()
    # Print payoff comparison for constrained pairs
    constraints = report["constraints"]
    if constraints:
        absorbing_idx = states.index(absorbing)
        print("  Payoff comparison (why constraints apply):")
        print(f"  {'Player':<10}  {'State B':<20}  {'u(B)':>12}  {'u(A)':>12}  {'Rule'}")
        print(f"  {'-'*10}  {'-'*20}  {'-'*12}  {'-'*12}  ------")
        seen = set()
        for c in constraints:
            key = (c["pi"], c["bi"])
            if key in seen:
                continue
            seen.add(key)
            pi, bi = c["pi"], c["bi"]
            u_b = float(payoff_array[bi, pi])
            u_a = float(payoff_array[absorbing_idx, pi])
            print(f"  {players[pi]:<10}  {states[bi]:<20}  {u_b:>12.4f}  {u_a:>12.4f}  {c['rule']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Search ordinal rankings (Thin Wrapper)")
    parser.add_argument("file", help="Payoff table path or basename")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument(
        "--allow-non-canonical-states",
        action="store_true",
        help="Allow non-canonical state names from reduced payoff tables.",
    )
    parser.add_argument(
        "--effectivity-rule",
        type=str,
        default=None,
        choices=("heyen_lehtomaa_2021", "unanimous_consent", "deployer_exit", "free_exit", "adjacent_step"),
    )
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weak-orders", action="store_true")
    parser.add_argument("--weak-equality-solve", action="store_true")
    parser.add_argument("--weak-equality-max-vars", type=int, default=None)
    parser.add_argument("--ranking-order", default="lexicographic",
                        choices=("lexicographic", "payoff", "random"))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--write-output", type=str, default=None,
                        help="Write first verified equilibrium to this path")
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--write-all-output-dir", type=str, default=None)
    parser.add_argument("--dedup-by", choices=("none", "transition", "strategy"), default="none")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--disable-newton", action="store_true",
                        help="Disable Newton's method in weak-equality solver (use only Scipy)")
    parser.add_argument("--use-broyden", action="store_true",
                        help="Use Broyden's Good Method instead of Newton for warm-up "
                             "(residual-only per iteration, no Jacobian recomputation)")
    parser.add_argument(
        "--prune-lccs",
        action="store_true",
        help=(
            "Automatically compute LCCS (weak), and if it identifies a unique absorbing state, "
            "apply safe pruning to reduce the weak-order search space. Two rules are used: "
            "Rule 1 (unilateral exit): if player i can unilaterally exit the absorbing state "
            "to state B and ui(B) < ui(absorbing), require tier_i[B] > tier_i[absorbing]; "
            "Rule 2 (strict dominance): if the absorbing state has strictly max payoff for "
            "player i over all states, all non-absorbing states rank below it. "
            "Both rules are safe: no valid equilibrium is discarded. "
            "Requires --weak-orders."
        ),
    )

    args = parser.parse_args()

    payoff_path = Path(args.file)
    if args.scenario:
        config = _build_payoff_config(
            args.scenario,
            str(payoff_path),
            effectivity_rule=args.effectivity_rule,
            allow_non_canonical_states=args.allow_non_canonical_states,
        )
        config_source = f"scenario:{args.scenario}"
    else:
        config = _build_inferred_payoff_config(
            payoff_path,
            effectivity_rule=args.effectivity_rule,
            allow_non_canonical_states=args.allow_non_canonical_states,
        )
        config_source = "inferred"

    setup = setup_experiment(config)
    players = setup["players"]
    states = setup["state_names"]
    n_states = len(states)

    # Compute total search space size for header
    if args.weak_orders:
        order_arrays = _generate_weak_orders(n_states)
        n_orders = order_arrays.shape[0]
    else:
        n_orders = math.factorial(n_states)
    total_triples = n_orders ** len(players)
    n_to_test = min(total_triples, args.max_combinations) if args.max_combinations else total_triples

    output_dir = args.write_all_output_dir
    if args.write_all and not output_dir:
        output_dir = str(REPO_ROOT / "strategy_tables" / f"ordinal_all_{payoff_path.stem}")

    # ── LCCS auto-pruning ─────────────────────────────────────────────────────
    lccs_absorbing_state: str | None = None
    if args.prune_lccs:
        if not args.weak_orders:
            print("ERROR: --prune-lccs requires --weak-orders")
            sys.exit(1)
        lccs_members, _ = compute_lccs(
            list(players), list(states),
            setup["payoffs"][list(players)], setup["effectivity"], weak=True,
        )
        if len(lccs_members) != 1:
            print(
                f"WARNING: --prune-lccs: LCCS (weak) contains {len(lccs_members)} states "
                f"{sorted(lccs_members)} — pruning requires a unique absorbing state; skipping."
            )
        else:
            lccs_absorbing_state = next(iter(lccs_members))
            # Build committee structure to evaluate pruning rules upfront
            payoff_array = setup["payoffs"].loc[states, players].to_numpy()
            player_idx_map = {p: i for i, p in enumerate(players)}
            committee_idxs_pre: list = []
            for proposer in players:
                proposer_rows = []
                for current_state in states:
                    row = []
                    for next_state in states:
                        committee = get_approval_committee(
                            setup["effectivity"], players, proposer, current_state, next_state
                        )
                        row.append(tuple(player_idx_map[p] for p in committee))
                    proposer_rows.append(row)
                committee_idxs_pre.append(proposer_rows)
            _, pruning_report_pre = _compute_absorbing_pruning_masks(
                order_arrays, payoff_array, list(states), list(players),
                committee_idxs_pre, lccs_absorbing_state,
            )
            _print_pruning_report(
                pruning_report_pre, list(players), list(states), payoff_array, total_triples,
            )

    # ── Header ───────────────────────────────────────────────────────────────
    print("Ordinal Ranking Verification Search")
    print("-" * 80)
    try:
        rel = payoff_path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = payoff_path
    print(f"file: {rel}")
    print(f"config_source: {config_source}")
    print(f"players: {players}")
    print(f"allow_non_canonical_states: {args.allow_non_canonical_states}")
    if args.effectivity_rule:
        print(f"effectivity_rule: {args.effectivity_rule}")
    print(f"total_ranking_triples: {total_triples:,d}")
    if lccs_absorbing_state:
        print(f"prune_lccs: absorbing={lccs_absorbing_state}")
    if args.max_combinations:
        print(f"max_combinations: {args.max_combinations:,d}")
    print(f"stop_on_success: {args.stop_on_success}")
    print(f"ranking_order: {args.ranking_order}")
    print(f"weak_orders: {args.weak_orders}")
    if args.weak_orders:
        print(f"weak_equality_solve: {args.weak_equality_solve}")
        if args.weak_equality_solve and args.weak_equality_max_vars is not None:
            print(f"weak_equality_max_vars: {args.weak_equality_max_vars}")
    if args.write_all or output_dir:
        print(f"write_all: True")
    if args.dedup_by != "none":
        print(f"dedup_by: {args.dedup_by}")
    print(f"workers: {args.workers}")
    print(f"shuffle: {args.shuffle}")
    if args.weak_equality_solve:
        print(f"newton_guess_limit: {_NEWTON_GUESS_LIMIT}")
        print(f"newton_max_iters: {_NEWTON_MAX_ITERS}")
        warm_up_method = "broyden" if args.use_broyden else ("newton" if not args.disable_newton else "none")
        print(f"warm_up_method: {warm_up_method}")
    print()

    # ── Search ───────────────────────────────────────────────────────────────
    _effectivity_rule = setup.get("effectivity_rule", "heyen_lehtomaa_2021")

    solver = EquilibriumSolver(
        players=players,
        states=states,
        effectivity=setup["effectivity"],
        protocol=setup["protocol"],
        payoffs=setup["payoffs"],
        discounting=setup["discounting"],
        unanimity_required=setup["unanimity_required"],
        power_rule=setup["power_rule"],
        forbidden_proposals=setup["forbidden_proposals"],
        effectivity_rule=_effectivity_rule,
    )

    extra_metadata: dict = {"effectivity_rule": _effectivity_rule}
    if config.get("power_rule") == "power_threshold":
        extra_metadata["min_power"] = config.get("min_power", 0.501)
    for player in players:
        for field in ("base_temp", "ideal_temp", "delta_temp", "m_damage", "power", "protocol"):
            val = config.get(field, {})
            extra_metadata[f"{field}_{player}"] = val.get(player, 0.0) if isinstance(val, dict) else 0.0

    df, result = solve_with_ordinal_ranking_n3(
        solver,
        max_combinations=args.max_combinations,
        workers=args.workers,
        weak_orders=args.weak_orders,
        weak_equality_solve=args.weak_equality_solve,
        weak_equality_max_vars=args.weak_equality_max_vars,
        ranking_order=args.ranking_order,
        progress_every=args.progress_every,
        stop_on_success=args.stop_on_success,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        write_all_dir=output_dir,
        dedup_by=args.dedup_by,
        payoff_path=payoff_path,
        use_newton=(not args.disable_newton),
        use_broyden=args.use_broyden,
        extra_metadata=extra_metadata,
        lccs_absorbing_state=lccs_absorbing_state,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    n_workers = result.get("n_workers", args.workers)
    wall_time = result.get("wall_time", 0.0)
    rate = result.get("rate", 0.0)
    t_numba = result.get("t_numba", 0.0)
    t_tie_struct = result.get("t_tie_struct", 0.0)
    t_solver = result.get("t_solver", 0.0)
    n_solver_calls = result.get("n_solver_calls", 0)
    n_skipped = result.get("n_skipped", 0)
    all_successes = result.get("all_successes", [])
    n_hits = len(all_successes)
    interrupted = result.get("interrupted", False)

    def _fmt_pct(val: float, base: float) -> str:
        return f" ({100.0 * val / base:5.1f}%)" if base > 0 else ""

    def _fmt_us(val: float) -> str:
        tested = result.get("tested", 0)
        if tested > 0:
            us = (val * 1_000_000) / tested
            return f" [{us:>9.1f} μs/combo]"
        return ""

    print("Summary")
    print("-" * 80)
    print(f"tested_combinations:  {result['tested']:>12,d}")
    print(f"wall_time:            {wall_time:>12.2f}s")
    print(f"rate:                 {rate:>12.0f}/s")
    if n_workers > 0:
        if t_numba > 0:
            msg = f"{t_numba / n_workers:>12.2f}s (avg per worker)"
            print(f"  - numba_time:       {msg:<35}{_fmt_us(t_numba)}")
        if t_tie_struct > 0:
            msg = f"{t_tie_struct / n_workers:>12.2f}s (avg per worker)"
            print(f"  - tie_struct_time:  {msg:<35}{_fmt_us(t_tie_struct)}")
        t_sv = t_solver / n_workers
        if t_sv > 0:
            msg = f"{t_sv:>12.2f}s (avg per worker)"
            print(f"  - solver_time:      {msg:<35}{_fmt_us(t_solver)}")
            t_nb_newton = result.get("t_solver_nb_newton", 0.0) / n_workers
            t_root = result.get("t_solver_root", 0.0) / n_workers
            t_root_v = result.get("t_solver_root_v_solve", 0.0) / n_workers
            t_root_p = result.get("t_solver_root_p_agg", 0.0) / n_workers
            t_root_m = result.get("t_solver_root_mapping", 0.0) / n_workers
            t_root_r = result.get("t_solver_root_residuals", 0.0) / n_workers
            t_fin = result.get("t_solver_finalize", 0.0) / n_workers
            t_fin_rb = result.get("t_solver_finalize_rebuild", 0.0) / n_workers
            t_fin_vf = result.get("t_solver_finalize_verify", 0.0) / n_workers
            t_fin_so = result.get("t_solver_finalize_solver_obj", 0.0) / n_workers
            t_chk = result.get("t_solver_check", 0.0) / n_workers
            t_setup = result.get("t_solver_setup", 0.0) / n_workers
            t_setup_cp = result.get("t_solver_setup_copy", 0.0) / n_workers
            t_setup_ix = result.get("t_solver_setup_indices", 0.0) / n_workers
            t_setup_nb = result.get("t_solver_setup_numba", 0.0) / n_workers
            t_setup_g = result.get("t_solver_setup_guesses", 0.0) / n_workers
            if t_nb_newton > 0:
                msg = f"{t_nb_newton:>12.2f}s{_fmt_pct(t_nb_newton, t_sv)}"
                print(f"    - nb_newton:      {msg:<35}{_fmt_us(result.get('t_solver_nb_newton', 0.0))}")
            if t_root > 0:
                msg = f"{t_root:>12.2f}s{_fmt_pct(t_root, t_sv)}"
                print(f"    - root_search:    {msg:<35}{_fmt_us(result.get('t_solver_root', 0.0))}")
                if t_root_v > 0:
                    msg = f"{t_root_v:>12.2f}s{_fmt_pct(t_root_v, t_sv)}"
                    print(f"      - V_solve:      {msg:<35}{_fmt_us(result.get('t_solver_root_v_solve', 0.0))}")
                    msg = f"{t_root_p:>12.2f}s{_fmt_pct(t_root_p, t_sv)}"
                    print(f"      - P_aggregate:  {msg:<35}{_fmt_us(result.get('t_solver_root_p_agg', 0.0))}")
                    msg = f"{t_root_m:>12.2f}s{_fmt_pct(t_root_m, t_sv)}"
                    print(f"      - mapping:      {msg:<35}{_fmt_us(result.get('t_solver_root_mapping', 0.0))}")
                    msg = f"{t_root_r:>12.2f}s{_fmt_pct(t_root_r, t_sv)}"
                    print(f"      - residuals:    {msg:<35}{_fmt_us(result.get('t_solver_root_residuals', 0.0))}")
            if t_fin > 0:
                msg = f"{t_fin:>12.2f}s{_fmt_pct(t_fin, t_sv)}"
                print(f"    - finalize:       {msg:<35}{_fmt_us(result.get('t_solver_finalize', 0.0))}")
                if t_fin_rb > 0:
                    msg = f"{t_fin_rb:>12.2f}s{_fmt_pct(t_fin_rb, t_sv)}"
                    print(f"      - rebuild:      {msg:<35}{_fmt_us(result.get('t_solver_finalize_rebuild', 0.0))}")
                    msg = f"{t_fin_vf:>12.2f}s{_fmt_pct(t_fin_vf, t_sv)}"
                    print(f"      - verify:       {msg:<35}{_fmt_us(result.get('t_solver_finalize_verify', 0.0))}")
                    msg = f"{t_fin_so:>12.2f}s{_fmt_pct(t_fin_so, t_sv)}"
                    print(f"      - solver_obj:   {msg:<35}{_fmt_us(result.get('t_solver_finalize_solver_obj', 0.0))}")
            if t_chk > 0:
                msg = f"{t_chk:>12.2f}s{_fmt_pct(t_chk, t_sv)}"
                print(f"    - check:          {msg:<35}{_fmt_us(result.get('t_solver_check', 0.0))}")
            if t_setup > 0:
                msg = f"{t_setup:>12.2f}s{_fmt_pct(t_setup, t_sv)}"
                print(f"    - setup/other:    {msg:<35}{_fmt_us(result.get('t_solver_setup', 0.0))}")
                if t_setup_cp > 0:
                    msg = f"{t_setup_cp:>12.2f}s{_fmt_pct(t_setup_cp, t_sv)}"
                    print(f"      - copy_arrays:  {msg:<35}{_fmt_us(result.get('t_solver_setup_copy', 0.0))}")
                    msg = f"{t_setup_ix:>12.2f}s{_fmt_pct(t_setup_ix, t_sv)}"
                    print(f"      - find_indices: {msg:<35}{_fmt_us(result.get('t_solver_setup_indices', 0.0))}")
                    msg = f"{t_setup_nb:>12.2f}s{_fmt_pct(t_setup_nb, t_sv)}"
                    print(f"      - build_numba:  {msg:<35}{_fmt_us(result.get('t_solver_setup_numba', 0.0))}")
                    msg = f"{t_setup_g:>12.2f}s{_fmt_pct(t_setup_g, t_sv)}"
                    print(f"      - guesses:      {msg:<35}{_fmt_us(result.get('t_solver_setup_guesses', 0.0))}")
    if args.weak_equality_solve:
        print(f"total_solver_calls:   {n_solver_calls:>12,d}")
        print(f"skipped_max_vars:     {n_skipped:>12,d}")
        hist = result.get("n_free_histogram", {})
        calls_by_nf = result.get("solver_calls_by_n_free", {}) or {}
        time_by_nf = result.get("solver_time_by_n_free", {}) or {}
        
        if hist:
            # Build combined bins with distribution + timing
            bins: dict[int, dict] = {}
            for nf, cnt in hist.items():
                b = (nf // 10) * 10
                if b not in bins:
                    bins[b] = {"count": 0, "calls": 0, "time": 0.0}
                bins[b]["count"] += cnt
            
            for nf_raw, calls_raw in calls_by_nf.items():
                nf = int(nf_raw)
                calls = int(calls_raw)
                b = (nf // 10) * 10
                if b not in bins:
                    bins[b] = {"count": 0, "calls": 0, "time": 0.0}
                bins[b]["calls"] += calls
                bins[b]["time"] += float(time_by_nf.get(nf_raw, 0.0))
            
            print("\nFree Variables Distribution and Solver Cost")
            print("  bin(vars)         count       calls   total_time(s)   avg_ms/call")
            for b in sorted(bins.keys()):
                cnt = bins[b]["count"]
                calls = bins[b]["calls"]
                ttot = bins[b]["time"]
                avg_ms = (ttot / calls * 1000.0) if calls > 0 else 0.0
                bin_str = f"{b:>2d}-{b+9:<9d}"
                print(f"  {bin_str} {cnt:>10,d} {calls:>10,d} {ttot:>14.2f} {avg_ms:>13.2f}")
            
            if args.weak_equality_max_vars is not None:
                max_v = args.weak_equality_max_vars
                skipped_nfs = [nf for nf in hist.keys() if nf > max_v]
                if skipped_nfs:
                    print(f"Note: Combinations with > {max_v} free variables were skipped (range {min(skipped_nfs)}-{max(skipped_nfs)}).")

        ec = result.get("exit_stats_counts")
        if ec is not None and np.any(ec):
            _outcome_labels = ["nb_newton_hit", "success(scipy)", "conv_badresid",
                               "maxfev", "xtol", "bad_progress", "exception", "nb_skip"]
            print("\nSolver Exit Reasons (rows=guess_idx, cols=outcome)")
            header = f"  {'guess':>5s}" + "".join(f"  {lbl:>13s}" for lbl in _outcome_labels)
            print(header)
            for gi in range(ec.shape[0]):
                if np.any(ec[gi]):
                    row_str = f"  {gi:>5d}" + "".join(f"  {int(ec[gi, oi]):>13,d}" for oi in range(ec.shape[1]))
                    print(row_str)
        _print_solver_execution_summary(
            result=result,
            all_successes=all_successes,
            n_hits=n_hits,
            tested=result["tested"],
            n_solver_calls=n_solver_calls,
            output_dir=output_dir,
            dedup_by=args.dedup_by,
        )
    print(f"interrupted:          {str(interrupted):>12s}")
    print(f"verified_successes:   {n_hits:>12,d}")
    if all_successes:
        n_frees = [s.get("n_free", 0) for s in all_successes]
        if len(n_frees) > 1:
            print(f"  - free_vars:        {', '.join(map(str, n_frees))}")
        else:
            print(f"  - free_vars:        {n_frees[0]}")
    print(f"stored_successes:     {n_hits:>12,d}")
    if output_dir and all_successes:
        manifest_rows = result.get("manifest", [])
        written = len(manifest_rows)
        skipped_dedup = n_hits - written
        print(f"  write_all done: {written} files, {skipped_dedup} skipped (dedup_by={args.dedup_by}), manifest rows={written}")
        print(f"written_successes:    {written:>12,d}")
        print(f"write_all_dir:        {output_dir}")
        print(f"manifest:             {Path(output_dir) / 'manifest.csv'}")
    if config.get("payoff_table"):
        print(f"Payoffs: loaded from '{Path(config['payoff_table']).name}'")

    # ── First Verified Success ────────────────────────────────────────────────
    if all_successes:
        first = result.get("first_success") or all_successes[0]
        rankings = first["rankings"]  # tuple of np.ndarray, one per player

        cli_ok, cli_msg = _verify_via_cli(
            config=config,
            strategy_df=df,
            solver=solver,
            effectivity_rule=args.effectivity_rule or "heyen_lehtomaa_2021",
        )

        print()
        print("First Verified Success")
        print("-" * 80)
        states_list = list(states)
        for player_idx, player in enumerate(players):
            ranking = rankings[player_idx]
            if args.weak_orders:
                print(f"{player}: {_format_weak_order(states_list, ranking)}")
            else:
                print(f"{player}: {_format_ranking(states_list, ranking)}")
        print(f"cli_verification: {cli_ok}")
        print(f"cli_message: {cli_msg}")

        if cli_ok:
            out_path = _resolve_output_path(payoff_path, args.write_output)
            meta: dict = {
                "payoff_source": "precomputed_table",
                "payoff_table": str(config.get("payoff_table", "")),
                "power_rule": config.get("power_rule", "power_threshold"),
                "unanimity_required": config.get("unanimity_required", True),
                "discounting": config.get("discounting", 0.99),
            }
            write_strategy_table_excel(
                df=df,
                excel_file_path=str(out_path),
                players=players,
                effectivity=setup["effectivity"],
                states=states,
                metadata=meta,
                value_functions=solver.value_functions,
                static_payoffs=solver.payoffs,
                transition_matrix=solver.transition_matrix,
            )
            print(f"written: {out_path}")

        print()
        print("Transition Matrix")
        print("-" * 80)
        P = solver.transition_matrix
        print(P.loc[states, states].to_string(float_format=lambda x: f"{x:.6f}"))
        print()
        print("Value Functions")
        print("-" * 80)
        V = solver.value_functions
        print(V.loc[states, players].to_string(float_format=lambda x: f"{x:.6f}"))
    else:
        print("\nNo equilibrium found.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        # Final flush and immediate exit to bypass problematic atexit hangs.
        sys.stdout.flush()
        sys.stderr.flush()
        import os
        os._exit(0)
