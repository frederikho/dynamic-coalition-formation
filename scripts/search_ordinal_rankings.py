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
from lib.equilibrium.ordinal_ranking.ranking_orders import _generate_weak_orders
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
            "payoff_table": config.get("payoff_table", ""),
            "power_rule": config.get("power_rule", "power_threshold"),
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
        choices=("heyen_lehtomaa_2021", "unanimous_consent", "deployer_exit", "free_exit"),
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
    print()

    # ── Search ───────────────────────────────────────────────────────────────
    solver = EquilibriumSolver(
        players=players,
        states=states,
        effectivity=setup["effectivity"],
        protocol=setup["protocol"],
        payoffs=setup["payoffs"],
        discounting=setup["discounting"],
        unanimity_required=setup["unanimity_required"],
        forbidden_proposals=setup["forbidden_proposals"],
    )

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
            if us >= 100:
                return f" [{us:>8.0f} μs/combo]"
            elif us >= 1:
                return f" [{us:>8.1f} μs/combo]"
            else:
                return f" [{us:>8.2f} μs/combo]"
        return ""

    print("Summary")
    print("-" * 80)
    print(f"tested_combinations:  {result['tested']:>12,d}")
    print(f"wall_time:            {wall_time:>12.2f}s")
    print(f"rate:                 {rate:>12.0f}/s")
    if n_workers > 0:
        if t_numba > 0:
            print(f"  - numba_time:       {t_numba / n_workers:>12.2f}s (avg per worker){_fmt_us(t_numba)}")
        if t_tie_struct > 0:
            print(f"  - tie_struct_time:  {t_tie_struct / n_workers:>12.2f}s (avg per worker){_fmt_us(t_tie_struct)}")
        t_sv = t_solver / n_workers
        if t_sv > 0:
            print(f"  - solver_time:      {t_sv:>12.2f}s (avg per worker){_fmt_us(t_solver)}")
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
                print(f"    - nb_newton:      {t_nb_newton:>12.2f}s{_fmt_pct(t_nb_newton, t_sv)}{_fmt_us(result.get('t_solver_nb_newton', 0.0))}")
            if t_root > 0:
                print(f"    - root_search:    {t_root:>12.2f}s{_fmt_pct(t_root, t_sv)}{_fmt_us(result.get('t_solver_root', 0.0))}")
                if t_root_v > 0:
                    print(f"      - V_solve:      {t_root_v:>12.2f}s{_fmt_pct(t_root_v, t_sv)}{_fmt_us(result.get('t_solver_root_v_solve', 0.0))}")
                    print(f"      - P_aggregate:  {t_root_p:>12.2f}s{_fmt_pct(t_root_p, t_sv)}{_fmt_us(result.get('t_solver_root_p_agg', 0.0))}")
                    print(f"      - mapping:      {t_root_m:>12.2f}s{_fmt_pct(t_root_m, t_sv)}{_fmt_us(result.get('t_solver_root_mapping', 0.0))}")
                    print(f"      - residuals:    {t_root_r:>12.2f}s{_fmt_pct(t_root_r, t_sv)}{_fmt_us(result.get('t_solver_root_residuals', 0.0))}")
            if t_fin > 0:
                print(f"    - finalize:       {t_fin:>12.2f}s{_fmt_pct(t_fin, t_sv)}{_fmt_us(result.get('t_solver_finalize', 0.0))}")
                if t_fin_rb > 0:
                    print(f"      - rebuild:      {t_fin_rb:>12.2f}s{_fmt_pct(t_fin_rb, t_sv)}{_fmt_us(result.get('t_solver_finalize_rebuild', 0.0))}")
                    print(f"      - verify:       {t_fin_vf:>12.2f}s{_fmt_pct(t_fin_vf, t_sv)}{_fmt_us(result.get('t_solver_finalize_verify', 0.0))}")
                    print(f"      - solver_obj:   {t_fin_so:>12.2f}s{_fmt_pct(t_fin_so, t_sv)}{_fmt_us(result.get('t_solver_finalize_solver_obj', 0.0))}")
            if t_chk > 0:
                print(f"    - check:          {t_chk:>12.2f}s{_fmt_pct(t_chk, t_sv)}{_fmt_us(result.get('t_solver_check', 0.0))}")
            if t_setup > 0:
                print(f"    - setup/other:    {t_setup:>12.2f}s{_fmt_pct(t_setup, t_sv)}{_fmt_us(result.get('t_solver_setup', 0.0))}")
                if t_setup_cp > 0:
                    print(f"      - copy_arrays:  {t_setup_cp:>12.2f}s{_fmt_pct(t_setup_cp, t_sv)}{_fmt_us(result.get('t_solver_setup_copy', 0.0))}")
                    print(f"      - find_indices: {t_setup_ix:>12.2f}s{_fmt_pct(t_setup_ix, t_sv)}{_fmt_us(result.get('t_solver_setup_indices', 0.0))}")
                    print(f"      - build_numba:  {t_setup_nb:>12.2f}s{_fmt_pct(t_setup_nb, t_sv)}{_fmt_us(result.get('t_solver_setup_numba', 0.0))}")
                    print(f"      - guesses:      {t_setup_g:>12.2f}s{_fmt_pct(t_setup_g, t_sv)}{_fmt_us(result.get('t_solver_setup_guesses', 0.0))}")
    if args.weak_equality_solve:
        print(f"total_solver_calls:   {n_solver_calls:>12,d}")
        print(f"skipped_max_vars:     {n_skipped:>12,d}")
        hist = result.get("n_free_histogram", {})
        if hist:
            print("\nFree Variables Distribution (n_free bin: count)")
            bins: dict[int, int] = {}
            for nf, cnt in hist.items():
                b = (nf // 10) * 10
                bins[b] = bins.get(b, 0) + cnt
            
            for b in sorted(bins.keys()):
                cnt = bins[b]
                bin_str = f"{b}-{b+9}"
                print(f"  {bin_str:>7s} vars: {cnt:>12,d}")
            
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
    print(f"interrupted:          {str(interrupted):>12s}")
    print(f"verified_successes:   {n_hits:>12,d}")
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
            write_strategy_table_excel(
                df=df,
                excel_file_path=str(out_path),
                players=players,
                effectivity=setup["effectivity"],
                states=states,
                metadata={"payoff_table": config.get("payoff_table", "")},
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
