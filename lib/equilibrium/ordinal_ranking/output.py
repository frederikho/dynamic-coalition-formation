"""Writing strategy tables and manifests for ordinal ranking search results."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.equilibrium.excel_writer import write_strategy_table_excel
from lib.equilibrium.ordinal_ranking.induced_strategies import (
    _induce_profile_from_rankings,
    _induce_profile_from_weak_orders,
)
from lib.equilibrium.solver import EquilibriumSolver


def _build_committee_idxs(solver: EquilibriumSolver) -> list[list[list[tuple[int, ...]]]]:
    from lib.utils import get_approval_committee
    players = solver.players
    states = solver.states
    player_idx_map = {p: i for i, p in enumerate(players)}
    committee_idxs: list[list[list[tuple[int, ...]]]] = []
    for proposer in players:
        proposer_rows = []
        for current_state in states:
            row = []
            for next_state in states:
                committee = get_approval_committee(solver.effectivity, players, proposer, current_state, next_state)
                row.append(tuple(player_idx_map[p] for p in committee))
            proposer_rows.append(row)
        committee_idxs.append(proposer_rows)
    return committee_idxs


def _solve_induced(solver: EquilibriumSolver) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Given solver with populated p_proposals/r_acceptances, compute strategy_df, P, V."""
    strategy_df = solver._create_strategy_dataframe()
    P, _, _ = solver._compute_transition_probabilities(strategy_df)
    V = solver._solve_value_functions(P)
    return strategy_df, P, V


class StreamingWriter:
    """Writes each equilibrium hit to disk immediately as it is found."""

    def __init__(
        self,
        solver: EquilibriumSolver,
        output_dir: Path,
        payoff_path: Path,
        dedup_by: str = "none",
        weak_orders: bool = False,
        committee_idxs: list | None = None,
    ) -> None:
        self.solver = solver
        self.output_dir = Path(output_dir)
        self.payoff_path = Path(payoff_path)
        self.dedup_by = dedup_by
        self.weak_orders = weak_orders
        self.seen_keys: set[str] = set()
        self.manifest_rows: list[dict[str, Any]] = []
        self.manifest_path = self.output_dir / "manifest.csv"
        self._total_seen: int = 0  # counts every success handed to write()

        self.committee_idxs = committee_idxs if committee_idxs is not None else _build_committee_idxs(solver)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, success: dict) -> bool:
        """Induce strategy, write xlsx, flush manifest. Returns True if written (not a duplicate)."""
        solver = self.solver
        players = solver.players
        states = solver.states

        if self.weak_orders:
            _induce_profile_from_weak_orders(solver, players, states, success["rankings"], self.committee_idxs)
        else:
            _induce_profile_from_rankings(solver, players, states, success["rankings"], self.committee_idxs)

        strategy_df, P, V = _solve_induced(solver)

        if self.dedup_by == "transition":
            key = hashlib.md5(P.to_numpy().tobytes()).hexdigest()
        elif self.dedup_by == "strategy":
            key = hashlib.md5(strategy_df.to_numpy().tobytes()).hexdigest()
        else:
            key = f"idx_{self._total_seen:05d}"

        self._total_seen += 1

        if self.dedup_by != "none" and key in self.seen_keys:
            self.seen_keys.add(key)
            return False
        self.seen_keys.add(key)

        filename = f"eq_{key}.xlsx"
        file_path = self.output_dir / filename

        metadata: dict[str, Any] = {
            "payoff_table": str(self.payoff_path),
            "dedup_key": key,
            "weak_orders": self.weak_orders,
        }
        for p_idx, p_name in enumerate(players):
            metadata[f"ranking_{p_name}"] = str(list(success["rankings"][p_idx]))

        write_strategy_table_excel(
            df=strategy_df,
            excel_file_path=str(file_path),
            players=players,
            effectivity=solver.effectivity,
            states=states,
            metadata=metadata,
            value_functions=V,
            static_payoffs=solver.payoffs,
            transition_matrix=P,
        )

        row: dict[str, Any] = {
            "filename": filename,
            "key": key,
            **{f"ranking_{p}": list(success["rankings"][p_idx]) for p_idx, p in enumerate(players)},
        }
        self.manifest_rows.append(row)
        self._flush_manifest()
        return True

    def _flush_manifest(self) -> None:
        if not self.manifest_rows:
            return
        keys = list(self.manifest_rows[0].keys())
        with open(self.manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.manifest_rows)


def _write_all_successes(
    all_successes: list[dict[str, Any]],
    solver: EquilibriumSolver,
    payoff_path: Path,
    output_dir: Path,
    dedup_by: str = "none",
    weak_orders: bool = False,
    committee_idxs: list | None = None,
) -> list[dict[str, Any]]:
    """Write all verified equilibria to a directory with a manifest (batch mode)."""
    writer = StreamingWriter(
        solver=solver,
        output_dir=output_dir,
        payoff_path=payoff_path,
        dedup_by=dedup_by,
        weak_orders=weak_orders,
        committee_idxs=committee_idxs,
    )
    for success in all_successes:
        writer.write(success)
    return writer.manifest_rows
