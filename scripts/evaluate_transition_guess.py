#!/usr/bin/env python3
"""Evaluate absorbing-state predictors against verified n=3 strategy tables.

Two predictors are compared:
  heuristic  — short-term best-response: approve iff short-term payoff weakly
               improves for every required approver; propose to the short-term
               best viable target.
  lcs        — Largest Consistent Set (Chwe 1994): set of all outcomes that
               can possibly be farsightedly stable given the payoff structure
               and the effectivity correspondence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.find import _parse_players_from_payoff_table
from lib.equilibrium.lcs import compute_lcs
from lib.equilibrium.scenarios import fill_players, get_scenario
from lib.equilibrium.solver import EquilibriumSolver
from lib.effectivity import heyen_lehtomaa_2021
from lib.utils import get_approval_committee



def _load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        df = pd.read_excel(path, sheet_name="Metadata", header=None)
    except Exception:
        return None
    if df.shape[1] < 2:
        return None
    meta: dict[str, Any] = {}
    for _, row in df.iterrows():
        key = row.iloc[0]
        value = row.iloc[1]
        if isinstance(key, str) and key and not key.startswith("---"):
            meta[key] = value
    return meta


def _parse_players(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _load_short_term_values(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="Short-term Values", header=1, index_col=0)


def _load_transition_matrix(path: Path) -> pd.DataFrame:
    for sheet in ("Transition Table", "Transition Matrix"):
        try:
            return pd.read_excel(path, sheet_name=sheet, header=1, index_col=0)
        except Exception:
            continue
    raise ValueError("No Transition Table/Transition Matrix sheet found")


def _load_config_from_metadata(path: Path, metadata: dict[str, Any], players: list[str]) -> dict[str, Any]:
    scenario_name = metadata.get("scenario_name")
    if not isinstance(scenario_name, str):
        raise ValueError("Missing scenario_name in metadata")
    config = get_scenario(scenario_name)
    if config.get("players") is None:
        config = fill_players(config, players or _parse_players_from_payoff_table(path))
    return config


def _guess_transition_matrix(
    players: list[str],
    states: list[str],
    short_term_values: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    effectivity = heyen_lehtomaa_2021(players, states)
    solver = EquilibriumSolver(
        players=players,
        states=states,
        effectivity=effectivity,
        protocol=config["protocol"],
        payoffs=short_term_values.loc[states, players],
        discounting=config["discounting"],
        unanimity_required=config["unanimity_required"],
        verbose=False,
        random_seed=0,
        initialization_mode="uniform",
        logger=None,
    )

    for proposer in players:
        for current_state in states:
            viable_targets: list[str] = []
            for next_state in states:
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                if all(
                    float(short_term_values.loc[next_state, approver]) >= float(short_term_values.loc[current_state, approver])
                    for approver in committee
                ):
                    viable_targets.append(next_state)
            if not viable_targets:
                viable_targets = [current_state]
            proposer_values = short_term_values.loc[viable_targets, proposer].astype(float)
            best_value = float(proposer_values.max())
            best_targets = [
                state for state in viable_targets
                if np.isclose(float(short_term_values.loc[state, proposer]), best_value, rtol=0.0, atol=1e-12)
            ]

            for next_state in states:
                solver.p_proposals[(proposer, current_state, next_state)] = (
                    1.0 / len(best_targets) if next_state in best_targets else 0.0
                )

            for next_state in states:
                committee = get_approval_committee(effectivity, players, proposer, current_state, next_state)
                for approver in committee:
                    solver.r_acceptances[(proposer, current_state, next_state, approver)] = (
                        1.0
                        if float(short_term_values.loc[next_state, approver]) >= float(short_term_values.loc[current_state, approver])
                        else 0.0
                    )

    P, _, _ = solver._compute_transition_probabilities_fast()
    return P.loc[states, states]


def _absorbing_states(P: pd.DataFrame, edge_threshold: float = 0.05) -> tuple[str, ...]:
    """Return all states in absorbing SCCs of the transition graph.

    An edge x→y is included iff P[x,y] > edge_threshold.  A singleton SCC is
    absorbing only if it has a self-loop; multi-node SCCs are absorbing if they
    have no outgoing edges to states outside the component.

    Using a graph-based approach (rather than requiring P[x,x]=1.0) makes
    detection robust to both hard equilibria (P[x,x]=1.0 exactly) and files
    where strategies have not been fully projected to hard values.
    """
    states = [str(s) for s in P.index]
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # Build adjacency list from thresholded edges
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, s in enumerate(states):
        for j, t in enumerate(states):
            if float(P.loc[s, t]) > edge_threshold:
                adj[i].append(j)

    # Tarjan's SCC
    index_counter = [0]
    stack: list[int] = []
    on_stack = [False] * n
    index = [-1] * n
    lowlink = [-1] * n
    sccs: list[list[int]] = []

    def strongconnect(v: int) -> None:
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        for w in adj[v]:
            if index[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            scc: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)

    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n * 10 + 100))
    for v in range(n):
        if index[v] == -1:
            strongconnect(v)

    # Identify absorbing SCCs: no outgoing edges to outside the component
    absorbing: list[str] = []
    for scc in sccs:
        scc_set = set(scc)
        has_outgoing = any(
            w not in scc_set
            for v in scc
            for w in adj[v]
        )
        if has_outgoing:
            continue
        if len(scc) > 1:
            # Multi-node SCC with no outgoing edges → absorbing set
            absorbing.extend(states[v] for v in scc)
        else:
            # Singleton: absorbing only if it has a self-loop
            v = scc[0]
            if v in adj[v]:
                absorbing.append(states[v])

    return tuple(sorted(absorbing))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a payoff-based transition guess against verified n=3 strategy tables."
    )
    parser.add_argument(
        "--root",
        default="strategy_tables",
        help="Directory to scan for top-level .xlsx strategy tables",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of files processed",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Evaluate one specific strategy table in detail",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.file is not None:
        file_path = Path(args.file)
        if not file_path.exists():
            candidate = root / file_path.name
            if candidate.exists():
                file_path = candidate
        files = [file_path]
    else:
        files = sorted(root.glob("*.xlsx"))
        if args.limit is not None:
            files = files[: args.limit]

    evaluated = 0
    exact_transition_matches = 0
    exact_absorbing_matches = 0
    lcs_contains_actual = 0   # actual absorbing ⊆ LCS  (Chwe Prop. 2 guarantee)
    lcs_exact_matches = 0     # LCS == actual absorbing (tight prediction)
    rows: list[dict[str, Any]] = []

    for path in files:
        metadata = _load_metadata(path)
        if not metadata:
            continue
        if not _truthy(metadata.get("verification_success")):
            continue
        if int(float(metadata.get("n_players", 0) or 0)) != 3:
            continue
        try:
            short_term_values = _load_short_term_values(path)
            actual_P = _load_transition_matrix(path)
        except Exception:
            continue

        players = _parse_players(metadata.get("players"))
        states = [str(s) for s in short_term_values.index.tolist()]
        if not players or len(states) != 5:
            continue

        try:
            config = _load_config_from_metadata(path, metadata, players)
            guessed_P = _guess_transition_matrix(players, states, short_term_values, config)
        except Exception as exc:
            rows.append({
                "file": str(path),
                "status": f"error: {exc}",
            })
            continue

        actual_P = actual_P.loc[states, states].astype(float)
        guessed_P = guessed_P.loc[states, states].astype(float)
        transition_match = bool(np.allclose(actual_P.to_numpy(), guessed_P.to_numpy(), rtol=0.0, atol=1e-9))
        actual_abs = _absorbing_states(actual_P)
        guessed_abs = _absorbing_states(guessed_P)
        absorbing_match = actual_abs == guessed_abs

        # LCS prediction
        try:
            effectivity = heyen_lehtomaa_2021(players, states)
            u = short_term_values.loc[states, players].astype(float)
            lcs, indirect_dom = compute_lcs(players, states, u, effectivity)
        except Exception as exc:
            lcs = None
            indirect_dom = None
            lcs_exc = str(exc)
        else:
            lcs_exc = None

        lcs_abs_contains = (
            lcs is not None
            and all(s in lcs for s in actual_abs)
        )
        lcs_abs_exact = (
            lcs is not None
            and frozenset(actual_abs) == lcs
        )

        evaluated += 1
        exact_transition_matches += int(transition_match)
        exact_absorbing_matches += int(absorbing_match)
        lcs_contains_actual += int(lcs_abs_contains)
        lcs_exact_matches += int(lcs_abs_exact)
        rows.append({
            "file": str(path),
            "status": "ok",
            "transition_match": transition_match,
            "absorbing_match": absorbing_match,
            "actual_absorbing": actual_abs,
            "guessed_absorbing": guessed_abs,
            "lcs": lcs,
            "lcs_contains_actual": lcs_abs_contains,
            "lcs_exact": lcs_abs_exact,
            "lcs_error": lcs_exc,
            "indirect_dom": indirect_dom,
            "actual_P": actual_P,
            "guessed_P": guessed_P,
            "short_term_values": short_term_values.loc[states, players],
        })

    if args.file is not None:
        print("Transition Guess Evaluation")
        print("-" * 80)
        if not rows:
            print("No evaluable result.")
            return
        row = rows[0]
        if row.get("status") != "ok":
            print(f"error: {row.get('status')}")
            return
        print(f"file: {row['file']}")
        print(f"transition_match:    {row['transition_match']}")
        print(f"absorbing_match:     {row['absorbing_match']}")
        print(f"actual_absorbing:    {row['actual_absorbing']}")
        print(f"guessed_absorbing:   {row['guessed_absorbing']}")
        if row.get("lcs_error"):
            print(f"lcs_error:           {row['lcs_error']}")
        else:
            lcs_sorted = tuple(sorted(row["lcs"])) if row["lcs"] is not None else None
            print(f"lcs:                 {lcs_sorted}")
            print(f"lcs_contains_actual: {row['lcs_contains_actual']}")
            print(f"lcs_exact:           {row['lcs_exact']}")

        print("\nShort-term Values")
        print("-" * 80)
        print(row["short_term_values"].to_string(float_format=lambda x: f"{x:.6f}"))
        print("\nActual Transition Matrix")
        print("-" * 80)
        print(row["actual_P"].to_string(float_format=lambda x: f"{x:.6f}"))
        print("\nGuessed Transition Matrix")
        print("-" * 80)
        print(row["guessed_P"].to_string(float_format=lambda x: f"{x:.6f}"))
        print("\nDifference (guessed - actual)")
        print("-" * 80)
        diff = row["guessed_P"] - row["actual_P"]
        print(diff.to_string(float_format=lambda x: f"{x:.6f}"))

        if row.get("indirect_dom") and row.get("lcs") is not None:
            states_list = list(row["short_term_values"].index)
            print("\nIndirect Dominance (a ≪ b)")
            print("-" * 80)
            header = f"{'a \\ b':<20}" + "".join(f"{b:<20}" for b in states_list)
            print(header)
            for a in states_list:
                cells = "".join(
                    f"{'<<':<20}" if row["indirect_dom"].get((a, b), False) else f"{'.':<20}"
                    for b in states_list
                )
                print(f"{a:<20}{cells}")
        return

    print("Transition Guess Evaluation")
    print("-" * 80)
    print("Heuristic:")
    print("  approvals: approve iff short-term payoff weakly improves for every required approver")
    print("  proposals: each proposer chooses their short-term best among approval-viable targets")
    print("LCS (Chwe 1994):")
    print("  set of all farsightedly possibly-stable outcomes given payoffs + effectivity")
    print("")
    print("Summary")
    print("-" * 80)
    print(f"evaluated_files:             {evaluated}")
    print(f"exact_transition_matches:    {exact_transition_matches}")
    print(f"exact_absorbing_matches:     {exact_absorbing_matches}  (heuristic == actual)")
    print(f"lcs_contains_actual:         {lcs_contains_actual}  (actual ⊆ LCS  — Chwe guarantee)")
    print(f"lcs_exact_matches:           {lcs_exact_matches}  (LCS == actual absorbing set)")

    # LCS size distribution
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("lcs") is not None]
    if ok_rows:
        sizes = [len(r["lcs"]) for r in ok_rows]
        from collections import Counter
        size_counts = sorted(Counter(sizes).items())
        print(f"lcs_size_distribution:       { {k: v for k, v in size_counts} }")

    mismatch_limit = 20
    print("\nHeuristic absorbing mismatches")
    print("-" * 80)
    mismatch_rows = [row for row in rows if row.get("status") == "ok" and (not row["transition_match"] or not row["absorbing_match"])]
    if not mismatch_rows:
        print("none")
    else:
        print(f"showing first {min(mismatch_limit, len(mismatch_rows))} of {len(mismatch_rows)} mismatches")
        print(f"{'file':<40} {'P':<5} {'A':<5} {'actual_abs':<28} {'guessed_abs':<28} {'lcs':<28}")
        print(f"{'-' * 40} {'-' * 5} {'-' * 5} {'-' * 28} {'-' * 28} {'-' * 28}")
        for row in mismatch_rows[:mismatch_limit]:
            file_label = Path(row["file"]).name
            actual_abs = str(row["actual_absorbing"])
            guessed_abs = str(row["guessed_absorbing"])
            lcs_str = str(tuple(sorted(row["lcs"]))) if row.get("lcs") is not None else "err"
            print(
                f"{file_label:<40} "
                f"{str(row['transition_match']):<5} "
                f"{str(row['absorbing_match']):<5} "
                f"{actual_abs:<28.28} "
                f"{guessed_abs:<28.28} "
                f"{lcs_str:<28.28}"
            )

    print("\nLCS non-exact predictions (LCS ≠ actual absorbing)")
    print("-" * 80)
    lcs_nonexact = [r for r in rows if r.get("status") == "ok" and r.get("lcs") is not None and not r["lcs_exact"]]
    if not lcs_nonexact:
        print("none")
    else:
        print(f"showing first {min(mismatch_limit, len(lcs_nonexact))} of {len(lcs_nonexact)} cases")
        print(f"{'file':<42} {'actual_abs':<36} lcs")
        print(f"{'-' * 42} {'-' * 36} {'-' * 36}")
        for row in lcs_nonexact[:mismatch_limit]:
            file_label = Path(row["file"]).name
            actual_abs = str(row["actual_absorbing"])
            lcs_str = str(tuple(sorted(row["lcs"])))
            print(f"{file_label:<42} {actual_abs:<36} {lcs_str}")


if __name__ == "__main__":
    main()
