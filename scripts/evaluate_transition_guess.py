#!/usr/bin/env python3
"""Evaluate absorbing-state predictors against verified n=3 strategy tables.

Predictors:
  Heuristic  — short-term best-response (STBR)
  LCS        — Largest Consistent Set (Chwe 1994)
  LCCS       — Largest Cautious Consistent Set (Mauleon & Vannetelbosch 2004)
  HREFS      — Largest History-dependent Rational Expectation Farsighted Stable Set (Dutta & Vartiainen 2020)
  HSREFS     — Largest History-dependent Strongly Rational Expectation Farsighted Stable Set (Dutta & Vartiainen 2020)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.equilibrium.find import _parse_players_from_payoff_table
from lib.equilibrium.lcs import compute_lcs, compute_lccs, compute_largest_hrefs
from lib.equilibrium.scenarios import fill_players, get_scenario
from lib.equilibrium.solver import EquilibriumSolver
from lib.effectivity import get_effectivity, get_forbidden_proposals
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
    effectivity_rule: str = "heyen_lehtomaa_2021",
) -> pd.DataFrame:
    effectivity = get_effectivity(effectivity_rule, players, states)
    forbidden = get_forbidden_proposals(effectivity_rule, players, states)
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
                if (proposer, current_state, next_state) in forbidden:
                    continue
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
    states = [str(s) for s in P.index]
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, s in enumerate(states):
        for j, t in enumerate(states):
            if float(P.loc[s, t]) > edge_threshold:
                adj[i].append(j)

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
                if w == v: break
            sccs.append(scc)

    for v in range(n):
        if index[v] == -1: strongconnect(v)

    absorbing: list[str] = []
    for scc in sccs:
        scc_set = set(scc)
        has_outgoing = any(w not in scc_set for v in scc for w in adj[v])
        if has_outgoing: continue
        if len(scc) > 1:
            absorbing.extend(states[v] for v in scc)
        else:
            v = scc[0]
            if v in adj[v]: absorbing.append(states[v])
    return tuple(sorted(absorbing))


def _run_payoff_table_mode(path: Path, effectivity_rule: str | None = None) -> None:
    print("Stability Prediction from Payoff Table")
    print("-" * 80)
    print(f"file: {path}")

    try:
        meta_df = pd.read_excel(path, sheet_name="Metadata", header=None)
        meta: dict[str, Any] = {row.iloc[0]: row.iloc[1] for _, row in meta_df.iterrows() if isinstance(row.iloc[0], str)}
    except Exception:
        meta = {}

    players = _parse_players(meta.get("players"))
    if not players:
        # Fallback to inference from columns
        try:
            df = pd.read_excel(path, sheet_name="Payoffs", header=1, index_col=0)
            players = [c for c in df.columns if c not in ("W_SAI", "Source file") and not str(c).startswith("W_SAI")]
        except Exception:
            print("error: could not determine players")
            return

    u: pd.DataFrame | None = None
    for sheet in ("Payoffs", "Short-term Values"):
        try:
            raw = pd.read_excel(path, sheet_name=sheet, header=1, index_col=0)
            u = raw[[p for p in players if p in raw.columns]].astype(float)
            break
        except Exception: continue
    if u is None: return

    states = [str(s) for s in u.index.tolist()]
    file_rule = meta.get("effectivity_rule", "heyen_lehtomaa_2021")
    rule_to_use = effectivity_rule or file_rule
    print(f"Effectivity Rule: {rule_to_use}")
    effectivity = get_effectivity(rule_to_use, players, states)
    
    try:
        lcs, _ = compute_lcs(players, states, u, effectivity)
        lccs_s, _ = compute_lccs(players, states, u, effectivity, weak=False)
        lccs_w, _ = compute_lccs(players, states, u, effectivity, weak=True)
        hrefs = compute_largest_hrefs(players, states, u, effectivity, strong=False)
        hsrefs = compute_largest_hrefs(players, states, u, effectivity, strong=True)
    except Exception as exc:
        print(f"Prediction error: {exc}")
        return

    print(f"LCS (Strict):  {tuple(sorted(lcs))}")
    print(f"LCCS (Strict): {tuple(sorted(lccs_s))}")
    print(f"LCCS (Weak):   {tuple(sorted(lccs_w))}")
    print(f"HREFS (L):     {tuple(sorted(hrefs))}")
    print(f"HSREFS (L):    {tuple(sorted(hsrefs))}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate farsighted stability predictors.")
    parser.add_argument("--root", default="strategy_tables", help="Directory to scan")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--file", type=str, default=None, help="Evaluate a specific file")
    parser.add_argument("--effectivity-rule", type=str, default=None, help="Filter by or force effectivity rule")
    args = parser.parse_args()

    root = Path(args.root)
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists(): file_path = root / file_path.name
        try:
            sheet_names = pd.ExcelFile(file_path).sheet_names
        except Exception: return
        if "Short-term Values" not in sheet_names and "Payoffs" in sheet_names:
            _run_payoff_table_mode(file_path, effectivity_rule=args.effectivity_rule)
            return
        files = [file_path]
    else:
        files = sorted(root.glob("*.xlsx"))
        if args.limit: files = files[:args.limit]

    stats = {
        "Heuristic":   {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
        "LCS (S)":     {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
        "LCS (W)":     {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
        "LCCS (S)":    {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
        "LCCS (W)":    {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
        "HREFS (L)":   {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
        "HSREFS (L)":  {"evaluated": 0, "contains": 0, "exact": 0, "sizes": []},
    }
    rows: list[dict[str, Any]] = []

    for path in tqdm(files, desc="Computing stability predictors", unit="file"):
        metadata = _load_metadata(path)
        if not (metadata and _truthy(metadata.get("verification_success")) and int(float(metadata.get("n_players", 0))) == 3):
            continue
        try:
            u_df = _load_short_term_values(path)
            actual_P = _load_transition_matrix(path)
        except Exception: continue

        players = _parse_players(metadata.get("players"))
        states = [str(s) for s in u_df.index.tolist()]
        if not players or len(states) < 5: continue

        file_rule = metadata.get("effectivity_rule", "heyen_lehtomaa_2021")
        if args.effectivity_rule and args.effectivity_rule != file_rule:
            continue

        try:
            config = _load_config_from_metadata(path, metadata, players)
            guessed_P = _guess_transition_matrix(players, states, u_df, config, effectivity_rule=file_rule)
            u = u_df.loc[states, players].astype(float)
            effectivity = get_effectivity(file_rule, players, states)
            
            actual_abs = _absorbing_states(actual_P.loc[states, states])
            guessed_abs = _absorbing_states(guessed_P.loc[states, states])
            
            # Predictors
            preds = {
                "Heuristic":   set(guessed_abs),
                "LCS (S)":     compute_lcs(players, states, u, effectivity, weak=False)[0],
                "LCS (W)":     compute_lcs(players, states, u, effectivity, weak=True)[0],
                "LCCS (S)":    compute_lccs(players, states, u, effectivity, weak=False)[0],
                "LCCS (W)":    compute_lccs(players, states, u, effectivity, weak=True)[0],
                "HREFS (L)":   compute_largest_hrefs(players, states, u, effectivity, strong=False),
                "HSREFS (L)":  compute_largest_hrefs(players, states, u, effectivity, strong=True),
            }
            
            for name, members in preds.items():
                stats[name]["evaluated"] += 1
                stats[name]["contains"] += int(all(s in members for s in actual_abs))
                stats[name]["exact"] += int(frozenset(actual_abs) == members)
                stats[name]["sizes"].append(len(members))
                
            rows.append({"file": path.name, "actual": actual_abs, "preds": preds})
        except Exception as exc:
            print(f"Error processing {path.name}: {exc}")

    print("Stability Predictor Evaluation (Summary Table)")
    print("=" * 105)
    print("Legend:")
    print("  (S) Strict  : based on indirect strict dominance (all deviators must gain)")
    print("  (W) Weak    : based on indirect weak dominance (all deviators >= equal, some must gain)")
    print("  (L) Largest : the unique largest set containing all coherent stable sets")
    print("-" * 105)
    print(f"{'Stability Concept':<15} | {'Contains Actual':<22} | {'Exact Match':<22} | {'Mean Size':>10} | {'Size Dist'}")
    print("-" * 105)
    for name, s in stats.items():
        n = s["evaluated"]
        if n == 0: continue
        c_pct = (s["contains"] / n) * 100
        e_pct = (s["exact"] / n) * 100
        m_size = np.mean(s["sizes"])
        dist = dict(sorted(Counter(s["sizes"]).items()))
        
        c_str = f"{s['contains']}/{n} ({c_pct:.1f}%)"
        e_str = f"{s['exact']}/{n} ({e_pct:.1f}%)"
        
        print(f"{name:<15} | {c_str:<22} | {e_str:<22} | {m_size:>10.2f} | {dist}")
    print("-" * 105)
    evaluated_n = stats['LCS (S)']['evaluated']
    print(f"evaluated_files: {evaluated_n}")
    print()

    # Show some failures for LCCS
    lccs_fails = [r for r in rows if not all(s in r["preds"]["LCCS (S)"] for s in r["actual"])]
    if lccs_fails:
        print("Notable LCCS (Strict) Failures (actual absorbing ⊈ LCCS)")
        print("-" * 150)
        print(f"{'file':<45} | {'actual_abs':<40} | {'lccs_s':<20} | {'lcs_s'}")
        print("-" * 150)
        for r in lccs_fails:
            actual_str = str(r['actual'])
            lccs_s_str = str(tuple(sorted(r['preds']['LCCS (S)'])))
            lcs_s_str = str(tuple(sorted(r['preds']['LCS (S)'])))
            print(f"{r['file']:<45} | {actual_str:<40} | {lccs_s_str:<20} | {lcs_s_str}")
        print()

if __name__ == "__main__":
    main()
