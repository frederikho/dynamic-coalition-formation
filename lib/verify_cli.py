#!/usr/bin/env python3
"""
CLI wrapper to verify whether an Excel strategy profile is an equilibrium.

Usage examples:
    python -m lib.verify_cli eq_n4_weak_gov_unan_b02e87_test.xlsx
    python -m lib.verify_cli ./strategy_tables/my_profile.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow running as a script: `python lib/verify_cli.py ...`
if __package__ is None or __package__ == "":
    repo_root = str(Path(__file__).resolve().parent.parent)
    script_dir = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == script_dir:
        sys.path[0] = repo_root
    elif repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import pandas as pd

from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.mdp import MDP
from lib.probabilities_optimized import (
    TransitionProbabilitiesOptimized as TransitionProbabilities,
)
from lib.equilibrium.solver import format_strategy_df_compact
import numpy as np

from lib.utils import (
    derive_effectivity,
    get_payoff_matrix,
    get_geoengineering_levels,
    verify_equilibrium,
)
from lib.effectivity import check_effectivity, get_effectivity
from lib.equilibrium.excel_writer import write_strategy_table_excel


DEFAULT_STRATEGY_DIR = Path("./strategy_tables")
DEFAULT_PAYOFF_TABLE_DIR = Path("./payoff_tables")


def _deployer_key(state) -> str:
    """Return the payoff-table row key for a state (mirrors find.py)."""
    if state.geo_deployment_level == 0:
        return "( )"
    members = sorted(country.name for country in state.strongest_coalition.members)
    return "(" + "".join(members) + ")"


def _load_payoff_table(
    path: Path, state_objects: List, players: List[str]
) -> "pd.DataFrame":
    """Load a precomputed payoff table and map rows to framework state names."""
    if not path.exists():
        fallback = DEFAULT_PAYOFF_TABLE_DIR / path.name
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Payoff table '{path.name}' not found.\n"
                f"Searched: {path.resolve()}, {fallback.resolve()}\n"
                f"Tip: place the file in payoff_tables/ or provide the full path."
            )

    df = pd.read_excel(str(path), sheet_name="Payoffs", header=1, index_col=0)

    missing = [p for p in players if p not in df.columns]
    if missing:
        raise ValueError(
            f"Payoff table {path.name} is missing player columns: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    state_names = [s.name for s in state_objects]
    payoffs = pd.DataFrame(index=state_names, columns=players, dtype=np.float64)
    for state in state_objects:
        # Prefer direct state name match; fall back to computed deployer key.
        if state.name in df.index:
            key = state.name
        else:
            key = _deployer_key(state)
            if key not in df.index:
                raise ValueError(
                    f"Payoff table {path.name} has no row for deployer key '{key}' "
                    f"(needed by state '{state.name}').\n"
                    f"Available keys: {df.index.tolist()}"
                )
        payoffs.loc[state.name] = df.loc[key, players].values
    return payoffs


def _read_metadata_from_xlsx(xlsx_path: Path) -> Dict[str, Any]:
    """Read metadata key/value pairs from a metadata sheet if present."""
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception:
        return {}

    metadata_sheet_name = None
    for possible_name in ("Metadata", "metadata", "Tabelle2", "Sheet2"):
        if possible_name in xl.sheet_names:
            metadata_sheet_name = possible_name
            break

    if metadata_sheet_name is None:
        return {}

    try:
        metadata_df = pd.read_excel(xlsx_path, sheet_name=metadata_sheet_name)
    except Exception:
        return {}

    metadata: Dict[str, Any] = {}
    for _, row in metadata_df.iterrows():
        param = row.get("Parameter")
        value = row.get("Value")

        if pd.isna(param):
            continue

        key = str(param).strip()
        if not key or key.startswith("---"):
            continue

        metadata[key] = value

    return metadata


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _players_from_strategy_df(df: pd.DataFrame) -> List[str]:
    proposers = _ordered_unique([col[0] for col in df.columns])
    players: List[str] = []
    for proposer in proposers:
        p = str(proposer).replace("Proposer ", "").strip()
        if p:
            players.append(p)
    return players


def _states_from_strategy_df(df: pd.DataFrame) -> List[str]:
    return _ordered_unique([col[1] for col in df.columns])


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def _infer_n_from_filename(stem: str) -> int | None:
    match = re.search(r"(?:^|_)n(\d+)(?:_|$)", stem)
    if match:
        return int(match.group(1))
    return None


def _infer_power_rule_from_filename(stem: str) -> str | None:
    s = stem.lower()
    if "weak_gov" in s or "weak_governance" in s:
        return "weak_governance"
    if "power_thresh" in s or "power_threshold" in s:
        return "power_threshold"
    return None


def _infer_unanimity_from_filename(stem: str) -> bool | None:
    s = stem.lower()
    if "_unan_" in f"_{s}_":
        return True
    if "_maj_" in f"_{s}_":
        return False
    return None


def _infer_min_power_from_filename(stem: str) -> float | None:
    """Infer min power from filename tokens like mp0.5, minpower0.501, p0501."""
    s = stem.lower()

    patterns = [
        r"(?:^|_)(?:mp|minpower|min_power)(\d+(?:\.\d+)?)(?:_|$)",
        r"(?:^|_)p(0\d{2,4})(?:_|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if not match:
            continue
        raw = match.group(1)
        if raw.startswith("0") and raw.isdigit() and len(raw) >= 3:
            # e.g., 0501 -> 0.501
            return float(f"0.{raw[1:]}")
        return float(raw)

    return None


def _require_float(metadata: Dict[str, Any], key: str, context: str) -> float:
    if key not in metadata or pd.isna(metadata[key]):
        raise ValueError(f"Missing required value '{key}' in metadata for {context}.")
    try:
        return float(metadata[key])
    except Exception as exc:
        raise ValueError(f"Value '{key}' in metadata is not numeric for {context}: {metadata[key]!r}") from exc


def _build_config(xlsx_path: Path, strategy_df: pd.DataFrame) -> Dict[str, Any]:
    """Build verification config strictly from metadata/filename/file content."""
    metadata = _read_metadata_from_xlsx(xlsx_path)

    file_stem = xlsx_path.stem
    states = _states_from_strategy_df(strategy_df)

    # Priority for players: metadata -> strategy table columns.
    players: List[str]
    if "players" in metadata and str(metadata["players"]).strip():
        players = [p.strip() for p in str(metadata["players"]).split(",") if p.strip()]
    else:
        players = _players_from_strategy_df(strategy_df)

    if not players:
        raise ValueError("Could not infer players from metadata or strategy table columns.")

    n_from_filename = _infer_n_from_filename(file_stem)
    n_from_metadata = None
    if "n_players" in metadata and not pd.isna(metadata["n_players"]):
        try:
            n_from_metadata = int(metadata["n_players"])
        except Exception as exc:
            raise ValueError(f"Metadata field 'n_players' is invalid: {metadata['n_players']!r}") from exc

    if n_from_filename is not None and n_from_filename != len(players):
        raise ValueError(
            f"Filename implies n={n_from_filename}, but strategy table has {len(players)} players ({players})."
        )
    if n_from_metadata is not None and n_from_metadata != len(players):
        raise ValueError(
            f"Metadata implies n_players={n_from_metadata}, but strategy table has {len(players)} players ({players})."
        )

    if not states:
        raise ValueError("Could not infer any states from strategy table columns.")

    config: Dict[str, Any] = {
        "players": players,
        "state_names": states,
    }

    inferred_power_rule = _infer_power_rule_from_filename(file_stem)
    inferred_unanimity = _infer_unanimity_from_filename(file_stem)
    inferred_min_power = _infer_min_power_from_filename(file_stem)

    # Required global parameters (metadata first, filename fallback where possible)
    if "power_rule" in metadata and str(metadata["power_rule"]).strip():
        config["power_rule"] = str(metadata["power_rule"]).strip()
    elif inferred_power_rule is not None:
        config["power_rule"] = inferred_power_rule
    else:
        raise ValueError("Missing required value 'power_rule' in metadata or filename.")

    if "unanimity_required" in metadata:
        config["unanimity_required"] = _safe_bool(metadata["unanimity_required"])
    elif inferred_unanimity is not None:
        config["unanimity_required"] = inferred_unanimity
    else:
        raise ValueError("Missing required value 'unanimity_required' in metadata or filename (_unan_/_maj_).")

    if config["power_rule"] == "power_threshold":
        if "min_power" in metadata and not pd.isna(metadata["min_power"]):
            config["min_power"] = float(metadata["min_power"])
        elif inferred_min_power is not None:
            config["min_power"] = inferred_min_power
        else:
            raise ValueError(
                "Missing required value 'min_power' for power_threshold in metadata or filename."
            )
    else:
        # For weak governance this parameter is structurally unused.
        config["min_power"] = None

    config["discounting"] = _require_float(metadata, "discounting", "discount factor")

    # Required player-specific parameters in metadata
    for field in ("base_temp", "ideal_temp", "delta_temp", "m_damage", "power", "protocol"):
        config[field] = {}

    for player in players:
        for field in ("base_temp", "ideal_temp", "delta_temp", "m_damage", "power", "protocol"):
            key = f"{field}_{player}"
            config[field][player] = _require_float(metadata, key, f"player '{player}'")

    return config


def _parse_coalition_structure(state_name: str, all_countries: List[Country]) -> List[Coalition]:
    """Parse coalition state label like '(RUSUSA)' or '(CT)(FW)' into Coalition objects.

    Handles both single-character names (W, T, C) and multi-character names
    (NDE, USA, RUS) by greedy left-to-right matching of known player names.
    """
    country_map = {c.name: c for c in all_countries}

    if len(all_countries) == 2 and state_name in {"N", "A", "B"}:
        if state_name == "N":
            return [Coalition([c]) for c in all_countries]
        chosen = all_countries[0] if state_name == "A" else all_countries[1]
        remaining = [c for c in all_countries if c.name != chosen.name]
        return [Coalition([chosen])] + [Coalition([c]) for c in remaining]

    if state_name == "( )":
        return [Coalition([c]) for c in all_countries]

    # Extract the text inside each pair of parentheses.
    coalition_parts: List[str] = []
    current = ""
    depth = 0
    for char in state_name:
        if char == "(":
            depth += 1
            if depth == 1:
                current = ""
        elif char == ")":
            depth -= 1
            if depth == 0 and current:
                coalition_parts.append(current)
                current = ""
        elif depth == 1:
            current += char

    # Sort names longest-first so greedy matching never picks a short prefix
    # when a longer name starts the same way.
    sorted_names = sorted(country_map.keys(), key=len, reverse=True)

    countries_in_coalitions: set = set()
    coalitions: List[Coalition] = []
    for part in coalition_parts:
        coalition_members = []
        remaining = part
        while remaining:
            matched = False
            for name in sorted_names:
                if remaining.startswith(name):
                    coalition_members.append(country_map[name])
                    countries_in_coalitions.add(name)
                    remaining = remaining[len(name):]
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Cannot parse '{remaining}' in coalition part '{part}' "
                    f"of state '{state_name}'. Known players: {list(country_map)}"
                )
        if coalition_members:
            coalitions.append(Coalition(coalition_members))

    for country in all_countries:
        if country.name not in countries_in_coalitions:
            coalitions.append(Coalition([country]))

    return coalitions


def _run_verification(xlsx_path: Path, skip_effectivity_check: bool = False, effectivity_rule: str = "heyen_lehtomaa_2021") -> Tuple[bool, str, Dict[str, Any]]:
    strategy_df_raw = pd.read_excel(xlsx_path, header=[0, 1], index_col=[0, 1, 2])

    config = _build_config(xlsx_path, strategy_df_raw)

    players = config["players"]
    states = config["state_names"]

    # Build countries and states for payoff construction.
    all_countries = [
        Country(
            name=player,
            base_temp=config["base_temp"][player],
            delta_temp=config["delta_temp"][player],
            ideal_temp=config["ideal_temp"][player],
            m_damage=config["m_damage"][player],
            power=config["power"][player],
        )
        for player in players
    ]

    state_objects = [
        State(
            name=state_name,
            coalitions=_parse_coalition_structure(state_name, all_countries),
            all_countries=all_countries,
            power_rule=config["power_rule"],
            min_power=config["min_power"],
        )
        for state_name in states
    ]

    # Use precomputed payoff table if the profile was solved with one.
    metadata = _read_metadata_from_xlsx(xlsx_path)
    payoff_source = str(metadata.get("payoff_source", "")).strip()
    payoff_table_name = str(metadata.get("payoff_table", "")).strip()

    if payoff_source == "precomputed_table" and payoff_table_name:
        payoff_table_path = Path(payoff_table_name)
        payoffs = _load_payoff_table(payoff_table_path, state_objects, players)
        print(f"Payoffs: loaded from '{payoff_table_path.name}'")
    else:
        payoffs = get_payoff_matrix(states=state_objects, columns=players)

    geoengineering = get_geoengineering_levels(states=state_objects)

    file_effectivity = derive_effectivity(df=strategy_df_raw, players=players, states=states)
    effectivity = get_effectivity(effectivity_rule, players, states)

    if not skip_effectivity_check:
        effectivity_violations = check_effectivity(file_effectivity, players, states, rule=effectivity_rule)
        if effectivity_violations:
            lines = "\n  ".join(effectivity_violations)
            raise ValueError(
                f"Effectivity rule violations ({len(effectivity_violations)}):\n  {lines}"
            )

    # For transition probability computation, use file-derived effectivity when skipping
    # the check (so V/P reflect actual file structure). The rule-based effectivity is
    # always kept in `details` so the enriched Excel highlights what OUGHT to be there.
    tp_effectivity = file_effectivity if skip_effectivity_check else effectivity

    strategy_df = strategy_df_raw.copy()
    strategy_df.fillna(0.0, inplace=True)

    transition_probabilities = TransitionProbabilities(
        df=strategy_df,
        effectivity=tp_effectivity,
        players=players,
        states=states,
        protocol=config["protocol"],
        unanimity_required=config["unanimity_required"],
    )
    P, P_proposals, P_approvals = transition_probabilities.get_probabilities()

    mdp = MDP(n_states=len(states), transition_probs=P, discounting=config["discounting"])

    V = pd.DataFrame(index=states, columns=players)
    for player in players:
        V.loc[:, player] = mdp.solve_value_func(payoffs.loc[:, player])

    result = {
        "scenario_name": xlsx_path.stem,
        "V": V,
        "P": P,
        "geoengineering": geoengineering,
        "payoffs": payoffs,
        "P_proposals": P_proposals,
        "P_approvals": P_approvals,
        "players": players,
        "state_names": states,
        "effectivity": effectivity,
        "strategy_df": strategy_df,
    }

    success, message = verify_equilibrium(result)

    compact_strategy = format_strategy_df_compact(
        strategy_df_raw,
        players=players,
        states=states,
    )

    details = {
        "players": players,
        "n_states": len(states),
        "power_rule": config["power_rule"],
        "min_power": config["min_power"],
        "unanimity_required": config["unanimity_required"],
        "discounting": config["discounting"],
        "compact_strategy": compact_strategy,
        "V": V,
        "P": P,
        "payoffs": payoffs,
        "effectivity": effectivity,
        "strategy_df": strategy_df_raw,
        "state_names": states,
    }
    return success, message, details


def _resolve_profile_path(profile_arg: str, strategy_dir: Path) -> Path:
    candidate = Path(profile_arg)
    if candidate.exists():
        return candidate.resolve()

    in_default_dir = strategy_dir / profile_arg
    if in_default_dir.exists():
        return in_default_dir.resolve()

    raise FileNotFoundError(
        f"Could not find strategy profile '{profile_arg}' "
        f"(looked at '{candidate}' and '{in_default_dir}')."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify if an Excel strategy profile is an equilibrium."
    )
    parser.add_argument(
        "profile",
        type=str,
        help="Profile filename or full path, e.g. eq_n4_weak_gov_unan_b02e87_test.xlsx",
    )
    parser.add_argument(
        "--strategy-dir",
        type=str,
        default=str(DEFAULT_STRATEGY_DIR),
        help="Default directory for profile files (used when profile is not an existing path).",
    )
    parser.add_argument(
        "--effectivity-rule",
        type=str,
        default="heyen_lehtomaa_2021",
        choices=["heyen_lehtomaa_2021", "unanimous_consent", "deployer_exit", "free_exit"],
        help="Effectivity rule to validate against (default: heyen_lehtomaa_2021).",
    )
    parser.add_argument(
        "--enrich",
        type=str,
        nargs="?",
        const="__same__",
        default=None,
        metavar="PATH",
        help=(
            "Write enriched file (adds Value Functions, Short-term Values, Transition Matrix "
            "sheets). If PATH is omitted, overwrites the input file. "
            "Enrichment happens even when not an equilibrium."
        ),
    )

    args = parser.parse_args()

    try:
        profile_path = _resolve_profile_path(args.profile, Path(args.strategy_dir))

        effectivity_ok = True
        try:
            success, message, details = _run_verification(profile_path, effectivity_rule=args.effectivity_rule)
        except ValueError as exc:
            effectivity_ok = False
            print(f"Warning: {exc}", file=sys.stderr)
            success, message, details = _run_verification(profile_path, skip_effectivity_check=True, effectivity_rule=args.effectivity_rule)
            success = False
            message = "Effectivity rule violations (see above)."

        print(f"Profile: {profile_path}")
        print(
            "Config: "
            f"players={details['players']} | "
            f"states={details['n_states']} | "
            f"rule={details['power_rule']} | "
            f"min_power={details['min_power']} | "
            f"unanimity_required={details['unanimity_required']}"
        )
        print(details["compact_strategy"])

        if args.enrich is not None:
            output_path = profile_path if args.enrich == "__same__" else Path(args.enrich)
            metadata = _read_metadata_from_xlsx(profile_path) or None
            write_strategy_table_excel(
                df=details["strategy_df"],
                excel_file_path=str(output_path),
                players=details["players"],
                effectivity=details["effectivity"],
                states=details["state_names"],
                metadata=metadata,
                value_functions=details["V"],
                geo_levels=None,
                deploying_coalitions=None,
                static_payoffs=details["payoffs"],
                transition_matrix=details["P"],
            )
            print(f"Written: {output_path}")

        if success:
            print("Result: EQUILIBRIUM ✅")
            print(message)
            raise SystemExit(0)

        print("Result: NOT AN EQUILIBRIUM ❌")
        print(message)
        raise SystemExit(1)

    except KeyError as exc:
        key = exc.args[0]
        if isinstance(key, tuple) and len(key) == 3:
            state, row_type, player = key
            print(
                f"Verification failed: missing row ({state!r}, {row_type!r}, {player!r}) "
                f"in strategy table — check that every acceptance row for {player!r} "
                f"in state {state!r} is present in your Excel file.",
                file=sys.stderr,
            )
        else:
            print(f"Verification failed: missing key {key!r} in strategy table.", file=sys.stderr)
        raise SystemExit(2)

    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
