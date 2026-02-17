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

import pandas as pd

# Allow running as a script: `python lib/verify_cli.py ...`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.mdp import MDP
from lib.probabilities_optimized import (
    TransitionProbabilitiesOptimized as TransitionProbabilities,
)
from lib.utils import (
    derive_effectivity,
    get_payoff_matrix,
    get_geoengineering_levels,
    verify_equilibrium,
)


DEFAULT_STRATEGY_DIR = Path("./strategy_tables")


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
    """Parse coalition state label like '(CT)(FW)' into Coalition objects."""
    country_map = {c.name: c for c in all_countries}

    if state_name == "( )":
        return [Coalition([c]) for c in all_countries]

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

    countries_in_coalitions = set()
    coalitions: List[Coalition] = []
    for part in coalition_parts:
        coalition_members = []
        for country_name in part:
            if country_name in country_map:
                coalition_members.append(country_map[country_name])
                countries_in_coalitions.add(country_name)
        if coalition_members:
            coalitions.append(Coalition(coalition_members))

    for country in all_countries:
        if country.name not in countries_in_coalitions:
            coalitions.append(Coalition([country]))

    return coalitions


def _run_verification(xlsx_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
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

    payoffs = get_payoff_matrix(states=state_objects, columns=players)
    geoengineering = get_geoengineering_levels(states=state_objects)

    effectivity = derive_effectivity(df=strategy_df_raw, players=players, states=states)

    strategy_df = strategy_df_raw.copy()
    strategy_df.fillna(0.0, inplace=True)

    transition_probabilities = TransitionProbabilities(
        df=strategy_df,
        effectivity=effectivity,
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

    details = {
        "players": players,
        "n_states": len(states),
        "power_rule": config["power_rule"],
        "min_power": config["min_power"],
        "unanimity_required": config["unanimity_required"],
        "discounting": config["discounting"],
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

    args = parser.parse_args()

    try:
        profile_path = _resolve_profile_path(args.profile, Path(args.strategy_dir))
        success, message, details = _run_verification(profile_path)

        print(f"Profile: {profile_path}")
        print(
            "Config: "
            f"players={details['players']} | "
            f"states={details['n_states']} | "
            f"rule={details['power_rule']} | "
            f"min_power={details['min_power']} | "
            f"unanimity_required={details['unanimity_required']}"
        )

        if success:
            print("Result: EQUILIBRIUM ✅")
            print(message)
            raise SystemExit(0)

        print("Result: NOT AN EQUILIBRIUM ❌")
        print(message)
        raise SystemExit(1)

    except Exception as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
