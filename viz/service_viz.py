"""
Lightweight visualization service for coalition formation transition graphs.
Exposes HTTP endpoints to compute and serve transition probability graphs from XLSX strategy profiles.
"""

import argparse
import copy
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.probabilities_optimized import TransitionProbabilitiesOptimized
from lib.utils import derive_effectivity, list_members
from lib.verify_cli import safe_bool
from lib.mdp import MDP, absorbing_sets, limiting_distribution


def generate_all_partitions(elements):
    """
    Generate all possible partitions of a set.
    A partition is a way of grouping elements into non-empty subsets.
    
    This generates Bell number B(n) partitions for n elements.
    """
    if len(elements) == 0:
        yield []
        return
    
    if len(elements) == 1:
        yield [[elements[0]]]
        return
    
    first = elements[0]
    rest = elements[1:]
    
    # For each partition of the rest
    for partition in generate_all_partitions(rest):
        # Add first element to each existing subset
        for i, subset in enumerate(partition):
            yield partition[:i] + [subset + [first]] + partition[i+1:]
        # Add first element as a new singleton subset
        yield [[first]] + partition


def partition_to_state_name(partition):
    """
    Convert a partition to coalition structure notation.
    
    Examples:
        [[A], [B], [C]] -> '( )' (all singletons)
        [[A, B], [C]] -> '(AB)'
        [[A, B, C]] -> '(ABC)'
        [[A, C], [B]] -> '(AC)'
    """
    # Filter out singletons and sort coalitions
    coalitions = sorted(
        [sorted(subset) for subset in partition if len(subset) > 1],
        key=lambda x: (len(x), x)
    )
    
    if not coalitions:
        return '( )'
    
    # Join coalitions
    coalition_strs = [''.join(coal) for coal in coalitions]
    return ''.join(f'({c})' for c in coalition_strs)


def generate_coalition_structures(n: int) -> list:
    """
    Generate all possible coalition structure names for n players.
    Uses letters W, T, C for n=3 (to match existing convention),
    and A, B, C, D, E, F for other player counts.
    
    Returns list of state names following Bell numbers:
    n=2: 2, n=3: 5, n=4: 15, n=5: 52, n=6: 203, etc.
    """
    # Use consistent player naming
    if n == 3:
        player_letters = ['W', 'T', 'C']
    else:
        player_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:n]
    
    # Generate all partitions
    all_partitions = list(generate_all_partitions(player_letters))
    
    # Convert to state names
    state_names = []
    seen = set()
    
    for partition in all_partitions:
        state_name = partition_to_state_name(partition)
        if state_name not in seen:
            state_names.append(state_name)
            seen.add(state_name)
    
    # Sort: all singletons first, then by size and alphabetically
    def sort_key(name):
        if name == '( )':
            return (0, '')
        # Count coalitions and total size
        coalitions = name.strip('()').split(')(')
        return (1, len(coalitions), sum(len(c) for c in coalitions), name)
    
    state_names.sort(key=sort_key)
    
    return state_names


app = FastAPI(title="Coalition Formation Visualizer API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default profiles directory (absolute path to repository's strategy_tables)
DEFAULT_PROFILES_DIR = str(Path(__file__).parent.parent / "strategy_tables")

# Default configuration (matches main.py base_config)
DEFAULT_CONFIG = {
    "players": ["W", "T", "C"],
    "base_temp": {},
    "ideal_temp": {},
    "delta_temp": {},
    "power": {},
    "protocol": {},
    "m_damage": {},
    "power_rule": "power_threshold",
    "min_power": None,
    "state_names": [],
    "unanimity_required": True,
    "discounting": 0.99,
}


def parse_coalition_structure(state_name: str, all_countries: List[Country]) -> List[Coalition]:
    """
    Parse a state name like '(CT)' or '(WTC)' and create the corresponding coalition structure.
    
    Args:
        state_name: Coalition structure notation like '( )', '(CT)', '(WTC)', etc.
        all_countries: List of all Country objects
    
    Returns:
        List of Coalition objects representing this structure
    """
    country_map = {c.name: c for c in all_countries}
    players = list(country_map.keys())
    
    # Use the library function which handles multi-character names correctly
    from lib.utils import list_coalitions
    coalition_member_lists = list_coalitions(state_name, players)
    
    coalitions = []
    countries_in_coalitions = set()
    
    for member_names in coalition_member_lists:
        coalition_countries = [country_map[name] for name in member_names]
        coalitions.append(Coalition(coalition_countries))
        for name in member_names:
            countries_in_coalitions.add(name)
            
    # Add singletons for countries not in any coalition
    for country in all_countries:
        if country.name not in countries_in_coalitions:
            coalitions.append(Coalition([country]))
            
    return coalitions


def read_metadata_from_xlsx(xlsx_path: str) -> Dict[str, Any]:
    """
    Read metadata from the second sheet of an XLSX file.

    Args:
        xlsx_path: Path to the XLSX strategy profile

    Returns:
        Dict containing metadata extracted from the file
    """
    try:
        # Try multiple possible sheet names for metadata
        xl = pd.ExcelFile(xlsx_path)
        metadata_sheet_name = None

        # Check for common metadata sheet names
        for possible_name in ['Metadata', 'metadata', 'Tabelle2', 'Sheet2']:
            if possible_name in xl.sheet_names:
                metadata_sheet_name = possible_name
                break

        if metadata_sheet_name is None:
            # No metadata sheet found
            return {}

        # Read the Metadata sheet
        metadata_df = pd.read_excel(xlsx_path, sheet_name=metadata_sheet_name)
        
        # Convert to a dictionary
        metadata = {}
        for _, row in metadata_df.iterrows():
            param = row.get('Parameter')
            value = row.get('Value')
            
            # Skip section headers and NaN values
            if pd.isna(param) or pd.isna(value) or str(param).startswith('---'):
                continue
                
            # Clean parameter name and store value
            param_clean = str(param).strip()
            # Convert boolean-like integer fields to actual booleans
            if param_clean == 'verification_success':
                metadata[param_clean] = bool(int(value)) if not pd.isna(value) else False
            else:
                metadata[param_clean] = value

        return metadata
    except Exception as e:
        print(f"Warning: Could not read metadata from {xlsx_path}: {e}")
        return {}


def compute_mixing_time(P: pd.DataFrame, pi: np.ndarray, epsilon: float = 0.01) -> int:
    """
    Compute the mixing time of a Markov chain.

    Mixing time is the smallest t such that ||P^t - π|| < ε
    where π is the stationary distribution repeated for each row.

    Args:
        P: Transition probability matrix
        pi: Stationary distribution
        epsilon: Threshold for convergence (default: 0.01)

    Returns:
        Mixing time (number of steps)
    """
    n = len(P)
    P_array = P.values

    # Target: each row should be close to pi
    target = np.tile(pi, (n, 1))

    P_t = P_array.copy()
    for t in range(1, 10000):  # Max 10000 iterations
        # Compute total variation distance
        dist = np.max(np.abs(P_t - target))

        if dist < epsilon:
            return t

        P_t = P_t @ P_array

    return -1  # Did not converge


def _players_from_strategy_df(df: pd.DataFrame) -> List[str]:
    """Infer player list from acceptance rows in the strategy DataFrame."""
    players = []
    # Index is (state, kind, player)
    for _, kind, player in df.index:
        if kind == 'Acceptance' and pd.notna(player) and player not in players:
            players.append(str(player))
    return players


def compute_stability(
    players: List[str],
    state_names: List[str],
    static_payoffs: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, bool]]:
    """Internal / external stability of every state in the game.

    Each state is a partition of the players into coalitions (the non-singleton
    groups written in the state name, plus the implied singletons). Stability is
    evaluated on the static one-period payoffs ``u``, mirroring the classical
    d'Aspremont et al. (1983) benchmark used in ``lib/results_analysis``.

    Internal stability: no member of any coalition strictly prefers to leave,
    going it alone while the remaining members stay together (the same exit rule
    the framework's dynamics use).

    External stability: no outsider strictly prefers to join a coalition. Two
    readings are reported:

    ``external_open``
        d'Aspremont's open membership: incumbents cannot refuse, so a structure
        is stable only if no outsider would gain from accession.

    ``external_consent``
        Accession also requires all incumbents to (weakly) gain, matching the
        approval committees used in this framework. Consent can only rescue
        structures that open membership already accepts.

    Returns a dict mapping each state name to
    ``{"internal", "external_open", "external_consent"}``, or an empty dict when
    the concept is not defined for this game (see below).

    This delegates to ``lib.results_analysis.benchmarks`` so that the colours in
    the visualiser and the numbers in the written analysis cannot drift apart.
    That matters more than it sounds: the two differ on whether a *member* may
    leave one coalition and join another in a single deviation. Heyen & Lehtomaa
    (2021) take the narrow reading — a leaver stands alone — which is what the
    shared implementation applies. Reporting the broad reading here would make
    the graphs contradict the analysis they illustrate.
    """
    import pandas as pd

    from lib.results_analysis.benchmarks import PayoffGame, internal_external_stability

    frame = pd.DataFrame(
        [[static_payoffs[s][p] for p in players] for s in state_names],
        index=state_names,
        columns=players,
    )
    try:
        game = PayoffGame.from_resolved(frame, players, allow_contaminated=True)
        verdicts = internal_external_stability(game)
    except (ValueError, KeyError) as exc:
        # The benchmark is only defined when every "one coalition plus
        # singletons" structure is present. Reduced or non-canonical state
        # spaces legitimately fail that test; the frontend renders such nodes as
        # "unknown" rather than guessing a verdict.
        logger.warning("Stability colouring unavailable for this profile: %s", exc)
        return {}

    return {
        state: {
            "internal": bool(verdicts.loc[state, "internal"]),
            "external_open": bool(verdicts.loc[state, "external_open"]),
            "external_consent": bool(verdicts.loc[state, "external_consent"]),
        }
        for state in verdicts.index
    }


def compute_gamma_core(
    players: List[str],
    state_names: List[str],
    static_payoffs: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """Gamma-core membership of every single-coalition state.

    NTU gamma-core of Chander & Tulkens (1997): a coalition ``S`` deviates to the
    structure ``pi(S)`` (``S`` cooperates, everyone else is a singleton) and
    blocks the current structure iff *every* member of ``S`` is strictly better
    off there. The core is the set of structures no coalition can block. This
    delegates to ``lib.results_analysis.benchmarks.gamma_core`` so the graph
    colours cannot drift from the written analysis.

    Multi-coalition states are skipped: the gamma-characteristic function lives
    on the "one coalition plus singletons" sub-lattice. States whose verdict
    cannot be computed (missing structures, non-numeric payoffs) are omitted so
    the frontend renders them as "unknown".

    Returns a dict mapping each single-coalition state name to
    ``{"in_core": bool, "blockers": [["S", ...], ...]}``.
    """
    import pandas as pd

    from lib.results_analysis.benchmarks import PayoffGame, gamma_core

    frame = pd.DataFrame(
        [[static_payoffs[s][p] for p in players] for s in state_names],
        index=state_names,
        columns=players,
    )
    try:
        game = PayoffGame.from_resolved(frame, players, allow_contaminated=True)
        core = gamma_core(game)
    except (ValueError, KeyError) as exc:
        logger.warning("Gamma-core colouring unavailable for this profile: %s", exc)
        return {}

    unblocked = set(core["ntu_unblocked"])
    return {
        state: {
            "in_core": state in unblocked,
            "blockers": [list(b) for b in core["ntu_blockers"].get(state, [])],
        }
        for state in game.state_names
        if state in unblocked or state in core["ntu_blockers"]
    }


def compute_ricke_stability(
    players: List[str],
    state_names: List[str],
    static_payoffs: Dict[str, Dict[str, float]],
    power: Dict[str, float],
    min_power: Optional[float],
) -> Dict[str, Dict[str, Any]]:
    """Ricke, Moreno-Cruz & Caldeira (2013)'s static exclusion-game verdict per state.

    A coalition is a Ricke "winning coalition" if it holds a majority power
    share (exceeds ``min_power``) and is "stable" in their sense -- no member
    wants to leave, which (since only one coalition ever acts in their game)
    collapses onto this framework's own narrow internal-stability test. This
    delegates to ``lib.results_analysis.benchmarks.ricke_winning_coalitions``
    so the graph colouring cannot drift from the written analysis.

    Undefined (and skipped) when ``min_power`` is not set, e.g. under
    ``weak_governance`` where there is no majority-power concept to apply.

    Returns a dict mapping each single-coalition state name to
    ``{"majority": bool, "stable": bool, "winning": bool}``, where ``winning``
    is ``majority and stable`` -- Ricke's own uniqueness claim is falsified
    whenever more than one state in a game has ``winning: True``.
    """
    import pandas as pd

    from lib.results_analysis.benchmarks import PayoffGame, ricke_winning_coalitions

    if min_power is None or not power:
        return {}

    frame = pd.DataFrame(
        [[static_payoffs[s][p] for p in players] for s in state_names],
        index=state_names,
        columns=players,
    )
    try:
        game = PayoffGame.from_resolved(frame, players, allow_contaminated=True)
        verdicts = ricke_winning_coalitions(game, min_power=min_power, power=power)
    except (ValueError, KeyError) as exc:
        logger.warning("Ricke-stability colouring unavailable for this profile: %s", exc)
        return {}

    return {
        state: {
            "majority": bool(verdicts.loc[state, "majority"]),
            "stable": bool(verdicts.loc[state, "stable"]),
            "winning": bool(verdicts.loc[state, "majority"] and verdicts.loc[state, "stable"]),
        }
        for state in verdicts.index
    }


def _compute_stability_broad(
    players: List[str],
    state_names: List[str],
    static_payoffs: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, bool]]:
    """Broad-deviation variant, retained for reference and not currently used.

    Differs from the benchmark by also treating a coalition member's move into
    another existing coalition as a deviation. See Section 5 of the appendix
    draft for why this is a different concept rather than a stricter version of
    the same one.
    """
    from lib.utils import get_player_coalition

    def partition_of(state_name: str) -> List[List[str]]:
        coalitions = []
        seen = set()
        for p in players:
            coal = get_player_coalition(p, state_name, players)
            key = tuple(sorted(coal))
            if key not in seen:
                seen.add(key)
                coalitions.append(sorted(coal))
        return coalitions

    def canonical_name(partition: List[List[str]]) -> str:
        non_singletons = sorted(
            [sorted(c) for c in partition if len(c) > 1],
            key=lambda x: (len(x), x),
        )
        if not non_singletons:
            return '( )'
        return ''.join(f'({"".join(c)})' for c in non_singletons)

    canon_to_state = {
        canonical_name(partition_of(sn)): sn for sn in state_names
    }

    results = {}
    for state_name in state_names:
        partition = partition_of(state_name)
        u_state = static_payoffs[state_name]

        internal = True
        for coalition in partition:
            if len(coalition) < 2:
                continue
            for member in coalition:
                remainder = [m for m in coalition if m != member]
                new_partition = [
                    c for c in partition if sorted(c) != sorted(coalition)
                ]
                if len(remainder) >= 2:
                    new_partition.append(remainder)
                after = canon_to_state.get(canonical_name(new_partition))
                if after is None or after not in static_payoffs:
                    continue
                if u_state[member] < static_payoffs[after][member]:
                    internal = False

        external_open = True
        external_consent = True
        for coalition in partition:
            for joiner in players:
                if joiner in coalition:
                    continue
                enlarged = sorted(coalition + [joiner])
                # Build the partition after joiner leaves its current coalition
                # (if any) and joins ``coalition``. A leftover singleton is
                # implicit and dropped from the written state name.
                new_partition = []
                for c in partition:
                    if sorted(c) == sorted(coalition):
                        continue  # replaced by enlarged below
                    if joiner in c:
                        remainder = [m for m in c if m != joiner]
                        if len(remainder) >= 2:
                            new_partition.append(sorted(remainder))
                    else:
                        new_partition.append(c)
                new_partition.append(sorted(enlarged))
                after = canon_to_state.get(canonical_name(new_partition))
                if after is None or after not in static_payoffs:
                    continue
                u_after = static_payoffs[after]
                if u_after[joiner] <= u_state[joiner]:
                    continue
                external_open = False
                if all(u_after[m] >= u_state[m] for m in coalition):
                    external_consent = False

        results[state_name] = {
            "internal": internal,
            "external_open": external_open,
            "external_consent": external_consent,
        }

    return results


def compute_transition_graph(
    xlsx_path: str,
    config: Dict[str, Any] = None,
    n: int = 3
) -> Dict[str, Any]:
    """
    Compute transition graph from an XLSX strategy profile.
    Now reads metadata from the second sheet of the file to configure computation.

    Args:
        xlsx_path: Path to the XLSX strategy profile
        config: Configuration dict (uses DEFAULT_CONFIG if None, overridden by file metadata)
        n: Number of players (2-6)

    Returns:
        Dict with 'nodes', 'edges', and 'metadata'
    """
    # Always use a deep copy to avoid state pollution between requests
    if config is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
    else:
        config = copy.deepcopy(config)

    # Read metadata from file
    file_metadata = read_metadata_from_xlsx(xlsx_path)
    
    # Override config with file metadata if present
    if file_metadata:
        if 'n_players' in file_metadata:
            n = int(file_metadata['n_players'])
        if 'players' in file_metadata:
            # Parse comma-separated player list
            players_str = str(file_metadata['players']).strip()
            config['players'] = [p.strip() for p in players_str.split(',')]
        if 'states' in file_metadata:
            # Parse comma-separated state list
            states_str = str(file_metadata['states']).strip()
            config['state_names'] = [s.strip() for s in states_str.split(',')]
        if 'power_rule' in file_metadata:
            config['power_rule'] = str(file_metadata['power_rule'])
        if 'min_power' in file_metadata and not pd.isna(file_metadata['min_power']):
            config['min_power'] = float(file_metadata['min_power'])
        elif config.get('power_rule') == 'power_threshold':
            logger.warning("min_power missing from metadata for power_threshold file; defaulting to 0.501")
            config['min_power'] = 0.501
        else:
            config['min_power'] = None
        if 'unanimity_required' in file_metadata:
            # Shared with lib.verify_cli so the graph and the analysis cannot
            # disagree about the approval rule. Excel returns True as 1, which a
            # string comparison against "true" silently reads as False.
            config['unanimity_required'] = safe_bool(file_metadata['unanimity_required'])
        if 'discounting' in file_metadata:
            config['discounting'] = float(file_metadata['discounting'])
        
        # Parse player-specific parameters from metadata
        for player in config['players']:
            for param in ['base_temp', 'ideal_temp', 'delta_temp', 'm_damage', 'power', 'protocol']:
                key = f'{param}_{player}'
                if key in file_metadata and not pd.isna(file_metadata[key]):
                    if param not in config:
                        config[param] = {}
                    config[param][player] = float(file_metadata[key])

        # Fill in 0 for temperature/damage params that are absent from metadata.
        # (Files generated with --payoff-table omit these since payoffs come from
        # an external table; 0 makes it obvious the value was not provided.)
        for player in config['players']:
            for param in ['base_temp', 'ideal_temp', 'delta_temp', 'm_damage', 'power', 'protocol']:
                if param not in config:
                    config[param] = {}
                if player not in config[param]:
                    # For protocol and power, default to uniform if missing
                    if param in ('protocol', 'power'):
                        config[param][player] = 1.0 / len(config['players'])
                    else:
                        config[param][player] = 0.0
    
    # 1. Read strategy profile first to get actual state names and players
    strategy_df = pd.read_excel(xlsx_path, header=[0, 1], index_col=[0, 1, 2])

    # Infer players if not specified in metadata
    if 'players' not in file_metadata:
        actual_players = _players_from_strategy_df(strategy_df)
        if actual_players:
            config['players'] = actual_players
            n = len(actual_players)
            logger.info(f"Inferred players from strategy table: {actual_players}")
            # Ensure protocol and power are also initialized correctly for inferred players
            for player in config['players']:
                for param in ('protocol', 'power', 'base_temp', 'ideal_temp', 'delta_temp', 'm_damage'):
                    if param not in config:
                        config[param] = {}
                    if player not in config[param]:
                        if param in ('protocol', 'power'):
                            config[param][player] = 1.0 / n
                        else:
                            config[param][player] = 0.0

    # Extract actual state names from the DataFrame columns (second level of MultiIndex)
    actual_state_names = []
    for col in strategy_df.columns:
        state_name = col[1]  # Second level is the state name
        if state_name not in actual_state_names:
            actual_state_names.append(state_name)

    # Use actual state names from file, not from metadata
    config["state_names"] = actual_state_names

    # 2. Initialize countries
    all_countries = []
    for player in config["players"]:
        try:
            country = Country(
                name=player,
                base_temp=config["base_temp"][player],
                delta_temp=config["delta_temp"][player],
                ideal_temp=config["ideal_temp"][player],
                m_damage=config["m_damage"][player],
                power=config["power"][player]
            )
            all_countries.append(country)
        except KeyError as e:
            logger.error(f"Missing config parameter for player {player}: {e}")
            logger.error(f"Available config keys: {list(config.keys())}")
            raise

    # 3. Initialize coalition structures dynamically from state names
    states = []
    for state_name in config["state_names"]:
        coalitions = parse_coalition_structure(state_name, all_countries)
        state = State(
            name=state_name,
            coalitions=coalitions,
            all_countries=all_countries,
            power_rule=config["power_rule"],
            min_power=config["min_power"]
        )
        states.append(state)

    # 4. Effectivity correspondence.
    #
    # Take it from the effectivity *rule*, exactly as lib.verify_cli does when it
    # reconstructs a profile. Deriving it instead from the strategy file's NaN
    # pattern (the previous behaviour) can disagree with the rule the profile was
    # solved under, and then the graph drawn here is not the chain that was
    # solved and verified — approval committees differ, so transitions a veto
    # rules out can appear as edges. Falling back to the file pattern would
    # reintroduce exactly that silent divergence, so an unreadable rule is an
    # error rather than a guess.
    effectivity_rule = str(
        file_metadata.get("effectivity_rule") or "heyen_lehtomaa_2021"
    ).strip()
    try:
        from lib.effectivity import get_effectivity

        effectivity = get_effectivity(
            effectivity_rule, config["players"], config["state_names"]
        )
    except Exception as e:
        logger.error(f"Error building effectivity for rule {effectivity_rule!r}: {e}")
        logger.error(traceback.format_exc())
        raise

    strategy_df.fillna(0., inplace=True)

    # 5. Compute transition probabilities
    try:
        transition_probabilities = TransitionProbabilitiesOptimized(
            df=strategy_df,
            effectivity=effectivity,
            players=config["players"],
            states=config["state_names"],
            protocol=config["protocol"],
            unanimity_required=config["unanimity_required"]
        )
        P, P_proposals, P_approvals = transition_probabilities.get_probabilities()
    except Exception as e:
        logger.error(f"Error computing transition probabilities: {e}")
        logger.error(traceback.format_exc())
        raise

    # 6. Compute static payoffs and long-term value functions via MDP
    discounting = config.get("discounting", 0.99)
    static_payoffs = {state.name: state.payoffs for state in states}

    # If the Results sheet has V columns for all players, read value functions directly
    # from the file instead of recomputing via MDP. This is important when country
    # parameters are from --payoff-table RICE runs because the MDP
    # would use wrong static payoffs (u=0) and produce wrong V.
    _results_sheet = None
    _short_term_sheet = None
    try:
        _xl = pd.ExcelFile(xlsx_path)
        # Check for new sheet name first, fall back to old name for backwards compatibility
        _results_sheet_name = None
        if 'Long-term Values' in _xl.sheet_names:
            _results_sheet_name = 'Long-term Values'
        elif 'Results' in _xl.sheet_names:
            _results_sheet_name = 'Results'
        if _results_sheet_name is not None:
            _results_sheet = pd.read_excel(xlsx_path, sheet_name=_results_sheet_name, header=1, index_col=0)
        if 'Short-term Values' in _xl.sheet_names:
            _short_term_sheet = pd.read_excel(xlsx_path, sheet_name='Short-term Values', header=1, index_col=0)
    except Exception as _e:
        logger.warning(f"Could not read Long-term Values/Results sheet: {_e}")

    # Build payoff matrix: shape (n_states, n_players) ordered by config["players"]
    n_states_count = len(config["state_names"])
    payoff_matrix = np.array([
        [static_payoffs[sn][p] for p in config["players"]]
        for sn in config["state_names"]
    ])  # shape: (n_states, n_players)

    mdp = MDP(n_states=n_states_count, transition_probs=P, discounting=discounting)
    long_term_values: dict[str, dict[str, float]] = {}
    for state_name in config["state_names"]:
        long_term_values[state_name] = {}
    for pi_idx, player in enumerate(config["players"]):
        player_payoffs = payoff_matrix[:, pi_idx]
        try:
            V = mdp.solve_value_func(player_payoffs)
            for si, state_name in enumerate(config["state_names"]):
                long_term_values[state_name][player] = float(V[si])
        except Exception as e:
            logger.warning(f"Could not solve MDP for player {player}: {e}")
            for state_name in config["state_names"]:
                long_term_values[state_name][player] = None

    # Override V from Long-term Values sheet if present — more reliable than MDP-computed V
    # when country parameters are fallback values (RICE / payoff-table scenarios).
    if _results_sheet is not None:
        v_players = [p for p in config["players"] if p in _results_sheet.columns]
        if v_players:
            logger.info("Using precomputed V from Long-term Values sheet")
            for state_name in config["state_names"]:
                if state_name in _results_sheet.index:
                    for player in v_players:
                        long_term_values[state_name][player] = float(_results_sheet.loc[state_name, player])

    # Override static payoffs (u) from Short-term Values sheet if present.
    if _short_term_sheet is not None:
        u_players = [p for p in config["players"] if p in _short_term_sheet.columns]
        if u_players:
            logger.info("Using precomputed u from Short-term Values sheet")
            for state_name in config["state_names"]:
                if state_name in _short_term_sheet.index:
                    for player in u_players:
                        static_payoffs[state_name][player] = float(_short_term_sheet.loc[state_name, player])

    # 7. Get geoengineering levels and deploying coalitions for metadata
    geo_levels = {state.name: state.geo_deployment_level for state in states}

    # Standard player order: H, W, T, C, F (and A, B, D, E, G if needed)
    standard_order = ['H', 'W', 'T', 'C', 'F', 'A', 'B', 'D', 'E', 'G']

    def sort_by_standard_order(names):
        """Sort player names by standard order."""
        return sorted(names, key=lambda x: standard_order.index(x) if x in standard_order else 999)

    # Get deploying coalition for each state
    deploying_coalitions = {}
    for state in states:
        # If G=0, no one actually deploys
        if state.geo_deployment_level == 0:
            deployer_name = "None"
        else:
            strongest = state.strongest_coalition
            member_names = [country.name for country in strongest.members]

            # Format as coalition name using standard order
            if len(member_names) == 0:
                deployer_name = "( )"
            elif len(member_names) == 1:
                deployer_name = member_names[0]
            else:
                sorted_names = sort_by_standard_order(member_names)
                deployer_name = f"({''.join(sorted_names)})"

        deploying_coalitions[state.name] = deployer_name

    # Override geo_levels and deploying_coalitions from Results sheet if available
    if _results_sheet is not None:
        if 'G (°C cooling)' in _results_sheet.columns:
            for sn in config["state_names"]:
                if sn in _results_sheet.index:
                    geo_levels[sn] = float(_results_sheet.loc[sn, 'G (°C cooling)'])
        if 'Deployed by' in _results_sheet.columns:
            for sn in config["state_names"]:
                if sn in _results_sheet.index:
                    val = _results_sheet.loc[sn, 'Deployed by']
                    # pandas reads "None" as NaN; treat NaN as no deployment
                    deploying_coalitions[sn] = "None" if pd.isna(val) else str(val)

    # 8. Compute static internal/external stability of each state (used for node coloring)
    stability = compute_stability(
        players=config["players"],
        state_names=config["state_names"],
        static_payoffs=static_payoffs,
    )

    # 8b. Gamma-core membership of each single-coalition state (node coloring)
    gamma = compute_gamma_core(
        players=config["players"],
        state_names=config["state_names"],
        static_payoffs=static_payoffs,
    )

    # 8c. Ricke et al. (2013) static exclusion-game verdict of each single-coalition
    # state (node coloring) -- undefined under weak_governance (no min_power).
    ricke = compute_ricke_stability(
        players=config["players"],
        state_names=config["state_names"],
        static_payoffs=static_payoffs,
        power=config.get("power", {}),
        min_power=config.get("min_power"),
    )

    # 9. Convert to graph format
    nodes = []
    for i, state_name in enumerate(config["state_names"]):
        nodes.append({
            "id": state_name,
            "label": state_name,
            "meta": {
                "index": i,
                "geo_level": geo_levels[state_name],
                "deploying_coalition": deploying_coalitions[state_name],
                "payoffs": static_payoffs[state_name],
                "values": long_term_values[state_name],
                "stability": stability.get(state_name, {}),
                "gamma_core": gamma.get(state_name, {}),
                "ricke": ricke.get(state_name, {}),
            }
        })

    # 10. Build edges
    edges = []
    edge_id = 0
    for i, source_state in enumerate(config["state_names"]):
        for j, target_state in enumerate(config["state_names"]):
            prob = P.iloc[i, j]
            if prob > 0:  # Only include edges with positive probability
                # Compute breakdown: which proposers and approval patterns contribute
                breakdown = []
                is_self_loop = source_state == target_state

                def _get_committee_info(proposer, src, tgt):
                    """Get approval committee details for a proposer/transition."""
                    committee = []
                    not_in_committee = []
                    approvals = {}
                    for responder in config["players"]:
                        eff_key = (proposer, src, tgt, responder)
                        if eff_key in transition_probabilities.effectivity and transition_probabilities.effectivity[eff_key] == 1:
                            committee.append(responder)
                            try:
                                col_key = (f"Proposer {proposer}", tgt)
                                acc_prob = transition_probabilities.df.loc[(src, 'Acceptance', responder), col_key]
                                if pd.notna(acc_prob):
                                    approvals[responder] = float(acc_prob)
                            except:
                                pass
                        else:
                            not_in_committee.append(responder)
                    return committee, not_in_committee, approvals

                for proposer in config["players"]:
                    if is_self_loop:
                        # For self-loops, show ALL contributions: direct stay proposals
                        # AND rejected proposals for other states that cause staying
                        for other_state in config["state_names"]:
                            other_key = (proposer, source_state, other_state)
                            if other_key not in P_proposals or P_proposals[other_key] <= 0:
                                continue
                            if other_key not in P_approvals:
                                continue

                            committee, not_in_committee, approvals = _get_committee_info(
                                proposer, source_state, other_state
                            )

                            if other_state == source_state:
                                # Direct "propose to stay" path
                                path_prob = transition_probabilities.protocol[proposer] * P_proposals[other_key] * P_approvals[other_key]
                                if path_prob > 0:
                                    breakdown.append({
                                        "type": "direct",
                                        "proposer": proposer,
                                        "proposed_target": other_state,
                                        "prop_prob": float(P_proposals[other_key]),
                                        "approval_prob": float(P_approvals[other_key]),
                                        "path_prob": float(path_prob),
                                        "committee": committee,
                                        "not_in_committee": not_in_committee,
                                        "approvals": approvals
                                    })
                            else:
                                # Rejected proposal: proposer wanted other_state but got rejected
                                p_rejected = 1.0 - P_approvals[other_key]
                                path_prob = transition_probabilities.protocol[proposer] * P_proposals[other_key] * p_rejected
                                if path_prob > 1e-12:
                                    breakdown.append({
                                        "type": "rejection",
                                        "proposer": proposer,
                                        "proposed_target": other_state,
                                        "prop_prob": float(P_proposals[other_key]),
                                        "approval_prob": float(P_approvals[other_key]),
                                        "rejection_prob": float(p_rejected),
                                        "path_prob": float(path_prob),
                                        "committee": committee,
                                        "not_in_committee": not_in_committee,
                                        "approvals": approvals
                                    })
                    else:
                        # Non-self-loop: standard breakdown (proposer proposes this target)
                        prop_key = (proposer, source_state, target_state)
                        if prop_key in P_proposals and P_proposals[prop_key] > 0:
                            app_key = prop_key
                            if app_key in P_approvals:
                                committee, not_in_committee, approvals = _get_committee_info(
                                    proposer, source_state, target_state
                                )
                                path_prob = transition_probabilities.protocol[proposer] * P_proposals[prop_key] * P_approvals[app_key]

                                breakdown.append({
                                    "type": "direct",
                                    "proposer": proposer,
                                    "proposed_target": target_state,
                                    "prop_prob": float(P_proposals[prop_key]),
                                    "approval_prob": float(P_approvals[app_key]),
                                    "path_prob": float(path_prob),
                                    "committee": committee,
                                    "not_in_committee": not_in_committee,
                                    "approvals": approvals
                                })

                # Sort breakdown: largest contributions first
                breakdown.sort(key=lambda x: -x["path_prob"])

                edges.append({
                    "id": f"e{edge_id}",
                    "source": source_state,
                    "target": target_state,
                    "p": float(prob),
                    "meta": {
                        "is_self_loop": source_state == target_state,
                        "breakdown": breakdown
                    }
                })
                edge_id += 1

    # 11. Compute stationary distribution, mixing time, and expected geoengineering level
    try:
        pi = limiting_distribution(P)
        # E_π[G] = Σ π_i * G_i
        G_values = np.array([geo_levels[state_name] for state_name in config["state_names"]])
        expected_G = float(np.dot(pi, G_values))

        # Create stationary distribution dict
        pi_dict = {state_name: float(pi[i]) for i, state_name in enumerate(config["state_names"])}

        # Detect absorbing sets for diagnostics (must come before mixing/absorption time)
        closed_classes = absorbing_sets(P)
        absorbing_state_sets = [
            [config["state_names"][i] for i in members] for members in closed_classes
        ]

        # Ergodic iff the whole state space is one closed communicating class.
        is_ergodic = (
            len(closed_classes) == 1
            and len(closed_classes[0]) == len(config["state_names"])
        )

        # Compute mixing time (only meaningful for ergodic chains)
        mixing_time = compute_mixing_time(P, pi) if is_ergodic else None

        # For non-ergodic chains with absorbing sets, compute absorption time
        absorption_time = None
        if not is_ergodic and len(absorbing_state_sets) > 0:
            absorbing_indices = {i for members in closed_classes for i in members}
            transient_indices = [i for i in range(len(P)) if i not in absorbing_indices]

            if len(transient_indices) > 0:
                Q = P.values[np.ix_(transient_indices, transient_indices)]
                try:
                    N = np.linalg.inv(np.eye(len(transient_indices)) - Q)
                    t_absorb = N @ np.ones(len(transient_indices))
                    absorption_time = {
                        "max": float(np.max(t_absorb)),
                        "mean": float(np.mean(t_absorb)),
                        "by_state": {config["state_names"][transient_indices[i]]: float(t_absorb[i])
                                     for i in range(len(transient_indices))}
                    }
                except np.linalg.LinAlgError:
                    pass
        
    except Exception as e:
        print(f"Warning: Could not compute stationary distribution: {e}")
        import traceback as _tb
        _tb.print_exc()
        expected_G = None
        mixing_time = None
        absorption_time = None
        pi_dict = None
        absorbing_state_sets = []
        is_ergodic = None

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "profile_path": xlsx_path,
            "num_players": len(config["players"]),
            "num_states": len(nodes),
            "num_transitions": len(edges),
            "expected_geo_level": expected_G,
            "stationary_distribution": pi_dict,
            "mixing_time": mixing_time,
            "scenario_name": file_metadata.get("scenario_name"),
            "scenario_description": file_metadata.get("scenario_description"),
            "config": {
                "power_rule": config["power_rule"],
                "unanimity_required": config["unanimity_required"],
                "min_power": config["min_power"]
            },
            "file_metadata": file_metadata,  # Include all file metadata
            "absorption_time": absorption_time,
            "chain_diagnostics": {
                "is_ergodic": is_ergodic,
                "num_absorbing_sets": len(absorbing_state_sets),
                "absorbing_sets": absorbing_state_sets if absorbing_state_sets else None
            }
        }
    }


def _sanitize_json(obj: Any) -> Any:
    """Recursively replace NaN values with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    elif isinstance(obj, float) or isinstance(obj, np.float64):
        return None if np.isnan(obj) else obj
    return obj


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "Coalition Formation Visualizer API",
        "endpoints": {
            "/graph": "GET - Compute transition graph from XLSX profile",
            "/profiles": "GET - List available strategy profiles"
        }
    }


@app.get("/graph")
async def get_graph(
    profile: str = Query(..., description="Path to XLSX strategy profile (relative or absolute)")
):
    """
    Compute and return transition graph from an XLSX strategy profile.
    All configuration parameters (n, power_rule, min_power, unanimity) are now read from the file's Metadata sheet.

    Recomputes on every request - no caching.
    """
    try:
        # Resolve path
        profile_path = Path(profile)

        # If not absolute and doesn't exist, try prepending the default profiles dir
        if not profile_path.is_absolute() and not profile_path.exists():
            profile_path = Path(DEFAULT_PROFILES_DIR) / profile_path

        if not profile_path.exists():
            logger.error(f"Profile not found: {profile_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found: {profile_path}"
            )

        # Compute graph (configuration is read from file metadata)
        graph_data = compute_transition_graph(str(profile_path))

        # Sanitize for JSON compliance (replace NaN with None)
        return _sanitize_json(graph_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing graph for {profile}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error computing graph: {str(e)}\n\n{traceback.format_exc()}"
        )


@app.get("/profiles")
async def list_profiles(profiles_dir: str = Query(DEFAULT_PROFILES_DIR, description="Directory containing XLSX profiles")):
    """
    List available strategy profile XLSX files.
    """
    try:
        profiles_path = Path(profiles_dir)
        if not profiles_path.exists():
            return {"profiles": [], "error": f"Directory not found: {profiles_dir}"}

        xlsx_files = list(profiles_path.glob("*.xlsx"))
        # Filter out lock files
        xlsx_files = [f for f in xlsx_files if not f.name.startswith(".~lock")]

        profiles = sorted(
            [
                {
                    "name": f.stem,
                    "path": str(f),
                    "filename": f.name,
                    "created_at": f.stat().st_mtime,
                }
                for f in xlsx_files
            ],
            key=lambda p: p["created_at"],
            reverse=True
        )

        return {"profiles": profiles}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing profiles: {str(e)}"
        )


@app.get("/download")
async def download_profile(
    profile: str = Query(..., description="Path to XLSX strategy profile (relative or absolute)")
):
    """
    Download an XLSX strategy profile file.
    """
    try:
        from fastapi.responses import FileResponse

        # Resolve path
        profile_path = Path(profile)

        # If not absolute and doesn't exist, try prepending the default profiles dir
        if not profile_path.is_absolute() and not profile_path.exists():
            profile_path = Path(DEFAULT_PROFILES_DIR) / profile_path

        if not profile_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found: {profile_path}"
            )

        return FileResponse(
            path=str(profile_path),
            filename=profile_path.name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading profile: {str(e)}"
        )


def main():
    """CLI entry point for the visualization service."""
    parser = argparse.ArgumentParser(
        description="Coalition Formation Visualization Service"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--profiles-dir",
        default=DEFAULT_PROFILES_DIR,
        help=f"Directory containing XLSX strategy profiles (default: {DEFAULT_PROFILES_DIR})"
    )

    args = parser.parse_args()

    print(f"Starting Coalition Formation Visualizer API")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Profiles dir: {args.profiles_dir}")
    print(f"  API docs: http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
