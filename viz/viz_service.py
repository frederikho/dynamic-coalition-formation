"""
Lightweight visualization service for coalition formation transition graphs.
Exposes HTTP endpoints to compute and serve transition probability graphs from XLSX strategy profiles.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path to import lib modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.probabilities import TransitionProbabilities
from lib.utils import derive_effectivity


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

# Default configuration (matches main.py base_config)
DEFAULT_CONFIG = {
    "players": ["W", "T", "C"],
    "base_temp": {"W": 21.5, "T": 14.0, "C": 11.5},
    "ideal_temp": {"W": 13.0, "T": 13.0, "C": 13.0},
    "delta_temp": {"W": 3.0, "T": 3.0, "C": 3.0},
    "power": {"W": 1/3, "T": 1/3, "C": 1/3},
    "protocol": {"W": 1/3, "T": 1/3, "C": 1/3},
    "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
    "power_rule": "weak_governance",
    "min_power": None,
    "state_names": ['( )', '(TC)', '(WC)', '(WT)', '(WTC)'],
    "unanimity_required": True,
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
    # Create a mapping from country names to country objects
    country_map = {c.name: c for c in all_countries}
    
    if state_name == '( )':
        # All singletons
        return [Coalition([c]) for c in all_countries]
    
    # Parse coalitions from the state name
    # Remove outer spaces and split by ')(' to get individual coalitions
    coalitions_str = state_name.strip().strip('(').strip(')')
    
    if not coalitions_str:
        # Empty means all singletons
        return [Coalition([c]) for c in all_countries]
    
    # Split by ')(' to find multiple coalitions
    coalition_parts = []
    current = ""
    depth = 0
    for char in state_name:
        if char == '(':
            depth += 1
            if depth == 1:
                current = ""
        elif char == ')':
            depth -= 1
            if depth == 0 and current:
                coalition_parts.append(current)
                current = ""
        elif depth == 1:
            current += char
    
    # Build set of countries in coalitions
    countries_in_coalitions = set()
    coalitions = []
    
    for part in coalition_parts:
        # Each character is a country name
        coalition_countries = []
        for country_name in part:
            if country_name in country_map:
                coalition_countries.append(country_map[country_name])
                countries_in_coalitions.add(country_name)
        if coalition_countries:
            coalitions.append(Coalition(coalition_countries))
    
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
        # Read the Metadata sheet
        metadata_df = pd.read_excel(xlsx_path, sheet_name='Metadata')
        
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
            metadata[param_clean] = value
        
        return metadata
    except Exception as e:
        print(f"Warning: Could not read metadata from {xlsx_path}: {e}")
        return {}


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
        else:
            config['min_power'] = None
        if 'unanimity_required' in file_metadata:
            unanimity_val = file_metadata['unanimity_required']
            config['unanimity_required'] = unanimity_val if isinstance(unanimity_val, bool) else str(unanimity_val).lower() == 'true'
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
    
    # 1. Read strategy profile first to get actual state names from columns
    strategy_df = pd.read_excel(xlsx_path, header=[0, 1], index_col=[0, 1, 2])
    
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
        country = Country(
            name=player,
            base_temp=config["base_temp"][player],
            delta_temp=config["delta_temp"][player],
            ideal_temp=config["ideal_temp"][player],
            m_damage=config["m_damage"][player],
            power=config["power"][player]
        )
        all_countries.append(country)

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

    # 4. Derive effectivity
    effectivity = derive_effectivity(
        df=strategy_df,
        players=config["players"],
        states=config["state_names"]
    )
    strategy_df.fillna(0., inplace=True)

    # 5. Compute transition probabilities
    transition_probabilities = TransitionProbabilities(
        df=strategy_df,
        effectivity=effectivity,
        players=config["players"],
        states=config["state_names"],
        protocol=config["protocol"],
        unanimity_required=config["unanimity_required"]
    )
    P, P_proposals, P_approvals = transition_probabilities.get_probabilities()

    # 6. Get geoengineering levels for metadata
    geo_levels = {state.name: state.geo_deployment_level for state in states}

    # 7. Convert to graph format
    nodes = []
    for i, state_name in enumerate(config["state_names"]):
        nodes.append({
            "id": state_name,
            "label": state_name,
            "meta": {
                "index": i,
                "geo_level": geo_levels[state_name]
            }
        })

    edges = []
    edge_id = 0
    for i, source_state in enumerate(config["state_names"]):
        for j, target_state in enumerate(config["state_names"]):
            prob = P.iloc[i, j]
            if prob > 0:  # Only include edges with positive probability
                edges.append({
                    "id": f"e{edge_id}",
                    "source": source_state,
                    "target": target_state,
                    "p": float(prob),
                    "meta": {
                        "is_self_loop": source_state == target_state
                    }
                })
                edge_id += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "profile_path": xlsx_path,
            "num_players": len(config["players"]),
            "num_states": len(nodes),
            "num_transitions": len(edges),
            "config": {
                "power_rule": config["power_rule"],
                "unanimity_required": config["unanimity_required"],
                "min_power": config["min_power"]
            },
            "file_metadata": file_metadata  # Include all file metadata
        }
    }


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

        # If not absolute, check if it exists as-is first
        if not profile_path.is_absolute():
            if not profile_path.exists():
                # Try prepending strategy_tables/ if it doesn't start with it
                if not str(profile_path).startswith("strategy_tables"):
                    profile_path = Path("strategy_tables") / profile_path

        if not profile_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found: {profile_path}"
            )

        # Compute graph (configuration is read from file metadata)
        graph_data = compute_transition_graph(str(profile_path))

        return graph_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error computing graph: {str(e)}"
        )


@app.get("/profiles")
async def list_profiles(profiles_dir: str = Query("strategy_tables", description="Directory containing XLSX profiles")):
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

        profiles = [
            {
                "name": f.stem,
                "path": str(f),
                "filename": f.name
            }
            for f in sorted(xlsx_files)
        ]

        return {"profiles": profiles}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing profiles: {str(e)}"
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
        default="strategy_tables",
        help="Directory containing XLSX strategy profiles (default: strategy_tables)"
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
