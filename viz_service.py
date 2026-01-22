"""
Lightweight visualization service for coalition formation transition graphs.
Exposes HTTP endpoints to compute and serve transition probability graphs from XLSX strategy profiles.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from lib.country import Country
from lib.coalition import Coalition
from lib.state import State
from lib.probabilities import TransitionProbabilities
from lib.utils import derive_effectivity


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


def compute_transition_graph(
    xlsx_path: str,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Compute transition graph from an XLSX strategy profile.

    Args:
        xlsx_path: Path to the XLSX strategy profile
        config: Configuration dict (uses DEFAULT_CONFIG if None)

    Returns:
        Dict with 'nodes', 'edges', and 'metadata'
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # 1. Initialize countries
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

    # 2. Initialize coalition structures
    W, T, C = all_countries
    coalition_map = {
        '( )': [Coalition([W]), Coalition([T]), Coalition([C])],
        '(TC)': [Coalition([W]), Coalition([T, C])],
        '(WC)': [Coalition([T]), Coalition([W, C])],
        '(WT)': [Coalition([C]), Coalition([W, T])],
        '(WTC)': [Coalition([W, T, C])]
    }

    states = [State(
        name=name,
        coalitions=coalition_map[name],
        all_countries=all_countries,
        power_rule=config["power_rule"],
        min_power=config["min_power"]
    ) for name in config["state_names"]]

    # 3. Read strategy profile
    strategy_df = pd.read_excel(xlsx_path, header=[0, 1], index_col=[0, 1, 2])

    effectivity = derive_effectivity(
        df=strategy_df,
        players=config["players"],
        states=config["state_names"]
    )
    strategy_df.fillna(0., inplace=True)

    # 4. Compute transition probabilities
    transition_probabilities = TransitionProbabilities(
        df=strategy_df,
        effectivity=effectivity,
        players=config["players"],
        states=config["state_names"],
        protocol=config["protocol"],
        unanimity_required=config["unanimity_required"]
    )
    P, P_proposals, P_approvals = transition_probabilities.get_probabilities()

    # 5. Get geoengineering levels for metadata
    geo_levels = {state.name: state.geo_deployment_level for state in states}

    # 6. Convert to graph format
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
            "num_states": len(nodes),
            "num_transitions": len(edges),
            "config": {
                "power_rule": config["power_rule"],
                "unanimity_required": config["unanimity_required"],
                "min_power": config["min_power"]
            }
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
    profile: str = Query(..., description="Path to XLSX strategy profile (relative or absolute)"),
    power_rule: str = Query("weak_governance", description="Power rule: 'weak_governance' or 'power_threshold'"),
    min_power: float = Query(None, description="Minimum power threshold (for power_threshold rule)"),
    unanimity: bool = Query(True, description="Whether unanimity is required for approval")
):
    """
    Compute and return transition graph from an XLSX strategy profile.

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

        # Build config with user parameters
        config = DEFAULT_CONFIG.copy()
        config["power_rule"] = power_rule
        config["min_power"] = min_power if power_rule == "power_threshold" else None
        config["unanimity_required"] = unanimity

        # Compute graph
        graph_data = compute_transition_graph(str(profile_path), config)

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
