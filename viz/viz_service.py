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
from typing import Dict, List, Any

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

# Default profiles directory (absolute path to repository's strategy_tables)
DEFAULT_PROFILES_DIR = str(Path(__file__).parent.parent / "strategy_tables")

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


def find_strongly_connected_components(adjacency_matrix: np.ndarray) -> list:
    """Find strongly connected components using Tarjan's algorithm."""
    n = adjacency_matrix.shape[0]
    index_counter = [0]
    stack = []
    lowlink = [0] * n
    index = [0] * n
    on_stack = [False] * n
    index_initialized = [False] * n
    sccs = []
    
    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        index_initialized[v] = True
        stack.append(v)
        on_stack[v] = True
        
        # Consider successors
        for w in range(n):
            if adjacency_matrix[v, w] > 0:  # There's an edge from v to w
                if not index_initialized[w]:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])
        
        # If v is a root node, pop the stack and generate an SCC
        if lowlink[v] == index[v]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == v:
                    break
            sccs.append(scc)
    
    for v in range(n):
        if not index_initialized[v]:
            strongconnect(v)
    
    return sccs


def compute_stationary_distribution(P: pd.DataFrame) -> np.ndarray:
    """
    Compute the limiting distribution of a Markov chain.
    
    For chains with absorbing sets, computes the absorption probabilities
    assuming a uniform initial distribution over all states.

    Args:
        P: Transition probability matrix (DataFrame)

    Returns:
        Stationary distribution as numpy array
    """
    n = len(P)
    P_array = P.values
    
    # Find strongly connected components
    sccs = find_strongly_connected_components(P_array)
    
    # Identify absorbing sets (SCCs with no outgoing edges)
    absorbing_sets = []
    for scc in sccs:
        is_absorbing = True
        for i in scc:
            for j in range(n):
                if j not in scc and P_array[i, j] > 1e-10:
                    is_absorbing = False
                    break
            if not is_absorbing:
                break
        if is_absorbing:
            absorbing_sets.append(scc)
    
    # If there are absorbing sets, compute absorption probabilities
    if len(absorbing_sets) > 0:
        # Flatten absorbing sets to get all absorbing states
        absorbing_states = []
        for abs_set in absorbing_sets:
            absorbing_states.extend(abs_set)
        
        logger.info(f"Found {len(absorbing_sets)} absorbing sets with {len(absorbing_states)} total absorbing states")
        for i, abs_set in enumerate(absorbing_sets):
            set_states = [P.index[idx] for idx in abs_set]
            logger.info(f"  Absorbing set {i+1}: {set_states}")
        
        # Partition states into transient (T) and absorbing (A)
        transient_states = [i for i in range(n) if i not in absorbing_states]
        logger.info(f"Found {len(transient_states)} transient states")
        
        if len(transient_states) == 0:
            # All states are in absorbing sets
            # Assume uniform initial distribution
            pi = np.zeros(n)
            initial_uniform = np.ones(n) / n
            
            for abs_set in absorbing_sets:
                # Compute stationary distribution within this absorbing set
                P_sub = P_array[np.ix_(abs_set, abs_set)]
                try:
                    n_sub = len(abs_set)
                    A_sub = P_sub.T - np.eye(n_sub)
                    A_sub[-1, :] = np.ones(n_sub)
                    b_sub = np.zeros(n_sub)
                    b_sub[-1] = 1.0
                    pi_sub = np.linalg.solve(A_sub, b_sub)
                    pi_sub = np.maximum(pi_sub, 0)
                    pi_sub = pi_sub / pi_sub.sum()
                    
                    # Weight by initial probability mass in this set
                    initial_mass = sum(initial_uniform[i] for i in abs_set)
                    for i, state_idx in enumerate(abs_set):
                        pi[state_idx] = pi_sub[i] * initial_mass
                except:
                    # Fallback: uniform within set
                    for state_idx in abs_set:
                        pi[state_idx] = initial_uniform[state_idx]
            
            pi = pi / pi.sum()
            return pi
        
        # Extract Q (transient-to-transient) submatrix
        Q = P_array[np.ix_(transient_states, transient_states)]
        
        # Compute fundamental matrix N = (I - Q)^(-1)
        try:
            I = np.eye(len(transient_states))
            N = np.linalg.inv(I - Q)
            
            # For each absorbing set, compute absorption probability
            pi = np.zeros(n)
            initial_uniform = np.ones(n) / n
            
            for abs_set in absorbing_sets:
                # R matrix: transient -> this absorbing set
                R_set = P_array[np.ix_(transient_states, abs_set)]
                
                # Absorption probabilities: B_set = N * R_set
                B_set = N @ R_set
                
                # From transient states
                for i, t_idx in enumerate(transient_states):
                    # Total probability of absorbing into this set
                    total_abs_prob = B_set[i, :].sum()
                    
                    if total_abs_prob > 1e-10:
                        # Compute stationary distribution within the absorbing set
                        P_sub = P_array[np.ix_(abs_set, abs_set)]
                        try:
                            n_sub = len(abs_set)
                            A_sub = P_sub.T - np.eye(n_sub)
                            A_sub[-1, :] = np.ones(n_sub)
                            b_sub = np.zeros(n_sub)
                            b_sub[-1] = 1.0
                            pi_sub = np.linalg.solve(A_sub, b_sub)
                            pi_sub = np.maximum(pi_sub, 0)
                            pi_sub = pi_sub / pi_sub.sum()
                            
                            # Distribute probability according to stationary distribution within set
                            for j, a_idx in enumerate(abs_set):
                                pi[a_idx] += initial_uniform[t_idx] * total_abs_prob * pi_sub[j]
                        except:
                            # Fallback: uniform within set
                            for a_idx in abs_set:
                                pi[a_idx] += initial_uniform[t_idx] * total_abs_prob / len(abs_set)
            
            # States already in absorbing sets
            for abs_set in absorbing_sets:
                P_sub = P_array[np.ix_(abs_set, abs_set)]
                try:
                    n_sub = len(abs_set)
                    A_sub = P_sub.T - np.eye(n_sub)
                    A_sub[-1, :] = np.ones(n_sub)
                    b_sub = np.zeros(n_sub)
                    b_sub[-1] = 1.0
                    pi_sub = np.linalg.solve(A_sub, b_sub)
                    pi_sub = np.maximum(pi_sub, 0)
                    pi_sub = pi_sub / pi_sub.sum()
                    
                    initial_mass = sum(initial_uniform[i] for i in abs_set)
                    for i, state_idx in enumerate(abs_set):
                        pi[state_idx] += pi_sub[i] * initial_mass
                except:
                    for state_idx in abs_set:
                        pi[state_idx] += initial_uniform[state_idx]
            
            # Normalize
            pi = pi / pi.sum()
            
            # Log final distribution by absorbing set
            logger.info("Final stationary distribution by absorbing set:")
            for i, abs_set in enumerate(absorbing_sets):
                set_prob = sum(pi[idx] for idx in abs_set)
                set_states = [P.index[idx] for idx in abs_set]
                logger.info(f"  Set {i+1} ({set_states}): {set_prob*100:.2f}%")
            
            return pi
            
        except np.linalg.LinAlgError:
            # Fall back to standard method if fundamental matrix computation fails
            pass
    
    # Standard method for ergodic chains (no absorbing states, or fallback)
    # Solve (P^T - I)π = 0 with constraint Σπ = 1
    A = P_array.T - np.eye(n)

    # Replace last equation with normalization constraint Σπ = 1
    A[-1, :] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(A, b)
        # Ensure non-negative and normalized (numerical errors)
        pi = np.maximum(pi, 0)
        pi = pi / pi.sum()
        return pi
    except np.linalg.LinAlgError:
        # If singular, use eigenvalue method
        eigenvalues, eigenvectors = np.linalg.eig(P_array.T)
        # Find eigenvector for eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)  # Ensure non-negative
        pi = pi / pi.sum()  # Normalize
        return pi


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

    # 4. Derive effectivity
    try:
        effectivity = derive_effectivity(
            df=strategy_df,
            players=config["players"],
            states=config["state_names"]
        )
    except Exception as e:
        logger.error(f"Error deriving effectivity: {e}")
        logger.error(traceback.format_exc())
        raise

    strategy_df.fillna(0., inplace=True)

    # 5. Compute transition probabilities
    try:
        transition_probabilities = TransitionProbabilities(
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

    # 6. Get geoengineering levels and deploying coalitions for metadata
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

    # 7. Convert to graph format
    nodes = []
    for i, state_name in enumerate(config["state_names"]):
        nodes.append({
            "id": state_name,
            "label": state_name,
            "meta": {
                "index": i,
                "geo_level": geo_levels[state_name],
                "deploying_coalition": deploying_coalitions[state_name]
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

    # 8. Compute stationary distribution, mixing time, and expected geoengineering level
    try:
        pi = compute_stationary_distribution(P)
        # E_π[G] = Σ π_i * G_i
        G_values = np.array([geo_levels[state_name] for state_name in config["state_names"]])
        expected_G = float(np.dot(pi, G_values))

        # Compute mixing time
        mixing_time = compute_mixing_time(P, pi)

        # Create stationary distribution dict
        pi_dict = {state_name: float(pi[i]) for i, state_name in enumerate(config["state_names"])}
        
        # Detect absorbing sets for diagnostics
        sccs = find_strongly_connected_components(P.values)
        absorbing_sets = []
        for scc in sccs:
            is_absorbing = True
            for i in scc:
                for j in range(len(P)):
                    if j not in scc and P.values[i, j] > 1e-10:
                        is_absorbing = False
                        break
                if not is_absorbing:
                    break
            if is_absorbing:
                # Convert indices to state names
                absorbing_sets.append([config["state_names"][i] for i in scc])
        
        # Check if chain is ergodic (single SCC containing all states)
        is_ergodic = len(sccs) == 1 and len(sccs[0]) == len(config["state_names"])
        
    except Exception as e:
        print(f"Warning: Could not compute stationary distribution: {e}")
        import traceback
        traceback.print_exc()
        expected_G = None
        mixing_time = None
        pi_dict = None
        absorbing_sets = []
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
            "chain_diagnostics": {
                "is_ergodic": is_ergodic,
                "num_absorbing_sets": len(absorbing_sets),
                "absorbing_sets": absorbing_sets if len(absorbing_sets) > 0 else None
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

        return graph_data

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
