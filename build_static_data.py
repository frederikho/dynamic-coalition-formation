#!/usr/bin/env python3
"""
Build script to precompute all transition graphs from XLSX strategy profiles.
Generates static JSON files for frontend consumption.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from viz.viz_service import compute_transition_graph


def build_static_data(
    strategy_tables_dir: str = "strategy_tables",
    output_dir: str = "viz/public/data"
):
    """
    Precompute all XLSX strategy profiles to JSON files.

    Args:
        strategy_tables_dir: Directory containing XLSX files
        output_dir: Directory to write JSON files
    """
    strategy_dir = Path(strategy_tables_dir)
    output_path = Path(output_dir)

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    xlsx_output_path = output_path / "xlsx"
    xlsx_output_path.mkdir(parents=True, exist_ok=True)

    # Find all XLSX files (excluding lock files)
    xlsx_files = [
        f for f in strategy_dir.glob("*.xlsx")
        if not f.name.startswith(".~lock")
    ]

    print(f"Found {len(xlsx_files)} XLSX files in {strategy_tables_dir}/")
    print()

    # Precompute each file
    profiles = []
    for xlsx_file in sorted(xlsx_files):
        print(f"Processing {xlsx_file.name}...", end=" ")

        try:
            # Compute transition graph
            graph_data = compute_transition_graph(str(xlsx_file))

            # Generate output filename (same as xlsx but .json)
            json_filename = xlsx_file.stem + ".json"
            json_path = output_path / json_filename

            # Write JSON file
            with open(json_path, 'w') as f:
                json.dump(graph_data, f, indent=2)

            # Copy XLSX file for download
            xlsx_copy_path = xlsx_output_path / xlsx_file.name
            shutil.copy2(xlsx_file, xlsx_copy_path)

            # Add to profiles list
            profiles.append({
                "name": xlsx_file.stem,
                "filename": json_filename,
                "path": str(xlsx_file),
                "num_states": graph_data["metadata"]["num_states"],
                "num_transitions": graph_data["metadata"]["num_transitions"],
                "scenario_name": graph_data["metadata"].get("scenario_name"),
                "scenario_description": graph_data["metadata"].get("scenario_description")
            })

            file_size = json_path.stat().st_size
            print(f"✓ ({file_size/1024:.1f} KB)")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Write profiles manifest
    profiles_path = output_path / "profiles.json"
    with open(profiles_path, 'w') as f:
        json.dump({"profiles": profiles}, f, indent=2)

    print()
    print(f"✓ Generated {len(profiles)} profile JSON files")
    print(f"✓ Wrote manifest to {profiles_path}")

    # Calculate total size
    total_size = sum((output_path / p["filename"]).stat().st_size for p in profiles)
    total_size += profiles_path.stat().st_size
    print(f"✓ Total size: {total_size/1024:.1f} KB")


if __name__ == "__main__":
    build_static_data()
