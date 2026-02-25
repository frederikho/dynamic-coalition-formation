#!/usr/bin/env python3
"""
Build script to precompute all transition graphs from XLSX strategy profiles.
Generates static JSON files for frontend consumption.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from viz.service_viz import compute_transition_graph


def build_static_data(
    strategy_tables_dir: str = "strategy_tables",
    output_dir: str = "viz/data"
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

    # Remove stale generated profile artifacts from prior runs.
    # Keep unrelated static JSON files (e.g., eq_*.json) untouched.
    # Also preserve any manual flags (e.g. "hidden": true) set in the previous manifest.
    previous_profile_json_files = set()
    previous_profile_xlsx_files = set()
    previous_profile_flags = {}   # filename -> dict of flags to preserve (e.g. {"hidden": True})
    profiles_path = output_path / "profiles.json"
    if profiles_path.exists():
        try:
            with open(profiles_path, 'r') as f:
                previous_manifest = json.load(f)
            for profile in previous_manifest.get("profiles", []):
                filename = profile.get("filename")
                if filename:
                    previous_profile_json_files.add(filename)
                    # Preserve hidden flag (and any future manual flags)
                    flags = {k: v for k, v in profile.items() if k == "hidden"}
                    if flags:
                        previous_profile_flags[filename] = flags
                path_from_manifest = profile.get("path")
                if path_from_manifest:
                    previous_profile_xlsx_files.add(Path(path_from_manifest).name)
                elif filename:
                    previous_profile_xlsx_files.add(f"{Path(filename).stem}.xlsx")
        except Exception as e:
            print(f"Warning: Could not read existing profiles manifest for cleanup: {e}")

    current_profile_json_files = {f"{xlsx_file.stem}.json" for xlsx_file in xlsx_files}
    current_profile_xlsx_files = {xlsx_file.name for xlsx_file in xlsx_files}

    stale_json_files = previous_profile_json_files - current_profile_json_files
    stale_xlsx_files = previous_profile_xlsx_files - current_profile_xlsx_files

    for stale_json in sorted(stale_json_files):
        stale_json_path = output_path / stale_json
        if stale_json_path.exists():
            stale_json_path.unlink()
            print(f"Removed stale profile JSON: {stale_json}")

    for stale_xlsx in sorted(stale_xlsx_files):
        stale_xlsx_path = xlsx_output_path / stale_xlsx
        if stale_xlsx_path.exists():
            stale_xlsx_path.unlink()
            print(f"Removed stale profile XLSX copy: {stale_xlsx}")

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

            # Extract end_time from file metadata as a stable created_at timestamp.
            # Filesystem mtime is unreliable after git clone (all files get clone time),
            # but end_time is embedded in the file content and survives git.
            file_meta = graph_data["metadata"].get("file_metadata", {})
            end_time_str = file_meta.get("end_time", "")
            created_at = 0
            if end_time_str:
                try:
                    created_at = datetime.strptime(str(end_time_str), "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    pass

            # Add to profiles list, restoring any manual flags from the previous manifest
            entry = {
                "name": xlsx_file.stem,
                "filename": json_filename,
                "path": str(xlsx_file),
                "created_at": created_at,
                "num_states": graph_data["metadata"]["num_states"],
                "num_transitions": graph_data["metadata"]["num_transitions"],
                "scenario_name": graph_data["metadata"].get("scenario_name"),
                "scenario_description": graph_data["metadata"].get("scenario_description")
            }
            if json_filename in previous_profile_flags:
                entry.update(previous_profile_flags[json_filename])
            profiles.append(entry)

            file_size = json_path.stat().st_size
            print(f"✓ ({file_size/1024:.1f} KB)")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Sort by created_at descending (most recently computed first) before writing manifest
    profiles.sort(key=lambda p: p["created_at"], reverse=True)

    # Write profiles manifest
    with open(profiles_path, 'w') as f:
        json.dump({"profiles": profiles}, f, indent=2)

    print()
    print(f"✓ Generated {len(profiles)} profile JSON files")
    print(f"✓ Wrote manifest to {profiles_path}")

    # Calculate total size
    total_size = sum((output_path / p["filename"]).stat().st_size for p in profiles)
    total_size += profiles_path.stat().st_size
    print(f"✓ Total size: {total_size/1024:.1f} KB")
    print()
    print("Tip: To test the static site locally, serve the repository root over HTTP:")
    print("  python3 -m http.server 8765")
    print("Then open http://127.0.0.1:8765/ in your browser.")
    print("If you also built the frontend with --vite-build, this serves the production artifacts.")


def run_vite_build(vite_base_path: str):
    """
    Build the frontend with Vite for GitHub Pages deployment.
    
    Args:
        vite_base_path: Base path for Vite build (e.g., '/repo-name/')
    """
    print()
    print("=" * 80)
    print("Building frontend with Vite...")
    print("=" * 80)
    
    viz_dir = Path(__file__).parent / "viz"
    env = os.environ.copy()
    env["VITE_BASE_PATH"] = vite_base_path
    
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=viz_dir,
            env=env,
            check=True
        )
        print()
        print(f"✓ Frontend build complete!")
        print(f"✓ Output directory: repository root (index.html) with assets/data in viz/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Vite build failed: {e}")
        return False
    except FileNotFoundError:
        print("✗ npm not found. Please install Node.js and npm.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build static data and optionally rebuild frontend with Vite"
    )
    parser.add_argument(
        "--vite-build",
        action="store_true",
        help="Also run Vite build for GitHub Pages (uses REPO_NAME from .env)"
    )
    parser.add_argument(
        "--vite-base",
        type=str,
        default=None,
        help="Base path for Vite build (default: /REPO_NAME/ from .env, or /)"
    )
    
    args = parser.parse_args()
    
    # Build static data first
    build_static_data()
    
    # Optionally build frontend
    if args.vite_build:
        # Determine base path
        if args.vite_base:
            vite_base_path = args.vite_base
        else:
            # Try to read from .env
            repo_name = os.getenv("REPO_NAME")
            if repo_name:
                vite_base_path = f"/{repo_name}/"
            else:
                vite_base_path = "/"
        
        success = run_vite_build(vite_base_path)
        sys.exit(0 if success else 1)
