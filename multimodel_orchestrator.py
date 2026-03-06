#!/usr/bin/env python3
"""
Model orchestrator for multi-period dynamic coalition formation analysis.

Runs RICE50x GAMS models and the coalition equilibrium finder across multiple
time periods, chaining the SAI deployment history from one period to the next.

Usage (activate .venv first):
    source .venv/bin/activate
    python orchestrate.py \\
        --periods 2035-2060 2060-2080 2080-2100 \\
        --impact burke \\
        --countries usa chn nde \\
        --policy bau_impact \\
        [--scenario power_threshold_RICE_n3] \\
        [--rice-dir /home/frederik/Code/RICE50x] \\
        [--max-workers 4]

        
The script runs every step silently in the background — only high-level progress
is shown on the console.  Detailed GAMS logs are written to the GAMS workdir.
"""

import argparse
import concurrent.futures
import logging
import pickle
import shlex
import subprocess
import sys
import io
import time
import shutil
from contextlib import redirect_stdout
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

COALITION_DIR = Path(__file__).parent.resolve()
DEFAULT_RICE_DIR = Path("/home/frederik/Code/RICE50x")

# Make lib.* importable regardless of working directory
if str(COALITION_DIR) not in sys.path:
    sys.path.insert(0, str(COALITION_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(msg, flush=True)


def _elapsed(t0: float) -> str:
    """Format elapsed time since t0 as 'H:MM:SS'."""
    secs = int(time.monotonic() - t0)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _section(title: str) -> None:
    _log(f"\n{'━' * 64}")
    _log(f"  {title}")
    _log(f"{'━' * 64}")


def _step(n: int, total: int, label: str, t0: Optional[float] = None) -> None:
    elapsed = f"  ({_elapsed(t0)} elapsed)" if t0 is not None else ""
    _log(f"\n[{n}/{total}] {label}{elapsed}")


def _skip(label: str) -> None:
    _log(f"  →  Skipped — {label}  (use --fresh to rerun)")


def _ok(label: str) -> None:
    _log(f"  ✓  {label}")


def _warn(label: str) -> None:
    _log(f"  ⚠  {label}")


# ─────────────────────────────────────────────────────────────────────────────
# Winning coalition export + plot
# ─────────────────────────────────────────────────────────────────────────────

def export_winning_coalition_gdx(
    source_gdx: Path,
    rice_dir: Path,
    policy: str,
    impact: str,
    countries_slug: str,
    start_year: int,
    end_year: int,
) -> Path:
    dest_dir = rice_dir / "results" / "winning_coalitions"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_name = (
        f"results_ssp2_{policy}_{impact}_{countries_slug}_"
        f"winning_coalitions_{start_year}-{end_year}.gdx"
    )
    dest_path = dest_dir / dest_name
    shutil.copy2(source_gdx, dest_path)
    return dest_path


def compute_sai_timeseries(
    gdx_path: Path,
    start_year: int,
    end_year: int,
) -> tuple[list[int], list[float]]:
    from lib.ingest_payoffs import (
        parse_gdx_parameter,
        parse_gdx_variable_levels,
        SymbolNotFoundError,
        YEAR_SYMBOL,
        SAI_SYMBOL,
        _in_year_range,
    )

    year_map = parse_gdx_parameter(gdx_path, YEAR_SYMBOL)
    try:
        sai_levels = parse_gdx_variable_levels(gdx_path, SAI_SYMBOL)
    except SymbolNotFoundError:
        sai_levels = {}

    rows: list[tuple[int, float]] = []
    for period_key, calendar_year in year_map.items():
        if _in_year_range(calendar_year, start_year, end_year):
            year_int = int(round(calendar_year))
            rows.append((year_int, float(sai_levels.get(period_key, 0.0))))

    if not rows:
        raise RuntimeError(
            f"No W_SAI values found in {gdx_path.name} for years "
            f"{start_year}–{end_year}."
        )

    rows.sort(key=lambda x: x[0])
    t_values = [t for t, _ in rows]
    y_values = [y for _, y in rows]
    return t_values, y_values


def _period_sizes(t_values: list[int], periods: list[tuple[int, int]]) -> list[int]:
    sizes: list[int] = []
    for start, end in periods:
        count = sum(1 for t in t_values if start <= t <= end)
        sizes.append(max(1, count))
    return sizes


def _build_period_label_line(width: int, sizes: list[int], labels: list[str]) -> str:
    line = [" "] * width
    total = sum(sizes)
    starts = [0]
    for size in sizes[:-1]:
        starts.append(starts[-1] + size)
    ends = starts[1:] + [total]
    for i, label in enumerate(labels):
        left = int(round(starts[i] / total * width))
        right = int(round(ends[i] / total * width))
        center = (left + right) // 2
        pos = max(left, min(center - len(label) // 2, width - len(label)))
        for j, ch in enumerate(label):
            if 0 <= pos + j < width:
                line[pos + j] = ch
        if i < len(labels) - 1 and 0 <= right < width:
            line[right] = "|"
    return "".join(line)


def _build_centered_segment_label_line(width: int, sizes: list[int], labels: list[str]) -> str:
    """Center one label in each period segment without drawing separators."""
    line = [" "] * width
    total = sum(sizes)
    starts = [0]
    for size in sizes[:-1]:
        starts.append(starts[-1] + size)
    ends = starts[1:] + [total]
    for i, label in enumerate(labels):
        left = int(round(starts[i] / total * width))
        right = int(round(ends[i] / total * width))
        center = (left + right) // 2
        pos = max(left, min(center - len(label) // 2, width - len(label)))
        for j, ch in enumerate(label):
            if 0 <= pos + j < width:
                line[pos + j] = ch
    return "".join(line)


def render_sai_plot_to_ascii(
    t_values: list[int],
    y_values: list[float],
    periods: list[tuple[int, int]],
    absorbing_labels: Optional[list[str]] = None,
) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib is required to render the ASCII plot. "
            "Install it with: pip install matplotlib"
        ) from exc

    term_width = max(80, shutil.get_terminal_size((100, 30)).columns)
    ascii_width = min(140, term_width - 2)
    ascii_height = 28

    fig_w = ascii_width / 10.0
    fig_h = ascii_height / 5.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100, facecolor="black")
    ax.set_facecolor("black")

    ax.plot(t_values, y_values, color="#ffffff", linewidth=4.6, marker="o", markersize=5.5)
    ax.set_xlim(min(t_values), max(t_values))
    y_min = min(y_values)
    y_max = max(y_values)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.margins(x=0.02, y=0.0)

    for spine in ax.spines.values():
        spine.set_color("#a8a8a8")
        spine.set_linewidth(1.1)
    ax.tick_params(colors="#c8c8c8", labelsize=8)

    boundaries = [end for _, end in periods[:-1]]
    for boundary in boundaries:
        ax.axvline(boundary, color="#9a9a9a", linestyle="--", linewidth=1.0, alpha=0.7)

    xticks = [t_values[0]] + boundaries + [t_values[-1]]
    ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(int(y_min // 10) * 10, int(y_max // 10 + 1) * 10, 5))

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.17)
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    gray = (0.2126 * rgba[:, :, 0] + 0.7152 * rgba[:, :, 1] + 0.0722 * rgba[:, :, 2]).astype(np.uint8)

    h, w = gray.shape
    block_h = max(1, h // ascii_height)
    block_w = max(1, w // ascii_width)
    cropped_h = block_h * ascii_height
    cropped_w = block_w * ascii_width
    cropped = gray[:cropped_h, :cropped_w]
    sampled = cropped.reshape(ascii_height, block_h, ascii_width, block_w).mean(axis=(1, 3))

    ramp = np.array(list(" .:-=+*#%@"))
    bucket_idx = (sampled.astype(np.float32) / 255.0 * (len(ramp) - 1)).astype(int)
    bucket_idx = np.clip(bucket_idx, 0, len(ramp) - 1)
    lines = ["".join(ramp[row]) for row in bucket_idx]
    # Remove excessive blank rows at the bottom so period labels are directly readable.
    while lines and not lines[-1].strip():
        lines.pop()

    labels = [f"{s}-{e}" for s, e in periods]
    sizes = _period_sizes(t_values, periods)
    period_line = _build_period_label_line(ascii_width, sizes, labels)
    out_lines = lines + [period_line]
    if absorbing_labels is not None:
        coalition_line = _build_centered_segment_label_line(
            ascii_width, sizes, absorbing_labels
        )
        out_lines.append(coalition_line)
    return "\n".join(out_lines)

# ─────────────────────────────────────────────────────────────────────────────
# GAMS execution
# ─────────────────────────────────────────────────────────────────────────────

# Substrings that indicate a GAMS model error even when the exit code is 0.
# GAMS occasionally exits 0 for infeasible models with a status message.
_GAMS_ERROR_MARKERS = [
    "*** ERROR",
    "COMPILATION ERROR",
    "EXECUTION ERROR",
    "Exit code: 3",   # resource/time limit
    "Exit code: 2",   # execution error
]


def _check_gams_log_for_errors(log_path: Path) -> Optional[str]:
    """Scan a GAMS log file for known error markers.

    Returns the first suspicious line, or None if the log looks clean.
    """
    try:
        with open(log_path) as f:
            for line in f:
                for marker in _GAMS_ERROR_MARKERS:
                    if marker in line:
                        return line.rstrip()
    except OSError:
        pass
    return None


def _tail(path: Path, n: int = 40) -> str:
    """Return last n lines of a file as a single string."""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except OSError:
        return "(could not read file)"


def _log_contains(path: Path, needle: str) -> bool:
    """Return True if `needle` appears anywhere in file `path`."""
    try:
        with open(path) as f:
            return needle in f.read()
    except OSError:
        return False


def _coalition_lookup_explanation(
    cmd_args: list[str], returncode: int, log_path: Path
) -> Optional[str]:
    """Return a tentative user-facing hint for failed coalition lookup runs."""
    if returncode != 144:
        return None
    if "--cooperation=coalitions" not in cmd_args:
        return None
    if "--sel_coalition=sai_farsighted" not in cmd_args:
        return None
    if not _log_contains(
        log_path,
        "*** Abandoned SolveLink=6 instance with handle 1 (submitted, but not solved)",
    ):
        return None

    coalition_code: Optional[str] = None
    for arg in cmd_args:
        if arg.startswith("--sai_coalition="):
            coalition_code = arg.split("=", 1)[1]
            break

    if coalition_code:
        return (
            "Possible cause: coalition lookup issue. "
            f"Check whether the coalition '{coalition_code}' exists in "
            "'coalitions/coal_sai_farsighted.gms'."
        )
    return (
        "Possible cause: coalition lookup issue. "
        "Check whether the requested coalition exists in "
        "'coalitions/coal_sai_farsighted.gms'."
    )


def _run_one_gams_job(cmd_args: list[str], cwd: Path, log_path: Path) -> None:
    """Run one GAMS command, directing all output to log_path.

    Raises RuntimeError on failure (non-zero exit code *or* GAMS error in log).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd_args,
            cwd=str(cwd),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    error_line = _check_gams_log_for_errors(log_path)
    cmd_str = f"cd {cwd} && " + " ".join(cmd_args)

    if proc.returncode != 0:
        snippet = _tail(log_path)
        coalition_hint = _coalition_lookup_explanation(
            cmd_args, proc.returncode, log_path
        )
        raise RuntimeError(
            f"GAMS exited with code {proc.returncode}.\n"
            f"Command: {cmd_str}\n"
            f"Log: {log_path}\n"
            + (f"{coalition_hint}\n" if coalition_hint else "")
            + (f"Error marker: {error_line}\n" if error_line else "")
            + f"Last 40 lines of log:\n{snippet}"
        )

    if error_line:
        snippet = _tail(log_path)
        raise RuntimeError(
            f"GAMS exited 0 but log contains an error marker.\n"
            f"Command: {cmd_str}\n"
            f"Log: {log_path}\n"
            f"Error marker: {error_line}\n"
            f"Last 40 lines of log:\n{snippet}"
        )


def _build_gams_jobs(
    countries: list[str],
    impact: str,
    policy: str,
    gams_workdir: str,
    sai_history_gdx: Optional[str] = None,
    sai_history_from: Optional[int] = None,
    extra_gams_args: Optional[list[str]] = None,
) -> list[tuple[str, list[str]]]:
    """Build all GAMS command lines for one phase.

    Returns a list of (label, cmd_args) tuples.
    The label is a short identifier used for log file naming.
    """
    history_extra: list[str] = []
    if sai_history_gdx is not None:
        history_extra = [
            f"--load_sai_history={sai_history_gdx}",
            f"--load_sai_history_until={sai_history_from}",
        ]

    base = [
        "gams", "run_rice50x.gms",
        "--mod_sai=1",
        f"--impact={impact}",
        f"--policy={policy}",
        f"--workdir={gams_workdir}",
    ] + history_extra
    if extra_gams_args:
        base += extra_gams_args

    jobs: list[tuple[str, list[str]]] = []

    # 1. Baseline: no SAI deployment (--can_deploy=no).
    #    Without this explicit flag, mod_sai.gms defaults to can_deploy="usa",
    #    which would produce _usa_deployed.gdx instead of the no-deployment GDX.
    #    (When --load_sai_history is set, GAMS already defaults to "no", but
    #    we set it explicitly here for consistency across all phases.)
    jobs.append(("baseline", base + ["--can_deploy=no"]))

    # 2. Unilateral (noncoop) runs — one per country
    for country in countries:
        cmd = base + [
            "--cooperation=noncoop",
            f"--can_deploy={country}",
        ]
        jobs.append((f"noncoop_{country}", cmd))

    # 3. Coalition runs — all subsets of size 2 up to N (including grand coalition)
    for size in range(2, len(countries) + 1):
        for subset in combinations(countries, size):
            coalition_code = "".join(subset)
            cmd = base + [
                "--cooperation=coalitions",
                "--sel_coalition=sai_farsighted",
                f"--sai_coalition={coalition_code}",
            ]
            jobs.append((f"coalition_{coalition_code}", cmd))

    return jobs


def _rename_no_deployed_gdx(workdir: Path) -> None:
    """Rename *_no_deployed.gdx → the same name without '_no_deployed'.

    mod_sai.gms appends '_no_deployed' when can_deploy="no".  The resulting
    file has no recognisable country token, so discover_state_files() can't
    map it to '( )'.  Renaming to strip '_no_deployed' leaves just the common
    prefix (e.g. results_ssp2_bau_impact_burke.gdx), which maps to '( )' via
    the empty-deployer-token path in discover_state_files().
    """
    renamed = 0
    for gdx in workdir.glob("*_no_deployed.gdx"):
        new_name = gdx.name.replace("_no_deployed", "")
        dest = gdx.parent / new_name
        gdx.rename(dest)
        _log(f"  Renamed {gdx.name}  →  {new_name}  (baseline / '( )' state)")
        renamed += 1
    if renamed == 0:
        raise RuntimeError(
            f"No '*_no_deployed.gdx' found in {workdir} after GAMS phase.\n"
            "Expected the baseline run (--can_deploy=no) to produce one."
        )


def _gams_outputs_exist(workdir: Path, n_countries: int) -> bool:
    """Return True if all expected GAMS GDX outputs already exist in workdir.

    Expected count: 2^N files (1 baseline + N noncoop + all coalition subsets).
    Also verifies the _no_deployed rename has already happened.
    """
    if not workdir.exists():
        return False
    gdx_files = list(workdir.glob("*.gdx"))
    n_expected = 2 ** n_countries
    if len(gdx_files) < n_expected:
        return False
    # Rename step must have already run (no *_no_deployed.gdx remaining)
    if any("_no_deployed" in f.name for f in gdx_files):
        return False
    return True


def run_gams_phase(
    countries: list[str],
    impact: str,
    policy: str,
    rice_dir: Path,
    workdir: Path,
    sai_history_gdx: Optional[Path] = None,
    sai_history_from: Optional[int] = None,
    max_workers: int = 4,
    extra_gams_args: Optional[list[str]] = None,
) -> None:
    """Run all GAMS simulations for one phase in parallel.

    Args:
        countries:       RICE50x country codes (lowercase), e.g. ['usa', 'chn', 'nde'].
        impact:          Impact function name, e.g. 'burke'.
        policy:          Policy name, e.g. 'bau_impact'.
        rice_dir:        Absolute path to the RICE50x directory.
        workdir:         Absolute path where GAMS writes output GDX files.
        sai_history_gdx: Absolute path to previous-phase GDX file (or None).
        sai_history_from:Year from which to load SAI history (= end year of prev phase).
        max_workers:     Maximum number of parallel GAMS jobs.
        extra_gams_args: Additional '--key=value' args appended to every GAMS run.

    Raises:
        RuntimeError: if any GAMS job fails.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "gams_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    history_arg = str(sai_history_gdx) if sai_history_gdx is not None else None
    jobs = _build_gams_jobs(
        countries,
        impact,
        policy,
        str(workdir),
        history_arg,
        sai_history_from,
        extra_gams_args,
    )

    _log(f"  Submitting {len(jobs)} GAMS jobs (max_workers={max_workers})...")

    # Track failures; we want to see ALL that fail, not just the first
    failures: list[tuple[str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(
                _run_one_gams_job,
                cmd,
                rice_dir,
                log_dir / f"{label}.log",
            ): label
            for label, cmd in jobs
        }
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            exc = future.exception()
            if exc is not None:
                _log(f"  ✗  {label}")
                failures.append((label, str(exc)))
            else:
                _ok(label)

    if failures:
        details = "\n\n".join(
            f"  [{label}]:\n{msg}" for label, msg in failures
        )
        raise RuntimeError(
            f"{len(failures)} GAMS job(s) failed:\n\n{details}"
        )

    # Rename *_no_deployed.gdx so discover_state_files() can map it to '( )'
    _rename_no_deployed_gdx(workdir)


# ─────────────────────────────────────────────────────────────────────────────
# Payoff ingest
# ─────────────────────────────────────────────────────────────────────────────

def run_ingest(
    gdx_dir: Path,
    payoff_table_path: Path,
    start_year: Optional[int],
    end_year: int,
    policy: str,
    impact: str,
    extra_metadata: Optional[dict[str, object]] = None,
) -> None:
    """Ingest welfare and W_SAI from GDX files into a payoff table Excel file.

    Args:
        gdx_dir:           Directory containing the GDX files from GAMS.
        payoff_table_path: Destination .xlsx file path.
        start_year:        First year to include (None = no lower bound).
        end_year:          Last year to include (inclusive).
        policy:            Policy name used to filter matching GDX filenames.
        impact:            Impact function used to filter matching GDX filenames.
        extra_metadata:    Optional metadata key/value pairs for the Excel file.
    """
    from lib.ingest_payoffs import ingest

    _log(f"  GDX source dir:  {gdx_dir}")
    _log(f"  Payoff table:    {payoff_table_path}")
    stem_prefix = f"results_ssp2_{policy}_{impact}"
    _log(f"  GDX filter:      stem starts with '{stem_prefix}'")

    buf = io.StringIO()
    with redirect_stdout(buf):
        ingest(
            input_dir=gdx_dir,
            output_path=payoff_table_path,
            players=None,          # auto-detect from filenames
            start_year=start_year,
            end_year=end_year,
            extra_metadata=extra_metadata,
            required_stem_prefix=stem_prefix,
        )

    # Surface the key lines from ingest output
    for line in buf.getvalue().splitlines():
        stripped = line.strip()
        if any(kw in stripped for kw in [
            "Auto-detected", "Wrote payoff", "Skipped", "Warning", "ordering check",
            "Found", "Summing",
        ]):
            _log(f"    {stripped}")


# ─────────────────────────────────────────────────────────────────────────────
# Equilibrium finder
# ─────────────────────────────────────────────────────────────────────────────

def _make_file_only_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes only to a file (no console/Rich output).

    find_equilibrium accepts a logger kwarg; by passing our own file-only
    logger we prevent the Rich console handler from printing solver internals
    to stdout during orchestration.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    name = f"orch_{log_path.stem}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False  # don't bubble up to the root logger

    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logger.addHandler(fh)
    return logger


def _build_equilibrium_command(
    scenario: str,
    payoff_table_path: Path,
    output_file: Path,
) -> str:
    """Build the equivalent CLI command for reproducing one solver run."""
    cmd_args = [
        "python3",
        str(COALITION_DIR / "find_equilibrium.py"),
        scenario,
        "--payoff-table",
        str(payoff_table_path),
        "--output",
        str(output_file),
        "--fresh",
    ]
    return " ".join(shlex.quote(arg) for arg in cmd_args)


def run_find_equilibrium(
    scenario: str,
    payoff_table_path: Path,
    output_file: Path,
    log_dir: Path,
) -> dict:
    """Run the equilibrium solver and return the result dictionary.

    The result contains (among other keys):
      'P'             : transition probability DataFrame (states × states)
      'geoengineering': DataFrame indexed by state name, column 'G' = W_SAI
      'state_names'   : list of framework state names
      'players'       : list of player names
      'verification_success': bool

    Args:
        scenario:          Equilibrium scenario name (e.g. 'power_threshold_RICE_n3').
        payoff_table_path: Absolute path to the ingested payoff table.
        output_file:       Where to save the strategy Excel file.
        log_dir:           Directory for the solver log file.

    Returns:
        Result dict from find_equilibrium().

    Raises:
        RuntimeError: if equilibrium verification fails.
    """
    from lib.equilibrium.find import (
        find_equilibrium,
        _parse_players_from_payoff_table,
    )
    from lib.equilibrium.scenarios import get_scenario, fill_players

    config = get_scenario(scenario)

    # Inject players from payoff table filename if not hardcoded in scenario
    if config.get("players") is None:
        players = _parse_players_from_payoff_table(payoff_table_path)
        config = fill_players(config, players)

    config["payoff_table"] = str(payoff_table_path)

    _log(f"  Scenario:  {scenario}")
    _log(f"  Players:   {config['players']}")
    _log(f"  Output:    {output_file}")
    eq_cmd = _build_equilibrium_command(scenario, payoff_table_path, output_file)

    # Build a file-only logger so the Rich console handler is bypassed.
    # All solver progress is written to this log file instead of the terminal.
    log_path = log_dir / f"solver_{output_file.stem}.log"
    solver_logger = _make_file_only_logger(log_path)
    _log(f"  Solver log: {log_path}")
    _log(f"  Command:    {eq_cmd}")
    _log(f"  (running — may take several minutes)")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = find_equilibrium(
        config=config,
        output_file=str(output_file),
        # Keep terminal quiet via file-only logger, but enable verbose logging
        # so the solver writes progress into the log file.
        verbose=True,
        load_from_checkpoint=False,   # always fresh (equivalent to --fresh)
        logger=solver_logger,
    )

    verified = result.get("verification_success", False)
    msg = result.get("verification_message", "")
    status = "PASSED" if verified else "FAILED"
    _log(f"  Verification: {status}  —  {msg}")

    if not verified:
        raise RuntimeError(
            f"Equilibrium verification FAILED for '{output_file.stem}'.\n"
            f"Message: {msg}\n"
            f"Check solver log: {log_path}\n"
            f"Equilibrium command: {eq_cmd}"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Absorbing state extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_absorbing_states(
    P: pd.DataFrame,
    geo_levels: pd.DataFrame,
    tol: float = 1e-6,
) -> tuple[list[str], list[float]]:
    """Find absorbing states (P[s,s] ≈ 1) and their W_SAI values.

    Args:
        P:          Transition probability DataFrame (state × state).
        geo_levels: DataFrame indexed by state name, column 'G' = W_SAI.
        tol:        Tolerance for considering P[s,s] == 1.

    Returns:
        (absorbing_states, sai_values) — parallel lists.
    """
    absorbing: list[str] = []
    sai_values: list[float] = []

    for state in P.index:
        p_stay = float(P.loc[state, state])
        if abs(p_stay - 1.0) < tol:
            absorbing.append(state)
            g = float(geo_levels.loc[state, "G"]) if geo_levels is not None else 0.0
            sai_values.append(g)

    return absorbing, sai_values


def _check_sai_consistency(
    absorbing_states: list[str],
    sai_values: list[float],
    tol: float = 1e-3,
) -> float:
    """Assert all absorbing states share the same W_SAI value.

    Returns the common value.
    Raises ValueError if values differ or no absorbing states exist.
    """
    if not absorbing_states:
        raise ValueError("No absorbing states found.")

    ref = sai_values[0]
    differing = [
        f"{s}={v:.6f}"
        for s, v in zip(absorbing_states, sai_values)
        if abs(v - ref) > tol
    ]
    if differing:
        all_vals = ", ".join(
            f"{s}={v:.6f}" for s, v in zip(absorbing_states, sai_values)
        )
        raise ValueError(
            f"Absorbing states have different W_SAI values: {all_vals}\n"
            "Cannot determine which SAI history to load for the next phase."
        )

    return ref


# ─────────────────────────────────────────────────────────────────────────────
# GDX history lookup
# ─────────────────────────────────────────────────────────────────────────────

def find_history_gdx(
    gdx_dir: Path,
    absorbing_state: str,
    policy: str,
    impact: str,
) -> Path:
    """Find the GDX file to load as SAI history in the next phase.

    Always returns a GDX — even when the absorbing state is '( )' (no new
    deployment).  Loading the '( )' GDX tells RICE50x that W_SAI was zero in
    the previous period, constraining its optimizer so it does not freely
    deploy SAI in those years as part of a longer-horizon optimal path.

    Args:
        gdx_dir:         Directory containing GDX files from the GAMS phase.
        absorbing_state: Framework state name, e.g. '( )' or '(CHNNDE)'.
        policy:          Policy name used to filter matching GDX filenames.
        impact:          Impact function used to filter matching GDX filenames.

    Returns:
        Absolute Path to the GDX file for the absorbing coalition.

    Raises:
        RuntimeError: if the expected GDX file is not found.
    """
    from lib.ingest_payoffs import discover_state_files

    stem_prefix = f"results_ssp2_{policy}_{impact}"
    key_to_file, skipped = discover_state_files(
        gdx_dir,
        required_stem_prefix=stem_prefix,
    )

    # For power_threshold with equal power, the deployer coalition equals the
    # state coalition, so the deployer key matches the framework state name.
    deployer_key = absorbing_state

    if deployer_key not in key_to_file:
        available = list(key_to_file.keys())
        raise RuntimeError(
            f"Cannot find GDX file for deployer key '{deployer_key}' in {gdx_dir}.\n"
            f"Available deployer keys: {available}"
        )

    gdx_path = key_to_file[deployer_key]
    _log(f"  SAI history GDX: {gdx_path.name}"
         + ("  (no new deployment — constrains next phase to W_SAI=0)"
            if absorbing_state == "( )" else ""))
    return gdx_path


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration loop
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate(
    periods: list[tuple[int, int]],
    impact: str,
    countries: list[str],
    policy: str,
    scenario: str = "power_threshold_RICE_n3",
    rice_dir: Path = DEFAULT_RICE_DIR,
    max_workers: int = 4,
    fresh: bool = False,
    extra_gams_args: Optional[list[str]] = None,
    payoff_metadata: Optional[dict[str, object]] = None,
    payoff_range: Optional[tuple[Optional[int], int]] = None,
) -> list[dict]:
    """Run the full multi-period orchestration.

    For each period:
      1. Run all GAMS simulations in parallel (with SAI history from previous phase).
      2. Ingest GDX outputs into a payoff table.
      3. Find the coalition equilibrium.
      4. Extract absorbing states and their W_SAI values.
      5. Determine the SAI history GDX for the next phase.

    Args:
        periods:     List of (start_year, end_year) tuples.
        impact:      Impact function (e.g. 'burke', 'andreoni', 'kalkuhl').
        countries:   RICE50x country codes, lowercase (e.g. ['usa', 'chn', 'nde']).
        policy:      Policy name (e.g. 'bau_impact', 'bau', 'cba').
        scenario:    Equilibrium scenario name.
        rice_dir:    Absolute path to the RICE50x repository.
        max_workers: Maximum parallel GAMS jobs per phase.
        fresh:       If True, re-run every step even if outputs already exist.
        extra_gams_args: Additional '--key=value' args appended to every GAMS run.
        payoff_metadata: Extra key/value rows to write into payoff metadata.
        payoff_range: Optional payoff summation range override:
                      (None, end) means [period_start, end] per phase;
                      (start, end) means fixed [start, end] for all phases.

    Returns:
        List of per-period result dicts with keys:
          'period'           : str  (e.g. '2035-2060')
          'absorbing_states' : list[str]
          'w_sai'            : float
          'history_gdx'      : str or None
    """
    t0 = time.monotonic()

    countries_slug = "".join(countries)
    payoff_dir  = COALITION_DIR / "payoff_tables"
    strategy_dir = COALITION_DIR / "strategy_tables"
    payoff_dir.mkdir(parents=True, exist_ok=True)
    strategy_dir.mkdir(parents=True, exist_ok=True)

    _section("Orchestration parameters")
    _log(f"  Impact:       {impact}")
    _log(f"  Countries:    {countries}  (slug: {countries_slug})")
    _log(f"  Policy:       {policy}")
    _log(f"  Scenario:     {scenario}")
    _log(f"  Periods:      {[f'{s}-{e}' for s, e in periods]}")
    _log(f"  RICE dir:     {rice_dir}")
    _log(f"  Max workers:  {max_workers}")
    _log(f"  Fresh:        {fresh}")
    if payoff_range is not None:
        pr_start, pr_end = payoff_range
        if pr_start is None:
            _log(f"  Payoff range: [period_start, {pr_end}]")
        else:
            _log(f"  Payoff range: [{pr_start}, {pr_end}]")
    if extra_gams_args:
        _log(f"  Extra GAMS args: {extra_gams_args}")
    if payoff_metadata:
        _log(f"  Payoff metadata extras: {payoff_metadata}")

    if not rice_dir.exists():
        raise RuntimeError(f"RICE50x directory not found: {rice_dir}")

    phase_results: list[dict] = []
    history_gdx: Optional[Path] = None
    history_from: Optional[int] = None
    last_winning_state: Optional[str] = None
    last_winning_gdx: Optional[Path] = None
    last_period_label: Optional[str] = None

    n_phases = len(periods)
    for phase_idx, (period_start, period_end) in enumerate(periods):
        period_label = f"{period_start}-{period_end}"
        slug = f"{impact}_{countries_slug}_{period_label}"

        if payoff_range is None:
            ingest_start, ingest_end = period_start, period_end
            payoff_range_label = period_label
        else:
            pr_start, pr_end = payoff_range
            ingest_start = period_start if pr_start is None else pr_start
            ingest_end = pr_end
            if ingest_start > ingest_end:
                raise RuntimeError(
                    f"Invalid payoff range for phase {period_label}: "
                    f"{ingest_start}-{ingest_end} (start > end)."
                )
            payoff_range_label = f"{ingest_start}-{ingest_end}"

        _section(f"Phase {phase_idx + 1}/{n_phases}  ·  {period_label}")

        gams_workdir   = rice_dir / "results" / slug
        if payoff_range is None:
            payoff_table = payoff_dir / f"{slug}.xlsx"
        else:
            payoff_table = payoff_dir / f"{slug}_payoff_{payoff_range_label}.xlsx"
        strategy_file  = strategy_dir / f"{slug}.xlsx"
        result_cache   = strategy_dir / f"{slug}.pkl"
        solver_log_dir = COALITION_DIR / "logs"

        # ── Step 1: GAMS simulations ─────────────────────────────────────────
        _step(1, 4, "Running GAMS simulations", t0)
        if not fresh and _gams_outputs_exist(gams_workdir, len(countries)):
            _skip(f"GDX outputs already in {gams_workdir.name}")
        else:
            run_gams_phase(
                countries=countries,
                impact=impact,
                policy=policy,
                rice_dir=rice_dir,
                workdir=gams_workdir,
                sai_history_gdx=history_gdx,
                sai_history_from=history_from,
                max_workers=max_workers,
                extra_gams_args=extra_gams_args,
            )

        # ── Step 2: Ingest payoffs ───────────────────────────────────────────
        _step(2, 4, "Ingesting payoffs from GDX files", t0)
        if not fresh and payoff_table.exists():
            _skip(f"payoff table already exists: {payoff_table.name}")
        else:
            _log(f"  Ingest years:     {ingest_start}-{ingest_end}")
            run_ingest(
                gdx_dir=gams_workdir,
                payoff_table_path=payoff_table,
                start_year=ingest_start,
                end_year=ingest_end,
                policy=policy,
                impact=impact,
                extra_metadata=payoff_metadata,
            )

        # ── Step 3: Find equilibrium ─────────────────────────────────────────
        _step(3, 4, "Finding coalition equilibrium", t0)
        if not fresh and strategy_file.exists() and result_cache.exists():
            _skip(f"strategy table and result cache already exist: {strategy_file.name}")
            try:
                with open(result_cache, "rb") as _f:
                    result = pickle.load(_f)
                if not isinstance(result, dict):
                    raise ValueError("cached result is not a dict")
                if not result.get("verification_success", False):
                    raise RuntimeError(
                        f"Cached result for '{slug}' has verification_success=False.\n"
                        f"Delete {result_cache} or rerun with --fresh."
                    )
            except RuntimeError:
                # Keep existing behavior for explicitly failed verification.
                raise
            except Exception as exc:
                _warn(
                    f"Could not load cached result ({type(exc).__name__}: {exc}). "
                    "Recomputing equilibrium and refreshing cache."
                )
                result = run_find_equilibrium(
                    scenario=scenario,
                    payoff_table_path=payoff_table,
                    output_file=strategy_file,
                    log_dir=solver_log_dir,
                )
                with open(result_cache, "wb") as _f:
                    pickle.dump(result, _f)
        else:
            result = run_find_equilibrium(
                scenario=scenario,
                payoff_table_path=payoff_table,
                output_file=strategy_file,
                log_dir=solver_log_dir,
            )
            with open(result_cache, "wb") as _f:
                pickle.dump(result, _f)

        # ── Step 4: Extract absorbing states ─────────────────────────────────
        _step(4, 4, "Extracting absorbing states", t0)
        P          = result["P"]
        geo_levels = result["geoengineering"]

        absorbing, sai_values = extract_absorbing_states(P, geo_levels)

        if not absorbing:
            raise RuntimeError(
                f"No absorbing states found for period {period_label}!\n"
                f"Transition matrix:\n{P}"
            )

        _log(f"  Absorbing state(s): {absorbing}")
        _log(f"  W_SAI value(s):     {[f'{v:.4f}' for v in sai_values]}")

        # Check all absorbing states share the same W_SAI
        w_sai = _check_sai_consistency(absorbing, sai_values)
        _log(f"  Common W_SAI:       {w_sai:.4f}")

        if len(absorbing) > 1:
            _warn(
                f"Multiple absorbing states detected ({absorbing}). "
                f"Using '{absorbing[0]}' for next phase's SAI history."
            )

        # Determine GDX for the next phase's SAI history.
        # Always pass a GDX — even '( )' (no deployment) — so RICE is
        # constrained to W_SAI=0 from the previous period and does not
        # re-optimize SAI deployment in years that have already been decided.
        winning_state = absorbing[0]
        next_history_gdx = find_history_gdx(
            gams_workdir,
            winning_state,
            policy=policy,
            impact=impact,
        )

        phase_results.append({
            "period":          period_label,
            "absorbing_states": absorbing,
            "w_sai":           w_sai,
            "history_gdx":     str(next_history_gdx) if next_history_gdx else None,
        })

        # Carry history into the next phase.
        # load_sai_history_until fixes N_SAI for all years <= that value.
        # We want the just-computed period to be fixed through its end year,
        # so use period_end (e.g. 2065 for phase 2050-2065).
        history_gdx  = next_history_gdx
        history_from = period_end

        if phase_idx == n_phases - 1:
            last_winning_state = winning_state
            last_winning_gdx = next_history_gdx
            last_period_label = period_label

    # ── Final summary ──────────────────────────────────────────────────────
    _section(f"RESULTS SUMMARY  (total elapsed: {_elapsed(t0)})")
    col_period  = 18
    col_states  = 36
    col_sai     = 12
    _log(
        f"  {'Period':<{col_period}}  {'Absorbing State(s)':<{col_states}}  {'W_SAI':>{col_sai}}"
    )
    _log(f"  {'─' * col_period}  {'─' * col_states}  {'─' * col_sai}")
    for r in phase_results:
        states_str = ", ".join(r["absorbing_states"])
        _log(
            f"  {r['period']:<{col_period}}  {states_str:<{col_states}}  {r['w_sai']:>{col_sai}.4f}"
        )
    _log("")

    # ── Winning coalition export + plot ───────────────────────────────────
    if last_winning_gdx is None or last_winning_state is None or last_period_label is None:
        _warn("No winning coalition GDX found to export.")
        return phase_results

    full_start = periods[0][0]
    full_end = periods[-1][1]
    dest_gdx = export_winning_coalition_gdx(
        source_gdx=last_winning_gdx,
        rice_dir=rice_dir,
        policy=policy,
        impact=impact,
        countries_slug=countries_slug,
        start_year=full_start,
        end_year=full_end,
    )

    _section("WINNING COALITION OUTPUT")
    _log(f"  Last period: {last_period_label}")
    _log(f"  Winning state: {last_winning_state}")
    _log(f"  Source GDX:   {last_winning_gdx}")
    _log(f"  Saved GDX:    {dest_gdx}")

    try:
        t_values, y_values = compute_sai_timeseries(dest_gdx, full_start, full_end)
        absorbing_labels = [
            ", ".join(r["absorbing_states"]) for r in phase_results
        ]
        _log("\nTerminal plot (W_SAI from GDX):\n")
        _log(
            render_sai_plot_to_ascii(
                t_values, y_values, periods, absorbing_labels=absorbing_labels
            )
        )
        _log(f"\nYears: {t_values[0]}  ...  {t_values[-1]}")
    except Exception as exc:
        _warn(f"Could not render W_SAI plot: {exc}")

    return phase_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_period(s: str) -> tuple[int, int]:
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Period must be in 'YYYY-YYYY' format, got '{s}'"
        )
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Period must be in 'YYYY-YYYY' format, got '{s}'"
        )
    if start >= end:
        raise argparse.ArgumentTypeError(
            f"Start year must be less than end year, got '{s}'"
        )
    return start, end


def _validate_periods(periods: list[tuple[int, int]]) -> None:
    """Ensure periods are contiguous (each starts where previous ends)."""
    if not periods:
        raise argparse.ArgumentTypeError("At least one period is required.")
    for i in range(1, len(periods)):
        prev_end = periods[i - 1][1]
        curr_start = periods[i][0]
        if curr_start != prev_end:
            raise argparse.ArgumentTypeError(
                "Periods must be contiguous, i.e. each period starts at the "
                f"previous period's end. Got {periods[i - 1][0]}-{prev_end} "
                f"followed by {curr_start}-{periods[i][1]}."
            )


def _parse_payoff_range(s: str) -> tuple[Optional[int], int]:
    """
    Parse payoff summation range.

    Accepted formats:
      YYYY        -> (None, YYYY)         # start is period_start (per phase)
      YYYY-YYYY   -> (start_year, end_year)
    """
    parts = s.split("-")
    if len(parts) == 1:
        try:
            return None, int(parts[0])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Payoff range must be 'YYYY' or 'YYYY-YYYY', got '{s}'"
            )
    if len(parts) == 2:
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Payoff range must be 'YYYY' or 'YYYY-YYYY', got '{s}'"
            )
        if start > end:
            raise argparse.ArgumentTypeError(
                f"Payoff range start must be <= end, got '{s}'"
            )
        return start, end
    raise argparse.ArgumentTypeError(
        f"Payoff range must be 'YYYY' or 'YYYY-YYYY', got '{s}'"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        type=_parse_period,
        required=True,
        metavar="YYYY-YYYY",
        help="One or more time periods (e.g. 2035-2060 2060-2080 2080-2100)",
    )
    parser.add_argument(
        "--impact",
        required=True,
        help="Impact function name (e.g. burke, andreoni, kalkuhl)",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        required=True,
        metavar="CODE",
        help="RICE50x country codes, lowercase (e.g. usa chn nde)",
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Policy name (e.g. bau_impact, bau, cba, cbudget, ctax)",
    )
    parser.add_argument(
        "--scenario",
        default="power_threshold_RICE_n3",
        help="Equilibrium scenario name (default: power_threshold_RICE_n3)",
    )
    parser.add_argument(
        "--rice-dir",
        type=Path,
        default=DEFAULT_RICE_DIR,
        help=f"Path to the RICE50x repository (default: {DEFAULT_RICE_DIR})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel GAMS jobs per phase (default: 4)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help=(
            "Re-run every step even if outputs already exist. "
            "Without this flag, steps whose output files are present are skipped."
        ),
    )
    parser.add_argument(
        "--gamma_ineq",
        type=float,
        default=None,
        help="Optional gamma_ineq override passed through to GAMS.",
    )
    parser.add_argument(
        "--max_gain",
        type=float,
        default=None,
        help="Optional max_gain override passed through to GAMS.",
    )
    parser.add_argument(
        "--max_damage",
        type=float,
        default=None,
        help="Optional max_damage override passed through to GAMS.",
    )
    parser.add_argument(
        "--t_ada_temp",
        type=float,
        default=None,
        help="Optional t_ada_temp override passed through to GAMS.",
    )
    parser.add_argument(
        "--sai_damage_coef",
        type=float,
        default=None,
        help="Optional sai_damage_coef override passed through to GAMS.",
    )
    parser.add_argument(
        "--payoff-range",
        type=_parse_payoff_range,
        default=None,
        metavar="YYYY|YYYY-YYYY",
        help=(
            "Override payoff summation years during ingest. "
            "Use 'YYYY' to sum each phase from period_start..YYYY (e.g. 2300), "
            "or 'YYYY-YYYY' for a fixed range across all phases."
        ),
    )

    args = parser.parse_args()
    _validate_periods(args.periods)
    extra_gams_args: list[str] = []
    if args.gamma_ineq is not None:
        extra_gams_args.append(f"--gamma_ineq={args.gamma_ineq}")
    if args.max_gain is not None:
        extra_gams_args.append(f"--max_gain={args.max_gain}")
    if args.max_damage is not None:
        extra_gams_args.append(f"--max_damage={args.max_damage}")
    if args.t_ada_temp is not None:
        extra_gams_args.append(f"--t_ada_temp={args.t_ada_temp}")
    if args.sai_damage_coef is not None:
        extra_gams_args.append(f"--sai_damage_coef={args.sai_damage_coef}")
    payoff_metadata = {
        "gamma_ineq": args.gamma_ineq,
        "max_gain": args.max_gain,
        "max_damage": args.max_damage,
        "t_ada_temp": args.t_ada_temp,
        "sai_damage_coef": args.sai_damage_coef,
        "payoff_range": (
            None
            if args.payoff_range is None
            else (f"period_start-{args.payoff_range[1]}" if args.payoff_range[0] is None
                  else f"{args.payoff_range[0]}-{args.payoff_range[1]}")
        ),
    }

    try:
        orchestrate(
            periods=args.periods,
            impact=args.impact,
            countries=args.countries,
            policy=args.policy,
            scenario=args.scenario,
            rice_dir=args.rice_dir,
            max_workers=args.max_workers,
            fresh=args.fresh,
            extra_gams_args=extra_gams_args or None,
            payoff_metadata=payoff_metadata,
            payoff_range=args.payoff_range,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        cmd_str = " ".join(sys.argv)
        _log(f"\n{'━' * 64}")
        _log(f"  ERROR: {exc}")
        _log(f"  Command: {cmd_str}")
        _log(f"{'━' * 64}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
