"""
Cycle detection and analysis for the inner fixed-point iteration.

The inner loop can fail to converge when the smoothed best-response map has a
limit cycle rather than a fixed point.  This module detects such cycles and
formats the history so the pattern is immediately visible.

Usage
-----
    from lib.equilibrium.cycle_analysis import detect_cycle, format_cycle_report

    history = [...]   # per-iteration max_change values from inner loop
    result  = detect_cycle(history)
    if result is not None:
        print(format_cycle_report(history, result))
"""

from __future__ import annotations
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def detect_cycle(
    history: list[float],
    max_period: Optional[int] = None,
    tol: float = 1e-3,
    min_confirmations: int = 3,
) -> Optional[int]:
    """
    Detect a repeating period in *history* (a list of scalar values).

    Strategy
    --------
    For each candidate period P (1, 2, 3, …) check whether the last
    ``min_confirmations * P`` values consist of the same block of length P
    repeated ``min_confirmations`` times.  The first P that passes is
    returned (shortest period wins).

    Parameters
    ----------
    history:
        Sequence of max_change values from the inner loop, one per iteration.
    max_period:
        Largest period to test.  Defaults to ``len(history) // (min_confirmations + 1)``
        so there is always enough data to confirm the result.
    tol:
        Maximum absolute difference allowed between corresponding positions
        in successive copies of the candidate block.
    min_confirmations:
        Number of full repetitions that must be present in the tail of
        *history* for a period to be accepted.

    Returns
    -------
    int or None
        The detected period, or ``None`` if no cycle was found.
    """
    n = len(history)
    if max_period is None:
        max_period = n // (min_confirmations + 1)

    if max_period < 1 or n < 2:
        return None

    arr = np.asarray(history, dtype=float)

    for period in range(1, max_period + 1):
        needed = period * min_confirmations
        if needed > n:
            break

        # Take the last `needed` values and reshape into (min_confirmations, period)
        tail = arr[-needed:].reshape(min_confirmations, period)

        # All rows should be (approximately) equal
        max_diff = np.max(np.abs(tail - tail[0]))
        if max_diff <= tol:
            return period

    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_cycle_report(
    history: list[float],
    period: int,
    row_width: int = 10,
    cycles_to_show: int = 3,
    value_fmt: str = ".3f",
) -> str:
    """
    Format a human-readable cycle report.

    The detected cycle is shown as ``cycles_to_show`` consecutive rows, each
    containing one full period.  Because the rows are identical (up to
    tolerance), the repetition is immediately obvious.

    Parameters
    ----------
    history:
        Full per-iteration max_change history.
    period:
        Detected cycle period (from :func:`detect_cycle`).
    row_width:
        Number of values to print per line within a single period row.
        Useful when the period is long.
    cycles_to_show:
        How many full repetitions of the cycle to display.
    value_fmt:
        Python format spec for individual values (e.g. ``".3f"``).
    """
    lines: list[str] = []

    # Extract one canonical cycle (the last complete period in history)
    arr = np.asarray(history, dtype=float)
    one_cycle = arr[-period:].tolist()

    lines.append(
        f"↺  Period-{period} cycle detected "
        f"(verified over {min(len(history), period * 4)} iterations)."
    )
    if cycles_to_show > 1:
        lines.append(f"   Showing {cycles_to_show} consecutive repetitions "
                     f"— identical rows confirm the cycle:")
    lines.append("")

    sep = "  "

    for rep in range(cycles_to_show):
        # Break the period into sub-rows of row_width
        for start in range(0, period, row_width):
            chunk = one_cycle[start:start + row_width]
            formatted = sep.join(format(v, value_fmt) for v in chunk)
            # Label only the first sub-row of each repetition
            if start == 0:
                label = f"  [{rep + 1}]  " if cycles_to_show > 1 else "   "
            else:
                label = "        " if cycles_to_show > 1 else "   "
            lines.append(label + formatted)
        if rep < cycles_to_show - 1:
            lines.append("")   # blank line between repetitions

    return "\n".join(lines)


def format_no_cycle_report(history: list[float], row_width: int = 20) -> str:
    """
    Compact fallback when no cycle is found: show the last *row_width* values
    as a single arrow-separated sequence.
    """
    tail = history[-row_width:]
    seq = "  →  ".join(f"{v:.3f}" for v in tail)
    return (
        f"  No short cycle detected (tested up to period {len(history) // 4}).\n"
        f"  Last {len(tail)} max_changes:\n"
        f"    {seq}"
    )
