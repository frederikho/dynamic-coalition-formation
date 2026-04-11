"""Terminal progress output for ordinal-ranking search."""

from __future__ import annotations

import math
import time


def _print_progress(done: int, total: int | None, start_time: float) -> None:
    elapsed = max(1e-9, time.perf_counter() - start_time)
    rate = done / elapsed
    frac = done / total if total else 0.0
    width = 30
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = 100.0 * frac
    remaining = (total - done) / rate if (total and rate > 0) else float("inf")
    if math.isfinite(remaining) and 0 <= remaining < 360_000:  # cap at 100 hours
        secs = int(round(remaining))
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        eta = f"{hours:d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
    else:
        eta = "∞"
    total_str = f"{total:.3e}" if total and total > 1_000_000_000 else f"{total:,d}" if total else "?"
    print(
        f"\r\033[2K[{bar}] {done:>9,d}/{total_str}  {pct:.2e}%  "
        f"rate={rate:8.0f}/s  eta={eta}",
        end="",
        flush=True,
    )
