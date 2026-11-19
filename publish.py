#!/usr/bin/env python3
"""
Curate which strategy tables are published on the GitHub Pages site.

The published set lives in ``published/`` and is the ONLY thing committed for
the site (everything under viz/data is generated from it by build_static.py,
in CI):

    published/strategy_tables/<name>.xlsx   the chosen strategy profiles
    published/payoff_tables/<file>.xlsx     the payoff tables those profiles name
                                            in their Metadata sheet (deduplicated:
                                            profiles sharing a table share one copy)

Usage:
    python publish.py add NAME [NAME ...]     NAME = file stem, e.g. eq_n3_power_threshold_RICE_7078fd
    python publish.py remove NAME [NAME ...]
    python publish.py list

NAME is searched recursively in strategy_tables/, then in viz/data/xlsx/ (the
previously published copies). Fails loudly if a name is not found, is ambiguous,
or its payoff table is missing.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PUB = ROOT / "published"
PUB_STRATEGY = PUB / "strategy_tables"
PUB_PAYOFF = PUB / "payoff_tables"
SEARCH_DIRS = [ROOT / "strategy_tables", ROOT / "viz" / "data" / "xlsx"]


def payoff_table_of(xlsx: Path) -> str | None:
    """Payoff-table path named in the profile's Metadata sheet, or None if it has none."""
    xl = pd.ExcelFile(xlsx)
    if "Metadata" not in xl.sheet_names:
        return None
    meta = xl.parse("Metadata", header=None)
    hit = meta[meta[0].astype(str) == "payoff_table"]
    if hit.empty:
        return None
    value = str(hit.iloc[0, 1]).strip()
    return value if value and value.lower() != "nan" else None


def find_strategy(name: str) -> Path:
    stem = name[:-5] if name.endswith(".xlsx") else name
    matches = [m for d in SEARCH_DIRS if d.exists() for m in d.rglob(f"{stem}.xlsx")
               if not m.name.startswith(".~lock")]
    # Prefer strategy_tables/ over the older published copies if both exist
    preferred = [m for m in matches if SEARCH_DIRS[0] in m.parents]
    matches = preferred or matches
    if not matches:
        sys.exit(f"ERROR: no strategy table named '{stem}' in {', '.join(map(str, SEARCH_DIRS))}")
    if len({m.read_bytes() for m in matches}) > 1:
        sys.exit(f"ERROR: '{stem}' matches different files: {[str(m) for m in matches]}")
    return matches[0]


def cmd_add(names):
    for name in names:
        src = find_strategy(name)
        table = payoff_table_of(src)
        PUB_STRATEGY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, PUB_STRATEGY / src.name)
        msg = f"+ {src.name}"
        if table:
            # Same resolution as lib/verify_cli.py: the recorded path, else payoff_tables/<basename>
            # (profiles solved on another machine record that machine's absolute path)
            table_src = next((c for c in (Path(table), ROOT / table, ROOT / "payoff_tables" / Path(table).name)
                              if c.exists()), None)
            if table_src is None:
                sys.exit(f"ERROR: {src.name} needs payoff table '{table}', which does not exist "
                         f"(also looked for payoff_tables/{Path(table).name})")
            PUB_PAYOFF.mkdir(parents=True, exist_ok=True)
            dest = PUB_PAYOFF / table_src.name
            if dest.exists() and dest.read_bytes() != table_src.read_bytes():
                sys.exit(f"ERROR: a different '{dest.name}' is already published; resolve manually")
            shutil.copy2(table_src, dest)
            msg += f"  (payoff table: {table_src.name})"
        print(msg)


def used_payoff_tables() -> set[str]:
    used = set()
    for f in PUB_STRATEGY.glob("*.xlsx") if PUB_STRATEGY.exists() else []:
        table = payoff_table_of(f)
        if table:
            used.add(Path(table).name)
    return used


def cmd_remove(names):
    for name in names:
        stem = name[:-5] if name.endswith(".xlsx") else name
        target = PUB_STRATEGY / f"{stem}.xlsx"
        if not target.exists():
            sys.exit(f"ERROR: '{stem}' is not published")
        target.unlink()
        print(f"- {target.name}")
    # Drop payoff tables no remaining profile needs
    used = used_payoff_tables()
    for f in PUB_PAYOFF.glob("*.xlsx") if PUB_PAYOFF.exists() else []:
        if f.name not in used:
            f.unlink()
            print(f"- payoff table {f.name} (no longer used)")


def cmd_list(_names):
    files = sorted(PUB_STRATEGY.glob("*.xlsx")) if PUB_STRATEGY.exists() else []
    if not files:
        print("Nothing published yet. Use: python publish.py add NAME")
        return
    for f in files:
        table = payoff_table_of(f)
        print(f"{f.stem}  ->  {Path(table).name if table else '(no payoff table)'}")
    print(f"\n{len(files)} strategy tables, "
          f"{len(list(PUB_PAYOFF.glob('*.xlsx'))) if PUB_PAYOFF.exists() else 0} payoff tables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["add", "remove", "list"])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    if a.command != "list" and not a.names:
        ap.error(f"'{a.command}' needs at least one NAME")
    {"add": cmd_add, "remove": cmd_remove, "list": cmd_list}[a.command](a.names)
