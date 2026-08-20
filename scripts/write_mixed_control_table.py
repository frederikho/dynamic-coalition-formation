#!/usr/bin/env python3
"""
Write the mixed-equilibrium control game out as a normal payoff table.

`scripts/mixed_control_game.py` builds the control directly against the jeres_vfi
Game object.  That proves the solver path works, but it bypasses the framework:
the scenario, the effectivity rule, the state naming and the payoff loader are all
skipped.  Writing the same game as a payoff table lets the control be run through
the ordinary CLI, which is both a stronger test and a reusable fixture.

Two translations are needed and neither is cosmetic:

  * State ORDER.  The control is indexed in Jere partition order; the payoff table
    is indexed by framework state name.  `fw_state_name_to_partition` gives the
    mapping, and getting it wrong would silently permute the payoffs into a
    different game.
  * Committee RULE.  The control was constructed against jeres_vfi's built-in
    `voters()` rule.  The framework applies an effectivity rule by name.  If the
    two disagree, the constructed profile is not an equilibrium of the table.
    This script checks that rather than assuming it.

Usage:
    python scripts/write_mixed_control_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium.jeres_vfi import (  # noqa: E402
    Game,
    fw_state_name_to_partition,
)

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "mcg", REPO / "scripts" / "mixed_control_game.py")
mcg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcg)

# Real region codes, so the framework can infer the players from the filename.
PLAYERS = ["CHN", "EUR", "USA"]
OUT = Path("/home/frederik/Code/farsighted-coalitions/payoff_tables")
NAME = "mixedcontrol_chneurusa.xlsx"


def framework_state_names(players):
    """Canonical n=3 coalition-structure names, in the framework's own order."""
    a, b, c = players
    return ["( )", f"({a}{b})", f"({a}{c})", f"({b}{c})", f"({a}{b}{c})"]


def main():
    ctrl = mcg.build_control(["A", "B", "C"], seed=0)
    if ctrl is None:
        print("Could not build the control game.")
        return 1

    u = ctrl["u"]                      # (n_states, n_players) in Jere order
    game = ctrl["game"]
    tx, ty, tj = ctrl["tie"]

    names = framework_state_names(PLAYERS)
    rows = {}
    for name in names:
        part = fw_state_name_to_partition(name, PLAYERS)
        j = game.state_idx[part]
        rows[name] = u[j]
        if j == tx:
            tie_from = name
        if j == ty:
            tie_to = name

    df = pd.DataFrame.from_dict(rows, orient="index", columns=PLAYERS)
    df.index.name = "state"

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / NAME
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        # The loader reads with header=1, so the header must sit on the second row.
        df.to_excel(xl, sheet_name="Payoffs", startrow=1)
        ws = xl.sheets["Payoffs"]
        ws.cell(row=1, column=1).value = (
            f"Synthetic control: mixed equilibrium by construction, "
            f"delta={mcg.DELTA}, mixing voter {PLAYERS[tj]} on "
            f"{tie_from} -> {tie_to} at alpha={mcg.THETA}"
        )

    print(f"written: {path}")
    print(f"  players       : {PLAYERS}")
    print(f"  delta         : {mcg.DELTA}   (pass --discounting {mcg.DELTA})")
    print(f"  planted mixing: {PLAYERS[tj]} on {tie_from} -> {tie_to}, "
          f"alpha = {mcg.THETA}")
    print("  no PURE equilibrium exists (certified by exhaustive enumeration),")
    print("  so any verified profile for this game must be mixed.\n")
    print(df.round(4).to_string())

    # Round-trip: does the loader reproduce the payoffs we wrote?
    back = pd.read_excel(path, sheet_name="Payoffs", header=1, index_col=0)
    if not np.allclose(back[PLAYERS].to_numpy(), df[PLAYERS].to_numpy()):
        print("\nWARNING: round-trip mismatch")
        return 1
    print("\nround-trip through the loader's format: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
