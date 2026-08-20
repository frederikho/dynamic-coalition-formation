"""The unit convention a profile was solved in must be recorded and honoured.

Payoff normalisation leaves the equilibrium set exactly unchanged, so it never
changes a verdict in exact arithmetic.  What it does change is the SCALE of V,
and `atol` is an absolute tolerance measured against that scale.  A profile
solved with normalisation on has a V spread of order 0.1; reconstructed in raw
RICE units the same profile's smallest per-player V spread is order 1e-6.  The
same `atol` number is therefore ~5 orders of magnitude weaker in raw units, and
a tolerance chosen to be strict during solving becomes near-vacuous on reload.

These tests pin down that the convention is recorded in the profile and that
`run_verification` reconstructs V in it -- while keeping the reported payoffs in
RAW units, because cross-player quantities (welfare sums) are only meaningful
there.
"""

import pandas as pd
import pytest

from lib.verify_cli import _read_metadata_from_xlsx, run_verification

TABLE = "payoff_tables/kalkuhl_eurnderus_2035-2100.xlsx"


@pytest.fixture(scope="module")
def profile(tmp_path_factory):
    """Solve one small game so the test never depends on a stale artefact."""
    from pathlib import Path
    import subprocess
    import sys

    if not Path(TABLE).exists():
        pytest.skip(f"{TABLE} not present")
    out = tmp_path_factory.mktemp("profiles") / "eurnderus.xlsx"
    proc = subprocess.run(
        [sys.executable, "-m", "lib.equilibrium.find", "power_threshold_RICE_n3",
         "--payoff-table", TABLE, "--discounting", "0.99",
         "--effectivity-rule", "heyen_lehtomaa_2021",
         "--solver-approach", "jeres_vfi", "--jeres-n-restarts", "1",
         "--jeres-max-iter", "2000", "--verify-atol", "1e-12",
         "--jeres-tol", "1e-14", "--fresh", "--output", str(out), "--oneline"],
        capture_output=True, text=True, timeout=300,
    )
    if not out.exists():
        pytest.fail(f"solver did not produce a profile:\n{proc.stdout}{proc.stderr}")
    return out


def test_profile_records_its_unit_convention(profile):
    """A profile must say which convention it was solved in."""
    metadata = _read_metadata_from_xlsx(profile)
    assert "normalise_payoffs" in metadata, (
        "profile does not record normalise_payoffs, so its unit convention "
        "cannot be recovered on reload"
    )
    assert str(metadata["normalise_payoffs"]).strip().lower() in {"true", "1"}


def test_verification_reconstructs_V_in_the_recorded_convention(profile):
    """V used for verification must be on the scale atol was chosen for."""
    ok, _, details = run_verification(profile, atol=1e-12, quiet=True)
    assert ok
    V = details["V"].astype(float)
    spread = (V.max(axis=0) - V.min(axis=0)).min()
    assert 1e-3 < spread < 1.0, (
        f"smallest per-player V spread is {spread:.3e}; a normalised profile "
        "should reconstruct on a unit-ish scale, not in raw RICE units"
    )


def test_reported_payoffs_stay_raw_for_cross_player_sums(profile):
    """Welfare sums compare players to each other, so they need raw units."""
    _, _, details = run_verification(profile, atol=1e-12, quiet=True)
    payoffs = details["payoffs"].astype(float)
    assert payoffs.to_numpy().max() < 0, (
        "reported payoffs are not raw RICE welfare; cross-player sums such as "
        "compare.py's mpe_expected_welfare would be meaningless"
    )
    assert "payoffs_normalised" in details


def test_normalisation_does_not_change_the_verdict(profile):
    """The invariance claim, exercised end to end on a real profile."""
    ok_norm, _, _ = run_verification(profile, atol=1e-12, quiet=True)
    ok_raw, _, _ = run_verification(profile, atol=1e-12, quiet=True,
                                    force_raw_units=True)
    assert ok_norm == ok_raw
