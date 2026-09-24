"""Adapter from Jere's SMPE solver (lib/equilibrium/smpe) to the framework.

What is tested:
- under heyen_lehtomaa_2021 our committees, minus the proposer, are exactly
  Jere's own rule (so the adapter changes nothing for our default rule);
- the solver's output passes OUR verifier at atol 1e-12, on a cell where no
  pure equilibrium exists (so it must mix);
- other effectivity rules, including forbidden proposals, are honoured:
  the output verifies under that rule;
- unsupported settings raise instead of solving a different game.
"""
from pathlib import Path

import numpy as np
import pytest

from lib.equilibrium.find import (
    find_equilibrium, setup_experiment, _infer_or_parse_players_from_payoff_table,
)
from lib.equilibrium.scenarios import get_scenario, fill_players
from lib.equilibrium.smpe import build_committees, state_maps, solve_with_smpe
from lib.equilibrium.smpe.smpe import CoalitionGeometry

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "payoff_tables" / "kalkuhl_chneurrus_2035-2100.xlsx"


def _config(delta, rule="heyen_lehtomaa_2021", table=TABLE):
    config = get_scenario("power_threshold_RICE_n3")
    config["payoff_table"] = str(table)
    config["discounting"] = delta
    config["effectivity_rule"] = rule
    config["normalise_payoffs"] = True
    return fill_players(config, _infer_or_parse_players_from_payoff_table(table))


def _committees(rule):
    setup = setup_experiment(_config(0.9, rule))
    players, states = setup["players"], setup["state_names"]
    geom, fw_to_j, _ = state_maps(players, states)
    comm, own = build_committees(players, states, setup["effectivity"],
                                 setup.get("forbidden_proposals", frozenset()),
                                 fw_to_j, rule)
    return geom, comm, own


def _solve(delta, rule="heyen_lehtomaa_2021"):
    return find_equilibrium(_config(delta, rule), verbose=False,
                            solver_approach="smpe", verify_atol=1e-12)


def test_heyen_committees_equal_jere_rule():
    geom, comm, own = _committees("heyen_lehtomaa_2021")
    native = CoalitionGeometry(geom.players)
    assert np.array_equal(comm.voters, native.voters)
    assert not comm.blocked.any()
    # the only difference: the proposer on its own unilateral-exit committee
    assert len(own) == 9


def test_adjacent_step_blocks_non_adjacent_proposals():
    _, comm, _ = _committees("adjacent_step")
    assert comm.blocked.sum() == 24


@pytest.mark.parametrize("delta", [0.8, 0.95])
def test_verifies_under_our_verifier(delta):
    """0.8: pure region.  0.95: no pure equilibrium exists here (exhaustive
    ordinal_ranking), so a verified profile must be mixed."""
    res = _solve(delta)
    assert res["verification_success"], res["verification_message"]
    if delta == 0.95:
        df = res["strategy_df"]
        vals = df.values.astype(float)
        assert ((vals > 1e-9) & (vals < 1 - 1e-9)).any(), "expected a mixed profile"


@pytest.mark.parametrize("rule", ["adjacent_step", "deployer_exit", "unanimous_consent"])
def test_other_effectivity_rules(rule):
    res = _solve(0.9, rule)
    assert res["solver_result"]["committee_source"] == rule
    assert res["verification_success"], res["verification_message"]


def test_majority_approval_rejected():
    class _S:  # minimal solver stand-in; the adapter must refuse before using it
        unanimity_required = False

    with pytest.raises(ValueError, match="unanimous approval only"):
        solve_with_smpe(_S(), {})
