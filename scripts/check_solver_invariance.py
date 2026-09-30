#!/usr/bin/env python3
"""
Regression checks for the property that lets the solver normalise payoffs.

WHY THIS EXISTS.  find_equilibrium() rescales every player's payoffs to [0,1]
before solving, by default.  That is only legitimate because the equilibrium
concept is invariant to per-player positive affine maps: every condition compares
a player against *themselves* (V_i(y) vs V_i(x)), never one player against
another.  If that invariance is ever broken -- by a condition that mixes players,
or by a welfare term leaking into the solve path -- normalisation would silently
change the answer rather than merely the conditioning.  These checks fail loudly
if that happens.

Normalisation is not cosmetic: measured over the 91 n=3 payoff tables in this
repo it takes the solve rate from 42/91 to 67/91, halves both timeouts and
failures, and cuts wall time by a third.  See scripts/bench_solver.py.

Run:
    python scripts/check_solver_invariance.py       # standalone, prints results
    pytest scripts/check_solver_invariance.py       # or under pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib.equilibrium.find import normalise_payoffs, payoff_scale, payoff_spreads  # noqa: E402
from lib.mdp import MDP  # noqa: E402

STATES = ["( )", "(AB)", "(AC)", "(BC)", "(ABC)"]


def _rice_like(seed=0, n_players=3):
    """Payoffs with the RICE shape: huge level, minuscule spread."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-22.0, -3.0, size=n_players)
    spread = rng.uniform(1e-5, 2e-2, size=n_players)
    data = base + rng.random((len(STATES), n_players)) * spread
    return pd.DataFrame(data, index=STATES, columns=list("ABC")[:n_players])


def test_normalisation_maps_onto_the_unit_interval():
    out = normalise_payoffs(_rice_like())
    assert np.allclose(out.min().to_numpy(), 0.0)
    assert np.allclose(out.max().to_numpy(), 1.0)


def test_each_players_preference_order_is_preserved():
    """The core claim: normalisation cannot change who prefers what."""
    payoffs = _rice_like(seed=1)
    out = normalise_payoffs(payoffs)
    for p in payoffs.columns:
        assert list(payoffs[p].rank()) == list(out[p].rank()), f"{p} reordered"


def test_value_gap_signs_are_identical_and_magnitudes_widen():
    """V_i -> a_i*V_i + b_i with a_i > 0, so every comparison survives.

    Also asserts the practical point: the gaps the verifier must resolve get
    strictly larger, which is the entire reason for doing this.
    """
    rng = np.random.default_rng(5)
    n = len(STATES)
    P = rng.random((n, n))
    P /= P.sum(axis=1, keepdims=True)

    payoffs = _rice_like(seed=2)
    norm = normalise_payoffs(payoffs)
    mdp = MDP(n_states=n, transition_probs=pd.DataFrame(P), discounting=0.99)

    for p in payoffs.columns:
        V_raw = np.asarray(mdp.solve_value_func(payoffs[p].to_numpy()))
        V_norm = np.asarray(mdp.solve_value_func(norm[p].to_numpy()))
        g_raw = V_raw[:, None] - V_raw[None, :]
        g_norm = V_norm[:, None] - V_norm[None, :]
        assert np.array_equal(np.sign(g_raw), np.sign(g_norm)), f"{p}: sign flipped"
        assert np.max(np.abs(g_norm)) > np.max(np.abs(g_raw)), f"{p}: gaps not widened"


def test_arbitrary_affine_map_leaves_gap_signs_alone():
    """Not just [0,1]: ANY per-player positive affine map must be safe."""
    rng = np.random.default_rng(11)
    n = len(STATES)
    P = rng.random((n, n))
    P /= P.sum(axis=1, keepdims=True)
    mdp = MDP(n_states=n, transition_probs=pd.DataFrame(P), discounting=0.99)

    payoffs = _rice_like(seed=3)
    scales = rng.uniform(0.01, 100.0, size=payoffs.shape[1])
    shifts = rng.uniform(-500.0, 500.0, size=payoffs.shape[1])
    mapped = payoffs * scales + shifts

    for k, p in enumerate(payoffs.columns):
        V_a = np.asarray(mdp.solve_value_func(payoffs[p].to_numpy()))
        V_b = np.asarray(mdp.solve_value_func(mapped[p].to_numpy()))
        g_a = V_a[:, None] - V_a[None, :]
        g_b = V_b[:, None] - V_b[None, :]
        assert np.array_equal(np.sign(g_a), np.sign(g_b)), f"{p}: sign flipped"
        # And the map is exactly the one the theory predicts.
        assert np.allclose(g_b, scales[k] * g_a, rtol=1e-9, atol=1e-12)


def test_flat_player_raises_rather_than_dividing_by_zero():
    payoffs = _rice_like(seed=4)
    payoffs["B"] = -3.0  # indifferent between every state
    try:
        normalise_payoffs(payoffs)
    except ValueError as exc:
        assert "identical payoffs in every state" in str(exc)
    else:
        raise AssertionError("a flat player must raise, not divide by zero")


def test_mip_probabilities_are_snapped_onto_the_unit_interval():
    """Regression: MIP noise outside [0,1] made valid equilibria unverifiable.

    The framework verifier tests probabilities EXACTLY -- `p == 1.`, `p == 0.`,
    or `0. <= p <= 1.` in the indifference branch. HiGHS reports solutions to
    within its primal feasibility tolerance, so a variable bounded to [0,1] can
    return -1e-14. That value fails BOTH the indifference branch (not >= 0) and
    the strict-loss branch (not == 0), rejecting a good profile over a rounding
    artefact.

    Observed on kalkuhl_eurndeusa_2035-2100: two acceptance cells at
    -1.0011e-14 made the game report unsolvable at verify_atol=1e-2, while the
    same game solved at 1e-10 only because the tighter tolerance routed
    verification down a branch that never touched those cells.
    """
    from lib.equilibrium.jeres_vfi.solver import _clean_probability

    assert _clean_probability(-1.0011058459439017e-14) == 0.0
    assert _clean_probability(-0.0) == 0.0
    assert _clean_probability(1.0 + 1e-14) == 1.0

    # Snapping, not merely clipping: 1 - 1e-14 is inside [0,1] but still fails
    # the verifier's `p == 1.` test unless it is snapped to exactly 1.
    assert _clean_probability(1.0 - 1e-14) == 1.0
    assert _clean_probability(1e-14) == 0.0

    # Genuine interior mixing must survive untouched.
    for p in (0.25, 0.5, 0.902385, 0.999):
        assert _clean_probability(p) == p


def test_non_finite_mip_probability_raises():
    """A NaN probability is a solver failure, not something to silently clamp."""
    from lib.equilibrium.jeres_vfi.solver import _clean_probability

    for bad in (float("nan"), float("inf"), float("-inf")):
        with_raise = False
        try:
            _clean_probability(bad)
        except ValueError:
            with_raise = True
        assert with_raise, f"{bad} should raise"


def test_payoff_scale_binds_on_the_narrowest_player():
    payoffs = pd.DataFrame(
        {"A": [-13.0, -12.0, -12.5, -12.2, -12.9],
         "B": [-3.0, -3.25, -3.1, -3.2, -3.05]},
        index=STATES,
    )
    spreads = payoff_spreads(payoffs)
    assert spreads["A"] > spreads["B"]
    assert payoff_scale(payoffs) == spreads["B"]


def main() -> int:
    checks = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {check.__name__}\n      {exc}")
        else:
            print(f"ok    {check.__name__}")
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
