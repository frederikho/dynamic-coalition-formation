"""Static coalition-formation benchmarks on a partition-function payoff table.

Two concepts are implemented, both defined over the sub-lattice of coalition
structures of the form "one coalition S, everybody else a singleton" — the shape
the classical literature assumes, and (for n=3) the whole state space:

  * internal / external stability, d'Aspremont et al. (1983), as used in the
    IEA literature (Carraro & Siniscalco 1993, Barrett 1994);
  * the gamma-core, Chander & Tulkens (1997), in a no-transfer (NTU) reading
    that matches this framework, plus the classical transferable-utility (TU)
    reading as a diagnostic.

Both take payoffs already resolved onto framework state names — as produced by
``lib.equilibrium.find.setup_experiment`` or ``lib.verify_cli.run_verification``
— so no payoff-table parsing happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from lib.utils import list_coalitions

# RICE runs that failed to converge are written to the payoff tables as a large
# negative sentinel. Any stability calculation on such a row is meaningless.
SENTINEL_THRESHOLD = -9999.0


class ContaminatedPayoffTable(ValueError):
    """Raised when a payoff table contains sentinel (failed-run) payoffs."""


@dataclass(frozen=True)
class PayoffGame:
    """A partition-function-form game over coalition structures.

    Attributes:
        players: Player names, in payoff-column order.
        state_names: Coalition-structure names, in payoff-row order.
        payoffs: states x players payoff matrix.
        geo: Geoengineering (SAI) level per state, or None if the table has none.
        structure_of: frozenset(S) -> state name, over the "one coalition plus
            singletons" sub-lattice. The empty frozenset maps to all-singletons.
        partition_index: canonical partition -> state name, over *every* state,
            including structures with several non-trivial coalitions.
    """

    players: List[str]
    state_names: List[str]
    payoffs: pd.DataFrame
    geo: Optional[pd.Series]
    structure_of: Dict[FrozenSet[str], str]
    partition_index: Dict[FrozenSet[FrozenSet[str]], str]

    @classmethod
    def from_resolved(
        cls,
        payoffs: pd.DataFrame,
        players: Sequence[str],
        geo: Optional[pd.DataFrame | pd.Series] = None,
        allow_contaminated: bool = False,
    ) -> "PayoffGame":
        """Build a game from a state-indexed payoff matrix.

        Arguments:
            payoffs: states x players payoffs, indexed by framework state name.
            players: Player names; must all be columns of ``payoffs``.
            geo: Per-state SAI level, either a Series or a one-column DataFrame.
            allow_contaminated: Permit sentinel payoffs instead of raising.

        Raises:
            ValueError: on duplicate states, missing players, non-numeric
                payoffs, or a state space that does not contain every
                "one coalition plus singletons" structure.
            ContaminatedPayoffTable: on sentinel payoffs, unless allowed.
        """
        players = list(players)
        state_names = [str(s) for s in payoffs.index]

        duplicates = sorted({s for s in state_names if state_names.count(s) > 1})
        if duplicates:
            raise ValueError(f"Payoff matrix has duplicate state rows: {duplicates}")

        missing = [p for p in players if p not in payoffs.columns]
        if missing:
            raise ValueError(
                f"Payoff matrix is missing player columns {missing}; "
                f"has {list(payoffs.columns)}"
            )

        values = payoffs.loc[:, players].apply(pd.to_numeric, errors="coerce")
        values.index = state_names
        if values.isna().any().any():
            bad = values.index[values.isna().any(axis=1)].tolist()
            raise ValueError(f"Payoff matrix has non-numeric entries in states {bad}")

        contaminated = values.index[(values <= SENTINEL_THRESHOLD).any(axis=1)].tolist()
        if contaminated and not allow_contaminated:
            raise ContaminatedPayoffTable(
                f"Payoff table has sentinel values (<= {SENTINEL_THRESHOLD}) in states "
                f"{contaminated}. These mark failed or missing RICE runs, so stability "
                "results computed from them are meaningless. Pass "
                "allow_contaminated=True only if you intend to report them as such."
            )

        geo_series = _as_geo_series(geo, state_names)
        partition_index = _index_partitions(state_names, players)
        structure_of = {
            _sole_coalition(partition): state
            for partition, state in partition_index.items()
            if _sole_coalition(partition) is not None
        }

        return cls(
            players=players,
            state_names=state_names,
            payoffs=values,
            geo=geo_series,
            structure_of=structure_of,
            partition_index=partition_index,
        )

    def coalition_of(self, state: str) -> FrozenSet[str]:
        """The non-trivial coalition in ``state`` (empty set if all singletons)."""
        coalitions = list_coalitions(state, self.players)
        if not coalitions:
            return frozenset()
        if len(coalitions) > 1:
            raise ValueError(
                f"State {state!r} contains {len(coalitions)} non-trivial coalitions; "
                "the static benchmarks are only defined for structures with one."
            )
        return frozenset(coalitions[0])

    def state_for(self, coalition: FrozenSet[str]) -> str:
        """State name of the structure with ``coalition`` and everyone else alone."""
        key = frozenset(coalition)
        if len(key) < 2:
            key = frozenset()
        if key not in self.structure_of:
            raise KeyError(
                f"No state in this game corresponds to coalition {sorted(coalition)} "
                "plus singletons."
            )
        return self.structure_of[key]

    def partition_of(self, state: str) -> FrozenSet[FrozenSet[str]]:
        """Full partition of ``state``, singletons included."""
        return _partition(state, self.players)

    def state_for_partition(
        self, partition: FrozenSet[FrozenSet[str]]
    ) -> Optional[str]:
        """State name for a partition, or None if the game does not contain it.

        Returning None rather than raising lets the stability tests skip
        deviations that lead outside a reduced state space, instead of failing
        outright on games that legitimately omit some structures.
        """
        return self.partition_index.get(frozenset(partition))

    def welfare(self, state: str) -> float:
        """Utilitarian sum of payoffs in ``state``."""
        return float(self.payoffs.loc[state, self.players].sum())

    def largest_coalition_size(self, state: str) -> int:
        """Size of the biggest coalition in ``state`` (1 if all singletons)."""
        coalitions = list_coalitions(state, self.players)
        return max((len(c) for c in coalitions), default=1)


def _as_geo_series(geo, state_names: List[str]) -> Optional[pd.Series]:
    if geo is None:
        return None
    if isinstance(geo, pd.DataFrame):
        if geo.shape[1] != 1:
            raise ValueError(
                f"Geoengineering frame must have exactly one column, got {list(geo.columns)}"
            )
        geo = geo.iloc[:, 0]
    series = pd.to_numeric(pd.Series(geo), errors="coerce")
    series.index = [str(s) for s in series.index]
    missing = [s for s in state_names if s not in series.index]
    if missing:
        raise ValueError(f"Geoengineering levels missing for states {missing}")
    return series.reindex(state_names)


def _partition(state: str, players: List[str]) -> FrozenSet[FrozenSet[str]]:
    """Canonical partition of a state: its coalitions plus the implied singletons."""
    blocks = [frozenset(c) for c in list_coalitions(state, players)]
    covered = {p for block in blocks for p in block}
    blocks.extend(frozenset({p}) for p in players if p not in covered)
    return frozenset(blocks)


def _sole_coalition(
    partition: FrozenSet[FrozenSet[str]],
) -> Optional[FrozenSet[str]]:
    """The single non-trivial coalition of a partition, or None if there are several.

    All-singleton partitions map to the empty frozenset, which is the key the
    gamma-core uses for "nobody cooperates".
    """
    blocks = [b for b in partition if len(b) > 1]
    if len(blocks) > 1:
        return None
    return blocks[0] if blocks else frozenset()


def _index_partitions(
    state_names: List[str], players: List[str]
) -> Dict[FrozenSet[FrozenSet[str]], str]:
    """Map every state to its canonical partition."""
    index: Dict[FrozenSet[FrozenSet[str]], str] = {}
    for state in state_names:
        key = _partition(state, players)
        if key in index:
            raise ValueError(
                f"States {index[key]!r} and {state!r} describe the same coalition "
                "structure."
            )
        index[key] = state
    return index


def require_cartel_lattice(game: "PayoffGame") -> None:
    """Check the game contains every "one coalition plus singletons" structure.

    The gamma-core enumerates all coalitions and needs each of them to have a
    corresponding state; internal/external stability does not, so this is
    enforced where it is actually required rather than at load time.
    """
    required = [frozenset()] + [
        frozenset(combo)
        for size in range(2, len(game.players) + 1)
        for combo in combinations(game.players, size)
    ]
    absent = [
        sorted(key) or ["<all singletons>"]
        for key in required
        if key not in game.structure_of
    ]
    if absent:
        raise ValueError(
            "This calculation needs every 'one coalition plus singletons' structure, "
            f"but the state space is missing: {absent}"
        )


def _margin_summary(margins: List[float]) -> Dict[str, float]:
    """Summarise how close a state's stability verdicts are to flipping.

    Exact ties are reported separately from small-but-positive margins. A tie
    means two structures give literally the same payoffs — which happens by
    construction whenever they share a deploying coalition — so the verdict there
    is decided by the weak/strict convention rather than by the numbers. A small
    positive margin is the worrying case: the verdict is real but rests on a
    difference near the resolution of the underlying RICE runs.
    """
    positive = [m for m in margins if m > 0]
    return {
        "min_margin": min(positive) if positive else float("nan"),
        "n_exact_ties": sum(1 for m in margins if m == 0),
    }


def internal_external_stability(game: PayoffGame) -> pd.DataFrame:
    """Internal and external stability of each single-coalition structure.

    Let ``pi(S)`` be the structure with coalition ``S`` and all others singletons.

    Internal stability: no member wants to leave, where leaving means becoming a
    singleton while every other block of the partition stays intact. This is the
    narrow deviation set, and it is the one Heyen & Lehtomaa (2021) use — their
    stated static predictions for the power-threshold example are reproducible
    only under it. A member leaving one coalition to join another is a *broad*
    deviation and is deliberately not counted here.

    States with several non-trivial coalitions (possible for n >= 4) are handled:
    a member's departure affects only its own block. Deviations leading to
    structures the game does not contain are skipped rather than assumed.

    External stability comes in two readings:

    ``external_open``
        d'Aspremont's open membership: an outsider joins if it gains, and
        incumbents cannot refuse. For all j not in S, u_j(pi(S+j)) <= u_j(pi(S)).

    ``external_consent``
        Accession also requires the incumbents to (weakly) gain, matching the
        approval committees used in this framework. A structure is externally
        stable unless some outsider *and* every incumbent would gain from
        accession.

    Returns:
        DataFrame indexed by state with boolean columns ``internal``,
        ``external_open``, ``external_consent``, ``stable_open``,
        ``stable_consent``, and the string column ``blocking_accessions``.
    """
    rows = []
    for state in game.state_names:
        partition = game.partition_of(state)
        coalitions = sorted((c for c in partition if len(c) > 1), key=lambda c: sorted(c))
        singletons = sorted(next(iter(c)) for c in partition if len(c) == 1)
        u = game.payoffs

        internal = True
        margins = []
        for coalition in coalitions:
            for member in sorted(coalition):
                # Narrow deviation: the leaver stands alone, every other block
                # of the partition is untouched. Allowing the leaver to join
                # another coalition instead is a different concept (see the
                # module docstring and Section 5 of the appendix draft).
                after_partition = (partition - {coalition}) | {
                    coalition - {member},
                    frozenset({member}),
                }
                after = game.state_for_partition(after_partition)
                if after is None:
                    continue
                gap = u.loc[state, member] - u.loc[after, member]
                margins.append(abs(gap))
                if gap < 0:
                    internal = False

        external_open = True
        external_consent = True
        blocking = []

        if not coalitions:
            # Accession is not defined at the all-singleton structure: there is
            # no coalition to join. Taking d'Aspremont literally would make it
            # vacuously stable in every game, which is useless as a benchmark.
            # The standard reading is that no coalition forms only if no *pair*
            # wants to form, and with no incumbents to overrule, both variants
            # coincide on the mutual-gain test.
            for first, second in combinations(game.players, 2):
                after = game.state_for_partition(
                    (partition - {frozenset({first}), frozenset({second})})
                    | {frozenset({first, second})}
                )
                if after is None:
                    continue
                margins.append(min(
                    abs(u.loc[after, first] - u.loc[state, first]),
                    abs(u.loc[after, second] - u.loc[state, second]),
                ))
                if (u.loc[after, first] > u.loc[state, first]
                        and u.loc[after, second] > u.loc[state, second]):
                    external_open = False
                    external_consent = False
                    blocking.append(f"{first}+{second}")
        else:
            for coalition in coalitions:
                # Only unattached players may accede. A member of another
                # coalition moving across is re-partnering, not accession, and
                # belongs to the broad reading this benchmark deliberately omits.
                for joiner in singletons:
                    after = game.state_for_partition(
                        (partition - {coalition, frozenset({joiner})})
                        | {coalition | {joiner}}
                    )
                    if after is None:
                        continue
                    joiner_gap = u.loc[after, joiner] - u.loc[state, joiner]
                    margins.append(abs(joiner_gap))
                    if joiner_gap <= 0:
                        continue
                    external_open = False
                    margins.extend(
                        abs(u.loc[after, m] - u.loc[state, m]) for m in coalition
                    )
                    if all(u.loc[after, m] >= u.loc[state, m] for m in coalition):
                        external_consent = False
                        blocking.append(joiner)

        rows.append(
            {
                "state": state,
                "internal": internal,
                "external_open": external_open,
                "external_consent": external_consent,
                "stable_open": internal and external_open,
                "stable_consent": internal and external_consent,
                "blocking_accessions": ",".join(sorted(blocking)),
                **_margin_summary(margins),
            }
        )

    return pd.DataFrame(rows).set_index("state")


def ricke_winning_coalitions(
    game: PayoffGame, min_power: float, power: Dict[str, float]
) -> pd.DataFrame:
    """Ricke, Moreno-Cruz & Caldeira (2013)'s static exclusion-game benchmark.

    Ricke et al. model coalition formation as a one-shot "exclusion game": a
    coalition needs a majority power share to deploy, and is "stable" if "no
    member has an incentive to leave the coalition for another" (their section
    2.1). A "winning coalition" is a stable majority coalition, which they
    assert is unique "by definition."

    In their game only one coalition ever acts, so a member who leaves has
    nowhere to go but non-member (outsider) status -- their stability test is
    this framework's own narrow ``internal`` stability (the leaver stands
    alone), gated by the majority-power requirement. This function reuses that
    exact narrow-deviation logic from :func:`internal_external_stability`
    rather than redefining it, so the two benchmarks cannot drift apart.

    Args:
        game: The payoff game.
        min_power: The majority-power threshold a coalition's combined power
            must exceed (matches this framework's ``min_power`` convention:
            the same rule usually used to decide who *can deploy*, reused here
            for who *can be a Ricke-style winning coalition*).
        power: Each player's power share.

    Returns:
        DataFrame indexed by state (over every single-coalition state in the
        game, i.e. ``game.structure_of``), with boolean columns ``majority``
        (combined power exceeds ``min_power``) and ``stable`` (no member wants
        to leave to become a singleton). A state's Ricke "winning coalition"
        status is ``majority & stable``; Ricke's uniqueness claim is falsified
        whenever more than one state has both True.
    """
    narrow = internal_external_stability(game)

    rows = []
    for coalition, state in game.structure_of.items():
        if len(coalition) < 2:
            continue
        p = sum(power.get(m, 0.0) for m in coalition)
        rows.append({
            "state": state,
            "majority": p > min_power,
            "stable": bool(narrow.loc[state, "internal"]) if state in narrow.index else False,
            "power": p,
        })

    return pd.DataFrame(rows).set_index("state")


def gamma_core(game: PayoffGame) -> Dict[str, object]:
    """Gamma-core analysis, in both no-transfer and transferable-utility form.

    The gamma-characteristic function assumes that when ``S`` deviates, the
    complement breaks apart into singletons — so ``v_gamma(S)`` is read directly
    off the structure ``pi(S)``.

    NTU (headline): ``S`` blocks structure ``pi`` iff *every* member of ``S`` is
    strictly better off in ``pi(S)`` than in ``pi``. This is the reading that
    matches a framework without side payments, and it yields a set of unblocked
    structures directly comparable to the MPE's absorbing states.

    TU (diagnostic): ``v(S) = sum of member payoffs in pi(S)``, and the core is
    the usual set of efficient, unblockable imputations; non-emptiness is decided
    by a small exact LP. Comparing the two flags the interesting case where the
    TU core is non-empty but the NTU core is not: the gains from cooperation
    exist but cannot be realised without transfers.

    Returns:
        Dict with ``ntu_unblocked`` (list of state names), ``ntu_blockers``
        (state -> list of blocking coalitions as sorted tuples),
        ``tu_core_nonempty`` (bool), ``tu_grand_value``, ``tu_surplus``
        (v(N) minus the largest total achievable by a two-part split), and
        ``transfers_would_help`` (bool).
    """
    require_cartel_lattice(game)

    players = game.players
    grand = game.state_for(frozenset(players))

    all_coalitions = [
        frozenset(combo)
        for size in range(2, len(players) + 1)
        for combo in combinations(players, size)
    ]

    ntu_blockers: Dict[str, List[tuple]] = {}
    ntu_unblocked: List[str] = []
    ntu_margins: List[float] = []
    for state in game.state_names:
        try:
            game.coalition_of(state)
        except ValueError:
            continue
        blockers = []
        for coalition in all_coalitions:
            deviation = game.state_for(coalition)
            if deviation == state:
                continue
            gains = [
                game.payoffs.loc[deviation, member] - game.payoffs.loc[state, member]
                for member in coalition
            ]
            # How far this coalition is from flipping between blocking and not:
            # the member whose gain is closest to zero decides.
            closest = min(abs(g) for g in gains)
            if closest > 0:
                ntu_margins.append(closest)
            if all(g > 0 for g in gains):
                blockers.append(tuple(sorted(coalition)))
        ntu_blockers[state] = blockers
        if not blockers:
            ntu_unblocked.append(state)

    v = {}
    for coalition in all_coalitions:
        structure = game.state_for(coalition)
        v[coalition] = float(game.payoffs.loc[structure, sorted(coalition)].sum())
    singletons_state = game.state_for(frozenset())
    for player in players:
        v[frozenset({player})] = float(game.payoffs.loc[singletons_state, player])

    grand_key = frozenset(players)
    tu_nonempty = _tu_core_nonempty(players, v, grand_key)

    # Is full cooperation even efficient? Measured on total welfare of actual
    # structures rather than on v_gamma, because the gamma-characteristic
    # function is not a partition function: v_gamma(S) + v_gamma(N\S) double
    # counts externalities and is not the welfare of any single structure.
    others = [s for s in game.state_names if s != grand]
    grand_efficiency_surplus = (
        game.welfare(grand) - max(game.welfare(s) for s in others) if others else 0.0
    )

    return {
        "ntu_unblocked": ntu_unblocked,
        "ntu_blockers": ntu_blockers,
        "ntu_grand_unblocked": grand in ntu_unblocked,
        "ntu_min_margin": min(ntu_margins) if ntu_margins else float("nan"),
        "tu_core_nonempty": tu_nonempty,
        "tu_grand_value": v[grand_key],
        "grand_efficiency_surplus": grand_efficiency_surplus,
        "transfers_would_help": bool(tu_nonempty and grand not in ntu_unblocked),
    }


def _tu_core_nonempty(players: List[str], v: Dict[FrozenSet[str], float],
                      grand: FrozenSet[str]) -> bool:
    """Decide TU-core non-emptiness by linear programming.

    Feasibility of {x : sum(x) = v(N), x(S) >= v(S) for all S} is posed as a
    minimisation with no objective; ``linprog`` reports whether the polytope is
    non-empty.
    """
    n = len(players)
    index = {p: i for i, p in enumerate(players)}

    # -x(S) <= -v(S)
    A_ub, b_ub = [], []
    for coalition, value in v.items():
        if coalition == grand:
            continue
        row = np.zeros(n)
        for member in coalition:
            row[index[member]] = -1.0
        A_ub.append(row)
        b_ub.append(-value)

    A_eq = np.ones((1, n))
    b_eq = np.array([v[grand]])

    result = linprog(
        c=np.zeros(n),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * n,
        method="highs",
    )
    if result.status not in (0, 2):
        raise RuntimeError(
            f"TU-core feasibility LP failed with status {result.status}: {result.message}"
        )
    return bool(result.status == 0)
