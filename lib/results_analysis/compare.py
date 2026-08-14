"""Compare farsighted MPE predictions against the static benchmarks.

Reads the CSVs written by ``scripts/rice_hardness_report.py``, reconstructs each
solved equilibrium, and emits one row per (payoff table x effectivity rule) with
the prediction of each solution concept and a classification of how they differ.

Usage:
    python -m lib.results_analysis.compare \\
        reports/rice_hardness_jeres_vfi_heyen_lehtomaa_2021_20260813_105433.csv \\
        reports/rice_hardness_jeres_vfi_adjacent_step_20260813_104442.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.mdp import absorbing_sets, limiting_distribution
from lib.results_analysis.benchmarks import (
    ContaminatedPayoffTable,
    PayoffGame,
    gamma_core,
    internal_external_stability,
)
from lib.utils import get_approval_committee
from lib.verify_cli import run_verification

KNOWN_EFFECTIVITY_RULES = (
    "heyen_lehtomaa_2021",
    "unanimous_consent",
    "deployer_exit",
    "free_exit",
    "adjacent_step",
)

# The exclusive relation between the MPE's predicted set and a static concept's.
# These five partition every pair of non-empty sets. Everything else the
# comparison records — cycling, frozen chains, empty static sets, knife-edge
# margins, rule sensitivity — lives in separate boolean columns, because those
# are properties of one side alone and can co-occur with any relation.
RELATIONS = (
    "AGREE",           # same set
    "MPE_SELECTS",     # MPE picks a strict subset: equilibrium selection
    "MPE_ADDS",        # MPE admits strictly more
    "OVERLAPPING",     # share some structures, neither contains the other
    "DISJOINT",        # no shared prediction at all
)


def rule_from_filename(path: Path) -> str:
    """Infer the effectivity rule from a hardness-report filename.

    Raises rather than guessing: an unlabelled CSV would silently produce rows
    attributed to the wrong governance rule.
    """
    stem = path.stem
    matches = [rule for rule in KNOWN_EFFECTIVITY_RULES if rule in stem]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"Cannot tell which effectivity rule {path.name} used. "
            f"Pass --rule explicitly (one of {list(KNOWN_EFFECTIVITY_RULES)})."
        )
    raise ValueError(
        f"Filename {path.name} mentions several effectivity rules {matches}; "
        "pass --rule explicitly."
    )


def mpe_outcome(profile_path: Path, effectivity_rule: str,
                verify_atol: float) -> Dict[str, object]:
    """Reconstruct a solved equilibrium and reduce it to its predicted outcome.

    Returns the absorbing structures, the long-run distribution over them, and
    the outcome scalars (expected SAI level, welfare, coalition size) implied by
    that distribution starting from a uniform prior over states.
    """
    verified, message, details = run_verification(
        profile_path,
        effectivity_rule=effectivity_rule,
        atol=verify_atol,
        quiet=True,
    )

    P = details["P"]
    state_names = details["state_names"]
    payoffs = details["payoffs"]
    geo = details.get("geoengineering")

    game = PayoffGame.from_resolved(payoffs, details["players"], geo=geo,
                                    allow_contaminated=True)

    classes = absorbing_sets(P)
    pi = limiting_distribution(P)
    pi_series = pd.Series(pi, index=state_names)

    absorbing_states = sorted({state_names[i] for members in classes for i in members})
    cyclic = [
        sorted(state_names[i] for i in members) for members in classes if len(members) > 1
    ]

    expected_geo = (
        float((pi_series * game.geo.reindex(state_names)).sum())
        if game.geo is not None
        else np.nan
    )
    welfare = pd.Series({s: game.welfare(s) for s in state_names})
    sizes = pd.Series({s: game.largest_coalition_size(s) for s in state_names})

    return {
        "game": game,
        "details": details,
        "verified_on_reload": verified,
        "verification_message": message.splitlines()[0] if message else "",
        "absorbing_states": absorbing_states,
        "n_absorbing_classes": len(classes),
        "cyclic_classes": cyclic,
        "limiting_distribution": pi_series,
        "expected_geo": expected_geo,
        "expected_welfare": float((pi_series * welfare).sum()),
        "expected_coalition_size": float((pi_series * sizes).sum()),
    }


def decisive_value_gaps(details: Dict[str, object]) -> Dict[str, float]:
    """Smallest value difference the equilibrium's strategies actually rely on.

    The approval condition says: approve iff V(next) > V(current). Where an
    approver plays a pure 0 or 1, the equilibrium is resting on the *sign* of
    that difference. If the smallest such difference is below the tolerance used
    to verify the profile, the designation is an artefact of the tolerance
    rather than a property of the model — a different atol would accept a
    different absorbing set.

    This is the dynamic counterpart of ``ies_min_margin``, which measures the
    same fragility on static payoffs.
    """
    V = details["V"].astype(float)
    # Values that differ only by floating-point residue are mathematically equal:
    # the approver is genuinely indifferent, any approval probability is valid,
    # and nothing rests on the difference. Only differences above this floor are
    # real strict preferences that a tighter tolerance could turn decisive.
    numerical_zero = 1e-12 * float(V.abs().to_numpy().max())

    players, states = details["players"], details["state_names"]
    effectivity = details["effectivity"]
    strategy = details["strategy_df"]
    forbidden = details.get("forbidden_proposals", frozenset())

    gaps = []
    for proposer in players:
        for current in states:
            for nxt in states:
                if (proposer, current, nxt) in forbidden or current == nxt:
                    continue
                for approver in get_approval_committee(
                    effectivity, players, proposer, current, nxt
                ):
                    try:
                        p_approve = float(strategy.loc[
                            (current, "Acceptance", approver),
                            (f"Proposer {proposer}", nxt)])
                    except KeyError:
                        continue
                    if p_approve not in (0.0, 1.0):
                        continue  # mixing: the player is indifferent by design
                    gap = abs(V.loc[nxt, approver] - V.loc[current, approver])
                    if gap > numerical_zero:
                        gaps.append(gap)

    return {
        "mpe_min_decisive_gap": min(gaps) if gaps else float("nan"),
        "mpe_value_spread": float((V.max() - V.min()).max()),
    }


def _mean_over(states: List[str], values: pd.Series) -> float:
    """Unweighted mean of ``values`` over a predicted set of states."""
    if not states:
        return np.nan
    return float(np.mean([values[s] for s in states]))


def set_relation(mpe_states: Sequence[str], static_states: Sequence[str]) -> Optional[str]:
    """How the MPE's predicted set stands to a static concept's predicted set.

    This is the only genuinely exclusive axis of the comparison: the five
    outcomes partition every pair of non-empty sets. Properties of one side
    alone — a cycling or frozen chain, an empty or all-inclusive static set —
    are *not* encoded here. They are reported as independent flags, because a
    cycling MPE can equally well be a subset, a superset, or disjoint, and
    folding them into one label would hide whichever came second.

    Returns None when either side is empty, in which case the containment tests
    hold vacuously and carry no information.
    """
    mpe_set, static_set = set(mpe_states), set(static_states)
    if not mpe_set or not static_set:
        return None
    if mpe_set == static_set:
        return "AGREE"
    if mpe_set < static_set:
        return "MPE_SELECTS"
    if static_set < mpe_set:
        return "MPE_ADDS"
    if mpe_set & static_set:
        return "OVERLAPPING"
    return "DISJOINT"


def _compare_against(name: str, mpe_states: List[str], static_states: List[str],
                     outcome: Dict[str, object], game: PayoffGame,
                     sizes: pd.Series, geo: pd.Series,
                     welfare: pd.Series, uninformative: bool) -> Dict[str, object]:
    """Relation and signed gaps between the MPE and one static concept.

    The gaps replace the direction labels this taxonomy used to carry. Coalition
    size, deployment and welfare do not move together in a free-driver setting —
    a larger coalition averages over more heterogeneous ideal cooling levels and
    can deploy *less* — so collapsing them into a single "more/less cooperative"
    verdict picked one proxy and hid the disagreement between the other two.
    """
    relation = None if uninformative else set_relation(mpe_states, static_states)
    return {
        f"relation_vs_{name}": relation,
        f"delta_size_vs_{name}": (
            outcome["expected_coalition_size"] - _mean_over(static_states, sizes)
        ),
        f"delta_geo_vs_{name}": outcome["expected_geo"] - _mean_over(static_states, geo),
        f"delta_welfare_vs_{name}": (
            outcome["expected_welfare"] - _mean_over(static_states, welfare)
        ),
    }


def analyse_case(payoff_table: str, profile_path: Path, effectivity_rule: str,
                 verify_atol: float) -> Dict[str, object]:
    """Build the full comparison row for one (payoff table, effectivity rule)."""
    outcome = mpe_outcome(profile_path, effectivity_rule, verify_atol)
    game: PayoffGame = outcome["game"]

    contaminated = bool(
        (game.payoffs <= -9999.0).any().any()
    )

    ies = internal_external_stability(game)
    core = gamma_core(game)

    ies_open = ies.index[ies["stable_open"]].tolist()
    ies_consent = ies.index[ies["stable_consent"]].tolist()
    ntu_unblocked = core["ntu_unblocked"]

    sizes = pd.Series({s: game.largest_coalition_size(s) for s in game.state_names})
    geo = game.geo if game.geo is not None else pd.Series(np.nan, index=game.state_names)
    welfare = pd.Series({s: game.welfare(s) for s in game.state_names})

    mpe_states = outcome["absorbing_states"]
    n_states = len(game.state_names)
    # Every state absorbing means P is the identity: the chain never moves, so
    # the equilibrium makes no prediction and trivially "contains" any static
    # set. Same for a static concept that declares every structure stable.
    mpe_frozen = len(mpe_states) == n_states

    row = {
        "payoff_table": payoff_table,
        "effectivity_rule": effectivity_rule,
        "profile": profile_path.name,
        "players": ",".join(game.players),
        "contaminated_payoffs": contaminated,
        "verified_on_reload": outcome["verified_on_reload"],
        # --- farsighted MPE ---
        "mpe_absorbing": "|".join(mpe_states),
        "mpe_n_absorbing_classes": outcome["n_absorbing_classes"],
        "mpe_cyclic_classes": "|".join(
            "+".join(cycle) for cycle in outcome["cyclic_classes"]
        ),
        "mpe_expected_geo": outcome["expected_geo"],
        "mpe_expected_welfare": outcome["expected_welfare"],
        "mpe_expected_coalition_size": outcome["expected_coalition_size"],
        **decisive_value_gaps(outcome["details"]),
        # --- internal / external stability ---
        "ies_stable_open": "|".join(ies_open),
        "ies_stable_consent": "|".join(ies_consent),
        "ies_open_geo": _mean_over(ies_open, geo),
        "ies_open_welfare": _mean_over(ies_open, welfare),
        "ies_open_coalition_size": _mean_over(ies_open, sizes),
        "ies_min_margin": float(ies["min_margin"].min()),
        "ies_n_exact_ties": int(ies["n_exact_ties"].sum()),
        # --- gamma core ---
        "gamma_ntu_unblocked": "|".join(ntu_unblocked),
        "gamma_ntu_grand_unblocked": core["ntu_grand_unblocked"],
        "gamma_ntu_geo": _mean_over(ntu_unblocked, geo),
        "gamma_ntu_welfare": _mean_over(ntu_unblocked, welfare),
        "gamma_ntu_coalition_size": _mean_over(ntu_unblocked, sizes),
        "gamma_ntu_min_margin": core["ntu_min_margin"],
        "gamma_tu_core_nonempty": core["tu_core_nonempty"],
        "gamma_transfers_would_help": core["transfers_would_help"],
        "grand_efficiency_surplus": core["grand_efficiency_surplus"],
        # --- flags: properties of one side alone, orthogonal to the relation ---
        "mpe_cyclic": bool(outcome["cyclic_classes"]),
        "mpe_frozen": mpe_frozen,
        "mpe_point_valued": len(mpe_states) == 1,
        "ies_empty": not ies_open,
        "ies_all_stable": len(ies_open) == n_states,
        "ies_point_valued": len(ies_open) == 1,
        "ies_consent_empty": not ies_consent,
        "ies_consent_all_stable": len(ies_consent) == n_states,
        "ies_consent_point_valued": len(ies_consent) == 1,
        "gamma_empty": not ntu_unblocked,
        "gamma_all_unblocked": len(ntu_unblocked) == n_states,
        "gamma_point_valued": len(ntu_unblocked) == 1,
    }
    row.update(_compare_against("ies", mpe_states, ies_open, outcome, game,
                                sizes, geo, welfare,
                                uninformative=mpe_frozen or len(ies_open) == n_states))
    row.update(_compare_against(
        "ies_consent", mpe_states, ies_consent, outcome, game, sizes, geo, welfare,
        uninformative=mpe_frozen or len(ies_consent) == n_states))
    row.update(_compare_against("gamma", mpe_states, ntu_unblocked, outcome, game,
                                sizes, geo, welfare,
                                uninformative=mpe_frozen or len(ntu_unblocked) == n_states))
    # RICE payoffs separate the structures only in the fourth or fifth
    # significant figure, so record how large the deciding margins are relative
    # to the payoff scale. A knife-edge verdict is a modelling artefact, not a
    # result about coalition formation.
    scale = float(game.payoffs.abs().to_numpy().max())
    row["payoff_scale"] = scale
    row["ies_relative_margin"] = row["ies_min_margin"] / scale if scale else np.nan
    # The equilibrium was verified at `verify_atol`. If the smallest value
    # difference its strategies depend on is below that, a tighter tolerance
    # would have rejected the profile: the absorbing set is tolerance-dependent.
    row["tolerance_headroom"] = row["mpe_min_decisive_gap"] / verify_atol
    row["tolerance_sensitive"] = bool(
        np.isfinite(row["tolerance_headroom"]) and row["tolerance_headroom"] < 1.0
    )
    row["knife_edge"] = bool(
        np.isfinite(row["ies_relative_margin"]) and row["ies_relative_margin"] < 1e-4
    )
    return row


def add_rule_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag payoff tables whose MPE prediction depends on the effectivity rule.

    This is the one column no static concept can produce: internal/external
    stability and the gamma-core are blind to the protocol, so they return the
    same answer for every rule by construction.
    """
    frame = frame.copy()
    per_table = frame.groupby("payoff_table")["mpe_absorbing"].nunique()
    n_rules = frame.groupby("payoff_table")["effectivity_rule"].nunique()
    sensitive = (per_table > 1) & (n_rules > 1)
    frame["rule_sensitive"] = frame["payoff_table"].map(sensitive).fillna(False)
    return frame


def _load_cases(csv_paths: List[Path], rules: Optional[List[str]],
                n_players: Optional[int],
                exclude_patterns: Optional[List[str]] = None) -> List[tuple]:
    """Collect (payoff_table, profile_path, rule) triples from hardness CSVs."""
    if rules is not None and len(rules) != len(csv_paths):
        raise ValueError(
            f"Got {len(rules)} --rule values for {len(csv_paths)} CSVs; "
            "supply one per CSV or none at all."
        )
    excluded = [re.compile(pattern) for pattern in (exclude_patterns or [])]

    cases = []
    for position, csv_path in enumerate(csv_paths):
        rule = rules[position] if rules else rule_from_filename(csv_path)
        frame = pd.read_csv(csv_path)

        for _, record in frame.iterrows():
            if n_players is not None and record.get("n_players") != n_players:
                continue
            table_name = str(record.get("payoff_table", ""))
            if any(pattern.search(table_name) for pattern in excluded):
                continue
            if not bool(record.get("verification_success", False)):
                continue
            output_file = record.get("output_file")
            if not isinstance(output_file, str) or not output_file.strip():
                continue
            profile_path = Path(output_file)
            if not profile_path.is_absolute():
                profile_path = REPO_ROOT / profile_path
            if not profile_path.exists():
                raise FileNotFoundError(
                    f"{csv_path.name} references a solved profile that is gone: "
                    f"{profile_path}"
                )
            cases.append((str(record["payoff_table"]), profile_path, rule))
    return cases


def build_comparison(csv_paths: List[Path], rules: Optional[List[str]] = None,
                     n_players: Optional[int] = 3, verify_atol: float = 1e-5,
                     skip_contaminated: bool = True,
                     exclude_patterns: Optional[List[str]] = None,
                     verbose: bool = True) -> tuple:
    """Run the full comparison. Returns (frame, problems)."""
    cases = _load_cases(csv_paths, rules, n_players, exclude_patterns)
    rows, problems = [], []

    for payoff_table, profile_path, rule in cases:
        try:
            row = analyse_case(payoff_table, profile_path, rule, verify_atol)
        except ContaminatedPayoffTable as exc:
            problems.append((payoff_table, rule, f"contaminated: {exc}"))
            continue
        except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
            problems.append((payoff_table, rule, f"{type(exc).__name__}: {exc}"))
            continue

        if not row["verified_on_reload"]:
            problems.append(
                (payoff_table, rule,
                 "equilibrium did not re-verify on reload; excluded")
            )
            continue

        if row["contaminated_payoffs"] and skip_contaminated:
            problems.append(
                (payoff_table, rule,
                 "sentinel payoffs (<= -9999) present; excluded from statistics")
            )
            continue

        rows.append(row)
        if verbose:
            print(
                f"  {payoff_table:52s} {rule:20s} "
                f"{row['relation_vs_ies'] or '(voided)'}",
                file=sys.stderr,
            )

    if not rows:
        raise RuntimeError(
            "No usable cases. Problems encountered:\n  "
            + "\n  ".join(f"{t} [{r}]: {m}" for t, r, m in problems)
        )

    return add_rule_sensitivity(pd.DataFrame(rows)), problems


def _relation_block(frame: pd.DataFrame, name: str) -> List[str]:
    """Relation counts for one static concept, over the rows where it is defined."""
    column = f"relation_vs_{name}"
    defined = frame[frame[column].notna()]
    voided = len(frame) - len(defined)

    counts = defined[column].value_counts()
    lines = [f"Relation defined for {len(defined)} of {len(frame)} rows."]
    if voided:
        lines.append(
            f"({voided} voided: one side named every structure, so containment "
            "holds vacuously.)"
        )
    lines.append("")
    for relation in RELATIONS:
        lines.append(f"- `{relation}`: {int(counts.get(relation, 0))}")

    # An AGREE between two single-structure predictions is a much stronger claim
    # than an AGREE between two shortlists that happen to coincide.
    point = defined[defined["mpe_point_valued"] & defined[f"{name}_point_valued"]]
    agree_point = int((point[column] == "AGREE").sum())
    lines += [
        "",
        f"Of the `AGREE` rows, {agree_point} have both concepts naming a single "
        f"structure (the rest are coinciding shortlists).",
    ]
    return lines


def write_digest(frame: pd.DataFrame, problems: List[tuple], path: Path) -> None:
    """Write a markdown digest ranking the cases worth looking at."""
    lines = [
        "# Farsighted MPE vs. static coalition-formation benchmarks",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. "
        f"{len(frame)} cases over {frame['payoff_table'].nunique()} payoff tables "
        f"and {frame['effectivity_rule'].nunique()} effectivity rules.",
        "",
        "## Divergence from internal/external stability",
        "",
    ]
    lines += _relation_block(frame, "ies")
    lines += [
        "",
        "## Divergence from internal/external stability (consent variant)",
        "",
        "Accession additionally requires the incumbents to gain, matching this "
        "framework's approval committees. This is the like-for-like benchmark; "
        "open membership above lets an outsider join over the incumbents' "
        "objection, which the framework never permits.",
        "",
    ]
    lines += _relation_block(frame, "ies_consent")
    lines += ["", "## Divergence from the NTU gamma-core", ""]
    lines += _relation_block(frame, "gamma")

    lines += [
        "",
        "## Cross-cutting counts",
        "",
        f"- MPE prediction depends on the effectivity rule: "
        f"{int(frame['rule_sensitive'].sum())} rows "
        f"({int(frame.loc[frame['rule_sensitive'], 'payoff_table'].nunique())} tables)",
        f"- Grand coalition unblocked in the NTU gamma-core: "
        f"{int(frame['gamma_ntu_grand_unblocked'].sum())}",
        f"- TU core non-empty but grand coalition NTU-blocked "
        f"(transfers would help): {int(frame['gamma_transfers_would_help'].sum())}",
        f"- Knife-edge IS/ES verdicts (smallest deciding margin < 1e-4 of the "
        f"payoff scale): {int(frame['knife_edge'].sum())} of {len(frame)}",
        f"- Cases containing payoff-identical structures (verdict decided by the "
        f"weak/strict convention): {int((frame['ies_n_exact_ties'] > 0).sum())}",
        f"- **Tolerance-sensitive equilibria** (smallest decisive value gap below "
        f"the verification atol, so a tighter atol would give a different "
        f"absorbing set): {int(frame['tolerance_sensitive'].sum())} of {len(frame)}",
        "",
        "### Single-side flags (orthogonal to the relation)",
        "",
        f"- MPE has a cycling closed class: {int(frame['mpe_cyclic'].sum())}",
        f"- MPE frozen (every state absorbing, P = I): {int(frame['mpe_frozen'].sum())}",
        f"- MPE names a single structure: {int(frame['mpe_point_valued'].sum())} "
        f"of {len(frame)}",
        f"- IS/ES set empty: {int(frame['ies_empty'].sum())}; "
        f"names a single structure: {int(frame['ies_point_valued'].sum())}",
        f"- Gamma-core NTU set empty: {int(frame['gamma_empty'].sum())}; "
        f"names a single structure: {int(frame['gamma_point_valued'].sum())}",
        "",
        "## Cases worth a figure",
        "",
    ]

    interesting = frame[
        frame["relation_vs_ies"].isin(("MPE_SELECTS", "MPE_ADDS", "DISJOINT",
                                       "OVERLAPPING"))
        | frame["mpe_cyclic"]
        | frame["rule_sensitive"]
    ].copy()
    interesting["abs_delta_geo"] = interesting["delta_geo_vs_ies"].abs()
    interesting = interesting.sort_values("abs_delta_geo", ascending=False)

    if interesting.empty:
        lines.append("_No divergent cases._")
    else:
        lines += [
            "| payoff table | rule | relation | MPE absorbing | IS/ES stable "
            "| Δsize | ΔE[W_SAI] | Δwelfare | rel. margin | render key |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, record in interesting.head(25).iterrows():
            lines.append(
                f"| {record['payoff_table']} | {record['effectivity_rule']} "
                f"| {record['relation_vs_ies']}"
                f"{' +cyc' if record['mpe_cyclic'] else ''} "
                f"| {record['mpe_absorbing']} "
                f"| {record['ies_stable_open'] or '—'} "
                f"| {record['delta_size_vs_ies']:+.2f} "
                f"| {record['delta_geo_vs_ies']:+.3f} "
                f"| {record['delta_welfare_vs_ies']:+.4f} "
                f"| {record['ies_relative_margin']:.2e}"
                f"{' ⚠' if record['knife_edge'] else ''} "
                f"| `{Path(record['profile']).stem}` |"
            )
        lines += [
            "",
            "Render any of these with:",
            "",
            "```bash",
            "python viz/render_graph.py <render key> --coloring absorbing -o <name>.png",
            "```",
        ]

    if problems:
        lines += ["", "## Excluded cases", ""]
        for table, rule, message in problems:
            lines.append(f"- `{table}` [{rule}]: {message}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark solved farsighted equilibria against static concepts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_csv", nargs="+", type=Path,
        help="Hardness-report CSV(s) from scripts/rice_hardness_report.py.",
    )
    parser.add_argument(
        "--rule", action="append", choices=KNOWN_EFFECTIVITY_RULES,
        help="Effectivity rule for each CSV, in order. Inferred from the "
             "filename when omitted.",
    )
    parser.add_argument(
        "--n-players", type=int, default=3,
        help="Restrict to cases with this many players (0 for no restriction).",
    )
    parser.add_argument(
        "--verify-atol", type=float, default=1e-5,
        help="Tolerance for re-verifying reloaded profiles (matches the solver runs).",
    )
    parser.add_argument(
        "--exclude", action="append", metavar="REGEX",
        help="Skip payoff tables whose name matches this regex. Repeatable. "
             "Use --exclude simple_cycle for the tables the hardness harness "
             "poses without --allow-non-canonical-states/--effectivity-rule "
             "free_exit, whose equilibria answer a different question.",
    )
    parser.add_argument(
        "--include-contaminated", action="store_true",
        help="Keep tables containing sentinel (-10000) payoffs. Their stability "
             "results are meaningless; they are excluded by default.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "reports",
        help="Where to write the CSV and markdown digest.",
    )
    args = parser.parse_args()

    frame, problems = build_comparison(
        args.results_csv,
        rules=args.rule,
        n_players=args.n_players or None,
        verify_atol=args.verify_atol,
        skip_contaminated=not args.include_contaminated,
        exclude_patterns=args.exclude,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"benchmark_comparison_{stamp}.csv"
    digest_path = args.out_dir / f"benchmark_comparison_{stamp}.md"

    frame.to_csv(csv_path, index=False)
    write_digest(frame, problems, digest_path)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {digest_path}")
    print(f"\n{len(frame)} cases analysed, {len(problems)} excluded.")
    for label, column in (("open membership", "relation_vs_ies"),
                          ("consent", "relation_vs_ies_consent")):
        print(f"\nRelation to internal/external stability ({label}):")
        for relation, count in frame[column].value_counts().items():
            print(f"  {relation:16s} {count}")
        voided = int(frame[column].isna().sum())
        if voided:
            print(f"  {'(voided)':16s} {voided}")


if __name__ == "__main__":
    main()
