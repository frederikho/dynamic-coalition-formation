"""
Effectivity correspondence rules for coalition formation games.

Effectivity determines which players must approve a proposed transition
from one coalition structure to another.
"""

from lib.utils import list_members, list_coalitions, get_player_coalition, _exit_committee


def heyen_lehtomaa_2021(players: list, states: list) -> dict:
    """
    Generate effectivity correspondence based on Heyen & Lehtomaa (2021),
    generalised to work for any number of players n >= 3.

    Rules:
    1. Status quo (x → x): Only proposer approves.
    2. Exit-type transition (proposer leaves their current coalition, no cross-coalition
       mergers occur):
       a. Unilateral exit (proposer is the only player leaving): only proposer approves.
       b. Multi-lateral exit (other players also become singletons from the same coalition):
          "others approve, not proposer" — all co-exiters except the proposer must approve.
          This covers grand-coalition full dissolution and partial multi-lateral exits.
    3. Non-exit transitions (joining, merging, restructuring across coalitions):
       Players whose coalition membership changed must approve, except the proposer.

    Args:
        players: List of player names
        states: List of state names

    Returns:
        Effectivity dictionary with keys (proposer, current_state, next_state, responder)
        Values are 1 (in committee) or 0 (not in committee)
    """
    effectivity = {}

    for proposer in players:
        for current_state in states:
            current_members = list_members(current_state, players)

            for next_state in states:
                next_members = list_members(next_state, players)

                for responder in players:
                    key = (proposer, current_state, next_state, responder)

                    # Rule 1: Status quo - only proposer approves
                    if current_state == next_state:
                        effectivity[key] = 1 if responder == proposer else 0
                        continue

                    # Rules 2a/2b: Exit-type transition.
                    # _exit_committee() returns the set of actively exiting players
                    # (proposer + any co-exiters), or empty if not exit-type.
                    committee = _exit_committee(proposer, current_state, next_state, players)

                    if committee:
                        if len(committee) == 1:
                            # 2a. Unilateral exit: only proposer approves
                            effectivity[key] = 1 if responder == proposer else 0
                        else:
                            # 2b. Multi-lateral exit: others approve, not proposer
                            effectivity[key] = 1 if (
                                responder in committee and responder != proposer
                            ) else 0
                        continue

                    # Rule 3: Non-exit transitions (joining, merging, restructuring)
                    # Proposer never in approval committee
                    if responder == proposer:
                        effectivity[key] = 0
                        continue

                    # Players whose coalition membership changed must approve
                    was_in_coalition = responder in current_members
                    is_in_coalition = responder in next_members

                    joining = is_in_coalition and not was_in_coalition
                    leaving = was_in_coalition and not is_in_coalition

                    current_coalition = get_player_coalition(responder, current_state, players)
                    next_coalition = get_player_coalition(responder, next_state, players)
                    switching = (was_in_coalition and is_in_coalition and
                                 current_coalition != next_coalition)

                    effectivity[key] = 1 if (joining or leaving or switching) else 0

    return effectivity


def deployer_exit(players: list, states: list) -> dict:
    """
    Effectivity rule for reduced deployer-state models (e.g. a 2-player / 3-state
    'battle of the sexes' setup with states '( )', '(CHN)', '(USA)').

    Rules:
    - Status quo (x → x): only proposer approves.
    - (X) → ( ): only player X must approve.  If X is the proposer the transition
                  is self-approved (unilateral exit).  If another player proposes,
                  X must consent.
    - All other non-status-quo transitions: every non-proposer must approve
                  (unanimity among all players except the proposer).
    """
    from lib.utils import list_members

    effectivity = {}
    for proposer in players:
        for current_state in states:
            for next_state in states:
                for responder in players:
                    key = (proposer, current_state, next_state, responder)

                    if current_state == next_state:
                        effectivity[key] = 1 if responder == proposer else 0
                        continue

                    # (X) → ( ): only the named deployer X must approve
                    current_deployers = list_members(current_state, players)
                    if next_state == "( )" and len(current_deployers) == 1:
                        deployer = current_deployers[0]
                        effectivity[key] = 1 if responder == deployer else 0
                    else:
                        # All other transitions: unanimity (every non-proposer approves)
                        effectivity[key] = 0 if responder == proposer else 1

    return effectivity


def deployer_exit_forbidden_proposals(players: list, states: list) -> frozenset:
    """
    Return forbidden (proposer, current_state, next_state) triples for deployer_exit.

    Non-deployers cannot propose exiting a deployer state to ( ).  That is, for
    any state '(X)', only player X may propose the transition '(X) → ( )'.
    """
    from lib.utils import list_members
    forbidden = set()
    for current_state in states:
        if current_state == "( )":
            continue
        deployers = list_members(current_state, players)
        if len(deployers) != 1:
            continue
        deployer = deployers[0]
        for proposer in players:
            if proposer != deployer:
                forbidden.add((proposer, current_state, "( )"))
    return frozenset(forbidden)


# Registry of forbidden-proposal functions, keyed by effectivity rule name.
# Rules with no forbidden proposals are absent from this registry.
FORBIDDEN_PROPOSALS_RULES = {
    'deployer_exit': deployer_exit_forbidden_proposals,
}


def get_forbidden_proposals(rule_name: str, players: list, states: list) -> frozenset:
    """Return forbidden (proposer, current_state, next_state) triples for a rule.

    Returns an empty frozenset if the rule has no forbidden proposals.
    """
    fn = FORBIDDEN_PROPOSALS_RULES.get(rule_name)
    if fn is None:
        return frozenset()
    return fn(players, states)


def unanimous_consent(players: list, states: list) -> dict:
    """
    Alternative effectivity rule: All players must unanimously approve all transitions.

    This is a more restrictive rule that gives every player veto power over any
    proposed change. It prevents unilateral exit and requires full consensus for
    any coalition formation or dissolution.

    Rationale: This rule might be more appropriate for contexts where:
    - International treaties require universal consent
    - Property rights are strong (no forced membership)
    - Exit from coalitions requires permission from all parties
    - Coalition formation needs buy-in from all countries, not just affected ones

    Note: This makes coalition changes much harder and likely results in more
    stable but potentially less efficient outcomes.

    Args:
        players: List of player names
        states: List of state names

    Returns:
        Effectivity dictionary with keys (proposer, current_state, next_state, responder)
        Values are 1 (in committee) or 0 (not in committee)
    """
    effectivity = {}

    for proposer in players:
        for current_state in states:
            for next_state in states:
                for responder in players:
                    key = (proposer, current_state, next_state, responder)

                    # Status quo: only proposer approves
                    if current_state == next_state:
                        effectivity[key] = 1 if responder == proposer else 0
                    else:
                        # All transitions require unanimous approval from all players
                        effectivity[key] = 1

    return effectivity


def check_effectivity(
    file_effectivity: dict,
    players: list,
    states: list,
    rule: str = "heyen_lehtomaa_2021",
) -> list[str]:
    """
    Compare a file-derived effectivity against the rule-generated one.

    Returns a list of violation strings (empty = no violations).

    Two kinds of violations:
      - EXTRA:   file includes a player in the committee who shouldn't be there.
                 Always reported — this is an unconditional rule breach.
      - MISSING: file excludes a player who should be in the committee.
                 This is always reported, including when the whole committee is
                 left blank in the file. Otherwise a feasible deviation can be
                 silently reinterpreted as impossible during verification.

    Status-quo transitions (x → x) are skipped (their committee is always just
    the proposer and they are not represented as ordinary acceptance rows).

    Args:
        file_effectivity: Effectivity dict derived from the strategy Excel (via derive_effectivity).
        players:          List of player names.
        states:           List of state names.
        rule:             Name of the rule to validate against (default: heyen_lehtomaa_2021).

    Returns:
        List of human-readable violation strings.
    """
    expected = EFFECTIVITY_RULES[rule](players, states)
    violations = []

    for proposer in players:
        for current_state in states:
            for next_state in states:
                if current_state == next_state:
                    continue

                for responder in players:
                    key = (proposer, current_state, next_state, responder)
                    expected_val = expected.get(key, 0)
                    file_val = file_effectivity.get(key, 0)
                    if expected_val == file_val:
                        continue
                    if file_val == 1 and expected_val == 0:
                        violations.append(
                            f"EXTRA:   {responder} in committee for "
                            f"{proposer}: {current_state} → {next_state} "
                            f"(rule says not in committee)"
                        )
                    elif file_val == 0 and expected_val == 1:
                        violations.append(
                            f"MISSING: {responder} not in committee for "
                            f"{proposer}: {current_state} → {next_state} "
                            f"(rule says should be in committee)"
                        )

    return violations


# Registry of available effectivity rules
EFFECTIVITY_RULES = {
    'heyen_lehtomaa_2021': heyen_lehtomaa_2021,
    'unanimous_consent': unanimous_consent,
    'deployer_exit': deployer_exit,
}


def get_effectivity(rule_name: str, players: list, states: list) -> dict:
    """
    Get effectivity correspondence using a named rule.

    Args:
        rule_name: Name of the effectivity rule to use
        players: List of player names
        states: List of state names

    Returns:
        Effectivity dictionary

    Raises:
        ValueError: If rule_name is not recognized
    """
    if rule_name not in EFFECTIVITY_RULES:
        available = ', '.join(EFFECTIVITY_RULES.keys())
        raise ValueError(f"Unknown effectivity rule '{rule_name}'. Available: {available}")

    return EFFECTIVITY_RULES[rule_name](players, states)
