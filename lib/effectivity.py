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
            current_members = list_members(current_state)

            for next_state in states:
                next_members = list_members(next_state)

                for responder in players:
                    key = (proposer, current_state, next_state, responder)

                    # Rule 1: Status quo - only proposer approves
                    if current_state == next_state:
                        effectivity[key] = 1 if responder == proposer else 0
                        continue

                    # Rules 2a/2b: Exit-type transition.
                    # _exit_committee() returns the set of actively exiting players
                    # (proposer + any co-exiters), or empty if not exit-type.
                    committee = _exit_committee(proposer, current_state, next_state)

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

                    current_coalition = get_player_coalition(responder, current_state)
                    next_coalition = get_player_coalition(responder, next_state)
                    switching = (was_in_coalition and is_in_coalition and
                                 current_coalition != next_coalition)

                    effectivity[key] = 1 if (joining or leaving or switching) else 0

    return effectivity


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


# Registry of available effectivity rules
EFFECTIVITY_RULES = {
    'heyen_lehtomaa_2021': heyen_lehtomaa_2021,
    'unanimous_consent': unanimous_consent,
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
