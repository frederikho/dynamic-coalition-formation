"""
Effectivity correspondence rules for coalition formation games.

Effectivity determines which players must approve a proposed transition
from one coalition structure to another.
"""

from lib.utils import list_members


def heyen_lehtomaa_2021(players: list, states: list) -> dict:
    """
    Generate effectivity correspondence based on Heyen & Lehtomaa (2021).

    Rules derived from strategy tables:
    1. Status quo (x → x): Only proposer approves
    2. Transitions to ( ):
       - From grand coalition: all non-proposers approve
       - Proposer in coalition: only proposer approves (unilateral exit)
       - Proposer not in coalition: coalition members approve
    3. From grand coalition to smaller coalition:
       - Proposer not in result: only proposer approves (unilateral exit)
       - Proposer stays in result: all non-proposers approve
    4. Other transitions: Players whose coalition membership changes must approve,
       EXCEPT the proposer (proposer makes the proposal, doesn't approve it)

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

                    # Rule 2: Transitions involving ( )
                    if next_state == '( )':
                        # Special case: From grand coalition (all in coalition)
                        if len(current_members) == len(players):
                            # Breaking up grand coalition: others approve, not proposer
                            effectivity[key] = 1 if responder != proposer and responder in current_members else 0
                            continue
                        # Normal case: proposer in coalition - unilateral exit
                        elif proposer in current_members:
                            effectivity[key] = 1 if responder == proposer else 0
                            continue
                        # Proposer NOT in coalition: coalition members must approve
                        else:
                            effectivity[key] = 1 if responder in current_members else 0
                            continue

                    # Rule 3: From grand coalition to smaller coalition
                    if len(current_members) == len(players) and next_state != '( )':
                        # Two sub-cases:
                        # 3a. Proposer NOT in resulting coalition: only proposer approves (unilateral exit)
                        if proposer not in next_members:
                            effectivity[key] = 1 if responder == proposer else 0
                            continue
                        # 3b. Proposer stays in resulting coalition: all non-proposers approve
                        else:
                            effectivity[key] = 1 if responder != proposer else 0
                            continue

                    # Rule 4: Non-status quo transitions (except special cases above)
                    # Proposer never in approval committee
                    if responder == proposer:
                        effectivity[key] = 0
                        continue

                    # Check if responder's coalition membership changed
                    was_in_coalition = responder in current_members
                    is_in_coalition = responder in next_members

                    # Joining: was not in coalition, now is
                    joining = is_in_coalition and not was_in_coalition

                    # Leaving: was in coalition, now not
                    leaving = was_in_coalition and not is_in_coalition

                    # Switching: in both, but different coalitions
                    # If in both lists but the coalition composition changed, they switched
                    switching = (was_in_coalition and is_in_coalition and
                                current_members != next_members)

                    # Responder must approve if their membership changed
                    if joining or leaving or switching:
                        effectivity[key] = 1
                    else:
                        effectivity[key] = 0

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
