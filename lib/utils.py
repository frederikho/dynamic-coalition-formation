import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from lib.state import State


def _escape_latex(s: str) -> str:
    """Escape common LaTeX special characters in a string for safe captions."""
    if not isinstance(s, str):
        return s
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def get_payoff_matrix(states: List[State], columns: List[str]) -> pd.DataFrame:
    """
    Calculate a payoff matrix for all states and countries in the game.

    Arguments:
        states: A list of State instances (all states considered in the game).
        columns: List (str) of all player names included in the game.
    """
    assert all(isinstance(state, State) for state in states)

    state_names = [state.name for state in states]
    payoffs_df = pd.DataFrame(index=state_names, columns=columns,
                              dtype=np.float64)

    for state in states:
        assert list(state.payoffs.keys()) == columns,\
            "Payoff matrix cols and payoff dict keys do not match!"
        payoffs_df.loc[state.name, :] = state.payoffs

    return payoffs_df


def get_geoengineering_levels(states: List[State]) -> pd.DataFrame:
    """
    Returns the geoengineering deployment level for a given state.

    Arguments:
        states: A list of State instances (all states considered in the game).
    """
    assert all(isinstance(state, State) for state in states)

    G = {}
    for state in states:
        G[state.name] = state.geo_deployment_level

    return pd.DataFrame.from_dict(G, orient='index', columns=["G"])


def get_deploying_coalitions(states: List[State]) -> Dict[str, str]:
    """
    Returns which coalition deploys geoengineering in each state.

    Arguments:
        states: A list of State instances (all states considered in the game).

    Returns:
        Dictionary mapping state names to deploying coalition names.
        For example:
        - "( )" -> "W" (singleton W deploys)
        - "(CF)" -> "(CF)" (coalition CF deploys)
        - "(CT)(FW)" -> "(CT)" (coalition CT deploys, FW doesn't)
    """
    assert all(isinstance(state, State) for state in states)

    # Standard player order: H, W, T, C, F (and A, B, D, E, G if needed)
    standard_order = ['H', 'W', 'T', 'C', 'F', 'A', 'B', 'D', 'E', 'G']

    def sort_by_standard_order(names):
        """Sort player names by standard order."""
        return sorted(names, key=lambda x: standard_order.index(x) if x in standard_order else 999)

    deployers = {}
    for state in states:
        # If G=0, no one actually deploys
        if state.geo_deployment_level == 0:
            deployer_name = "None"
        else:
            # Get the strongest coalition that deploys
            strongest = state.strongest_coalition

            # Get member names
            member_names = [country.name for country in strongest.members]

            # Format as coalition name using standard order
            if len(member_names) == 0:
                deployer_name = "( )"
            elif len(member_names) == 1:
                deployer_name = member_names[0]
            else:
                sorted_names = sort_by_standard_order(member_names)
                deployer_name = f"({''.join(sorted_names)})"

        deployers[state.name] = deployer_name

    return deployers


def list_members(state: str) -> List[str]:
    """ Lists all the member countries of all existing coalitions.

    For instance:
        list_members('(WTC)') returns ['W', 'T', 'C']
        list_members('(CT)(FW)') returns ['C', 'T', 'F', 'W']
        list_members('( )') returns []
    """
    import re
    # Find all parenthesized groups and extract letters
    coalitions = re.findall(r'\(([A-Z]*)\)', state)
    # Flatten and return all members from all coalitions
    members = []
    for coalition in coalitions:
        members.extend(list(coalition))
    return members


def list_coalitions(state: str) -> List[List[str]]:
    """Lists all non-singleton coalitions in a state as lists of members.

    For instance:
        list_coalitions('(WTC)') returns [['W', 'T', 'C']]
        list_coalitions('(CT)(FW)') returns [['C', 'T'], ['F', 'W']]
        list_coalitions('( )') returns []
        list_coalitions('(CT)') returns [['C', 'T']]
    """
    import re
    # Find all parenthesized groups with at least 2 letters
    coalitions = re.findall(r'\(([A-Z]{2,})\)', state)
    return [list(coal) for coal in coalitions]


def get_player_coalition(player: str, state: str) -> List[str]:
    """Returns the coalition that a player belongs to in a given state.

    Returns a sorted list of all members in the player's coalition,
    or [player] if the player is a singleton.

    For instance:
        get_player_coalition('C', '(CF)(TW)') returns ['C', 'F']
        get_player_coalition('C', '(CFTW)') returns ['C', 'F', 'T', 'W']
        get_player_coalition('C', '( )') returns ['C']
        get_player_coalition('W', '(CF)') returns ['W']
    """
    import re
    # Find all parenthesized groups
    coalitions = re.findall(r'\(([A-Z]*)\)', state)

    for coalition in coalitions:
        if player in coalition:
            return sorted(list(coalition))

    # Player is a singleton
    return [player]


def get_approval_committee(effectivity: Dict[tuple, int], players: List[str],
                           proposer: str, current_state: str,
                           next_state: str) -> List[str]:
    """Returns the list of all players who belong to the approval committee
    when proposer proposes the transition (current_state) -> (next_state).

    Arguments:
        effectivity: The effectivity correspondence, from derive_effectivity().
        players: The list (string) of all countries in the game.
        proposer: The current proposer country.
        current_state: Current coalition structure of the game.
        next_state: The next coalition structure suggested by proposer.
    """

    comm = [player for player in players
            if effectivity[(proposer, current_state, next_state, player)] == 1]

    return comm


def derive_effectivity(df: pd.DataFrame, players: List[str],
                       states: List[str]) -> Dict[tuple, int]:
    """ Defines the effectivity correspondence from the strategy profiles.

    For each possible proposer, every possible state transition, and
    every possible other player as a responder, the effectivity matrix
    has a value of 1 if that responder is in the approval committee, and
    0 otherwise.

    Note that the set of responders is the full set of players. That is, 
    we also consider cases where a proposer can "propose" a transition to
    itself. This is important, as in most settings countries might be able
    to unilaterally exit their current coalition structure.

    Arguments:
        df: A DataFrame instance containing the strategies of all players.
        players: A list (str) of all the players in the game.
        states: A list (str) of all the considered states of the system.

    Returns:
        effectivity: a dictionary with keys being the 4-tuples
                     (proposer, current_state, next_state, responder), and
                     the value being a boolean 0 or 1. Each entry tells
                     whether the responder is a member of the approval
                     committee, when the proposer suggests a transition from
                     the current_state to next_state.
    """

    effectivity = {}

    for proposer in players:
        for current_state in states:
            for next_state in states:
                for responder in players:

                    # If the corresponding 'acceptance' cell is not empty,
                    # the player is a member of the approval committee.
                    resp_val = df.loc[(current_state, 'Acceptance', responder),
                                      (f'Proposer {proposer}', next_state)]
                    is_member = int(~np.isnan(resp_val))

                    idx = (proposer, current_state, next_state, responder)
                    effectivity[idx] = is_member


    return effectivity


def _coalition_structure_as_frozensets(state: str) -> frozenset:
    """Represent a coalition structure as a frozenset of frozensets for semantic comparison.

    Singletons are not listed in state names but every player is implicitly
    present.  This function only returns the *non-singleton* coalitions, which
    is sufficient for exit-type detection.

    '(CT)(FW)' -> frozenset({frozenset({'C','T'}), frozenset({'F','W'})})
    '(WTC)'    -> frozenset({frozenset({'W','T','C'})})
    '( )'      -> frozenset()
    """
    return frozenset(frozenset(c) for c in list_coalitions(state))


def _exit_committee(proposer: str, current_state: str, next_state: str) -> set:
    """Return the set of players who actively exit in this transition, or an
    empty set if the transition is not exit-type.

    A transition is *exit-type* when:
      1. No cross-coalition mergers occur (every next coalition is a subset of
         some current coalition — no players from different current coalitions
         end up together).
      2. The proposer was in a non-singleton coalition and becomes a singleton.

    The returned set always includes the proposer (if exit-type).  It also
    includes every co-member of the proposer's current coalition who also
    becomes a singleton, but only when that coalition had ≥ 3 members.  In a
    2-member coalition the remaining member is stranded with no agency and is
    therefore not included.

    This is a private helper used by is_unilateral_breakout() and by the
    effectivity rules in lib/effectivity.py.  It was introduced when the
    framework was generalised beyond n=3: the original is_unilateral_breakout()
    used hard-coded n=3 checks and string-based state comparison, both of
    which fail for multi-coalition states such as (CT)(FW) → (CT) and for
    non-canonical state-name orderings like '(CT)' vs '(TC)'.
    """
    current_sets = _coalition_structure_as_frozensets(current_state)
    next_sets    = _coalition_structure_as_frozensets(next_state)

    # Condition 1: no cross-coalition mergers.
    for ns in next_sets:
        if not any(ns.issubset(cs) for cs in current_sets):
            return set()   # merger detected → not exit-type

    # Condition 2: proposer must leave a non-singleton coalition.
    proposer_current = frozenset(get_player_coalition(proposer, current_state))
    if len(proposer_current) < 2:
        return set()   # proposer already singleton

    proposer_next = frozenset(get_player_coalition(proposer, next_state))
    if len(proposer_next) > 1:
        return set()   # proposer stays in a coalition → not exiting

    # Build the committee: proposer plus any co-members who also become singletons
    # (only meaningful when the coalition had ≥ 3 members).
    committee = {proposer}
    if len(proposer_current) >= 3:
        for member in proposer_current - {proposer}:
            if len(frozenset(get_player_coalition(member, next_state))) == 1:
                committee.add(member)

    return committee


def is_unilateral_breakout(proposer: str, current_state: str,
                           next_state: str, n_players: int) -> bool:
    """Return True iff the transition is a unilateral exit by the proposer alone.

    **Why this function was extended beyond the original n=3 logic**

    The original implementation used hard-coded checks covering only the two
    exit patterns that arise for n=3 (leaving the grand coalition, or leaving a
    2-player coalition into full singletons).  When the framework was
    generalised to n=4, new patterns appeared — e.g. a proposer leaving one
    coalition in a multi-coalition state such as (CT)(FW) → (CT) — and string
    comparison failed because state names are not canonical ('(CT)' vs '(TC)').

    The function now delegates to the private helper _exit_committee(), which
    uses frozensets for order-independent semantic comparison and covers the
    general case for any n.  A transition is unilateral when _exit_committee()
    returns a set containing only the proposer.

    For instance:
        'T' proposing '(WTC)' -> '(WC)' returns True  (n=3, T exits alone).
        'C' proposing '(CFTW)' -> '(FTW)' returns True  (n=4, C exits alone).
        'W' proposing '(CT)(FW)' -> '(CT)' returns True  (n=4, F stranded).
        'C' proposing '(CT)(FW)' -> '(FW)' returns True  (n=4, T stranded).
        'C' proposing '(CFTW)' -> '(TW)' returns False  (n=4, F also exits).
    """
    return _exit_committee(proposer, current_state, next_state) == {proposer}


def verify_proposals(players: List[str], states: List[str],
                     P_proposals: Dict[tuple, float],
                     P_approvals: Dict[tuple, float],
                     V: pd.DataFrame) -> Tuple[bool, str]:
    """Checks that the proposal strategies of all players constitute a
    valid equilibrium, as specified in Condition 1 in section A.5.

    Arguments:
        players: A list of all countries in the game.
        states: A list of all possible states in the system.
        P_proposals: A dictionary with keys determined by triplets
                     (i, x, y). Each value is the probability that player i,
                     IF chosen as proposer, suggests a move from the current
                     state x to a new state y.
        P_approvals: A dictionary with keys determined by triplets
                     (i, x, y). Each value is the probability that the
                     transition proposed by player i, to move from current
                     state x to a new state y, gets accepted by the
                     approval committee.
        V: A dataframe containing the long-run expected payoff for all
           players in all states.
    """

    for proposer in players:
        for current_state in states:

            # All next states for which the proposer attaches
            # a positive proposition probability while in current_state.
            pos_prob_next_states = []

            # Expectation of the proposition value:
            # E = p_accepted * V_next + p_rejected * V_current
            expected_values = {}

            for next_state in states:

                p_proposed = P_proposals[(proposer, current_state,
                                         next_state)]

                if p_proposed > 0.:
                    pos_prob_next_states.append(next_state)

                # Probability that the approval committee accepts.
                p_approved = P_approvals[(proposer, current_state,
                                         next_state)]
                p_rejected = 1 - p_approved

                V_current = V.loc[current_state, proposer]
                V_next = V.loc[next_state, proposer]
                expected_values[next_state] =\
                    p_approved * V_next + p_rejected * V_current

            # Next state(s) that give the highest possible expected
            # long-run payoff.
            argmaxes = [key for key, val in expected_values.items()
                        if np.isclose(val, max(expected_values.values()),
                        atol=1e-12)]

            try:
                # Any state with a positive proposal probability must be one
                # of the best alternatives.
                assert set(pos_prob_next_states).issubset(argmaxes)
            except AssertionError:
                error_msg = (
                         f"Proposal strategy error with player {proposer}! "
                         f"In state {current_state}, positive probability "
                         f"on state(s) {pos_prob_next_states}, but the argmax "
                         f"states are: {argmaxes}."
                         )
                return False, error_msg

    return True, "Test passed."


def verify_approvals(players: List[str], states: List[str],
                     effectivity: Dict[tuple, int], V: pd.DataFrame,
                     strategy_df: pd.DataFrame) -> Tuple[bool, str]:
    """Checks that the approval strategies of all players constitute a
    valid equilibrium, as specified in Condition 2 in section A.5.

    Arguments:
        players: A list of all countries in the game.
        states: A list of all possible states in the system.
        effectivity: The effectivity correspondence, from derive_effectivity().
        V: A dataframe containing the long-run expected payoff for all
           players in all states.
        strategy_df: A dataframe containing the strategies of all players.
    """

    for proposer in players:
        for current_state in states:
            for next_state in states:

                # Approval committee for this transition.
                approvers = get_approval_committee(
                    effectivity, players, proposer, current_state, next_state)

                for approver in approvers:
                    V_current = V.loc[current_state, approver]
                    V_next = V.loc[next_state, approver]
                    p_approve = strategy_df.loc[
                                    (current_state, 'Acceptance', approver),
                                    (f'Proposer {proposer}', next_state)]

                    if np.isclose(V_next, V_current, atol=1e-12):
                        passed = (0. <= p_approve <= 1.)
                    elif V_next > V_current:
                        passed = (p_approve == 1.)
                    elif V_next < V_current:
                        passed = (p_approve == 0.)
                    else:
                        msg = 'Unknown error during approval consistency check'
                        raise ValueError(msg)

                    if not passed:
                        error_msg = (
                            f"Approval strategy error with player {approver}! "
                            f"When player {proposer} proposes the transition "
                            f"{current_state} -> {next_state}, the values are "
                            f"V(current) = {V_current:.5f} "
                            f"and V(next) = {V_next:.5f}, "
                            f"but approval probability is {p_approve}."
                            )
                        return False, error_msg

    return True, "Test passed."


def verify_equilibrium(result: Dict[str, Any]):
    """Checks that the experiment results and strategy profiles are a
    valid equilibrium.

    Arguments:
        results: A dictionary from main.run_experiment().
    """

    proposals_ok = verify_proposals(players=result["players"],
                                    states=result["state_names"],
                                    P_proposals=result["P_proposals"],
                                    P_approvals=result["P_approvals"],
                                    V=result["V"])

    approvals_ok = verify_approvals(players=result["players"],
                                    states=result["state_names"],
                                    effectivity=result["effectivity"],
                                    V=result["V"],
                                    strategy_df=result["strategy_df"])

    if proposals_ok[0] and approvals_ok[0]:
        return True, "All tests passed."
    else:
        messages = [check[1] for check in [proposals_ok, approvals_ok]
                    if not check[0]]

        # Prepend V values once when either verification fails
        V = result["V"]
        full_message = f"The value functions V are:\n{V}\n\n" + '\n'.join(messages)

        return False, full_message


def write_latex_tables(result: Dict[str, Any], variables: List[str],
                       results_path: str = "./results",
                       float_format: str = "%.5f") -> None:
    """Writes experiment results as .tex tables.

    Arguments:
        results: A dictionary from main.run_experiment().
        variables: A list of items in results.keys() to store.
        results_path: Folder to store the .tex files in.
        float_format: How many digits to include in the .tex tables.
    """

    experiment = result['scenario_name']
    for variable in variables:

        path = f"{results_path}/{variable}_{experiment}.tex"
        raw_caption = f"{experiment}: {variable}"
        safe_caption = _escape_latex(raw_caption)
        result[variable].to_latex(buf=path, float_format=float_format,
                      caption=safe_caption)
