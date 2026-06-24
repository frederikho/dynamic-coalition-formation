"""
Game setup for Jere's MIP-VFI coalition formation solver.

A Game instance holds all parameters (players, states, payoffs) and provides
game-mechanic helpers. Solver functions take a Game as their first argument,
replacing the module-level globals in the original jeres_implementation.py.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

EPS_IND = 1e-12  # indifference threshold for numerical stability


# ---------------------------------------------------------------------------
# Partition combinatorics
# ---------------------------------------------------------------------------

def gen_partitions(players):
    """Yield all set partitions of `players` as tuples of frozensets."""
    players = list(players)
    if not players:
        yield ()
        return
    first, rest = players[0], players[1:]
    for p in gen_partitions(rest):
        yield (frozenset([first]),) + p
        for i, block in enumerate(p):
            new_p = list(p)
            new_p[i] = block | frozenset([first])
            yield tuple(new_p)


def canon_partition(p):
    """Canonical form: tuple of frozensets sorted by min element."""
    return tuple(sorted((frozenset(c) for c in p), key=lambda s: sorted(s)))


def all_partitions(n_players: int) -> list:
    """All canonical partitions of {0, …, n_players-1}, sorted by structure."""
    parts = set(canon_partition(p) for p in gen_partitions(range(n_players)))
    return sorted(parts, key=lambda p: (len(p), sorted(len(c) for c in p)))


# ---------------------------------------------------------------------------
# Game dataclass
# ---------------------------------------------------------------------------

@dataclass
class Game:
    """
    All game parameters for Jere's MIP-VFI solver.

    Attributes
    ----------
    players   : list of player names (strings)
    payoffs   : ndarray of shape (n_states, n_players), Jere state order
    states    : list of canonical partitions (tuples of frozensets)
    state_idx : {partition: index} lookup dict
    n_players : int
    n_states  : int
    """
    players: list
    payoffs: np.ndarray
    states: list
    state_idx: dict
    n_players: int
    n_states: int

    @classmethod
    def from_payoffs(cls, players: list, payoffs_np: np.ndarray) -> "Game":
        """
        Build a Game from a player list and a payoffs array.

        Parameters
        ----------
        players    : list of player names, length n
        payoffs_np : (n_states, n) ndarray indexed in Jere state order
                     (partitions sorted by structure — all singletons first)
        """
        n = len(players)
        states = all_partitions(n)
        state_idx = {s: i for i, s in enumerate(states)}
        return cls(
            players=list(players),
            payoffs=np.asarray(payoffs_np, dtype=float),
            states=states,
            state_idx=state_idx,
            n_players=n,
            n_states=len(states),
        )


# ---------------------------------------------------------------------------
# Game mechanics
# ---------------------------------------------------------------------------

def _player_block(state, player: int) -> frozenset:
    for c in state:
        if player in c:
            return c
    raise ValueError(f"Player {player} not in any block of state {state}")


def changed_players(game: Game, state, next_state) -> frozenset:
    """Players whose coalition block changes between state and next_state."""
    return frozenset(
        p for p in range(game.n_players)
        if _player_block(state, p) != _player_block(next_state, p)
    )


def apply_proposal(state, proposer: int) -> tuple:
    """Result of proposer splitting off as a singleton."""
    proposed = frozenset([proposer])
    new_parts = [frozenset(b - proposed) for b in state if b - proposed]
    new_parts.append(proposed)
    return canon_partition(new_parts)


def is_unilateral_exit(game: Game, state, next_state, proposer: int) -> bool:
    """True iff next_state results from proposer leaving their coalition alone."""
    coal = _player_block(state, proposer)
    if len(coal) == 1:
        return False
    exit_result = apply_proposal(state, proposer)
    return next_state == exit_result and exit_result in game.state_idx


def voters(game: Game, state, next_state, proposer: int) -> frozenset:
    """Approval committee: changed players minus the proposer, or empty for unilateral exit."""
    if is_unilateral_exit(game, state, next_state, proposer):
        return frozenset()
    return changed_players(game, state, next_state) - {proposer}


# ---------------------------------------------------------------------------
# Framework state-name → Jere partition
# ---------------------------------------------------------------------------

def fw_state_name_to_partition(name: str, players: list) -> tuple:
    """
    Convert a framework state name (e.g. '( )', '(CHNUSA)', '(CHN)(RUSUSA)')
    to a canonical Jere partition of player indices.
    """
    import re
    player_idx = {p: i for i, p in enumerate(players)}
    n = len(players)

    if name == "( )":
        return canon_partition([frozenset({i}) for i in range(n)])

    groups = re.findall(r'\(([^)]+)\)', name)
    used: set = set()
    partitions = []
    for group in groups:
        remaining = group
        members: set = set()
        while remaining:
            matched = False
            for p in sorted(players, key=len, reverse=True):
                if remaining.startswith(p):
                    members.add(player_idx[p])
                    remaining = remaining[len(p):]
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Cannot parse '{remaining}' in state '{name}'")
        partitions.append(frozenset(members))
        used.update(members)

    for i in range(n):
        if i not in used:
            partitions.append(frozenset({i}))

    return canon_partition(partitions)
