from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class MDP:
    """
    General Markov Decision Process placeholder.

    Arguments:
        n_states: Number of possible states in the system.
        transition_probs: n_states * n_states matrix of probabilities.
        discounting: Discounting (and farsightedness) parameter.

    Implementation of this class follows closely the structure in:
    http://aima.cs.berkeley.edu/python/mdp.html and
    https://github.com/akaAlbo/deeprlbootcamp/tree/master/lab1
    """
    def __init__(self,
                 n_states: int,
                 transition_probs: pd.DataFrame,
                 discounting: float):

        self.n_states = n_states
        self.transition_probs = transition_probs
        self.discounting = discounting

    def solve_value_func(self, payoffs: np.ndarray) -> np.ndarray:
        """ Solve the linear system of value functions
        for an individual player.

        Arguments:
            payoffs: A vector of payoffs size n_states for a single country.
        """

        A = np.zeros((self.n_states, self.n_states))
        b = np.zeros(self.n_states)
        P = self.transition_probs

        for state in range(self.n_states):
            for next_state, prob in enumerate(P.iloc[state, :]):
                A[state][next_state] = self.discounting * prob

        A -= np.eye(self.n_states)
        b = -(1-self.discounting) * payoffs
        x = np.linalg.solve(A, b)

        assert np.allclose(np.dot(A, x), b)

        return x


def _as_array(P) -> np.ndarray:
    """Coerce a transition matrix to a float array and check it is stochastic."""
    P_array = np.asarray(P.values if isinstance(P, pd.DataFrame) else P, dtype=np.float64)
    if P_array.ndim != 2 or P_array.shape[0] != P_array.shape[1]:
        raise ValueError(f"Transition matrix must be square, got shape {P_array.shape}.")
    row_sums = P_array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        bad = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=1e-8))
        raise ValueError(
            f"Transition matrix rows must sum to 1; rows {bad.tolist()} sum to "
            f"{row_sums[bad].tolist()}."
        )
    return P_array


def absorbing_sets(P, tol: float = 1e-10) -> List[List[int]]:
    """Find the closed communicating classes of a Markov chain.

    A closed class is a strongly connected component of the support graph with no
    outgoing edge of probability above ``tol``. These are the sets the chain ends
    up in: singletons are ordinary absorbing states, larger sets are cycles the
    chain never escapes.

    Arguments:
        P: n x n transition matrix (DataFrame or array).
        tol: Probabilities at or below this are treated as absent edges.

    Returns:
        A list of closed classes, each a sorted list of state indices. The classes
        themselves are ordered by their smallest member.
    """
    P_array = _as_array(P)
    n = P_array.shape[0]

    support = (P_array > tol).astype(np.int8)
    n_components, labels = connected_components(
        csr_matrix(support), directed=True, connection="strong"
    )

    closed = []
    for component in range(n_components):
        members = np.flatnonzero(labels == component)
        # Closed iff no member puts mass on a state outside the component.
        outside = np.ones(n, dtype=bool)
        outside[members] = False
        if support[np.ix_(members, outside)].any():
            continue
        closed.append(sorted(int(i) for i in members))

    return sorted(closed, key=lambda members: members[0])


def _class_stationary(P_sub: np.ndarray) -> np.ndarray:
    """Stationary distribution within a closed communicating class.

    ``P_sub`` is irreducible and stochastic by construction, so pi P = pi together
    with the normalisation sum(pi) = 1 has a unique solution. For a periodic class
    this is the Cesaro limit rather than a pointwise limit, which is what the
    long-run occupancy share means for a cycling equilibrium.
    """
    size = P_sub.shape[0]
    if size == 1:
        return np.ones(1)

    A = P_sub.T - np.eye(size)
    A[-1, :] = 1.0
    b = np.zeros(size)
    b[-1] = 1.0

    pi = np.linalg.solve(A, b)
    if np.any(pi < -1e-9):
        raise ValueError(f"Stationary distribution has negative entries: {pi.tolist()}")
    return np.clip(pi, 0.0, None) / pi.sum()


def limiting_distribution(P, initial: Optional[np.ndarray] = None,
                          tol: float = 1e-10) -> np.ndarray:
    """Long-run occupancy distribution of a Markov chain.

    Mass starting in a transient state is routed to the closed classes by their
    absorption probabilities, then spread within each class by that class's own
    stationary distribution. With several closed classes the answer depends on
    where the chain starts, so ``initial`` is an explicit argument rather than an
    assumption baked into the result.

    Arguments:
        P: n x n transition matrix (DataFrame or array).
        initial: Initial distribution over states. Defaults to uniform.
        tol: Edge-support threshold passed to :func:`absorbing_sets`.

    Returns:
        Length-n probability vector, supported on the closed classes.
    """
    P_array = _as_array(P)
    n = P_array.shape[0]

    if initial is None:
        initial = np.full(n, 1.0 / n)
    else:
        initial = np.asarray(initial, dtype=np.float64)
        if initial.shape != (n,):
            raise ValueError(f"initial must have shape ({n},), got {initial.shape}.")
        if np.any(initial < 0) or not np.isclose(initial.sum(), 1.0):
            raise ValueError("initial must be a probability distribution over states.")

    classes = absorbing_sets(P_array, tol=tol)
    if not classes:
        raise ValueError(
            "Transition matrix has no closed communicating class; this cannot "
            "happen for a finite stochastic matrix and indicates a malformed P."
        )

    absorbing_idx = [i for members in classes for i in members]
    transient_idx = [i for i in range(n) if i not in set(absorbing_idx)]

    pi = np.zeros(n)

    # Mass that starts inside a closed class stays in it.
    for members in classes:
        share = initial[members].sum()
        if share > 0:
            pi[members] += share * _class_stationary(P_array[np.ix_(members, members)])

    if transient_idx:
        Q = P_array[np.ix_(transient_idx, transient_idx)]
        fundamental = np.linalg.inv(np.eye(len(transient_idx)) - Q)
        transient_mass = initial[transient_idx]
        for members in classes:
            R = P_array[np.ix_(transient_idx, members)]
            # Probability of ending in this class, per transient starting state.
            absorption = (fundamental @ R).sum(axis=1)
            share = float(transient_mass @ absorption)
            if share > 0:
                pi[members] += share * _class_stationary(P_array[np.ix_(members, members)])

    total = pi.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Limiting distribution sums to {total}, expected 1.")
    return pi / total
