"""
Equilibrium solver for coalition formation games using smoothed fixed-point iteration.

This module implements an algorithm to find equilibrium strategy profiles
from random initialization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import pickle
import time
from lib.probabilities_optimized import TransitionProbabilitiesOptimized
from lib.mdp import MDP
from lib.utils import derive_effectivity, get_approval_committee, verify_equilibrium
from lib.equilibrium.cycle_analysis import (
    detect_cycle, format_cycle_report,
    detect_partial_cycle, format_partial_cycle_report,
)
import copy


def sigmoid(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Smoothed sigmoid function for acceptance probabilities.

    Args:
        x: Value difference (V_next - V_current)
        temperature: Smoothing parameter (tau_r in algorithm)

    Returns:
        Probability in [0, 1]
    """
    z = x / temperature
    # Clip to avoid overflow in exp for extreme z
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax function for proposal probabilities.

    Args:
        values: Expected values for each option
        temperature: Smoothing parameter (tau_p in algorithm)

    Returns:
        Probability distribution over options
    """
    # Numerical stability: subtract max before exp
    exp_values = np.exp((values - np.max(values)) / temperature)
    return exp_values / np.sum(exp_values)


def format_strategy_df_compact(df: pd.DataFrame, players: List[str], states: List[str]) -> str:
    """Return a compact string representation of a strategy DataFrame.

    Each player name is shortened to its first letter (e.g. CHN->C, NDE->N, USA->U).
    States are ordered: ( ) first, then by coalition size, then grand coalition.
    Within each state rows are ordered: Prop, then Acc for each player.
    """
    abbrev = {p: p[0] for p in players}
    abbrev_players = [abbrev[p] for p in players]

    def shorten(s: str) -> str:
        for full, short in abbrev.items():
            s = s.replace(full, short)
        return s

    def state_size(state_name: str) -> int:
        return len(state_name.strip().lstrip('(').rstrip(')').replace(' ', ''))

    compact_df = df.copy()

    proposer_order = {f'P:{a}': i for i, a in enumerate(abbrev_players)}
    state_size_map = {shorten(s): state_size(s) for s in states}
    new_cols = [(shorten(c0).replace('Proposer ', 'P:'), shorten(c1)) for c0, c1 in compact_df.columns]
    sorted_cols = sorted(
        new_cols,
        key=lambda t: (proposer_order.get(t[0], 999), state_size_map.get(t[1], 999))
    )
    compact_df.columns = pd.MultiIndex.from_tuples(new_cols, names=compact_df.columns.names)
    compact_df = compact_df.reindex(
        columns=pd.MultiIndex.from_tuples(sorted_cols, names=compact_df.columns.names)
    )

    def shorten_row(t):
        state, kind, player = t
        return (
            shorten(state),
            'Prop' if kind == 'Proposition' else 'Acc',
            shorten(player) if isinstance(player, str) else '',
        )

    compact_df.index = pd.MultiIndex.from_tuples(
        [shorten_row(t) for t in compact_df.index], names=compact_df.index.names
    )

    sorted_states = sorted(states, key=state_size)
    desired_rows = []
    for state in sorted_states:
        s = shorten(state)
        desired_rows.append((s, 'Prop', ''))
        for p in abbrev_players:
            desired_rows.append((s, 'Acc', p))

    existing = set(compact_df.index.tolist())
    desired_rows = [r for r in desired_rows if r in existing]
    compact_df = compact_df.reindex(
        pd.MultiIndex.from_tuples(desired_rows, names=compact_df.index.names)
    )

    return compact_df.to_string(float_format=lambda x: f'{x:.2f}')


class EquilibriumSolver:
    """Finds equilibrium strategy profiles via smoothed fixed-point iteration."""

    def __init__(self,
                 players: List[str],
                 states: List[str],
                 effectivity: Dict[tuple, int],
                 protocol: Dict[str, float],
                 payoffs: pd.DataFrame,
                 discounting: float,
                 unanimity_required: bool,
                 verbose: bool = True,
                 random_seed: Optional[int] = None,
                 logger=None):
        """
        Args:
            players: List of player names
            states: List of state names
            effectivity: Effectivity correspondence
            protocol: Probability distribution over proposers
            payoffs: Static payoffs for each player in each state
            discounting: Discount factor (delta)
            unanimity_required: Whether unanimous approval is required
            verbose: Whether to print progress
            random_seed: Random seed for initialization (if None, generates one)
            logger: Logger instance (if None, uses print statements)
        """
        self.players = players
        self.states = states
        self.effectivity = effectivity
        self.protocol = protocol
        self.payoffs = payoffs
        self.discounting = discounting
        self.unanimity_required = unanimity_required
        self.verbose = verbose
        self.logger = logger

        # Set random seed (generate if not provided)
        if random_seed is None:
            random_seed = np.random.randint(0, 2**31)
        self.random_seed = random_seed

        # Initialize strategy tables
        self.p_proposals = {}  # Proposal probabilities
        self.r_acceptances = {}  # Acceptance probabilities
        self._initialize_strategies()

        # Create TransitionProbabilities object once for fast updates
        # (will be initialized with current strategy dicts in first use)
        self.tp = None  # Lazy initialization on first solve() call
    def _log(self, message, level='info'):
        """Log message using logger if available, otherwise print."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)

    def _initialize_strategies(self):
        """Initialize proposal and acceptance probabilities randomly with uniform distribution."""
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.random_seed)

        # Initialize proposal probabilities: random values that sum to 1 for each (proposer, current_state)
        for proposer in self.players:
            for current_state in self.states:
                # Generate random values and normalize to sum to 1
                random_values = rng.uniform(0, 1, len(self.states))
                normalized = random_values / random_values.sum()

                for idx, next_state in enumerate(self.states):
                    key = (proposer, current_state, next_state)
                    self.p_proposals[key] = normalized[idx]

        # Initialize acceptance probabilities: random uniform [0,1] for all committee members
        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:
                    committee = get_approval_committee(
                        self.effectivity, self.players,
                        proposer, current_state, next_state
                    )
                    for approver in committee:
                        key = (proposer, current_state, next_state, approver)
                        self.r_acceptances[key] = rng.uniform(0, 1)

    def _create_strategy_dataframe(self) -> pd.DataFrame:
        """Create a strategy DataFrame from current proposal and acceptance probabilities."""
        # Create multi-index columns: (Proposer X, state)
        columns = pd.MultiIndex.from_product(
            [[f'Proposer {p}' for p in self.players], self.states],
            names=['Proposer', 'Next State']
        )

        # Create multi-index rows: (state, Proposition/Acceptance, player/nan)
        rows = []
        for state in self.states:
            rows.append((state, 'Proposition', np.nan))
            for player in self.players:
                rows.append((state, 'Acceptance', player))
        index = pd.MultiIndex.from_tuples(
            rows,
            names=['Current State', 'Type', 'Player']
        )

        # Create DataFrame (initialize with NaN)
        df = pd.DataFrame(np.nan, index=index, columns=columns)

        # Sort both index and columns to improve performance and avoid warnings
        df = df.sort_index(axis=0).sort_index(axis=1)

        # Fill in proposal probabilities
        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:
                    key = (proposer, current_state, next_state)
                    df.loc[(current_state, 'Proposition', np.nan),
                           (f'Proposer {proposer}', next_state)] = self.p_proposals[key]

        # Fill in acceptance probabilities (only for committee members)
        # Non-committee cells remain as NaN
        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:
                    committee = get_approval_committee(
                        self.effectivity, self.players,
                        proposer, current_state, next_state
                    )
                    for approver in committee:
                        key = (proposer, current_state, next_state, approver)
                        df.loc[(current_state, 'Acceptance', approver),
                               (f'Proposer {proposer}', next_state)] = self.r_acceptances[key]

        return df

    def _compact_strategy_str(self) -> str:
        """Return the strategy DataFrame as a compact string with abbreviated labels."""
        return format_strategy_df_compact(
            self._create_strategy_dataframe(),
            players=self.players,
            states=self.states,
        )

    def _compute_transition_probabilities_fast(self) -> Tuple:
        """Compute transition probabilities from strategy dicts (fast path).

        Uses pre-created TP object and updates arrays directly from dicts,
        bypassing DataFrame construction for maximum speed.
        """
        # Lazy initialization of TP object on first call
        if self.tp is None:
            # Create initial DataFrame for first-time setup
            initial_df = self._create_strategy_dataframe()
            initial_df.fillna(0., inplace=True)

            self.tp = TransitionProbabilitiesOptimized(
                df=initial_df,
                effectivity=self.effectivity,
                players=self.players,
                states=self.states,
                protocol=self.protocol,
                unanimity_required=self.unanimity_required
            )

        # Fast update: dict -> arrays directly (no DataFrame construction)
        self.tp.update_from_dicts(self.p_proposals, self.r_acceptances)

        # Compute probabilities from updated arrays
        return self.tp.get_probabilities()

    def _compute_transition_probabilities(self, strategy_df: pd.DataFrame) -> Tuple:
        """Compute transition probabilities from strategy DataFrame.

        This is the slower path used for verification and final output.
        For inner loop, use _compute_transition_probabilities_fast() instead.
        """
        # Fill NaN values with 0.0 (NaN values indicate non-committee members or zero probabilities)
        strategy_df_filled = strategy_df.copy()
        strategy_df_filled.fillna(0., inplace=True)

        # Check for invalid probability values
        if (strategy_df_filled < -1e-10).any().any() or (strategy_df_filled > 1.0 + 1e-10).any().any():
            self._log("WARNING: Strategy DataFrame contains values outside [0,1]:", level='warning')
            mask = (strategy_df_filled < -1e-10) | (strategy_df_filled > 1.0 + 1e-10)
            if mask.any().any():
                self._log(str(strategy_df_filled[mask].stack()), level='warning')

        # Use optimized version (113x faster computation, but limited by DataFrame conversion)
        TPClass = TransitionProbabilitiesOptimized
        tp = TPClass(
            df=strategy_df_filled,
            effectivity=self.effectivity,
            players=self.players,
            states=self.states,
            protocol=self.protocol,
            unanimity_required=self.unanimity_required
        )
        return tp.get_probabilities()

    def _solve_value_functions(self, P: pd.DataFrame) -> pd.DataFrame:
        """Solve for value functions given transition probabilities."""
        mdp = MDP(
            n_states=len(self.states),
            transition_probs=P,
            discounting=self.discounting
        )

        V = pd.DataFrame(index=self.states, columns=self.players)
        for player in self.players:
            V.loc[:, player] = mdp.solve_value_func(
                self.payoffs.loc[:, player].values
            )

        return V

    def _update_acceptances(self, V: pd.DataFrame, tau_r: float,
                            old_acceptances: Optional[Dict] = None,
                            players_subset: Optional[List] = None,
                            keys_subset: Optional[set] = None) -> Dict:
        """Update acceptance probabilities using smoothed sigmoid.

        Args:
            V: Value functions
            tau_r: Smoothing temperature for acceptances
            old_acceptances: Current acceptance dict; required when any subset is given
                             so non-selected entries keep their existing values.
            players_subset: If given (and keys_subset is None), only recompute for
                            approvers in this set.
            keys_subset: If given, only recompute these exact (proposer, current_state,
                         next_state, approver) keys. Takes precedence over players_subset.

        Returns:
            Updated acceptance probabilities
        """
        new_acceptances = {}

        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:
                    committee = get_approval_committee(
                        self.effectivity, self.players,
                        proposer, current_state, next_state
                    )

                    for approver in committee:
                        key = (proposer, current_state, next_state, approver)

                        # Determine whether to recompute or carry forward old value
                        if keys_subset is not None:
                            skip = key not in keys_subset
                        elif players_subset is not None:
                            skip = approver not in players_subset
                        else:
                            skip = False

                        if skip:
                            new_acceptances[key] = old_acceptances[key]
                            continue

                        V_current = V.loc[current_state, approver]
                        V_next = V.loc[next_state, approver]

                        # Smoothed acceptance: sigmoid((V_next - V_current) / tau_r)
                        new_acceptances[key] = sigmoid(
                            V_next - V_current, tau_r
                        )

        return new_acceptances

    def _compute_approval_probability(self, proposer: str, current_state: str,
                                     next_state: str, P_approvals: Dict) -> float:
        """Compute approval probability for a transition."""
        key = (proposer, current_state, next_state)
        return P_approvals.get(key, 0.0)

    def _update_proposals(self, V: pd.DataFrame, P_approvals: Dict,
                         tau_p: float,
                         old_proposals: Optional[Dict] = None,
                         players_subset: Optional[List] = None) -> Dict:
        """Update proposal probabilities using smoothed softmax.

        Args:
            V: Value functions
            P_approvals: Approval probabilities
            tau_p: Smoothing temperature for proposals
            old_proposals: Current proposal dict; required when players_subset is given
                           so non-selected proposers keep their existing values.
            players_subset: If given, only recompute for proposers in this set.

        Returns:
            Updated proposal probabilities
        """
        new_proposals = {}

        for proposer in self.players:
            for current_state in self.states:
                if players_subset is not None and proposer not in players_subset:
                    for next_state in self.states:
                        key = (proposer, current_state, next_state)
                        new_proposals[key] = old_proposals[key]
                    continue

                # Compute expected values for each next state
                expected_values = np.zeros(len(self.states))

                for i, next_state in enumerate(self.states):
                    p_approved = self._compute_approval_probability(
                        proposer, current_state, next_state, P_approvals
                    )
                    p_rejected = 1.0 - p_approved

                    V_current = V.loc[current_state, proposer]
                    V_next = V.loc[next_state, proposer]

                    # Expected value: p_approved * V_next + p_rejected * V_current
                    expected_values[i] = p_approved * V_next + p_rejected * V_current

                # Softmax over expected values
                probs = softmax(expected_values, tau_p)

                for i, next_state in enumerate(self.states):
                    key = (proposer, current_state, next_state)
                    new_proposals[key] = probs[i]

        return new_proposals

    def _apply_damping(self, old_dict: Dict, new_dict: Dict, damping: float) -> Dict:
        """
        Apply damping to strategy updates.

        Convention:
        - damping = 1.0 -> full damping (no change; keep old)
        - damping = 0.0 -> no damping (full update; take new)
        - intermediate -> convex combination

        Args:
            old_dict: Old strategy dictionary
            new_dict: New strategy dictionary
            damping: Damping factor in [0, 1]

        Returns:
            Damped strategy dictionary
        """
        if not (0.0 <= damping <= 1.0):
            raise ValueError(f"damping must be in [0, 1], got {damping}")

        damped = {}
        for key, old_val in old_dict.items():
            new_val = new_dict[key]
            damped[key] = damping * old_val + (1.0 - damping) * new_val
        return damped

    def _verify_candidate_equilibrium(self, strategy_df: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
        """Verify if a strategy profile is a valid equilibrium.

        Args:
            strategy_df: Strategy DataFrame to verify

        Returns:
            (success, message, V): Verification result, message, and value functions
        """
        # Compute transition probabilities
        tp = TransitionProbabilitiesOptimized(
            df=strategy_df,
            effectivity=self.effectivity,
            players=self.players,
            states=self.states,
            protocol=self.protocol,
            unanimity_required=self.unanimity_required
        )
        P, P_proposals, P_approvals = tp.get_probabilities()

        # Solve value functions
        V = self._solve_value_functions(P)

        # Build result structure for verification
        result = {
            'V': V,
            'P': P,
            'P_proposals': P_proposals,
            'P_approvals': P_approvals,
            'players': self.players,
            'state_names': self.states,
            'effectivity': self.effectivity,
            'strategy_df': strategy_df,
        }

        # Verify equilibrium
        success, message = verify_equilibrium(result)

        return success, message, V

    def _polish_exact_equilibrium(self, max_iter: int = 20, tol: float = 1e-9) -> Tuple[bool, str, int]:
        """Iteratively enforce exact best responses after annealing.

        One-shot projection can be internally inconsistent because projecting
        strategies changes transitions, which changes V. This loop alternates:
        solve V -> project exact -> verify, until verified, stalled, or cycling.

        When a cycle is detected, additionally tries the element-wise average of
        the cycling strategies as a candidate mixed equilibrium.
        """
        proposal_keys = list(self.p_proposals.keys())
        acceptance_keys = list(self.r_acceptances.keys())

        # Map signature -> index into seen_arrays (for cycle-period detection)
        seen_signatures = {}
        seen_arrays = []   # (p_arr, r_arr) after each projection, for averaging
        prev_p = None
        prev_r = None
        last_message = "Verification not run."

        for it in range(1, max_iter + 1):
            strategy_df = self._create_strategy_dataframe()
            P, _, _ = self._compute_transition_probabilities(strategy_df)
            V = self._solve_value_functions(P)

            # Capture pre-projection arrays for change reporting
            p_before = np.fromiter((self.p_proposals[k] for k in proposal_keys), dtype=np.float64)
            r_before = np.fromiter((self.r_acceptances[k] for k in acceptance_keys), dtype=np.float64)

            self._project_to_exact_equilibrium(V)
            projected_strategy_df = self._create_strategy_dataframe()
            success, message, _ = self._verify_candidate_equilibrium(projected_strategy_df)
            last_message = message

            p_arr = np.fromiter((self.p_proposals[k] for k in proposal_keys), dtype=np.float64)
            r_arr = np.fromiter((self.r_acceptances[k] for k in acceptance_keys), dtype=np.float64)
            signature = (
                np.round(p_arr, 12).tobytes(),
                np.round(r_arr, 12).tobytes(),
            )

            if self.verbose:
                self._log(f"\n  --- Polish iteration {it} ---")
                self._log(f"  V:\n{V.round(6).to_string()}")
                # Report what projection changed
                acc_flipped = int(np.sum(np.abs(r_arr - r_before) > 0.5))
                acc_0to1 = int(np.sum((r_before < 0.5) & (r_arr > 0.5)))
                acc_1to0 = int(np.sum((r_before > 0.5) & (r_arr < 0.5)))
                prop_changed = int(np.sum(np.abs(p_arr - p_before) > 1e-9))
                self._log(
                    f"  Projection: {acc_flipped} acceptance(s) flipped "
                    f"({acc_0to1} reject→approve, {acc_1to0} approve→reject); "
                    f"{prop_changed} proposal weight(s) changed"
                )
                # Show specific acceptance flips
                for i, k in enumerate(acceptance_keys):
                    if abs(r_arr[i] - r_before[i]) > 0.5:
                        proposer, cur, nxt, approver = k
                        self._log(
                            f"    acc flip: {approver} in {cur}, "
                            f"{proposer} proposes {nxt}: "
                            f"{r_before[i]:.2f} → {r_arr[i]:.2f} "
                            f"(V_cur={V.loc[cur, approver]:.6f}, V_nxt={V.loc[nxt, approver]:.6f})"
                        )
                # Show proposal argmax changes vs prev iteration
                if prev_p is not None:
                    for i, k in enumerate(proposal_keys):
                        proposer, cur, nxt = k
                        if abs(p_arr[i] - prev_p[i]) > 1e-9:
                            self._log(
                                f"    prop change: {proposer} in {cur} → {nxt}: "
                                f"{prev_p[i]:.4f} → {p_arr[i]:.4f}"
                            )
                self._log(f"  Verification: {'✓' if success else '✗'} {message.splitlines()[0] if not success else ''}")

            if success:
                return True, message, it

            if signature in seen_signatures:
                cycle_start_idx = seen_signatures[signature]
                cycle_period = len(seen_arrays) - cycle_start_idx
                cycling_p = [a[0] for a in seen_arrays[cycle_start_idx:]]
                cycling_r = [a[1] for a in seen_arrays[cycle_start_idx:]]
                avg_p = np.mean(cycling_p, axis=0)
                avg_r = np.mean(cycling_r, axis=0)

                # Apply the average strategy
                for i, k in enumerate(proposal_keys):
                    self.p_proposals[k] = avg_p[i]
                for i, k in enumerate(acceptance_keys):
                    self.r_acceptances[k] = avg_r[i]

                avg_strategy_df = self._create_strategy_dataframe()
                avg_success, avg_message, _ = self._verify_candidate_equilibrium(avg_strategy_df)

                if self.verbose:
                    self._log(
                        f"\n  ↺ Cycle of period {cycle_period} detected "
                        f"(iter {cycle_start_idx + 1}–{it}); trying average strategy..."
                    )
                    if avg_success:
                        self._log("  ✓ Average strategy verified as equilibrium.")
                    else:
                        self._log(f"  ✗ Average strategy did not verify: "
                                  f"{avg_message.splitlines()[0]}")

                if avg_success:
                    return True, avg_message, it

                return False, f"{message}\n\nPolishing stopped: detected a repeated exact-strategy state.", it

            seen_signatures[signature] = len(seen_arrays)
            seen_arrays.append((p_arr.copy(), r_arr.copy()))

            if prev_p is not None and prev_r is not None:
                if np.allclose(p_arr, prev_p, rtol=0, atol=tol) and np.allclose(r_arr, prev_r, rtol=0, atol=tol):
                    return False, f"{message}\n\nPolishing stopped: exact strategies no longer change.", it

            prev_p = p_arr
            prev_r = r_arr

        return False, f"{last_message}\n\nPolishing stopped: reached max_iter={max_iter}.", max_iter

    def _project_to_exact_equilibrium(self, V: pd.DataFrame):
        """Project strategies to exact equilibrium conditions.

        This enforces strict acceptance rules and best-response proposals.
        """
        # DEBUG: Track if we're setting strategies for split coalition states
        split_states = [s for s in self.states if s.count('(') > 1]

        # Project acceptances: 1 if V_next > V_current, 0 if V_next < V_current
        for proposer in self.players:
            for current_state in self.states:
                for next_state in self.states:
                    committee = get_approval_committee(
                        self.effectivity, self.players,
                        proposer, current_state, next_state
                    )

                    for approver in committee:
                        V_current = V.loc[current_state, approver]
                        V_next = V.loc[next_state, approver]

                        key = (proposer, current_state, next_state, approver)

                        if np.isclose(V_next, V_current, rtol=0, atol=1e-9):
                            # Indifferent: keep current value (could be anything in [0,1])
                            pass
                        elif V_next > V_current:
                            self.r_acceptances[key] = 1.0
                        else:  # V_next < V_current
                            self.r_acceptances[key] = 0.0

        # Compute final transition probabilities
        strategy_df = self._create_strategy_dataframe()
        P, P_proposals, P_approvals = self._compute_transition_probabilities(strategy_df)

        # Project proposals: only propose states that maximize expected value
        for proposer in self.players:
            for current_state in self.states:
                # Compute expected values
                expected_values = {}
                for next_state in self.states:
                    p_approved = P_approvals[(proposer, current_state, next_state)]
                    p_rejected = 1.0 - p_approved

                    V_current = V.loc[current_state, proposer]
                    V_next = V.loc[next_state, proposer]

                    expected_values[next_state] = (
                        p_approved * V_next + p_rejected * V_current
                    )

                # Find argmax states
                max_value = max(expected_values.values())
                argmax_states = [
                    state for state, val in expected_values.items()
                    if np.isclose(val, max_value, atol=1e-9)
                ]

                # Zero out non-argmax states and renormalize the existing weights
                # within the argmax set.  Any mixture over argmax states is a valid
                # exact best response; preserving the smoothed weights minimises the
                # perturbation to V and helps the polish loop find a self-consistent
                # fixed point.  Uniform (1/N) is used only as a fallback when the
                # total weight on argmax states is numerically zero.
                argmax_set = set(argmax_states)
                total_weight = sum(
                    self.p_proposals[(proposer, current_state, ns)]
                    for ns in argmax_set
                )
                for next_state in self.states:
                    key = (proposer, current_state, next_state)
                    if next_state in argmax_set:
                        if total_weight > 0:
                            self.p_proposals[key] = self.p_proposals[key] / total_weight
                        else:
                            self.p_proposals[key] = 1.0 / len(argmax_set)
                    else:
                        self.p_proposals[key] = 0.0

    def _save_checkpoint(self, checkpoint_path: str, outer_iter: int, tau_p: float, tau_r: float, config_hash: str):
        """Save current solver state to checkpoint file."""
        checkpoint = {
            'outer_iter': outer_iter,
            'tau_p': tau_p,
            'tau_r': tau_r,
            'p_proposals': self.p_proposals.copy(),
            'r_acceptances': self.r_acceptances.copy(),
            'config_hash': config_hash,
            'random_seed': self.random_seed,
            'timestamp': datetime.now().isoformat(),
            'states': self.states,
            'players': self.players,
        }

        # Ensure directory exists
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        if self.verbose:
            self._log(f"  Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict]:
        """Load solver state from checkpoint file."""
        if not Path(checkpoint_path).exists():
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            # Validate states and players match current solver configuration.
            # A mismatch means the checkpoint was created with different state
            # names (e.g. after a code change to coalition naming), making the
            # stored r_acceptances keys incompatible with the current effectivity.
            if 'states' in checkpoint and checkpoint['states'] != self.states:
                if self.verbose:
                    self._log(
                        f"Warning: Checkpoint states {checkpoint['states']} do not match "
                        f"current states {self.states}. Ignoring checkpoint.",
                        level='warning'
                    )
                return None
            if 'players' in checkpoint and checkpoint['players'] != self.players:
                if self.verbose:
                    self._log(
                        f"Warning: Checkpoint players {checkpoint['players']} do not match "
                        f"current players {self.players}. Ignoring checkpoint.",
                        level='warning'
                    )
                return None

            if self.verbose:
                self._log(f"Loaded checkpoint from iteration {checkpoint['outer_iter']}")
                self._log(f"  Checkpoint created: {checkpoint['timestamp']}")
                self._log(f"  Resuming from tau_p={checkpoint['tau_p']:.4f}, tau_r={checkpoint['tau_r']:.4f}")

            return checkpoint
        except Exception as e:
            if self.verbose:
                self._log(f"Warning: Could not load checkpoint: {e}", level='warning')
            return None

    def _print_timing_report(self, timing_stats: Dict):
        """Print timing analysis to log."""
        self._log("\n" + "="*80)
        self._log("TIMING ANALYSIS")
        self._log("="*80)

        inner_ops = ['create_df', 'compute_transitions', 'solve_values',
                     'update_acceptances', 'update_proposals', 'damping']
        outer_ops = ['inner_total', 'checkpoint_save', 'cycle_analysis', 'early_verification', 'outer_total']

        self._log("Inner loop operations (per iteration):")
        for op_name in inner_ops:
            timings = timing_stats.get(op_name, [])
            if timings:
                total_time = sum(timings)
                avg_time = total_time / len(timings)
                self._log(f"  {op_name:23s}: {total_time:8.3f}s total, {avg_time*1000:7.3f}ms avg ({len(timings)} calls)")

        self._log("\nOuter loop operations (per outer iteration):")
        for op_name in outer_ops:
            timings = timing_stats.get(op_name, [])
            if timings:
                total_time = sum(timings)
                avg_time = total_time / len(timings)
                self._log(f"  {op_name:23s}: {total_time:8.3f}s total, {avg_time*1000:7.3f}ms avg ({len(timings)} calls)")

        total_outer = sum(timing_stats.get('outer_total', []))
        total_inner = sum(timing_stats.get('inner_total', []))
        total_checkpoint = sum(timing_stats.get('checkpoint_save', []))
        total_cycle = sum(timing_stats.get('cycle_analysis', []))
        total_verification = sum(timing_stats.get('early_verification', []))
        overhead = total_outer - total_inner - total_checkpoint - total_cycle - total_verification

        self._log("\nBreakdown:")
        if total_outer > 0:
            self._log(f"  {'Total outer time':23s}: {total_outer:8.3f}s (100.0%)")
            self._log(f"  {'  Inner loops':23s}: {total_inner:8.3f}s ({100*total_inner/total_outer:5.1f}%)")
            self._log(f"  {'  Checkpoint saves':23s}: {total_checkpoint:8.3f}s ({100*total_checkpoint/total_outer:5.1f}%)")
            self._log(f"  {'  Cycle analysis':23s}: {total_cycle:8.3f}s ({100*total_cycle/total_outer:5.1f}%)")
            self._log(f"  {'  Early verifications':23s}: {total_verification:8.3f}s ({100*total_verification/total_outer:5.1f}%)")
            self._log(f"  {'  Other overhead':23s}: {overhead:8.3f}s ({100*overhead/total_outer:5.1f}%)")
        else:
            self._log("  No outer iteration timing data available")
        self._log("="*80)

    def solve(self,
              tau_p_init: float = 1.0,
              tau_r_init: float = 1.0,
              tau_decay: float = 0.95,
              tau_min: float = 0.01,
              max_outer_iter: int = 50,
              max_inner_iter: int = 100,
              damping: float = 0.5,
              inner_tol: float = 1e-6,
              outer_tol: Optional[float] = None,
              consecutive_tol: int = 2,
              verify_every_n: int = 1,
              tau_margin: float = 0.01,
              max_cycles_at_tau_min: int = 8,
              cycle_break_tau_threshold: Optional[float] = None,
              exact_polish_enabled: bool = True,
              exact_polish_max_iter: int = 20,
              exact_polish_tol: float = 1e-9,
              project_to_exact: bool = True,
              checkpoint_dir: str = './checkpoints',
              load_from_checkpoint: bool = False,
              config_hash: str = None,
              max_time_seconds: Optional[float] = None,
              disable_checkpoints: bool = False,
              disable_timing_report: bool = False,
              update_k_players: Optional[int] = None,
              link_player_updates: bool = True,
              update_k_acceptances: Optional[int] = None,
              log_strategy_every_inner: bool = True,
              inner_cycle_check_interval: int = 20) -> Tuple[pd.DataFrame, Dict]:
        """
        Solve for equilibrium strategy profile using smoothed fixed-point iteration.

        Args:
            tau_p_init: Initial smoothing temperature for proposals
            tau_r_init: Initial smoothing temperature for acceptances
            tau_decay: Decay rate for temperatures
            tau_min: Minimum temperature threshold
            max_outer_iter: Maximum outer loop iterations (safety valve)
            max_inner_iter: Maximum inner loop iterations (fixed-point)
            damping: Damping factor for updates (lambda)
            inner_tol: Convergence tolerance for inner loop
            outer_tol: Convergence tolerance for outer loop (defaults to 10*inner_tol)
            consecutive_tol: Number of consecutive converged outer iterations required
            verify_every_n: Only run early verification every nth stable outer iteration
                            (1 = every iteration, 10 = every 10th). Reduces cost when
                            verification dominates runtime.
            tau_margin: Margin for checking if tau is near tau_min (default 0.01 = 1%)
            max_cycles_at_tau_min: Stop if this many qualifying low-tau cycles are detected
            cycle_break_tau_threshold: Tau gate for cycle stopping (defaults to tau_min*(1+tau_margin))
            exact_polish_enabled: Run iterative exact-strategy polishing after annealing
            exact_polish_max_iter: Maximum polish iterations
            exact_polish_tol: Stagnation tolerance for polish iterations
            project_to_exact: Whether to project to exact equilibrium at end
            checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
            load_from_checkpoint: Whether to load from checkpoint if exists
            config_hash: Hash of configuration for checkpoint filename
            update_k_players: If set, only update this many randomly sampled players per
                              inner iteration (asynchronous / Gauss-Seidel style update).
                              None (default) updates all players simultaneously.
            link_player_updates: If True (default), proposals and acceptances are updated
                                 for the same sampled players. If False, proposals use
                                 update_k_players and acceptances are sampled independently
                                 (governed by update_k_acceptances).
            update_k_acceptances: If set, update exactly this many randomly sampled
                                  individual acceptance keys per inner iteration, independent
                                  of player-level sampling. Overrides link_player_updates
                                  for acceptances. None (default) uses player-level sampling.
            log_strategy_every_inner: If True, log the full strategy profile (proposals and
                                      acceptances) after every inner iteration. Intended for
                                      debugging; off by default.
            inner_cycle_check_interval: Check for a confirmed cycle every this many inner
                                        iterations. When a cycle is detected at low tau, the
                                        inner loop breaks immediately (avoids burning all
                                        max_inner_iter iterations on a known limit cycle).
                                        Set to 0 to disable. Default: 20.

        Returns:
            strategy_df: Equilibrium strategy DataFrame
            result: Dictionary with convergence information
        """
        # Set outer tolerance if not provided
        if outer_tol is None:
            outer_tol = 10 * inner_tol
        elif outer_tol < inner_tol:
            if self.verbose:
                self._log(
                    f"Warning: outer_tol ({outer_tol:.2e}) < inner_tol ({inner_tol:.2e}); ",
                    level='warning'
                )
            outer_tol = inner_tol

        tau_p = tau_p_init
        tau_r = tau_r_init
        if cycle_break_tau_threshold is None:
            cycle_break_tau_threshold = tau_min * (1 + tau_margin)

        outer_iter = 0
        converged = False

        # Track convergence history for consecutive check
        recent_max_changes = []

        # Counter for throttling early verification (reset after each attempt)
        iters_since_verify = 0
        low_tau_cycle_count = 0

        # Checkpoint setup
        checkpoint_path = None
        if config_hash:
            checkpoint_path = str(Path(checkpoint_dir) / f"checkpoint_{config_hash}.pkl")

            # Try to load from checkpoint
            if load_from_checkpoint and not disable_checkpoints:
                checkpoint = self._load_checkpoint(checkpoint_path)
                if checkpoint:
                    # Restore state
                    outer_iter = checkpoint['outer_iter'] + 1  # Resume from next iteration
                    tau_p = checkpoint['tau_p']
                    tau_r = checkpoint['tau_r']
                    self.p_proposals = checkpoint['p_proposals']
                    self.r_acceptances = checkpoint['r_acceptances']
                    # Restore random seed (for completeness, even though it doesn't affect resumed state)
                    if 'random_seed' in checkpoint:
                        self.random_seed = checkpoint['random_seed']

                    if self.verbose:
                        self._log(f"Resuming from outer iteration {outer_iter}")
                else:
                    if self.verbose:
                        self._log("No checkpoint found, starting fresh")

        if update_k_players is not None:
            k_update = min(update_k_players, len(self.players))
        else:
            k_update = None

        all_acceptance_keys = list(self.r_acceptances.keys())
        if update_k_acceptances is not None:
            k_acc = min(update_k_acceptances, len(all_acceptance_keys))
        else:
            k_acc = None

        if self.verbose:
            self._log("Starting equilibrium solver...")
            self._log(f"Initial tau_p={tau_p:.4f}, tau_r={tau_r:.4f}")
            self._log(f"Outer tolerance: {outer_tol:.2e}, consecutive required: {consecutive_tol}")
            if k_update is not None or k_acc is not None:
                proposal_desc = f"proposals: k={k_update}/{len(self.players)} players"
                if k_acc is not None:
                    acc_desc = f"acceptances: k={k_acc}/{len(all_acceptance_keys)} keys (independent)"
                elif not link_player_updates:
                    acc_desc = f"acceptances: k={k_update}/{len(self.players)} players (independent)"
                else:
                    acc_desc = f"acceptances: linked to proposals"
                self._log(f"Asynchronous updates — {proposal_desc}; {acc_desc}")
            if checkpoint_path:
                self._log(f"Checkpoints will be saved to: {checkpoint_path}")

        # Timing statistics
        timing_stats = {
            'create_df': [],
            'compute_transitions': [],
            'solve_values': [],
            'update_acceptances': [],
            'update_proposals': [],
            'damping': [],
            'checkpoint_save': [],
            'early_verification': [],
            'cycle_analysis': [],
            'outer_total': [],  # Total time per outer iteration
            'inner_total': []   # Total time for inner loop per outer iteration
        }

        start_time = time.time()
        stopping_reason = None

        max_change = float('inf')
        any_outer_started = False
        try:
          while outer_iter < max_outer_iter:
            any_outer_started = True
            if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                stopping_reason = 'time_budget'
                break
            outer_iter_start = time.time()  # Track total outer iteration time

            if self.verbose:
                timestamp = datetime.now().strftime('%H:%M:%S')
                self._log(f"\nOuter iteration {outer_iter + 1}/{max_outer_iter} [{timestamp}]")
                self._log(f"  tau_p={tau_p:.4f}, tau_r={tau_r:.4f}")

            # Inner fixed-point iteration
            inner_loop_start = time.time()
            max_change = float('inf')  # Track final max_change from inner loop
            inner_history = []         # Full per-iteration max_change history
            inner_converged = False
            effective_max_inner_iter = max_inner_iter
            partial_cycle_period = None  # Set if a partial cycle is detected mid-run
            inner_iter = 0
            while inner_iter < effective_max_inner_iter:
                if max_time_seconds is not None and (time.time() - start_time) >= max_time_seconds:
                    stopping_reason = 'time_budget'
                    break
                # Save old strategies
                old_proposals = self.p_proposals.copy()
                old_acceptances = self.r_acceptances.copy()

                # 1. Compute transition probabilities (FAST PATH: dict -> arrays directly)
                t0 = time.time()
                P, P_proposals, P_approvals = self._compute_transition_probabilities_fast()
                timing_stats['compute_transitions'].append(time.time() - t0)

                # 3. Solve value functions
                t0 = time.time()
                V = self._solve_value_functions(P)
                timing_stats['solve_values'].append(time.time() - t0)

                # Sample which players/keys to update this iteration
                proposal_players = (
                    list(np.random.choice(self.players, size=k_update, replace=False))
                    if k_update is not None else None
                )
                if k_acc is not None:
                    # Key-level sampling for acceptances, independent of proposals
                    acc_keys = set(
                        all_acceptance_keys[i]
                        for i in np.random.choice(len(all_acceptance_keys), size=k_acc, replace=False)
                    )
                    acc_players = None
                elif link_player_updates:
                    # Same player sample for both
                    acc_keys = None
                    acc_players = proposal_players
                else:
                    # Independent player-level sample for acceptances
                    acc_keys = None
                    acc_players = (
                        list(np.random.choice(self.players, size=k_update, replace=False))
                        if k_update is not None else None
                    )

                # 4. Update acceptances (smoothed)
                t0 = time.time()
                new_acceptances = self._update_acceptances(
                    V, tau_r, old_acceptances=old_acceptances,
                    players_subset=acc_players, keys_subset=acc_keys
                )
                timing_stats['update_acceptances'].append(time.time() - t0)

                # 5. Update proposals (smoothed)
                t0 = time.time()
                new_proposals = self._update_proposals(
                    V, P_approvals, tau_p, old_proposals=old_proposals,
                    players_subset=proposal_players
                )
                timing_stats['update_proposals'].append(time.time() - t0)

                # Measure convergence BEFORE applying damping: compare new vs old strategies.
                # Using |new - old| is correct for both full and partial (async) updates:
                # non-updated keys are copied from old, so they contribute 0 change.
                # This avoids the trivial-zero bug of residual-after-damping when damping=0.
                acceptance_change = max(
                    abs(new_acceptances[k] - old_acceptances[k])
                    for k in old_acceptances
                )
                proposal_change = max(
                    abs(new_proposals[k] - old_proposals[k])
                    for k in old_proposals
                )
                max_change = max(proposal_change, acceptance_change)

                # 6. Apply damping
                t0 = time.time()
                self.r_acceptances = self._apply_damping(
                    old_acceptances, new_acceptances, damping
                )
                self.p_proposals = self._apply_damping(
                    old_proposals, new_proposals, damping
                )
                timing_stats['damping'].append(time.time() - t0)
                inner_history.append(max_change)

                if inner_iter % 10 == 0 and self.verbose:
                    self._log(f"    Inner iter {inner_iter}: max_change={max_change:.6f}")

                if max_change < inner_tol:
                    if self.verbose:
                        self._log(f"    Converged after {inner_iter + 1} iterations")
                    inner_converged = True

                if log_strategy_every_inner and self.verbose:
                    self._log(f"\n    --- Strategy after inner iter {inner_iter} ---")
                    self._log(f"\n{self._compact_strategy_str()}\n")

                if inner_converged:
                    break

                # Periodic mid-inner cycle detection: break early when a confirmed cycle is
                # found at low tau.  Avoids wasting all max_inner_iter iterations on a known
                # limit cycle (the final polish will find the equilibrium anyway).
                if (inner_cycle_check_interval > 0
                        and inner_iter > 0
                        and inner_iter % inner_cycle_check_interval == 0):
                    at_low_tau = (tau_p <= cycle_break_tau_threshold
                                  and tau_r <= cycle_break_tau_threshold)
                    if at_low_tau:
                        t0 = time.time()
                        period_mid = detect_cycle(inner_history)
                        timing_stats['cycle_analysis'].append(time.time() - t0)
                        if period_mid is not None:
                            if self.verbose:
                                self._log(
                                    f"    ↺ Period-{period_mid} cycle confirmed at inner iter {inner_iter} "
                                    f"(low tau), breaking inner loop early."
                                )
                            break

                # After exhausting the initial budget, check for a partial cycle and
                # extend the run by 3x the candidate period so detect_cycle can confirm.
                if inner_iter == max_inner_iter - 1 and partial_cycle_period is None:
                    t0 = time.time()
                    partial = detect_partial_cycle(inner_history)
                    timing_stats['cycle_analysis'].append(time.time() - t0)
                    if partial is not None:
                        partial_cycle_period, overlap = partial
                        effective_max_inner_iter = max_inner_iter + 3 * partial_cycle_period
                        if self.verbose:
                            self._log(
                                f"    {format_partial_cycle_report(partial_cycle_period, overlap)}"
                            )

                inner_iter += 1

            if stopping_reason == 'time_budget':
                # Record inner timing and break out of outer loop cleanly
                inner_loop_time = time.time() - inner_loop_start
                timing_stats['inner_total'].append(inner_loop_time)
                break

            # Report oscillation when inner loop fails to converge
            if not inner_converged:
                t0 = time.time()
                period = detect_cycle(inner_history)
                timing_stats['cycle_analysis'].append(time.time() - t0)
                if self.verbose:
                    iters_run = len(inner_history)
                    if period is not None:
                        report = format_cycle_report(inner_history, period, cycles_to_show=1)
                        self._log(f"    ↺ No convergence after {iters_run} iterations.")
                        for line in report.split('\n'):
                            self._log(f"      {line}")
                    elif partial_cycle_period is not None:
                        # Extension ran but detect_cycle still couldn't confirm
                        self._log(
                            f"    ~ No convergence after {iters_run} iterations. "
                            f"Partial period-{partial_cycle_period} cycle was not confirmed "
                            f"after extension."
                        )
                low_tau_cycle = (
                    period is not None and
                    tau_p <= cycle_break_tau_threshold and
                    tau_r <= cycle_break_tau_threshold
                )
                if low_tau_cycle:
                    low_tau_cycle_count += 1
                    if self.verbose:
                        self._log(
                            f"    ↺ Low-tau cycle count: {low_tau_cycle_count}/{max_cycles_at_tau_min} "
                            f"(tau threshold={cycle_break_tau_threshold:.4e})"
                        )
                else:
                    low_tau_cycle_count = 0
            else:
                low_tau_cycle_count = 0

            inner_loop_time = time.time() - inner_loop_start
            timing_stats['inner_total'].append(inner_loop_time)
            if low_tau_cycle_count >= max_cycles_at_tau_min:
                stopping_reason = 'cycle_stuck'
                break

            # Track this outer iteration's convergence
            recent_max_changes.append(max_change)
            if len(recent_max_changes) > consecutive_tol:
                recent_max_changes.pop(0)  # Keep only last consecutive_tol changes

            if self.verbose:
                self._log(f"  Outer iteration max_change: {max_change:.6e}")

            # Save checkpoint after each outer iteration
            if checkpoint_path and config_hash and not disable_checkpoints:
                t0 = time.time()
                self._save_checkpoint(checkpoint_path, outer_iter, tau_p, tau_r, config_hash)
                timing_stats['checkpoint_save'].append(time.time() - t0)

            # Check if strategies have stabilized
            consecutive_converged = (len(recent_max_changes) >= consecutive_tol and
                                    all(change < outer_tol for change in recent_max_changes))

            # Early termination: if strategies are stable, try projection and verification
            if consecutive_converged:
                iters_since_verify += 1
                run_verification = (iters_since_verify >= verify_every_n)
            else:
                iters_since_verify = 0
                run_verification = False

            if consecutive_converged and not run_verification and self.verbose:
                self._log(f"\n  Strategies stable, skipping verification (next in {verify_every_n - iters_since_verify} iters)...")

            if run_verification:
                iters_since_verify = 0  # Reset counter after each attempt
                t_verify_start = time.time()  # Time the entire verification block

                if self.verbose:
                    self._log(f"\n  Strategies stable for {consecutive_tol} iterations, attempting early verification...")

                # Save current strategies
                old_proposals = self.p_proposals.copy()
                old_acceptances = self.r_acceptances.copy()

                # Create current strategy DataFrame
                current_strategy_df = self._create_strategy_dataframe()

                # Project and iterate until V stabilises, then verify
                success, message, polish_iters = self._polish_exact_equilibrium(
                    max_iter=exact_polish_max_iter, tol=exact_polish_tol
                )
                if self.verbose:
                    self._log(f"  Polish finished in {polish_iters} iteration(s).")

                # Recompute V from the polished strategy for the result dict
                projected_strategy_df = self._create_strategy_dataframe()
                _, _, V_projected = self._verify_candidate_equilibrium(projected_strategy_df)

                timing_stats['early_verification'].append(time.time() - t_verify_start)

                if success:
                    # Found valid equilibrium - stop early!
                    converged = True
                    if self.verbose:
                        self._log(f"  ✓ Equilibrium verification PASSED")
                        self._log(f"\n  Early termination (equilibrium found):")
                        self._log(f"    - Strategies stable for {consecutive_tol} iterations")
                        self._log(f"    - Projected equilibrium verified")
                        self._log(f"    - tau_p={tau_p:.4e}, tau_r={tau_r:.4e} (not yet at tau_min={tau_min:.4e})")
                    # Record total outer iteration time before breaking
                    outer_iter_total = time.time() - outer_iter_start
                    timing_stats['outer_total'].append(outer_iter_total)
                    # Keep the projected strategies (already in self.p_proposals, self.r_acceptances)
                    break
                else:
                    # Verification failed - restore old strategies and continue annealing
                    self.p_proposals = old_proposals
                    self.r_acceptances = old_acceptances
                    if self.verbose:
                        self._log(f"  ✗ Equilibrium verification failed:\n{message}")
                        self._log(f"  Continuing annealing...")

            # Check regular convergence criterion (temperature + stability)
            tau_near_min = (tau_p <= tau_min * (1 + tau_margin) and
                           tau_r <= tau_min * (1 + tau_margin))

            if tau_near_min and consecutive_converged:
                converged = True
                if self.verbose:
                    self._log(f"\n  Convergence criterion met (annealing complete):")
                    self._log(f"    - tau_p={tau_p:.4e} <= {tau_min * (1 + tau_margin):.4e}")
                    self._log(f"    - tau_r={tau_r:.4e} <= {tau_min * (1 + tau_margin):.4e}")
                    self._log(f"    - Last {consecutive_tol} max_changes < {outer_tol:.2e}")
                # Record total outer iteration time before breaking
                outer_iter_total = time.time() - outer_iter_start
                timing_stats['outer_total'].append(outer_iter_total)
                break

            # Decay temperatures
            tau_p = max(tau_p * tau_decay, tau_min)
            tau_r = max(tau_r * tau_decay, tau_min)

            # Record total outer iteration time
            outer_iter_total = time.time() - outer_iter_start
            timing_stats['outer_total'].append(outer_iter_total)

            outer_iter += 1

        except KeyboardInterrupt:
            self._log("\n\n" + "="*80, level='warning')
            self._log("INTERRUPTED BY USER", level='warning')
            self._log("="*80, level='warning')
            if not disable_timing_report:
                self._print_timing_report(timing_stats)
            raise

        # Determine the stopping reason
        early_stop_via_verification = False
        if stopping_reason == 'time_budget':
            pass
        elif converged and tau_p > tau_min * (1 + tau_margin):
            # Stopped early via verification (not via temperature convergence)
            early_stop_via_verification = True
            stopping_reason = 'early_verification'
        elif converged:
            # Stopped via regular convergence criterion
            stopping_reason = 'converged'
        elif stopping_reason == 'cycle_stuck':
            pass
        else:
            # Hit safety valve
            stopping_reason = 'max_iter'

        if self.verbose:
            if early_stop_via_verification:
                self._log(f"\nStopped early after {outer_iter + 1} outer iterations (equilibrium verified)")
            elif converged:
                self._log(f"\nAnnealing converged after {outer_iter + 1} outer iterations")
            elif stopping_reason == 'time_budget':
                self._log(f"\nStopped due to time budget (max_time_seconds={max_time_seconds})")
            elif stopping_reason == 'cycle_stuck':
                self._log(
                    f"\nStopped due to persistent low-tau cycles "
                    f"({low_tau_cycle_count} >= {max_cycles_at_tau_min})"
                )
            else:
                self._log(f"\nAnnealing stopped at max_outer_iter={max_outer_iter} (safety valve)")
                self._log(f"  Final max_change: {max_change:.6e} (outer_tol: {outer_tol:.2e})")

        # Final projection to exact equilibrium (unless we already did it via early stopping)
        if project_to_exact and not early_stop_via_verification and stopping_reason != 'time_budget':
            if self.verbose:
                self._log("\nProjecting to exact equilibrium...")

            # Iterate project→V→project until V stabilises
            success_polish, message_polish, polish_iters = self._polish_exact_equilibrium(
                max_iter=exact_polish_max_iter, tol=exact_polish_tol
            )
            if self.verbose:
                status = "succeeded" if success_polish else "did not verify"
                self._log(f"  Polish {status} after {polish_iters} iteration(s).")
        elif early_stop_via_verification and self.verbose:
            self._log("\nSkipping final projection (already projected and verified)")

        # Final strategy DataFrame
        final_strategy_df = self._create_strategy_dataframe()

        if stopping_reason == 'time_budget' and any_outer_started:
            outer_iterations_reported = outer_iter + 1
        else:
            outer_iterations_reported = outer_iter + 1 if converged else outer_iter

        result = {
            'converged': converged,
            'stopping_reason': stopping_reason,  # 'early_verification', 'converged', 'cycle_stuck', 'time_budget', or 'max_iter'
            'outer_iterations': outer_iterations_reported,
            'final_tau_p': tau_p,
            'final_tau_r': tau_r,
            'final_max_change': max_change,
            'outer_tol': outer_tol,
            'recent_max_changes': recent_max_changes.copy(),
            'timing_stats': timing_stats,
        }

        if self.verbose:
            self._log("\nSolver complete!")
            if not disable_timing_report:
                self._print_timing_report(timing_stats)

            self._log("="*80)

        return final_strategy_df, result
