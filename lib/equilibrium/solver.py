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

    def _compute_transition_probabilities(self, strategy_df: pd.DataFrame) -> Tuple:
        """Compute transition probabilities from strategy DataFrame."""
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

    def _update_acceptances(self, V: pd.DataFrame, tau_r: float) -> Dict:
        """Update acceptance probabilities using smoothed sigmoid.

        Args:
            V: Value functions
            tau_r: Smoothing temperature for acceptances

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
                        V_current = V.loc[current_state, approver]
                        V_next = V.loc[next_state, approver]

                        # Smoothed acceptance: sigmoid((V_next - V_current) / tau_r)
                        key = (proposer, current_state, next_state, approver)
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
                         tau_p: float) -> Dict:
        """Update proposal probabilities using smoothed softmax.

        Args:
            V: Value functions
            P_approvals: Approval probabilities
            tau_p: Smoothing temperature for proposals

        Returns:
            Updated proposal probabilities
        """
        new_proposals = {}

        for proposer in self.players:
            for current_state in self.states:
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

                        if np.isclose(V_next, V_current, atol=1e-12):
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
                    if np.isclose(val, max_value, atol=1e-12)
                ]

                # Distribute probability uniformly over argmax states
                for next_state in self.states:
                    key = (proposer, current_state, next_state)
                    if next_state in argmax_states:
                        self.p_proposals[key] = 1.0 / len(argmax_states)
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
            'timestamp': datetime.now().isoformat()
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

            if self.verbose:
                self._log(f"Loaded checkpoint from iteration {checkpoint['outer_iter']}")
                self._log(f"  Checkpoint created: {checkpoint['timestamp']}")
                self._log(f"  Resuming from tau_p={checkpoint['tau_p']:.4f}, tau_r={checkpoint['tau_r']:.4f}")

            return checkpoint
        except Exception as e:
            if self.verbose:
                self._log(f"Warning: Could not load checkpoint: {e}", level='warning')
            return None

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
              tau_margin: float = 0.01,
              project_to_exact: bool = True,
              checkpoint_dir: str = './checkpoints',
              load_from_checkpoint: bool = False,
              config_hash: str = None) -> Tuple[pd.DataFrame, Dict]:
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
            tau_margin: Margin for checking if tau is near tau_min (default 0.01 = 1%)
            project_to_exact: Whether to project to exact equilibrium at end
            checkpoint_dir: Directory to save checkpoints (default: './checkpoints')
            load_from_checkpoint: Whether to load from checkpoint if exists
            config_hash: Hash of configuration for checkpoint filename

        Returns:
            strategy_df: Equilibrium strategy DataFrame
            result: Dictionary with convergence information
        """
        # Set outer tolerance if not provided
        if outer_tol is None:
            outer_tol = 10 * inner_tol

        tau_p = tau_p_init
        tau_r = tau_r_init

        outer_iter = 0
        converged = False

        # Track convergence history for consecutive check
        recent_max_changes = []

        # Checkpoint setup
        checkpoint_path = None
        if config_hash:
            checkpoint_path = str(Path(checkpoint_dir) / f"checkpoint_{config_hash}.pkl")

            # Try to load from checkpoint
            if load_from_checkpoint:
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

        if self.verbose:
            self._log("Starting equilibrium solver...")
            self._log(f"Initial tau_p={tau_p:.4f}, tau_r={tau_r:.4f}")
            self._log(f"Outer tolerance: {outer_tol:.2e}, consecutive required: {consecutive_tol}")
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
            'outer_total': [],  # Total time per outer iteration
            'inner_total': []   # Total time for inner loop per outer iteration
        }

        while outer_iter < max_outer_iter:
            outer_iter_start = time.time()  # Track total outer iteration time

            if self.verbose:
                timestamp = datetime.now().strftime('%H:%M:%S')
                self._log(f"\nOuter iteration {outer_iter + 1}/{max_outer_iter} [{timestamp}]")
                self._log(f"  tau_p={tau_p:.4f}, tau_r={tau_r:.4f}")

            # Inner fixed-point iteration
            inner_loop_start = time.time()
            max_change = float('inf')  # Track final max_change from inner loop
            for inner_iter in range(max_inner_iter):
                # Save old strategies
                old_proposals = self.p_proposals.copy()
                old_acceptances = self.r_acceptances.copy()

                # 1. Create strategy DataFrame
                t0 = time.time()
                strategy_df = self._create_strategy_dataframe()
                timing_stats['create_df'].append(time.time() - t0)

                # 2. Compute transition probabilities
                t0 = time.time()
                P, P_proposals, P_approvals = self._compute_transition_probabilities(
                    strategy_df
                )
                timing_stats['compute_transitions'].append(time.time() - t0)

                # 3. Solve value functions
                t0 = time.time()
                V = self._solve_value_functions(P)
                timing_stats['solve_values'].append(time.time() - t0)

                # 4. Update acceptances (smoothed)
                t0 = time.time()
                new_acceptances = self._update_acceptances(V, tau_r)
                timing_stats['update_acceptances'].append(time.time() - t0)

                # 5. Update proposals (smoothed)
                t0 = time.time()
                new_proposals = self._update_proposals(V, P_approvals, tau_p)
                timing_stats['update_proposals'].append(time.time() - t0)

                # 6. Apply damping
                t0 = time.time()
                self.r_acceptances = self._apply_damping(
                    old_acceptances, new_acceptances, damping
                )
                self.p_proposals = self._apply_damping(
                    old_proposals, new_proposals, damping
                )
                timing_stats['damping'].append(time.time() - t0)

                # Check convergence
                proposal_change = np.max([
                    abs(new_proposals[k] - old_proposals[k])
                    for k in old_proposals
                ])
                acceptance_change = np.max([
                    abs(new_acceptances[k] - old_acceptances[k])
                    for k in old_acceptances
                ])
                max_change = max(proposal_change, acceptance_change)

                if inner_iter % 10 == 0 and self.verbose:
                    self._log(f"    Inner iter {inner_iter}: max_change={max_change:.6f}")

                if max_change < inner_tol:
                    if self.verbose:
                        self._log(f"    Converged after {inner_iter + 1} iterations")
                    break

            inner_loop_time = time.time() - inner_loop_start
            timing_stats['inner_total'].append(inner_loop_time)

            # Track this outer iteration's convergence
            recent_max_changes.append(max_change)
            if len(recent_max_changes) > consecutive_tol:
                recent_max_changes.pop(0)  # Keep only last consecutive_tol changes

            if self.verbose:
                self._log(f"  Outer iteration max_change: {max_change:.6e}")

            # Save checkpoint after each outer iteration
            if checkpoint_path and config_hash:
                t0 = time.time()
                self._save_checkpoint(checkpoint_path, outer_iter, tau_p, tau_r, config_hash)
                timing_stats['checkpoint_save'].append(time.time() - t0)

            # Check if strategies have stabilized
            consecutive_converged = (len(recent_max_changes) >= consecutive_tol and
                                    all(change < outer_tol for change in recent_max_changes))

            # Early termination: if strategies are stable, try projection and verification
            if consecutive_converged:
                t_verify_start = time.time()  # Time the entire verification block

                if self.verbose:
                    self._log(f"\n  Strategies stable for {consecutive_tol} iterations, attempting early verification...")

                # Save current strategies
                old_proposals = self.p_proposals.copy()
                old_acceptances = self.r_acceptances.copy()

                # Create current strategy DataFrame
                current_strategy_df = self._create_strategy_dataframe()

                # Compute current value functions
                P, _, _ = self._compute_transition_probabilities(current_strategy_df)
                V_current = self._solve_value_functions(P)

                # Project to exact equilibrium (temporarily)
                self._project_to_exact_equilibrium(V_current)

                # Create projected strategy DataFrame
                projected_strategy_df = self._create_strategy_dataframe()

                # Verify the projected equilibrium
                success, message, V_projected = self._verify_candidate_equilibrium(projected_strategy_df)

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

        # Determine the stopping reason
        early_stop_via_verification = False
        if converged and tau_p > tau_min * (1 + tau_margin):
            # Stopped early via verification (not via temperature convergence)
            early_stop_via_verification = True
            stopping_reason = 'early_verification'
        elif converged:
            # Stopped via regular convergence criterion
            stopping_reason = 'converged'
        else:
            # Hit safety valve
            stopping_reason = 'max_iter'

        if self.verbose:
            if early_stop_via_verification:
                self._log(f"\nStopped early after {outer_iter + 1} outer iterations (equilibrium verified)")
            elif converged:
                self._log(f"\nAnnealing converged after {outer_iter + 1} outer iterations")
            else:
                self._log(f"\nAnnealing stopped at max_outer_iter={max_outer_iter} (safety valve)")
                self._log(f"  Final max_change: {max_change:.6e} (outer_tol: {outer_tol:.2e})")

        # Final projection to exact equilibrium (unless we already did it via early stopping)
        if project_to_exact and not early_stop_via_verification:
            if self.verbose:
                self._log("\nProjecting to exact equilibrium...")

            # Recompute value functions one more time
            strategy_df = self._create_strategy_dataframe()
            P, _, _ = self._compute_transition_probabilities(strategy_df)
            V = self._solve_value_functions(P)

            # Project to exact equilibrium
            self._project_to_exact_equilibrium(V)
        elif early_stop_via_verification and self.verbose:
            self._log("\nSkipping final projection (already projected and verified)")

        # Final strategy DataFrame
        final_strategy_df = self._create_strategy_dataframe()

        result = {
            'converged': converged,
            'stopping_reason': stopping_reason,  # 'early_verification', 'converged', or 'max_iter'
            'outer_iterations': outer_iter + 1 if converged else outer_iter,
            'final_tau_p': tau_p,
            'final_tau_r': tau_r,
            'final_max_change': max_change,
            'outer_tol': outer_tol,
            'recent_max_changes': recent_max_changes.copy(),
            'timing_stats': timing_stats,
        }

        if self.verbose:
            self._log("\nSolver complete!")
            self._log("\n" + "="*80)
            self._log("TIMING ANALYSIS")
            self._log("="*80)

            # Order for display: inner loop operations first, then outer loop operations
            inner_ops = ['create_df', 'compute_transitions', 'solve_values',
                        'update_acceptances', 'update_proposals', 'damping']
            outer_ops = ['inner_total', 'checkpoint_save', 'early_verification', 'outer_total']

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

            # Compute overhead
            total_outer = sum(timing_stats.get('outer_total', []))
            total_inner = sum(timing_stats.get('inner_total', []))
            total_checkpoint = sum(timing_stats.get('checkpoint_save', []))
            total_verification = sum(timing_stats.get('early_verification', []))
            overhead = total_outer - total_inner - total_checkpoint - total_verification

            self._log("\nBreakdown:")
            if total_outer > 0:
                self._log(f"  {'Total outer time':23s}: {total_outer:8.3f}s (100.0%)")
                self._log(f"  {'  Inner loops':23s}: {total_inner:8.3f}s ({100*total_inner/total_outer:5.1f}%)")
                self._log(f"  {'  Checkpoint saves':23s}: {total_checkpoint:8.3f}s ({100*total_checkpoint/total_outer:5.1f}%)")
                self._log(f"  {'  Early verifications':23s}: {total_verification:8.3f}s ({100*total_verification/total_outer:5.1f}%)")
                self._log(f"  {'  Other overhead':23s}: {overhead:8.3f}s ({100*overhead/total_outer:5.1f}%)")
            else:
                self._log("  No outer iteration timing data available")

            self._log("="*80)

        return final_strategy_df, result
