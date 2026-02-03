"""
Equilibrium solver module for coalition formation games.

This module provides algorithms to find equilibrium strategy profiles
from random initialization using smoothed fixed-point iteration.
"""

from .solver import EquilibriumSolver, sigmoid, softmax

__all__ = ['EquilibriumSolver', 'sigmoid', 'softmax']
