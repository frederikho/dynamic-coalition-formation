"""Post-processing of solved equilibria: benchmark the farsighted MPE against
static coalition-formation concepts (internal/external stability, gamma-core).

The solver tells us *which* strategy profiles are equilibria. This package asks
what those equilibria mean economically, and how the predictions differ from the
concepts the coalition-formation literature normally uses.
"""

from lib.results_analysis.benchmarks import (
    PayoffGame,
    gamma_core,
    internal_external_stability,
)

__all__ = [
    "PayoffGame",
    "gamma_core",
    "internal_external_stability",
]
