import numpy as np
from typing import List, Dict
from lib.country import Country
from lib.coalition import Coalition


class State:
    """ Encodes the current state of the dynamic coalition game.

    Arguments:
        name: Coalition's name, e.g., '(TC)'.
        coalitions: List of Coalition instances that exist in current state.
        all_countries: List of all Country instances that exist in the game.
        power_rule: Rule to determine the strongest coalition.
        min_power: Minimum required world power share to do geoengineering.
    """
    def __init__(self,
                 name: str,
                 coalitions: List[Coalition],
                 all_countries: List[Country],
                 power_rule: str,
                 min_power: float = None):

        # Safety checks.
        assert all(isinstance(country, Country) for country in all_countries)
        assert all(isinstance(coal, Coalition) for coal in coalitions)

        self.name = name
        self.coalitions = coalitions
        self.all_countries = all_countries
        self.power_rule = power_rule
        self.min_power = min_power
        self.coalition_powers = [coal.total_power for coal in self.coalitions]

        assert np.isclose(np.sum(self.coalition_powers), 1., atol=1e-9),\
            "Coalition powers must sum up to 1."

    @property
    def strongest_coalition(self) -> Coalition:
        """
        Returns the coalition that, according to self.power_rule,
        gets to implement geoengineering.
        """
        if self.power_rule == "power_threshold":
            # Coalition with the highest share of the world power
            # gets to implement geoengineering. If powers are tied,
            # the one with the highest preferred G-level (free-driver) wins.
            def sort_key(coalition): return (coalition.total_power, coalition.avg_ideal_G)
        elif self.power_rule == "weak_governance":
            # Free-driver case: Coalition with the highest average ideal
            # geoengineering level gets to deploy.
            def sort_key(coalition): return coalition.avg_ideal_G
        else:
            msg = ("Incorrect power threshold specification. "
                   "Must be in ['power_threshold', 'weak_governance']")
            raise ValueError(msg)

        sorted_coalitions = sorted(self.coalitions, key=sort_key, reverse=True)
        strongest_coalition = sorted_coalitions[0]
        assert isinstance(strongest_coalition, Coalition)

        return strongest_coalition

    @property
    def geo_deployment_level(self) -> float:
        """Geoengineering deployment chosen by the strongest coalition."""
        winner = self.strongest_coalition
        winner_power = winner.total_power
        G = winner.avg_ideal_G

        if self.power_rule == "power_threshold":
            assert self.min_power is not None, ("Minimum power threshold "
                                                "is not defined.")
            # If minimum power threshold is not exceeded,
            # nobody gets to deploy geoengineering.
            if winner_power < self.min_power:
                G = 0.

            # If multiple coalitions exceed the threshold and have tied power,
            # the strongest_coalition property has already resolved this via 
            # G-level tie-breaking. We only raise an error if both power AND 
            # G-level are tied between different coalitions.
            else:
                for coal in self.coalitions:
                    if coal is winner:
                        continue
                    if (np.isclose(coal.total_power, winner_power, atol=1e-9) and 
                        np.isclose(coal.avg_ideal_G, G, atol=1e-9)):
                        raise ValueError(
                            f"State '{self.name}' has multiple coalitions tied for both "
                            f"power ({winner_power:.4f}) and G-level ({G:.4f}). "
                            "Deployment is ambiguous."
                        )

        return G

    @property
    def payoffs(self) -> Dict[str, float]:
        """Calculate the payoffs for all countries, given the current
        coalition structure and corresponding geoengineering deployment."""
        G = self.geo_deployment_level

        names = [country.name for country in self.all_countries]
        payoffs = [country.payoff(G) for country in self.all_countries]

        return dict(zip(names, payoffs))
