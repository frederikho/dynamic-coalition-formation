"""
Scenario configurations for equilibrium finding.

This module contains predefined scenarios with different player configurations,
power rules, damage parameters, and other settings.

Defining a new scenario only requires 'power_rule'. Everything else has defaults:
  - m_damage:           1.0 for all players
  - unanimity_required: True
  - min_power:          0.501 if power_threshold, None if weak_governance

The scenario name is the dict key and is automatically injected as 'scenario_name'
when retrieved via get_scenario().
"""

import warnings


class _ScenarioRegistry(dict):
    """Dict subclass that warns when a key is overwritten."""

    def __setitem__(self, key, value):
        if key in self:
            warnings.warn(
                f"Duplicate scenario key '{key}' in scenarios.py — "
                "previous definition will be overwritten.",
                UserWarning,
                stacklevel=2,
            )
        super().__setitem__(key, value)


def _apply_defaults(config):
    """Fill in default values for fields omitted from a scenario dict."""
    if "m_damage" not in config:
        config["m_damage"] = {p: 1.0 for p in config["players"]}
    if "unanimity_required" not in config:
        config["unanimity_required"] = True
    if "min_power" not in config:
        config["min_power"] = 0.501 if config["power_rule"] == "power_threshold" else None
    return config


# ---------------------------------------------------------------------------
# Base configurations for different player counts
# ---------------------------------------------------------------------------

def get_base_config_n3():
    """Base configuration for 3-player games (W, T, C)."""
    players = ["W", "T", "C"]
    return {
        "players": players,
        "base_temp": {"W": 21.5, "T": 14.0, "C": 11.5},
        "ideal_temp": {player: 13.0 for player in players},
        "delta_temp": {player: 3.0 for player in players},
        "power": {player: 1/3 for player in players},
        "protocol": {player: 1/3 for player in players},
        "discounting": 0.99,
        "state_names": None  # Will be generated automatically
    }


def get_base_config_n4():
    """Base configuration for 4-player games (W, T, C, F)."""
    players = ["W", "T", "C", "F"]
    return {
        "players": players,
        "base_temp": {"W": 21.5, "T": 14.0, "C": 11.5, "F": 9.0},
        "ideal_temp": {player: 13.0 for player in players},
        "delta_temp": {player: 3.0 for player in players},
        "power": {player: 1/4 for player in players},
        "protocol": {player: 1/4 for player in players},
        "discounting": 0.99,
        "state_names": None  # Will be generated automatically
    }


def get_base_config_n5():
    """Base configuration for 5-player games (W, T, C, F, H)."""
    players = ["W", "T", "C", "F", "H"]
    return {
        "players": players,
        "base_temp": {"H": 25.0, "W": 21.5, "T": 14.0, "C": 11.5, "F": 9.0},
        "ideal_temp": {player: 13.0 for player in players},
        "delta_temp": {player: 3.0 for player in players},
        "power": {player: 1/5 for player in players},
        "protocol": {player: 1/5 for player in players},
        "discounting": 0.99,
        "state_names": None  # Will be generated automatically
    }


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS = _ScenarioRegistry()

# ========== 3-player scenarios ==========

SCENARIOS["weak_governance_n3"] = {
    **get_base_config_n3(),
    "scenario_description": "Standard 3-player weak governance scenario.",
    "power_rule": "weak_governance",
}

SCENARIOS["power_threshold_n3"] = {
    **get_base_config_n3(),
    "scenario_description": "3-player power threshold requiring majority support (>50%).",
    "power_rule": "power_threshold",
}

SCENARIOS["power_threshold_no_unanimity_n3"] = {
    **get_base_config_n3(),
    "scenario_description": "Power threshold without unanimity; W less vulnerable (0.75), T more vulnerable (1.25).",
    "m_damage": {"W": 0.75, "T": 1.25, "C": 1.0},
    "power_rule": "power_threshold",
    "unanimity_required": False,
}

SCENARIOS["high_discounting_n3"] = {
    **get_base_config_n3(),
    "scenario_description": "Weak governance with high discounting (δ=0.95).",
    "power_rule": "weak_governance",
    "discounting": 0.95,
}

SCENARIOS["unequal_power_n3"] = {
    **get_base_config_n3(),
    "scenario_description": "Asymmetric power: W dominant (50%), T moderate (30%), C weak (20%).",
    "power": {"W": 0.5, "T": 0.3, "C": 0.2},
    "power_rule": "power_threshold",
}

# ========== 4-player scenarios ==========

SCENARIOS["weak_governance_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Standard 4-player weak governance.",
    "power_rule": "weak_governance",
}

SCENARIOS["weak_governance_no_unanimity_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Standard 4-player weak governance without unanimity requirement.",
    "power_rule": "weak_governance",
    "unanimity_required": False,
}

SCENARIOS["power_threshold_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "4-player power threshold requiring majority support (>50%).",
    "power_rule": "power_threshold",
}

SCENARIOS["power_threshold_no_unanimity_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Power threshold without unanimity.",
    "power_rule": "power_threshold",
    "unanimity_required": False,
}

SCENARIOS["power_threshold_high_discounting_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Power threshold with high discounting (δ=0.90).",
    "power_rule": "power_threshold",
    "discounting": 0.90,
}

SCENARIOS["weak_governance_high_discounting_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Weak governance with high discounting (δ=0.90).",
    "power_rule": "weak_governance",
    "discounting": 0.90,
}

SCENARIOS["weak_governance_low_discounting_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Weak governance with very low discounting (δ=0.999).",
    "power_rule": "weak_governance",
    "discounting": 0.999,
}

SCENARIOS["dominant_warm_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "W holds dominant power (50%) while others share the rest equally.",
    "power": {"W": 0.5, "T": 0.2, "C": 0.15, "F": 0.15},
    "power_rule": "power_threshold",
}

SCENARIOS["very_dominant_warm_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "W holds very dominant power (60%) while others share the rest equally.",
    "power": {"W": 0.6, "T": 0.15, "C": 0.15, "F": 0.1},
    "power_rule": "power_threshold",
}

SCENARIOS["dominant_cold_n4"] = {
    **get_base_config_n4(),
    "power": {"W": 0.2, "T": 0.2, "C": 0.5, "F": 0.1},
    "power_rule": "power_threshold",
}

SCENARIOS["very_dominant_cold_n4"] = {
    **get_base_config_n4(),
    "power": {"W": 0.1, "T": 0.2, "C": 0.6, "F": 0.1},
    "power_rule": "power_threshold",
}

SCENARIOS["lower_power_high_threshold_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Power threshold with a lower threshold (49%).",
    "power_rule": "power_threshold",
    "min_power": 0.49,
}

SCENARIOS["unequal_protocol_favouring_W_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Unequal proposal probabilities favouring W.",
    "protocol": {"W": 0.4, "T": 0.3, "C": 0.2, "F": 0.1},
    "power_rule": "power_threshold",
}

SCENARIOS["unequal_protocol_favouring_F_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Unequal proposal probabilities favouring F.",
    "protocol": {"W": 0.1, "T": 0.2, "C": 0.3, "F": 0.4},
    "power_rule": "power_threshold",
}

SCENARIOS["weak_governance_high_damage_sensitivity_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "All countries highly sensitive to climate deviations (m_damage=2.0), weak governance.",
    "m_damage": {player: 2.0 for player in ["W", "T", "C", "F"]},
    "power_rule": "weak_governance",
}

SCENARIOS["power_threshold_high_damage_sensitivity_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "All countries highly sensitive to climate deviations (m_damage=2.0), power threshold.",
    "m_damage": {player: 2.0 for player in ["W", "T", "C", "F"]},
    "power_rule": "power_threshold",
}

SCENARIOS["power_threshold_low_damage_sensitivity_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "All countries less sensitive to climate impacts (m_damage=0.5), power threshold.",
    "m_damage": {player: 0.5 for player in ["W", "T", "C", "F"]},
    "power_rule": "power_threshold",
}

SCENARIOS["weak_governance_low_damage_sensitivity_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "All countries less sensitive to climate impacts (m_damage=0.5), weak governance.",
    "m_damage": {player: 0.5 for player in ["W", "T", "C", "F"]},
    "power_rule": "weak_governance",
}

SCENARIOS["power_threshold_equal_geoengineering_preferences_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "All players want the same geoengineering level (ideal temp = base temp), power threshold.",
    "ideal_temp": {"W": 21.5, "T": 14.0, "C": 11.5, "F": 9.0},
    "power_rule": "power_threshold",
}

SCENARIOS["weak_governance_equal_geoengineering_preferences_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "All players want the same geoengineering level (ideal temp = base temp), weak governance.",
    "ideal_temp": {"W": 21.5, "T": 14.0, "C": 11.5, "F": 9.0},
    "power_rule": "weak_governance",
}

SCENARIOS["weak_governance_similar_geoengineering_preferences_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Players have ideal temperatures close to their base temperature, weak governance.",
    "ideal_temp": {"W": 19, "T": 13.0, "C": 13, "F": 11},
    "power_rule": "weak_governance",
}

SCENARIOS["power_threshold_similar_geoengineering_preferences_n4"] = {
    **get_base_config_n4(),
    "scenario_description": "Players have ideal temperatures close to their base temperature, power threshold.",
    "ideal_temp": {"W": 19, "T": 13.0, "C": 13, "F": 11},
    "power_rule": "power_threshold",
}

# ========== 5-player scenarios ==========

SCENARIOS["weak_governance_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "Standard 5-player weak governance including hot country H (base temp 25°C).",
    "power_rule": "weak_governance",
}

SCENARIOS["power_threshold_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "5-player power threshold with equal power distribution.",
    "power_rule": "power_threshold",
}

SCENARIOS["power_threshold_no_unanimity_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "5-player power threshold without unanimity; H least vulnerable (0.5).",
    "m_damage": {"W": 0.75, "T": 1.25, "C": 1.0, "F": 1.1, "H": 0.5},
    "power_rule": "power_threshold",
    "unanimity_required": False,
}

SCENARIOS["unequal_power_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "Gradually declining power from W=30% to H=10%.",
    "power": {"W": 0.3, "T": 0.25, "C": 0.2, "F": 0.15, "H": 0.1},
    "power_rule": "power_threshold",
}

SCENARIOS["hot_country_advantage_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "H has low damage sensitivity (0.5) while others face higher impacts (1.2).",
    "power": {"W": 0.2, "T": 0.2, "C": 0.2, "F": 0.2, "H": 0.2},
    "m_damage": {"W": 1.2, "T": 1.2, "C": 1.2, "F": 1.2, "H": 0.5},
    "power_rule": "power_threshold",
    "unanimity_required": False,
}

SCENARIOS["varied_damage_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "Graduated damage sensitivity: H least vulnerable (0.3), F most (1.5).",
    "m_damage": {"W": 0.6, "T": 0.9, "C": 1.2, "F": 1.5, "H": 0.3},
    "power_rule": "power_threshold",
    "unanimity_required": False,
}

SCENARIOS["high_discounting_n5"] = {
    **get_base_config_n5(),
    "scenario_description": "5-player weak governance with high discounting (δ=0.95).",
    "power_rule": "weak_governance",
    "discounting": 0.95,
}


# ---------------------------------------------------------------------------
# Access functions
# ---------------------------------------------------------------------------

def get_scenario(name):
    """
    Get a scenario configuration by name.

    Injects 'scenario_name' from the registry key and fills in defaults for
    any omitted fields (m_damage, unanimity_required, min_power).

    Args:
        name: Scenario name (registry key)

    Returns:
        Scenario configuration dictionary

    Raises:
        KeyError: If scenario name not found
    """
    if name not in SCENARIOS:
        available = ', '.join(sorted(SCENARIOS.keys()))
        raise KeyError(f"Unknown scenario '{name}'. Available scenarios: {available}")
    config = SCENARIOS[name].copy()
    config["scenario_name"] = name
    _apply_defaults(config)
    return config


def list_scenarios(filter_players=None):
    """
    List available scenario names.

    Args:
        filter_players: Optional filter for number of players (e.g., 3, 4, or 5)

    Returns:
        List of scenario names
    """
    if filter_players is None:
        return sorted(SCENARIOS.keys())

    suffix = f"_n{filter_players}"
    return sorted([name for name in SCENARIOS.keys() if name.endswith(suffix)])


def get_scenario_summary(name):
    """
    Get a human-readable summary of a scenario.

    Args:
        name: Scenario name

    Returns:
        Dictionary with summary information
    """
    config = get_scenario(name)

    return {
        'name': name,
        'players': config['players'],
        'n_players': len(config['players']),
        'power_rule': config['power_rule'],
        'min_power': config.get('min_power'),
        'unanimity_required': config['unanimity_required'],
        'discounting': config['discounting'],
        'damage_params': config['m_damage'],
        'power_distribution': config['power']
    }
