"""
Scenario configurations for equilibrium finding.

This module contains predefined scenarios with different player configurations,
power rules, damage parameters, and other settings.
"""

# Base configurations for different player counts

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
        "base_temp": { "H": 25.0, "W": 21.5, "T": 14.0, "C": 11.5, "F": 9.0},
        "ideal_temp": {player: 13.0 for player in players},
        "delta_temp": {player: 3.0 for player in players},
        "power": {player: 1/5 for player in players},
        "protocol": {player: 1/5 for player in players},
        "discounting": 0.99,
        "state_names": None  # Will be generated automatically
    }


# All scenario definitions

SCENARIOS = {
    # ========== 3-player scenarios ==========
    
    "weak_governance_n3": {
        **get_base_config_n3(),
        "scenario_name": "weak_governance_n3",
        "scenario_description": "Standard 3-player weak governance scenario with equal damage sensitivity and unanimity requirement.",
        "experiment_name": "weak_governance_n3",
        "m_damage": {player: 1.0 for player in ["W", "T", "C"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "power_threshold_n3": {
        **get_base_config_n3(),
        "scenario_name": "power_threshold_n3",
        "scenario_description": "3-player scenario with power threshold governance requiring majority support (>50%) for coalition effectiveness.",
        "experiment_name": "power_threshold_n3",
        "m_damage": {player: 1.0 for player in ["W", "T", "C"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
    
    "power_threshold_no_unanimity_n3": {
        **get_base_config_n3(),
        "scenario_name": "power_threshold_no_unanimity_n3",
        "scenario_description": "Power threshold without unanimity requirement; W less vulnerable (0.75), T more vulnerable (1.25) to climate impacts.",
        "experiment_name": "power_threshold_no_unanimity_n3",
        "m_damage": {"W": 0.75, "T": 1.25, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "high_discounting_n3": {
        **get_base_config_n3(),
        "scenario_name": "high_discounting_n3",
        "scenario_description": "Weak governance with high discounting (δ=0.95), emphasizing near-term payoffs over long-term outcomes.",
        "experiment_name": "high_discounting_n3",
        "m_damage": {player: 1.0 for player in ["W", "T", "C"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True,
        "discounting": 0.95
    },
    
    "unequal_power_n3": {
        **get_base_config_n3(),
        "scenario_name": "unequal_power_n3",
        "scenario_description": "Asymmetric power distribution with W dominant (50%), T moderate (30%), and C weak (20%).",
        "experiment_name": "unequal_power_n3",
        "power": {"W": 0.5, "T": 0.3, "C": 0.2},
        "m_damage": {player: 1.0 for player in ["W", "T", "C"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
    
    # ========== 4-player scenarios ==========
    
    "weak_governance_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "weak_governance_n4",
        "scenario_description": "Standard 4-player weak governance with equal damage sensitivity and unanimity requirement.","experiment_name": "weak_governance_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "weak_governance_no_unanimity_n4": {
        **get_base_config_n4(),

        "scenario_name": "weak_governance_no_unanimity_n4",
        "scenario_description": "Standard 4-player weak governance with equal damage sensitivity and no unanimity requirement.",
        "experiment_name": "weak_governance_no_unanimity_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": False
    },
    
    "power_threshold_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "power_threshold_n4",
        "scenario_description": "4-player power threshold governance requiring majority support (>50%) for effectiveness.",
        "experiment_name": "power_threshold_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
    
    "power_threshold_no_unanimity_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "power_threshold_no_unanimity_n4",
        "scenario_description": "Power threshold without unanimity; varied damage parameters (W=0.75, T=1.25, C=1.0, F=1.1).","experiment_name": "power_threshold_no_unanimity_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "high_discounting_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "high_discounting_n4",
        "scenario_description": "Weak governance with high discounting (δ=0.95), emphasizing near-term over long-term payoffs.","experiment_name": "high_discounting_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True,
        "discounting": 0.95
    },
    
    "low_discounting_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "low_discounting_n4",
        "scenario_description": "Weak governance with very low discounting (δ=0.999), heavily weighting long-term outcomes.","experiment_name": "low_discounting_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True,
        "discounting": 0.999
    },
    
    "unequal_power_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "unequal_power_n4",
        "scenario_description": "Gradually declining power distribution (W=40%, T=30%, C=20%, F=10%) with power threshold.","experiment_name": "unequal_power_n4",
        "power": {"W": 0.4, "T": 0.3, "C": 0.2, "F": 0.1},
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
    
    "dominant_warm_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "dominant_warm_n4",
        "scenario_description": "Warm country W holds dominant power (60%) while others share remaining influence equally.","experiment_name": "dominant_warm_n4",
        "power": {"W": 0.6, "T": 0.2, "C": 0.1, "F": 0.1},
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },

    "dominant_cold_n4": {
        **get_base_config_n4(),
        "experiment_name": "dominant_cold_n4",
        "power": {"W": 0.2, "T": 0.2, "C": 0.5, "F": 0.1},
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
        
    "balanced_power_high_threshold_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "balanced_power_high_threshold_n4",
        "scenario_description": "Equal power distribution but high approval threshold (67%) for coalition effectiveness.","experiment_name": "balanced_power_high_threshold_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.67,
        "unanimity_required": True
    },
    
    "varied_damage_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "varied_damage_n4",
        "scenario_description": "Graduated damage sensitivity from low (W=1.5) to high (F=0.75) without unanimity requirement.","experiment_name": "varied_damage_n4",
        "m_damage": {"W": 1.5, "T": 1.25, "C": 1.0, "F": 0.75},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "extreme_damage_asymmetry_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "extreme_damage_asymmetry_n4",
        "scenario_description": "Extreme differences in climate vulnerability, no unanimity.",
        "experiment_name": "extreme_damage_asymmetry_n4",
        "m_damage": {"W": 2.5, "T": 1.75, "C": 1.25, "F": 0.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "weak_governance_varied_damage_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "weak_governance_varied_damage_n4",
        "scenario_description": "Weak governance with moderate variation in damage parameters across players.","experiment_name": "weak_governance_varied_damage_n4",
        "m_damage": {"W": 0.8, "T": 1.0, "C": 1.2, "F": 1.4},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "low_threshold_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "low_threshold_n4",
        "scenario_description": "Lower power threshold (34%) making coalition formation easier than standard scenarios.","experiment_name": "low_threshold_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.34,
        "unanimity_required": True
    },
    
    "unequal_protocol_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "unequal_protocol_n4",
        "scenario_description": "Unequal proposal probabilities mirroring power distribution under weak governance.","experiment_name": "unequal_protocol_n4",
        "protocol": {"W": 0.4, "T": 0.3, "C": 0.2, "F": 0.1},
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "mixed_asymmetry_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "mixed_asymmetry_n4",
        "scenario_description": "Combined asymmetries in both power distribution and climate damage sensitivity.","experiment_name": "mixed_asymmetry_n4",
        "power": {"W": 0.35, "T": 0.35, "C": 0.2, "F": 0.1},
        "m_damage": {"W": 0.8, "T": 1.0, "C": 1.3, "F": 1.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "alliance_structure_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "alliance_structure_n4",
        "scenario_description": "Two pairs (W-T and C-F) with matched characteristics suggesting alliance formation.","experiment_name": "alliance_structure_n4",
        "power": {"W": 0.3, "T": 0.3, "C": 0.2, "F": 0.2},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.5, "F": 1.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "high_damage_sensitivity_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "high_damage_sensitivity_n4",
        "scenario_description": "All countries highly sensitive to climate deviations (m_damage=2.0 for all).","experiment_name": "high_damage_sensitivity_n4",
        "m_damage": {player: 2.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "low_damage_sensitivity_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "low_damage_sensitivity_n4",
        "scenario_description": "All countries less sensitive to climate impacts (m_damage=0.5 for all).","experiment_name": "low_damage_sensitivity_n4",
        "m_damage": {player: 0.5 for player in ["W", "T", "C", "F"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "moderate_discounting_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "moderate_discounting_n4",
        "scenario_description": "Moderate time preference (δ=0.97) balancing near and long-term considerations.","experiment_name": "moderate_discounting_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True,
        "discounting": 0.97
    },

    "higher_discounting_n4": {
        **get_base_config_n4(),
        
        "scenario_name": "higher_discounting_n4",
        "scenario_description": "High time preference (δ=0.90) emphasizing near-term considerations.","experiment_name": "higher_discounting_n4",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True,
        "discounting": 0.90
    },
    
    # ========== 5-player scenarios ==========
    
    "weak_governance_n5": {
        **get_base_config_n5(),
        
        "scenario_name": "weak_governance_n5",
        "scenario_description": "Standard 5-player weak governance including hot country H with base temperature 25°C.","experiment_name": "weak_governance_n5",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F", "H"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True
    },
    
    "power_threshold_n5": {
        **get_base_config_n5(),
        
        "scenario_name": "power_threshold_n5",
        "scenario_description": "5-player power threshold scenario with equal power distribution and unanimity requirement.","experiment_name": "power_threshold_n5",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F", "H"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
    
    "power_threshold_no_unanimity_n5": {
        **get_base_config_n5(),
        
        "scenario_name": "power_threshold_no_unanimity_n5",
        "scenario_description": "5-player power threshold without unanimity; H least vulnerable (0.5) to climate impacts.","experiment_name": "power_threshold_no_unanimity_n5",
        "m_damage": {"W": 0.75, "T": 1.25, "C": 1.0, "F": 1.1, "H": 0.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "unequal_power_n5": {
        **get_base_config_n5(),
        
        "scenario_name": "unequal_power_n5",
        "scenario_description": "Gradually declining power distribution across five players from W=30% to H=10%.","experiment_name": "unequal_power_n5",
        "power": {"W": 0.3, "T": 0.25, "C": 0.2, "F": 0.15, "H": 0.1},
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F", "H"]},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": True
    },
    
    "hot_country_advantage_n5": {
        **get_base_config_n5(),
        
        "scenario_name": "hot_country_advantage_n5",
        "scenario_description": "Hot country H has low damage sensitivity (0.5) while others face higher impacts (1.2).","experiment_name": "hot_country_advantage_n5",
        "power": {"W": 0.2, "T": 0.2, "C": 0.2, "F": 0.2, "H": 0.2},
        "m_damage": {"W": 1.2, "T": 1.2, "C": 1.2, "F": 1.2, "H": 0.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "varied_damage_n5": {
        **get_base_config_n5(),
        
        "scenario_name": "varied_damage_n5",
        "scenario_description": "Graduated damage sensitivity with hot country H least vulnerable (0.3) and F most (1.5).","experiment_name": "varied_damage_n5",
        "m_damage": {"W": 0.6, "T": 0.9, "C": 1.2, "F": 1.5, "H": 0.3},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },
    
    "high_discounting_n5": {
        **get_base_config_n5(),

        "scenario_name": "high_discounting_n5",
        "scenario_description": "5-player weak governance with high discounting (δ=0.95) favoring immediate gains.","experiment_name": "high_discounting_n5",
        "m_damage": {player: 1.0 for player in ["W", "T", "C", "F", "H"]},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": True,
        "discounting": 0.95
    },

    # ========== EXPERIMENTAL SCENARIOS FOR CYCLING EXPLORATION ==========

    # 3-player cycling experiments

    "cycle_experiment_1_n3": {
        **get_base_config_n3(),
        "scenario_name": "cycle_experiment_1_n3",
        "scenario_description": "Rock-paper-scissors power structure: nearly equal power (0.34, 0.33, 0.33) with no unanimity to allow majority forcing.",
        "experiment_name": "cycle_experiment_1_n3",
        "power": {"W": 0.34, "T": 0.33, "C": 0.33},
        "m_damage": {"W": 0.75, "T": 1.25, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "cycle_experiment_2_n3": {
        **get_base_config_n3(),
        "scenario_name": "cycle_experiment_2_n3",
        "scenario_description": "Extreme damage asymmetry (W=2.5, T=0.5, C=1.5) with no unanimity, creating conflicting preferences.",
        "experiment_name": "cycle_experiment_2_n3",
        "power": {"W": 0.4, "T": 0.35, "C": 0.25},
        "m_damage": {"W": 2.5, "T": 0.5, "C": 1.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "cycle_experiment_3_n3": {
        **get_base_config_n3(),
        "scenario_name": "cycle_experiment_3_n3",
        "scenario_description": "Very low discounting (δ=0.5) with no unanimity, making players myopic and potentially unstable.",
        "experiment_name": "cycle_experiment_3_n3",
        "m_damage": {"W": 0.75, "T": 1.25, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.5
    },

    "cycle_experiment_4_n3": {
        **get_base_config_n3(),
        "scenario_name": "cycle_experiment_4_n3",
        "scenario_description": "Inverse damage-temperature relationship: cold countries more vulnerable (W=0.5, T=1.0, C=2.0) with no unanimity.",
        "experiment_name": "cycle_experiment_4_n3",
        "power": {"W": 0.35, "T": 0.33, "C": 0.32},
        "m_damage": {"W": 0.5, "T": 1.0, "C": 2.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    # 4-player cycling experiments

    "cycle_experiment_1_n4": {
        **get_base_config_n4(),
        "scenario_name": "cycle_experiment_1_n4",
        "scenario_description": "Balanced power (0.26, 0.26, 0.26, 0.22) with extreme damage variation, no unanimity.",
        "experiment_name": "cycle_experiment_1_n4",
        "power": {"W": 0.26, "T": 0.26, "C": 0.26, "F": 0.22},
        "m_damage": {"W": 2.0, "T": 0.6, "C": 1.4, "F": 0.8},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "cycle_experiment_2_n4": {
        **get_base_config_n4(),
        "scenario_name": "cycle_experiment_2_n4",
        "scenario_description": "Two pairs structure (W-T: 0.27 each, C-F: 0.23 each) with contrasting damage (W-T: 0.7, C-F: 1.8), no unanimity.",
        "experiment_name": "cycle_experiment_2_n4",
        "power": {"W": 0.27, "T": 0.27, "C": 0.23, "F": 0.23},
        "m_damage": {"W": 0.7, "T": 0.7, "C": 1.8, "F": 1.8},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "cycle_experiment_3_n4": {
        **get_base_config_n4(),
        "scenario_name": "cycle_experiment_3_n4",
        "scenario_description": "Very low discounting (δ=0.3) with heterogeneous damage and power, no unanimity.",
        "experiment_name": "cycle_experiment_3_n4",
        "power": {"W": 0.3, "T": 0.28, "C": 0.24, "F": 0.18},
        "m_damage": {"W": 1.5, "T": 0.8, "C": 1.2, "F": 0.6},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.3
    },

    "cycle_experiment_4_n4": {
        **get_base_config_n4(),
        "scenario_name": "cycle_experiment_4_n4",
        "scenario_description": "Low threshold (0.4) with asymmetric power and damage, no unanimity - easier coalition formation.",
        "experiment_name": "cycle_experiment_4_n4",
        "power": {"W": 0.35, "T": 0.25, "C": 0.25, "F": 0.15},
        "m_damage": {"W": 0.6, "T": 1.5, "C": 1.8, "F": 0.9},
        "power_rule": "power_threshold",
        "min_power": 0.4,
        "unanimity_required": False
    },

    "cycle_experiment_5_n4": {
        **get_base_config_n4(),
        "scenario_name": "cycle_experiment_5_n4",
        "scenario_description": "Non-monotonic damage (W=1.2, T=0.6, C=1.8, F=0.4) creating complex preference ordering, no unanimity.",
        "experiment_name": "cycle_experiment_5_n4",
        "power": {"W": 0.28, "T": 0.27, "C": 0.26, "F": 0.19},
        "m_damage": {"W": 1.2, "T": 0.6, "C": 1.8, "F": 0.4},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "cycle_experiment_6_n4": {
        **get_base_config_n4(),
        "scenario_name": "cycle_experiment_6_n4",
        "scenario_description": "Weak governance with very heterogeneous damage (0.3 to 2.5) and low discounting (δ=0.7), no unanimity.",
        "experiment_name": "cycle_experiment_6_n4",
        "m_damage": {"W": 2.5, "T": 0.3, "C": 1.8, "F": 0.9},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": False,
        "discounting": 0.7
    },

    # ========== ULTRA-EXTREME SCENARIOS FOR FORCING CYCLES ==========

    "ultra_cycle_1_n3": {
        **get_base_config_n3(),
        "scenario_name": "ultra_cycle_1_n3",
        "scenario_description": "Ultra-myopic players (δ=0.1), perfectly balanced power, extreme damage asymmetry.",
        "experiment_name": "ultra_cycle_1_n3",
        "power": {"W": 0.334, "T": 0.333, "C": 0.333},
        "m_damage": {"W": 3.0, "T": 0.3, "C": 1.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.1
    },

    "ultra_cycle_2_n3": {
        **get_base_config_n3(),
        "scenario_name": "ultra_cycle_2_n3",
        "scenario_description": "Very myopic (δ=0.3), alternating damage (high-low-high), no unanimity.",
        "experiment_name": "ultra_cycle_2_n3",
        "power": {"W": 0.35, "T": 0.33, "C": 0.32},
        "m_damage": {"W": 2.5, "T": 0.4, "C": 2.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.3
    },

    "ultra_cycle_3_n3": {
        **get_base_config_n3(),
        "scenario_name": "ultra_cycle_3_n3",
        "scenario_description": "Myopic (δ=0.5), very low threshold (0.4), extreme asymmetry.",
        "experiment_name": "ultra_cycle_3_n3",
        "power": {"W": 0.45, "T": 0.35, "C": 0.2},
        "m_damage": {"W": 0.2, "T": 3.0, "C": 1.5},
        "power_rule": "power_threshold",
        "min_power": 0.4,
        "unanimity_required": False,
        "discounting": 0.5
    },

    "ultra_cycle_1_n4": {
        **get_base_config_n4(),
        "scenario_name": "ultra_cycle_1_n4",
        "scenario_description": "Ultra-myopic 4-player (δ=0.1), balanced power, wild damage variation.",
        "experiment_name": "ultra_cycle_1_n4",
        "power": {"W": 0.255, "T": 0.25, "C": 0.25, "F": 0.245},
        "m_damage": {"W": 3.0, "T": 0.2, "C": 2.5, "F": 0.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.1
    },

    "ultra_cycle_2_n4": {
        **get_base_config_n4(),
        "scenario_name": "ultra_cycle_2_n4",
        "scenario_description": "Very myopic (δ=0.2), two dominant pairs, conflicting preferences.",
        "experiment_name": "ultra_cycle_2_n4",
        "power": {"W": 0.3, "T": 0.28, "C": 0.25, "F": 0.17},
        "m_damage": {"W": 0.3, "T": 2.8, "C": 0.4, "F": 2.5},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.2
    },

    "ultra_cycle_3_n4": {
        **get_base_config_n4(),
        "scenario_name": "ultra_cycle_3_n4",
        "scenario_description": "Myopic (δ=0.4), low threshold (0.4), circular damage pattern.",
        "experiment_name": "ultra_cycle_3_n4",
        "power": {"W": 0.28, "T": 0.27, "C": 0.26, "F": 0.19},
        "m_damage": {"W": 2.0, "T": 0.5, "C": 1.5, "F": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.4,
        "unanimity_required": False,
        "discounting": 0.4
    },

    # ========== HETEROGENEOUS IDEAL TEMPERATURES (KEY INSIGHT!) ==========

    "hetero_ideal_1_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_1_n3",
        "scenario_description": "DIFFERENT ideal temperatures: W prefers warm (16°C), T temperate (13°C), C cold (10°C). No unanimity, balanced power.",
        "experiment_name": "hetero_ideal_1_n3",
        "ideal_temp": {"W": 16.0, "T": 13.0, "C": 10.0},
        "power": {"W": 0.34, "T": 0.33, "C": 0.33},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "hetero_ideal_2_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_2_n3",
        "scenario_description": "Extreme ideal temp differences: W=18°C (hot preference), T=13°C, C=8°C (very cold). No unanimity.",
        "experiment_name": "hetero_ideal_2_n3",
        "ideal_temp": {"W": 18.0, "T": 13.0, "C": 8.0},
        "power": {"W": 0.34, "T": 0.33, "C": 0.33},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "hetero_ideal_3_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_3_n3",
        "scenario_description": "Different ideal temps (W=16, T=13, C=10) with heterogeneous damage (creates double asymmetry).",
        "experiment_name": "hetero_ideal_3_n3",
        "ideal_temp": {"W": 16.0, "T": 13.0, "C": 10.0},
        "power": {"W": 0.34, "T": 0.33, "C": 0.33},
        "m_damage": {"W": 0.8, "T": 1.0, "C": 1.2},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "hetero_ideal_4_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_4_n3",
        "scenario_description": "Different ideals (W=17, T=13, C=9) with low discounting (δ=0.7) - myopic + conflicting goals.",
        "experiment_name": "hetero_ideal_4_n3",
        "ideal_temp": {"W": 17.0, "T": 13.0, "C": 9.0},
        "power": {"W": 0.34, "T": 0.33, "C": 0.33},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.7
    },

    "hetero_ideal_5_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_5_n3",
        "scenario_description": "Different ideals (W=16, T=13, C=10), very myopic (δ=0.3), low threshold (0.4) - maximum cycling potential.",
        "experiment_name": "hetero_ideal_5_n3",
        "ideal_temp": {"W": 16.0, "T": 13.0, "C": 10.0},
        "power": {"W": 0.35, "T": 0.33, "C": 0.32},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.4,
        "unanimity_required": False,
        "discounting": 0.3
    },

    "hetero_ideal_6_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_6_n3",
        "scenario_description": "Weak governance + different ideals (W=16, T=13, C=10) - free-driver problem with conflicting goals.",
        "experiment_name": "hetero_ideal_6_n3",
        "ideal_temp": {"W": 16.0, "T": 13.0, "C": 10.0},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
        "power_rule": "weak_governance",
        "min_power": None,
        "unanimity_required": False
    },

    "hetero_ideal_7_n3": {
        **get_base_config_n3(),
        "scenario_name": "hetero_ideal_7_n3",
        "scenario_description": "Moderate different ideals (W=15, T=13, C=11) - subtle but real conflict.",
        "experiment_name": "hetero_ideal_7_n3",
        "ideal_temp": {"W": 15.0, "T": 13.0, "C": 11.0},
        "power": {"W": 0.34, "T": 0.33, "C": 0.33},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    # ========== 4-PLAYER WITH HETEROGENEOUS IDEALS ==========

    "hetero_ideal_1_n4": {
        **get_base_config_n4(),
        "scenario_name": "hetero_ideal_1_n4",
        "scenario_description": "4-player with divergent ideal temps (W=17, T=14, C=11, F=8) - testing complex state space for cycles.",
        "experiment_name": "hetero_ideal_1_n4",
        "ideal_temp": {"W": 17.0, "T": 14.0, "C": 11.0, "F": 8.0},
        "power": {"W": 0.26, "T": 0.26, "C": 0.25, "F": 0.23},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0, "F": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False
    },

    "hetero_ideal_2_n4": {
        **get_base_config_n4(),
        "scenario_name": "hetero_ideal_2_n4",
        "scenario_description": "4-player extreme ideals (W=19, T=15, C=10, F=6) with low discounting (δ=0.5) - maximum cycling potential.",
        "experiment_name": "hetero_ideal_2_n4",
        "ideal_temp": {"W": 19.0, "T": 15.0, "C": 10.0, "F": 6.0},
        "power": {"W": 0.26, "T": 0.26, "C": 0.25, "F": 0.23},
        "m_damage": {"W": 1.0, "T": 1.0, "C": 1.0, "F": 1.0},
        "power_rule": "power_threshold",
        "min_power": 0.501,
        "unanimity_required": False,
        "discounting": 0.5
    },
}


def get_scenario(name):
    """
    Get a scenario configuration by name.
    
    Args:
        name: Scenario name
        
    Returns:
        Scenario configuration dictionary
        
    Raises:
        KeyError: If scenario name not found
    """
    if name not in SCENARIOS:
        available = ', '.join(sorted(SCENARIOS.keys()))
        raise KeyError(f"Unknown scenario '{name}'. Available scenarios: {available}")
    return SCENARIOS[name].copy()


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
