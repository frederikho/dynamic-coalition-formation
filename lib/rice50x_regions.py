"""
RICE50x region codes and country names.

The keys are the lowercase GDX region codes used in RICE50x model output files.
The values are the full country/region names.

Player names in this framework use the uppercase version of the GDX code
(e.g. 'nde' → player name 'NDE', 'usa' → 'USA').
"""

# Mapping: GDX code (lowercase) → full country/region name
RICE50X_REGIONS = {
    "arg":    "Argentina",
    "aus":    "Australia",
    "aut":    "Austria",
    "bel":    "Belgium",
    "bgr":    "Bulgaria",
    "blt":    "Baltic States",
    "bra":    "Brazil",
    "can":    "Canada",
    "chl":    "Chile",
    "chn":    "China",
    "cor":    "South Korea",
    "cro":    "Croatia",
    "dnk":    "Denmark",
    "egy":    "Egypt",
    "esp":    "Spain",
    "fin":    "Finland",
    "fra":    "France",
    "gbr":    "United Kingdom",
    "golf57": "Rest of Gulf countries",
    "grc":    "Greece",
    "hun":    "Hungary",
    "idn":    "Indonesia",
    "irl":    "Ireland",
    "ita":    "Italy",
    "jpn":    "Japan",
    "meme":   "Mediterranean (excl. Gulf countries)",
    "mex":    "Mexico",
    "mys":    "Malaysia",
    "nde":    "India",
    "nld":    "Netherlands",
    "noan":   "Morocco-Tunisia",
    "noap":   "Algeria-Libya",
    "nor":    "Norway",
    "oeu":    "Rest of Europe",
    "osea":   "Rest of South-East Asia",
    "pol":    "Poland",
    "prt":    "Portugal",
    "rcam":   "Rest of Central America and Caribbean",
    "rcz":    "Czechia",
    "rfa":    "Germany",
    "ris":    "Rest of CIS",
    "rjan57": "Rest of Pacific",
    "rom":    "Romania",
    "rsaf":   "Rest of Sub-Saharan Africa",
    "rsam":   "Rest of South America",
    "rsas":   "Rest of South Asia",
    "rsl":    "Slovakia",
    "rus":    "Russia",
    "sau":    "Saudi Arabia",
    "slo":    "Slovenia",
    "sui":    "Switzerland",
    "swe":    "Sweden",
    "tha":    "Thailand",
    "tur":    "Turkey",
    "ukr":    "Ukraine",
    "usa":    "United States",
    "vnm":    "Vietnam",
    "zaf":    "South Africa",
}

# Player name (uppercase code) → full country/region name
RICE50X_PLAYER_NAMES = {code.upper(): name for code, name in RICE50X_REGIONS.items()}


def get_country_name(player: str) -> str:
    """Return the full country name for a player name (RICE code, case-insensitive).

    Falls back to the player name itself if not found (e.g. for W, T, C, F, H).
    """
    return RICE50X_PLAYER_NAMES.get(player.upper(), player)
