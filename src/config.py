"""Project-wide configuration."""
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"
MAPS_DIR = OUTPUTS / "maps"

# Study region: Northeast US (lat_min, lon_min, lat_max, lon_max)
# Covers MA, RI, CT, eastern NY, southern NH/VT, northern NJ
REGION_NAME = "northeast_us"
BBOX = {
    "lat_min": 40.5,
    "lat_max": 43.5,
    "lon_min": -74.5,
    "lon_max": -70.5,
}

# Alternative regions (for easy swap during interview demo)
REGIONS = {
    "northeast_us": {
        "lat_min": 40.5, "lat_max": 43.5,
        "lon_min": -74.5, "lon_max": -70.5,
    },
    "alabama": {  # Birmingham focus - Accelerate Wind CEO location
        "lat_min": 32.5, "lat_max": 35.0,
        "lon_min": -88.5, "lon_max": -85.0,
    },
}

# Date range for historical analysis
DATE_START = "2023-01-01"
DATE_END = "2023-12-31"

# Wind turbine parameters (Accelerate Wind-style edge-of-roof turbine)
HUB_HEIGHT_M = 10.0       # rooftop-mounted, low hub height
ROTOR_DIAMETER_M = 1.5    # small distributed turbine
RATED_POWER_KW = 1.5      # typical small wind
CUT_IN_SPEED = 2.5        # m/s
RATED_SPEED = 11.0        # m/s
CUT_OUT_SPEED = 25.0      # m/s

# Air density (kg/m³) at sea level, 15°C
AIR_DENSITY = 1.225

# Interpolation grid resolution (degrees)
GRID_RESOLUTION = 0.05    # ~5km

# Suitability scoring weights
SUITABILITY_WEIGHTS = {
    "wind_speed": 0.6,
    "consistency": 0.25,   # low variance is good
    "elevation_bonus": 0.15,
}