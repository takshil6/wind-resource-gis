"""Composite suitability scoring for wind turbine site assessment.

Produces a 0-100 score per grid cell from normalized features.
Configurable weights via src.config.SUITABILITY_WEIGHTS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import SUITABILITY_WEIGHTS


def normalize(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Min-max normalize to [0, 1], clipping outliers."""
    if lo is None:
        lo = np.nanpercentile(arr, 5)
    if hi is None:
        hi = np.nanpercentile(arr, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def consistency_score(mean_v: np.ndarray, std_v: np.ndarray) -> np.ndarray:
    """Lower coefficient of variation = higher score.

    Steady wind is more valuable than gusty wind for turbine economics.
    """
    cv = np.where(mean_v > 0, std_v / mean_v, np.nan)
    # Invert: low CV = high score. Typical CV range 0.3-0.8.
    return 1 - normalize(cv, lo=0.3, hi=0.8)


def coastal_score(distance_km: np.ndarray, decay_km: float = 30.0) -> np.ndarray:
    """Exponential decay with distance from coast.

    Within ~30km of coast: meaningful sea-breeze contribution.
    """
    return np.exp(-distance_km / decay_km)


def elevation_score(
    elevation: np.ndarray, ideal_low: float = 50, ideal_high: float = 400
) -> np.ndarray:
    """Bonus for elevated, exposed terrain — penalty for valleys/peaks.

    Mid-range ridges (50-400m) score highest. Sea level OK (coastal),
    very high elevation penalized (turbulence + access).
    """
    score = np.where(
        np.isnan(elevation),
        0.5,  # neutral if missing
        np.where(
            elevation < ideal_low,
            normalize(elevation, lo=-10, hi=ideal_low) * 0.7,
            np.where(
                elevation <= ideal_high,
                1.0,
                np.maximum(0, 1 - (elevation - ideal_high) / 600),
            ),
        ),
    )
    return score


def composite_score(
    wind_speed: np.ndarray,
    wind_std: np.ndarray,
    elevation: np.ndarray,
    coast_distance_km: np.ndarray,
    weights: dict | None = None,
) -> dict[str, np.ndarray]:
    """Combine features into composite 0-100 suitability score.

    Returns dict with the score plus each component for transparency.
    """
    w = weights or SUITABILITY_WEIGHTS

    f_wind = normalize(wind_speed, lo=3.0, hi=8.0)
    f_consistency = consistency_score(wind_speed, wind_std)
    f_elevation = elevation_score(elevation)
    f_coast = coastal_score(coast_distance_km)

    # Coastal bonus is multiplicative on elevation_bonus weight
    elev_blended = 0.6 * f_elevation + 0.4 * f_coast

    score = (
        w["wind_speed"] * f_wind
        + w["consistency"] * f_consistency
        + w["elevation_bonus"] * elev_blended
    )
    score_100 = np.clip(score * 100, 0, 100)

    return {
        "score": score_100,
        "f_wind": f_wind,
        "f_consistency": f_consistency,
        "f_elevation": f_elevation,
        "f_coast": f_coast,
    }


def rank_top_sites(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    score: np.ndarray,
    wind_speed: np.ndarray,
    capacity_factor: np.ndarray,
    n: int = 25,
) -> pd.DataFrame:
    """Return top-N sites by suitability score as a dataframe."""
    df = pd.DataFrame({
        "lat": grid_lat.ravel(),
        "lon": grid_lon.ravel(),
        "score": score.ravel(),
        "wind_speed_ms": wind_speed.ravel(),
        "capacity_factor": capacity_factor.ravel(),
    })
    df = df.dropna().sort_values("score", ascending=False).head(n).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df