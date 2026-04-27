"""Spatial interpolation methods for wind speed surfaces.

Implements:
- IDW (Inverse Distance Weighting): fast baseline
- Ordinary Kriging: statistically rigorous, with uncertainty
- Validation: Leave-One-Out Cross-Validation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pykrige.ok import OrdinaryKriging
from src.config import BBOX, GRID_RESOLUTION


def make_target_grid(
    bbox: dict = BBOX,
    res: float = GRID_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular grid of (lat, lon) for interpolation output."""
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + res, res)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + res, res)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    return grid_lat, grid_lon


def idw_interpolate(
    points_lat: np.ndarray,
    points_lon: np.ndarray,
    values: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    power: float = 2.0,
    k: int = 8,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation.

    Args:
        points_lat/lon/values: known data points
        target_lat/lon: where to predict (can be 2D meshgrid)
        power: distance weighting exponent (2 is standard)
        k: number of nearest neighbors to use

    Returns:
        Predicted values matching target_lat shape.
    """
    src_pts = np.column_stack([points_lat, points_lon])
    tree = cKDTree(src_pts)

    target_shape = target_lat.shape
    target_pts = np.column_stack([target_lat.ravel(), target_lon.ravel()])

    distances, indices = tree.query(target_pts, k=min(k, len(values)))

    # Handle exact-match points (distance == 0)
    distances = np.where(distances < 1e-10, 1e-10, distances)
    weights = 1.0 / (distances ** power)
    weights /= weights.sum(axis=1, keepdims=True)

    predicted = np.sum(values[indices] * weights, axis=1)
    return predicted.reshape(target_shape)


def kriging_interpolate(
    points_lat: np.ndarray,
    points_lon: np.ndarray,
    values: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    variogram_model: str = "spherical",
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary Kriging interpolation with uncertainty.

    Returns:
        (predicted, variance) — both matching target_lat shape.
    """
    OK = OrdinaryKriging(
        points_lon,  # pykrige uses (x=lon, y=lat) order
        points_lat,
        values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )

    target_shape = target_lat.shape
    # Use 'points' mode with flattened arrays (works for any shape)
    predicted, variance = OK.execute(
        "points",
        target_lon.ravel(),
        target_lat.ravel(),
    )
    return predicted.reshape(target_shape), variance.reshape(target_shape)


def loocv(
    points_lat: np.ndarray,
    points_lon: np.ndarray,
    values: np.ndarray,
    method: str = "idw",
    **kwargs,
) -> pd.DataFrame:
    """Leave-One-Out Cross-Validation.

    For each point, predict it using all others, compare to actual.
    Returns dataframe with predictions, residuals, and summary stats.
    """
    n = len(values)
    predicted = np.zeros(n)

    for i in range(n):
        mask = np.arange(n) != i
        if method == "idw":
            pred = idw_interpolate(
                points_lat[mask], points_lon[mask], values[mask],
                np.array([[points_lat[i]]]),
                np.array([[points_lon[i]]]),
                **kwargs,
            )
            predicted[i] = pred[0, 0]
        elif method == "kriging":
            try:
                pred, _ = kriging_interpolate(
                    points_lat[mask], points_lon[mask], values[mask],
                    np.array([[points_lat[i]]]),
                    np.array([[points_lon[i]]]),
                    **kwargs,
                )
                predicted[i] = pred[0, 0]
            except Exception:
                predicted[i] = np.nan
        else:
            raise ValueError(f"Unknown method: {method}")

    df = pd.DataFrame({
        "actual": values,
        "predicted": predicted,
        "residual": values - predicted,
    })
    return df


def loocv_metrics(df: pd.DataFrame) -> dict:
    """Compute RMSE, MAE, R² from LOOCV results."""
    df = df.dropna()
    residuals = df["residual"]
    actual = df["actual"]
    predicted = df["predicted"]

    rmse = float(np.sqrt((residuals ** 2).mean()))
    mae = float(residuals.abs().mean())
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    bias = float(residuals.mean())

    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias, "n": len(df)}