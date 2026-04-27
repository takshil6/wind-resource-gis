"""Fetch ERA5 wind data on a regular grid covering the study region."""
import time
import numpy as np
import pandas as pd
from src.config import BBOX, GRID_RESOLUTION, DATA_RAW
from src.data.wind_fetcher import fetch_point


def make_grid(bbox: dict = BBOX, res: float = GRID_RESOLUTION) -> pd.DataFrame:
    """Build a regular lat/lon grid over the bbox."""
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + res, res)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + res, res)
    grid = pd.DataFrame(
        [(lat, lon) for lat in lats for lon in lons],
        columns=["lat", "lon"],
    )
    grid["grid_id"] = grid.index.map(lambda i: f"g{i:05d}")
    return grid


def fetch_grid_summary(
    grid: pd.DataFrame,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Pull yearly summary stats for each grid point.

    Returns one row per grid cell with annual mean wind speed,
    Weibull-ready stats, etc. Avoids storing 8760 hours × N grid points.
    """
    if sample_n:
        grid = grid.sample(n=sample_n, random_state=42).reset_index(drop=True)

    rows = []
    for i, row in grid.iterrows():
        try:
            df = fetch_point(row["lat"], row["lon"])
            ws10 = df["wind_speed_10m"].dropna()
            ws100 = df["wind_speed_100m"].dropna()
            rows.append({
                "grid_id": row["grid_id"],
                "lat": row["lat"],
                "lon": row["lon"],
                "ws10_mean": ws10.mean(),
                "ws10_std": ws10.std(),
                "ws10_p50": ws10.median(),
                "ws10_p90": ws10.quantile(0.9),
                "ws100_mean": ws100.mean(),
                "ws100_std": ws100.std(),
                "n_hours": len(df),
            })
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(grid)} grid points done")
        except Exception as e:
            print(f"  grid {row['grid_id']} FAILED: {e}")
        time.sleep(0.5)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    grid = make_grid()
    print(f"Grid size: {len(grid)} points (full region)")

    # Phase 1: sample subset so it finishes in reasonable time
    # Phase 4: bump up or run overnight for full coverage
    summary = fetch_grid_summary(grid, sample_n=80)

    out = DATA_RAW / "grid_wind_summary.parquet"
    summary.to_parquet(out, index=False)
    print(f"\nSaved {len(summary)} grid points to {out}")
    print(summary[["lat", "lon", "ws10_mean", "ws100_mean"]].head())