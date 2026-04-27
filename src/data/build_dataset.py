"""Consolidate raw data into clean, analysis-ready datasets."""
import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED


def build_station_summary() -> pd.DataFrame:
    """Aggregate hourly station data into per-station annual stats."""
    hourly = pd.read_parquet(DATA_RAW / "station_wind_hourly.parquet")

    summary = (
        hourly.groupby(["station_id", "station_name", "lat", "lon"])
        .agg(
            ws10_mean=("wind_speed_10m", "mean"),
            ws10_std=("wind_speed_10m", "std"),
            ws10_max=("wind_speed_10m", "max"),
            ws100_mean=("wind_speed_100m", "mean"),
            ws100_std=("wind_speed_100m", "std"),
            wd10_mean=("wind_direction_10m", "mean"),
            n_hours=("wind_speed_10m", "size"),
        )
        .reset_index()
    )
    return summary


def build_grid_dataset() -> pd.DataFrame:
    """Grid summary + elevation, cleaned."""
    df = pd.read_parquet(DATA_RAW / "grid_with_elevation.parquet")
    df = df.dropna(subset=["ws10_mean", "lat", "lon"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    stations = build_station_summary()
    stations.to_parquet(DATA_PROCESSED / "stations_summary.parquet", index=False)
    print(f"Stations summary: {len(stations)} rows")
    print(stations.describe()[["ws10_mean", "ws100_mean"]])

    grid = build_grid_dataset()
    grid.to_parquet(DATA_PROCESSED / "grid_summary.parquet", index=False)
    print(f"\nGrid summary: {len(grid)} rows")
    print(grid.describe()[["ws10_mean", "ws100_mean", "elevation_m"]])