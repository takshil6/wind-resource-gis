"""Fetch elevation from USGS Elevation Point Query Service (EPQS)."""
import time
import pandas as pd
import requests
from src.config import DATA_RAW

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"


def fetch_elevation(lat: float, lon: float) -> float | None:
    """Single point elevation lookup. Returns meters or None."""
    try:
        r = requests.get(
            EPQS_URL,
            params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326},
            timeout=10,
        )
        r.raise_for_status()
        return float(r.json()["value"])
    except Exception:
        return None


def enrich_with_elevation(df: pd.DataFrame, sleep_s: float = 0.05) -> pd.DataFrame:
    """Add elevation_m column to a dataframe with lat/lon."""
    df = df.copy()
    elevs = []
    for i, row in df.iterrows():
        elevs.append(fetch_elevation(row["lat"], row["lon"]))
        if (i + 1) % 20 == 0:
            print(f"  elevation: {i+1}/{len(df)}")
        time.sleep(sleep_s)
    df["elevation_m"] = elevs
    return df


if __name__ == "__main__":
    grid = pd.read_parquet(DATA_RAW / "grid_wind_summary.parquet")
    print(f"Enriching {len(grid)} grid points with elevation...")
    enriched = enrich_with_elevation(grid)

    out = DATA_RAW / "grid_with_elevation.parquet"
    enriched.to_parquet(out, index=False)
    print(f"\nSaved to {out}")
    print(enriched[["lat", "lon", "ws10_mean", "elevation_m"]].head())
    print(f"\nMissing elevations: {enriched['elevation_m'].isna().sum()}")