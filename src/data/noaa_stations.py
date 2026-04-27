"""Find NOAA ISD weather stations within the study region."""
import pandas as pd
import requests
from pathlib import Path
from src.config import BBOX, DATA_RAW

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"


def fetch_station_inventory(force_refresh: bool = False) -> pd.DataFrame:
    """Download the NOAA ISD station inventory (~30K stations globally)."""
    cache = DATA_RAW / "isd-history.csv"
    if cache.exists() and not force_refresh:
        return pd.read_csv(cache, dtype={"USAF": str, "WBAN": str})

    print("Downloading NOAA ISD station inventory...")
    r = requests.get(ISD_HISTORY_URL, timeout=60)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return pd.read_csv(cache, dtype={"USAF": str, "WBAN": str})


def filter_stations_in_region(
    stations: pd.DataFrame,
    bbox: dict = BBOX,
    min_end_year: int = 2023,
) -> pd.DataFrame:
    """Filter to stations active in our region with recent data."""
    stations = stations.copy()
    stations["LAT"] = pd.to_numeric(stations["LAT"], errors="coerce")
    stations["LON"] = pd.to_numeric(stations["LON"], errors="coerce")
    stations["END"] = pd.to_datetime(stations["END"], format="%Y%m%d", errors="coerce")

    mask = (
        stations["LAT"].between(bbox["lat_min"], bbox["lat_max"])
        & stations["LON"].between(bbox["lon_min"], bbox["lon_max"])
        & (stations["CTRY"] == "US")
        & (stations["END"].dt.year >= min_end_year)
        & stations["LAT"].notna()
        & stations["LON"].notna()
    )
    filtered = stations[mask].copy()
    filtered["station_id"] = filtered["USAF"] + "-" + filtered["WBAN"]
    return filtered.reset_index(drop=True)


if __name__ == "__main__":
    inv = fetch_station_inventory()
    print(f"Total stations globally: {len(inv):,}")

    regional = filter_stations_in_region(inv)
    print(f"Stations in region: {len(regional)}")
    print(regional[["station_id", "STATION NAME", "STATE", "LAT", "LON", "ELEV(M)"]].head(15))

    out = DATA_RAW / "stations_in_region.csv"
    regional.to_csv(out, index=False)
    print(f"\nSaved to {out}")