"""Fetch hourly wind data from Open-Meteo (ERA5-backed historical API).

Why Open-Meteo: free, no API key, ERA5 reanalysis under the hood, handles
caching and retries cleanly. NOAA's direct ISD-Lite parser works but is
slower for our use case (FTP, fixed-width format, station-by-station).
"""
import time
from pathlib import Path
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests
from src.config import DATA_RAW, DATE_START, DATE_END

# Cached, retrying session — re-runs are instant after first fetch
_cache = requests_cache.CachedSession(
    str(DATA_RAW / ".openmeteo_cache"), expire_after=-1
)
_session = retry(_cache, retries=5, backoff_factor=0.2)
_client = openmeteo_requests.Client(session=_session)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

WIND_VARS = [
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
]


def fetch_point(
    lat: float,
    lon: float,
    start: str = DATE_START,
    end: str = DATE_END,
) -> pd.DataFrame:
    """Get hourly wind data for a single lat/lon."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(WIND_VARS),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    responses = _client.weather_api(API_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        **{
            var: hourly.Variables(i).ValuesAsNumpy()
            for i, var in enumerate(WIND_VARS)
        },
    })
    df["lat"] = lat
    df["lon"] = lon
    return df


def fetch_for_stations(
    stations_df: pd.DataFrame,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    """Fetch wind data at each station's coordinates.

    Returns a long-format dataframe: one row per (station, timestamp).
    """
    frames = []
    total = len(stations_df)
    for i, (_, row) in enumerate(stations_df.iterrows()):
        try:
            df = fetch_point(row["LAT"], row["LON"])
            df["station_id"] = row["station_id"]
            df["station_name"] = row["STATION NAME"]
            df["elevation_m"] = row.get("ELEV(M)")
            frames.append(df)
            print(f"  [{i+1}/{total}] {row['station_id']} — {len(df)} hours")
        except Exception as e:
            print(f"  [{i+1}/{total}] {row['station_id']} FAILED: {e}")
        time.sleep(sleep_s)

    if not frames:
        raise RuntimeError("No data fetched for any station")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    stations = pd.read_csv(DATA_RAW / "stations_in_region.csv")
    # Sample to keep first run fast — bump up later
    sample = stations.sample(n=min(15, len(stations)), random_state=42)
    print(f"Fetching wind data for {len(sample)} stations...")

    data = fetch_for_stations(sample)
    out = DATA_RAW / "station_wind_hourly.parquet"
    data.to_parquet(out, index=False)
    print(f"\nSaved {len(data):,} rows to {out}")
    print(data.head())