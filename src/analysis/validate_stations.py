"""Validate interpolated surface against held-out NOAA station readings.

This is the gold-standard validation: ERA5 reanalysis is a model itself,
so we check if our interpolated surface (built from ERA5 samples) actually
agrees with independent ground-truth NOAA station readings.
"""
import numpy as np
import pandas as pd
from src.config import DATA_PROCESSED, DATA_INTERIM
from src.analysis.interpolation import idw_interpolate, kriging_interpolate, loocv_metrics


def main():
    stations = pd.read_parquet(DATA_PROCESSED / "stations_summary.parquet")
    grid_samples = pd.read_parquet(DATA_PROCESSED / "grid_summary.parquet")
    grid_samples = grid_samples.dropna(subset=["ws10_mean"]).reset_index(drop=True)

    # Use grid samples to predict at station locations
    src_lat = grid_samples["lat"].to_numpy()
    src_lon = grid_samples["lon"].to_numpy()
    src_ws = grid_samples["ws10_mean"].to_numpy()

    target_lat = stations["lat"].to_numpy().reshape(-1, 1)
    target_lon = stations["lon"].to_numpy().reshape(-1, 1)

    # IDW prediction at station locations
    idw_pred = idw_interpolate(
        src_lat, src_lon, src_ws, target_lat, target_lon, power=2.0, k=8
    ).ravel()

    # Kriging prediction
    krg_pred, krg_var = kriging_interpolate(
        src_lat, src_lon, src_ws, target_lat, target_lon
    )
    krg_pred = krg_pred.ravel()

    df = stations[["station_id", "station_name", "lat", "lon", "ws10_mean"]].copy()
    df = df.rename(columns={"ws10_mean": "actual"})
    df["idw_pred"] = idw_pred
    df["krg_pred"] = krg_pred
    df["idw_residual"] = df["actual"] - df["idw_pred"]
    df["krg_residual"] = df["actual"] - df["krg_pred"]

    # Metrics
    idw_metrics = loocv_metrics(
        df.rename(columns={"idw_pred": "predicted", "idw_residual": "residual"})
    )
    krg_metrics = loocv_metrics(
        df.rename(columns={"krg_pred": "predicted", "krg_residual": "residual"})
    )

    print("=" * 60)
    print("EXTERNAL VALIDATION: predicted vs actual NOAA stations")
    print("=" * 60)
    print(f"\nIDW:     RMSE={idw_metrics['rmse']:.3f}  MAE={idw_metrics['mae']:.3f}  "
          f"R²={idw_metrics['r2']:.3f}  bias={idw_metrics['bias']:+.3f}")
    print(f"Kriging: RMSE={krg_metrics['rmse']:.3f}  MAE={krg_metrics['mae']:.3f}  "
          f"R²={krg_metrics['r2']:.3f}  bias={krg_metrics['bias']:+.3f}")

    # Note: stations report at ~10m AGL but ERA5 wind_speed_10m is also ~10m,
    # so direct comparison is fair. ERA5 will systematically smooth peaks though.

    df.to_csv(DATA_INTERIM / "station_validation.csv", index=False)
    pd.DataFrame([
        {"method": "idw_external", **idw_metrics},
        {"method": "kriging_external", **krg_metrics},
    ]).to_csv(DATA_INTERIM / "external_validation_metrics.csv", index=False)
    print(f"\nSaved to {DATA_INTERIM}/station_validation.csv")


if __name__ == "__main__":
    main()