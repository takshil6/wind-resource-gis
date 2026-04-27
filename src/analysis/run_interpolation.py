"""Run interpolation pipeline: IDW + Kriging + validation."""
import numpy as np
import pandas as pd
from src.config import DATA_PROCESSED, DATA_INTERIM
from src.analysis.interpolation import (
    make_target_grid,
    idw_interpolate,
    kriging_interpolate,
    loocv,
    loocv_metrics,
)


def main():
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    # Load: combine station + grid points as our "known" data
    stations = pd.read_parquet(DATA_PROCESSED / "stations_summary.parquet")
    grid_samples = pd.read_parquet(DATA_PROCESSED / "grid_summary.parquet")

    # Use ERA5 grid samples as primary input — denser, more uniform
    # Stations are reserved for validation later
    pts = grid_samples.dropna(subset=["ws10_mean", "ws100_mean"]).reset_index(drop=True)
    print(f"Interpolating from {len(pts)} known points")

    lats = pts["lat"].to_numpy()
    lons = pts["lon"].to_numpy()
    ws10 = pts["ws10_mean"].to_numpy()
    ws100 = pts["ws100_mean"].to_numpy()

    # Build target grid
    grid_lat, grid_lon = make_target_grid()
    print(f"Target grid: {grid_lat.shape} = {grid_lat.size:,} cells")

    # ---- IDW ----
    print("\n[1/3] Running IDW...")
    idw_ws10 = idw_interpolate(lats, lons, ws10, grid_lat, grid_lon, power=2.0, k=8)
    idw_ws100 = idw_interpolate(lats, lons, ws100, grid_lat, grid_lon, power=2.0, k=8)

    # ---- Kriging ----
    print("[2/3] Running Ordinary Kriging...")
    krg_ws10, krg_var10 = kriging_interpolate(lats, lons, ws10, grid_lat, grid_lon)
    krg_ws100, _ = kriging_interpolate(lats, lons, ws100, grid_lat, grid_lon)

    # ---- LOOCV ----
    print("[3/3] Running LOOCV validation...")
    idw_cv = loocv(lats, lons, ws10, method="idw", power=2.0, k=8)
    idw_metrics = loocv_metrics(idw_cv)
    print(f"  IDW:     RMSE={idw_metrics['rmse']:.3f} m/s, "
          f"MAE={idw_metrics['mae']:.3f}, R²={idw_metrics['r2']:.3f}")

    krg_cv = loocv(lats, lons, ws10, method="kriging")
    krg_metrics = loocv_metrics(krg_cv)
    print(f"  Kriging: RMSE={krg_metrics['rmse']:.3f} m/s, "
          f"MAE={krg_metrics['mae']:.3f}, R²={krg_metrics['r2']:.3f}")

    # Pick winner
    winner = "kriging" if krg_metrics["rmse"] < idw_metrics["rmse"] else "idw"
    print(f"\n  Winner: {winner.upper()}")

    # Save everything
    np.savez(
        DATA_INTERIM / "interpolation_surfaces.npz",
        grid_lat=grid_lat, grid_lon=grid_lon,
        idw_ws10=idw_ws10, idw_ws100=idw_ws100,
        krg_ws10=krg_ws10, krg_ws100=krg_ws100,
        krg_var10=krg_var10,
    )

    pd.DataFrame([
        {"method": "idw", **idw_metrics},
        {"method": "kriging", **krg_metrics},
    ]).to_csv(DATA_INTERIM / "validation_metrics.csv", index=False)

    idw_cv.to_csv(DATA_INTERIM / "loocv_idw.csv", index=False)
    krg_cv.to_csv(DATA_INTERIM / "loocv_kriging.csv", index=False)

    print(f"\nSaved surfaces and validation to {DATA_INTERIM}")


if __name__ == "__main__":
    main()