"""Phase 3 pipeline: power surfaces + suitability scoring + top sites."""
import numpy as np
import pandas as pd
from src.config import DATA_INTERIM, DATA_PROCESSED, HUB_HEIGHT_M
from src.analysis.power import (
    power_density_weibull,
    estimate_weibull_from_mean_std,
    capacity_factor_from_weibull,
)
from src.analysis.geo_features import distance_to_coast_km
from src.analysis.suitability import composite_score, rank_top_sites


def main():
    # Load interpolation surfaces
    s = np.load(DATA_INTERIM / "interpolation_surfaces.npz")
    grid_lat = s["grid_lat"]
    grid_lon = s["grid_lon"]

    # Use kriging surface (won validation)
    ws10 = s["krg_ws10"]
    ws100 = s["krg_ws100"]

    # Estimate std at grid cells from input data spread (approximation)
    # Better: re-run kriging on std as a separate target. For Phase 3 we
    # use a typical CV from the source data.
    grid_summary = pd.read_parquet(DATA_PROCESSED / "grid_summary.parquet")
    typical_cv = (grid_summary["ws10_std"] / grid_summary["ws10_mean"]).median()
    ws10_std = ws10 * typical_cv
    print(f"Using typical CV = {typical_cv:.3f}")

    # Weibull params + capacity factor at hub height
    print("Computing Weibull params...")
    if HUB_HEIGHT_M >= 80:
        ws_hub = ws100
        ws_hub_std = ws100 * typical_cv
    else:
        ws_hub = ws10
        ws_hub_std = ws10_std

    c, k = estimate_weibull_from_mean_std(ws_hub, ws_hub_std)

    print("Computing power density...")
    wpd = power_density_weibull(c, k)

    print("Computing capacity factor (this is the slow one)...")
    cf = capacity_factor_from_weibull(c, k)

    # Elevation: interpolate from grid_summary onto target grid
    print("Interpolating elevation onto target grid...")
    from src.analysis.interpolation import idw_interpolate
    elev_pts = grid_summary.dropna(subset=["elevation_m"])
    if len(elev_pts) >= 5:
        elevation = idw_interpolate(
            elev_pts["lat"].to_numpy(),
            elev_pts["lon"].to_numpy(),
            elev_pts["elevation_m"].to_numpy(),
            grid_lat, grid_lon, power=2.0, k=6,
        )
    else:
        print("  Warning: <5 elevation points, using zeros")
        elevation = np.zeros_like(grid_lat)

    # Coastal distance
    print("Computing coastal distance...")
    coast_dist = distance_to_coast_km(grid_lat, grid_lon)

    # Composite suitability
    print("Computing composite suitability score...")
    result = composite_score(ws_hub, ws_hub_std, elevation, coast_dist)

    # Land mask: zero out offshore cells (Accelerate Wind = rooftop = land only)
    print("Applying land mask...")
    from src.analysis.geo_features import land_mask
    on_land = land_mask(grid_lat, grid_lon)
    print(f"  {on_land.sum()}/{on_land.size} cells on land "
          f"({100*on_land.sum()/on_land.size:.0f}%)")
    score_land_only = np.where(on_land, result["score"], np.nan)

    # Top sites (land-only)
    top = rank_top_sites(
        grid_lat, grid_lon, score_land_only, ws_hub, cf, n=25
    )
    print("\nTop 10 sites by suitability:")
    print(top.head(10).to_string(index=False))

    # Save
    np.savez(
        DATA_INTERIM / "suitability_surfaces.npz",
        grid_lat=grid_lat, grid_lon=grid_lon,
        ws_hub=ws_hub, wpd=wpd, capacity_factor=cf,
        elevation=elevation, coast_dist_km=coast_dist,
        score=result["score"], score_land_only=score_land_only,
        on_land=on_land,
        f_wind=result["f_wind"], f_consistency=result["f_consistency"],
        f_elevation=result["f_elevation"], f_coast=result["f_coast"],
    )
    top.to_csv(DATA_INTERIM / "top_sites.csv", index=False)

    # Summary stats (land cells only)
    cf_land = np.where(on_land, cf, np.nan)
    print(f"\nWind power density (W/m²): "
          f"min={np.nanmin(wpd):.1f}, mean={np.nanmean(wpd):.1f}, "
          f"max={np.nanmax(wpd):.1f}")
    print(f"Capacity factor (%, land only): "
          f"min={100*np.nanmin(cf_land):.1f}, mean={100*np.nanmean(cf_land):.1f}, "
          f"max={100*np.nanmax(cf_land):.1f}")
    print(f"Suitability score (land only): "
          f"min={np.nanmin(score_land_only):.1f}, "
          f"mean={np.nanmean(score_land_only):.1f}, "
          f"max={np.nanmax(score_land_only):.1f}")

    print(f"\nSaved suitability surfaces to {DATA_INTERIM}/suitability_surfaces.npz")
    print(f"Saved top sites to {DATA_INTERIM}/top_sites.csv")

if __name__ == "__main__":
    main()