"""Generate Phase 2 diagnostic plots: kriging uncertainty map + LOOCV scatter."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[2]
DATA_INTERIM = ROOT / "data/interim"
MAPS_OUT     = ROOT / "outputs/maps"
MAPS_OUT.mkdir(parents=True, exist_ok=True)


def plot_kriging_uncertainty():
    npz = np.load(DATA_INTERIM / "interpolation_surfaces.npz")
    grid_lat = npz["grid_lat"]
    grid_lon = npz["grid_lon"]
    krg_ws10 = npz["krg_ws10"]
    krg_var10 = npz["krg_var10"]
    krg_std10 = np.sqrt(np.maximum(krg_var10, 0))

    # Load known point locations for overlay
    from src.config import DATA_PROCESSED
    pts = pd.read_parquet(DATA_PROCESSED / "grid_summary.parquet").dropna(subset=["ws10_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean wind speed surface
    ax = axes[0]
    im = ax.pcolormesh(grid_lon, grid_lat, krg_ws10, cmap="YlOrRd", shading="auto")
    ax.scatter(pts["lon"], pts["lat"], s=10, c="black", alpha=0.5, label="Known pts")
    plt.colorbar(im, ax=ax, label="Mean wind speed (m/s)")
    ax.set_title("Kriging — Mean Wind Speed at 10 m")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=7)

    # Uncertainty (std dev)
    ax = axes[1]
    im2 = ax.pcolormesh(grid_lon, grid_lat, krg_std10, cmap="Blues", shading="auto")
    ax.scatter(pts["lon"], pts["lat"], s=10, c="red", alpha=0.5, label="Known pts")
    plt.colorbar(im2, ax=ax, label="Std dev (m/s)")
    ax.set_title("Kriging — Uncertainty (σ)\nLow near stations, high in gaps")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=7)

    plt.tight_layout()
    out = MAPS_OUT / "kriging_uncertainty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_validation_scatter():
    idw = pd.read_csv(DATA_INTERIM / "loocv_idw.csv")
    krg = pd.read_csv(DATA_INTERIM / "loocv_kriging.csv")
    metrics = pd.read_csv(DATA_INTERIM / "validation_metrics.csv").set_index("method")
    ext = pd.read_csv(DATA_INTERIM / "external_validation_metrics.csv").set_index("method")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, df, label, key in [
        (axes[0], idw,  "IDW LOOCV",     "idw"),
        (axes[1], krg, "Kriging LOOCV", "kriging"),
    ]:
        lo = min(df["actual"].min(), df["predicted"].min()) - 0.2
        hi = max(df["actual"].max(), df["predicted"].max()) + 0.2
        ax.scatter(df["actual"], df["predicted"], alpha=0.6, s=25, edgecolors="k", lw=0.4)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="1:1 line")
        m = metrics.loc[key]
        ext_m = ext.loc[f"{key}_external"]
        ax.set_title(f"{label}\nLOOCV RMSE={m['rmse']:.3f} | Ext RMSE={ext_m['rmse']:.3f}")
        ax.set_xlabel("Actual wind speed (m/s)")
        ax.set_ylabel("Predicted wind speed (m/s)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.suptitle("LOOCV validation — predicted vs actual (ws10_mean)", y=1.01)
    plt.tight_layout()
    out = MAPS_OUT / "validation_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_kriging_uncertainty()
    plot_validation_scatter()
