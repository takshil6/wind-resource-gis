"""Generate Phase 3 plots: suitability map, feature decomposition, power metrics, top sites table."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = ROOT / "data/interim"
MAPS_OUT = ROOT / "outputs/maps"
MAPS_OUT.mkdir(parents=True, exist_ok=True)


def _load():
    npz = np.load(DATA_INTERIM / "suitability_surfaces.npz")
    top = pd.read_csv(DATA_INTERIM / "top_sites.csv")
    return npz, top


def plot_suitability_map(npz, top):
    grid_lat = npz["grid_lat"]
    grid_lon = npz["grid_lon"]
    on_land = npz["on_land"].astype(bool)
    score = np.ma.masked_where(~on_land, npz["score_land_only"])

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(grid_lon, grid_lat, score, cmap=cmap,
                       vmin=np.nanpercentile(npz["score_land_only"], 5),
                       vmax=np.nanpercentile(npz["score_land_only"], 95),
                       shading="auto")
    plt.colorbar(im, ax=ax, label="Suitability score (0–100)")

    ax.scatter(top["lon"], top["lat"], marker="*", s=120,
               c="navy", zorder=5, label="Top 25 sites")
    for _, row in top.head(5).iterrows():
        ax.annotate(f"#{int(row['rank'])}", (row["lon"], row["lat"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="navy")

    ax.set_title("Wind Site Suitability Score — Land Only\n"
                 "(offshore = grey; higher = better for rooftop wind)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=8)

    out = MAPS_OUT / "suitability_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_feature_decomposition(npz):
    grid_lat = npz["grid_lat"]
    grid_lon = npz["grid_lon"]
    on_land = npz["on_land"].astype(bool)

    panels = [
        ("f_wind",        "Wind Speed Factor",                      "YlOrRd"),
        ("f_consistency", "Consistency Factor\n(low CV = steady)",  "Blues"),
        ("f_elevation",   "Elevation Factor",                       "Greens"),
        ("f_coast",       "Coastal Proximity Factor",               "PuBu"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, cmap_name) in zip(axes.flat, panels):
        data = np.ma.masked_where(~on_land, npz[key])
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgrey")
        im = ax.pcolormesh(grid_lon, grid_lat, data, cmap=cmap,
                           vmin=0, vmax=1, shading="auto")
        plt.colorbar(im, ax=ax, label="0–1")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("Feature Decomposition — Suitability Score Components\n"
                 "(offshore = grey)", y=1.01)
    plt.tight_layout()
    out = MAPS_OUT / "feature_decomposition.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_power_metrics(npz):
    grid_lat = npz["grid_lat"]
    grid_lon = npz["grid_lon"]
    on_land = npz["on_land"].astype(bool)

    panels = [
        (np.ma.masked_where(~on_land, npz["wpd"]),
         "Wind Power Density (W/m²)", "W/m²", "YlOrRd"),
        (np.ma.masked_where(~on_land, npz["capacity_factor"] * 100),
         "Capacity Factor (%)", "%", "RdYlGn"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (data, title, label, cmap_name) in zip(axes, panels):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("lightgrey")
        im = ax.pcolormesh(grid_lon, grid_lat, data, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, label=label)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.suptitle("Power Metrics — Land Cells Only (offshore = grey)")
    plt.tight_layout()
    out = MAPS_OUT / "power_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_top_sites_table(top):
    rows = top.head(10).copy()
    rows["score"] = rows["score"].round(1)
    rows["wind_speed_ms"] = rows["wind_speed_ms"].round(2)
    rows["capacity_factor"] = (rows["capacity_factor"] * 100).round(1)

    col_labels = ["Rank", "Lat", "Lon", "Score", "Wind (m/s)", "CF (%)"]
    table_data = rows[["rank", "lat", "lon", "score",
                        "wind_speed_ms", "capacity_factor"]].values.tolist()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.6)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data) + 1):
        bg = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    ax.set_title("Top 10 Wind Sites — Northeast US (Land-Only Suitability Score)",
                 fontsize=11, pad=12)
    out = MAPS_OUT / "top_sites_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    npz, top = _load()
    plot_suitability_map(npz, top)
    plot_feature_decomposition(npz)
    plot_power_metrics(npz)
    plot_top_sites_table(top)
