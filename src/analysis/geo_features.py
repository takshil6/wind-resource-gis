"""Geospatial feature engineering: coastal distance, terrain effects."""
from __future__ import annotations
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree


def load_us_coast() -> gpd.GeoDataFrame:
    """Load US coastline from Natural Earth (bundled with geopandas datasets)."""
    # Natural Earth low-res — fine for regional analysis
    url = (
        "https://naciscdn.org/naturalearth/110m/physical/"
        "ne_110m_coastline.zip"
    )
    return gpd.read_file(url)


def distance_to_coast_km(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    coast: gpd.GeoDataFrame | None = None,
) -> np.ndarray:
    """Compute distance from each grid cell to nearest coastline (km).

    Uses simple equirectangular approximation — accurate enough for
    regional-scale (within a few %).
    """
    if coast is None:
        coast = load_us_coast()

    # Extract coast vertices as points
    coast_points: list[tuple[float, float]] = []
    for geom in coast.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coast_points.extend([(p[1], p[0]) for p in geom.coords])
        elif geom.geom_type == "MultiLineString":
            for ls in geom.geoms:
                coast_points.extend([(p[1], p[0]) for p in ls.coords])

    coast_arr = np.array(coast_points)  # (lat, lon)

    # Equirectangular projection — convert deg to km using local lat
    mean_lat = np.deg2rad(grid_lat.mean())
    KM_PER_DEG = 111.0

    coast_xy = np.column_stack([
        coast_arr[:, 0] * KM_PER_DEG,
        coast_arr[:, 1] * KM_PER_DEG * np.cos(mean_lat),
    ])
    grid_xy = np.column_stack([
        grid_lat.ravel() * KM_PER_DEG,
        grid_lon.ravel() * KM_PER_DEG * np.cos(mean_lat),
    ])

    tree = cKDTree(coast_xy)
    dist_km, _ = tree.query(grid_xy, k=1)
    return dist_km.reshape(grid_lat.shape)


def load_land_polygons() -> gpd.GeoDataFrame:
    """Natural Earth land polygons. 50m resolution for better bay/island fidelity."""
    url = "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
    return gpd.read_file(url)


def land_mask(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    land: gpd.GeoDataFrame | None = None,
) -> np.ndarray:
    """Boolean mask: True where grid cell is on land.

    Uses Natural Earth 50m land polygons. Inner bays (Narragansett, LIS)
    are still imperfect but vastly better than 110m for our region.
    """
    from shapely.geometry import Point
    if land is None:
        land = load_land_polygons()

    pts = gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lat, lon in zip(
            grid_lat.ravel(), grid_lon.ravel()
        )],
        crs="EPSG:4326",
    )
    land = land.to_crs("EPSG:4326")
    joined = gpd.sjoin(pts, land[["geometry"]], how="left", predicate="within")
    # sjoin keeps original index; collapse duplicates from overlapping polys
    is_land = joined.groupby(level=0)["index_right"].first().notna()
    return is_land.to_numpy().reshape(grid_lat.shape)