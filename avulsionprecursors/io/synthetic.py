"""Build a tiny synthetic valley DEM and centerline for tests and examples."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

# EPSG:32610 (UTM 10N, metres). Elevations are metres (native = m).
SYNTHETIC_EPSG = 32610
CELL_SIZE = 10.0
ORIGIN_X = 500_000.0
ORIGIN_Y = 5_001_200.0  # north-up: origin is top-left
NCOLS = 80
NROWS = 120
CHANNEL_X = ORIGIN_X + 40 * CELL_SIZE  # 500400


def valley_elevation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Channel, levees, floodplain, and valley walls with a mild downstream slope.

    z_channel = 20 − 0.002 * (y − ymin)   (metres)
    levees at |x − channel| ≈ 40 m, +5 m
    floodplain at 70–110 m from channel, +2 m
    valley walls beyond 140 m, +40 m
    """
    ymin = ORIGIN_Y - NROWS * CELL_SIZE
    z_chan = 20.0 - 0.002 * (y - ymin)
    dx = np.abs(x - CHANNEL_X)
    z = z_chan + 2.0
    z = np.where(dx <= 15.0, z_chan, z)
    levee = (dx >= 30.0) & (dx <= 50.0)
    z = np.where(levee, z_chan + 5.0, z)
    wall = dx >= 140.0
    z = np.where(wall, z_chan + 40.0, z)
    return z


def write_synthetic_valley(out_dir: Path) -> dict:
    """Write dem.tif, centerline.shp, and example labels. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = ORIGIN_X + (np.arange(NCOLS) + 0.5) * CELL_SIZE
    ys = ORIGIN_Y - (np.arange(NROWS) + 0.5) * CELL_SIZE
    xx, yy = np.meshgrid(xs, ys)
    dem = valley_elevation(xx, yy).astype(np.float32)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, CELL_SIZE, CELL_SIZE)
    dem_path = out_dir / "dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=NROWS,
        width=NCOLS,
        count=1,
        dtype="float32",
        crs=f"EPSG:{SYNTHETIC_EPSG}",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)

    ymin = ORIGIN_Y - NROWS * CELL_SIZE
    line = LineString(
        [
            (CHANNEL_X, ymin + 50.0),
            (CHANNEL_X, ORIGIN_Y - 50.0),
        ]
    )
    cl_path = out_dir / "centerline.shp"
    gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs=f"EPSG:{SYNTHETIC_EPSG}").to_file(cl_path)
    return {"dem": dem_path, "centerline": cl_path, "dir": out_dir}


def write_synthetic_labels(
    profiles_dir: Path,
    study_name: str,
    labels_dir: Path,
) -> int:
    """Write channel / levee / floodplain labels from the known synthetic geometry.

    Uses the profile CSVs so dist_along matches the extract step. Picks:
    channel near distance 0 (thalweg), ridges near ±40 m, floodplains near ±90 m.
    """
    import pandas as pd

    profiles_dir = Path(profiles_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for csv_path in sorted(profiles_dir.glob("cross_section_*.csv")):
        node_id = int(csv_path.stem.split("_")[-1])
        df = pd.read_csv(csv_path)
        dist_col = "distance" if "distance" in df.columns else "distance_m"
        elev_col = "elevation" if "elevation" in df.columns else "elevation_m"
        prof = df[[dist_col, elev_col]].dropna()
        if prof.empty:
            continue

        def nearest(target: float):
            i = (prof[dist_col] - target).abs().idxmin()
            # dist_along on the line is centered distance plus half-length offset
            # used by the labeler: CSV is already centered at the node (0).
            # Lambda's load_labels uses the stored dist_along; calculate_lambda
            # only uses elevation for Hm/Har and dist_along for SAR run length,
            # so relative distances between ridge and floodplain matter.
            return float(prof.loc[i, dist_col]), float(prof.loc[i, elev_col])

        # Convert centered distance to dist_along by adding an offset so all
        # distances are positive; SAR uses differences only.
        offset = 500.0
        chan_d, chan_z = nearest(0.0)
        r1_d, r1_z = nearest(-40.0)
        fp1_d, fp1_z = nearest(-90.0)
        r2_d, r2_z = nearest(40.0)
        fp2_d, fp2_z = nearest(90.0)
        rows = [
            ("channel", chan_d + offset, chan_z),
            ("ridge1", r1_d + offset, r1_z),
            ("floodplain1", fp1_d + offset, fp1_z),
            ("ridge2", r2_d + offset, r2_z),
            ("floodplain2", fp2_d + offset, fp2_z),
        ]
        out = pd.DataFrame(rows, columns=["label", "dist_along", "elevation"])
        out.to_csv(labels_dir / f"{study_name}_node_{node_id}_labels.csv", index=False)
        n += 1
    return n
