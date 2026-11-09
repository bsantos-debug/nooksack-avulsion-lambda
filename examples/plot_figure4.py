"""Figure 4 style map: hillshade plus the red visit overlay.

Hillshade and the overlay follow ``figure_plotting/dem_ops.py`` and
``Figure4_interactive.py`` in Jake Gearon's rulesofriveravulsion repository.
The walk is this package's: it starts on the lower-ridge crest, and steps
back toward the channel are blocked only until the path is past the crest or
downstream of the start.

    python examples/plot_figure4.py
    python examples/plot_figure4.py nooksack
    python examples/plot_figure4.py nooksack 177
"""

from __future__ import annotations

import io
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from avulsionprecursors.analysis.routing import accumulate_visit_frequency
from avulsionprecursors.config import load_config
from avulsionprecursors.io.synthetic import write_synthetic_labels
from avulsionprecursors.pipeline.extract import extract_cross_sections
from avulsionprecursors.pipeline.lambda_calc import assemble_lambda_table
from avulsionprecursors.pipeline.lambda_routing import (
    _channel_point,
    _cross_section_line,
    _lower_ridge_seed,
    _outward_pixel_vector,
    channel_geometry,
    channel_mask,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALLEY = HERE / "synthetic_valley"
OUT = HERE / "routing_figure4.png"
# Whatcom 2024 topobathy mosaic (lidar plus bathymetry), reprojected to
# UTM metres. Elevations are still NAVD88 US survey feet.
NOOKSACK_DEM = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack/BathymetryData/Topobathy_reprojected.tif"
)
US_SURVEY_FOOT = 0.3048006096012192
NOOKSACK_XS = ROOT / "cross_sections.shp"
NOOKSACK_CENTERLINE = ROOT / "centerline_points_from_xs.shp"
NOOKSACK_LABELS = ROOT / "data" / "labels"
NOOKSACK_BANKFULL = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack/Model_outputs/Bankfull_flow_extent_2025.shp"
)
NOOKSACK_OUT = HERE / "nooksack_routing_figure4.png"
NOOKSACK_SERIES = HERE / "nooksack_routing"

# Figure 4 tool defaults: theta 10, alpha 1, beta 0.5, 100 trials,
# pct_over 0.075, sigma 2. Steps are shortened to this small DEM.
N_TRIALS = 100
N_STEPS = 80
THETA = 10.0
ALPHA = 1.0
BETA = 0.5
PCT_OVER = 0.075
SIGMA = 2.0
# Nooksack figures. The walk is longer than one station interval so the
# path can leave the crest. The window is a little larger than that length.
NOOKSACK_PATH_M = 40000.0
NOOKSACK_HALF_WIDTH_M = 8000.0
# The walker averages the mosaic to this cell size. Rise is scored per
# pixel, so a 1 m step cannot feel the valley slope.
NOOKSACK_CELL_M = 30.0
# Hillshade margin around the walks. The mosaic covers the county, so the
# figure stays on the river instead of the full raster.
NOOKSACK_PAD_M = 4000.0


def _read_nooksack_elevation(src, window, out_shape) -> np.ndarray:
    """Average the 2024 mosaic onto ``out_shape`` and convert feet to metres.

    Nodata is about -3e38. Real elevations in this mosaic are above -500 ft.
    """
    elevation = src.read(
        1,
        window=window,
        out_shape=out_shape,
        resampling=Resampling.average,
    ).astype(float)
    elevation[~np.isfinite(elevation) | (elevation < -1000.0)] = np.nan
    return elevation * US_SURVEY_FOOT


def _start_outside_polygon(
    seed_point: Point,
    channel_point: Point,
    polygon,
    cell_m: float,
) -> Point:
    """Move an enclosed seed outward along its bank direction.

    The returned point is at least half a routing cell beyond the polygon,
    so rounding to the 30 m routing grid does not put it back inside.
    """
    if polygon is None or polygon.is_empty:
        return seed_point
    clearance = max(float(cell_m) / 2.0, 0.5)
    if not polygon.covers(seed_point) and seed_point.distance(polygon) >= clearance:
        return seed_point
    dx = float(seed_point.x - channel_point.x)
    dy = float(seed_point.y - channel_point.y)
    length = float(np.hypot(dx, dy))
    if length < 1e-9:
        raise RuntimeError("Cannot move a bankfull seed without a channel-to-ridge direction.")
    ux, uy = dx / length, dy / length
    step = max(float(cell_m) / 4.0, 1.0)
    for distance in np.arange(step, 5000.0 + step, step):
        candidate = Point(seed_point.x + ux * distance, seed_point.y + uy * distance)
        if not polygon.covers(candidate) and candidate.distance(polygon) >= clearance:
            return candidate
    raise RuntimeError("Could not move the start outside the bankfull polygon within 5 km.")


def hillshade(elevation: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Gearon's ``func_hillshade`` (azimuth 315, altitude 45)."""
    x, y = np.gradient(elevation)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x**2 + y**2))
    aspect = np.arctan2(-x, y)
    azimuth_rad = np.clip(360.0 - azimuth, 0, 360) * np.pi / 180.0
    altitude_rad = np.clip(altitude, 0, 90) * np.pi / 180.0
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(
        azimuth_rad - np.pi / 2.0 - aspect
    )
    return (shaded + 1) / 2


def visit_overlay(average_visits: np.ndarray, pct_over: float, sigma: float) -> np.ndarray:
    """Red RGBA overlay. Gearon keeps cells whose smoothed visits exceed pct_over.

    The color scale runs across those kept cells. Scaling to the whole grid,
    which is mostly zeros, draws the path in near-white.
    """
    finite = np.isfinite(average_visits)
    smoothed = ndimage.gaussian_filter(np.where(finite, average_visits, 0.0), sigma=sigma)
    keep = smoothed > pct_over
    labeled, n = ndimage.label(keep)
    if n:
        counts = np.bincount(labeled.ravel())
        small = counts < 2
        small[0] = False
        keep = keep & ~small[labeled]
    shown = smoothed[keep]
    lo = float(shown.min()) if shown.size else pct_over
    hi = float(shown.max()) if shown.size else lo + 1.0
    lo = max(lo, 1e-6)
    if hi <= lo:
        hi = lo * 10.0
    # Log scale: a few cells are visited hundreds of times, and a linear
    # scale paints the rest of the path the same color as the paper.
    rgba = plt.get_cmap("Reds")(Normalize(vmin=np.log10(lo), vmax=np.log10(hi))(np.log10(np.clip(smoothed, lo, None))))
    rgba[..., 3] = np.where(keep, 1.0, 0.0)
    return rgba


def main() -> None:
    cfg = load_config(VALLEY / "config.yaml")
    if not cfg.prepared_dem_path().exists():
        extract_cross_sections(cfg)
    if not any(cfg.labels_dir().glob(f"{cfg.study_name}_node_*_labels.csv")):
        write_synthetic_labels(cfg.profiles_dir(), cfg.study_name, cfg.labels_dir())

    labeled = assemble_lambda_table(cfg, require_slope=False)
    labeled = labeled.sort_values("dist_out")
    node = labeled.iloc[len(labeled) // 2]
    node_id = int(node["node_id"])
    node_point = Point(float(node["x"]), float(node["y"]))
    cross_sections = gpd.read_file(cfg.cross_sections_path())
    line = _cross_section_line(cross_sections, node_id)
    seed = _lower_ridge_seed(node, line, node_point)
    if seed is None:
        raise RuntimeError(f"Node {node_id} has no lower-ridge crest.")
    side, seed_point = seed

    dem_path = cfg.prepared_dem_path()
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(float)
        if src.nodata is not None and np.isfinite(src.nodata):
            elevation[elevation == src.nodata] = np.nan
        transform = src.transform
        xres = abs(float(src.res[0]))
        yres = abs(float(src.res[1]))
        seed_row, seed_col = src.index(float(seed_point.x), float(seed_point.y))
        outward = _outward_pixel_vector(transform, _channel_point(node, line, node_point), seed_point)
        left = float(transform.c)
        right = left + float(transform.a) * src.width
        top = float(transform.f)
        bottom = top + float(transform.e) * src.height

    visit, _ = accumulate_visit_frequency(
        elevation,
        int(seed_row),
        int(seed_col),
        xres=xres,
        yres=yres,
        n_steps=N_STEPS,
        n_trials=N_TRIALS,
        theta=THETA,
        alpha=ALPHA,
        beta=BETA,
        outward=outward,
        rng=np.random.default_rng(0),
    )
    average = visit / float(N_TRIALS)
    shade = hillshade(np.where(np.isfinite(elevation), elevation, np.nanmean(elevation)))
    overlay = visit_overlay(average, PCT_OVER, SIGMA)

    _draw(
        shade,
        overlay,
        seed_point,
        (left, right, bottom, top),
        f"Synthetic valley, node {node_id} ({side})\nred: visit frequency > 0.075 after smoothing",
        OUT,
    )
    print(f"Wrote {OUT}")
    print(f"Start: {side} at ({seed_point.x:.1f}, {seed_point.y:.1f}), outward {outward}")
    print(f"Peak mean visits: {average.max():.3f}")


def plot_nooksack(
    node_id: int = 177,
    half_width: float = NOOKSACK_HALF_WIDTH_M,
    path_length: float = NOOKSACK_PATH_M,
    src=None,
    cross_sections=None,
    channel_geom=None,
    bankfull_geom=None,
    geometry_only: bool = False,
    return_seed: bool = False,
    return_frequency: bool = False,
):
    """Satellite and 10 m hillshade for one Nooksack node.

    The walk and the hillshade both use the lidar averaged to
    ``NOOKSACK_CELL_M``. The walk stops after ``path_length`` metres.
    """
    label_path = NOOKSACK_LABELS / f"Nooksack_node_{node_id}_labels.csv"
    if not NOOKSACK_DEM.exists():
        raise FileNotFoundError(NOOKSACK_DEM)
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    labels = pd.read_csv(label_path)
    row = {}
    for _, rec in labels.iterrows():
        name = str(rec["label"]).lower()
        row[f"{name}_dist_along"] = float(rec["dist_along"])
        row[f"{name}_elevation"] = float(rec["elevation"])
    row = pd.Series(row)
    for key in (
        "ridge1_elevation",
        "ridge2_elevation",
        "ridge1_dist_along",
        "ridge2_dist_along",
        "floodplain1_elevation",
        "floodplain2_elevation",
        "channel_dist_along",
    ):
        if key not in row.index:
            row[key] = np.nan

    if cross_sections is None:
        cross_sections = gpd.read_file(NOOKSACK_XS)
    match = cross_sections[cross_sections["node_id"] == node_id]
    if match.empty:
        raise RuntimeError(f"No cross section for node {node_id}.")
    line = match.geometry.iloc[0]
    node_point = line.interpolate(float(line.project(line.centroid)))
    # The channel pick is the node used to aim the walk; the line midpoint is
    # only a fallback inside _channel_point.
    channel = _point_on_transect_safe(line, row)
    seed = _lower_ridge_seed(row, line, channel)
    if seed is None:
        raise RuntimeError(f"Node {node_id} has no lower-ridge crest.")
    side, seed_point = seed

    own_src = src is None
    if own_src:
        src = rasterio.open(NOOKSACK_DEM)
    if channel_geom is None:
        channel_geom = channel_geometry(gpd.read_file(NOOKSACK_CENTERLINE))
    if bankfull_geom is None:
        bankfull = gpd.read_file(NOOKSACK_BANKFULL)
        if bankfull.crs != src.crs:
            bankfull = bankfull.to_crs(src.crs)
        bankfull_geom = bankfull.geometry.unary_union
    seed_point = _start_outside_polygon(
        seed_point,
        channel,
        bankfull_geom,
        NOOKSACK_CELL_M,
    )
    try:
        return _plot_nooksack_on_src(
            src,
            node_id,
            side,
            seed_point,
            row,
            line,
            channel,
            channel_geom,
            half_width,
            path_length,
            geometry_only=geometry_only,
            return_seed=return_seed,
            return_frequency=return_frequency,
        )
    finally:
        if own_src:
            src.close()


def _plot_nooksack_on_src(
    src,
    node_id,
    side,
    seed_point,
    row,
    line,
    channel,
    channel_geom,
    half_width,
    path_length,
    geometry_only: bool = False,
    return_seed: bool = False,
    return_frequency: bool = False,
):
    bounds = (
        float(seed_point.x) - half_width,
        float(seed_point.y) - half_width,
        float(seed_point.x) + half_width,
        float(seed_point.y) + half_width,
    )
    window = from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets(op="floor").round_lengths(op="ceil")
    window = window.intersection(Window(0, 0, src.width, src.height))
    if window.width < 2 or window.height < 2:
        print(f"Skip node {node_id}: the crest is outside the DEM.")
        return
    native = abs(float(src.res[0]))
    walk_cell = float(NOOKSACK_CELL_M)
    out_h = max(1, int(round(window.height * native / walk_cell)))
    out_w = max(1, int(round(window.width * native / walk_cell)))
    elevation = _read_nooksack_elevation(src, window, (out_h, out_w))
    transform = src.window_transform(window) * Affine.scale(window.width / out_w, window.height / out_h)
    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    left = float(transform.c)
    right = left + float(transform.a) * out_w
    top = float(transform.f)
    bottom = top + float(transform.e) * out_h

    seed_col_f, seed_row_f = (~transform) * (float(seed_point.x), float(seed_point.y))
    seed_row = int(round(seed_row_f))
    seed_col = int(round(seed_col_f))
    outward = _outward_pixel_vector(transform, _channel_point(row, line, channel), seed_point)
    blocked = channel_mask(elevation.shape, transform, channel_geom)
    if 0 <= seed_row < blocked.shape[0] and 0 <= seed_col < blocked.shape[1]:
        blocked[seed_row, seed_col] = False
    walk_z = elevation.copy()
    walk_z[blocked] = np.nan
    n_steps = max(1, int(round(path_length / walk_cell)))
    visit, _ = accumulate_visit_frequency(
        walk_z,
        seed_row,
        seed_col,
        xres=xres,
        yres=yres,
        n_steps=n_steps,
        n_trials=N_TRIALS,
        theta=THETA,
        alpha=ALPHA,
        beta=BETA,
        outward=outward,
        rng=np.random.default_rng(0),
    )
    average = visit / float(N_TRIALS)
    valid = np.isfinite(elevation)
    fill = float(np.nanmean(elevation)) if np.any(valid) else 0.0
    shade = hillshade(np.where(valid, elevation, fill))
    shade = np.where(valid, shade, np.nan)
    overlay = visit_overlay(average, PCT_OVER, SIGMA)
    smoothed = ndimage.gaussian_filter(np.where(valid, average, 0.0), sigma=SIGMA)
    occupied = smoothed > PCT_OVER
    ys, xs = np.nonzero(occupied if np.any(occupied) else visit > 0)
    if ys.size == 0:
        print(f"Skip node {node_id}: the walk did not leave a path.")
        return None
    if geometry_only:
        parts = [
            shape(geom)
            for geom, value in shapes(occupied.astype(np.uint8), mask=occupied, transform=transform)
            if value
        ]
        geometry = None if not parts else unary_union(parts)
        frequency = None
        if return_frequency:
            peak = float(smoothed.max())
            normalized = smoothed / peak if peak > 0 else np.zeros_like(smoothed)
            freq_rows, freq_cols = np.nonzero(valid & (normalized >= PCT_OVER))
            freq_x = (
                transform.c
                + (freq_cols.astype(float) + 0.5) * transform.a
                + (freq_rows.astype(float) + 0.5) * transform.b
            )
            freq_y = (
                transform.f
                + (freq_cols.astype(float) + 0.5) * transform.d
                + (freq_rows.astype(float) + 0.5) * transform.e
            )
            frequency = (freq_x, freq_y, normalized[freq_rows, freq_cols])
        return (geometry, seed_point, frequency) if return_seed else geometry
    span_x = float((xs.max() - xs.min()) * xres)
    span_y = float((ys.max() - ys.min()) * yres)
    pad = int(round(800.0 / walk_cell))
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(out_h, int(ys.max()) + pad + 1)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(out_w, int(xs.max()) + pad + 1)
    shade = shade[r0:r1, c0:c1]
    overlay = overlay[r0:r1, c0:c1]
    blocked = blocked[r0:r1, c0:c1]
    x0, y0 = transform * (c0, r0)
    x1, y1 = transform * (c1, r1)
    left, right = (x0, x1) if x0 < x1 else (x1, x0)
    bottom, top = (y0, y1) if y0 < y1 else (y1, y0)
    NOOKSACK_SERIES.mkdir(parents=True, exist_ok=True)
    out = NOOKSACK_SERIES / f"node_{node_id}.png"
    title = (
        f"Nooksack 2024, node {node_id} ({side}), "
        f"{path_length / 1000:.0f} km walk on {walk_cell:.0f} m cells"
    )
    _draw(
        shade,
        overlay,
        seed_point,
        (left, right, bottom, top),
        title,
        out,
        blocked,
        crs="EPSG:26910",
    )
    if node_id == 177:
        shutil.copy(out, NOOKSACK_OUT)
        print(f"Wrote {NOOKSACK_OUT}")
    print(f"Wrote {out}")
    print(f"Start: {side} at ({seed_point.x:.1f}, {seed_point.y:.1f}), outward {outward}")
    print(
        f"Window {out_w} x {out_h} at {walk_cell:g} m, "
        f"{n_steps} steps ({path_length / 1000:.1f} km)"
    )
    print(f"Visit span {span_x:.0f} m east-west, {span_y:.0f} m north-south")
    print(f"Peak mean visits: {average.max():.3f}")


def plot_every_other_nooksack_node() -> None:
    """One figure for every other labeled node, upstream to downstream."""
    cross_sections = gpd.read_file(NOOKSACK_XS).sort_values("dist_out", ascending=False)
    labeled = []
    for node_id in cross_sections["node_id"].astype(int):
        if (NOOKSACK_LABELS / f"Nooksack_node_{node_id}_labels.csv").exists():
            labeled.append(int(node_id))
    chosen = labeled[::2]
    print(f"{len(labeled)} labeled nodes, plotting every other: {len(chosen)}")
    channel_geom = channel_geometry(gpd.read_file(NOOKSACK_CENTERLINE))
    bankfull = gpd.read_file(NOOKSACK_BANKFULL).to_crs(cross_sections.crs)
    bankfull_geom = bankfull.geometry.unary_union
    with rasterio.open(NOOKSACK_DEM) as src:
        for index, node_id in enumerate(chosen, start=1):
            out = NOOKSACK_SERIES / f"node_{node_id}.png"
            if out.exists():
                print(f"[{index}/{len(chosen)}] node {node_id} already written")
                continue
            print(f"[{index}/{len(chosen)}] node {node_id}")
            try:
                plot_nooksack(
                    node_id,
                    src=src,
                    cross_sections=cross_sections,
                    channel_geom=channel_geom,
                    bankfull_geom=bankfull_geom,
                )
            except Exception as exc:
                print(f"Skip node {node_id}: {exc}")


def plot_all_paths_on_one_dem(normalized_frequency: bool = False) -> None:
    """Every other node's walk, drawn together on one routing-grid hillshade."""
    cross_sections = gpd.read_file(NOOKSACK_XS).sort_values("dist_out", ascending=False)
    labeled = []
    for node_id in cross_sections["node_id"].astype(int):
        if (NOOKSACK_LABELS / f"Nooksack_node_{node_id}_labels.csv").exists():
            labeled.append(int(node_id))
    chosen = labeled[::2]
    print(f"Collecting paths for {len(chosen)} nodes")
    channel_geom = channel_geometry(gpd.read_file(NOOKSACK_CENTERLINE))
    bankfull = gpd.read_file(NOOKSACK_BANKFULL).to_crs(cross_sections.crs)
    bankfull_geom = bankfull.geometry.unary_union
    paths = []
    start_points = []
    frequencies = []
    path_colors = []
    path_edges = []
    with rasterio.open(NOOKSACK_DEM) as src:
        for index, node_id in enumerate(chosen, start=1):
            print(f"[{index}/{len(chosen)}] node {node_id}", flush=True)
            try:
                result = plot_nooksack(
                    node_id,
                    src=src,
                    cross_sections=cross_sections,
                    channel_geom=channel_geom,
                    bankfull_geom=bankfull_geom,
                    geometry_only=True,
                    return_seed=True,
                    return_frequency=normalized_frequency,
                )
            except Exception as exc:
                print(f"Skip node {node_id}: {exc}")
                continue
            if result is None:
                continue
            geom, start_point, frequency = result
            if geom is not None and not geom.is_empty:
                paths.append(geom)
                start_points.append(start_point)
                if frequency is not None:
                    frequencies.append(frequency)
                if index % 2:
                    path_colors.append("#d7191c")
                    path_edges.append("#7f0000")
                else:
                    path_colors.append("#f2b701")
                    path_edges.append("#8a6500")
                print(f"  path {index}: {len(paths)} kept", flush=True)
        if not paths:
            raise RuntimeError("No paths to draw.")
        minx, miny, maxx, maxy = unary_union(paths).bounds
        if frequencies:
            minx = min(minx, *(float(x.min()) for x, _, _ in frequencies if x.size))
            miny = min(miny, *(float(y.min()) for _, y, _ in frequencies if y.size))
            maxx = max(maxx, *(float(x.max()) for x, _, _ in frequencies if x.size))
            maxy = max(maxy, *(float(y.max()) for _, y, _ in frequencies if y.size))
        pad = float(NOOKSACK_PAD_M)
        bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        window = window.intersection(Window(0, 0, src.width, src.height))
        native = abs(float(src.res[0]))
        cell = float(NOOKSACK_CELL_M)
        out_h = max(1, int(round(window.height * native / cell)))
        out_w = max(1, int(round(window.width * native / cell)))
        elevation = _read_nooksack_elevation(src, window, (out_h, out_w))
        transform = src.window_transform(window) * Affine.scale(window.width / out_w, window.height / out_h)
    valid = np.isfinite(elevation)
    fill = float(np.nanmean(elevation)) if np.any(valid) else 0.0
    shade = hillshade(np.where(valid, elevation, fill))
    shade = np.where(valid, shade, np.nan)
    left = float(transform.c)
    right = left + float(transform.a) * out_w
    top = float(transform.f)
    bottom = top + float(transform.e) * out_h
    frame = gpd.GeoDataFrame({"geometry": paths}, geometry="geometry")
    width_m = right - left
    height_m = top - bottom
    fig_w = 16.0
    fig_h = max(4.0, fig_w * height_m / width_m)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    ax.imshow(
        _stretch_shade(shade),
        cmap="gray",
        extent=[left, right, bottom, top],
        origin="upper",
        interpolation="nearest",
    )
    if channel_geom is not None and not channel_geom.is_empty:
        gpd.GeoSeries([channel_geom]).plot(ax=ax, color="#5aa6e6", alpha=0.45, zorder=2)
    if bankfull_geom is not None and not bankfull_geom.is_empty:
        gpd.GeoSeries([bankfull_geom]).boundary.plot(
            ax=ax,
            color="#00c8c8",
            linewidth=0.8,
            linestyle="--",
            zorder=3.5,
        )
    if normalized_frequency:
        combined = np.zeros(elevation.shape, dtype=float)
        for freq_x, freq_y, freq_value in frequencies:
            cols = np.floor((freq_x - left) / float(transform.a)).astype(int)
            rows = np.floor((top - freq_y) / abs(float(transform.e))).astype(int)
            inside = (
                (rows >= 0)
                & (rows < combined.shape[0])
                & (cols >= 0)
                & (cols < combined.shape[1])
            )
            np.maximum.at(combined, (rows[inside], cols[inside]), freq_value[inside])
        shown = np.ma.masked_less(combined, PCT_OVER)
        frequency_image = ax.imshow(
            shown,
            cmap="magma",
            vmin=PCT_OVER,
            vmax=1.0,
            alpha=0.82,
            extent=[left, right, bottom, top],
            origin="upper",
            interpolation="nearest",
            zorder=3,
        )
        colorbar = fig.colorbar(frequency_image, ax=ax, fraction=0.025, pad=0.015)
        colorbar.set_label("Normalized visitation frequency (per-walk peak = 1)")
    else:
        frame.plot(
            ax=ax,
            color=path_colors,
            alpha=0.6,
            edgecolor=path_edges,
            linewidth=0.35,
            zorder=3,
        )
    gpd.GeoSeries(start_points).plot(
        ax=ax,
        color="#ffff00",
        edgecolor="#111111",
        linewidth=0.45,
        markersize=18,
        zorder=4,
    )
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    if normalized_frequency:
        ax.set_title(
            f"Nooksack 2024 topobathy, normalized visitation frequency for {len(paths)} "
            f"walks on {NOOKSACK_CELL_M:.0f} m cells\n"
            "yellow: starts; cyan: bankfull edge; blue: closed channel; 40 km allowed"
        )
    else:
        ax.set_title(
            f"Nooksack 2024 topobathy, {len(paths)} walks on one "
            f"{NOOKSACK_CELL_M:.0f} m hillshade\n"
            "yellow: starts; cyan: bankfull edge; red/gold: paths; blue: closed channel"
        )
    ax.set_facecolor("#e6e6e6")
    fig.tight_layout()
    out = HERE / (
        "nooksack_routing_normalized_frequency.png"
        if normalized_frequency
        else "nooksack_routing_all_paths.png"
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out} ({len(paths)} paths)")


def _point_on_transect_safe(line, row) -> Point:
    """Channel label on the transect, or the line's midpoint."""
    dist = row.get("channel_dist_along", np.nan)
    if dist is None or not np.isfinite(float(dist)):
        return line.interpolate(float(line.length) / 2.0)
    dist = float(dist)
    if 0.0 <= dist <= float(line.length):
        return line.interpolate(dist)
    return line.interpolate(float(line.length) / 2.0)


def _for_figure(shade: np.ndarray, overlay: np.ndarray, channel=None, max_side: int = 3200):
    """Thin a native-resolution hillshade for the PNG.

    The shade is averaged in blocks. That keeps banks and bar edges, which
    bilinear resampling of the elevation does not. Each block of the red
    overlay keeps the cell that was drawn, so a thin path is not dropped.
    A channel cell is kept if any fine cell in the block is channel.
    """
    if channel is None:
        channel = np.zeros(shade.shape[:2], dtype=bool)
    step = max(1, int(np.ceil(max(shade.shape) / max_side)))
    if step == 1:
        return shade, overlay, channel, shade.shape[0], shade.shape[1]
    height = shade.shape[0] // step * step
    width = shade.shape[1] // step * step
    rows = height // step
    cols = width // step
    with np.errstate(all="ignore"):
        shade_block = np.nanmean(
            shade[:height, :width].reshape(rows, step, cols, step),
            axis=(1, 3),
        )
    flat = overlay[:height, :width].reshape(rows, step * step, cols, overlay.shape[-1])
    pick = flat[..., 3].argmax(axis=1)
    overlay_block = flat[np.arange(rows)[:, None], pick, np.arange(cols)[None, :]]
    channel_block = channel[:height, :width].reshape(rows, step, cols, step).any(axis=(1, 3))
    rows = min(shade_block.shape[0], overlay_block.shape[0], channel_block.shape[0])
    cols = min(shade_block.shape[1], overlay_block.shape[1], channel_block.shape[1])
    return (
        shade_block[:rows, :cols],
        overlay_block[:rows, :cols],
        channel_block[:rows, :cols],
        rows * step,
        cols * step,
    )


def _satellite_image(extent, shape, crs: str) -> np.ndarray | None:
    """Esri World Imagery for this extent, with north at the top."""
    from PIL import Image

    left, right, bottom, top = extent
    height, width = int(shape[0]), int(shape[1])
    # The hillshade may be only a few hundred 10 m cells. Ask for a sharper photo.
    long_side = max(height, width, 1)
    scale = float(np.clip(1000.0 / long_side, 0.5, 6.0))
    req_w = max(2, int(round(width * scale)))
    req_h = max(2, int(round(height * scale)))
    wkid = crs.split(":")[-1]
    query = urllib.parse.urlencode(
        {
            "bbox": f"{left},{bottom},{right},{top}",
            "bboxSR": wkid,
            "imageSR": wkid,
            "size": f"{req_w},{req_h}",
            "format": "png",
            "f": "image",
        }
    )
    url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?" + query
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "avulsionprecursors"})
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=float) / 255.0
    except Exception as exc:
        print(f"Satellite imagery was not loaded: {exc}")
        return None
    return image


def _path_outline(overlay: np.ndarray) -> np.ndarray:
    """Red edge of the cells the walk occupied, so the ground inside stays visible."""
    occupied = overlay[..., 3] > 0
    edge = occupied & ~ndimage.binary_erosion(occupied, border_value=0)
    edge = ndimage.binary_dilation(edge, iterations=1)
    rgba = np.zeros((*occupied.shape, 4), dtype=float)
    rgba[edge] = (0.75, 0.05, 0.05, 1.0)
    return rgba


def _mark_start(ax, seed_point) -> None:
    ax.plot(
        seed_point.x,
        seed_point.y,
        marker="o",
        color="yellow",
        markersize=6,
        markeredgecolor="black",
        markeredgewidth=0.4,
    )


def _style_map(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.set_facecolor("#e6e6e6")


def _stretch_shade(shade: np.ndarray) -> np.ndarray:
    finite = np.isfinite(shade)
    if not np.any(finite):
        return shade
    lo, hi = np.percentile(shade[finite], [2, 98])
    if hi <= lo:
        return shade
    stretched = np.clip((shade - lo) / (hi - lo), 0.0, 1.0)
    return np.where(finite, stretched, np.nan)


def _draw(
    shade,
    overlay,
    seed_point,
    extent_parts,
    title: str,
    path: Path,
    channel=None,
    crs=None,
) -> None:
    left, right, bottom, top = extent_parts
    shade = _stretch_shade(shade)
    extent = [left, right, bottom, top]
    height, width = shade.shape[:2]
    dpi = max(140, int(round(max(width, height) / 8.0)))
    panel_w = 8.0 * width / max(width, height)
    panel_h = 8.0 * height / max(width, height)
    outline = _path_outline(overlay)

    if crs:
        image = _satellite_image((left, right, bottom, top), shade.shape[:2], crs)
        fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, panel_h), dpi=dpi)
        photo, ground = axes
        if image is not None:
            photo.imshow(np.clip(image, 0.0, 1.0), extent=extent, origin="upper", interpolation="bilinear")
        else:
            photo.imshow(shade, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
        if channel is not None and np.any(channel):
            blue = np.zeros((*channel.shape, 4), dtype=float)
            blue[channel.astype(bool)] = (0.35, 0.65, 0.95, 0.45)
            photo.imshow(blue, extent=extent, origin="upper", interpolation="nearest")
        photo.imshow(overlay, extent=extent, origin="upper", interpolation="nearest")
        _mark_start(photo, seed_point)
        _style_map(photo, "Satellite and walk")

        ground.imshow(shade, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
        ground.imshow(outline, extent=extent, origin="upper", interpolation="nearest")
        _mark_start(ground, seed_point)
        _style_map(ground, "Hillshade and path extent")
        fig.suptitle(title.split("\n")[0], fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig, ax = plt.subplots(figsize=(panel_w, panel_h), dpi=dpi)
        ax.imshow(shade, cmap="gray", extent=extent, origin="upper", interpolation="nearest")
        ax.imshow(overlay, extent=extent, origin="upper", interpolation="nearest")
        _mark_start(ax, seed_point)
        _style_map(ax, title)
        fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "nooksack":
        if len(sys.argv) > 2 and sys.argv[2] == "every-other":
            plot_every_other_nooksack_node()
        elif len(sys.argv) > 2 and sys.argv[2] == "together":
            plot_all_paths_on_one_dem()
        elif len(sys.argv) > 2 and sys.argv[2] == "frequency":
            plot_all_paths_on_one_dem(normalized_frequency=True)
        else:
            chosen = int(sys.argv[2]) if len(sys.argv) > 2 else 177
            plot_nooksack(chosen)
    else:
        main()
