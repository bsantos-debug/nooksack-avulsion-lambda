"""Lambda using the slope of the most frequent avulsion-routing paths.

The thalweg-slope lambda (``calculate_lambda``) is unchanged. This command
writes a separate CSV and plot. At each labeled node it runs the softmax
random walk from Gearon et al. (2024), starting at the crest of the lower
ridge, and replaces Sm with the visit-weighted map slope of steps that land
on the most frequently visited cells. The channel, the centerline buffered
by half its measured width, is nodata for the whole walk. Steps back toward
the channel are also refused until the walk has moved past the crest or
downstream at least as far as the crest is from the channel. Har, SAR, Hm,
and the lower-ridge rule stay the same, so

    γ = SAR / S_routing
    β = Har / Hm
    Λ = γ × β
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine, xy
from rasterio.windows import Window
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from avulsionprecursors.analysis.metrics import (
    alluvial_ridge_height,
    apply_ridge_parameters,
    choose_lower_ridge_bank,
)
from avulsionprecursors.analysis.routing import (
    RoutingPathSlope,
    aggregate_elevation,
    aggregation_factor,
    routing_slope_at_seed,
)
from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.exceptions import InputValidationError
from avulsionprecursors.io.validation import validate_config
from avulsionprecursors.pipeline.lambda_calc import assemble_lambda_table, plot_lambda_vs_distance


def calculate_lambda_routing(cfg: WorkflowConfig) -> pd.DataFrame:
    """Write Λ computed with the frequent-path routing slope.

    Requires the extract step (cross-sections, centerline points, prepared
    DEM) and the same labels as ``calculate_lambda``. Each walk starts at
    the lower-ridge crest. Does not overwrite ``<study>_lambda_results.csv``.
    """
    validate_config(cfg)
    dem_path = cfg.prepared_dem_path()
    if not dem_path.exists():
        raise InputValidationError(
            f"Prepared DEM not found:\n  {dem_path}\n"
            "Run the extract step first."
        )
    points_path = cfg.centerline_points_path()
    xs_path = cfg.cross_sections_path()
    if not points_path.exists() or not xs_path.exists():
        raise InputValidationError(
            "Cross-sections or centerline points are missing. Run the extract step first."
        )

    route = cfg.routing
    print(f"Routing-slope lambda for {cfg.study_name}")
    print(
        "Walk start: crest of the lower ridge. The channel at its "
        "measured width, is closed for the whole walk."
    )
    print(
        "Sm source: visit-weighted map slope on the most frequent softmax paths "
        f"(cells with at least {route.frequency_threshold:.1%} of the peak visit count)"
    )
    print(
        f"Walk: theta={route.theta:g}, alpha={route.alpha:g}, beta={route.beta:g}, "
        f"trials={route.n_trials}, cell={route.cell_m:g} (0 = native)"
    )
    print("Thalweg-slope lambda is left unchanged.")

    points = gpd.read_file(points_path)
    cross_sections = gpd.read_file(xs_path)
    labeled = assemble_lambda_table(cfg, require_slope=False)
    slope_rows, path_rows = _routing_slopes_for_nodes(
        cfg, labeled, cross_sections, dem_path, points
    )
    slopes = pd.DataFrame(slope_rows)
    slopes_path = cfg.routing_slopes_csv_path()
    slopes_path.parent.mkdir(parents=True, exist_ok=True)
    slopes.to_csv(slopes_path, index=False)
    print(f"Saved routing slopes to {slopes_path}")

    if path_rows:
        paths = gpd.GeoDataFrame(path_rows, geometry="geometry", crs=points.crs)
        paths.to_file(cfg.routing_paths_path())
        print(f"Saved {len(paths)} modal paths to {cfg.routing_paths_path()}")

    labeled = labeled.merge(
        slopes[
            [
                "node_id",
                "routing_slope",
                "routing_side",
                "modal_path_slope",
                "n_frequent_cells",
                "n_frequent_visits",
            ]
        ],
        on="node_id",
        how="left",
    )
    labeled["thalweg_slope"] = labeled["slope"]
    labeled["slope"] = labeled["routing_slope"]

    df = apply_ridge_parameters(labeled, slope_min=cfg.extract.slope_min)
    df["slope_source"] = "routing_modal_path"
    n_valid = int(df["flag_valid_lambda"].sum()) if "flag_valid_lambda" in df.columns else 0
    n_adv = int((df["slope"].astype(float) <= 0).sum()) if "slope" in df.columns else 0
    print(f"Calculated routing lambda for {n_valid} nodes ({n_adv} non-positive routing slopes omitted from Λ)")
    if n_valid:
        valid = df.loc[df["flag_valid_lambda"], "lambda"]
        print(f"Routing lambda range: {valid.min():.4f} - {valid.max():.4f}")

    out = cfg.lambda_routing_csv_path()
    df.to_csv(out, index=False)
    print(f"Saved routing lambda to {out}")
    plot_lambda_vs_distance(
        df,
        cfg.study_name,
        cfg.lambda_routing_plot_path(),
        title=f"{cfg.study_name} — lambda from routing-path slope",
    )
    return df


# Do not expand the channel beyond its measured width.
CHANNEL_WIDEN = 0.0


def channel_geometry(points: gpd.GeoDataFrame, widen: float = CHANNEL_WIDEN):
    """Centerline buffered by half the measured width, then widened.

    The default ``widen`` of 0 makes the closed band equal to the measured
    channel width. Returns None when widths are missing, so a study without
    a channel width is unchanged.
    """
    if points is None or points.empty or "width" not in points.columns:
        return None
    ordered = points.sort_values("dist_out") if "dist_out" in points.columns else points
    xs = ordered.geometry.x.to_numpy(dtype=float)
    ys = ordered.geometry.y.to_numpy(dtype=float)
    widths = ordered["width"].to_numpy(dtype=float)
    parts = []
    for i in range(len(xs) - 1):
        w0, w1 = widths[i], widths[i + 1]
        if not np.isfinite(w0) and not np.isfinite(w1):
            continue
        if not np.isfinite(w0):
            w0 = w1
        if not np.isfinite(w1):
            w1 = w0
        half = (float(w0) + float(w1)) / 4.0 * (1.0 + float(widen))
        if half <= 0:
            continue
        parts.append(LineString([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])]).buffer(half))
    if not parts:
        return None
    return unary_union(parts)


def channel_mask(shape: tuple[int, int], transform, geometry) -> np.ndarray:
    """True on walker cells whose center lies in the channel buffer."""
    blocked = np.zeros(shape, dtype=bool)
    if geometry is None or geometry.is_empty:
        return blocked
    burned = rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return burned.astype(bool)


def _routing_slopes_for_nodes(
    cfg: WorkflowConfig,
    labeled: pd.DataFrame,
    cross_sections: gpd.GeoDataFrame,
    dem_path,
    centerline_points: gpd.GeoDataFrame,
) -> tuple[list[dict], list[dict]]:
    route = cfg.routing
    rng = np.random.default_rng(route.seed)
    channel = channel_geometry(centerline_points)
    slope_rows: list[dict] = []
    path_rows: list[dict] = []

    with rasterio.open(dem_path) as src:
        xres = abs(float(src.res[0]))
        yres = abs(float(src.res[1]))
        cell = min(xres, yres)
        factor = aggregation_factor(cell, float(route.cell_m))
        walk_cell = cell * factor
        path_length = float(route.path_length) if route.path_length is not None else float(cfg.extract.spacing)
        n_steps = max(1, int(round(path_length / walk_cell)))
        pad = n_steps * factor + factor
        detail = f", averaged from {cell:g}" if factor > 1 else ""
        print(
            f"Walk length {path_length:g} working-CRS units "
            f"({n_steps} steps at cell size {walk_cell:g}{detail})"
        )

        for _, row in labeled.iterrows():
            node_id = int(row["node_id"])
            node_point = Point(float(row["x"]), float(row["y"]))
            line = _cross_section_line(cross_sections, node_id)
            seed = _lower_ridge_seed(row, line, node_point)
            chosen: Optional[RoutingPathSlope] = None
            chosen_transform = None
            chosen_side = "none"
            seed_x = np.nan
            seed_y = np.nan
            if seed is not None:
                chosen_side, seed_point = seed
                seed_x = float(seed_point.x)
                seed_y = float(seed_point.y)
                window = _read_window(src, seed_x, seed_y, pad)
                if window is not None:
                    elevation, chosen_transform, _, _ = window
                    elevation = aggregate_elevation(elevation, factor)
                    if factor > 1:
                        chosen_transform = chosen_transform * Affine.scale(factor, factor)
                    seed_col_f, seed_row_f = (~chosen_transform) * (seed_x, seed_y)
                    seed_row = int(round(seed_row_f))
                    seed_col = int(round(seed_col_f))
                    blocked = channel_mask(elevation.shape, chosen_transform, channel)
                    if blocked.shape == elevation.shape:
                        if 0 <= seed_row < blocked.shape[0] and 0 <= seed_col < blocked.shape[1]:
                            blocked[seed_row, seed_col] = False
                        elevation = elevation.copy()
                        elevation[blocked] = np.nan
                    channel_point = _channel_point(row, line, node_point)
                    chosen = routing_slope_at_seed(
                        elevation,
                        seed_row,
                        seed_col,
                        xres=abs(float(chosen_transform.a)),
                        yres=abs(float(chosen_transform.e)),
                        n_steps=n_steps,
                        n_trials=int(route.n_trials),
                        theta=float(route.theta),
                        alpha=float(route.alpha),
                        beta=float(route.beta),
                        frequency_threshold=float(route.frequency_threshold),
                        outward=_outward_pixel_vector(chosen_transform, channel_point, seed_point),
                        rng=rng,
                    )

            routing_slope = float(chosen.slope) if chosen is not None else float("nan")
            print(
                f"  node {node_id}: routing slope {routing_slope:.6g} "
                f"({chosen_side}, {0 if chosen is None else chosen.n_frequent_visits} frequent visits)"
            )
            dist_out = row["dist_out"] if pd.notna(row["dist_out"]) else np.nan
            slope_rows.append(
                {
                    "node_id": node_id,
                    "dist_out": dist_out,
                    "x": float(node_point.x),
                    "y": float(node_point.y),
                    "seed_x": seed_x,
                    "seed_y": seed_y,
                    "routing_side": chosen_side,
                    "routing_slope": routing_slope,
                    "modal_path_slope": float(chosen.modal_path_slope) if chosen is not None else np.nan,
                    "modal_path_length": float(chosen.modal_path_length) if chosen is not None else np.nan,
                    "n_frequent_cells": 0 if chosen is None else chosen.n_frequent_cells,
                    "n_frequent_visits": 0 if chosen is None else chosen.n_frequent_visits,
                    "n_steps": n_steps,
                    "path_length": path_length,
                }
            )
            if chosen is not None and chosen.path_rows.size >= 2 and chosen_transform is not None:
                xs, ys = xy(chosen_transform, chosen.path_rows, chosen.path_cols, offset="center")
                path_rows.append(
                    {
                        "node_id": node_id,
                        "slope": routing_slope,
                        "side": chosen_side,
                        "geometry": LineString(zip(xs, ys)),
                    }
                )
    return slope_rows, path_rows


def _cross_section_line(cross_sections: gpd.GeoDataFrame, node_id: int):
    match = cross_sections[cross_sections["node_id"] == node_id]
    if match.empty:
        return None
    geom = match.geometry.iloc[0]
    if geom is None or geom.is_empty or geom.geom_type != "LineString":
        return None
    return geom


def _lower_ridge_seed(row, line, node_point: Point) -> Optional[tuple[str, Point]]:
    """Crest of the lower ridge, as a point on the cross-section line."""
    if line is None:
        return None
    ridge_z = [
        _finite_or_nan(row["ridge1_elevation"]),
        _finite_or_nan(row["ridge2_elevation"]),
    ]
    floodplain_z = [
        _finite_or_nan(row["floodplain1_elevation"]),
        _finite_or_nan(row["floodplain2_elevation"]),
    ]
    har = alluvial_ridge_height(ridge_z, floodplain_z)
    use_bank1 = bool(choose_lower_ridge_bank(ridge_z[0:1], ridge_z[1:2], har[0:1], har[1:2])[0])
    if use_bank1:
        side, dist = "ridge1", row["ridge1_dist_along"]
    else:
        side, dist = "ridge2", row["ridge2_dist_along"]
    if not np.isfinite(dist):
        return None
    return side, _point_on_transect(line, node_point, float(dist), row["channel_dist_along"])


def _channel_point(row, line, node_point: Point) -> Point:
    """Labeled channel point on the transect, or the node if it is missing."""
    dist = _finite_or_nan(row["channel_dist_along"])
    if line is None or not np.isfinite(dist):
        return node_point
    return _point_on_transect(line, node_point, dist, dist)


def _outward_pixel_vector(transform, channel_point: Point, seed_point: Point):
    """Pixel ``(drow, dcol)`` from the channel to the ridge seed."""
    seed_col, seed_row = (~transform) * (float(seed_point.x), float(seed_point.y))
    channel_col, channel_row = (~transform) * (float(channel_point.x), float(channel_point.y))
    drow = float(seed_row - channel_row)
    dcol = float(seed_col - channel_col)
    if float(np.hypot(drow, dcol)) < 1e-6:
        return None
    return (drow, dcol)


def _point_on_transect(line, node_point: Point, dist_along: float, channel_dist_along) -> Point:
    """Place a label distance onto the cross-section line.

    GUI labels store distance along the line. Synthetic labels store the
    same offset on the channel pick and an arbitrary origin, so a distance
    outside the line is measured from the node by its offset from the channel.
    """
    length = float(line.length)
    if 0.0 <= dist_along <= length:
        return line.interpolate(dist_along)
    s_node = float(line.project(node_point))
    channel = _finite_or_nan(channel_dist_along)
    if not np.isfinite(channel):
        return line.interpolate(s_node)
    s = min(max(s_node + (dist_along - channel), 0.0), length)
    return line.interpolate(s)


def _finite_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _read_window(src, x: float, y: float, pad: int):
    row, col = src.index(x, y)
    if row < 0 or col < 0 or row >= src.height or col >= src.width:
        return None
    row_off = row - pad
    col_off = col - pad
    size = 2 * pad + 1
    window = Window(col_off, row_off, size, size)
    arr = src.read(1, window=window, boundless=True, fill_value=np.nan)
    arr = np.asarray(arr, dtype=float)
    nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr[arr == nodata] = np.nan
    transform = src.window_transform(window)
    return arr, transform, pad, pad
