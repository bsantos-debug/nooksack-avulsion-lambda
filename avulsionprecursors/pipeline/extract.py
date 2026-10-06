"""Extract perpendicular DEM profiles along a river centerline."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.exceptions import InputValidationError
from avulsionprecursors.geometry.transects import (
    construct_cross_section_line,
    half_lengths_to_valley_walls,
    is_valid_width,
    perpendicular_azimuths,
    smooth_centerline_for_azimuth,
    snap_to_centerline,
    thalweg_signed_slopes_at_points,
    width_from_polygon,
)
from avulsionprecursors.io.prepare import PreparedInputs, prepare_inputs
from avulsionprecursors.io.raster import mask_nodata
from avulsionprecursors.io.validation import validate_config
from avulsionprecursors.io.vectors import load_vector, merge_lines, merge_polygons


def extract_cross_sections(
    cfg: WorkflowConfig,
    prepared: PreparedInputs | None = None,
) -> gpd.GeoDataFrame:
    """Build stations, transects, slopes, and DEM profiles.

    Required inputs
    ---------------
    cfg.paths.dem : elevation raster (any GDAL format rasterio can read)
    cfg.paths.centerline : line vector of the river

    Units
    -----
    Horizontal distances: working CRS units (usually metres after reprojection).
    Elevations: native DEM values (not converted).

    Outputs
    -------
    Shapefiles of cross-sections and centerline points, plus one CSV profile
    per station in ``outputs/profiles``.
    """
    validate_config(cfg)
    if prepared is None:
        prepared = prepare_inputs(cfg)

    cfg.output_dir().mkdir(parents=True, exist_ok=True)
    cfg.profiles_dir().mkdir(parents=True, exist_ok=True)

    gdf = load_vector(prepared.centerline_path, "centerline")
    river = merge_lines(gdf)
    crs = gdf.crs
    ext = cfg.extract

    river_azimuth = smooth_centerline_for_azimuth(
        river, ext.centerline_smooth_window, ext.centerline_smooth_sample
    )
    gpd.GeoDataFrame({"id": [1]}, geometry=[river_azimuth], crs=crs).to_file(
        cfg.smoothed_centerline_path()
    )
    print(
        f"Azimuth centerline smoothed with {ext.centerline_smooth_window:g} "
        f"{prepared.horizontal_unit} Savitzky–Golay window → {cfg.smoothed_centerline_path()}"
    )

    points, point_ids, widths, slopes, dist_outs = _station_points(
        river, crs, cfg, prepared
    )

    if not Path(prepared.dem_path).exists():
        raise InputValidationError(f"Prepared DEM missing:\n  {prepared.dem_path}")

    channel_union = None
    if prepared.channel_polygon_path is not None:
        print(f"Channel polygon: {prepared.channel_polygon_path}")
        channel_polys = load_vector(prepared.channel_polygon_path, "channel polygon")
        channel_union = merge_polygons(channel_polys)

    print(
        f"Sm from thalweg long profile (min z in ±{ext.thalweg_half_width:g} "
        f"{prepared.horizontal_unit} strip; SG window {ext.slope_smooth_window:g})"
    )
    with rasterio.open(prepared.dem_path) as dem:
        cell = max(abs(float(dem.res[0])), abs(float(dem.res[1])), 1.0)
        sm = thalweg_signed_slopes_at_points(
            points,
            river,
            dem,
            sample_step=ext.slope_sample_step,
            smooth_window=ext.slope_smooth_window,
            thalweg_half=ext.thalweg_half_width,
            channel_polygon=channel_union,
            azimuth_line=river_azimuth,
            transect_step=cell,
        )
    slopes = [float(v) if np.isfinite(v) else np.nan for v in sm]
    n_down = sum(1 for v in slopes if np.isfinite(v) and v > 0)
    n_adv = sum(1 for v in slopes if np.isfinite(v) and v <= 0)
    n_miss = sum(1 for v in slopes if not np.isfinite(v))
    print(f"Signed thalweg Sm: {n_down} downhill, {n_adv} adverse/zero, {n_miss} missing")

    perp_azimuths = perpendicular_azimuths(
        points, river, river_azimuth, ext.flow_tangent_half_delta
    )
    print(
        f"Cross-section normals from smoothed flow azimuth "
        f"(SG window {ext.centerline_smooth_window:g}; circular mean of node ± "
        f"neighbors; tangent Δ = ±{ext.flow_tangent_half_delta:g})"
    )

    if channel_union is not None:
        print(f"Calculating widths from channel polygon: {prepared.channel_polygon_path}")
        probe_half = max(ext.cross_half_length, 500.0)
        n_w = 0
        for idx, (pt, perp) in enumerate(zip(points, perp_azimuths)):
            probe_line = construct_cross_section_line(pt, perp, probe_half)
            width_val = width_from_polygon(probe_line, channel_union)
            if is_valid_width(width_val):
                widths[idx] = width_val
                n_w += 1
        print(f"Calculated {n_w} widths from channel polygon")
        if n_w < len(points):
            print(f"Warning: {len(points) - n_w} points did not intersect the channel polygon")
    else:
        print("No channel polygon provided; widths left empty unless present on the points file")

    points_gdf = gpd.GeoDataFrame(
        {"node_id": point_ids, "dist_out": dist_outs, "width": widths, "slope": slopes},
        geometry=points,
        crs=crs,
    )

    print(
        f"Clipping transects at valley walls "
        f"(probe ±{ext.max_half_length:g}, Δz ≥ {ext.valley_wall_dz:g} native Z, "
        f"min half {ext.cross_half_length:g})"
    )
    cross_sections = []
    with rasterio.open(prepared.dem_path) as dem_walls:
        n_hit = 0
        for idx, pt in enumerate(points):
            half_plus, half_minus = half_lengths_to_valley_walls(
                dem_walls,
                pt,
                perp_azimuths[idx],
                max_half=ext.max_half_length,
                min_half=float(ext.cross_half_length),
                dz=ext.valley_wall_dz,
                persist=ext.valley_wall_persist,
                skip=ext.valley_wall_skip_channel,
                keep=ext.valley_wall_keep,
                channel_window=ext.channel_elev_window,
            )
            if half_plus < ext.max_half_length - 1.0 or half_minus < ext.max_half_length - 1.0:
                n_hit += 1
            cross_sections.append(
                construct_cross_section_line(pt, perp_azimuths[idx], half_plus, half_minus)
            )
    print(f"Valley wall hit on at least one side: {n_hit}/{len(points)}")

    cross_gdf = gpd.GeoDataFrame(
        {"node_id": point_ids, "dist_out": dist_outs, "width": widths, "slope": slopes},
        geometry=cross_sections,
        crs=crs,
    )
    cross_gdf.to_file(cfg.cross_sections_path())
    points_gdf.to_file(cfg.centerline_points_path())
    print(f"Wrote {len(cross_sections)} cross-sections → {cfg.cross_sections_path()}")

    _write_profiles(cfg, prepared, cross_sections, point_ids)
    return cross_gdf


def _station_points(river, crs, cfg: WorkflowConfig, prepared: PreparedInputs):
    points = []
    point_ids = []
    widths = []
    slopes = []
    dist_outs = []
    src = cfg.paths.centerline_points
    if src is not None and Path(src).exists():
        points_gdf = gpd.read_file(src).to_crs(crs)
        if points_gdf.empty:
            raise InputValidationError(f"No points found in {src}")
        for idx, row in points_gdf.iterrows():
            snapped = snap_to_centerline(row.geometry, river)
            points.append(snapped)
            point_ids.append(row.get("node_id", idx + 1))
            widths.append(row.get("width"))
            slopes.append(row.get("slope"))
            along = river.project(snapped)
            dist_outs.append(river.length - along)
    else:
        distance = 0.0
        pid = 1
        spacing = cfg.extract.spacing
        while distance <= river.length:
            snapped = river.interpolate(distance)
            points.append(snapped)
            point_ids.append(pid)
            widths.append(None)
            slopes.append(None)
            dist_outs.append(river.length - distance)
            distance += spacing
            pid += 1
    if not points:
        raise InputValidationError("No centerline stations were created. Check spacing vs. line length.")
    return points, point_ids, widths, slopes, dist_outs


def _write_profiles(cfg: WorkflowConfig, prepared: PreparedInputs, cross_sections, point_ids) -> None:
    ext = cfg.extract
    sample_step = prepared.sample_step
    masked_count = 0
    with rasterio.open(prepared.dem_path) as dem:
        for line, node_id in zip(cross_sections, point_ids):
            num = max(2, int(line.length / sample_step))
            distances = np.linspace(0, line.length, num)
            coords_line = list(line.coords)
            if len(coords_line) >= 3:
                s_channel = float(line.project(Point(coords_line[1])))
            else:
                s_channel = line.length / 2.0
            distances_centered = distances - s_channel
            points_along = [line.interpolate(d) for d in distances]
            coords = [(p.x, p.y) for p in points_along]
            elev = np.array([e[0] for e in dem.sample(coords)], dtype=float)
            elev = mask_nodata(elev, dem.nodata)

            if ext.max_elev_above_channel is not None:
                near = elev[np.abs(distances_centered) <= ext.channel_elev_window]
                near_valid = near[np.isfinite(near)]
                if near_valid.size > 0:
                    channel_ref = float(np.median(near_valid))
                    ceiling = channel_ref + ext.max_elev_above_channel
                    too_high = np.isfinite(elev) & (elev > ceiling)
                    masked_count += int(too_high.sum())
                    elev = np.where(too_high, np.nan, elev)

            df = pd.DataFrame(
                {
                    "distance": distances_centered,
                    "elevation": elev,
                    # Compatibility with existing label CSVs / older profiles
                    "distance_m": distances_centered,
                    "elevation_m": elev,
                }
            )
            csv_path = cfg.profiles_dir() / f"cross_section_{int(node_id):03d}.csv"
            df.to_csv(csv_path, index=False)

    print(f"Saved {len(cross_sections)} elevation profiles in {cfg.profiles_dir()}")
    if ext.max_elev_above_channel is not None:
        print(
            f"Masked {masked_count} samples > {ext.max_elev_above_channel:g} "
            f"native Z above near-channel elevation "
            f"(±{ext.channel_elev_window:g} horizontal units)"
        )
