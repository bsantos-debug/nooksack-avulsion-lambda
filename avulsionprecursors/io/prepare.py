"""Align DEM, centerline, and optional masks into one working CRS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.exceptions import CRSError, InputValidationError
from avulsionprecursors.io.raster import RasterInfo, describe_raster, write_reprojected_dem
from avulsionprecursors.io.validation import linear_unit_name, parse_crs, require_projected
from avulsionprecursors.io.vectors import (
    assign_source_crs,
    horizontal_crs,
    load_vector,
    merge_lines,
    merge_polygons,
)


@dataclass
class PreparedInputs:
    """Reprojected, optionally clipped inputs ready for extraction."""

    dem_path: Path
    centerline_path: Path
    channel_polygon_path: Optional[Path]
    clip_path: Optional[Path]
    working_crs: CRS
    raster: RasterInfo
    horizontal_unit: str
    vertical_unit: str
    sample_step: float


def _utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6) + 1
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _centroid_lonlat(gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    geom = gdf.unary_union.centroid
    src = horizontal_crs(gdf.crs)
    if src is None:
        raise CRSError("Cannot choose a UTM zone because the centerline has no CRS.")
    if src.is_geographic:
        return float(geom.x), float(geom.y)
    to_ll = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(geom.x, geom.y)
    return float(lon), float(lat)


def resolve_target_crs(cfg: WorkflowConfig, centerline: gpd.GeoDataFrame) -> CRS:
    target = str(cfg.crs.target).strip()
    if target.upper() == "UTM":
        lon, lat = _centroid_lonlat(centerline)
        crs = _utm_crs_from_lonlat(lon, lat)
        print(f"Working CRS: UTM zone from data centroid → {crs.to_string()} (≈ {lon:.3f}°, {lat:.3f}°)")
        return crs
    crs = parse_crs(target, "crs.target")
    require_projected(crs, "crs.target")
    print(f"Working CRS: {crs.to_string()}")
    return crs


def _write_vector(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Shapefile layer name from stem
    gdf.to_file(path)
    return path


def prepare_inputs(cfg: WorkflowConfig) -> PreparedInputs:
    """Reproject (and optionally clip) inputs into a single projected CRS.

    Elevations stay in the DEM's native vertical units.
    """
    centerline_gdf = load_vector(cfg.paths.centerline, "centerline")
    centerline_gdf = assign_source_crs(
        centerline_gdf,
        cfg.crs.centerline_source or cfg.crs.vector_source,
        "centerline",
    )
    centerline_gdf = centerline_gdf.set_crs(
        horizontal_crs(centerline_gdf.crs), allow_override=True
    )

    info = describe_raster(cfg.paths.dem)
    dem_crs = info.crs
    if dem_crs is None:
        if not cfg.crs.dem_source:
            raise CRSError(
                f"The DEM has no coordinate reference system:\n  {cfg.paths.dem}\n"
                "Assign a CRS in GIS software, or set crs.dem_source "
                "(for example EPSG:32610) in the config."
            )
        dem_crs = parse_crs(cfg.crs.dem_source, "crs.dem_source")
    dem_crs = horizontal_crs(dem_crs)

    working_crs = resolve_target_crs(cfg, centerline_gdf)

    clip_geom = None
    clip_path = None
    if cfg.paths.clip_boundary is not None:
        clip_gdf = load_vector(cfg.paths.clip_boundary, "clip boundary")
        clip_gdf = assign_source_crs(
            clip_gdf, cfg.crs.vector_source, "clip boundary"
        )
        clip_gdf = clip_gdf.to_crs(working_crs)
        clip_geom = merge_polygons(clip_gdf)
        clip_path = _write_vector(clip_gdf, cfg.prepared_clip_path())

    cfg.prepared_dir().mkdir(parents=True, exist_ok=True)
    dem_out = cfg.prepared_dem_path()
    raster = write_reprojected_dem(
        cfg.paths.dem,
        dem_out,
        working_crs,
        src_crs=dem_crs,
        clip_geom=clip_geom,
    )

    centerline_out_gdf = centerline_gdf.to_crs(working_crs)
    line = merge_lines(centerline_out_gdf)
    dem_box = box(*raster.bounds)
    if line.envelope.disjoint(dem_box) and not line.intersects(dem_box):
        raise InputValidationError(
            "The centerline does not overlap the DEM after reprojection.\n"
            "Check that both files cover the same river reach and that source "
            "CRS values are correct."
        )
    centerline_path = _write_vector(
        gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs=working_crs),
        cfg.prepared_centerline_path(),
    )

    channel_path = None
    if cfg.paths.channel_polygon is not None:
        poly = load_vector(cfg.paths.channel_polygon, "channel polygon")
        poly = assign_source_crs(poly, cfg.crs.vector_source, "channel polygon")
        poly = poly.to_crs(working_crs)
        channel_path = _write_vector(poly, cfg.prepared_channel_polygon_path())

    sample_step = cfg.extract.sample_step
    if sample_step is None:
        sample_step = max(min(raster.cell_size_x, raster.cell_size_y), 1e-6)

    horizontal_unit = linear_unit_name(working_crs) or "unknown"
    vertical_unit = cfg.vertical_unit
    print(
        f"DEM cell size: {raster.cell_size_x:.4g} × {raster.cell_size_y:.4g} {horizontal_unit}"
    )
    print(f"Profile sample step: {sample_step:.4g} {horizontal_unit}")
    print(
        f"Vertical values: native DEM units "
        f"({vertical_unit}; thresholds are in these units, not converted)."
    )
    if vertical_unit not in ("native",) and horizontal_unit:
        if vertical_unit in ("m",) and "foot" in horizontal_unit:
            print(
                "Warning: vertical unit is metres but the working CRS is in feet. "
                "Slope is (DEM Z) / (horizontal CRS units)."
            )
        if vertical_unit in ("ft", "us_ft") and "metre" in horizontal_unit:
            print(
                "Warning: vertical unit is feet but the working CRS is in metres. "
                "Slope is (DEM Z) / (horizontal CRS units). Lambda stays "
                "dimensionless if SAR and Sm use the same mixed units."
            )

    return PreparedInputs(
        dem_path=dem_out,
        centerline_path=centerline_path,
        channel_polygon_path=channel_path,
        clip_path=clip_path,
        working_crs=working_crs,
        raster=raster,
        horizontal_unit=horizontal_unit,
        vertical_unit=vertical_unit,
        sample_step=float(sample_step),
    )
