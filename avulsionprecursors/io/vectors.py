"""Vector loading, CRS assignment, and simple geometry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

from avulsionprecursors.exceptions import CRSError, InputValidationError


def load_vector(path: Union[str, Path], kind: str = "vector") -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise InputValidationError(
            f"Could not find the {kind} file:\n  {path}\n"
            "Check the path in your config. Relative paths are resolved from "
            "the folder that contains the config file."
        )
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        raise InputValidationError(
            f"Could not read the {kind} file:\n  {path}\n"
            f"Expected a vector dataset such as a shapefile (.shp) or GeoPackage (.gpkg).\n"
            f"Details: {exc}"
        ) from exc
    if gdf.empty:
        raise InputValidationError(f"The {kind} file has no features:\n  {path}")
    return gdf


def assign_source_crs(gdf: gpd.GeoDataFrame, source: Optional[str], label: str) -> gpd.GeoDataFrame:
    if gdf.crs is not None:
        return gdf
    if not source:
        raise CRSError(
            f"The {label} has no coordinate reference system (CRS).\n"
            "A CRS tells the program whether coordinates are metres, feet, or "
            "longitude/latitude. In GIS, assign a CRS to the file, or set "
            f"the matching source CRS in the config (for example EPSG:32610)."
        )
    try:
        crs = CRS.from_user_input(source)
    except Exception as exc:
        raise CRSError(
            f"Could not understand the source CRS {source!r} for {label}.\n"
            "Use an EPSG code such as EPSG:32610.\n"
            f"Details: {exc}"
        ) from exc
    return gdf.set_crs(crs, allow_override=True)


def horizontal_crs(crs) -> Optional[CRS]:
    """Drop a compound vertical CRS so outputs are 2D projected coordinates."""
    if crs is None:
        return None
    crs_obj = CRS.from_user_input(crs)
    try:
        if crs_obj.is_compound and crs_obj.sub_crs_list:
            return CRS.from_user_input(crs_obj.sub_crs_list[0])
    except Exception:
        pass
    return crs_obj


def as_single_linestring(geom) -> LineString:
    """Collapse MultiLineString parts into one LineString (longest part if unmergeable)."""
    if geom is None or geom.is_empty:
        raise InputValidationError("Centerline geometry is empty.")
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else list(geom.geoms)
        return max(parts, key=lambda g: g.length)
    raise InputValidationError(
        f"The centerline must be a line (LineString), not {geom.geom_type}."
    )


def merge_lines(gdf: gpd.GeoDataFrame) -> LineString:
    if len(gdf) == 1:
        return as_single_linestring(gdf.geometry.iloc[0])
    return as_single_linestring(linemerge(unary_union(gdf.geometry)))


def merge_polygons(gdf: gpd.GeoDataFrame):
    return unary_union(gdf.geometry)
