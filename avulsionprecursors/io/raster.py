"""DEM / raster loading, CRS, NoData, cell size, and reprojection.

Elevation values are left in the raster's native units. Reprojection changes
only the horizontal coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as raster_mask
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping

from avulsionprecursors.exceptions import CRSError, InputValidationError

SUPPORTED_RASTER_SUFFIXES = {
    ".tif",
    ".tiff",
    ".vrt",
    ".img",
    ".asc",
    ".nc",
    ".grd",
}


@dataclass
class RasterInfo:
    """Summary of a raster used for validation and messages."""

    path: Path
    crs: Optional[CRS]
    bounds: Tuple[float, float, float, float]
    width: int
    height: int
    cell_size_x: float
    cell_size_y: float
    nodata: Optional[float]
    dtype: str
    band_count: int
    transform: rasterio.Affine


def open_raster(path: Union[str, Path]) -> rasterio.DatasetReader:
    path = Path(path)
    try:
        return rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise InputValidationError(
            f"Could not open the raster file:\n  {path}\n"
            "Check that the file exists and is a supported raster format "
            f"(GeoTIFF .tif, .vrt, .img, .asc, …).\nDetails: {exc}"
        ) from exc


def describe_raster(path: Union[str, Path]) -> RasterInfo:
    path = Path(path)
    with open_raster(path) as src:
        crs = CRS.from_user_input(src.crs) if src.crs else None
        return RasterInfo(
            path=path,
            crs=crs,
            bounds=tuple(src.bounds),
            width=src.width,
            height=src.height,
            cell_size_x=abs(src.transform.a),
            cell_size_y=abs(src.transform.e),
            nodata=src.nodata,
            dtype=str(src.dtypes[0]),
            band_count=src.count,
            transform=src.transform,
        )


def elevation_valid(elev: float, nodata) -> bool:
    """True if a raster sample is usable (not NaN / inf / NoData / extreme)."""
    if elev is None:
        return False
    try:
        value = float(elev)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value) or abs(value) >= 1e6:
        return False
    if nodata is None:
        return True
    try:
        nd = float(nodata)
    except (TypeError, ValueError):
        return True
    if np.isnan(nd):
        return True
    return value != nd


def mask_nodata(elev: np.ndarray, nodata) -> np.ndarray:
    """Replace NoData and non-finite values with NaN."""
    out = np.asarray(elev, dtype=float)
    out = np.where(np.isfinite(out) & (np.abs(out) < 1e6), out, np.nan)
    if nodata is None:
        return out
    try:
        nd = float(nodata)
    except (TypeError, ValueError):
        return out
    if np.isnan(nd):
        return out
    return np.where(out == nd, np.nan, out)


def write_reprojected_dem(
    src_path: Path,
    dst_path: Path,
    dst_crs: CRS,
    src_crs: Optional[CRS] = None,
    clip_geom=None,
    resampling: Resampling = Resampling.bilinear,
) -> RasterInfo:
    """Reproject a DEM to ``dst_crs``, optionally clipping to a polygon.

    Pixel values (elevations) are resampled but not unit-converted.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open_raster(src_path) as src:
        working_src_crs = src_crs or (CRS.from_user_input(src.crs) if src.crs else None)
        if working_src_crs is None:
            raise CRSError(
                f"The DEM has no coordinate reference system:\n  {src_path}\n"
                "A CRS says whether the coordinates are metres, feet, or "
                "longitude/latitude. Assign a CRS in GIS software, or set "
                "crs.dem_source in the config (for example EPSG:32610)."
            )
        if src.count < 1:
            raise InputValidationError(f"The DEM has no bands:\n  {src_path}")

        transform, width, height = calculate_default_transform(
            working_src_crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        nodata = src.nodata
        if nodata is None:
            nodata = np.nan
        meta = src.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "count": 1,
                "nodata": nodata if np.isfinite(nodata) else -9999.0,
                "compress": "lzw",
            }
        )
        # Convert integer DEMs to float so NoData/NaN masking is safe
        if np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
            meta["dtype"] = "float32"

        destination = np.full((height, width), meta["nodata"], dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=working_src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=meta["nodata"],
            resampling=resampling,
        )

    if clip_geom is not None:
        # Write then mask so rasterio.mask can use the new grid
        with rasterio.open(dst_path, "w", **meta) as dst:
            dst.write(destination, 1)
        with rasterio.open(dst_path) as tmp:
            try:
                clipped, clip_transform = raster_mask(
                    tmp,
                    [mapping(clip_geom)],
                    crop=True,
                    nodata=meta["nodata"],
                    filled=True,
                )
            except ValueError as exc:
                raise InputValidationError(
                    "The clip boundary does not overlap the DEM. "
                    "Check that the polygon covers the river reach and uses "
                    f"the same area as the DEM.\nDetails: {exc}"
                ) from exc
        meta.update(
            {
                "transform": clip_transform,
                "height": clipped.shape[1],
                "width": clipped.shape[2],
            }
        )
        destination = clipped[0]

    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(np.asarray(destination, dtype=meta["dtype"]), 1)

    return describe_raster(dst_path)


def cell_size(info: RasterInfo) -> float:
    """Representative cell size in working CRS units (mean of |dx|, |dy|)."""
    return float(0.5 * (abs(info.cell_size_x) + abs(info.cell_size_y)))
