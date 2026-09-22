#!/usr/bin/env python3
"""Map numbered cross-sections on the Nooksack DEM.

Usage:
    poetry run python plot_cross_sections_on_dem.py
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import cm, patheffects as pe
from matplotlib.colors import LightSource
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import LineString, Point, box

DEFAULT_DEM = (
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/"
    "Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
)
DEFAULT_CENTERLINE = (
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/"
    "Nooksack_data/GIS/Centerline_flowaccumulation/Centerline_flowaccumulation.shp"
)
DEFAULT_XS = "cross_sections.shp"
DEFAULT_OUT = "data/Nooksack_cross_sections_numbered_on_dem.png"


def _horizontal_crs(crs):
    if crs is None:
        return crs
    try:
        if crs.is_compound and getattr(crs, "sub_crs_list", None):
            return crs.sub_crs_list[0]
    except Exception:
        pass
    return crs


def _load_line(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf.set_crs(_horizontal_crs(gdf.crs), allow_override=True)
    return gdf


def _transect_frame(geom: LineString) -> tuple[Point, np.ndarray]:
    mid = geom.interpolate(0.5, normalized=True)
    coords = np.asarray(geom.coords, dtype=float)
    vec = coords[-1] - coords[0]
    norm = float(np.hypot(vec[0], vec[1]))
    if norm < 1e-6:
        vec = np.array([1.0, 0.0])
    else:
        vec = vec / norm
    return mid, vec


def _short_tick(mid: Point, vec: np.ndarray, half_m: float) -> LineString:
    return LineString(
        [
            (mid.x + half_m * vec[0], mid.y + half_m * vec[1]),
            (mid.x - half_m * vec[0], mid.y - half_m * vec[1]),
        ]
    )


def read_dem(dem_path: Path, bounds, pad_m: float, target_px: int):
    minx, miny, maxx, maxy = bounds
    minx -= pad_m
    miny -= pad_m
    maxx += pad_m
    maxy += pad_m
    with rasterio.open(dem_path) as src:
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        win = win.round_offsets().round_lengths()
        width = max(int(win.width), 1)
        height = max(int(win.height), 1)
        scale = max(width, height) / float(target_px)
        out_w = max(int(width / scale), 40)
        out_h = max(int(height / scale), 40)
        data = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
            boundless=True,
            fill_value=src.nodata,
        ).astype(float)
        transform = src.window_transform(win)
        transform = transform * transform.scale(width / out_w, height / out_h)
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    data = np.where(np.isfinite(data) & (np.abs(data) < 1e10), data, np.nan)
    extent = [
        transform.c,
        transform.c + transform.a * data.shape[1],
        transform.f + transform.e * data.shape[0],
        transform.f,
    ]
    return data, extent


def hillshade_rgb(z: np.ndarray) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    finite = z[np.isfinite(z)]
    if finite.size < 10:
        return np.full(z.shape + (3,), 0.92)
    lo, hi = np.nanpercentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    fill = float(np.nanmedian(norm[np.isfinite(norm)]))
    rgb = np.asarray(
        ls.shade(
            np.nan_to_num(norm, nan=fill),
            cmap=cm.terrain,
            blend_mode="soft",
            vert_exag=4.0,
            dx=1,
            dy=1,
        )
    )
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb[~np.isfinite(z)] = 0.92
    return rgb


def _plot_centerline(ax, river, clip_bounds=None, lw=1.2):
    geom = river
    if clip_bounds is not None:
        geom = river.intersection(box(*clip_bounds))
    if geom.is_empty:
        return
    parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
    for part in parts:
        if part.geom_type == "LineString" and not part.is_empty:
            ax.plot(*part.xy, color="#08306b", lw=lw, zorder=4)


def _label_nodes(ax, rows, offset_m: float, fontsize: int, every: int = 1):
    stroke = [pe.withStroke(linewidth=2.4, foreground="white")]
    n = len(rows)
    for i, (nid, mid, vec) in enumerate(rows):
        if every > 1 and nid % every != 0 and nid not in (rows[0][0], rows[-1][0]):
            continue
        side = 1.0 if (i // 2) % 2 == 0 else -1.0
        x = mid.x + side * offset_m * vec[0]
        y = mid.y + side * offset_m * vec[1]
        ax.text(
            x,
            y,
            str(nid),
            fontsize=fontsize,
            color="#67000d",
            ha="center",
            va="center",
            zorder=6,
            path_effects=stroke,
            fontweight="bold",
        )


def plot_panel(ax, dem_path, river, rows, pad_m, target_px, tick_half, label_every, title):
    xs = [mid.x for _, mid, _ in rows]
    ys = [mid.y for _, mid, _ in rows]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    z, extent = read_dem(dem_path, bounds, pad_m=pad_m, target_px=target_px)
    ax.imshow(hillshade_rgb(z), extent=extent, origin="upper", zorder=0)
    clip = (extent[0], extent[2], extent[1], extent[3])
    _plot_centerline(ax, river, clip_bounds=clip, lw=1.35)
    for _, mid, vec in rows:
        tick = _short_tick(mid, vec, tick_half)
        ax.plot(*tick.xy, color="#e31a1c", lw=0.7, alpha=0.9, zorder=5)
        ax.plot(mid.x, mid.y, ".", color="#2171b5", ms=2.5, zorder=5)
    offset = tick_half + 80.0
    _label_nodes(ax, rows, offset_m=offset, fontsize=7 if label_every == 1 else 8, every=label_every)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Easting (m, NAD83 / UTM 10N)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)


def main() -> None:
    root = Path(__file__).resolve().parent
    xs_path = root / DEFAULT_XS
    cl_path = Path(DEFAULT_CENTERLINE)
    dem_path = Path(DEFAULT_DEM)
    out_path = root / DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = _load_line(xs_path)
    cl = _load_line(cl_path)
    if xs.crs and cl.crs and xs.crs != cl.crs:
        xs = xs.to_crs(cl.crs)
    river = cl.geometry.iloc[0]
    xs = xs.sort_values("node_id")

    rows = []
    for _, rec in xs.iterrows():
        geom = rec.geometry
        if geom is None or geom.is_empty:
            continue
        mid, vec = _transect_frame(geom)
        rows.append((int(rec["node_id"]), mid, vec))

    n = len(rows)
    n_panels = 4
    chunk = int(np.ceil(n / n_panels))
    print(f"Mapping {n} numbered cross-sections")

    fig = plt.figure(figsize=(12.5, 16.5), facecolor="white")
    gs = fig.add_gridspec(
        1 + n_panels,
        1,
        height_ratios=[1.35] + [1.0] * n_panels,
        hspace=0.28,
        left=0.07,
        right=0.98,
        top=0.95,
        bottom=0.03,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    plot_panel(
        ax0,
        dem_path,
        river,
        rows,
        pad_m=800,
        target_px=2200,
        tick_half=380,
        label_every=10,
        title=f"Overview  ·  {n} cross-sections  ·  labels every 10",
    )

    for i in range(n_panels):
        subset = rows[i * chunk : (i + 1) * chunk]
        if not subset:
            continue
        ax = fig.add_subplot(gs[i + 1, 0])
        lo, hi = subset[0][0], subset[-1][0]
        plot_panel(
            ax,
            dem_path,
            river,
            subset,
            pad_m=450,
            target_px=1600,
            tick_half=280,
            label_every=1,
            title=f"Cross-sections {lo}–{hi}",
        )

    fig.suptitle(
        "Nooksack numbered cross-sections on DEM (new flow-accumulation centerline)",
        fontsize=13,
        fontweight="semibold",
        y=0.985,
    )
    fig.savefig(out_path, dpi=170)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
