#!/usr/bin/env python3
"""Replace DEM bed slope with FlowFM water-surface slope and recompute Λ.

Keeps the existing labeled Har / SAR / β from data/Nooksack_lambda_results.csv
and recomputes γ = SAR / Sm_flowfm and Λ = γ β.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, LogNorm, Normalize
from matplotlib.ticker import FuncFormatter
from netCDF4 import Dataset
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent
DEM_PATH = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/"
    "Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
)
CL_PATH = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/"
    "Nooksack_data/GIS/Centerline_flowaccumulation/Centerline_flowaccumulation.shp"
)
NC_PATH = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/"
    "Nooksack_data/Model_outputs/FlowFM_merged_map.nc"
)
POINTS_PATH = ROOT / "centerline_points_from_xs.shp"
IN_CSV = ROOT / "data" / "Nooksack_lambda_results.csv"
OUT_CSV = ROOT / "data" / "Nooksack_lambda_results_flowfm.csv"
WINDOW_M = 5000.0
SLOPE_MIN = 1e-6


def _horizontal_crs(crs):
    try:
        if crs is not None and crs.is_compound and getattr(crs, "sub_crs_list", None):
            return crs.sub_crs_list[0]
    except Exception:
        pass
    return crs


def ls_slope(s, z):
    m = np.isfinite(s) & np.isfinite(z)
    s, z = s[m], z[m]
    if len(s) < 8:
        return np.nan
    ds = s - s.mean()
    denom = np.dot(ds, ds)
    if denom < 1e-12:
        return np.nan
    return abs(float(np.dot(ds, z - z.mean()) / denom))


def flowfm_slopes(line, pts: gpd.GeoDataFrame) -> np.ndarray:
    with Dataset(NC_PATH) as ds:
        x = np.asarray(ds["mesh2d_face_x"][:], dtype=float)
        y = np.asarray(ds["mesh2d_face_y"][:], dtype=float)
        var = ds["mesh2d_s1"]
        wl = np.asarray(var[-1, :], dtype=float)
        fill = getattr(var, "_FillValue", None)
        if fill is not None:
            wl = np.where(wl == fill, np.nan, wl)
    tree = cKDTree(np.column_stack([x, y]))
    step = 50.0
    s_line = np.arange(0.0, float(line.length) + 1e-6, step)
    xy = np.array([line.interpolate(float(s)).coords[0] for s in s_line])
    dist, idx = tree.query(xy, k=1)
    wse = np.where(dist < 75.0, wl[idx], np.nan)
    half = 0.5 * WINDOW_M
    out = []
    for p in pts.geometry:
        s0 = float(line.project(Point(p.x, p.y)))
        sel = (s_line >= s0 - half) & (s_line <= s0 + half)
        sm = ls_slope(s_line[sel], wse[sel])
        if np.isfinite(sm):
            sm = max(sm, SLOPE_MIN)
        out.append(sm)
    return np.asarray(out, dtype=float)


def line_collection(xy, z, cmap, norm, lw, **kwargs):
    ok = np.isfinite(xy).all(axis=1) & np.isfinite(z)
    xy, z = xy[ok], z[ok]
    segs = np.concatenate([xy[:-1, None, :], xy[1:, None, :]], axis=1)
    c = np.sqrt(np.clip(z[:-1], 1e-6, None) * np.clip(z[1:], 1e-6, None))
    lc = LineCollection(
        segs, cmap=cmap, norm=norm, linewidths=lw, capstyle="round", joinstyle="round", **kwargs
    )
    lc.set_array(c)
    return lc


def read_dem(bounds, pad_m, target_px):
    minx, miny, maxx, maxy = bounds
    minx -= pad_m
    miny -= pad_m
    maxx += pad_m
    maxy += pad_m
    with rasterio.open(DEM_PATH) as src:
        win = from_bounds(minx, miny, maxx, maxy, src.transform).round_offsets().round_lengths()
        w, h = max(int(win.width), 1), max(int(win.height), 1)
        scale = max(w, h) / float(target_px)
        out_w, out_h = max(int(w / scale), 40), max(int(h / scale), 40)
        data = src.read(
            1,
            window=win,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
            boundless=True,
            fill_value=src.nodata,
        ).astype(float)
        transform = src.window_transform(win)
        transform = transform * transform.scale(w / out_w, h / out_h)
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


def hillshade(z):
    ls = LightSource(azdeg=315, altdeg=45)
    finite = z[np.isfinite(z)]
    lo, hi = np.nanpercentile(finite, [8, 92])
    normed = np.clip((z - lo) / (hi - lo + 1e-6), 0, 1)
    fill = float(np.nanmedian(normed[np.isfinite(normed)]))
    rgb = np.asarray(
        ls.shade(np.nan_to_num(normed, nan=fill), cmap=plt.cm.terrain, blend_mode="soft", vert_exag=5, dx=1, dy=1)
    )
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb[~np.isfinite(z)] = 0.93
    return rgb


def add_colorbar(fig, ax, sm, label, ticks=None, href=None):
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.035, extend="both")
    cbar.set_label(label, fontsize=11)
    vmin, vmax = sm.norm.vmin, sm.norm.vmax
    if ticks is None:
        ticks = [t for t in [0.5, 1, 2, 5, 10, 20, 50, 100, 200] if vmin <= t <= vmax]
    else:
        ticks = [t for t in ticks if vmin <= t <= vmax]
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(t) for t in ticks])
    if href is not None and vmin <= href <= vmax:
        cbar.ax.axhline(href, color="white", lw=1.4, zorder=3)
        cbar.ax.axhline(href, color="black", lw=0.6, zorder=4)
    return cbar


def plot_dem_map(g, river, values, cmap, norm, out: Path, title: str, cbar_label: str, ticks=None, href=None):
    xy = np.column_stack([g.geometry.x.to_numpy(), g.geometry.y.to_numpy()])
    zdem, extent = read_dem(river.bounds, pad_m=2500, target_px=1800)
    rgb = hillshade(zdem)
    fig, ax = plt.subplots(figsize=(9.2, 11.2), facecolor="white")
    ax.imshow(rgb, extent=extent, origin="upper", zorder=0)
    ax.plot(*river.xy, color="0.15", lw=5.0, alpha=0.35, solid_capstyle="round", zorder=2)
    ax.add_collection(line_collection(xy, np.asarray(values, float), cmap, norm, lw=3.6))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v/1000:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v/1000:.0f}"))
    ax.set_xlabel("Easting (km, UTM 10N)")
    ax.set_ylabel("Northing (km)")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_colorbar(fig, ax, sm, cbar_label, ticks=ticks, href=href)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_satellite_map(g, river, values, cmap, norm, out: Path, title: str, cbar_label: str, ticks=None, href=None):
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt

    class EsriImagery(cimgt.GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{z}/{y}/{x}"
            )

    g84 = g.to_crs(4326)
    river84 = gpd.GeoSeries([river], crs=g.crs).to_crs(4326).iloc[0]
    xy = np.column_stack([g84.geometry.x.to_numpy(), g84.geometry.y.to_numpy()])
    minx, miny, maxx, maxy = river84.bounds
    pad_x = (maxx - minx) * 0.08
    pad_y = (maxy - miny) * 0.08
    extent = [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]
    tiles = EsriImagery()
    fig = plt.figure(figsize=(9.2, 11.2), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection=tiles.crs)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(tiles, 12)
    ax.plot(*river84.xy, color="k", lw=5.0, alpha=0.35, transform=ccrs.PlateCarree(), solid_capstyle="round", zorder=2)
    ax.add_collection(
        line_collection(xy, np.asarray(values, float), cmap, norm, lw=3.6, transform=ccrs.PlateCarree())
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.8", linestyle="--", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    add_colorbar(fig, ax, sm, cbar_label, ticks=ticks, href=href)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    lam = pd.read_csv(IN_CSV)
    pts = gpd.read_file(POINTS_PATH)
    cl = gpd.read_file(CL_PATH)
    cl = cl.set_crs(_horizontal_crs(cl.crs), allow_override=True)
    pts = pts.set_crs(_horizontal_crs(pts.crs), allow_override=True)
    if pts.crs != cl.crs:
        pts = pts.to_crs(cl.crs)
    river = cl.geometry.iloc[0]
    pts = pts.sort_values("dist_out")
    print("sampling FlowFM WSE slope, window", WINDOW_M, "m")
    pts["slope_flowfm"] = flowfm_slopes(river, pts)

    lam = lam.copy()
    if "slope_dem" not in lam.columns:
        lam["slope_dem"] = lam["slope"]
    lam["slope_flowfm"] = np.interp(
        lam["dist_out"].to_numpy(float),
        pts["dist_out"].to_numpy(float),
        pts["slope_flowfm"].to_numpy(float),
    )
    sm = lam["slope_flowfm"].clip(lower=SLOPE_MIN)
    lam["slope"] = sm
    lam["slope_source"] = "flowfm_wse_5km"
    g1 = lam["SAR_ridge1"].astype(float) / sm
    g2 = lam["SAR_ridge2"].astype(float) / sm
    lam["gamma_mean"] = np.nanmean(np.vstack([g1.to_numpy(), g2.to_numpy()]), axis=0)
    lam["lambda"] = lam["gamma_mean"] * lam["superelevation_mean"]
    lam["flag_valid_lambda"] = lam["lambda"].notna() & (lam["lambda"] > 0)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    lam.to_csv(OUT_CSV, index=False)

    v = lam[lam["flag_valid_lambda"] == True]  # noqa: E712
    print(f"wrote {OUT_CSV}")
    print(
        f"valid n={len(v)}  median Λ={v['lambda'].median():.2f}  "
        f"mean={v['lambda'].mean():.1f}  max={v['lambda'].max():.1f}  "
        f"pct>=2={100*(v['lambda']>=2).mean():.0f}%"
    )

    from calculate_lambda_bathymetry import plot_lambda_vs_distance

    plot_lambda_vs_distance(lam, "Nooksack (FlowFM Sm)", ROOT / "data" / "Nooksack_lambda_flowfm_plot.png")

    g = pts.copy()
    vmap = lam.loc[lam["lambda"].notna() & (lam["lambda"] > 0), ["dist_out", "lambda"]].sort_values("dist_out")
    g["lambda"] = np.interp(
        g["dist_out"].to_numpy(float),
        vmap["dist_out"].to_numpy(float),
        vmap["lambda"].to_numpy(float),
        left=np.nan,
        right=np.nan,
    )
    z = g["lambda"].to_numpy(float)
    z = z[np.isfinite(z) & (z > 0)]
    vmin = max(float(np.nanpercentile(z, 5)), 0.2)
    vmax = max(float(np.nanpercentile(z, 95)), vmin * 2)
    cmap = plt.get_cmap("plasma")
    norm = LogNorm(vmin=vmin, vmax=vmax)
    title = "Nooksack River  ·  Λ with FlowFM water-surface slope"
    plot_dem_map(
        g, river, g["lambda"].to_numpy(float), cmap, norm,
        ROOT / "data" / "Nooksack_lambda_on_dem.png",
        title, "Avulsion potential Λ", href=2.0,
    )
    plot_satellite_map(
        g, river, g["lambda"].to_numpy(float), cmap, norm,
        ROOT / "data" / "Nooksack_lambda_on_satellite.png",
        title, "Avulsion potential Λ", href=2.0,
    )


def plot_superelevation_maps():
    lam = pd.read_csv(OUT_CSV if OUT_CSV.exists() else IN_CSV)
    pts = gpd.read_file(POINTS_PATH)
    cl = gpd.read_file(CL_PATH)
    cl = cl.set_crs(_horizontal_crs(cl.crs), allow_override=True)
    pts = pts.set_crs(_horizontal_crs(pts.crs), allow_override=True)
    if pts.crs != cl.crs:
        pts = pts.to_crs(cl.crs)
    river = cl.geometry.iloc[0]
    pts = pts.sort_values("dist_out")
    vmap = lam.loc[
        lam["superelevation_mean"].notna() & (lam["superelevation_mean"] > 0),
        ["dist_out", "superelevation_mean"],
    ].sort_values("dist_out")
    g = pts.copy()
    g["beta"] = np.interp(
        g["dist_out"].to_numpy(float),
        vmap["dist_out"].to_numpy(float),
        vmap["superelevation_mean"].to_numpy(float),
        left=np.nan,
        right=np.nan,
    )
    z = g["beta"].to_numpy(float)
    z = z[np.isfinite(z)]
    vmin = 0.0
    vmax = float(np.nanpercentile(z, 98))
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=vmin, vmax=vmax)
    title = "Nooksack River  ·  superelevation β = Har / Hm"
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    plot_dem_map(
        g, river, g["beta"].to_numpy(float), cmap, norm,
        ROOT / "data" / "Nooksack_superelevation_on_dem.png",
        title, "Superelevation β", ticks=ticks,
    )
    plot_satellite_map(
        g, river, g["beta"].to_numpy(float), cmap, norm,
        ROOT / "data" / "Nooksack_superelevation_on_satellite.png",
        title, "Superelevation β", ticks=ticks,
    )


if __name__ == "__main__":
    main()
