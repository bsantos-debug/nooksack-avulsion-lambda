#!/usr/bin/env python3
"""Plan-view and river-coordinate maps of 2006→2013 floodplain deposition."""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin, xy as transform_xy
from rasterio.warp import reproject
from scipy.spatial import cKDTree
from shapely.geometry import box
from shapely.ops import linemerge, substring, unary_union

from compare_lidar_2006_ridges import VRT_PATH as DTM_2006
from compare_lidar_2013_ridges import DTM_PATH as DTM_2013
from compare_lidar_2013_ridges import ROOT, XS_PATH

CENTERLINE = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack_data/GIS/Centerline_flowaccumulation/"
    "Centerline_flowaccumulation.shp"
)
BELT_CSV = ROOT / "data" / "Nooksack_floodplain_belt_change.csv"
OUT = ROOT / "data" / "Nooksack_floodplain_deposition_2006_2013"

EVERSON_LONLAT = (-122.3427, 48.9207)
RES_M = 5.0
BUFFER_M = 650.0
CHANNEL_M = 90.0
FP_MAX_M = 600.0
DOD_CLIP_M = 2.0  # drop classification/building outliers from the color stretch


def load_centerline() -> tuple:
    cl = gpd.read_file(CENTERLINE).to_crs(26910)
    geom = unary_union(list(cl.geometry))
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    if geom.geom_type == "MultiLineString":
        geom = max(geom.geoms, key=lambda g: g.length)
    return geom, float(geom.length)


def warp_to_grid(src: rasterio.io.DatasetReader, west, north, height, width, transform):
    out = np.full((height, width), -9999.0, dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs="EPSG:26910",
        resampling=Resampling.average,
        src_nodata=src.nodata,
        dst_nodata=-9999.0,
    )
    out = out.astype(np.float64)
    out[out <= -9990] = np.nan
    return out


def hillshade_gray(z: np.ndarray) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    finite = z[np.isfinite(z)]
    if finite.size < 50:
        return np.full(z.shape + (3,), 0.92)
    lo, hi = np.nanpercentile(finite, [5, 95])
    normed = np.clip((z - lo) / (hi - lo + 1e-6), 0, 1)
    fill = float(np.nanmedian(normed[np.isfinite(normed)]))
    rgb = np.asarray(
        ls.shade(
            np.nan_to_num(normed, nan=fill),
            cmap=plt.cm.gray,
            blend_mode="soft",
            vert_exag=6.0,
            dx=RES_M,
            dy=RES_M,
        )
    )
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb[~np.isfinite(z)] = 0.93
    return rgb


def dod_cmap():
    # blue (erosion) — white — brown/orange (deposition)
    return mcolors.LinearSegmentedColormap.from_list(
        "fp_dod",
        ["#2166ac", "#67a9cf", "#d1e5f0", "#ffffff", "#fddbc7", "#ef8a62", "#b2182b"],
        N=256,
    )


def plot_river(ax, cl, everson_xy, xs, color="#1a1a1a", lw=1.0):
    ax.plot(*cl.xy, color=color, lw=lw, zorder=5)
    ax.scatter([everson_xy[0]], [everson_xy[1]], s=42, c="#c0392b", marker="*", zorder=6)
    for nid in (44, 54):
        rec = xs.loc[xs["node_id"] == nid]
        if rec.empty:
            continue
        line = rec.iloc[0].geometry
        mid_d = 0.5 * float(line.length)
        seg = substring(line, max(0.0, mid_d - 500.0), min(float(line.length), mid_d + 500.0))
        ax.plot(*seg.xy, color="#1f4e79", lw=0.8, zorder=4)


def main() -> None:
    cl, cl_len = load_centerline()
    corridor = cl.buffer(BUFFER_M)
    xs = gpd.read_file(XS_PATH)
    xs["node_id"] = xs["node_id"].astype(int)
    if xs.crs is None or str(xs.crs) != "EPSG:26910":
        xs = xs.to_crs(26910)
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True)
    everson_xy = to_xy.transform(*EVERSON_LONLAT)

    with rasterio.open(DTM_2006) as s06, rasterio.open(DTM_2013) as s13:
        west = max(s06.bounds.left, corridor.bounds[0])
        south = max(s06.bounds.bottom, corridor.bounds[1])
        east = min(s06.bounds.right, corridor.bounds[2])
        north = min(s06.bounds.top, corridor.bounds[3])
        west, south, east, north = [np.floor(v / RES_M) * RES_M for v in (west, south, east, north)]
        width = int(np.ceil((east - west) / RES_M))
        height = int(np.ceil((north - south) / RES_M))
        transform = from_origin(west, north, RES_M, RES_M)
        print(f"Warping {width}×{height} at {RES_M:.0f} m")
        z06 = warp_to_grid(s06, west, north, height, width, transform)
        z13 = warp_to_grid(s13, west, north, height, width, transform)

    dod = z13 - z06
    inside = ~geometry_mask([corridor], out_shape=dod.shape, transform=transform, invert=False)
    dod = np.where(inside, dod, np.nan)
    dod = np.where(np.abs(dod) > DOD_CLIP_M, np.nan, dod)

    rows, cols = np.where(np.isfinite(dod))
    xs_m, ys_m = transform_xy(transform, rows, cols)
    cell_xy = np.column_stack([np.asarray(xs_m), np.asarray(ys_m)])

    n_cl = int(cl_len / 10.0) + 1
    cl_s = np.linspace(0.0, cl_len, n_cl)
    cl_xy = np.array([[cl.interpolate(s).x, cl.interpolate(s).y] for s in cl_s])
    tangent = np.gradient(cl_xy, axis=0)
    tree = cKDTree(cl_xy)
    dist, idx = tree.query(cell_xy)
    v = cell_xy - cl_xy[idx]
    cross = tangent[idx, 0] * v[:, 1] - tangent[idx, 1] * v[:, 0]
    n_coord = np.sign(np.where(cross == 0, 1.0, cross)) * dist
    s_up = cl_s[idx]  # 0 at upstream end
    dist_out_m = cl_len - s_up

    keep_fp = (dist >= CHANNEL_M) & (dist <= FP_MAX_M)
    dod_fp = dod.copy()
    dod_fp[rows[~keep_fp], cols[~keep_fp]] = np.nan

    hs = hillshade_gray(z13)
    cmap = dod_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.50)

    s_bin, n_bin = 150.0, 20.0
    fp_cells = keep_fp & np.isfinite(dod[rows, cols])
    d_out = dist_out_m[fp_cells]
    n_c = n_coord[fp_cells]
    dz = dod[rows, cols][fp_cells]
    km0, km1 = 15.0, 58.0
    n0, n1 = -FP_MAX_M, FP_MAX_M
    n_km = int(np.floor((km1 - km0) * 1000 / s_bin))
    n_n = int(np.floor((n1 - n0) / n_bin))
    unwrap = np.full((n_n, n_km), np.nan)
    si = np.floor((d_out - km0 * 1000) / s_bin).astype(int)
    ni = np.floor((n_c - n0) / n_bin).astype(int)
    ok = (si >= 0) & (si < n_km) & (ni >= 0) & (ni < n_n)
    tmp = pd.DataFrame({"si": si[ok], "ni": ni[ok], "dz": dz[ok]})
    med = tmp.groupby(["ni", "si"], sort=False)["dz"].median()
    for (ni_i, si_i), val in med.items():
        unwrap[int(ni_i), int(si_i)] = val

    ex, ey = everson_xy
    hs_extent_km = [(west - ex) / 1000, (east - ex) / 1000, (south - ey) / 1000, (north - ey) / 1000]

    fig = plt.figure(figsize=(14.2, 11.6))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1.15], width_ratios=[1.05, 1.12], hspace=0.32, wspace=0.28
    )
    ax_all = fig.add_subplot(gs[0, 0])
    ax_ev = fig.add_subplot(gs[0, 1])
    ax_un = fig.add_subplot(gs[1, :])

    def show_map(ax, xlim, ylim, title):
        ax.imshow(hs, extent=hs_extent_km, origin="upper", zorder=0)
        ax.imshow(
            dod_fp,
            extent=hs_extent_km,
            origin="upper",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            zorder=2,
            alpha=0.88,
        )
        ax.plot(
            (np.asarray(cl.xy[0]) - ex) / 1000.0,
            (np.asarray(cl.xy[1]) - ey) / 1000.0,
            color="#1a1a1a",
            lw=1.1,
            zorder=5,
        )
        ax.scatter([0], [0], s=48, c="#c0392b", marker="*", zorder=6)
        for nid in (44, 54):
            rec = xs.loc[xs["node_id"] == nid]
            if rec.empty:
                continue
            line = rec.iloc[0].geometry
            mid_d = 0.5 * float(line.length)
            seg = substring(line, max(0.0, mid_d - 500.0), min(float(line.length), mid_d + 500.0))
            ax.plot(
                (np.asarray(seg.xy[0]) - ex) / 1000.0,
                (np.asarray(seg.xy[1]) - ey) / 1000.0,
                color="#1f4e79",
                lw=0.9,
                zorder=4,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("km east of Everson")
        ax.set_ylabel("km north of Everson")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2)

    show_map(ax_all, (-16.5, 5.5), (-11.5, 4.5), "Mainstem floodplain  ·  2013 minus 2006")
    ax_all.annotate("Everson", (0.15, 0.15), fontsize=9, color="#c0392b")
    ax_all.annotate("upstream", xy=(3.2, -6.8), fontsize=8, color="0.35")
    ax_all.plot([], [], color="#b2182b", lw=6, alpha=0.75, label="deposition")
    ax_all.plot([], [], color="#2166ac", lw=6, alpha=0.75, label="erosion")
    ax_all.legend(frameon=False, loc="lower left", fontsize=8)

    show_map(ax_ev, (-1.4, 2.6), (-6.6, 0.9), "Everson, looking at the floodplain south of town")
    ax_ev.annotate("Everson", (0.08, 0.12), fontsize=9, color="#c0392b")
    ax_ev.annotate("south floodplain\n(flooded side)", xy=(0.15, -1.55), fontsize=8, color="#1f4e79", ha="left")

    unwrap_extent = [km0, km0 + n_km * s_bin / 1000.0, n0, n0 + n_n * n_bin]
    im_u = ax_un.imshow(
        unwrap,
        extent=unwrap_extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
        zorder=2,
    )
    ax_un.axhline(0.0, color="0.15", lw=0.8, zorder=3)
    ax_un.axvline(46.63, color="#c0392b", lw=1.0, ls="--", zorder=3)
    ax_un.annotate("Everson", (46.63, 530), fontsize=9, color="#c0392b", ha="center")
    ax_un.axhspan(-CHANNEL_M, CHANNEL_M, color="0.85", alpha=0.55, zorder=1)
    ax_un.set_xlim(57.5, 16.0)
    ax_un.set_ylim(-FP_MAX_M, FP_MAX_M)
    ax_un.set_xlabel("Distance from outlet (km)   ·   upstream is to the left")
    ax_un.set_ylabel("Distance from channel (m)\nleft of flow   ←     →   right of flow")
    ax_un.set_title(
        "Unwrapped floodplain: ~18 cm sheet on both banks; thicker patches near 20–25 km and 40–45 km",
        fontsize=11,
    )
    ax_un.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.07, right=0.88, top=0.93, bottom=0.08)
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.68])
    cbar = fig.colorbar(im_u, cax=cax, extend="both")
    cbar.set_label("2013 − 2006  (m)\nred = deposition")
    fig.suptitle("Nooksack floodplain deposition, 2006–2013 lidar", fontsize=14, y=0.995)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.png")

    for lo, hi, name in [
        (-600, -200, "left far FP"),
        (-200, -90, "left near FP"),
        (90, 200, "right near FP"),
        (200, 600, "right far FP"),
    ]:
        m = (n_c >= lo) & (n_c < hi)
        if m.any():
            print(f"  {name:16s} n={int(m.sum()):7d}  median Δz={np.median(dz[m]):+.3f} m")


if __name__ == "__main__":
    main()
