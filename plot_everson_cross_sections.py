#!/usr/bin/env python3
"""Plot 2006 / 2013 / 2022 / 2024 cross-sections through Everson (ridge + floodplain)."""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import substring

from compare_lidar_2006_ridges import VRT_PATH as DTM_2006_PATH
from compare_lidar_2006_ridges import sample_nearest_valid
from compare_lidar_2013_ridges import (
    DEM_2022_VRT,
    DTM_PATH,
    PROFILES_DIR,
    ROOT,
    US_FT,
    XS_PATH,
    pick_from_2024_profile,
    sample_nearest,
    xy_on_line,
)

CENTERLINE = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack_data/GIS/Centerline_flowaccumulation/"
    "Centerline_flowaccumulation.shp"
)

EVERSON_LONLAT = (-122.3427, 48.9207)
# 54 = Everson crossing; 52–44 continue upstream (south/east of town).
NODES = (54, 52, 50, 48, 46, 44)
ZOOM_M = 500.0
OUT = ROOT / "data" / "Nooksack_Everson_2006_2013_2022_2024_xs"


def bank_ns(line) -> tuple[str, str]:
    mid = line.interpolate(0.5, normalized=True)
    p0 = line.coords[0]
    p1 = line.coords[-1]
    s = "N" if p0[1] >= mid.y else "S"
    e = "N" if p1[1] >= mid.y else "S"
    return s, e


def _peak_width_m(d: np.ndarray, z: np.ndarray, i: int, drop: float = 1.0) -> float:
    peak = float(z[i])
    left = i
    while left > 0 and np.isfinite(z[left]) and z[left] >= peak - drop:
        left -= 1
    right = i
    while right < len(z) - 1 and np.isfinite(z[right]) and z[right] >= peak - drop:
        right += 1
    return abs(float(d[right] - d[left]))


def broad_crest(d: np.ndarray, z: np.ndarray, d_ch: float, side: str) -> tuple[float, float]:
    """Highest point within 350 m that is at least ~30 m wide (not a mast/pile)."""
    if side == "left":
        m = np.isfinite(z) & (d < d_ch) & (d >= d_ch - 350.0)
    else:
        m = np.isfinite(z) & (d > d_ch) & (d <= d_ch + 350.0)
    if not np.any(m):
        return np.nan, np.nan
    zz = np.where(m, z, -np.inf)
    banned = np.zeros(z.shape, dtype=bool)
    for _ in range(8):
        zz = np.where(banned, -np.inf, zz)
        if not np.any(np.isfinite(zz) & (zz > -1e8) & m & ~banned):
            break
        i = int(np.nanargmax(zz))
        if _peak_width_m(d, z, i, drop=0.5) >= 20.0:
            return float(d[i]), float(z[i])
        peak = float(z[i])
        banned |= np.isfinite(z) & (np.abs(d - d[i]) < 35.0) & (z > peak - 1.5)
    i = int(np.nanargmax(np.where(m, z, -np.inf)))
    return float(d[i]), float(z[i])


def z_near(d_axis: np.ndarray, z: np.ndarray, x: float) -> float:
    if not np.isfinite(x) or z is None or d_axis.size == 0:
        return np.nan
    i = int(np.argmin(np.abs(d_axis - x)))
    return float(z[i]) if np.isfinite(z[i]) else np.nan


def profile_for_node(node_id: int, rec, dem06, dem13, dem22, to_2927) -> dict:
    line = rec.geometry
    length = float(line.length)
    mid = 0.5 * length
    picks = pick_from_2024_profile(node_id, line)
    if picks is None:
        raise SystemExit(f"No 2024 profile picks for node {node_id}")

    prof = pd.read_csv(PROFILES_DIR / f"cross_section_{node_id:03d}.csv")
    d24 = prof["distance_m"].to_numpy(dtype=float)
    z24 = pd.to_numeric(prof["elevation_m"], errors="coerce").to_numpy(dtype=float) * US_FT
    d_ch = picks["channel_dist_along"] - mid
    d24_th = d24 - d_ch
    d_l, z_l = broad_crest(d24, z24, d_ch, "left")
    d_r, z_r = broad_crest(d24, z24, d_ch, "right")

    step = 1.0
    dists_along = np.arange(0.0, length + step, step)
    d_th = dists_along - picks["channel_dist_along"]
    keep = np.abs(d_th) <= ZOOM_M + 25.0
    dists_along = dists_along[keep]
    d_th = d_th[keep]
    coords = [xy_on_line(line, d) for d in dists_along]
    xs22, ys22 = to_2927.transform([c[0] for c in coords], [c[1] for c in coords])
    z06 = sample_nearest_valid(dem06, coords) if dem06 is not None else np.full(len(coords), np.nan)
    z13 = sample_nearest(dem13, coords)
    z22 = sample_nearest(dem22, list(zip(xs22, ys22))) * US_FT

    ns1, ns2 = bank_ns(line)
    fp1_x = picks["floodplain1_dist_along"] - picks["channel_dist_along"]
    fp2_x = picks["floodplain2_dist_along"] - picks["channel_dist_along"]
    return {
        "node_id": node_id,
        "dist_out_km": float(rec["dist_out"]) / 1000.0,
        "picks": picks,
        "d_th": d_th,
        "z06": z06,
        "z13": z13,
        "z22": z22,
        "d24_th": d24_th,
        "z24": z24,
        "ns1": ns1,
        "ns2": ns2,
        "crest1_x": d_l - d_ch,
        "crest2_x": d_r - d_ch,
        "fp1_x": fp1_x,
        "fp2_x": fp2_x,
        "z06_r1": z_near(d_th, z06, d_l - d_ch),
        "z13_r1": z_near(d_th, z13, d_l - d_ch),
        "z22_r1": z_near(d_th, z22, d_l - d_ch),
        "z24_r1": z_l,
        "z06_r2": z_near(d_th, z06, d_r - d_ch),
        "z13_r2": z_near(d_th, z13, d_r - d_ch),
        "z22_r2": z_near(d_th, z22, d_r - d_ch),
        "z24_r2": z_r,
        "z06_f1": z_near(d_th, z06, fp1_x),
        "z13_f1": z_near(d_th, z13, fp1_x),
        "z22_f1": z_near(d_th, z22, fp1_x),
        "z24_f1": picks["floodplain1_z24_ft"] * US_FT,
        "z06_f2": z_near(d_th, z06, fp2_x),
        "z13_f2": z_near(d_th, z13, fp2_x),
        "z22_f2": z_near(d_th, z22, fp2_x),
        "z24_f2": picks["floodplain2_z24_ft"] * US_FT,
    }


def plot_one(ax, P: dict) -> None:
    d_th, z06, z13, z22 = P["d_th"], P["z06"], P["z13"], P["z22"]
    d24_th, z24 = P["d24_th"], P["z24"]
    m06 = np.isfinite(z06) & (np.abs(d_th) <= ZOOM_M)
    m13 = np.isfinite(z13) & (np.abs(d_th) <= ZOOM_M)
    m22 = np.isfinite(z22) & (np.abs(d_th) <= ZOOM_M)
    m24 = np.isfinite(z24) & (np.abs(d24_th) <= ZOOM_M)
    ax.plot(d_th[m06], z06[m06], color="#6b3fa0", lw=1.5, label="2006 lidar (ground, no bathy)")
    ax.plot(d_th[m13], z13[m13], color="#b36b00", lw=1.5, label="2013 lidar (ground, no bathy)")
    ax.plot(d_th[m22], z22[m22], color="#8a8a8a", lw=1.3, label="2022 topobathy")
    ax.plot(d24_th[m24], z24[m24], color="black", lw=1.5, label="2024 topobathy")
    ax.axvline(0.0, color="0.75", lw=0.7, ls=":")

    def mark(x, y, marker, color, text, dx):
        if not np.isfinite(x) or not np.isfinite(y) or abs(x) > ZOOM_M:
            return
        ax.scatter([x], [y], s=26, c=color, marker=marker, zorder=5, edgecolors="white", linewidths=0.4)
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            text,
            (x, y),
            xytext=(x + dx, y + 0.22),
            fontsize=7.5,
            color=color,
            ha=ha,
            va="bottom",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5),
        )

    mark(P["crest1_x"], P["z24_r1"], "v", "#1f4e79", f"{P['ns1']} levee", -80)
    mark(P["crest2_x"], P["z24_r2"], "v", "#1f4e79", f"{P['ns2']} levee", 80)
    mark(P["fp1_x"], P["z24_f1"], "s", "#2e7d4f", f"{P['ns1']} floodplain", -90)
    mark(P["fp2_x"], P["z24_f2"], "s", "#2e7d4f", f"{P['ns2']} floodplain", 90)

    ax.set_xlim(-ZOOM_M, ZOOM_M)
    s_levee = P["z24_r1"] if P["ns1"] == "S" else P["z24_r2"]
    s_fp = P["z24_f1"] if P["ns1"] == "S" else P["z24_f2"]
    s_z06 = P["z06_r1"] if P["ns1"] == "S" else P["z06_r2"]
    s_z13 = P["z13_r1"] if P["ns1"] == "S" else P["z13_r2"]
    s_fp06 = P["z06_f1"] if P["ns1"] == "S" else P["z06_f2"]
    s_fp13 = P["z13_f1"] if P["ns1"] == "S" else P["z13_f2"]
    bits = [f"Node {P['node_id']}   {P['dist_out_km']:.2f} km from outlet"]
    if np.isfinite(s_levee) and np.isfinite(s_z06):
        bits.append(f"S levee {s_z06:.2f}→{s_levee:.2f} m")
    elif np.isfinite(s_levee) and np.isfinite(s_z13):
        bits.append(f"S levee {s_z13:.2f}→{s_levee:.2f} m")
    if np.isfinite(s_fp) and np.isfinite(s_fp06):
        bits.append(f"S FP {s_fp06:.2f}→{s_fp:.2f} m")
    elif np.isfinite(s_fp) and np.isfinite(s_fp13):
        bits.append(f"S FP {s_fp13:.2f}→{s_fp:.2f} m")
    ax.set_title("   ·  ".join(bits), fontsize=9)
    ax.grid(True, alpha=0.25)


def locator(ax, xs, everson_xy, selected) -> None:
    ex, ey = everson_xy
    cl = gpd.read_file(CENTERLINE).to_crs(26910)
    for geom in cl.geometry:
        if geom is None:
            continue
        x = (np.asarray(geom.xy[0]) - ex) / 1000.0
        y = (np.asarray(geom.xy[1]) - ey) / 1000.0
        ax.plot(x, y, color="0.45", lw=1.6, zorder=2)

    everson = Point(ex, ey)
    for _, rec in xs.iterrows():
        nid = int(rec.node_id)
        line = rec.geometry
        mid = line.interpolate(0.5, normalized=True)
        if mid.distance(everson) > 4500:
            continue
        mid_d = 0.5 * float(line.length)
        seg = substring(line, max(0.0, mid_d - 550.0), min(float(line.length), mid_d + 550.0))
        x = (np.asarray(seg.xy[0]) - ex) / 1000.0
        y = (np.asarray(seg.xy[1]) - ey) / 1000.0
        if nid in selected:
            ax.plot(x, y, color="#1f4e79", lw=1.8, zorder=4)
            ax.text(x[len(x) // 2], y[len(y) // 2] + 0.08, str(nid), fontsize=8, color="#1f4e79", ha="center")
        else:
            ax.plot(x, y, color="0.78", lw=0.6, zorder=1)

    ax.scatter([0], [0], s=48, c="#c0392b", marker="*", zorder=5)
    ax.annotate("Everson", (0, 0), xytext=(0.12, 0.08), fontsize=9, color="#c0392b")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-2.2, 0.7)
    ax.set_aspect("equal")
    ax.annotate("upstream / south of town", xy=(0.35, -1.95), fontsize=8, color="0.35", ha="left")
    ax.set_title("Everson and upstream  ·  river runs south of town", fontsize=10)
    ax.set_xlabel("km east of Everson")
    ax.set_ylabel("km north of Everson")
    ax.grid(True, alpha=0.25)
    ax.plot([], [], color="#6b3fa0", lw=1.5, label="2006 lidar (ground, no bathy)")
    ax.plot([], [], color="#b36b00", lw=1.5, label="2013 lidar (ground, no bathy)")
    ax.plot([], [], color="#8a8a8a", lw=1.3, label="2022 topobathy")
    ax.plot([], [], color="black", lw=1.5, label="2024 topobathy")
    ax.legend(frameon=False, loc="lower left", fontsize=8)


def fmt(v: float) -> str:
    return "—" if not np.isfinite(v) else f"{v:.2f}"


def main() -> None:
    xs = gpd.read_file(XS_PATH)
    xs["node_id"] = xs["node_id"].astype(int)
    if xs.crs is None or str(xs.crs) != "EPSG:26910":
        xs = xs.to_crs(26910)
    to_ll = Transformer.from_crs("EPSG:4326", "EPSG:26910", always_xy=True)
    everson_xy = to_ll.transform(*EVERSON_LONLAT)
    to_2927 = Transformer.from_crs("EPSG:26910", "EPSG:2927", always_xy=True)

    profiles = []
    if not DTM_2006_PATH.exists():
        raise SystemExit(f"Missing 2006 DTM {DTM_2006_PATH}")
    with rasterio.open(DTM_2006_PATH) as dem06, rasterio.open(DTM_PATH) as dem13, rasterio.open(DEM_2022_VRT) as dem22:
        for nid in NODES:
            rec = xs.loc[xs["node_id"] == nid].iloc[0]
            profiles.append(profile_for_node(nid, rec, dem06, dem13, dem22, to_2927))

    fig = plt.figure(figsize=(13.2, 12.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.05, 1.15, 1.15, 1.15], hspace=0.40, wspace=0.24)
    ax_map = fig.add_subplot(gs[0, :])
    locator(ax_map, xs, everson_xy, set(NODES))
    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
    ]
    for ax, P in zip(axes, profiles):
        plot_one(ax, P)
        ax.set_xlabel("Distance from 2024 thalweg (m)")
        ax.set_ylabel("Elevation (m NAVD88)")

    fig.suptitle("Upstream of Everson (south of town): ridge and floodplain, 2006–2024", fontsize=13, y=0.995)
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}.png")

    rows = []
    print("\nEverson nodes  (levee = local max at least 20 m wide at 0.5 m below peak)")
    print(
        f"{'node':>5} {'km':>6} {'feature':<14} {'z06':>6} {'z13':>6} {'z22':>6} {'z24':>6} "
        f"{'06→13':>7} {'13→22':>7} {'22→24':>7} {'06→24':>7}"
    )
    for P in profiles:
        items = [
            (f"levee {P['ns1']}", P["z06_r1"], P["z13_r1"], P["z22_r1"], P["z24_r1"]),
            (f"levee {P['ns2']}", P["z06_r2"], P["z13_r2"], P["z22_r2"], P["z24_r2"]),
            (f"floodplain {P['ns1']}", P["z06_f1"], P["z13_f1"], P["z22_f1"], P["z24_f1"]),
            (f"floodplain {P['ns2']}", P["z06_f2"], P["z13_f2"], P["z22_f2"], P["z24_f2"]),
        ]
        for name, z06, z13, z22, z24 in items:
            d0613, d1322, d2224, d0624 = z13 - z06, z22 - z13, z24 - z22, z24 - z06
            print(
                f"{P['node_id']:5d} {P['dist_out_km']:6.2f} {name:<14} "
                f"{fmt(z06):>6} {fmt(z13):>6} {fmt(z22):>6} {fmt(z24):>6} "
                f"{fmt(d0613):>7} {fmt(d1322):>7} {fmt(d2224):>7} {fmt(d0624):>7}"
            )
            rows.append(
                {
                    "node_id": P["node_id"],
                    "dist_out_km": round(P["dist_out_km"], 2),
                    "feature": name,
                    "z2006_m": None if not np.isfinite(z06) else round(float(z06), 3),
                    "z2013_m": None if not np.isfinite(z13) else round(float(z13), 3),
                    "z2022_m": None if not np.isfinite(z22) else round(float(z22), 3),
                    "z2024_m": None if not np.isfinite(z24) else round(float(z24), 3),
                    "dz_2006_2013_m": None if not np.isfinite(d0613) else round(float(d0613), 3),
                    "dz_2013_2022_m": None if not np.isfinite(d1322) else round(float(d1322), 3),
                    "dz_2022_2024_m": None if not np.isfinite(d2224) else round(float(d2224), 3),
                    "dz_2006_2024_m": None if not np.isfinite(d0624) else round(float(d0624), 3),
                }
            )
    csv = ROOT / "data" / "Nooksack_Everson_ridge_floodplain_change.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"Wrote {csv}")


if __name__ == "__main__":
    main()
