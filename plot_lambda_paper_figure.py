#!/usr/bin/env python3
"""
Paper-style multi-panel figure for Λ spatial structure (cf. example Fig. A–F).

Panels:
  A  — Λ vs distance from outlet, raw + SG filters at L_c and L_Λ, Λ=2 line
  B  — LISA classes on L_c-smoothed Λ
  C  — experimental semivariogram + Robert model, L_c and L_Λ/2 marked
  D  — length-scale histogram (single-river marker if only one system)
  E  — L_c vs L_Λ (single-river marker)
  F  — map of Λ along centerline (satellite basemap when available)

Usage:
    poetry run python plot_lambda_paper_figure.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

from analyze_lambda_hotspots import (
    empirical_variogram,
    fit_robert_variogram,
    flag_hh_bordered_by_lh,
    load_lambda,
    robert_model,
    run_lisa,
    savgol_smooth_lambda,
)

DEFAULT_LAMBDA_CSV = "data/Nooksack_lambda_results.csv"
DEFAULT_POINTS = "centerline_points_from_xs.shp"
DEFAULT_OUT = "data/hotspots/Nooksack_paper_style_figure.png"
DEFAULT_RIVER = "Nooksack"


def _sg_at_scale(dist_down_m: np.ndarray, y: np.ndarray, scale_m: float) -> np.ndarray:
    """Savitzky–Golay along downstream distance with window ≈ scale_m."""
    order = np.argsort(dist_down_m)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    d = dist_down_m[order]
    yy = y[order]
    dx = float(np.median(np.diff(d)))
    if not np.isfinite(dx) or dx <= 0:
        dx = 200.0
    win = int(round(scale_m / dx))
    if win % 2 == 0:
        win += 1
    win = int(np.clip(win, 5, len(yy) - (1 - len(yy) % 2)))
    if win % 2 == 0:
        win -= 1
    win = max(5, win)
    poly = 2 if win > 5 else 1
    ys = savgol_filter(yy, window_length=win, polyorder=poly, mode="interp")
    return ys[inv]


def build_figure(
    df: pd.DataFrame,
    lisa_df: pd.DataFrame,
    vario: pd.DataFrame,
    fit: dict,
    points_path: Path,
    river_name: str,
    output_path: Path,
) -> Path:
    L_c = float(fit["LC"])          # 3r — characteristic correlation length
    L_lam = float(fit["L_lambda"])  # l1 — primary wavelength

    # Smooth at both scales (paper panel A)
    df = df.copy()
    df["lam_Lc"] = _sg_at_scale(df["dist_down_m"].to_numpy(), df["lambda"].to_numpy(), L_c)
    df["lam_Llam"] = _sg_at_scale(df["dist_down_m"].to_numpy(), df["lambda"].to_numpy(), L_lam)

    # Distance from outlet (km) — paper x-axis
    df["dist_out_km"] = df["dist_out"] / 1000.0
    # Prefer freshly computed L_c smooth for panel B
    lisa_df = lisa_df.drop(columns=[c for c in ("dist_out_km", "lam_Lc") if c in lisa_df.columns])
    lisa_df = lisa_df.merge(
        df[["node_id", "dist_out_km", "lam_Lc"]], on="node_id", how="left"
    )

    # Layout: A B C on top row conceptually via gridspec; D E small; F map
    fig = plt.figure(figsize=(14, 11), dpi=200)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.05, 1.35], hspace=0.35, wspace=0.3)

    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[1, 0:2])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 2])
    axE = fig.add_subplot(gs[2, 2])
    axF = fig.add_subplot(gs[2, 0:2], projection=ccrs.PlateCarree())

    # ----- Panel A -----
    ax = axA
    ax.scatter(
        df["dist_out_km"], df["lambda"],
        s=10, c="#9ecae1", alpha=0.85, zorder=2, label="Λ",
    )
    # Longer filter in red, shorter in blue (match paper color language)
    if L_lam >= L_c:
        ax.plot(df["dist_out_km"], df["lam_Llam"], color="#c0392b", lw=2.0,
                label=fr"${L_lam/1000:.0f}$ km filter $L_\Lambda$")
        ax.plot(df["dist_out_km"], df["lam_Lc"], color="#1a5276", lw=1.8,
                label=fr"${L_c/1000:.0f}$ km filter $L_c$")
    else:
        # Nooksack: L_c > L_Λ — still label by definition, color by size
        ax.plot(df["dist_out_km"], df["lam_Lc"], color="#c0392b", lw=2.0,
                label=fr"${L_c/1000:.0f}$ km filter $L_c$")
        ax.plot(df["dist_out_km"], df["lam_Llam"], color="#1a5276", lw=1.8,
                label=fr"${L_lam/1000:.0f}$ km filter $L_\Lambda$")
    ax.axhline(2.0, color="#c0392b", ls="--", lw=1.0, label=r"$\Lambda=2$")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from outlet (km)")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_title("A", loc="left", fontweight="bold", fontsize=12)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    # Invert x if outlet is left in paper (high dist_out on left)
    ax.set_xlim(df["dist_out_km"].max() * 1.02, max(df["dist_out_km"].min() * 0.9, 0))

    # Bracket annotations for scales near a high-Λ peak
    i_peak = int(np.argmax(df["lam_Lc"].to_numpy()))
    x_peak = float(df["dist_out_km"].iloc[i_peak])
    ax.annotate(
        "",
        xy=(x_peak - L_c / 2000.0, 0.15),
        xytext=(x_peak + L_c / 2000.0, 0.15),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="<->", color="#1a5276", lw=1.2),
    )
    ax.text(x_peak, 0.18, fr"$L_c={L_c/1000:.1f}$ km", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=7, color="#1a5276")

    # ----- Panel B: LISA -----
    ax = axB
    colors = {
        "HH": "#e74c3c",
        "LL": "#85c1e9",
        "LH": "#1a5276",
        "HL": "#f5b7b1",
        "ns": "#d5d8dc",
    }
    labels = {
        "HH": "High-High",
        "LL": "Low-Low",
        "LH": "Low-High",
        "HL": "High-Low",
        "ns": "n.s.",
    }
    for lab in ["ns", "LL", "LH", "HL", "HH"]:
        sub = lisa_df[lisa_df["lisa_sig_cluster"] == lab]
        if sub.empty:
            continue
        ax.scatter(
            sub["dist_out_km"], sub["lam_Lc"],
            s=18, c=colors[lab], label=labels[lab], zorder=3 if lab != "ns" else 1,
            edgecolors="none",
        )
    ax.axhline(2.0, color="#c0392b", ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Distance from outlet (km)")
    ax.set_ylabel(r"Smoothed $\Lambda$ ($L_c$)")
    ax.set_title("B", loc="left", fontweight="bold", fontsize=12)
    ax.legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(axA.get_xlim())

    # Highlight HH core
    hh = lisa_df[lisa_df["lisa_sig_cluster"] == "HH"]
    if not hh.empty:
        x_hh = float(hh.loc[hh["lam_Lc"].idxmax(), "dist_out_km"])
        y_hh = float(hh["lam_Lc"].max())
        ax.annotate(
            "High-High\n(avulsion-potential zone)",
            xy=(x_hh, y_hh),
            xytext=(x_hh + (5 if ax.get_xlim()[0] > ax.get_xlim()[1] else -5), y_hh * 0.4),
            fontsize=7,
            color="#922b21",
            arrowprops=dict(arrowstyle="->", color="#922b21", lw=1),
            ha="center",
        )

    # ----- Panel C: semivariogram -----
    ax = axC
    ax.plot(vario["lag_m"] / 1000.0, vario["gamma"], "o", ms=4, color="0.2", label="Experimental")
    h_fit = np.linspace(0, float(vario["lag_m"].max()), 400)
    g_fit = robert_model(
        h_fit,
        fit["c0"], fit["c"], fit["r"],
        fit["a1"], fit["b1"], fit["l1"],
        fit["a2"], fit["b2"], fit["l2"],
    )
    ax.plot(
        h_fit / 1000.0, g_fit, color="#c0392b", lw=1.8,
        label="Exponential +\nperiodic model",
    )
    ax.axvline(L_c / 1000.0, color="#c0392b", ls="--", lw=1.0)
    ax.axvline(L_lam / 2000.0, color="#2980b9", ls="--", lw=1.0)
    ymin, ymax = ax.get_ylim()
    ax.text(L_c / 1000.0, ymax, fr"$L_c\approx{L_c/1000:.0f}$ km", color="#c0392b",
            fontsize=7, rotation=90, va="top", ha="right")
    ax.text(L_lam / 2000.0, ymax, fr"$L_\Lambda/2\approx{L_lam/2000:.0f}$ km", color="#2980b9",
            fontsize=7, rotation=90, va="top", ha="right")
    ax.set_xlabel("Lag distance (km)")
    ax.set_ylabel("Semivariance")
    ax.set_title("C", loc="left", fontweight="bold", fontsize=12)
    ax.legend(
        fontsize=6, loc="lower right",
        title=fr"$L_c={L_c/1000:.1f}$ km" + "\n" + fr"$L_\Lambda={L_lam/1000:.1f}$ km",
        title_fontsize=7,
    )
    ax.grid(True, alpha=0.25)

    # ----- Panel D: histogram (single river) -----
    ax = axD
    ax.bar([0], [1], width=0.35, color="#85c1e9", label=fr"$L_c$ ({L_c/1000:.1f} km)")
    ax.bar([0.4], [1], width=0.35, color="#e67e22", label=fr"$L_\Lambda$ ({L_lam/1000:.1f} km)")
    ax.set_xticks([])
    ax.set_ylabel("Count")
    ax.set_xlabel("Length scale (km)")
    ax.set_title("D", loc="left", fontweight="bold", fontsize=12)
    ax.text(0.5, 0.9, f"{river_name}\n(single reach)", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="0.4")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 2.2)

    # ----- Panel E: L_c vs L_Λ -----
    ax = axE
    ax.scatter([L_lam / 1000.0], [L_c / 1000.0], s=80, c="#1a5276", zorder=3)
    ax.set_xlabel(r"$L_\Lambda$ (km)")
    ax.set_ylabel(r"$L_c$ (km)")
    ax.set_title("E", loc="left", fontweight="bold", fontsize=12)
    ax.text(
        0.05, 0.95,
        f"{river_name}\n"
        fr"$L_c={L_c/1000:.1f}$ km" + "\n"
        fr"$L_\Lambda={L_lam/1000:.1f}$ km" + "\n"
        "(multi-river regression\nneeds more reaches)",
        transform=ax.transAxes, va="top", fontsize=7, color="0.35",
    )
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, max(L_lam / 1000.0, L_c / 1000.0) * 1.4)
    ax.set_ylim(0, max(L_lam / 1000.0, L_c / 1000.0) * 1.4)

    # ----- Panel F: map -----
    ax = axF
    ax.set_title("F", loc="left", fontweight="bold", fontsize=12, pad=10)
    pts = gpd.read_file(points_path)
    pts["node_id"] = pts["node_id"].astype(int)
    g = pts.merge(df[["node_id", "lambda", "dist_out_km"]], on="node_id", how="inner")
    g = g.to_crs(epsg=4326)

    tiles = cimgt.QuadtreeTiles()  # OSM-like; fallback handled below
    try:
        ax.add_image(tiles, 12, interpolation="spline36")
    except Exception:
        try:
            ax.add_image(cimgt.GoogleTiles(style="satellite"), 12)
        except Exception as e:
            warnings.warn(f"Basemap unavailable ({e}); plotting points only.")
            ax.set_facecolor("#e8e8e8")

    sc = ax.scatter(
        g.geometry.x, g.geometry.y,
        c=g["lambda"], cmap="magma",
        norm=LogNorm(vmin=max(g["lambda"].min(), 0.1), vmax=max(g["lambda"].quantile(0.98), 2)),
        s=14, transform=ccrs.PlateCarree(), zorder=5,
    )
    # extent with padding
    pad = 0.02
    ax.set_extent(
        [
            g.geometry.x.min() - pad,
            g.geometry.x.max() + pad,
            g.geometry.y.min() - pad,
            g.geometry.y.max() + pad,
        ],
        crs=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"$\Lambda$")

    # Scale annotations using along-river peak neighborhood
    g_sorted = g.sort_values("dist_out_km", ascending=False)
    # mark L_c / L_Λ spans near mid-reach as text
    ax.text(
        0.02, 0.98,
        fr"$L_c={L_c/1000:.1f}$ km" + "\n" + fr"$L_\Lambda={L_lam/1000:.1f}$ km",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    # North arrow
    ax.annotate(
        "N", xy=(0.95, 0.12), xytext=(0.95, 0.02),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="k"),
    )

    fig.suptitle(
        f"{river_name}: spatial structure of $\\Lambda$ (paper-style)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    """Prefer ``poetry run python plot_results.py paper`` (unified entry point)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-csv", default=DEFAULT_LAMBDA_CSV)
    parser.add_argument("--points", default=DEFAULT_POINTS)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--river-name", default=DEFAULT_RIVER)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    def proj(p: str) -> Path:
        path = Path(p)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    df = load_lambda(proj(args.lambda_csv))
    x = df["dist_down_m"].to_numpy()
    z = df["log_lambda"].to_numpy()

    print("Fitting Robert variogram...")
    vario = empirical_variogram(x, z, n_lags=40)
    fit = fit_robert_variogram(vario)
    if fit is None:
        print("Variogram fit failed")
        return 1
    print(f"  L_c = {fit['LC']/1000:.2f} km,  L_Λ = {fit['L_lambda']/1000:.2f} km")

    # LISA on L_c-smoothed Λ (paper panel B)
    df_s = savgol_smooth_lambda(df, fit["LC"])
    spacing = float(np.median(np.diff(np.sort(x))))
    band = float(min(max(8.0 * spacing, 1500.0), fit["L_lambda"]))
    band = max(band, 2.5 * spacing)
    print(f"LISA band = {band:.0f} m")
    lisa_df, mi, lisa, w = run_lisa(df_s, df_s["log_lambda_smooth"].to_numpy(), band_m=band)
    lisa_df = flag_hh_bordered_by_lh(lisa_df, w)

    out = build_figure(
        df_s, lisa_df, vario, fit,
        proj(args.points), args.river_name, proj(args.output),
    )
    print(f"✅ Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
