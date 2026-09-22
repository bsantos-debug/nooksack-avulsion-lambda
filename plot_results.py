#!/usr/bin/env python3
"""
Unified plotting for Nooksack lambda / ridge analysis.

Figures:
  map         — Λ on satellite (or DEM) along the centerline
  widths      — ridge width vs channel width
  beta-gamma  — β, γ, Λ vs distance downstream
  paper       — multi-panel paper-style hotspot figure
  all         — write every figure above (default)

Usage:
    poetry run python plot_results.py
    poetry run python plot_results.py map --basemap satellite
    poetry run python plot_results.py widths
    poetry run python plot_results.py beta-gamma
    poetry run python plot_results.py paper
    poetry run python plot_results.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DEFAULT_LAMBDA_CSV = "data/Nooksack_lambda_results.csv"
DEFAULT_CENTERLINE = (
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/"
    "Nooksack_data/GIS/Centerline_flowaccumulation/Centerline_flowaccumulation.shp"
)
DEFAULT_POINTS = "centerline_points_from_xs.shp"
DEFAULT_DEM = (
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/"
    "Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
)
DEFAULT_RIVER = "Nooksack"


def _proj(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def run_map(
    lambda_csv: Path,
    centerline: Path,
    points: Path,
    dem_path: Path,
    output: Path,
    river_name: str,
    basemap: str = "satellite",
    zoom: int = 13,
) -> Path:
    from plot_lambda_on_dem import (
        load_centerline,
        load_lambda_results,
        plot_lambda_on_dem,
    )

    print(f"🗺️  Map ({basemap}): {output}")
    lambda_df = load_lambda_results(lambda_csv)
    centerline_gdf = load_centerline(centerline, points)
    plot_lambda_on_dem(
        lambda_df=lambda_df,
        centerline_gdf=centerline_gdf,
        dem_path=dem_path,
        centerline_path=centerline,
        output_path=output,
        river_name=river_name,
        basemap=basemap,
        zoom_level=zoom,
    )
    return output


def run_widths(
    lambda_csv: Path,
    output: Path,
    csv_out: Path,
    river_name: str,
) -> Path:
    from plot_ridge_vs_channel_width import compute_widths, plot_comparison

    print(f"📏 Widths: {output}")
    df = compute_widths(pd.read_csv(lambda_csv))
    cols = [
        "node_id",
        "dist_out",
        "channel_width_m",
        "ridge_width_1_m",
        "ridge_width_2_m",
        "ridge_width_mean_m",
        "ridge_channel_width_ratio",
    ]
    if "lambda" in df.columns:
        cols.append("lambda")
    if "width" in df.columns:
        cols.append("width")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(csv_out, index=False)
    print(f"✅ Widths CSV: {csv_out}")
    plot_comparison(df, river_name, output)
    return output


def run_beta_gamma(
    lambda_csv: Path,
    output: Path,
    river_name: str,
) -> Path:
    import matplotlib.pyplot as plt

    print(f"📈 β / γ downstream: {output}")
    df = pd.read_csv(lambda_csv)
    v = df[df["flag_valid_lambda"] == True].copy()  # noqa: E712
    if v.empty:
        raise ValueError("No valid lambda rows for beta-gamma plot")

    v = v.sort_values("dist_out", ascending=False)
    v["dist_down_km"] = (v["dist_out"].max() - v["dist_out"]) / 1000.0
    x = v["dist_down_km"].to_numpy(float)
    beta = v["superelevation_mean"].to_numpy(float)
    gamma = v["gamma_mean"].to_numpy(float)
    lam = v["lambda"].to_numpy(float)

    def sg(y: np.ndarray, win: int = 21) -> np.ndarray:
        win = min(win, len(y) - (1 - len(y) % 2))
        if win % 2 == 0:
            win -= 1
        win = max(5, win)
        return savgol_filter(y, window_length=win, polyorder=2, mode="interp")

    everson_down = None
    if (v["node_id"] == 103).any():
        everson_down = float(v.loc[v["node_id"] == 103, "dist_down_km"].iloc[0])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.scatter(x, beta, s=14, c="#1f77b4", alpha=0.55, edgecolors="none", label="XS")
    ax.plot(x, sg(beta), color="#0b3d91", lw=2, label="smoothed")
    ax.axhline(0.2, color="k", ls=":", lw=1, alpha=0.5)
    ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_ylabel("Superelevation β")
    ax.set_ylim(0, min(1.2, float(np.nanpercentile(beta, 99)) * 1.15))
    ax.set_title(f"{river_name}: β and γ downstream")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if everson_down is not None:
        ax.axvline(everson_down, color="#d62728", ls="--", lw=1, alpha=0.8)
        ax.text(
            everson_down,
            ax.get_ylim()[1] * 0.95,
            " Everson",
            color="#d62728",
            va="top",
            fontsize=9,
        )

    ax = axes[1]
    ax.scatter(x, gamma, s=14, c="#ff7f0e", alpha=0.55, edgecolors="none", label="XS")
    g_pos = np.clip(gamma, 1e-3, None)
    ax.plot(x, np.exp(sg(np.log(g_pos))), color="#b35c00", lw=2, label="smoothed")
    ax.axhline(3, color="k", ls=":", lw=1, alpha=0.5)
    ax.axhline(10, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Gradient advantage γ")
    ax.set_ylim(0.2, max(50.0, float(np.nanpercentile(gamma, 95))))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if everson_down is not None:
        ax.axvline(everson_down, color="#d62728", ls="--", lw=1, alpha=0.8)

    ax = axes[2]
    ax.scatter(x, lam, s=14, c="#2ca02c", alpha=0.55, edgecolors="none", label="XS")
    l_pos = np.clip(lam, 1e-3, None)
    ax.plot(x, np.exp(sg(np.log(l_pos))), color="#145a14", lw=2, label="smoothed")
    ax.axhline(2, color="k", ls="--", lw=1, alpha=0.7, label="Λ≈2")
    ax.axhline(11, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Avulsion potential Λ")
    ax.set_xlabel("Distance downstream [km]")
    ax.set_ylim(0.05, max(30.0, float(np.nanpercentile(lam, 95))))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if everson_down is not None:
        ax.axvline(everson_down, color="#d62728", ls="--", lw=1, alpha=0.8)

    for a in axes:
        a.grid(True, alpha=0.25)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Plot saved to: {output}")
    return output


def run_paper(
    lambda_csv: Path,
    points: Path,
    output: Path,
    river_name: str,
) -> Path:
    from analyze_lambda_hotspots import (
        empirical_variogram,
        fit_robert_variogram,
        flag_hh_bordered_by_lh,
        load_lambda,
        run_lisa,
        savgol_smooth_lambda,
    )
    from plot_lambda_paper_figure import build_figure

    print(f"📄 Paper figure: {output}")
    df = load_lambda(lambda_csv)
    x = df["dist_down_m"].to_numpy()
    z = df["log_lambda"].to_numpy()
    vario = empirical_variogram(x, z, n_lags=40)
    fit = fit_robert_variogram(vario)
    if fit is None:
        raise RuntimeError("Variogram fit failed")
    print(f"  L_c = {fit['LC']/1000:.2f} km,  L_Λ = {fit['L_lambda']/1000:.2f} km")

    df_s = savgol_smooth_lambda(df, fit["LC"])
    spacing = float(np.median(np.diff(np.sort(x))))
    band = float(min(max(8.0 * spacing, 1500.0), fit["L_lambda"]))
    band = max(band, 2.5 * spacing)
    lisa_df, _mi, _lisa, w = run_lisa(
        df_s, df_s["log_lambda_smooth"].to_numpy(), band_m=band
    )
    lisa_df = flag_hh_bordered_by_lh(lisa_df, w)
    return build_figure(df_s, lisa_df, vario, fit, points, river_name, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified plotting for lambda / ridge analysis"
    )
    parser.add_argument(
        "figure",
        nargs="?",
        default="all",
        choices=["all", "map", "widths", "beta-gamma", "paper"],
        help="Which figure(s) to generate (default: all)",
    )
    parser.add_argument("--lambda-csv", default=DEFAULT_LAMBDA_CSV)
    parser.add_argument("--centerline", default=DEFAULT_CENTERLINE)
    parser.add_argument("--points", default=DEFAULT_POINTS)
    parser.add_argument("--dem-path", default=DEFAULT_DEM)
    parser.add_argument("--river-name", default=DEFAULT_RIVER)
    parser.add_argument(
        "--basemap",
        choices=["satellite", "dem"],
        default="satellite",
        help="Basemap for map figure",
    )
    parser.add_argument("--zoom", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for default output paths (default: data)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path (single-figure modes only)",
    )
    args = parser.parse_args(argv)

    lambda_csv = _proj(args.lambda_csv)
    centerline = _proj(args.centerline)
    points = _proj(args.points)
    dem_path = _proj(args.dem_path)
    out_dir = _proj(args.output_dir)
    river = args.river_name
    figures = (
        ["map", "widths", "beta-gamma", "paper"]
        if args.figure == "all"
        else [args.figure]
    )

    if not lambda_csv.exists():
        print(f"❌ Lambda CSV not found: {lambda_csv}")
        return 1

    try:
        for fig in figures:
            if fig == "map":
                out = (
                    _proj(args.output)
                    if args.output and args.figure == "map"
                    else out_dir / f"{river}_lambda_on_satellite.png"
                )
                run_map(
                    lambda_csv,
                    centerline,
                    points,
                    dem_path,
                    out,
                    river,
                    basemap=args.basemap,
                    zoom=args.zoom,
                )
            elif fig == "widths":
                out = (
                    _proj(args.output)
                    if args.output and args.figure == "widths"
                    else out_dir / f"{river}_ridge_vs_channel_width.png"
                )
                csv_out = out_dir / f"{river}_ridge_channel_widths.csv"
                run_widths(lambda_csv, out, csv_out, river)
            elif fig == "beta-gamma":
                out = (
                    _proj(args.output)
                    if args.output and args.figure == "beta-gamma"
                    else out_dir / f"{river}_beta_gamma_downstream.png"
                )
                run_beta_gamma(lambda_csv, out, river)
            elif fig == "paper":
                out = (
                    _proj(args.output)
                    if args.output and args.figure == "paper"
                    else out_dir / "hotspots" / f"{river}_paper_style_figure.png"
                )
                run_paper(lambda_csv, points, out, river)
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
