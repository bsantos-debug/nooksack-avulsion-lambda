#!/usr/bin/env python3
"""
Compare alluvial ridge width to channel width from labeler picks.

Definitions (along-XS distance):
  - Channel width  W_ch = |ridge2 − ridge1|
  - Ridge width    W_r  = |ridge − floodplain|  (per bank; also mean of banks)

Usage:
    poetry run python plot_ridge_vs_channel_width.py
    poetry run python plot_ridge_vs_channel_width.py --lambda-csv data/Nooksack_lambda_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_LAMBDA_CSV = "data/Nooksack_lambda_results.csv"
DEFAULT_OUTPUT = "data/Nooksack_ridge_vs_channel_width.png"
DEFAULT_CSV_OUT = "data/Nooksack_ridge_channel_widths.csv"


def compute_widths(df: pd.DataFrame) -> pd.DataFrame:
    """Add channel and ridge widths from label distances."""
    required = [
        "ridge1_dist_along",
        "ridge2_dist_along",
        "floodplain1_dist_along",
        "floodplain2_dist_along",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    out["channel_width_m"] = (
        out["ridge2_dist_along"] - out["ridge1_dist_along"]
    ).abs()
    out["ridge_width_1_m"] = (
        out["ridge1_dist_along"] - out["floodplain1_dist_along"]
    ).abs()
    out["ridge_width_2_m"] = (
        out["ridge2_dist_along"] - out["floodplain2_dist_along"]
    ).abs()
    out["ridge_width_mean_m"] = np.nanmean(
        np.vstack(
            [
                out["ridge_width_1_m"].to_numpy(dtype=float),
                out["ridge_width_2_m"].to_numpy(dtype=float),
            ]
        ),
        axis=0,
    )
    out["ridge_channel_width_ratio"] = (
        out["ridge_width_mean_m"] / out["channel_width_m"].replace(0, np.nan)
    )
    return out


def plot_comparison(
    df: pd.DataFrame,
    river_name: str,
    output_path: Path,
) -> None:
    """Scatter + along-river panels for ridge vs channel width."""
    # Keep nodes with both ridges and at least one usable ridge width
    use = df[
        df["channel_width_m"].notna()
        & (df["channel_width_m"] > 0)
        & df["ridge_width_mean_m"].notna()
        & (df["ridge_width_mean_m"] > 0)
    ].copy()

    if len(use) == 0:
        raise ValueError("No nodes with valid ridge and channel widths")

    use = use.sort_values("dist_out")
    dist_down_km = (use["dist_out"].max() - use["dist_out"]) / 1000.0

    w_ch = use["channel_width_m"].to_numpy(dtype=float)
    w_r = use["ridge_width_mean_m"].to_numpy(dtype=float)
    w_r1 = use["ridge_width_1_m"].to_numpy(dtype=float)
    w_r2 = use["ridge_width_2_m"].to_numpy(dtype=float)
    ratio = use["ridge_channel_width_ratio"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # --- A: scatter mean ridge width vs channel width ---
    ax = axes[0, 0]
    ax.scatter(w_ch, w_r, c="steelblue", s=28, alpha=0.75, edgecolors="none")
    lim = np.nanpercentile(np.concatenate([w_ch, w_r]), 98)
    lim = max(float(lim), 10.0)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="1:1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Channel width (ridge₁ → ridge₂) [m]")
    ax.set_ylabel("Ridge width (ridge → floodplain, mean) [m]")
    ax.set_title("A  Ridge vs channel width")
    ax.legend(loc="upper left", frameon=False)
    med_r = float(np.nanmedian(ratio))
    ax.text(
        0.98,
        0.02,
        f"median Wᵣ/Wch = {med_r:.2f}\nn = {len(use)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    # --- B: both banks vs channel ---
    ax = axes[0, 1]
    ax.scatter(
        w_ch,
        w_r1,
        c="#2ca02c",
        s=22,
        alpha=0.65,
        edgecolors="none",
        label="Bank 1",
    )
    ax.scatter(
        w_ch,
        w_r2,
        c="#ff7f0e",
        s=22,
        alpha=0.65,
        edgecolors="none",
        label="Bank 2",
    )
    lim2 = np.nanpercentile(np.concatenate([w_ch, w_r1, w_r2]), 98)
    lim2 = max(float(lim2), 10.0)
    ax.plot([0, lim2], [0, lim2], "k--", lw=1, alpha=0.6)
    ax.set_xlim(0, lim2)
    ax.set_ylim(0, lim2)
    ax.set_xlabel("Channel width (ridge₁ → ridge₂) [m]")
    ax.set_ylabel("Ridge width (ridge → floodplain) [m]")
    ax.set_title("B  Per-bank ridge width")
    ax.legend(loc="upper left", frameon=False)

    # --- C: widths along river ---
    ax = axes[1, 0]
    ax.plot(dist_down_km, w_ch, color="#1f77b4", lw=1.4, label="Channel (ridge–ridge)")
    ax.plot(
        dist_down_km,
        w_r,
        color="#d62728",
        lw=1.4,
        label="Ridge (mean, ridge–floodplain)",
    )
    ax.set_xlabel("Distance downstream [km]")
    ax.set_ylabel("Width [m]")
    ax.set_title("C  Widths along river")
    ax.legend(loc="best", frameon=False)
    ax.set_ylim(bottom=0)

    # --- D: ratio along river ---
    ax = axes[1, 1]
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax.plot(dist_down_km, ratio, color="#9467bd", lw=1.4)
    ax.set_xlabel("Distance downstream [km]")
    ax.set_ylabel("Wᵣ / Wch")
    ax.set_title("D  Ridge / channel width ratio")
    ax.set_ylim(bottom=0)
    p95 = float(np.nanpercentile(ratio[np.isfinite(ratio)], 95))
    if np.isfinite(p95) and p95 > 0:
        ax.set_ylim(0, max(3.0, min(p95 * 1.2, np.nanmax(ratio) * 1.05)))

    fig.suptitle(
        f"{river_name}: alluvial ridge width vs channel width",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Plot saved to: {output_path}")


def main() -> None:
    """Prefer ``poetry run python plot_results.py widths`` (unified entry point)."""
    parser = argparse.ArgumentParser(
        description="Compare ridge width (ridge→floodplain) to channel width (ridge1→ridge2)"
    )
    parser.add_argument(
        "--lambda-csv",
        type=str,
        default=DEFAULT_LAMBDA_CSV,
        help=f"Lambda results CSV with label distances (default: {DEFAULT_LAMBDA_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Figure path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=DEFAULT_CSV_OUT,
        help=f"Widths CSV path (default: {DEFAULT_CSV_OUT})",
    )
    parser.add_argument(
        "--river-name",
        type=str,
        default="Nooksack",
        help="River name for title",
    )
    args = parser.parse_args()

    csv_path = Path(args.lambda_csv)
    print(f"📊 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = compute_widths(df)

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
        cols.append("width")  # network/XS width for reference

    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_csv, index=False)
    print(f"✅ Widths CSV: {out_csv}")

    valid = df[
        (df["channel_width_m"] > 0) & (df["ridge_width_mean_m"] > 0)
    ]
    print(
        f"n={len(valid)}  "
        f"channel median={valid['channel_width_m'].median():.1f} m  "
        f"ridge median={valid['ridge_width_mean_m'].median():.1f} m  "
        f"ratio median={valid['ridge_channel_width_ratio'].median():.2f}"
    )

    plot_comparison(df, args.river_name, Path(args.output))


if __name__ == "__main__":
    main()
