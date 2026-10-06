"""Compute Λ from labeled (or auto-picked) cross-sections."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from avulsionprecursors.analysis.metrics import apply_ridge_parameters
from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.exceptions import InputValidationError
from avulsionprecursors.geometry.reach_builder import build_reach_from_shapefiles
from avulsionprecursors.io.validation import validate_config


def load_labels(
    labels_dir: Path,
    study_name: str,
    node_id: int,
    min_mtime: Optional[datetime] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Load GUI labels for one node. Returns None if missing or skipped."""
    label_file = labels_dir / f"{study_name}_node_{int(node_id)}_labels.csv"
    if not label_file.exists():
        return None
    if min_mtime is not None:
        file_mtime = datetime.fromtimestamp(label_file.stat().st_mtime)
        if file_mtime < min_mtime:
            return None
    with open(label_file, "r") as f:
        first_line = f.readline().strip()
        if first_line.startswith("# SKIPPED"):
            return None
    try:
        df = pd.read_csv(label_file)
        if df.empty:
            return None
        labels = {}
        for _, row in df.iterrows():
            label = str(row["label"]).lower()
            if label in ["channel", "ridge1", "floodplain1", "ridge2", "floodplain2"]:
                labels[label] = {
                    "dist_along": float(row["dist_along"]),
                    "elevation": float(row["elevation"]),
                }
        if "channel" not in labels:
            return None
        return labels
    except Exception as exc:
        print(f"Warning: could not load labels for node {node_id}: {exc}")
        return None


def labels_from_profile(
    node_id: int,
    cross_section_line: LineString,
    profiles_dir: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Auto-pick channel = min elevation and ridges = max left/right of channel.

    Does not pick floodplains. If valley walls are higher than levees, the
    maxima will be the valley walls rather than alluvial ridges.
    """
    candidates = [
        profiles_dir / f"cross_section_{int(node_id):03d}.csv",
        profiles_dir / f"{int(node_id)}.csv",
    ]
    profile_path = next((p for p in candidates if p.exists()), None)
    if profile_path is None:
        return None

    df = pd.read_csv(profile_path)
    dist_col = "distance" if "distance" in df.columns else "distance_m"
    elev_col = "elevation" if "elevation" in df.columns else "elevation_m"
    if dist_col not in df.columns or elev_col not in df.columns:
        return None

    valid = df[elev_col].notna()
    if valid.sum() < 3:
        return None
    prof = df.loc[valid, [dist_col, elev_col]].copy()
    half_length = float(cross_section_line.length) / 2.0

    i_chan = prof[elev_col].idxmin()
    chan_dist_c = float(prof.loc[i_chan, dist_col])
    chan_elev = float(prof.loc[i_chan, elev_col])
    left = prof[prof[dist_col] < chan_dist_c]
    right = prof[prof[dist_col] > chan_dist_c]
    if left.empty or right.empty:
        return None
    i_r1 = left[elev_col].idxmax()
    i_r2 = right[elev_col].idxmax()
    return {
        "channel": {"dist_along": chan_dist_c + half_length, "elevation": chan_elev},
        "ridge1": {
            "dist_along": float(left.loc[i_r1, dist_col]) + half_length,
            "elevation": float(left.loc[i_r1, elev_col]),
        },
        "ridge2": {
            "dist_along": float(right.loc[i_r2, dist_col]) + half_length,
            "elevation": float(right.loc[i_r2, elev_col]),
        },
    }


def _parse_labels_since(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    text = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise InputValidationError(
        f"labels_since must be YYYY-MM-DD or YYYY-MM-DDTHH:MM, got {raw!r}."
    )


def plot_lambda_vs_distance(df: pd.DataFrame, study_name: str, output_path: Path) -> None:
    """Save Λ vs distance from outlet."""
    if df.empty or "lambda" not in df.columns or "dist_out" not in df.columns:
        print("Warning: cannot plot — missing lambda or dist_out")
        return
    df_plot = df.sort_values("dist_out").copy()
    df_plot["dist_out_display"] = df_plot["dist_out"] / 1000.0
    df_plot = df_plot[df_plot["lambda"].notna() & (df_plot["lambda"] > 0)]
    if df_plot.empty:
        print("Warning: no valid lambda values to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(
        df_plot["dist_out_display"],
        df_plot["lambda"],
        color="blue",
        s=20,
        zorder=2,
        alpha=0.6,
        edgecolor="none",
        label="Lambda values",
    )
    ax.axhline(y=2, color="darkred", linestyle="--", linewidth=2, label=r"$\Lambda$ = 2")
    ax.set_xlabel("Distance from outlet (×1000 working-CRS units)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Avulsion potential $\Lambda$", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)
    ax.invert_xaxis()
    ax.set_title(f"{study_name} — lambda vs distance downstream", fontsize=14, pad=15)
    stats_text = (
        f"N = {len(df_plot)}\n"
        f"Min: {df_plot['lambda'].min():.3f}\n"
        f"Max: {df_plot['lambda'].max():.3f}\n"
        f"Mean: {df_plot['lambda'].mean():.3f}\n"
        f"Median: {df_plot['lambda'].median():.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def calculate_lambda(cfg: WorkflowConfig) -> pd.DataFrame:
    """Attach labels, apply the lower-ridge rule, and write Λ.

    Required inputs
    ---------------
    Centerline points and cross-sections from the extract step, plus either
    labeler CSVs (channel, ridge1/2, floodplain1/2) or ``auto_minmax`` for
    channel/ridges only (floodplains still required for Har).
    """
    validate_config(cfg)
    points_path = cfg.centerline_points_path()
    xs_path = cfg.cross_sections_path()
    if not points_path.exists():
        raise InputValidationError(
            f"Centerline points not found:\n  {points_path}\n"
            "Run the extract step first."
        )
    if not xs_path.exists():
        raise InputValidationError(
            f"Cross-sections not found:\n  {xs_path}\n"
            "Run the extract step first."
        )

    labels_min_mtime = _parse_labels_since(cfg.lambda_calc.labels_since)
    study = cfg.study_name
    labels_dir = cfg.labels_dir()
    profiles_dir = cfg.profiles_dir()
    auto_minmax = bool(cfg.lambda_calc.auto_minmax)

    print(f"Calculating lambda for {study}")
    if auto_minmax:
        print(
            "Channel/ridge picks: profile min / left-right max. "
            "This can select valley walls when they are taller than levees. "
            "Floodplain labels are still required for Har."
        )
    else:
        print(f"Labels directory: {labels_dir}")
        if labels_min_mtime is not None:
            print(f"Only files saved since {labels_min_mtime:%Y-%m-%d %H:%M}")
    print("Hm definition: ridge crest − channel bed")
    print("Ridge rule: lower crest")
    print("Sm: signed downhill thalweg slope (SG-smoothed long profile)")

    reach = build_reach_from_shapefiles(
        centerline_path=str(cfg.prepared_centerline_path()),
        points_path=str(points_path),
    )
    cross_sections_gdf = gpd.read_file(xs_path)
    for node in reach.nodes:
        node_xs = cross_sections_gdf[cross_sections_gdf["node_id"] == node.node_id]
        if node_xs.empty:
            continue
        node.cross_section = node_xs.geometry.iloc[0]
        if "width" in node_xs.columns:
            width_val = node_xs["width"].iloc[0]
            if width_val is not None and not pd.isna(width_val) and width_val > 0:
                node.width = float(width_val)
        if "slope" in node_xs.columns:
            slope_val = node_xs["slope"].iloc[0]
            if slope_val is not None and not pd.isna(slope_val):
                node.slope = float(slope_val)

    rows = []
    for node in reach.nodes:
        if node.cross_section is None or node.slope is None:
            continue
        rows.append(
            {
                "node_id": node.node_id,
                "reach_id": node.reach_id,
                "dist_out": node.dist_out,
                "width": float(node.width) if node.width is not None else np.nan,
                "slope": float(node.slope),
                "elevation": float(node.elevation) if node.elevation is not None else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise InputValidationError("No nodes with cross-sections and slope were found.")

    label_columns = [
        "channel_dist_along",
        "channel_elevation",
        "ridge1_dist_along",
        "ridge1_elevation",
        "floodplain1_dist_along",
        "floodplain1_elevation",
        "ridge2_dist_along",
        "ridge2_elevation",
        "floodplain2_dist_along",
        "floodplain2_elevation",
    ]
    for col in label_columns:
        df[col] = np.nan

    for idx, row in df.iterrows():
        node_id = row["node_id"]
        node = next((n for n in reach.nodes if n.node_id == node_id), None)
        if not node:
            continue
        labels = None
        if auto_minmax and node.cross_section is not None:
            labels = labels_from_profile(int(node_id), node.cross_section, profiles_dir)
        if labels is None:
            labels = load_labels(labels_dir, study, node_id, min_mtime=labels_min_mtime)
        if not labels:
            continue
        for key, dist_col, elev_col in (
            ("channel", "channel_dist_along", "channel_elevation"),
            ("ridge1", "ridge1_dist_along", "ridge1_elevation"),
            ("floodplain1", "floodplain1_dist_along", "floodplain1_elevation"),
            ("ridge2", "ridge2_dist_along", "ridge2_elevation"),
            ("floodplain2", "floodplain2_dist_along", "floodplain2_elevation"),
        ):
            if key in labels:
                df.at[idx, dist_col] = labels[key]["dist_along"]
                df.at[idx, elev_col] = labels[key]["elevation"]

    bank1 = df["ridge1_dist_along"].notna() & df["floodplain1_dist_along"].notna()
    bank2 = df["ridge2_dist_along"].notna() & df["floodplain2_dist_along"].notna()
    df = df[df["channel_dist_along"].notna() & (bank1 | bank2)].copy()
    if df.empty:
        raise InputValidationError(
            "No nodes have channel plus at least one ridge and its floodplain.\n"
            f"Label CSVs should look like {study}_node_<id>_labels.csv in:\n  {labels_dir}\n"
            "Auto-minmax only picks channel and ridges; floodplains still need the labeler."
        )

    df = apply_ridge_parameters(df, slope_min=cfg.extract.slope_min)
    n_valid = int(df["flag_valid_lambda"].sum()) if "flag_valid_lambda" in df.columns else len(df)
    n_adv = int(df["flag_adverse_slope"].sum()) if "flag_adverse_slope" in df.columns else 0
    print(f"Calculated lambda for {n_valid} nodes ({n_adv} adverse/zero Sm omitted from Λ)")
    if n_valid:
        valid = df.loc[df["flag_valid_lambda"], "lambda"]
        print(f"Lambda range: {valid.min():.4f} - {valid.max():.4f}")
    print(
        f"Hm = ridge crest − channel bed: "
        f"{df['channel_depth'].min():.2f} – {df['channel_depth'].max():.2f} "
        f"(median {df['channel_depth'].median():.2f})"
    )
    out = cfg.lambda_csv_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved results to {out}")
    plot_lambda_vs_distance(df, study, cfg.lambda_plot_path())
    return df
