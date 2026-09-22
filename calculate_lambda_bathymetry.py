#!/usr/bin/env python3
"""
Calculate lambda from topobathy cross-section profiles (no XGBoost).

Workflow:
1. Build reach from centerline points + cross_sections.shp
2. Use labeler CSVs for channel / ridge / floodplain picks (default).
   Optional ``--auto-minmax`` picks channel=profile min and ridges=max
   left/right (often wrong when valley walls exceed levees).
3. Depth Hm = alluvial ridge crest − channel bed (from labels)
4. Sm from DEM bed slope along the centerline (default); optional FlowFM WSE
5. Lower-ridge rule (Gearon et al.): use bank with lower crest (default).
   If that bank has floodplain above the ridge (Har < 0) and the other bank
   does not, use the other bank.
6. Compute BASED metrics (gamma, superelevation, lambda)
7. Write CSV + lambda-vs-distance plot

Usage:
    poetry run python calculate_lambda_bathymetry.py
    poetry run python calculate_lambda_bathymetry.py --flowfm-slope
    poetry run python calculate_lambda_bathymetry.py --output data/Nooksack_lambda_results.csv
"""

# === USER CONFIGURATION ===
# Default discharge value (m³/s) - can be overridden with --discharge argument
# Nooksack discharge: 3,860 ft³/s = 109.3 m³/s (1 ft³ = 0.0283168 m³)
DEFAULT_DISCHARGE = 109.3  # Nooksack river discharge
DEFAULT_CENTERLINE = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack_data/GIS/Centerline_flowaccumulation/Centerline_flowaccumulation.shp"
DEFAULT_POINTS = "centerline_points_from_xs.shp"
DEFAULT_BATHYMETRY = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
DEFAULT_FLOWFM_NC = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack_data/Model_outputs/FlowFM_merged_map.nc"
DEFAULT_FLOWFM_SLOPE_WINDOW_M = 5000.0  # fixed reach window for WSE slope (m)
import sys
from pathlib import Path

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional
from shapely.geometry import LineString, Point

from avulsionprecursors.geometry.reach_builder import build_reach_from_shapefiles
from avulsionprecursors.analysis.based_bathymetry import BASEDAnalyzer
from avulsionprecursors.analysis.flowfm_depth import (
    DEFAULT_SLOPE_WINDOW_WIDTHS,
    slope_from_flowfm_waterlevel,
)
from avulsionprecursors.sword.base import SwordNode, SwordReach


def load_labels(
    labels_dir: Path,
    river_name: str,
    node_id: int,
    min_mtime: Optional[datetime] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load labeled points for a specific node.
    
    Returns:
        Dictionary with keys: 'channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2'
        Each value is a dict with 'dist_along' and 'elevation'
    """
    label_file = labels_dir / f"{river_name}_node_{int(node_id)}_labels.csv"
    
    if not label_file.exists():
        return None

    if min_mtime is not None:
        file_mtime = datetime.fromtimestamp(label_file.stat().st_mtime)
        if file_mtime < min_mtime:
            return None
    
    # Check if file is skipped
    with open(label_file, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith("# SKIPPED"):
            return None
    
    try:
        df = pd.read_csv(label_file)
        if df.empty:
            return None
        
        labels = {}
        for _, row in df.iterrows():
            label = row['label'].lower()
            if label in ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']:
                labels[label] = {
                    'dist_along': float(row['dist_along']),
                    'elevation': float(row['elevation'])
                }
        
        # Must have at least channel
        if 'channel' not in labels:
            return None
        
        return labels
    except Exception as e:
        print(f"⚠️  Warning: Could not load labels for node {node_id}: {e}")
        return None


def labels_from_profile(
    node_id: int,
    cross_section_line: LineString,
    profiles_dir: Path,
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Auto-pick channel / ridge positions from the elevation profile.

    - channel: minimum elevation (thalweg)
    - ridge1: maximum elevation on the left of the channel
    - ridge2: maximum elevation on the right of the channel
    """
    candidates = [
        profiles_dir / f"cross_section_{int(node_id):03d}.csv",
        profiles_dir / f"{int(node_id)}.csv",
    ]
    profile_path = next((p for p in candidates if p.exists()), None)
    if profile_path is None:
        return None

    df = pd.read_csv(profile_path)
    if "distance_m" not in df.columns or "elevation_m" not in df.columns:
        return None

    valid = df["elevation_m"].notna()
    if valid.sum() < 3:
        return None

    prof = df.loc[valid, ["distance_m", "elevation_m"]].copy()
    half_length = float(cross_section_line.length) / 2.0

    # Channel = thalweg (min elevation)
    i_chan = prof["elevation_m"].idxmin()
    chan_dist_c = float(prof.loc[i_chan, "distance_m"])
    chan_elev = float(prof.loc[i_chan, "elevation_m"])

    left = prof[prof["distance_m"] < chan_dist_c]
    right = prof[prof["distance_m"] > chan_dist_c]
    if left.empty or right.empty:
        return None

    i_r1 = left["elevation_m"].idxmax()
    i_r2 = right["elevation_m"].idxmax()

    return {
        "channel": {
            "dist_along": chan_dist_c + half_length,
            "elevation": chan_elev,
        },
        "ridge1": {
            "dist_along": float(left.loc[i_r1, "distance_m"]) + half_length,
            "elevation": float(left.loc[i_r1, "elevation_m"]),
        },
        "ridge2": {
            "dist_along": float(right.loc[i_r2, "distance_m"]) + half_length,
            "elevation": float(right.loc[i_r2, "elevation_m"]),
        },
    }


def build_reach_with_labels(
    centerline_path: str,
    points_path: str,
    cross_sections_path: str,
    labels_dir: Path,
    river_name: str,
    discharge: float,
    reach_id: int = 1,
    labels_min_mtime: Optional[datetime] = None,
) -> SwordReach:
    """
    Build a SwordReach with labeled cross-section attributes.
    
    Args:
        centerline_path: Path to centerline shapefile
        points_path: Path to centerline points shapefile
        cross_sections_path: Path to cross-sections shapefile
        labels_dir: Directory containing label CSV files
        river_name: Name of the river
        discharge: Discharge value (m³/s) to use for all nodes
        reach_id: Reach ID
        
    Returns:
        SwordReach with nodes containing labeled cross-section data
    """
    # Build base reach from shapefiles
    reach = build_reach_from_shapefiles(
        centerline_path=centerline_path,
        points_path=points_path,
        reach_id=reach_id
    )
    
    # Load cross-sections to get geometry, width, and slope
    cross_sections_gdf = gpd.read_file(cross_sections_path)
    
    # Update node widths and slopes from cross-sections (overrides points shapefile)
    # This ensures we use values calculated from channel polygon (width) and DEM (slope)
    width_updated_count = 0
    slope_updated_count = 0
    
    for node in reach.nodes:
        # Get cross-section data for this node
        node_cross_section = cross_sections_gdf[cross_sections_gdf['node_id'] == node.node_id]
        if not node_cross_section.empty:
            # Get cross-section geometry
            node.cross_section = node_cross_section.geometry.iloc[0]
            
            # Update width from cross-sections (calculated from channel polygon)
            if 'width' in node_cross_section.columns:
                width_val = node_cross_section['width'].iloc[0]
                if width_val is not None and not pd.isna(width_val) and width_val > 0:
                    node.width = float(width_val)
                    width_updated_count += 1
            
            # Update slope from cross-sections (calculated from DEM)
            if 'slope' in node_cross_section.columns:
                slope_val = node_cross_section['slope'].iloc[0]
                if slope_val is not None and not pd.isna(slope_val) and slope_val > 0:
                    node.slope = float(slope_val)
                    slope_updated_count += 1
        else:
            # If cross-section not found, node will be skipped in BASED analysis
            pass
    
    # Print update summary
    if width_updated_count > 0:
        print(f"📏 Updated {width_updated_count} node widths from cross-sections shapefile (channel polygon)")
    if slope_updated_count > 0:
        print(f"📐 Updated {slope_updated_count} node slopes from cross-sections shapefile (DEM)")
    if width_updated_count == 0 and slope_updated_count == 0:
        print("⚠️  Warning: No 'width' or 'slope' columns found in cross-sections shapefile")
    
    # Load labels for each node
    for node in reach.nodes:
        labels = load_labels(
            labels_dir, river_name, node.node_id, min_mtime=labels_min_mtime
        )
        
        if labels and node.cross_section:
            # Store label data and discharge on the node
            node._labels = labels
            node._discharge = discharge
        else:
            node._labels = None
            node._discharge = discharge
    
    return reach


def add_labels_to_dataframe(
    df: pd.DataFrame,
    reach: SwordReach,
    labels_dir: Path,
    river_name: str,
    profiles_dir: Optional[Path] = None,
    auto_minmax: bool = False,
    labels_min_mtime: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Add cross-section channel/ridge attributes to the DataFrame.

    By default, uses labeler CSVs (your clicked channel / ridge / floodplain).
    If ``auto_minmax`` is True, channel = profile min and ridges = max elev
    left/right of channel (can pick valley walls instead of levees).
    """
    # Initialize label columns
    label_columns = [
        'channel_dist_along', 'channel_elevation',
        'ridge1_dist_along', 'ridge1_elevation',
        'floodplain1_dist_along', 'floodplain1_elevation',
        'ridge2_dist_along', 'ridge2_elevation',
        'floodplain2_dist_along', 'floodplain2_elevation',
        'discharge_value'
    ]
    
    for col in label_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Add labels for each node
    for idx, row in df.iterrows():
        node_id = row['node_id']
        
        # Find the corresponding node
        node = next((n for n in reach.nodes if n.node_id == node_id), None)
        if not node:
            continue

        labels = None
        if auto_minmax and profiles_dir is not None and node.cross_section is not None:
            labels = labels_from_profile(int(node_id), node.cross_section, profiles_dir)
        if labels is None and not auto_minmax:
            labels = load_labels(
                labels_dir, river_name, node_id, min_mtime=labels_min_mtime
            )
        
        if labels:
            # Add channel position
            if 'channel' in labels:
                df.at[idx, 'channel_dist_along'] = labels['channel']['dist_along']
                df.at[idx, 'channel_elevation'] = labels['channel']['elevation']
            
            # Add ridge and floodplain positions
            if 'ridge1' in labels:
                df.at[idx, 'ridge1_dist_along'] = labels['ridge1']['dist_along']
                df.at[idx, 'ridge1_elevation'] = labels['ridge1']['elevation']
            
            if 'floodplain1' in labels:
                df.at[idx, 'floodplain1_dist_along'] = labels['floodplain1']['dist_along']
                df.at[idx, 'floodplain1_elevation'] = labels['floodplain1']['elevation']
            
            if 'ridge2' in labels:
                df.at[idx, 'ridge2_dist_along'] = labels['ridge2']['dist_along']
                df.at[idx, 'ridge2_elevation'] = labels['ridge2']['elevation']
            
            if 'floodplain2' in labels:
                df.at[idx, 'floodplain2_dist_along'] = labels['floodplain2']['dist_along']
                df.at[idx, 'floodplain2_elevation'] = labels['floodplain2']['elevation']
            
            # Add discharge
            if hasattr(node, '_discharge'):
                df.at[idx, 'discharge_value'] = node._discharge
            else:
                df.at[idx, 'discharge_value'] = np.nan
    
    # Filter to only nodes with channel + both ridges
    df = df[df['channel_dist_along'].notna() & df['ridge1_dist_along'].notna() & df['ridge2_dist_along'].notna()].copy()
    
    return df


def plot_lambda_vs_distance(
    df: pd.DataFrame,
    river_name: str,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot lambda vs distance downstream.
    
    Args:
        df: DataFrame with lambda and dist_out columns
        river_name: Name of the river for title
        output_path: Optional path to save the plot
    """
    if df.empty or 'lambda' not in df.columns or 'dist_out' not in df.columns:
        print("⚠️  Warning: Cannot plot - missing required columns")
        return
    
    # Sort by distance downstream
    df_plot = df.sort_values('dist_out').copy()
    
    # Convert distance to km
    df_plot['dist_out_km'] = df_plot['dist_out'] / 1000.0
    
    # Filter out invalid lambda values
    df_plot = df_plot[df_plot['lambda'].notna() & (df_plot['lambda'] > 0)]
    
    if df_plot.empty:
        print("⚠️  Warning: No valid lambda values to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Plot data points
    ax.scatter(
        df_plot['dist_out_km'],
        df_plot['lambda'],
        color='blue',
        s=20,
        zorder=2,
        alpha=0.6,
        edgecolor='none',
        label='Lambda values'
    )
    
    # Add lambda = 2 reference line
    ax.axhline(y=2, color='darkred', linestyle='--', linewidth=2, label=r'$\Lambda$ = 2')
    
    # Styling
    ax.set_xlabel('Distance from outlet (km)', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'Avulsion Potential $\Lambda$', fontsize=13, fontweight='bold', rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Invert x-axis (distance from outlet, so upstream is left)
    ax.invert_xaxis()
    
    # Set title
    ax.set_title(f'{river_name} - Lambda vs Distance Downstream', fontsize=14, fontweight='bold', pad=15)
    
    # Add statistics text box
    stats_text = (
        f"N = {len(df_plot)}\n"
        f"Min: {df_plot['lambda'].min():.3f}\n"
        f"Max: {df_plot['lambda'].max():.3f}\n"
        f"Mean: {df_plot['lambda'].mean():.3f}\n"
        f"Median: {df_plot['lambda'].median():.3f}"
    )
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        family='monospace'
    )
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    else:
        # Save to same directory as CSV output
        default_plot_path = Path(f"data/{river_name}_lambda_plot.png")
        default_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(default_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {default_plot_path}")
    
    plt.close()


def calculate_lambda_from_labels(
    centerline_path: str,
    points_path: str,
    cross_sections_path: str,
    labels_dir: Path,
    river_name: str,
    discharge: float,
    bathymetry_path: str = "bathy_15_ft_int.tif",
    params_path: str = None,
    bathymetry_is_depth: bool = True,
    bathymetry_units: str = 'ft',
    output_path: Optional[Path] = None,
    profiles_dir: Optional[Path] = None,
    auto_minmax: bool = False,
    flowfm_depth_nc: Optional[str] = DEFAULT_FLOWFM_NC,
    flowfm_time_index: object = -1,
    flowfm_slope_window_widths: float = DEFAULT_SLOPE_WINDOW_WIDTHS,
    flowfm_backwater_length_m: Optional[float] = DEFAULT_FLOWFM_SLOPE_WINDOW_M,
    use_flowfm_slope: bool = False,
    flowfm_slope_method: str = "reach_xs",
    flowfm_xs_wse: str = "max",
    ridge_rule: str = "lower",
    labels_min_mtime: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Calculate lambda values from labeled cross-sections using bathymetry.

    By default uses labeler CSVs for channel/ridge/floodplain, Hm = ridge −
    channel bed, DEM centerline bed slope for Sm, and the lower-ridge rule.
    Pass ``use_flowfm_slope=True`` for FlowFM reach-averaged WSE slope.

    ``ridge_rule``:
      - ``lower`` (Gearon et al.): use the bank with the lower ridge crest.
        If that crest is below its floodplain (Har < 0) and the other bank
        is not, switch to the other bank.
      - ``mean``: average both banks
    """
    print(f"📊 Calculating lambda for {river_name}")
    if auto_minmax:
        print("📍 Channel/ridge picks: profile min / left-right max elevations")
    else:
        print(f"📁 Labels directory: {labels_dir}")
        print("📍 Channel/ridge picks: from labeler CSVs")
        if labels_min_mtime is not None:
            print(f"   Only files saved since {labels_min_mtime:%Y-%m-%d %H:%M}")
    print("📏 Hm definition: ridge crest − channel bed")
    print(f"🏔️  Ridge rule: {ridge_rule}")
    # Handle discharge (use provided value or default from code)
    if discharge is None:
        discharge = DEFAULT_DISCHARGE
    
    print(f"💧 Discharge: {discharge} m³/s")
    print()
    
    # Build reach with labels
    print("🔨 Building reach from shapefiles...")
    reach = build_reach_with_labels(
        centerline_path=centerline_path,
        points_path=points_path,
        cross_sections_path=cross_sections_path,
        labels_dir=labels_dir,
        river_name=river_name,
        discharge=discharge,
        labels_min_mtime=labels_min_mtime,
    )
    
    print(f"✅ Built reach with {len(reach.nodes)} nodes")

    centerline_geom = None
    if flowfm_depth_nc and use_flowfm_slope:
        cl_gdf = gpd.read_file(centerline_path)
        if cl_gdf.empty:
            raise ValueError(f"No features found in centerline: {centerline_path}")
        centerline_geom = cl_gdf.geometry.iloc[0]
    
    # Run BASED analysis with bathymetry
    print("🔬 Running BASED analysis with bathymetry...")
    analyzer = BASEDAnalyzer(
        bathymetry_path=bathymetry_path,
        params_path=params_path,
        bathymetry_is_depth=bathymetry_is_depth,
        bathymetry_units=bathymetry_units
    )
    
    # Create initial DataFrame from nodes (as BASED does)
    print("📋 Creating initial DataFrame from nodes...")
    df = analyzer._nodes_to_dataframe(reach)
    
    if df.empty:
        raise ValueError("No nodes with cross-sections found.")
    
    print(f"✅ Found {len(df)} nodes with cross-sections")
    
    # Add channel/ridge picks
    if auto_minmax:
        print("📝 Auto-picking channel (min) and ridges (max left/right) from profiles...")
    else:
        print("📝 Adding labeled cross-section data...")
    if profiles_dir is None:
        profiles_dir = Path("cross_section_profiles")
    df = add_labels_to_dataframe(
        df,
        reach,
        labels_dir,
        river_name,
        profiles_dir=Path(profiles_dir),
        auto_minmax=auto_minmax,
        labels_min_mtime=labels_min_mtime,
    )
    
    if df.empty:
        raise ValueError(
            "No nodes with channel/ridge picks found. "
            "Check profile CSVs or label files."
        )
    
    print(f"✅ Found {len(df)} nodes with channel/ridge picks")
    print()
    
    # Run the BASED pipeline with the labeled DataFrame
    print("🔬 Running BASED calculations with bathymetry...")

    # Calculate discharge if params are available (optional)
    if analyzer.params is not None and 'discharge_value' in df.columns:
        df = analyzer._calculate_discharge(df)

    # Hm = alluvial ridge crest − channel bed (per bank; aggregation by ridge_rule)
    hm1 = (df["ridge1_elevation"] - df["channel_elevation"]).astype(float)
    hm2 = (df["ridge2_elevation"] - df["channel_elevation"]).astype(float)
    df["Hm_ridge1"] = hm1
    df["Hm_ridge2"] = hm2
    df["channel_depth"] = np.nanmean(np.vstack([hm1.to_numpy(), hm2.to_numpy()]), axis=0)
    df["channel_depth"] = df["channel_depth"].clip(lower=analyzer.min_slope)
    df["channel_depth_source"] = "ridge_crest_minus_channel_bed"
    df["ridge_rule"] = ridge_rule

    if flowfm_depth_nc and use_flowfm_slope and centerline_geom is not None:
        print(
            f"🌊 Slope from FlowFM ({flowfm_slope_method}, "
            f"XS WSE={flowfm_xs_wse}, window={flowfm_slope_window_widths:g}×width)"
        )
        df = slope_from_flowfm_waterlevel(
            df,
            reach,
            centerline_geom,
            flowfm_depth_nc,
            time_index=flowfm_time_index,
            window_widths=flowfm_slope_window_widths,
            backwater_length_m=flowfm_backwater_length_m,
            slope_min=max(analyzer.min_slope, 1e-5),
            method=flowfm_slope_method,
            xs_wse_reduce=flowfm_xs_wse,
        )
    else:
        print("🗻 Slope: DEM / node bed slope (paper-style Sm)")
        if "slope_dem" not in df.columns:
            df["slope_dem"] = df["slope"]
        df["slope_source"] = "dem_bed"

    df['slope'] = df['slope'].clip(lower=analyzer.min_slope)
    
    # Calculate ridge and floodplain parameters
    df = analyzer._calculate_ridge_parameters(df)

    # Gearon et al. ridge rule: alluvial ridge = lower of the two crest elevations.
    # If that bank is inverted (floodplain higher than the ridge, Har < 0) and
    # the other bank is not, use the other bank instead.
    if str(ridge_rule).lower() == "lower":
        r1 = df["ridge1_elevation"].astype(float)
        r2 = df["ridge2_elevation"].astype(float)
        har1 = df["Har_ridge1"].astype(float)
        har2 = df["Har_ridge2"].astype(float)
        pos1 = har1 >= 0
        pos2 = har2 >= 0
        only_pos1 = pos1 & ~pos2
        only_pos2 = pos2 & ~pos1
        use_bank1 = r1 <= r2
        switched = (only_pos1 & ~use_bank1) | (only_pos2 & use_bank1)
        both_inverted = ~pos1 & ~pos2
        use_bank1 = np.where(only_pos1, True, np.where(only_pos2, False, use_bank1))
        df["ridge_bank_used"] = np.where(use_bank1, 1, 2)
        df["ridge_inverted_switched"] = switched
        df["ridge_both_inverted"] = both_inverted
        for col in ("Har_ridge1", "SAR_ridge1", "Hm_ridge1"):
            if col in df.columns:
                df.loc[~use_bank1, col] = np.nan
        for col in ("Har_ridge2", "SAR_ridge2", "Hm_ridge2"):
            if col in df.columns:
                df.loc[use_bank1, col] = np.nan
        print(
            f"🏔️  Lower-ridge rule: bank1={int(use_bank1.sum())}, "
            f"bank2={int((~use_bank1).sum())} nodes"
        )
        if int(switched.sum()):
            print(
                f"   Switched {int(switched.sum())} nodes to the other bank "
                f"because the lower crest had floodplain above the ridge"
            )
        if int(both_inverted.sum()):
            print(
                f"   {int(both_inverted.sum())} nodes have floodplain above "
                f"both ridges; kept the lower crest"
            )
        # Recompute Har_mean from the retained bank only
        df["Har_mean"] = np.nanmean(
            np.vstack(
                [
                    df["Har_ridge1"].to_numpy(dtype=float),
                    df["Har_ridge2"].to_numpy(dtype=float),
                ]
            ),
            axis=0,
        )
    else:
        df["ridge_bank_used"] = 0  # both / mean
        df["ridge_inverted_switched"] = False
        df["ridge_both_inverted"] = False

    df = analyzer._calculate_gamma(df)
    df = analyzer._calculate_superelevation(df)

    # channel_depth / Har_mean reflect the active bank(s) after ridge rule
    print(
        f"📏 Hm = ridge crest − channel bed: "
        f"{df['channel_depth'].min():.2f} – {df['channel_depth'].max():.2f} m "
        f"(median {df['channel_depth'].median():.2f} m)"
    )
    
    # Calculate lambda
    df['lambda'] = df['gamma_mean'] * df['superelevation_mean']
    
    # Add quality flags
    df = analyzer._add_quality_flags(df)
    
    print(f"✅ Calculated lambda for {len(df)} nodes")
    print(f"📊 Lambda range: {df['lambda'].min():.4f} - {df['lambda'].max():.4f}")
    print(f"📊 Lambda mean: {df['lambda'].mean():.4f}")
    print()
    
    # Save results
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Saved results to {output_path}")
    else:
        output_path = Path(f"data/{river_name}_lambda_results.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"💾 Saved results to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate lambda values from labeled cross-sections"
    )
    parser.add_argument(
        "--centerline",
        type=str,
        default=DEFAULT_CENTERLINE,
        help="Path to centerline shapefile",
    )
    parser.add_argument(
        "--points",
        type=str,
        default=DEFAULT_POINTS,
        help="Path to centerline points shapefile",
    )
    parser.add_argument(
        "--cross-sections",
        type=str,
        default="cross_sections.shp",
        help="Path to cross-sections shapefile",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        help="Directory containing labeler CSV files (default picks)",
    )
    parser.add_argument(
        "--labels-since",
        type=str,
        default=None,
        help=(
            "Only use label CSVs saved at or after this local time "
            "(YYYY-MM-DD or YYYY-MM-DDTHH:MM). Skips leftover remapped files."
        ),
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default="cross_section_profiles",
        help="Directory with cross-section profile CSVs (only for --auto-minmax)",
    )
    parser.add_argument(
        "--auto-minmax",
        action="store_true",
        help="Ignore labeler CSVs; pick channel=min elev and ridges=max left/right",
    )
    parser.add_argument(
        "--manual-labels",
        action="store_true",
        help="Deprecated alias: manual labels are now the default",
    )
    parser.add_argument(
        "--river-name",
        type=str,
        default="Nooksack",
        help="Name of the river",
    )
    parser.add_argument(
        "--discharge",
        type=float,
        default=DEFAULT_DISCHARGE,
        help=f"Discharge value (m³/s) - Default: {DEFAULT_DISCHARGE} m³/s (set in code). Can also provide per-node discharge via CSV.",
    )
    parser.add_argument(
        "--bathymetry-path",
        type=str,
        default=DEFAULT_BATHYMETRY,
        help="Path to bathymetry/topobathy raster (.tif)",
    )
    parser.add_argument(
        "--params-path",
        type=str,
        default=None,
        help="Path to inverse power law parameters (optional, only needed for discharge correction)",
    )
    parser.add_argument(
        "--bathymetry-is-depth",
        action="store_true",
        default=False,
        help="Raster values are depths below water surface (default: False = elevation/topobathy).",
    )
    parser.add_argument(
        "--no-bathymetry-is-depth",
        dest="bathymetry_is_depth",
        action="store_false",
        help="Raster values are elevations (topobathy). Depth = max(z)-min(z) in channel.",
    )
    parser.add_argument(
        "--bathymetry-units",
        type=str,
        choices=['ft', 'm'],
        default='m',
        help="Units of bathymetry data (default: 'm')",
    )
    parser.add_argument(
        "--flowfm-slope",
        action="store_true",
        help=(
            "Use FlowFM water-surface slope for Sm instead of DEM bed slope "
            "(default is DEM bed slope)"
        ),
    )
    parser.add_argument(
        "--no-flowfm-slope",
        action="store_true",
        help=argparse.SUPPRESS,  # deprecated; DEM Sm is already the default
    )
    parser.add_argument(
        "--flowfm-depth-nc",
        type=str,
        default=DEFAULT_FLOWFM_NC,
        help=(
            "FlowFM/Delft3D-FM map NetCDF used only with --flowfm-slope "
            f"(default: {DEFAULT_FLOWFM_NC}). Hm is always ridge − channel bed."
        ),
    )
    parser.add_argument(
        "--flowfm-time-index",
        type=str,
        default="-1",
        help=(
            "Timestep for FlowFM slope: integer index (default -1 = last) "
            "or 'max' for the per-face maximum over all times"
        ),
    )
    parser.add_argument(
        "--flowfm-slope-window-widths",
        type=float,
        default=DEFAULT_SLOPE_WINDOW_WIDTHS,
        help=(
            "FlowFM WSE slope window as a multiple of local channel width "
            f"(used only if --flowfm-backwater-length-m is not set; "
            f"default {DEFAULT_SLOPE_WINDOW_WIDTHS:g})"
        ),
    )
    parser.add_argument(
        "--flowfm-backwater-length-m",
        type=float,
        default=DEFAULT_FLOWFM_SLOPE_WINDOW_M,
        help=(
            "Fixed centerline window (m) for FlowFM WSE slope "
            f"(default {DEFAULT_FLOWFM_SLOPE_WINDOW_M:.0f}). "
            "Overrides --flowfm-slope-window-widths. Use 0 for width-scaled windows."
        ),
    )
    parser.add_argument(
        "--flowfm-slope-method",
        type=str,
        choices=["reach_xs", "centerline"],
        default="reach_xs",
        help=(
            "FlowFM Sm method: 'reach_xs' = max/mean WSE along each XS then "
            "reach-average vs station (default); 'centerline' = sample WSE on centerline"
        ),
    )
    parser.add_argument(
        "--flowfm-xs-wse",
        type=str,
        choices=["max", "mean", "median"],
        default="max",
        help="How to reduce WSE along each cross-section for reach_xs (default: max)",
    )
    parser.add_argument(
        "--ridge-rule",
        type=str,
        choices=["lower", "mean"],
        default="lower",
        help=(
            "How to choose the alluvial ridge: 'lower' = bank with lower crest "
            "(Gearon et al., default); if that bank has floodplain above the "
            "ridge, use the other bank when it is still a real ridge. "
            "'mean' = average both banks"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/{river_name}_lambda_results.csv)",
    )
    
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent

    def proj_path(s: str) -> Path:
        p = Path(s)
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    # Validate inputs (relative paths are resolved from this script's project folder)
    centerline_path = proj_path(args.centerline)
    points_path = proj_path(args.points)
    cross_sections_path = proj_path(args.cross_sections)
    labels_dir = proj_path(args.labels_dir)
    profiles_dir = proj_path(args.profiles_dir)
    auto_minmax = bool(args.auto_minmax)
    if args.manual_labels and args.auto_minmax:
        print("❌ Error: use either --manual-labels (default) or --auto-minmax, not both")
        return 1
    
    if not centerline_path.exists():
        print(f"❌ Error: Centerline file not found: {centerline_path}")
        return 1
    
    if not points_path.exists():
        print(f"❌ Error: Points file not found: {points_path}")
        return 1
    
    if not cross_sections_path.exists():
        print(f"❌ Error: Cross-sections file not found: {cross_sections_path}")
        return 1

    if auto_minmax:
        if not profiles_dir.exists():
            print(f"❌ Error: Profiles directory not found: {profiles_dir}")
            return 1
    else:
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_csvs = sorted(
            {p for p in labels_dir.glob("*.csv") if "_labels" in p.name.lower()}
        )
        if not label_csvs:
            print("No label CSVs found.")
            print(f"  Folder: {labels_dir}")
            print(
                f"  Expected filenames like: {args.river_name}_node_<id>_labels.csv "
                "(from the cross-section labeler)."
            )
            print("  Use --labels-dir if your exports live somewhere else.")
            return 1
    
    # Calculate lambda
    try:
        bathy = proj_path(args.bathymetry_path)
        params = proj_path(args.params_path) if args.params_path else None
        out = proj_path(args.output) if args.output else None

        use_flowfm = bool(args.flowfm_slope) and not bool(args.no_flowfm_slope)
        flowfm_nc = None
        if use_flowfm:
            flowfm_nc = str(proj_path(args.flowfm_depth_nc))
            if not Path(flowfm_nc).is_file():
                print(f"❌ Error: FlowFM NetCDF not found: {flowfm_nc}")
                print("   Omit --flowfm-slope to use DEM bed slopes.")
                return 1
            print("ℹ️  --flowfm-slope: using FlowFM water-surface slopes")
        else:
            print("ℹ️  Using DEM bed slopes (default). Pass --flowfm-slope for FlowFM Sm.")

        t_raw = str(args.flowfm_time_index).strip().lower()
        flowfm_time: object
        if t_raw == "max":
            flowfm_time = "max"
        else:
            try:
                flowfm_time = int(t_raw)
            except ValueError:
                print(
                    f"❌ Error: --flowfm-time-index must be an integer or 'max', got {args.flowfm_time_index!r}"
                )
                return 1

        # 0 disables fixed window → width-scaled windows
        bw = args.flowfm_backwater_length_m
        if bw is not None and float(bw) <= 0:
            bw = None

        labels_min_mtime = None
        if args.labels_since:
            raw = args.labels_since.strip()
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    labels_min_mtime = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
            if labels_min_mtime is None:
                print(
                    f"❌ Error: --labels-since must be YYYY-MM-DD or YYYY-MM-DDTHH:MM, "
                    f"got {args.labels_since!r}"
                )
                return 1

        df = calculate_lambda_from_labels(
            centerline_path=str(centerline_path),
            points_path=str(points_path),
            cross_sections_path=str(cross_sections_path),
            labels_dir=labels_dir,
            river_name=args.river_name,
            discharge=args.discharge,
            bathymetry_path=str(bathy),
            params_path=str(params) if params is not None else None,
            bathymetry_is_depth=args.bathymetry_is_depth,
            bathymetry_units=args.bathymetry_units,
            output_path=out,
            profiles_dir=profiles_dir,
            auto_minmax=auto_minmax,
            flowfm_depth_nc=flowfm_nc,
            flowfm_time_index=flowfm_time,
            flowfm_slope_window_widths=args.flowfm_slope_window_widths,
            flowfm_backwater_length_m=bw,
            use_flowfm_slope=use_flowfm,
            flowfm_slope_method=args.flowfm_slope_method,
            flowfm_xs_wse=args.flowfm_xs_wse,
            ridge_rule=args.ridge_rule,
            labels_min_mtime=labels_min_mtime,
        )
        
        print("\n✅ Lambda calculation complete!")
        print(f"\n📊 Summary:")
        print(f"   Nodes processed: {len(df)}")
        print(f"   Lambda min: {df['lambda'].min():.4f}")
        print(f"   Lambda max: {df['lambda'].max():.4f}")
        print(f"   Lambda mean: {df['lambda'].mean():.4f}")
        print(f"   Lambda median: {df['lambda'].median():.4f}")
        
        # Generate plot
        print("\n📈 Generating lambda vs distance plot...")
        plot_output_path = None
        if out is not None:
            plot_output_path = out.with_name(out.name.replace(".csv", "_plot.png"))
        plot_lambda_vs_distance(df, args.river_name, plot_output_path)
        
        return 0
    except Exception as e:
        print(f"\n❌ Error calculating lambda: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
