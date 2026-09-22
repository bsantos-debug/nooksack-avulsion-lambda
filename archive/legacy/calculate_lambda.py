#!/usr/bin/env python3
"""
Calculate lambda values from labeled cross-sections.

This script:
1. Loads labeled cross-sections (from the labeler)
2. Extracts channel, ridge, and floodplain positions
3. Builds a SwordReach with all required attributes
4. Runs BASED analysis to calculate lambda
5. Outputs lambda values mapped along the river

Usage:
    poetry run python calculate_lambda.py
    poetry run python calculate_lambda.py --discharge 100.0
    poetry run python calculate_lambda.py --output lambda_results.csv
"""

# === USER CONFIGURATION ===
# Default discharge value (m³/s) - can be overridden with --discharge argument
# Nooksack discharge: 3,860 ft³/s = 109.3 m³/s (1 ft³ = 0.0283168 m³)
DEFAULT_DISCHARGE = 109.3  # Nooksack river discharge
import sys
from pathlib import Path

# Ensure repo root is on path so `import avulsionprecursors` resolves to ./avulsionprecursors/
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from shapely.geometry import LineString, Point

from avulsionprecursors.geometry.reach_builder import build_reach_from_shapefiles
from avulsionprecursors.analysis.based import BASEDAnalyzer
from avulsionprecursors.sword.base import SwordNode, SwordReach


def load_labels(labels_dir: Path, river_name: str, node_id: int) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load labeled points for a specific node.
    
    Returns:
        Dictionary with keys: 'channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2'
        Each value is a dict with 'dist_along' and 'elevation'
    """
    label_file = labels_dir / f"{river_name}_node_{int(node_id)}_labels.csv"
    
    if not label_file.exists():
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


def build_reach_with_labels(
    centerline_path: str,
    points_path: str,
    cross_sections_path: str,
    labels_dir: Path,
    river_name: str,
    discharge: float,
    reach_id: int = 1
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
        labels = load_labels(labels_dir, river_name, node.node_id)
        
        if labels and node.cross_section:
            # Store label data and discharge on the node
            node._labels = labels
            node._discharge = discharge
        else:
            node._labels = None
            node._discharge = discharge
    
    return reach


def add_labels_to_dataframe(df: pd.DataFrame, reach: SwordReach, labels_dir: Path, river_name: str) -> pd.DataFrame:
    """
    Add labeled cross-section data to the DataFrame.
    
    Args:
        df: DataFrame from BASED's _nodes_to_dataframe
        reach: SwordReach with nodes
        labels_dir: Directory containing label CSV files
        river_name: Name of the river
        
    Returns:
        DataFrame with labeled cross-section attributes added
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
        
        # Load labels for this node
        labels = load_labels(labels_dir, river_name, node_id)
        
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
    
    # Filter to only nodes with labels
    df = df[df['channel_dist_along'].notna()].copy()
    
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
    model_path: str = "basic_model_20250202_164836.joblib",
    params_path: str = "inverse_power_law_params.pickle",
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Calculate lambda values from labeled cross-sections.
    
    Args:
        centerline_path: Path to centerline shapefile
        points_path: Path to centerline points shapefile
        cross_sections_path: Path to cross-sections shapefile
        labels_dir: Directory containing label CSV files
        river_name: Name of the river
        discharge: Discharge value (m³/s)
        model_path: Path to XGBoost model file
        params_path: Path to inverse power law parameters
        output_path: Optional path to save results
        
    Returns:
        DataFrame with lambda values and all BASED metrics
    """
    print(f"📊 Calculating lambda for {river_name}")
    print(f"📁 Labels directory: {labels_dir}")
    
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
        discharge=discharge
    )
    
    print(f"✅ Built reach with {len(reach.nodes)} nodes")
    
    # Run BASED analysis
    print("🔬 Running BASED analysis...")
    analyzer = BASEDAnalyzer(
        model_path=model_path,
        params_path=params_path
    )
    
    # Create initial DataFrame from nodes (as BASED does)
    print("📋 Creating initial DataFrame from nodes...")
    df = analyzer._nodes_to_dataframe(reach)
    
    if df.empty:
        raise ValueError("No nodes with cross-sections found.")
    
    print(f"✅ Found {len(df)} nodes with cross-sections")
    
    # Add labels to DataFrame
    print("📝 Adding labeled cross-section data...")
    df = add_labels_to_dataframe(df, reach, labels_dir, river_name)
    
    if df.empty:
        raise ValueError("No nodes with labels found. Please label cross-sections first.")
    
    print(f"✅ Found {len(df)} nodes with labels")
    print()
    
    # Now run the rest of the BASED pipeline
    print("🔬 Running BASED calculations...")
    df['slope'] = df['slope'].clip(lower=analyzer.min_slope)
    df = analyzer._calculate_discharge(df)
    df = analyzer._predict_depth(df)
    df = analyzer._calculate_ridge_parameters(df)
    df = analyzer._calculate_gamma(df)
    df = analyzer._calculate_superelevation(df)
    
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
        default="/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Nooksack_research/Qgis/Crosssections/Modern_Nooksack_centerline_clean.shp",
        help="Path to centerline shapefile",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Nooksack_research/Qgis/Crosssections/Modern_Nooksack_Centerline_Points_400m.shp",
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
        help="Directory containing label CSV files",
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
        "--model-path",
        type=str,
        default="basic_model_20250202_164836.joblib",
        help="Path to XGBoost model file",
    )
    parser.add_argument(
        "--params-path",
        type=str,
        default="inverse_power_law_params.pickle",
        help="Path to inverse power law parameters",
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
    
    if not centerline_path.exists():
        print(f"❌ Error: Centerline file not found: {centerline_path}")
        return 1
    
    if not points_path.exists():
        print(f"❌ Error: Points file not found: {points_path}")
        return 1
    
    if not cross_sections_path.exists():
        print(f"❌ Error: Cross-sections file not found: {cross_sections_path}")
        return 1
    
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
        out_csv = proj_path(args.output) if args.output else None
        df = calculate_lambda_from_labels(
            centerline_path=str(centerline_path),
            points_path=str(points_path),
            cross_sections_path=str(cross_sections_path),
            labels_dir=labels_dir,
            river_name=args.river_name,
            discharge=args.discharge,
            model_path=str(proj_path(args.model_path)),
            params_path=str(proj_path(args.params_path)),
            output_path=out_csv,
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
        if out_csv is not None:
            plot_output_path = out_csv.with_name(
                out_csv.name.replace(".csv", "_plot.png")
            )
        plot_lambda_vs_distance(df, args.river_name, plot_output_path)
        
        return 0
    except Exception as e:
        print(f"\n❌ Error calculating lambda: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
