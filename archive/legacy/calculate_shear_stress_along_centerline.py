#!/usr/bin/env python3
"""
Calculate shear stress along a river centerline.

Calculates both:
1. Actual shear stress: τ = ρ_w * g * R * S
   where R is hydraulic radius (approximated as depth) and S is slope
2. Critical shear stress: τ_cr = (1/2) * f_w * (ρ_s - ρ_w) * g * d_50

Usage:
    poetry run python calculate_shear_stress_along_centerline.py --d50-csv data/nooksack_d50_along_centerline.csv --output data/shear_stress_along_centerline.csv --plot data/shear_stress_plot.png
    poetry run python calculate_shear_stress_along_centerline.py --d50-csv data/nooksack_d50_along_centerline.csv --slope 0.001 --depth 2.0 --output data/shear_stress_along_centerline.csv --plot data/shear_stress_plot.png
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Physical constants
RHO_WATER = 1000.0  # kg/m³ (water density at 4°C)
RHO_SEDIMENT = 2650.0  # kg/m³ (typical quartz sand/gravel density)
GRAVITY = 9.81  # m/s²
FRICTION_FACTOR = 0.03  # Typical friction factor for gravel-bed rivers (dimensionless)


def calculate_critical_shear_stress(
    d50_mm: float,
    f_w: float = FRICTION_FACTOR,
    rho_s: float = RHO_SEDIMENT,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY
) -> float:
    """
    Calculate critical shear stress for initiation of motion.
    
    Formula: τ_cr = (1/2) * f_w * (ρ_s - ρ_w) * g * d_50
    
    Args:
        d50_mm: Median grain size in millimeters
        f_w: Friction factor (default: 0.03)
        rho_s: Sediment density in kg/m³ (default: 2650)
        rho_w: Water density in kg/m³ (default: 1000)
        g: Acceleration due to gravity in m/s² (default: 9.81)
        
    Returns:
        Critical shear stress in Pa (N/m²)
    """
    # Convert d50 from mm to m
    d50_m = d50_mm / 1000.0
    
    # Calculate critical shear stress
    tau_cr = 0.5 * f_w * (rho_s - rho_w) * g * d50_m
    
    return max(0.0, tau_cr)  # Ensure non-negative


def calculate_actual_shear_stress(
    depth: float,
    slope: float,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY
) -> float:
    """
    Calculate actual shear stress in the channel.
    
    Formula: τ = ρ_w * g * R * S
    where R (hydraulic radius) is approximated as depth for wide channels.
    
    Args:
        depth: Channel depth in meters
        slope: Channel slope (dimensionless, m/m)
        rho_w: Water density in kg/m³ (default: 1000)
        g: Acceleration due to gravity in m/s² (default: 9.81)
        
    Returns:
        Actual shear stress in Pa (N/m²)
    """
    # For wide channels, hydraulic radius R ≈ depth
    hydraulic_radius = depth
    
    # Calculate shear stress
    tau = rho_w * g * hydraulic_radius * slope
    
    return max(0.0, tau)  # Ensure non-negative


def calculate_shear_stress_along_centerline(
    d50_csv_path: str,
    output_path: str,
    slope: Optional[float] = None,
    depth: Optional[float] = None,
    slope_csv_path: Optional[str] = None,
    depth_csv_path: Optional[str] = None,
    constant_slope: Optional[float] = None,
    constant_depth: Optional[float] = None,
    f_w: float = FRICTION_FACTOR,
    rho_s: float = RHO_SEDIMENT,
    rho_w: float = RHO_WATER
) -> None:
    """
    Calculate shear stress along centerline from D50 data.
    
    Args:
        d50_csv_path: Path to CSV with D50 data (from calculate_d50_along_centerline.py)
        output_path: Path to save output CSV
        slope: Constant slope value (m/m) to use for all points
        depth: Constant depth value (m) to use for all points
        slope_csv_path: Path to CSV with slope data (must have 'river_mile' or 'dist_along_m' column)
        depth_csv_path: Path to CSV with depth data (must have 'river_mile' or 'dist_along_m' column)
        constant_slope: Alias for slope parameter
        constant_depth: Alias for depth parameter
        f_w: Friction factor for critical shear stress
        rho_s: Sediment density
        rho_w: Water density
    """
    # Read D50 data
    df = pd.read_csv(d50_csv_path)
    
    print(f"Loaded {len(df)} points from {d50_csv_path}")
    
    # Use aliases if provided
    if constant_slope is not None:
        slope = constant_slope
    if constant_depth is not None:
        depth = constant_depth
    
    # Load slope data if provided
    slope_data = None
    if slope_csv_path:
        slope_data = pd.read_csv(slope_csv_path)
        print(f"Loaded slope data from {slope_csv_path}")
        # Try to match on river_mile or dist_along_m
        if 'river_mile' in slope_data.columns and 'river_mile' in df.columns:
            df = df.merge(slope_data[['river_mile', 'slope']], on='river_mile', how='left', suffixes=('', '_from_slope'))
            if 'slope_from_slope' in df.columns:
                df['slope'] = df['slope_from_slope'].fillna(df.get('slope', slope))
        elif 'dist_along_m' in slope_data.columns and 'dist_along_m' in df.columns:
            df = df.merge(slope_data[['dist_along_m', 'slope']], on='dist_along_m', how='left', suffixes=('', '_from_slope'))
            if 'slope_from_slope' in df.columns:
                df['slope'] = df['slope_from_slope'].fillna(df.get('slope', slope))
    
    # Load depth data if provided
    depth_data = None
    if depth_csv_path:
        depth_data = pd.read_csv(depth_csv_path)
        print(f"Loaded depth data from {depth_csv_path}")
        # Try to match on river_mile or dist_along_m
        if 'river_mile' in depth_data.columns and 'river_mile' in df.columns:
            df = df.merge(depth_data[['river_mile', 'depth']], on='river_mile', how='left', suffixes=('', '_from_depth'))
            if 'depth_from_depth' in df.columns:
                df['depth'] = df['depth_from_depth'].fillna(df.get('depth', depth))
        elif 'dist_along_m' in depth_data.columns and 'dist_along_m' in df.columns:
            df = df.merge(depth_data[['dist_along_m', 'depth']], on='dist_along_m', how='left', suffixes=('', '_from_depth'))
            if 'depth_from_depth' in df.columns:
                df['depth'] = df['depth_from_depth'].fillna(df.get('depth', depth))
    
    # Set constant values if provided
    if slope is not None:
        if 'slope' not in df.columns:
            df['slope'] = slope
        else:
            df['slope'] = df['slope'].fillna(slope)
        print(f"Using constant slope: {slope} m/m")
    
    if depth is not None:
        if 'depth' not in df.columns:
            df['depth'] = depth
        else:
            df['depth'] = df['depth'].fillna(depth)
        print(f"Using constant depth: {depth} m")
    
    # Calculate critical shear stress (always possible with D50)
    df['tau_cr_pa'] = df['d50_mm'].apply(
        lambda d: calculate_critical_shear_stress(d, f_w=f_w, rho_s=rho_s, rho_w=rho_w)
    )
    
    # Convert to N/m² (same as Pa) and also to dynes/cm² for reference
    df['tau_cr_dynes_per_cm2'] = df['tau_cr_pa'] * 10.0  # 1 Pa = 10 dynes/cm²
    
    # Calculate actual shear stress (if depth and slope are available)
    if 'depth' in df.columns and 'slope' in df.columns:
        df['tau_actual_pa'] = df.apply(
            lambda row: calculate_actual_shear_stress(
                row['depth'], 
                row['slope'],
                rho_w=rho_w
            ) if pd.notna(row['depth']) and pd.notna(row['slope']) else np.nan,
            axis=1
        )
        df['tau_actual_dynes_per_cm2'] = df['tau_actual_pa'] * 10.0
        
        # Calculate ratio of actual to critical (mobility parameter)
        df['tau_ratio'] = df['tau_actual_pa'] / df['tau_cr_pa']
        df['tau_ratio'] = df['tau_ratio'].replace([np.inf, -np.inf], np.nan)
        
        print(f"\nActual shear stress calculated for {df['tau_actual_pa'].notna().sum()} points")
    else:
        print("\n⚠️  Warning: Depth and/or slope not available. Only critical shear stress calculated.")
        print("   Provide --slope and --depth, or use --slope-csv and --depth-csv for variable values.")
    
    # Save results
    output_df = df.copy()
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to {output_path}")
    
    return df
    print(f"\nSummary statistics:")
    print(f"  Critical shear stress (τ_cr):")
    print(f"    Range: {df['tau_cr_pa'].min():.2f} - {df['tau_cr_pa'].max():.2f} Pa")
    print(f"    Mean: {df['tau_cr_pa'].mean():.2f} Pa")
    
    if 'tau_actual_pa' in df.columns:
        valid_tau = df['tau_actual_pa'].notna()
        if valid_tau.sum() > 0:
            print(f"  Actual shear stress (τ):")
            print(f"    Range: {df.loc[valid_tau, 'tau_actual_pa'].min():.2f} - {df.loc[valid_tau, 'tau_actual_pa'].max():.2f} Pa")
            print(f"    Mean: {df.loc[valid_tau, 'tau_actual_pa'].mean():.2f} Pa")
            print(f"  Mobility ratio (τ/τ_cr):")
            valid_ratio = df['tau_ratio'].notna()
            if valid_ratio.sum() > 0:
                print(f"    Range: {df.loc[valid_ratio, 'tau_ratio'].min():.3f} - {df.loc[valid_ratio, 'tau_ratio'].max():.3f}")
                print(f"    Mean: {df.loc[valid_ratio, 'tau_ratio'].mean():.3f}")
                print(f"    Points with τ/τ_cr > 1 (motion expected): {(df.loc[valid_ratio, 'tau_ratio'] > 1).sum()}")
    
    return df


def plot_shear_stress(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    show_plot: bool = False
) -> None:
    """
    Plot shear stress vs river mile.
    
    Args:
        df: DataFrame with shear stress data
        output_path: Path to save the plot (if None, plot is not saved)
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
    
    # Filter out zero D50 values for better visualization
    valid_mask = df['d50_mm'] > 0
    
    # Plot 1: Critical and Actual Shear Stress
    ax1 = axes[0]
    
    # Plot critical shear stress
    ax1.plot(df['river_mile'], df['tau_cr_pa'], 
            'b-', linewidth=2, alpha=0.8, label='Critical shear stress (τ_cr)')
    ax1.scatter(df.loc[valid_mask, 'river_mile'], df.loc[valid_mask, 'tau_cr_pa'], 
               s=15, alpha=0.5, color='blue', zorder=3)
    
    # Plot actual shear stress if available
    if 'tau_actual_pa' in df.columns:
        valid_tau = df['tau_actual_pa'].notna()
        if valid_tau.sum() > 0:
            ax1.plot(df.loc[valid_tau, 'river_mile'], df.loc[valid_tau, 'tau_actual_pa'], 
                    'r-', linewidth=2, alpha=0.8, label='Actual shear stress (τ)')
            ax1.scatter(df.loc[valid_tau & valid_mask, 'river_mile'], 
                       df.loc[valid_tau & valid_mask, 'tau_actual_pa'], 
                       s=15, alpha=0.5, color='red', zorder=3)
    
    # Add horizontal line at τ = τ_cr for reference
    if valid_mask.sum() > 0:
        mean_tau_cr = df.loc[valid_mask, 'tau_cr_pa'].mean()
        ax1.axhline(y=mean_tau_cr, color='gray', linestyle='--', alpha=0.5, 
                  label=f'Mean τ_cr: {mean_tau_cr:.2f} Pa')
    
    ax1.set_xlabel('River Mile (from upstream)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Shear Stress (Pa)', fontsize=14, fontweight='bold')
    ax1.set_title('Shear Stress vs River Mile\nNooksack River', 
                 fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=12, framealpha=0.9)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Set axis limits
    ax1.set_xlim(df['river_mile'].min() - 0.5, df['river_mile'].max() + 0.5)
    y_min = df['tau_cr_pa'].min()
    y_max = df['tau_cr_pa'].max()
    if 'tau_actual_pa' in df.columns and df['tau_actual_pa'].notna().sum() > 0:
        y_max = max(y_max, df['tau_actual_pa'].max())
    y_padding = (y_max - y_min) * 0.1 if y_max > y_min else y_max * 0.1
    ax1.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    # Plot 2: Mobility Ratio (τ/τ_cr)
    ax2 = axes[1]
    
    if 'tau_ratio' in df.columns:
        valid_ratio = df['tau_ratio'].notna() & valid_mask
        if valid_ratio.sum() > 0:
            ax2.plot(df.loc[valid_ratio, 'river_mile'], df.loc[valid_ratio, 'tau_ratio'], 
                    'g-', linewidth=2, alpha=0.8, label='Mobility ratio (τ/τ_cr)')
            ax2.scatter(df.loc[valid_ratio, 'river_mile'], df.loc[valid_ratio, 'tau_ratio'], 
                       s=15, alpha=0.5, color='green', zorder=3)
            
            # Add horizontal line at τ/τ_cr = 1 (critical threshold)
            ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                       label='τ/τ_cr = 1 (motion threshold)')
            
            # Shade regions where motion is expected
            ax2.fill_between(df.loc[valid_ratio, 'river_mile'], 1.0, 
                            df.loc[valid_ratio, 'tau_ratio'].max() * 1.1,
                            alpha=0.2, color='red', label='Motion expected (τ/τ_cr > 1)')
            
            ax2.set_ylabel('Mobility Ratio (τ/τ_cr)', fontsize=14, fontweight='bold')
            ax2.set_title('Mobility Ratio vs River Mile', 
                         fontsize=16, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=12, framealpha=0.9)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            # Set axis limits
            ax2.set_xlim(df['river_mile'].min() - 0.5, df['river_mile'].max() + 0.5)
            ratio_min = df.loc[valid_ratio, 'tau_ratio'].min()
            ratio_max = df.loc[valid_ratio, 'tau_ratio'].max()
            # Use log scale if range is large
            if ratio_max / ratio_min > 100:
                ax2.set_yscale('log')
                ax2.set_ylim(ratio_min * 0.5, ratio_max * 2)
            else:
                ratio_padding = (ratio_max - ratio_min) * 0.1
                ax2.set_ylim(max(0, ratio_min - ratio_padding), ratio_max + ratio_padding)
        else:
            ax2.text(0.5, 0.5, 'Mobility ratio not available\n(actual shear stress not calculated)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_ylabel('Mobility Ratio (τ/τ_cr)', fontsize=14, fontweight='bold')
            ax2.set_title('Mobility Ratio vs River Mile', 
                         fontsize=16, fontweight='bold', pad=15)
    else:
        ax2.text(0.5, 0.5, 'Mobility ratio not available\n(actual shear stress not calculated)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_ylabel('Mobility Ratio (τ/τ_cr)', fontsize=14, fontweight='bold')
        ax2.set_title('Mobility Ratio vs River Mile', 
                     fontsize=16, fontweight='bold', pad=15)
    
    ax2.set_xlabel('River Mile (from upstream)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate shear stress along river centerline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate only critical shear stress (using D50) with plot
  python calculate_shear_stress_along_centerline.py --d50-csv data/nooksack_d50_along_centerline.csv --output data/shear_stress.csv --plot data/shear_stress_plot.png
  
  # Calculate with constant slope and depth, save plot
  python calculate_shear_stress_along_centerline.py --d50-csv data/nooksack_d50_along_centerline.csv --slope 0.001 --depth 2.0 --output data/shear_stress.csv --plot data/shear_stress_plot.png
  
  # Use variable slope and depth from CSV files with plot
  python calculate_shear_stress_along_centerline.py --d50-csv data/nooksack_d50_along_centerline.csv --slope-csv data/slopes.csv --depth-csv data/depths.csv --output data/shear_stress.csv --plot data/shear_stress_plot.png
  
  # Display plot interactively (in addition to saving)
  python calculate_shear_stress_along_centerline.py --d50-csv data/nooksack_d50_along_centerline.csv --slope 0.001 --depth 2.0 --output data/shear_stress.csv --plot data/shear_stress_plot.png --show-plot
        """
    )
    
    parser.add_argument(
        '--d50-csv',
        type=str,
        required=True,
        help='Path to CSV file with D50 data (from calculate_d50_along_centerline.py)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save output CSV'
    )
    
    parser.add_argument(
        '--slope',
        type=float,
        default=None,
        help='Constant channel slope (m/m) to use for all points'
    )
    
    parser.add_argument(
        '--depth',
        type=float,
        default=None,
        help='Constant channel depth (m) to use for all points'
    )
    
    parser.add_argument(
        '--slope-csv',
        type=str,
        default=None,
        help='Path to CSV with variable slope data (must have river_mile or dist_along_m column)'
    )
    
    parser.add_argument(
        '--depth-csv',
        type=str,
        default=None,
        help='Path to CSV with variable depth data (must have river_mile or dist_along_m column)'
    )
    
    parser.add_argument(
        '--friction-factor',
        type=float,
        default=FRICTION_FACTOR,
        help=f'Friction factor for critical shear stress (default: {FRICTION_FACTOR})'
    )
    
    parser.add_argument(
        '--sediment-density',
        type=float,
        default=RHO_SEDIMENT,
        help=f'Sediment density in kg/m³ (default: {RHO_SEDIMENT})'
    )
    
    parser.add_argument(
        '--water-density',
        type=float,
        default=RHO_WATER,
        help=f'Water density in kg/m³ (default: {RHO_WATER})'
    )
    
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Path to save plot (e.g., data/shear_stress_plot.png). If not specified, no plot is created.'
    )
    
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display the plot interactively (in addition to saving if --plot is specified)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    d50_path = Path(args.d50_csv)
    if not d50_path.exists():
        print(f"Error: D50 CSV file not found: {d50_path}")
        return 1
    
    if args.slope_csv and not Path(args.slope_csv).exists():
        print(f"Error: Slope CSV file not found: {args.slope_csv}")
        return 1
    
    if args.depth_csv and not Path(args.depth_csv).exists():
        print(f"Error: Depth CSV file not found: {args.depth_csv}")
        return 1
    
    try:
        df = calculate_shear_stress_along_centerline(
            d50_csv_path=str(d50_path),
            output_path=args.output,
            slope=args.slope,
            depth=args.depth,
            slope_csv_path=args.slope_csv,
            depth_csv_path=args.depth_csv,
            f_w=args.friction_factor,
            rho_s=args.sediment_density,
            rho_w=args.water_density
        )
        
        # Create plot if requested
        if args.plot or args.show_plot:
            plot_shear_stress(
                df,
                output_path=args.plot,
                show_plot=args.show_plot
            )
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

