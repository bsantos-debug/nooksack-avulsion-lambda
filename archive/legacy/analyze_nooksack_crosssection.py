#!/usr/bin/env python3
"""
Analyze Nooksack River cross-section and calculate:
- Superelevation (β): ratio of ridge height (HAR) to channel depth (HM)
- Gradient advantage (γ): ratio of ridge slope (SAR) to channel slope (SM)

Also plots the cross-section with identified features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def load_crosssection(csv_path: str, meas_num: int = None) -> pd.DataFrame:
    """
    Load cross-section data from CSV.
    
    Args:
        csv_path: Path to CSV file
        meas_num: Measurement number to filter by. If None, loads all measurements.
    
    Returns:
        DataFrame with cross-section data for the specified measurement
    """
    df = pd.read_csv(csv_path)
    
    # Filter by measurement number if specified
    if meas_num is not None:
        if 'meas_num' not in df.columns:
            raise ValueError("CSV file must contain 'meas_num' column to filter by measurement")
        df = df[df['meas_num'] == meas_num].copy()
        if len(df) == 0:
            raise ValueError(f"No data found for measurement number {meas_num}")
    
    # Convert station from feet to meters for consistency
    # (keeping elevation in feet as that's what the data uses)
    df['station_m'] = df['station_ft'] * 0.3048
    
    # Sort by station distance
    df = df.sort_values('station_m').reset_index(drop=True)
    
    return df


def identify_channel_bed(df: pd.DataFrame, window_size: int = 20) -> dict:
    """
    Identify channel bed (lowest elevation point).
    
    Uses the global minimum as the channel bed, or the minimum
    in the central portion of the cross-section.
    """
    # Find global minimum
    min_idx = df['elev_navd88_ft'].idxmin()
    min_station = df.loc[min_idx, 'station_m']
    min_elev = df.loc[min_idx, 'elev_navd88_ft']
    
    # Also find minimum in central portion (middle 40%)
    center_start = df['station_m'].quantile(0.3)
    center_end = df['station_m'].quantile(0.7)
    center_mask = (df['station_m'] >= center_start) & (df['station_m'] <= center_end)
    center_min_idx = df.loc[center_mask, 'elev_navd88_ft'].idxmin()
    
    return {
        'global_min_idx': min_idx,
        'global_min_station': min_station,
        'global_min_elev': min_elev,
        'center_min_idx': center_min_idx,
        'center_min_station': df.loc[center_min_idx, 'station_m'],
        'center_min_elev': df.loc[center_min_idx, 'elev_navd88_ft']
    }


def identify_ridges(df: pd.DataFrame, channel_bed_idx: int, 
                    min_distance: float = 10.0, prominence: float = 0.5) -> dict:
    """
    Identify ridges on left and right sides of channel.
    
    Args:
        df: DataFrame with cross-section data
        channel_bed_idx: Index of channel bed point
        min_distance: Minimum distance between peaks (in meters)
        prominence: Minimum prominence of peaks (in feet)
    
    Returns:
        Dictionary with ridge information
    """
    # Get elevation profile and smooth it
    elev = df['elev_navd88_ft'].values
    station = df['station_m'].values
    
    # Smooth elevation profile to reduce noise
    if len(elev) > 5:
        window_length = min(11, len(elev) // 4 * 2 + 1)  # Must be odd
        if window_length >= 5:
            elev_smooth = savgol_filter(elev, window_length, 3)
        else:
            elev_smooth = elev
    else:
        elev_smooth = elev
    
    # Find peaks (ridges)
    min_distance_idx = int(min_distance / np.mean(np.diff(station)))
    peaks, properties = find_peaks(elev_smooth, 
                                  distance=max(1, min_distance_idx),
                                  prominence=prominence)
    
    # Separate into left and right ridges relative to channel
    channel_station = df.loc[channel_bed_idx, 'station_m']
    left_peaks = peaks[station[peaks] < channel_station]
    right_peaks = peaks[station[peaks] > channel_station]
    
    # Get the ridge closest to channel on each side (but not too close)
    left_ridge_idx = None
    right_ridge_idx = None
    
    if len(left_peaks) > 0:
        # Find peak with maximum prominence that's far enough from channel
        left_distances = channel_station - station[left_peaks]
        valid_left = left_distances > 5.0  # At least 5m from channel
        if valid_left.any():
            valid_left_peaks = left_peaks[valid_left]
            valid_left_prominences = properties['prominences'][np.isin(peaks, valid_left_peaks)]
            best_left = valid_left_peaks[np.argmax(valid_left_prominences)]
            left_ridge_idx = best_left
        else:
            # Use the most prominent peak even if close
            left_ridge_idx = left_peaks[np.argmax(properties['prominences'][np.isin(peaks, left_peaks)])]
    
    if len(right_peaks) > 0:
        right_distances = station[right_peaks] - channel_station
        valid_right = right_distances > 5.0
        if valid_right.any():
            valid_right_peaks = right_peaks[valid_right]
            valid_right_prominences = properties['prominences'][np.isin(peaks, valid_right_peaks)]
            best_right = valid_right_peaks[np.argmax(valid_right_prominences)]
            right_ridge_idx = best_right
        else:
            right_ridge_idx = right_peaks[np.argmax(properties['prominences'][np.isin(peaks, right_peaks)])]
    
    # Get ridge information
    ridges = {}
    if left_ridge_idx is not None:
        ridges['left'] = {
            'idx': left_ridge_idx,
            'station': station[left_ridge_idx],
            'elev': elev[left_ridge_idx]
        }
    
    if right_ridge_idx is not None:
        ridges['right'] = {
            'idx': right_ridge_idx,
            'station': station[right_ridge_idx],
            'elev': elev[right_ridge_idx]
        }
    
    return ridges


def calculate_channel_depth(df: pd.DataFrame, channel_bed_elev: float,
                           method: str = 'bankfull') -> float:
    """
    Calculate channel depth (HM).
    
    Methods:
    - 'bankfull': Use difference between channel bed and average of ridge elevations
    - 'min_max': Use difference between channel bed and average floodplain elevation
    - 'percentile': Use 90th percentile elevation minus channel bed
    """
    if method == 'bankfull':
        # Estimate bankfull as average of lower ridge elevations or floodplain
        # Use 75th percentile of elevations as proxy for bankfull
        bankfull_elev = df['elev_navd88_ft'].quantile(0.75)
        depth = bankfull_elev - channel_bed_elev
    elif method == 'min_max':
        # Use average of outer elevations (top 10% of elevation range)
        elev_range = df['elev_navd88_ft'].max() - df['elev_navd88_ft'].min()
        bankfull_elev = df['elev_navd88_ft'].min() + 0.9 * elev_range
        depth = bankfull_elev - channel_bed_elev
    else:  # percentile
        bankfull_elev = df['elev_navd88_ft'].quantile(0.90)
        depth = bankfull_elev - channel_bed_elev
    
    return max(depth, 0.1)  # Minimum depth of 0.1 ft


def calculate_ridge_height(ridge_elev: float, channel_bed_elev: float) -> float:
    """Calculate ridge height above channel bed (HAR)."""
    return ridge_elev - channel_bed_elev


def calculate_superelevation(ridge_height: float, channel_depth: float) -> float:
    """
    Calculate superelevation (β) = HAR / HM.
    
    β: ratio of ridge height (HAR) to channel depth (HM)
    """
    if channel_depth <= 0:
        return np.nan
    return ridge_height / channel_depth


def calculate_ridge_slope(df: pd.DataFrame, ridge_idx: int, 
                         channel_bed_idx: int, window: int = 10) -> float:
    """
    Calculate ridge slope (SAR) in cross-section direction.
    
    Uses the slope from channel bed to ridge along the cross-section.
    Slope is calculated as rise over run (elevation change / horizontal distance).
    """
    ridge_station = df.loc[ridge_idx, 'station_m']
    ridge_elev = df.loc[ridge_idx, 'elev_navd88_ft']
    channel_station = df.loc[channel_bed_idx, 'station_m']
    channel_elev = df.loc[channel_bed_idx, 'elev_navd88_ft']
    
    horizontal_dist = abs(ridge_station - channel_station)
    vertical_dist = ridge_elev - channel_elev
    
    if horizontal_dist < 0.1:  # Too close, avoid division by zero
        return np.nan
    
    slope = vertical_dist / horizontal_dist
    
    return slope


def calculate_channel_slope(df: pd.DataFrame, channel_bed_idx: int,
                           longitudinal_slope: float = None) -> float:
    """
    Calculate channel slope (SM).
    
    NOTE: For accurate gradient advantage (γ) calculations, the longitudinal
    channel slope (downstream slope along the river) should be provided.
    Cross-sectional slope is not the same as longitudinal slope.
    
    If longitudinal_slope is provided, use that. Otherwise,
    estimate from cross-section profile characteristics (which is not
    ideal for calculating γ since it's a cross-sectional gradient, not
    longitudinal).
    """
    if longitudinal_slope is not None:
        return abs(longitudinal_slope)
    
    # Estimate from local gradient around channel (cross-sectional)
    # This is NOT the same as longitudinal slope!
    # For proper γ calculation, use --longitudinal-slope parameter
    window = min(20, len(df) // 10)
    start_idx = max(0, channel_bed_idx - window)
    end_idx = min(len(df), channel_bed_idx + window)
    
    if end_idx - start_idx < 2:
        return np.nan
    
    local_station = df.loc[start_idx:end_idx, 'station_m'].values
    local_elev = df.loc[start_idx:end_idx, 'elev_navd88_ft'].values
    
    # Fit linear trend (cross-sectional slope)
    coeffs = np.polyfit(local_station, local_elev, 1)
    slope = abs(coeffs[0])
    
    return slope


def calculate_gradient_advantage(ridge_slope: float, channel_slope: float) -> float:
    """
    Calculate gradient advantage (γ) = SAR / SM.
    
    γ: ratio of ridge slope (SAR) to channel slope (SM)
    """
    if channel_slope <= 0 or np.isnan(channel_slope):
        return np.nan
    if np.isnan(ridge_slope):
        return np.nan
    return abs(ridge_slope) / channel_slope


def analyze_crosssection(csv_path: str, meas_num: int = None,
                        longitudinal_slope: float = None,
                        channel_depth_method: str = 'bankfull') -> dict:
    """
    Perform complete analysis of cross-section.
    
    Args:
        csv_path: Path to CSV file
        meas_num: Measurement number to analyze. If None, uses first measurement found.
        longitudinal_slope: Longitudinal channel slope for calculating γ
        channel_depth_method: Method for estimating channel depth
    
    Returns dictionary with all calculated metrics.
    """
    # Load data
    df = load_crosssection(csv_path, meas_num=meas_num)
    
    # Get measurement number from data if not specified
    if meas_num is None and 'meas_num' in df.columns:
        meas_num = df['meas_num'].iloc[0]
    
    # Identify channel bed
    channel_info = identify_channel_bed(df)
    channel_bed_idx = channel_info['center_min_idx']  # Use center minimum
    channel_bed_elev = channel_info['center_min_elev']
    
    # Identify ridges
    ridges = identify_ridges(df, channel_bed_idx)
    
    # Calculate channel depth
    channel_depth = calculate_channel_depth(df, channel_bed_elev, 
                                           method=channel_depth_method)
    
    # Calculate channel slope once (should be longitudinal slope)
    channel_slope = calculate_channel_slope(df, channel_bed_idx, longitudinal_slope)
    
    # Calculate metrics for each ridge
    results = {
        'meas_num': meas_num if meas_num is not None else df.get('meas_num', [None])[0] if 'meas_num' in df.columns else None,
        'channel_bed': {
            'idx': channel_bed_idx,
            'station_m': channel_info['center_min_station'],
            'elevation_ft': channel_bed_elev
        },
        'channel_depth_ft': channel_depth,
        'channel_slope': channel_slope,
        'ridges': {}
    }
    
    # Process left ridge
    if 'left' in ridges:
        ridge = ridges['left']
        ridge_height = calculate_ridge_height(ridge['elev'], channel_bed_elev)
        beta_left = calculate_superelevation(ridge_height, channel_depth)
        ridge_slope = calculate_ridge_slope(df, ridge['idx'], channel_bed_idx)
        gamma_left = calculate_gradient_advantage(ridge_slope, channel_slope)
        
        results['ridges']['left'] = {
            'idx': ridge['idx'],
            'station_m': ridge['station'],
            'elevation_ft': ridge['elev'],
            'ridge_height_ft': ridge_height,
            'beta': beta_left,
            'ridge_slope': ridge_slope,
            'gamma': gamma_left
        }
    
    # Process right ridge
    if 'right' in ridges:
        ridge = ridges['right']
        ridge_height = calculate_ridge_height(ridge['elev'], channel_bed_elev)
        beta_right = calculate_superelevation(ridge_height, channel_depth)
        ridge_slope = calculate_ridge_slope(df, ridge['idx'], channel_bed_idx)
        gamma_right = calculate_gradient_advantage(ridge_slope, channel_slope)
        
        results['ridges']['right'] = {
            'idx': ridge['idx'],
            'station_m': ridge['station'],
            'elevation_ft': ridge['elev'],
            'ridge_height_ft': ridge_height,
            'beta': beta_right,
            'ridge_slope': ridge_slope,
            'gamma': gamma_right
        }
    
    # Calculate mean values
    if 'left' in results['ridges'] and 'right' in results['ridges']:
        results['beta_mean'] = np.mean([results['ridges']['left']['beta'],
                                       results['ridges']['right']['beta']])
        results['gamma_mean'] = np.mean([results['ridges']['left']['gamma'],
                                        results['ridges']['right']['gamma']])
    elif 'left' in results['ridges']:
        results['beta_mean'] = results['ridges']['left']['beta']
        results['gamma_mean'] = results['ridges']['left']['gamma']
    elif 'right' in results['ridges']:
        results['beta_mean'] = results['ridges']['right']['beta']
        results['gamma_mean'] = results['ridges']['right']['gamma']
    
    # Store dataframe for plotting
    results['dataframe'] = df
    
    return results


def plot_crosssection(results: dict, save_path: str = None, show: bool = True):
    """
    Plot cross-section with identified features and metrics.
    """
    df = results['dataframe']
    channel_bed = results['channel_bed']
    meas_num = results.get('meas_num', 'Unknown')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot elevation profile
    ax.plot(df['station_m'], df['elev_navd88_ft'], 'k-', linewidth=1.5, 
            label='Bed Elevation', alpha=0.7)
    
    # Mark channel bed
    ax.plot(channel_bed['station_m'], channel_bed['elevation_ft'], 
           'bo', markersize=12, label='Channel Bed', zorder=5)
    
    # Mark ridges
    if 'left' in results['ridges']:
        ridge = results['ridges']['left']
        ax.plot(ridge['station_m'], ridge['elevation_ft'], 
               'r^', markersize=12, label='Left Ridge', zorder=5)
    
    if 'right' in results['ridges']:
        ridge = results['ridges']['right']
        ax.plot(ridge['station_m'], ridge['elevation_ft'], 
               'g^', markersize=12, label='Right Ridge', zorder=5)
    
    # Add fill for channel area
    channel_elev = channel_bed['elevation_ft']
    bankfull_elev = channel_elev + results['channel_depth_ft']
    ax.fill_between(df['station_m'], channel_elev, bankfull_elev,
                    alpha=0.2, color='blue', label='Channel Depth (HM)')
    
    # Add text box with metrics
    metrics_text = []
    metrics_text.append(f"Measurement #{meas_num}")
    metrics_text.append(f"Channel Depth (HM): {results['channel_depth_ft']:.2f} ft")
    
    if 'left' in results['ridges']:
        r = results['ridges']['left']
        metrics_text.append(f"\nLeft Ridge:")
        metrics_text.append(f"  Ridge Height (HAR): {r['ridge_height_ft']:.2f} ft")
        metrics_text.append(f"  Superelevation (β): {r['beta']:.3f}")
        metrics_text.append(f"  Ridge Slope (SAR): {r['ridge_slope']:.4f}")
        metrics_text.append(f"  Gradient Advantage (γ): {r['gamma']:.3f}")
    
    if 'right' in results['ridges']:
        r = results['ridges']['right']
        metrics_text.append(f"\nRight Ridge:")
        metrics_text.append(f"  Ridge Height (HAR): {r['ridge_height_ft']:.2f} ft")
        metrics_text.append(f"  Superelevation (β): {r['beta']:.3f}")
        metrics_text.append(f"  Ridge Slope (SAR): {r['ridge_slope']:.4f}")
        metrics_text.append(f"  Gradient Advantage (γ): {r['gamma']:.3f}")
    
    if 'beta_mean' in results:
        metrics_text.append(f"\nMean Values:")
        metrics_text.append(f"  β_mean: {results['beta_mean']:.3f}")
        metrics_text.append(f"  γ_mean: {results['gamma_mean']:.3f}")
    
    metrics_text.append(f"\nChannel Slope (SM): {results['channel_slope']:.6f}")
    
    ax.text(0.02, 0.98, '\n'.join(metrics_text),
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Distance Along Cross-Section (m)', fontsize=12)
    ax.set_ylabel('Elevation (ft, NAVD88)', fontsize=12)
    ax.set_title(f'Nooksack River Cross-Section Analysis - Measurement #{meas_num}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_crosssections(csv_path: str, save_path: str = None, show: bool = True):
    """
    Plot all cross-sections together, colored by time, to help pick which one to analyze.
    
    Args:
        csv_path: Path to CSV file with all cross-sections
        save_path: Path to save the plot (optional)
        show: Whether to display the plot interactively
    """
    # Load all data
    df_all = pd.read_csv(csv_path)
    
    if 'meas_num' not in df_all.columns:
        print("Error: CSV file must contain 'meas_num' column")
        return
    
    if 'meas_date' not in df_all.columns:
        print("Error: CSV file must contain 'meas_date' column for time-based coloring")
        return
    
    # Get unique measurements with dates
    measurements = sorted(df_all['meas_num'].unique())
    
    # Parse dates
    dates = []
    for meas in measurements:
        date_str = df_all[df_all['meas_num'] == meas]['meas_date'].iloc[0]
        try:
            date = pd.to_datetime(date_str)
            dates.append(date)
        except:
            dates.append(None)
    
    # Filter out measurements without valid dates
    valid_measurements = [m for m, d in zip(measurements, dates) if d is not None]
    valid_dates = [d for d in dates if d is not None]
    
    if len(valid_measurements) == 0:
        print("Error: No valid dates found in data")
        return
    
    # Normalize dates for colormap
    date_norm = Normalize(vmin=min(valid_dates).timestamp(), vmax=max(valid_dates).timestamp())
    # Use modern colormap access
    try:
        cmap = cm.colormaps.get_cmap('viridis')  # Matplotlib 3.5+
    except (AttributeError, KeyError):
        # Fallback: create colormap directly using plt
        cmap = plt.cm.viridis  # Works on older versions
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot each cross-section
    for meas, date in zip(valid_measurements, valid_dates):
        # Load this specific cross-section
        df = load_crosssection(csv_path, meas_num=meas)
        
        if len(df) == 0:
            continue
        
        # Get color based on date
        color = cmap(date_norm(date.timestamp()))
        
        # Plot with alpha to see overlapping lines
        ax.plot(df['station_m'], df['elev_navd88_ft'], 
               color=color, linewidth=1.5, alpha=0.7,
               label=f'#{meas} ({date.strftime("%Y-%m-%d")})')
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=date_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Date', fontsize=12)
    
    # Format colorbar dates
    date_format = '%Y-%m-%d'
    date_ticks = pd.date_range(start=min(valid_dates), end=max(valid_dates), periods=8)
    cbar.set_ticks([d.timestamp() for d in date_ticks])
    cbar.set_ticklabels([d.strftime(date_format) for d in date_ticks])
    
    ax.set_xlabel('Distance Along Cross-Section (m)', fontsize=12)
    ax.set_ylabel('Elevation (ft, NAVD88)', fontsize=12)
    ax.set_title('All Nooksack River Cross-Sections Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend but limit entries to avoid overcrowding
    # Only show every nth entry if there are many measurements
    if len(valid_measurements) > 20:
        # Show first, middle, and last few
        n_show = 10
        step = max(1, len(valid_measurements) // n_show)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[i] for i in range(0, len(handles), step)], 
                 [labels[i] for i in range(0, len(labels), step)],
                 loc='best', fontsize=8, ncol=2)
        ax.text(0.02, 0.02, f'Showing {len(valid_measurements)} cross-sections.\nUse --list to see all measurement numbers.',
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def list_measurements(csv_path: str):
    """List all available measurement numbers in the CSV file."""
    df = pd.read_csv(csv_path)
    if 'meas_num' not in df.columns:
        print("No 'meas_num' column found in CSV file")
        return
    
    measurements = sorted(df['meas_num'].unique())
    print(f"\nAvailable measurement numbers in {csv_path}:")
    print(f"Total: {len(measurements)} cross-sections\n")
    
    for i, meas in enumerate(measurements, 1):
        count = len(df[df['meas_num'] == meas])
        date = df[df['meas_num'] == meas]['meas_date'].iloc[0] if 'meas_date' in df.columns else 'N/A'
        print(f"  {i:2d}. Measurement #{meas:3d} - {count:3d} points (Date: {date})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Nooksack River cross-section and calculate metrics'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        default='nooksack_cross_sections_north_cedarville_12210700.csv',
        nargs='?',
        help='Path to cross-section CSV file'
    )
    parser.add_argument(
        '--meas-num',
        type=int,
        default=None,
        help='Measurement number to analyze. If not specified, analyzes first measurement found. Use --list to see available measurements.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available measurement numbers and exit'
    )
    parser.add_argument(
        '--plot-all',
        action='store_true',
        help='Plot all cross-sections together colored by time to help pick which one to analyze'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all measurements and save separate plots for each'
    )
    parser.add_argument(
        '--longitudinal-slope',
        type=float,
        default=None,
        help='Longitudinal channel slope (for calculating SM). If not provided, will estimate from cross-section.'
    )
    parser.add_argument(
        '--channel-depth-method',
        type=str,
        choices=['bankfull', 'min_max', 'percentile'],
        default='bankfull',
        help='Method for estimating channel depth'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for plot (if not specified, will show interactively). If --all is used, this becomes the output directory prefix.'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plot interactively'
    )
    
    args = parser.parse_args()
    
    # List measurements if requested
    if args.list:
        list_measurements(args.csv_path)
        return
    
    # Plot all cross-sections together if requested
    if args.plot_all:
        output_path = args.output or 'all_nooksack_crosssections.png'
        print("Plotting all cross-sections colored by time...")
        plot_all_crosssections(args.csv_path, save_path=output_path, show=not args.no_show)
        print("\nUse --meas-num <number> to analyze a specific cross-section")
        return
    
    # Load all measurements to get list
    df_all = pd.read_csv(args.csv_path)
    if 'meas_num' not in df_all.columns:
        print("Error: CSV file must contain 'meas_num' column")
        return
    
    measurements = sorted(df_all['meas_num'].unique())
    
    # Analyze all measurements if requested
    if args.all:
        print(f"Analyzing {len(measurements)} cross-sections...\n")
        output_prefix = args.output or 'nooksack_crosssection'
        
        for i, meas_num in enumerate(measurements, 1):
            print(f"Processing measurement #{meas_num} ({i}/{len(measurements)})...")
            try:
                results = analyze_crosssection(
                    args.csv_path,
                    meas_num=meas_num,
                    longitudinal_slope=args.longitudinal_slope,
                    channel_depth_method=args.channel_depth_method
                )
                
                # Create output filename
                if args.output:
                    # If output is a directory or has extension, handle it
                    if output_prefix.endswith('.png') or output_prefix.endswith('.pdf'):
                        base = output_prefix.rsplit('.', 1)[0]
                        ext = output_prefix.rsplit('.', 1)[1]
                        output_file = f"{base}_meas{meas_num}.{ext}"
                    else:
                        output_file = f"{output_prefix}_meas{meas_num}.png"
                else:
                    output_file = f"{output_prefix}_meas{meas_num}.png"
                
                plot_crosssection(results, save_path=output_file, show=False)
                
                print(f"  ✓ Saved: {output_file}")
                print(f"    β_mean: {results.get('beta_mean', 'N/A'):.3f}, "
                      f"γ_mean: {results.get('gamma_mean', 'N/A'):.3f}")
                print()
            except Exception as e:
                print(f"  ✗ Error processing measurement #{meas_num}: {e}\n")
        
        print(f"Completed analysis of {len(measurements)} cross-sections")
        return
    
    # Analyze single cross-section
    if args.meas_num is None:
        # Use first measurement if none specified
        args.meas_num = measurements[0]
        print(f"No measurement number specified. Using first measurement: #{args.meas_num}")
        print(f"(Use --list to see all available measurements)\n")
    
    # Analyze cross-section
    print(f"Analyzing measurement #{args.meas_num}...")
    results = analyze_crosssection(
        args.csv_path,
        meas_num=args.meas_num,
        longitudinal_slope=args.longitudinal_slope,
        channel_depth_method=args.channel_depth_method
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"CROSS-SECTION ANALYSIS RESULTS - Measurement #{results.get('meas_num', 'Unknown')}")
    print("="*60)
    print(f"\nChannel Bed:")
    print(f"  Station: {results['channel_bed']['station_m']:.2f} m")
    print(f"  Elevation: {results['channel_bed']['elevation_ft']:.2f} ft")
    print(f"\nChannel Depth (HM): {results['channel_depth_ft']:.2f} ft")
    print(f"Channel Slope (SM): {results['channel_slope']:.6f}")
    
    if 'left' in results['ridges']:
        r = results['ridges']['left']
        print(f"\nLeft Ridge:")
        print(f"  Station: {r['station_m']:.2f} m")
        print(f"  Elevation: {r['elevation_ft']:.2f} ft")
        print(f"  Ridge Height (HAR): {r['ridge_height_ft']:.2f} ft")
        print(f"  Superelevation (β) = HAR/HM: {r['beta']:.3f}")
        print(f"  Ridge Slope (SAR): {r['ridge_slope']:.6f}")
        print(f"  Gradient Advantage (γ) = SAR/SM: {r['gamma']:.3f}")
    
    if 'right' in results['ridges']:
        r = results['ridges']['right']
        print(f"\nRight Ridge:")
        print(f"  Station: {r['station_m']:.2f} m")
        print(f"  Elevation: {r['elevation_ft']:.2f} ft")
        print(f"  Ridge Height (HAR): {r['ridge_height_ft']:.2f} ft")
        print(f"  Superelevation (β) = HAR/HM: {r['beta']:.3f}")
        print(f"  Ridge Slope (SAR): {r['ridge_slope']:.6f}")
        print(f"  Gradient Advantage (γ) = SAR/SM: {r['gamma']:.3f}")
    
    if 'beta_mean' in results:
        print(f"\nMean Values:")
        print(f"  β_mean: {results['beta_mean']:.3f}")
        print(f"  γ_mean: {results['gamma_mean']:.3f}")
    
    print("\n" + "="*60)
    
    # Plot results
    plot_crosssection(results, save_path=args.output, show=not args.no_show)


if __name__ == '__main__':
    main()
