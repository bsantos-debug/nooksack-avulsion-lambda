#!/usr/bin/env python3
"""
Calculate D50 (median grain size) along a river centerline using the Nooksack River regression.

The regression equation is: D₅₀ = 0.038(RM) - 0.07
where:
- D₅₀ is the median grain size in inches
- RM is the river mile (distance from upstream end in miles)

The centerline starts at river mile 0 (upstream) and ends at river mile x (downstream).

Usage:
    poetry run python calculate_d50_along_centerline.py --centerline centerline.shp --output d50_results.csv
    poetry run python calculate_d50_along_centerline.py --centerline centerline.shp --spacing 100 --output d50_results.geojson
"""

import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


def calculate_d50_from_river_mile(river_mile: float) -> float:
    """
    Calculate D50 (median grain size) from river mile using the Nooksack River regression.
    
    Regression: D₅₀ = 0.038(RM) - 0.07
    where D₅₀ is in inches and RM is river mile.
    
    Args:
        river_mile: River mile (distance from upstream end in miles)
        
    Returns:
        D50 in inches
    """
    d50_inches = 0.038 * river_mile - 0.07
    # Ensure non-negative values (though regression may produce negative for very small RM)
    return max(0.0, d50_inches)


def sample_points_along_centerline(
    centerline: LineString,
    spacing: float = 50.0,
    start_river_mile: float = 0.0,
    crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Sample points along a centerline and calculate river miles and D50.
    
    Args:
        centerline: Shapely LineString representing the river centerline
        spacing: Distance between sample points in meters
        start_river_mile: Starting river mile (default 0.0 for upstream end)
        
    Returns:
        GeoDataFrame with sampled points, river miles, and D50 values
    """
    # Get the total length of the centerline in meters
    total_length_m = centerline.length
    
    # Convert spacing from meters to miles for river mile calculation
    # 1 mile = 1609.34 meters
    meters_to_miles = 1.0 / 1609.34
    
    # Sample points along the centerline
    points = []
    distances_along = []
    river_miles = []
    d50_values_inches = []
    d50_values_mm = []
    
    # Sample from upstream (start) to downstream (end)
    # Start at 0 and go to total_length
    current_distance = 0.0
    
    while current_distance <= total_length_m:
        # Interpolate point along the centerline
        point = centerline.interpolate(current_distance)
        
        # Calculate river mile (distance from upstream end in miles)
        river_mile = start_river_mile + (current_distance * meters_to_miles)
        
        # Calculate D50 using the regression
        d50_inches = calculate_d50_from_river_mile(river_mile)
        d50_mm = d50_inches * 25.4  # Convert inches to millimeters
        
        points.append(point)
        distances_along.append(current_distance)
        river_miles.append(river_mile)
        d50_values_inches.append(d50_inches)
        d50_values_mm.append(d50_mm)
        
        current_distance += spacing
    
    # Also include the endpoint
    if current_distance - spacing < total_length_m:
        point = centerline.interpolate(total_length_m)
        river_mile = start_river_mile + (total_length_m * meters_to_miles)
        d50_inches = calculate_d50_from_river_mile(river_mile)
        d50_mm = d50_inches * 25.4
        
        points.append(point)
        distances_along.append(total_length_m)
        river_miles.append(river_mile)
        d50_values_inches.append(d50_inches)
        d50_values_mm.append(d50_mm)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            'dist_along_m': distances_along,
            'river_mile': river_miles,
            'd50_inches': d50_values_inches,
            'd50_mm': d50_values_mm,
        },
        geometry=points,
        crs=crs  # Use provided CRS or None
    )
    
    return gdf


def process_centerline(
    centerline_path: str,
    output_path: str,
    spacing: float = 50.0,
    start_river_mile: float = 0.0,
    crs: Optional[str] = None
) -> None:
    """
    Process a centerline file and calculate D50 along it.
    
    Args:
        centerline_path: Path to centerline file (shapefile, GeoJSON, etc.)
        output_path: Path to output file (CSV or GeoJSON)
        spacing: Distance between sample points in meters
        start_river_mile: Starting river mile (default 0.0)
        crs: Optional CRS to use (if not specified, will use CRS from input file)
    """
    # Read centerline
    centerline_gdf = gpd.read_file(centerline_path)
    
    if centerline_gdf.empty:
        raise ValueError(f"No features found in {centerline_path}")
    
    # Get the first LineString geometry (or merge multiple if needed)
    if len(centerline_gdf) > 1:
        print(f"Warning: Multiple features found. Using the first one.")
    
    centerline_geom = centerline_gdf.geometry.iloc[0]
    
    # Ensure it's a LineString
    if not isinstance(centerline_geom, LineString):
        raise ValueError(f"Expected LineString geometry, got {type(centerline_geom)}")
    
    # Get CRS from input or use provided
    if crs:
        centerline_gdf = centerline_gdf.to_crs(crs)
        centerline_geom = centerline_gdf.geometry.iloc[0]
    elif centerline_gdf.crs is None:
        print("Warning: No CRS found. Using EPSG:4326. Consider specifying --crs.")
        centerline_gdf = centerline_gdf.set_crs('EPSG:4326')
        centerline_geom = centerline_gdf.geometry.iloc[0]
    
    # Calculate total length and end river mile
    total_length_m = centerline_geom.length
    meters_to_miles = 1.0 / 1609.34
    end_river_mile = start_river_mile + (total_length_m * meters_to_miles)
    
    print(f"Centerline length: {total_length_m:.2f} m ({total_length_m * meters_to_miles:.2f} miles)")
    print(f"River mile range: {start_river_mile:.2f} to {end_river_mile:.2f}")
    print(f"Sampling spacing: {spacing} m")
    
    # Sample points and calculate D50
    result_gdf = sample_points_along_centerline(
        centerline_geom,
        spacing=spacing,
        start_river_mile=start_river_mile,
        crs=centerline_gdf.crs
    )
    
    # Save output
    output_path_obj = Path(output_path)
    output_ext = output_path_obj.suffix.lower()
    
    if output_ext == '.csv':
        # Save as CSV (without geometry)
        result_df = pd.DataFrame(result_gdf.drop(columns='geometry'))
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path} (CSV)")
    elif output_ext in ['.geojson', '.json']:
        result_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Results saved to {output_path} (GeoJSON)")
    elif output_ext == '.shp':
        result_gdf.to_file(output_path)
        print(f"Results saved to {output_path} (Shapefile)")
    else:
        # Default to CSV
        result_df = pd.DataFrame(result_gdf.drop(columns='geometry'))
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path} (CSV, default format)")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  Number of points: {len(result_gdf)}")
    print(f"  D50 range (inches): {result_gdf['d50_inches'].min():.3f} - {result_gdf['d50_inches'].max():.3f}")
    print(f"  D50 range (mm): {result_gdf['d50_mm'].min():.2f} - {result_gdf['d50_mm'].max():.2f}")
    print(f"  Mean D50: {result_gdf['d50_inches'].mean():.3f} inches ({result_gdf['d50_mm'].mean():.2f} mm)")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate D50 along a river centerline using the Nooksack River regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default spacing (50m)
  python calculate_d50_along_centerline.py --centerline centerline.shp --output d50_results.csv
  
  # Custom spacing (100m intervals)
  python calculate_d50_along_centerline.py --centerline centerline.shp --spacing 100 --output d50_results.csv
  
  # Output as GeoJSON
  python calculate_d50_along_centerline.py --centerline centerline.shp --output d50_results.geojson
  
  # Start at a specific river mile
  python calculate_d50_along_centerline.py --centerline centerline.shp --start-mile 5.0 --output d50_results.csv
        """
    )
    
    parser.add_argument(
        '--centerline',
        type=str,
        required=True,
        help='Path to centerline file (shapefile, GeoJSON, etc.)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output file (CSV, GeoJSON, or shapefile)'
    )
    
    parser.add_argument(
        '--spacing',
        type=float,
        default=50.0,
        help='Distance between sample points in meters (default: 50.0)'
    )
    
    parser.add_argument(
        '--start-mile',
        type=float,
        default=0.0,
        help='Starting river mile at upstream end (default: 0.0)'
    )
    
    parser.add_argument(
        '--crs',
        type=str,
        default=None,
        help='CRS to use (e.g., EPSG:3857). If not specified, uses CRS from input file.'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    centerline_path = Path(args.centerline)
    if not centerline_path.exists():
        print(f"Error: Centerline file not found: {centerline_path}")
        return 1
    
    if args.spacing <= 0:
        print(f"Error: Spacing must be positive, got {args.spacing}")
        return 1
    
    try:
        process_centerline(
            centerline_path=str(centerline_path),
            output_path=args.output,
            spacing=args.spacing,
            start_river_mile=args.start_mile,
            crs=args.crs
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

