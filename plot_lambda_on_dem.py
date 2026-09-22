#!/usr/bin/env python3
"""
Plot lambda values on satellite imagery (or DEM) over the centerline.

Usage:
    poetry run python plot_lambda_on_dem.py
    poetry run python plot_lambda_on_dem.py --basemap satellite
    poetry run python plot_lambda_on_dem.py --basemap dem
    poetry run python plot_lambda_on_dem.py --lambda-csv data/Nooksack_lambda_results.csv
"""

import sys
from pathlib import Path

# Setup path for imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import rasterio
from rasterio.warp import transform_bounds
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from typing import Optional
import warnings

# === USER CONFIGURATION ===
DEFAULT_DEM_PATH = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
DEFAULT_CENTERLINE_PATH = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/Nooksack_data/GIS/Centerline_flowaccumulation/Centerline_flowaccumulation.shp"
DEFAULT_POINTS_PATH = "centerline_points_from_xs.shp"
DEFAULT_LAMBDA_CSV = "data/Nooksack_lambda_results.csv"
DEFAULT_OUTPUT = "data/Nooksack_lambda_on_satellite.png"

def load_lambda_results(lambda_csv: Path) -> pd.DataFrame:
    """Load lambda results from CSV."""
    if not lambda_csv.exists():
        raise FileNotFoundError(f"Lambda CSV not found: {lambda_csv}")

    df = pd.read_csv(lambda_csv)

    if "node_id" not in df.columns:
        raise ValueError("Lambda CSV must contain 'node_id' column")
    if "lambda" not in df.columns:
        raise ValueError("Lambda CSV must contain 'lambda' column")

    df = df[df["lambda"].notna() & (df["lambda"] > 0)]
    return df


def load_centerline(centerline_path: Path, points_path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load centerline points for node locations."""
    if not centerline_path.exists():
        raise FileNotFoundError(f"Centerline shapefile not found: {centerline_path}")

    if points_path and points_path.exists():
        points_gdf = gpd.read_file(points_path)
        if "node_id" in points_gdf.columns:
            return points_gdf
        print("⚠️  Warning: Points shapefile doesn't have 'node_id', using centerline only")

    centerline_gdf = gpd.read_file(centerline_path)
    # If LineString, densify to points
    geom = centerline_gdf.geometry.iloc[0]
    if geom.geom_type == "LineString":
        distances = np.linspace(0, geom.length, 100)
        points = [geom.interpolate(d) for d in distances]
        gdf = gpd.GeoDataFrame(
            {"node_id": range(1, len(points) + 1)},
            geometry=points,
            crs=centerline_gdf.crs,
        )
        return gdf
    return centerline_gdf


def plot_lambda_on_dem(
    lambda_df: pd.DataFrame,
    centerline_gdf: gpd.GeoDataFrame,
    dem_path: Path,
    centerline_path: Path,
    output_path: Optional[Path] = None,
    river_name: str = "River",
    basemap: str = "satellite",
    zoom_level: int = 12,
) -> None:
    """
    Plot lambda values over satellite imagery or DEM along the centerline.
    """
    basemap = str(basemap).lower()
    # Load the original centerline as a LineString for full extent
    centerline_line_gdf = gpd.read_file(centerline_path)
    if centerline_line_gdf.crs is None:
        centerline_line_gdf.set_crs('EPSG:4326', inplace=True)
    if centerline_line_gdf.crs != 'EPSG:4326':
        centerline_line_gdf = centerline_line_gdf.to_crs('EPSG:4326')
    
    # Get the LineString geometry for plotting the full centerline
    centerline_line = centerline_line_gdf.geometry.iloc[0] if centerline_line_gdf.geometry.iloc[0].geom_type == 'LineString' else None
    
    # Merge lambda values with centerline points (for colored points)
    if 'node_id' not in centerline_gdf.columns:
        # If no node_id, try to match by geometry or create sequential node_ids
        print("⚠️  Warning: Centerline doesn't have 'node_id', creating sequential IDs")
        centerline_gdf['node_id'] = range(1, len(centerline_gdf) + 1)
    
    # Join lambda values to centerline (left join to keep all centerline points, but only plot those with lambda)
    merged_gdf = centerline_gdf.merge(
        lambda_df[['node_id', 'lambda']],
        on='node_id',
        how='left'  # Keep all centerline points
    )
    
    # Filter to only nodes with lambda values (for colored points)
    lambda_points_gdf = merged_gdf[merged_gdf['lambda'].notna()].copy()
    
    if len(lambda_points_gdf) == 0:
        raise ValueError("No matching node_ids between lambda results and centerline")
    
    # Ensure CRS is set
    if merged_gdf.crs is None:
        merged_gdf.set_crs('EPSG:4326', inplace=True)
    
    # Reproject to WGS84 for display
    if merged_gdf.crs != 'EPSG:4326':
        merged_gdf = merged_gdf.to_crs('EPSG:4326')
        lambda_points_gdf = lambda_points_gdf.to_crs('EPSG:4326')
        centerline_line_gdf = centerline_line_gdf.to_crs('EPSG:4326')
    
    # Get extent from full centerline (not just lambda points)
    if centerline_line:
        bounds = centerline_line.bounds  # [minx, miny, maxx, maxy]
    else:
        bounds = merged_gdf.total_bounds  # [minx, miny, maxx, maxy]
    margin = 0.01  # 1% margin
    lon_margin = (bounds[2] - bounds[0]) * margin
    lat_margin = (bounds[3] - bounds[1]) * margin
    
    extent = [
        bounds[0] - lon_margin,  # west
        bounds[2] + lon_margin,  # east
        bounds[1] - lat_margin,  # south
        bounds[3] + lat_margin   # north
    ]
    
    # Create figure with Cartopy projection
    fig = plt.figure(figsize=(14, 10))
    if basemap == "satellite":
        tiles = cimgt.GoogleTiles(style="satellite")
        ax = plt.axes(projection=tiles.crs)
        print(f"🛰️  Adding satellite basemap (zoom={zoom_level})...")
        try:
            ax.add_image(tiles, int(zoom_level))
        except Exception as e:
            print(f"⚠️  Warning: Could not load Google satellite tiles: {e}")
            try:
                tiles = cimgt.QuadtreeTiles()
                ax = plt.axes(projection=tiles.crs)
                ax.add_image(tiles, int(zoom_level))
                print("   Fell back to QuadtreeTiles")
            except Exception as e2:
                print(f"⚠️  Satellite unavailable ({e2}); continuing without basemap")
                ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Load and plot DEM
        print(f"🗻 Loading DEM: {dem_path}")
        try:
            with rasterio.open(dem_path) as dem_src:
                dem_bounds = dem_src.bounds
                dem_crs = dem_src.crs
                
                # Transform extent from WGS84 to DEM CRS
                bounds_dem_crs = transform_bounds(
                    'EPSG:4326',
                    str(dem_crs),
                    extent[0],  # west
                    extent[2],  # south
                    extent[1],  # east
                    extent[3]   # north
                )
                
                # Clip bounds to DEM extent
                west = max(bounds_dem_crs[0], dem_bounds.left)
                east = min(bounds_dem_crs[2], dem_bounds.right)
                south = max(bounds_dem_crs[1], dem_bounds.bottom)
                north = min(bounds_dem_crs[3], dem_bounds.top)
                
                if west >= east or south >= north:
                    print("⚠️  Warning: Extent outside DEM bounds, using full DEM")
                    west, south, east, north = dem_bounds
                
                # Read DEM data
                window = rasterio.windows.from_bounds(
                    west, south, east, north, dem_src.transform
                )
                
                dem_data = dem_src.read(1, window=window, masked=True)
                transform = rasterio.windows.transform(window, dem_src.transform)
                
                # Create coordinate grids in DEM CRS
                height, width = dem_data.shape
                x = np.linspace(west, east, width)
                y = np.linspace(north, south, height)  # Note: y is flipped
                X, Y = np.meshgrid(x, y)
                
                # Downsample for faster display
                max_pixels = 1000
                if height * width > max_pixels * max_pixels:
                    downsample_factor = int(np.ceil(np.sqrt(height * width / (max_pixels * max_pixels))))
                    dem_data = dem_data[::downsample_factor, ::downsample_factor]
                    X = X[::downsample_factor, ::downsample_factor]
                    Y = Y[::downsample_factor, ::downsample_factor]
                
                # Transform to WGS84
                x_flat = X.flatten()
                y_flat = Y.flatten()
                lon_flat, lat_flat = rasterio.warp.transform(
                    str(dem_crs), 'EPSG:4326',
                    x_flat, y_flat
                )
                lon_coords = np.array(lon_flat).reshape(X.shape)
                lat_coords = np.array(lat_flat).reshape(Y.shape)
                
                # Mask invalid values
                dem_data = np.ma.masked_invalid(dem_data)
                
                # Plot DEM
                ax.pcolormesh(
                    lon_coords, lat_coords, dem_data,
                    cmap='terrain',
                    alpha=0.7,
                    transform=ccrs.PlateCarree(),
                    shading='auto',
                    vmin=np.nanpercentile(dem_data, 2),
                    vmax=np.nanpercentile(dem_data, 98)
                )
                
                print(f"✅ DEM plotted ({height}x{width} pixels)")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not load DEM: {e}")
            print("   Continuing without DEM background...")
    
    # Plot full centerline as a line
    print(f"📊 Plotting centerline with lambda values...")
    if centerline_line:
        # Extract coordinates from LineString
        if hasattr(centerline_line, 'coords'):
            coords = list(centerline_line.coords)
        else:
            coords = [(centerline_line.x, centerline_line.y)]  # Fallback for Point
        
        if len(coords) > 1:
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            ax.plot(
                x_coords, y_coords,
                'k-', linewidth=2, alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=5,
                label='Centerline'
            )

    # Calculate color normalization (use log scale for lambda)
    lambda_values = lambda_points_gdf['lambda'].values
    lambda_min = np.nanpercentile(lambda_values, 2)
    lambda_max = np.nanpercentile(lambda_values, 98)
    
    # Ensure minimum is positive for log scale
    lambda_min = max(lambda_min, 0.01)
    
    # Use LogNorm for lambda values
    norm = LogNorm(vmin=lambda_min, vmax=lambda_max)
    
    # Create a colormap with more color segments for smoother gradient
    colors = [
        '#1a0033',  # Dark purple
        '#330066',  # Purple
        '#0000cc',  # Dark blue
        '#0066ff',  # Blue
        '#00ccff',  # Cyan
        '#00ff99',  # Green-cyan
        '#66ff00',  # Yellow-green
        '#ffff00',  # Yellow
        '#ff9900',  # Orange
        '#ff6600',  # Red-orange
        '#ff3300',  # Red
        '#cc0000'   # Dark red
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('lambda_cmap', colors, N=n_bins)
    
    scatter = ax.scatter(
        lambda_points_gdf.geometry.x,
        lambda_points_gdf.geometry.y,
        c=lambda_points_gdf['lambda'],
        cmap=cmap,
        norm=norm,
        s=40,
        alpha=0.9,
        edgecolors='white',
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
        zorder=10,
        label='Lambda values'
    )
    
    cbar = plt.colorbar(
        scatter,
        ax=ax,
        orientation='vertical',
        pad=0.02,
        fraction=0.046,
        label='Avulsion Potential (λ)',
        extend='both'
    )
    cbar.ax.set_ylabel('Avulsion Potential (λ)', fontsize=12, fontweight='bold')
    
    def format_func(value, tick_number):
        if value == 0:
            return '0'
        if value >= 1:
            formatted = f'{value:.2f}'.rstrip('0').rstrip('.')
        elif value >= 0.1:
            formatted = f'{value:.3f}'.rstrip('0').rstrip('.')
        elif value >= 0.01:
            formatted = f'{value:.4f}'.rstrip('0').rstrip('.')
        else:
            formatted = f'{value:.5f}'.rstrip('0').rstrip('.')
        return formatted
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    basemap_label = "Satellite" if basemap == "satellite" else "DEM"
    ax.set_title(
        f'{river_name} - Lambda Values on {basemap_label}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    lambda_stats = f"λ: {lambda_min:.2f} - {lambda_max:.2f} (2nd-98th percentile)"
    ax.text(
        0.02, 0.02,
        lambda_stats,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        verticalalignment='bottom'
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Prefer ``poetry run python plot_results.py map`` (unified entry point)."""
    parser = argparse.ArgumentParser(
        description="Plot lambda values on satellite imagery (or DEM) over centerline"
    )
    parser.add_argument(
        "--lambda-csv",
        type=str,
        default=DEFAULT_LAMBDA_CSV,
        help=f"Path to lambda results CSV (default: {DEFAULT_LAMBDA_CSV})",
    )
    parser.add_argument(
        "--centerline",
        type=str,
        default=DEFAULT_CENTERLINE_PATH,
        help="Path to centerline shapefile"
    )
    parser.add_argument(
        "--points",
        type=str,
        default=DEFAULT_POINTS_PATH,
        help="Path to centerline points shapefile (optional, used if centerline is LineString)"
    )
    parser.add_argument(
        "--dem-path",
        type=str,
        default=DEFAULT_DEM_PATH,
        help="Path to DEM raster file (used with --basemap dem)"
    )
    parser.add_argument(
        "--basemap",
        type=str,
        choices=["satellite", "dem"],
        default="satellite",
        help="Background: satellite imagery (default) or DEM",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=12,
        help="Satellite tile zoom level (default: 12)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the plot (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--river-name",
        type=str,
        default="Nooksack",
        help="Name of the river for title"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    lambda_csv = Path(args.lambda_csv)
    centerline_path = Path(args.centerline)
    points_path = Path(args.points) if args.points else None
    dem_path = Path(args.dem_path)
    output_path = Path(args.output) if args.output else None
    
    # Load data
    print(f"📊 Loading lambda results: {lambda_csv}")
    lambda_df = load_lambda_results(lambda_csv)
    print(f"✅ Loaded {len(lambda_df)} lambda values")
    
    print(f"📍 Loading centerline: {centerline_path}")
    centerline_gdf = load_centerline(centerline_path, points_path)
    print(f"✅ Loaded {len(centerline_gdf)} centerline points")
    
    # Plot
    plot_lambda_on_dem(
        lambda_df=lambda_df,
        centerline_gdf=centerline_gdf,
        dem_path=dem_path,
        centerline_path=centerline_path,
        output_path=output_path,
        river_name=args.river_name,
        basemap=args.basemap,
        zoom_level=args.zoom,
    )


if __name__ == "__main__":
    main()
