#!/usr/bin/env python3
"""
Plot D50 vs River Mile from the calculated D50 along centerline data.

Usage:
    poetry run python plot_d50_vs_rivermile.py
    poetry run python plot_d50_vs_rivermile.py --input data/nooksack_d50_along_centerline.csv --output data/d50_vs_rivermile.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_d50_vs_rivermile(
    input_path: str,
    output_path: str = None,
    units: str = 'inches',
    show_regression: bool = True
) -> None:
    """
    Plot D50 vs River Mile.
    
    Args:
        input_path: Path to CSV file with D50 data
        output_path: Path to save the plot (if None, displays plot)
        units: 'inches' or 'mm' for D50 units
        show_regression: Whether to show the regression line
    """
    # Read data
    df = pd.read_csv(input_path)
    
    # Select units
    if units == 'mm':
        d50_col = 'd50_mm'
        ylabel = 'D₅₀ (mm)'
    else:
        d50_col = 'd50_inches'
        ylabel = 'D₅₀ (inches)'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Plot data points
    ax.plot(df['river_mile'], df[d50_col], 
            'b-', linewidth=1.5, alpha=0.7, label='D₅₀ along centerline')
    
    # Add scatter points for better visibility
    ax.scatter(df['river_mile'], df[d50_col], 
              s=10, alpha=0.5, color='blue', zorder=3)
    
    # Show regression line if requested
    if show_regression:
        # Calculate regression: D₅₀ = 0.038(RM) - 0.07
        river_mile_range = np.linspace(df['river_mile'].min(), df['river_mile'].max(), 100)
        if units == 'mm':
            d50_regression = (0.038 * river_mile_range - 0.07) * 25.4
            d50_regression = np.maximum(d50_regression, 0)  # Clamp negative values
        else:
            d50_regression = np.maximum(0.038 * river_mile_range - 0.07, 0)
        
        ax.plot(river_mile_range, d50_regression, 
               'r--', linewidth=2, alpha=0.8, 
               label='Regression: D₅₀ = 0.038(RM) - 0.07')
    
    # Styling
    ax.set_xlabel('River Mile (from upstream)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title('D₅₀ (Median Grain Size) vs River Mile\nNooksack River', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Set axis limits with some padding
    ax.set_xlim(df['river_mile'].min() - 0.5, df['river_mile'].max() + 0.5)
    y_min = df[d50_col].min()
    y_max = df[d50_col].max()
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
    
    # Formatting
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot D50 vs River Mile from calculated D50 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default input file and display plot
  python plot_d50_vs_rivermile.py
  
  # Specify input and output files
  python plot_d50_vs_rivermile.py --input data/nooksack_d50_along_centerline.csv --output data/d50_plot.png
  
  # Plot in millimeters
  python plot_d50_vs_rivermile.py --units mm --output data/d50_plot_mm.png
  
  # Hide regression line
  python plot_d50_vs_rivermile.py --no-regression
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/nooksack_d50_along_centerline.csv',
        help='Path to input CSV file with D50 data (default: data/nooksack_d50_along_centerline.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the plot (if not specified, displays plot)'
    )
    
    parser.add_argument(
        '--units',
        type=str,
        choices=['inches', 'mm'],
        default='inches',
        help='Units for D50: inches or mm (default: inches)'
    )
    
    parser.add_argument(
        '--no-regression',
        action='store_true',
        help='Hide the regression line'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    try:
        plot_d50_vs_rivermile(
            input_path=str(input_path),
            output_path=args.output,
            units=args.units,
            show_regression=not args.no_regression
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

