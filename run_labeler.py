#!/usr/bin/env python3
"""
Script to run the cross-section labeler on generated cross sections.

Usage:
    poetry run python run_labeler.py
    poetry run python run_labeler.py --node-ids 1 2 3  # label specific nodes only
    poetry run python run_labeler.py --no-skip  # re-label even if labels exist
"""

# === USER CONFIGURATION: Change these defaults to match your setup ===
DEFAULT_CROSS_SECTIONS = "cross_sections.shp"
DEFAULT_PROFILES_DIR = "cross_section_profiles"
DEFAULT_RIVER_NAME = "Nooksack"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_DEM_PATH = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
# === END USER CONFIGURATION ===

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from avulsionprecursors.gui.config import GUIConfig
from avulsionprecursors.pipeline.labeling import FileLabelingPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Label cross-sections using the interactive GUI"
    )
    parser.add_argument(
        "--cross-sections",
        type=str,
        default=DEFAULT_CROSS_SECTIONS,
        help=f"Path to cross-sections shapefile (default: {DEFAULT_CROSS_SECTIONS})",
    )
    parser.add_argument(
        "--profiles-dir",
        type=str,
        default=DEFAULT_PROFILES_DIR,
        help=f"Directory containing cross-section profile CSVs (default: {DEFAULT_PROFILES_DIR})",
    )
    parser.add_argument(
        "--river-name",
        type=str,
        default=DEFAULT_RIVER_NAME,
        help=f"Name of the river (default: {DEFAULT_RIVER_NAME})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for labels (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--node-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific node IDs to label (default: all nodes)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-label cross-sections even if labels already exist",
    )
    parser.add_argument(
        "--dem-path",
        type=str,
        default=DEFAULT_DEM_PATH,
        help=f"Path to DEM raster file (for background display). Default: {DEFAULT_DEM_PATH}",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Labels to pick in order (default: channel ridge1 floodplain1 ridge2 floodplain2). "
            "Example: --labels channel ridge1 ridge2"
        ),
    )
    parser.add_argument(
        "--no-satellite",
        action="store_true",
        help="Disable satellite imagery overlay",
    )
    parser.add_argument(
        "--no-dem",
        action="store_true",
        help="Disable DEM background layer",
    )
    
    args = parser.parse_args()

    def proj_path(s: str) -> Path:
        p = Path(s)
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    # Validate inputs (relative paths resolved from repo root)
    cross_sections_path = proj_path(args.cross_sections)
    profiles_dir = proj_path(args.profiles_dir)
    output_dir = proj_path(args.output_dir)
    
    if not cross_sections_path.exists():
        print(f"❌ Error: Cross-sections file not found: {cross_sections_path}")
        return 1
    
    if not profiles_dir.exists():
        print(f"❌ Error: Profiles directory not found: {profiles_dir}")
        return 1
    
    print(f"📁 Cross-sections: {cross_sections_path}")
    print(f"📁 Profiles directory: {profiles_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏞️  River name: {args.river_name}")
    if args.node_ids:
        print(f"🎯 Node IDs to label: {args.node_ids}")
    else:
        print(f"🎯 Labeling all nodes")
    print(f"⏭️  Skip existing: {not args.no_skip}")
    pick_labels = list(args.labels) if args.labels else GUIConfig().labels
    print(f"📍 Pick order: {' → '.join(pick_labels)}")
    print()
    
    # Try to find DEM path
    dem_path = DEFAULT_DEM_PATH
    if args.dem_path:
        # Use provided DEM path (either from --dem-path or from DEFAULT_DEM_PATH)
        dem_path = proj_path(args.dem_path)
        if not dem_path.exists():
            print(f"⚠️  Warning: DEM file not found: {dem_path}")
            print("   Continuing without DEM display...")
            dem_path = None
        else:
            if args.dem_path == DEFAULT_DEM_PATH:
                print(f"🗻 DEM: {dem_path} (using default from script)")
            else:
                print(f"🗻 DEM: {dem_path}")
    
    print()
    
    # Create pipeline
    try:
        # Show DEM status
        show_dem = not args.no_dem and dem_path is not None
        if show_dem:
            print(f"✅ DEM display: Enabled")
        else:
            print(f"⏭️  DEM display: Disabled")
        print(f"✅ Satellite display: {'Enabled' if not args.no_satellite else 'Disabled'}")
        print()
        
        gui_config = GUIConfig(labels=pick_labels) if args.labels else GUIConfig()
        pipeline = FileLabelingPipeline(
            cross_sections_path=cross_sections_path,
            profiles_dir=profiles_dir,
            river_name=args.river_name,
            output_dir=output_dir,
            gui_config=gui_config,
            dem_path=dem_path,
            show_satellite=not args.no_satellite,
            show_dem=show_dem,
        )
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return 1
    
    # Run labeling
    try:
        pipeline.run(
            node_ids=args.node_ids,
            skip_existing=not args.no_skip,
        )
        print("\n✅ Labeling complete!")
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Labeling interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during labeling: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
