#!/usr/bin/env python3
"""Open the interactive cross-section labeler (the picking window).

Two ways to launch:

1) Config-driven (package)::

    python -m avulsionprecursors label -c config/nooksack_labeler_no_levee.yaml

2) Legacy Nooksack working-tree paths (this script's defaults)::

    poetry run python run_labeler.py
    poetry run python run_labeler.py --river-name Nooksack_no_levee \\
        --labels-dir data/labels_no_levee

Archive old picks first so they are not overwritten::

    python studies/nooksack_2022_topobathy/archive_local_v1.py --also-sidecar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === Defaults match the local Nooksack working tree (gitignored) ===
DEFAULT_CROSS_SECTIONS = "cross_sections.shp"
DEFAULT_PROFILES_DIR = "cross_section_profiles"
DEFAULT_RIVER_NAME = "Nooksack"
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_LABELS_DIR = "data/labels"
DEFAULT_DEM_PATH = (
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
)


def _proj(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Delegate to the package CLI when a YAML config is provided.
    if "-c" in argv or "--config" in argv:
        from avulsionprecursors.cli import main as package_main

        return package_main(["label", *argv])

    parser = argparse.ArgumentParser(
        description="Interactive Nooksack cross-section labeler (picking GUI)"
    )
    parser.add_argument("--cross-sections", default=DEFAULT_CROSS_SECTIONS)
    parser.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--river-name",
        default=DEFAULT_RIVER_NAME,
        help="Prefix for label files: <river-name>_node_<id>_labels.csv",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--labels-dir",
        default=DEFAULT_LABELS_DIR,
        help="Directory for label CSVs (use a new dir for the no-levee pass)",
    )
    parser.add_argument("--node-ids", type=int, nargs="+", default=None)
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-open nodes that already have labels in --labels-dir",
    )
    parser.add_argument("--dem-path", default=DEFAULT_DEM_PATH)
    parser.add_argument("--levee", default=None, help="Mapped artificial levees overlay")
    parser.add_argument("--no-satellite", action="store_true")
    parser.add_argument("--no-dem", action="store_true")
    parser.add_argument(
        "--no-existing-overlay",
        action="store_true",
        help="Do not show faded previous picks from the current labels dir",
    )
    args = parser.parse_args(argv)

    from avulsionprecursors.gui.config import GUIConfig
    from avulsionprecursors.pipeline.labeling import FileLabelingPipeline

    cross_sections = _proj(args.cross_sections)
    profiles_dir = _proj(args.profiles_dir)
    output_dir = _proj(args.output_dir)
    labels_dir = _proj(args.labels_dir)
    dem_path = _proj(args.dem_path) if args.dem_path else None
    if dem_path is not None and not dem_path.exists():
        print(f"Warning: DEM not found ({dem_path}); continuing without DEM display")
        dem_path = None

    if not cross_sections.exists():
        print(f"Error: cross-sections not found: {cross_sections}")
        print("Run extract first, or point --cross-sections at your shapefile.")
        return 1
    if not profiles_dir.exists():
        print(f"Error: profiles dir not found: {profiles_dir}")
        return 1

    print(f"Cross-sections: {cross_sections}")
    print(f"Profiles:       {profiles_dir}")
    print(f"Labels dir:     {labels_dir}")
    print(f"River / study:  {args.river_name}")
    print(f"Label files:    {labels_dir / f'{args.river_name}_node_<id>_labels.csv'}")
    print(f"Skip existing:  {not args.no_skip}")
    print()

    pipeline = FileLabelingPipeline(
        cross_sections_path=cross_sections,
        profiles_dir=profiles_dir,
        river_name=args.river_name,
        output_dir=output_dir,
        labels_dir=labels_dir,
        gui_config=GUIConfig(),
        dem_path=dem_path,
        show_satellite=not args.no_satellite,
        show_dem=(not args.no_dem) and dem_path is not None,
        overlay_path=_proj(args.levee) if args.levee else None,
        overlay_existing_picks=not args.no_existing_overlay,
    )
    pipeline.run(
        node_ids=args.node_ids,
        skip_existing=not args.no_skip,
    )
    print("\nLabeling session finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
