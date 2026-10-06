"""Command-line entry point: extract, label, lambda, or run extract+lambda."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Sequence

from avulsionprecursors.config import WorkflowConfig, apply_cli_overrides, load_config
from avulsionprecursors.exceptions import WorkflowError
from avulsionprecursors.io.validation import validate_config
from avulsionprecursors.pipeline.extract import extract_cross_sections
from avulsionprecursors.pipeline.label import run_labeler
from avulsionprecursors.pipeline.lambda_calc import calculate_lambda


def _shared_flags() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-c",
        "--config",
        required=True,
        help="YAML configuration file (see config/example.yaml)",
    )
    parent.add_argument("--dem", help="Override paths.dem")
    parent.add_argument("--centerline", help="Override paths.centerline")
    parent.add_argument("--channel-polygon", dest="channel_polygon", default=None)
    parent.add_argument("--clip-boundary", dest="clip_boundary", default=None)
    parent.add_argument(
        "--levee",
        dest="levee",
        default=None,
        help="Line/polygon overlay of mapped artificial levees in the labeler",
    )
    parent.add_argument("--output-dir", dest="output_dir", default=None)
    parent.add_argument("--study-name", dest="study_name", default=None)
    parent.add_argument("--crs", dest="crs_target", default=None, help="UTM or EPSG:xxxx")
    parent.add_argument(
        "--vertical-units",
        dest="vertical_units",
        default=None,
        help="native, m, ft, or us_ft (not converted; documents DEM Z units)",
    )
    parent.add_argument("--spacing", type=float, default=None)
    parent.add_argument("--labels-dir", dest="labels_dir", default=None)
    parent.add_argument(
        "--auto-minmax",
        dest="auto_minmax",
        action="store_true",
        help="Pick channel=min and ridges=max left/right (can select valley walls)",
    )
    return parent


def _build_parser() -> argparse.ArgumentParser:
    parent = _shared_flags()
    parser = argparse.ArgumentParser(
        prog="avulsion-process",
        description=(
            "Extract river cross-sections from a DEM and compute avulsion "
            "potential Λ from alluvial-ridge geometry."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract", parents=[parent], help="Build cross-sections and DEM profiles")
    label_p = sub.add_parser("label", parents=[parent], help="Interactive labeling GUI")
    label_p.add_argument("--node-ids", nargs="+", type=int, default=None)
    label_p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-open nodes that already have labels",
    )
    label_p.add_argument(
        "--no-existing-overlay",
        action="store_true",
        help="Do not show faded previous picks",
    )
    sub.add_parser("lambda", parents=[parent], help="Compute Λ from labels")
    sub.add_parser("run", parents=[parent], help="extract, then lambda (skips the GUI)")
    return parser


def _load(args) -> WorkflowConfig:
    cfg = load_config(args.config)
    overrides = {
        "dem": args.dem,
        "centerline": args.centerline,
        "channel_polygon": args.channel_polygon,
        "clip_boundary": args.clip_boundary,
        "levee": args.levee,
        "output_dir": args.output_dir,
        "study_name": args.study_name,
        "crs_target": args.crs_target,
        "vertical_units": args.vertical_units,
        "spacing": args.spacing,
        "auto_minmax": True if getattr(args, "auto_minmax", False) else None,
        "labels_dir": args.labels_dir,
    }
    cwd = Path.cwd()
    for key in (
        "dem",
        "centerline",
        "channel_polygon",
        "clip_boundary",
        "levee",
        "output_dir",
        "labels_dir",
    ):
        value = overrides[key]
        if value:
            p = Path(value)
            overrides[key] = p.resolve() if p.is_absolute() else (cwd / p).resolve()
    cfg = apply_cli_overrides(cfg, overrides)
    validate_config(cfg)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _load(args)
        if args.command == "extract":
            extract_cross_sections(cfg)
        elif args.command == "label":
            skip = False if getattr(args, "no_skip", False) else None
            overlay_existing = False if getattr(args, "no_existing_overlay", False) else None
            run_labeler(
                cfg,
                node_ids=getattr(args, "node_ids", None),
                skip_existing=skip,
                overlay_existing=overlay_existing,
            )
        elif args.command == "lambda":
            calculate_lambda(cfg)
        elif args.command == "run":
            extract_cross_sections(cfg)
            calculate_lambda(cfg)
        else:
            parser.error(f"Unknown command {args.command}")
        return 0
    except WorkflowError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
