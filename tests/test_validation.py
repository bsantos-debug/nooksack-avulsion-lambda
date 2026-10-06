"""Input validation messages."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from avulsionprecursors.config import load_config, normalize_vertical_units
from avulsionprecursors.exceptions import CRSError, InputValidationError, UnitsError
from avulsionprecursors.io.validation import parse_crs, require_projected, validate_config


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(InputValidationError, match="Could not find the configuration file"):
        load_config(tmp_path / "nope.yaml")


def test_unsupported_vertical_units():
    with pytest.raises(UnitsError, match="Unsupported vertical unit"):
        normalize_vertical_units("inches")


def test_geographic_crs_rejected():
    crs = parse_crs("EPSG:4326")
    with pytest.raises(CRSError, match="geographic"):
        require_projected(crs, "crs.target")


def test_missing_dem_and_centerline(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "dem": "missing.tif",
                    "centerline": "missing.shp",
                    "output_dir": "out",
                }
            }
        )
    )
    cfg = load_config(cfg_path)
    with pytest.raises(InputValidationError, match="Could not find the DEM file"):
        validate_config(cfg)


def test_invalid_spacing(tmp_path: Path, synthetic_study: Path):
    cfg = load_config(synthetic_study / "config.yaml")
    cfg.extract.spacing = 0
    with pytest.raises(InputValidationError, match="spacing"):
        validate_config(cfg)


def test_max_half_less_than_min(tmp_path: Path, synthetic_study: Path):
    cfg = load_config(synthetic_study / "config.yaml")
    cfg.extract.max_half_length = 10
    cfg.extract.cross_half_length = 80
    with pytest.raises(InputValidationError, match="max_half_length"):
        validate_config(cfg)


def test_unwritable_output(synthetic_study: Path, tmp_path: Path):
    cfg = load_config(synthetic_study / "config.yaml")
    blocked = tmp_path / "not_a_directory.txt"
    blocked.write_text("cannot mkdir here")
    cfg.paths.output_dir = blocked
    with pytest.raises(InputValidationError, match="Cannot write"):
        validate_config(cfg)
