"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from avulsionprecursors.io.synthetic import write_synthetic_valley


def synthetic_config_dict(root: Path) -> dict:
    return {
        "study_name": "synthetic",
        "crs": {"target": "EPSG:32610"},
        "units": {"vertical": "m"},
        "paths": {
            "dem": "dem.tif",
            "centerline": "centerline.shp",
            "output_dir": "outputs",
        },
        "extract": {
            "spacing": 200,
            "cross_half_length": 80,
            "max_half_length": 250,
            "sample_step": 10,
            "valley_wall_dz": 30,
            "valley_wall_persist": 20,
            "valley_wall_skip_channel": 60,
            "valley_wall_keep": 10,
            "max_elev_above_channel": 50,
            "channel_elev_window": 20,
            "backwater_length": 400,
            "centerline_smooth_window": 80,
            "centerline_smooth_sample": 10,
            "flow_tangent_half_delta": 40,
            "slope_smooth_window": 800,
            "slope_sample_step": 10,
            "thalweg_half_width": 40,
            "slope_min": 1e-5,
        },
        "lambda_calc": {"auto_minmax": False},
    }


@pytest.fixture
def synthetic_study(tmp_path: Path) -> Path:
    write_synthetic_valley(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(synthetic_config_dict(tmp_path)))
    return tmp_path
