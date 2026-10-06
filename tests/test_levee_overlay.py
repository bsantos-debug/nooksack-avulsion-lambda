"""Mapped-levee overlay used by the labeling GUI."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import LineString, Point

from avulsionprecursors.config import load_config
from avulsionprecursors.exceptions import InputValidationError
from avulsionprecursors.gui.labeler import levee_profile_spans
from avulsionprecursors.io.validation import validate_config


def _profile() -> gpd.GeoDataFrame:
    pts = [Point(i, 0.0) for i in range(0, 101, 5)]
    return gpd.GeoDataFrame(
        {
            "dist_along": list(range(0, 101, 5)),
            "elevation": [0.0] * len(pts),
        },
        geometry=pts,
        crs="EPSG:32610",
    )


def test_levee_profile_spans_marks_crossing():
    profile = _profile()
    levees = gpd.GeoDataFrame(
        geometry=[LineString([(50.0, -20.0), (50.0, 20.0)])],
        crs="EPSG:32610",
    )
    spans = levee_profile_spans(profile, levees, buffer_m=10.0)
    assert spans
    lo, hi = spans[0]
    assert lo <= 50.0 <= hi


def test_levee_profile_spans_empty_when_far_away():
    profile = _profile()
    levees = gpd.GeoDataFrame(
        geometry=[LineString([(50.0, 200.0), (50.0, 220.0)])],
        crs="EPSG:32610",
    )
    assert levee_profile_spans(profile, levees, buffer_m=10.0) == []


def test_missing_levee_file_is_validated(synthetic_study: Path):
    cfg = load_config(synthetic_study / "config.yaml")
    cfg.paths.levee = synthetic_study / "missing_levees.geojson"
    with pytest.raises(InputValidationError, match="levee overlay"):
        validate_config(cfg)


def test_config_loads_optional_levee_path(synthetic_study: Path):
    levees = gpd.GeoDataFrame(
        geometry=[LineString([(0.0, 0.0), (10.0, 0.0)])],
        crs="EPSG:32610",
    )
    levee_path = synthetic_study / "levees.geojson"
    levees.to_file(levee_path, driver="GeoJSON")

    raw = yaml.safe_load((synthetic_study / "config.yaml").read_text())
    raw["paths"]["levee"] = "levees.geojson"
    cfg_path = synthetic_study / "config_levee.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)
    assert cfg.paths.levee == levee_path.resolve()
    validate_config(cfg)
