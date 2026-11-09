"""Softmax routing slope, kept separate from the thalweg-slope lambda."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from avulsionprecursors.analysis.routing import (
    accumulate_visit_frequency,
    aggregate_elevation,
    aggregation_factor,
    routing_slope_at_seed,
)
from avulsionprecursors.config import load_config
from avulsionprecursors.io.synthetic import write_synthetic_labels
from avulsionprecursors.pipeline.extract import extract_cross_sections
from avulsionprecursors.pipeline.lambda_routing import calculate_lambda_routing


def test_channel_mask_covers_the_centerline_and_not_the_outer_bank():
    import geopandas as gpd
    from rasterio.transform import Affine
    from shapely.geometry import Point

    from avulsionprecursors.pipeline.lambda_routing import channel_geometry, channel_mask

    points = gpd.GeoDataFrame(
        {"dist_out": [0.0, 100.0], "width": [20.0, 20.0]},
        geometry=[Point(5.0, 50.0), Point(95.0, 50.0)],
    )
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    mask = channel_mask((10, 10), transform, channel_geometry(points))
    assert mask[5, 5]
    assert not mask[0, 5]


def test_aggregate_elevation_averages_out_a_one_cell_spike():
    z = np.zeros((6, 6))
    z[1, 1] = 36.0
    coarse = aggregate_elevation(z, aggregation_factor(1.0, 3.0))
    assert coarse.shape == (2, 2)
    assert coarse[0, 0] == 4.0
    assert coarse[1, 1] == 0.0
    assert aggregation_factor(10.0, 0.0) == 1
    assert aggregation_factor(10.0, 30.0) == 3


def test_frequent_path_slope_follows_the_downhill_plane():
    ncols = 25
    z = 10.0 - 0.2 * np.arange(ncols)[None, :]
    z = np.repeat(z, 25, axis=0)
    result = routing_slope_at_seed(
        z,
        seed_row=12,
        seed_col=4,
        xres=10.0,
        yres=10.0,
        n_steps=8,
        n_trials=40,
        theta=30.0,
        alpha=1.0,
        beta=1.0,
        frequency_threshold=0.05,
        rng=np.random.default_rng(1),
    )
    # Eastward step drops 0.2 m over 10 m.
    assert result.n_frequent_visits > 0
    assert 0.015 <= result.slope <= 0.022
    assert result.path_cols[-1] > result.path_cols[0]


def test_walk_cannot_step_back_into_the_channel_at_the_crest():
    z = np.full((9, 9), 4.0)
    z[:, :4] = 0.0
    z[:, 4] = 5.0
    z[:, 5:] = 3.0
    result = routing_slope_at_seed(
        z,
        seed_row=4,
        seed_col=4,
        xres=10.0,
        yres=10.0,
        n_steps=4,
        n_trials=25,
        theta=30.0,
        alpha=1.0,
        beta=1.0,
        frequency_threshold=0.05,
        outward=(0.0, 4.0),
        rng=np.random.default_rng(1),
    )
    assert result.path_cols[1] > result.path_cols[0]
    assert result.path_cols.min() >= 4


def test_walk_can_rejoin_the_channel_downstream():
    nrows = 20
    rows = np.arange(nrows)[:, None]
    z = np.full((nrows, 10), 6.0)
    z[:, :3] = -0.05 * rows
    z[:, 0] -= 1.0
    z[:, 1] -= 0.5
    z[:, 3] = 9.0
    z[15:, 3] = z[15:, 2] + 0.3
    z[:, 4:] = 4.0 - 0.25 * rows
    visit, _ = accumulate_visit_frequency(
        z,
        1,
        3,
        xres=10.0,
        yres=10.0,
        n_steps=18,
        n_trials=15,
        theta=40.0,
        alpha=1.0,
        beta=1.0,
        outward=(0.0, 3.0),
        rng=np.random.default_rng(2),
    )
    assert visit[:12, :3].sum() == 0
    assert visit[15:, :3].sum() > 0


def test_missing_seed_returns_nan():
    z = np.full((5, 5), np.nan)
    result = routing_slope_at_seed(
        z,
        2,
        2,
        xres=1.0,
        yres=1.0,
        n_steps=3,
        n_trials=2,
        rng=np.random.default_rng(0),
    )
    assert np.isnan(result.slope)
    assert result.n_frequent_visits == 0


def test_routing_lambda_does_not_replace_thalweg_outputs(synthetic_study: Path):
    cfg = load_config(synthetic_study / "config.yaml")
    extract_cross_sections(cfg)
    write_synthetic_labels(cfg.profiles_dir(), cfg.study_name, cfg.labels_dir())

    cfg.routing.n_trials = 20
    cfg.routing.path_length = 80.0
    cfg.routing.beta = 1.0
    cfg.routing.seed = 2

    df = calculate_lambda_routing(cfg)
    assert not cfg.lambda_csv_path().exists()
    assert cfg.lambda_routing_csv_path().exists()
    assert cfg.routing_slopes_csv_path().exists()
    assert (df["slope_source"] == "routing_modal_path").all()
    assert "thalweg_slope" in df.columns
    assert df["slope"].notna().all()
    assert (df["slope"] > 0).all()
    assert 0.001 <= float(df["thalweg_slope"].median()) <= 0.004
    assert (df["flag_valid_lambda"]).any()
    # Equal levee crests: the lower-ridge rule keeps bank 1, about 40 m off the node.
    assert (df["routing_side"] == "ridge1").all()
    assert (df["ridge_bank_used"] == 1).all()
    slopes = pd.read_csv(cfg.routing_slopes_csv_path())
    seed_offset = np.hypot(slopes["seed_x"] - slopes["x"], slopes["seed_y"] - slopes["y"])
    assert (seed_offset > 25).all()
    assert (seed_offset < 55).all()
