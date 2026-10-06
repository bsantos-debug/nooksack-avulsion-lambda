"""End-to-end extract + lambda on a tiny synthetic DEM."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from avulsionprecursors.config import load_config
from avulsionprecursors.io.synthetic import write_synthetic_labels
from avulsionprecursors.pipeline.extract import extract_cross_sections
from avulsionprecursors.pipeline.lambda_calc import calculate_lambda


def test_extract_and_lambda_on_synthetic_dem(synthetic_study: Path):
    cfg = load_config(synthetic_study / "config.yaml")
    extract_cross_sections(cfg)

    xs = cfg.cross_sections_path()
    assert xs.exists()
    profiles = list(cfg.profiles_dir().glob("cross_section_*.csv"))
    assert len(profiles) >= 3

    sample = pd.read_csv(profiles[0])
    assert "distance" in sample.columns and "elevation" in sample.columns
    assert sample["elevation"].notna().sum() > 10

    n_labels = write_synthetic_labels(cfg.profiles_dir(), cfg.study_name, cfg.labels_dir())
    assert n_labels == len(profiles)

    df = calculate_lambda(cfg)
    assert not df.empty
    assert (df["lambda"] > 0).all()
    # Synthetic geometry: Hm ≈ 5 m, Har ≈ 3 m, Sm ≈ 0.002, Λ ≈ 18
    assert 4.0 <= float(df["channel_depth"].median()) <= 6.5
    assert 0.001 <= float(df["slope"].median()) <= 0.004
    assert float(df["lambda"].median()) > 5.0
    assert cfg.lambda_csv_path().exists()
