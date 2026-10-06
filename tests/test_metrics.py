"""Tests for Hm, Har, SAR, gamma, beta, lambda, and the lower-ridge rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from avulsionprecursors.analysis.metrics import (
    alluvial_ridge_height,
    apply_ridge_parameters,
    avulsion_lambda,
    channel_depth,
    choose_lower_ridge_bank,
    gamma_from_sar,
    ridge_aspect_slope,
    superelevation,
)


def test_channel_depth_and_har():
    assert channel_depth(15.0, 10.0) == 5.0
    assert alluvial_ridge_height(15.0, 12.0) == 3.0


def test_sar_gamma_beta_lambda():
    har = 3.0
    hm = 5.0
    sar = ridge_aspect_slope(har, 40.0, 90.0)
    assert abs(sar - 3.0 / 50.0) < 1e-12
    sm = 0.002
    gamma = gamma_from_sar(sar, sm)
    beta = superelevation(har, hm)
    lam = avulsion_lambda(gamma, beta)
    assert abs(gamma - 30.0) < 1e-9
    assert abs(beta - 0.6) < 1e-12
    assert abs(lam - 18.0) < 1e-9


def test_gamma_nan_if_adverse_or_zero_sm():
    sar = 0.06
    assert np.isnan(gamma_from_sar(sar, 0.0))
    assert np.isnan(gamma_from_sar(sar, -0.001))
    assert abs(gamma_from_sar(sar, 0.002) - 30.0) < 1e-9


def test_lower_ridge_picks_shorter_crest():
    use_bank1 = choose_lower_ridge_bank(
        ridge1_z=[10.0],
        ridge2_z=[12.0],
        har1=[1.0],
        har2=[2.0],
    )
    assert bool(use_bank1[0]) is True


def test_lower_ridge_keeps_lower_crest_when_har_negative():
    use_bank1 = choose_lower_ridge_bank(
        ridge1_z=[10.0],
        ridge2_z=[12.0],
        har1=[-1.0],
        har2=[2.0],
    )
    assert bool(use_bank1[0]) is True


def test_lower_ridge_keeps_lower_crest_if_both_har_negative():
    use_bank1 = choose_lower_ridge_bank(
        ridge1_z=[10.0],
        ridge2_z=[12.0],
        har1=[-1.0],
        har2=[-0.5],
    )
    assert bool(use_bank1[0]) is True


def test_apply_ridge_parameters_dataframe():
    df = pd.DataFrame(
        {
            "channel_elevation": [10.0],
            "ridge1_elevation": [15.0],
            "ridge2_elevation": [16.0],
            "floodplain1_elevation": [12.0],
            "floodplain2_elevation": [12.0],
            "ridge1_dist_along": [40.0],
            "ridge2_dist_along": [140.0],
            "floodplain1_dist_along": [90.0],
            "floodplain2_dist_along": [190.0],
            "slope": [0.002],
        }
    )
    out = apply_ridge_parameters(df)
    assert int(out.loc[0, "ridge_bank_used"]) == 1
    assert np.isnan(out.loc[0, "Hm_ridge2"])
    assert abs(out.loc[0, "lambda"] - 18.0) < 1e-8


def test_lower_ridge_uses_only_labeled_bank():
    use_bank1 = choose_lower_ridge_bank(
        ridge1_z=[10.0],
        ridge2_z=[np.nan],
        har1=[1.0],
        har2=[np.nan],
    )
    assert bool(use_bank1[0]) is True

    use_bank1 = choose_lower_ridge_bank(
        ridge1_z=[np.nan],
        ridge2_z=[12.0],
        har1=[np.nan],
        har2=[2.0],
    )
    assert bool(use_bank1[0]) is False


def test_apply_ridge_parameters_one_bank():
    df = pd.DataFrame(
        {
            "channel_elevation": [10.0],
            "ridge1_elevation": [15.0],
            "ridge2_elevation": [np.nan],
            "floodplain1_elevation": [12.0],
            "floodplain2_elevation": [np.nan],
            "ridge1_dist_along": [40.0],
            "ridge2_dist_along": [np.nan],
            "floodplain1_dist_along": [90.0],
            "floodplain2_dist_along": [np.nan],
            "slope": [0.002],
        }
    )
    out = apply_ridge_parameters(df)
    assert int(out.loc[0, "ridge_bank_used"]) == 1
    assert abs(out.loc[0, "lambda"] - 18.0) < 1e-8
    assert np.isnan(out.loc[0, "Hm_ridge2"])
