"""Alluvial-ridge metrics used to compute avulsion potential Λ.

Scientific definitions (Gearon et al.; unchanged)
-------------------------------------------------
Hm  = ridge crest − channel bed                         (native vertical units)
Har = ridge crest − floodplain elevation                (native vertical units)
SAR = |Har| / horizontal distance ridge→floodplain      (vertical / horizontal)
γ   = SAR / Sm
β   = Har / Hm                                          (superelevation)
Λ   = γ × β

Lower-ridge rule: use the bank with the lower crest. A missing bank is
ignored. If only one bank is labeled, that bank is used.

Sm is the signed downhill thalweg slope from a smoothed DEM long profile.
Adverse (negative) Sm is not used in Λ. Discharge is not used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SLOPE = 1e-6
_EPS = 1e-6


def channel_depth(ridge_z, channel_z):
    """Hm = ridge crest − channel bed (native vertical units)."""
    return np.asarray(ridge_z, dtype=float) - np.asarray(channel_z, dtype=float)


def alluvial_ridge_height(ridge_z, floodplain_z):
    """Har = ridge crest − floodplain elevation (native vertical units)."""
    return np.asarray(ridge_z, dtype=float) - np.asarray(floodplain_z, dtype=float)


def ridge_aspect_slope(har, ridge_dist, floodplain_dist, eps: float = _EPS):
    """SAR = |Har| / intervening horizontal distance."""
    run = np.abs(
        np.asarray(ridge_dist, dtype=float) - np.asarray(floodplain_dist, dtype=float)
    )
    run = np.clip(run, eps, None)
    return np.abs(np.asarray(har, dtype=float)) / run


def gamma_from_sar(sar, sm, slope_min: float = MIN_SLOPE):
    """γ = SAR / Sm. Adverse or zero Sm → NaN (Sm is not floored)."""
    sm = np.asarray(sm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.asarray(sar, dtype=float) / sm
    return np.where(sm > 0, gamma, np.nan)


def superelevation(har, hm, slope_min: float = MIN_SLOPE):
    """β = Har / Hm."""
    hm = np.clip(np.asarray(hm, dtype=float), slope_min, None)
    return np.asarray(har, dtype=float) / hm


def avulsion_lambda(gamma, beta):
    """Λ = γ × β."""
    return np.asarray(gamma, dtype=float) * np.asarray(beta, dtype=float)


def choose_lower_ridge_bank(
    ridge1_z,
    ridge2_z,
    har1=None,
    har2=None,
) -> np.ndarray:
    """Gearon et al. lower-crest rule.

    A missing bank (NaN ridge, or NaN Har if provided) is ignored. If only
    one bank is labeled, that bank is used.
    """
    r1 = np.asarray(ridge1_z, dtype=float)
    r2 = np.asarray(ridge2_z, dtype=float)
    has1 = np.isfinite(r1)
    has2 = np.isfinite(r2)
    if har1 is not None:
        has1 = has1 & np.isfinite(np.asarray(har1, dtype=float))
    if har2 is not None:
        has2 = has2 & np.isfinite(np.asarray(har2, dtype=float))
    return np.where(
        has1 & ~has2,
        True,
        np.where(has2 & ~has1, False, r1 <= r2),
    )


def apply_ridge_parameters(df: pd.DataFrame, slope_min: float = MIN_SLOPE) -> pd.DataFrame:
    """Add Har, SAR, Hm, γ, β, and Λ using the lower-ridge rule.

    Required columns (from labels + extraction)
    ------------------------------------------
    channel_elevation, slope (Sm), plus ridge/floodplain columns for each
    labeled bank. A node may have only one ridge + its floodplain; the other
    bank can be NaN.
    """
    required = (
        "channel_elevation",
        "ridge1_elevation",
        "ridge2_elevation",
        "floodplain1_elevation",
        "floodplain2_elevation",
        "ridge1_dist_along",
        "ridge2_dist_along",
        "floodplain1_dist_along",
        "floodplain2_dist_along",
        "slope",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Lambda needs channel, slope, and ridge/floodplain columns "
            f"(blank banks may be NaN). Missing columns: {missing}."
        )

    out = df.copy()
    out["Hm_ridge1"] = channel_depth(out["ridge1_elevation"], out["channel_elevation"])
    out["Hm_ridge2"] = channel_depth(out["ridge2_elevation"], out["channel_elevation"])
    out["Har_ridge1"] = alluvial_ridge_height(
        out["ridge1_elevation"], out["floodplain1_elevation"]
    )
    out["Har_ridge2"] = alluvial_ridge_height(
        out["ridge2_elevation"], out["floodplain2_elevation"]
    )
    out["SAR_ridge1"] = ridge_aspect_slope(
        out["Har_ridge1"], out["ridge1_dist_along"], out["floodplain1_dist_along"]
    )
    out["SAR_ridge2"] = ridge_aspect_slope(
        out["Har_ridge2"], out["ridge2_dist_along"], out["floodplain2_dist_along"]
    )

    use_bank1 = choose_lower_ridge_bank(
        out["ridge1_elevation"],
        out["ridge2_elevation"],
        out["Har_ridge1"],
        out["Har_ridge2"],
    )
    out["ridge_bank_used"] = np.where(use_bank1, 1, 2)
    out["ridge_rule"] = "lower"

    for col in ("Har_ridge1", "SAR_ridge1", "Hm_ridge1"):
        out.loc[~use_bank1, col] = np.nan
    for col in ("Har_ridge2", "SAR_ridge2", "Hm_ridge2"):
        out.loc[use_bank1, col] = np.nan

    out["Har_mean"] = np.nanmean(
        np.vstack(
            [out["Har_ridge1"].to_numpy(dtype=float), out["Har_ridge2"].to_numpy(dtype=float)]
        ),
        axis=0,
    )
    hm1 = out["Hm_ridge1"].astype(float)
    hm2 = out["Hm_ridge2"].astype(float)
    out["channel_depth"] = np.nanmean(
        np.vstack([hm1.to_numpy(dtype=float), hm2.to_numpy(dtype=float)]),
        axis=0,
    )
    out["channel_depth"] = out["channel_depth"].clip(lower=slope_min)
    out["channel_depth_source"] = "ridge_crest_minus_channel_bed"
    out["slope_source"] = "thalweg_sg"
    out["flag_adverse_slope"] = out["slope"].astype(float) <= 0

    sm = out["slope"].astype(float)
    g1 = gamma_from_sar(out["SAR_ridge1"], sm, slope_min)
    g2 = gamma_from_sar(out["SAR_ridge2"], sm, slope_min)
    out["gamma_mean"] = np.nanmean(np.vstack([np.asarray(g1), np.asarray(g2)]), axis=0)

    b1 = superelevation(out["Har_ridge1"], out["Hm_ridge1"].clip(lower=slope_min), slope_min)
    b2 = superelevation(out["Har_ridge2"], out["Hm_ridge2"].clip(lower=slope_min), slope_min)
    out["superelevation_mean"] = np.nanmean(
        np.vstack([np.asarray(b1, dtype=float), np.asarray(b2, dtype=float)]),
        axis=0,
    )
    out["lambda"] = avulsion_lambda(out["gamma_mean"], out["superelevation_mean"])
    out["flag_valid_lambda"] = (
        out["lambda"].notna() & (out["lambda"] > 0) & (sm > 0)
    )
    return out
