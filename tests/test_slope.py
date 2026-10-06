"""Tests for signed thalweg slope from a smoothed long profile."""

from __future__ import annotations

import numpy as np

from avulsionprecursors.geometry.transects import signed_slope_from_profile


def test_signed_slope_recovers_constant_drop():
    s = np.arange(0.0, 20000.0, 25.0)
    z = 50.0 - 0.003 * s
    sm = signed_slope_from_profile(s, z, smooth_window=10000.0)
    assert abs(float(np.nanmedian(sm)) - 0.003) < 2e-4


def test_smoothing_removes_short_waves_and_keeps_sign():
    s = np.arange(0.0, 30000.0, 25.0)
    z = 40.0 - 0.002 * s + 2.0 * np.sin(2.0 * np.pi * s / 2000.0)
    sm = signed_slope_from_profile(s, z, smooth_window=15000.0)
    mid = sm[200:-200]
    assert float(np.nanmedian(mid)) > 0.0015
    assert float(np.nanstd(mid)) < 0.0008


def test_adverse_profile_is_negative():
    s = np.arange(0.0, 5000.0, 25.0)
    z = 10.0 + 0.001 * s
    sm = signed_slope_from_profile(s, z, smooth_window=2000.0)
    assert float(np.nanmedian(sm)) < 0
