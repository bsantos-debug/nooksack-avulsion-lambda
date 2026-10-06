"""Valley-wall detection and transect helpers."""

from __future__ import annotations

import numpy as np

from avulsionprecursors.geometry.transects import first_valley_wall


def test_first_valley_wall_finds_sustained_rise():
    dist = np.arange(0.0, 200.0, 5.0)
    elev = np.full_like(dist, 10.0)
    elev[dist >= 140] = 50.0
    wall = first_valley_wall(dist, elev, zref=10.0, dz=30.0, persist=20.0, skip=60.0)
    assert 135 <= wall <= 145


def test_first_valley_wall_ignores_short_spike():
    dist = np.arange(0.0, 200.0, 5.0)
    elev = np.full_like(dist, 10.0)
    elev[dist == 80] = 50.0  # one sample only
    wall = first_valley_wall(dist, elev, zref=10.0, dz=30.0, persist=20.0, skip=0.0)
    assert wall == dist[-1]


def test_first_valley_wall_skips_near_channel_levees():
    dist = np.arange(0.0, 200.0, 5.0)
    elev = np.full_like(dist, 10.0)
    elev[(dist >= 30) & (dist <= 50)] = 15.0
    elev[dist >= 140] = 50.0
    wall = first_valley_wall(dist, elev, zref=10.0, dz=4.0, persist=15.0, skip=60.0)
    assert wall >= 130
