"""Bankfull-channel marks on the labeler, and partial-polygon thalweg masking."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from avulsionprecursors.geometry.transects import mask_thalweg_outside_channel
from avulsionprecursors.gui.labeler import channel_profile_spans


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


def test_channel_profile_spans_follow_the_polygon_intersection():
    profile = _profile()
    channel = gpd.GeoDataFrame(
        geometry=[Polygon([(40.0, -10.0), (70.0, -10.0), (70.0, 10.0), (40.0, 10.0)])],
        crs="EPSG:32610",
    )
    spans = channel_profile_spans(profile, channel)
    assert len(spans) == 1
    lo, hi = spans[0]
    assert abs(lo - 40.0) < 1e-6
    assert abs(hi - 70.0) < 1e-6


def test_channel_profile_spans_empty_when_transect_misses_the_polygon():
    profile = _profile()
    channel = gpd.GeoDataFrame(
        geometry=[Polygon([(40.0, 20.0), (70.0, 20.0), (70.0, 30.0), (40.0, 30.0)])],
        crs="EPSG:32610",
    )
    assert channel_profile_spans(profile, channel) == []


def test_partial_channel_mask_keeps_the_uncovered_reach():
    z = np.array([[1.0, 2.0], [5.0, 9.0]])
    inside = np.array([[True, False], [False, False]])
    node_inside = np.array([True, False])
    out = mask_thalweg_outside_channel(z, inside, node_inside)
    assert out[0, 0] == 1.0
    assert np.isnan(out[0, 1])
    assert out[1, 0] == 5.0
    assert out[1, 1] == 9.0
