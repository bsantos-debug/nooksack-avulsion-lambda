"""Centerline azimuth, valley-wall clip, and DEM sampling along transects.

Scientific rules (unchanged from the working workflow)
-----------------------------------------------------
- Nodes stay on the original centerline. A Savitzky–Golay smoothed copy is
  used only to estimate flow azimuth so raster stair-steps do not rotate
  cross-sections.
- Cross-sections are perpendicular to that azimuth. Azimuth at a node is the
  circular mean of tangents at the node and its along-centerline neighbors.
- Bed slope Sm is the signed downhill slope of a Savitzky–Golay smooth of
  the thalweg long profile (minimum DEM z in a short channel-perpendicular
  strip). Positive Sm is elevation drop in the downstream direction. Adverse
  (negative) slopes are kept and flagged; they are not floored or abs()'d.
- Each transect is probed to ``max_half_length``, then clipped at the first
  sustained rise of ``valley_wall_dz`` (native vertical units) above the
  near-channel elevation.
- Profile distance is 0 at the river node. Elevations are native DEM units.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.signal import savgol_filter
from shapely.geometry import LineString, Point
from shapely.prepared import prep

from avulsionprecursors.io.raster import elevation_valid, mask_nodata


def smooth_centerline_for_azimuth(
    line: LineString,
    window: float,
    sample_step: float = 10.0,
) -> LineString:
    """Savitzky–Golay smooth of x(s), y(s). Used only for transect orientation."""
    if window is None or window <= 0:
        return line
    length = float(line.length)
    if length <= 0:
        return line
    step = max(float(sample_step), 1e-6)
    n = max(int(np.ceil(length / step)) + 1, 5)
    distances = np.linspace(0.0, length, n)
    coords = np.array(
        [[line.interpolate(float(s)).x, line.interpolate(float(s)).y] for s in distances],
        dtype=float,
    )
    win = int(round(float(window) / step))
    if win % 2 == 0:
        win += 1
    win = max(5, win)
    max_win = n if n % 2 == 1 else n - 1
    win = min(win, max_win)
    if win < 5:
        return line
    polyorder = 3 if win > 5 else 2
    polyorder = min(polyorder, win - 1)
    xs = savgol_filter(coords[:, 0], win, polyorder, mode="interp")
    ys = savgol_filter(coords[:, 1], win, polyorder, mode="interp")
    smoothed = LineString(np.column_stack([xs, ys]))
    if smoothed.is_empty or smoothed.length <= 0:
        return line
    return smoothed


def snap_to_centerline(point: Point, line: LineString) -> Point:
    if point is None or point.is_empty:
        raise ValueError("Encountered empty point while snapping to centerline")
    return line.interpolate(line.project(point))


def is_valid_width(value) -> bool:
    try:
        return value is not None and not np.isnan(value) and value > 0
    except TypeError:
        return False


def is_valid_slope(value) -> bool:
    """True if Sm is finite and downhill (usable in γ = SAR / Sm)."""
    try:
        return value is not None and not np.isnan(value) and value > 0
    except TypeError:
        return False


def signed_slope_from_profile(
    s: np.ndarray,
    z: np.ndarray,
    smooth_window: float,
) -> np.ndarray:
    """Sm = −d(z_smooth)/ds. Positive means downhill as s increases downstream."""
    s = np.asarray(s, dtype=float)
    z = np.asarray(z, dtype=float)
    n = int(s.size)
    out = np.full(n, np.nan)
    if n < 5:
        return out
    finite = np.isfinite(s) & np.isfinite(z)
    if finite.sum() < 5:
        return out
    z_fill = z.copy()
    z_fill[~finite] = np.interp(s[~finite], s[finite], z[finite])
    step = float(np.median(np.diff(s)))
    if not np.isfinite(step) or step <= 0:
        return out
    win = int(round(float(smooth_window) / step))
    if win % 2 == 0:
        win += 1
    max_win = n if n % 2 == 1 else n - 1
    win = min(max(5, win), max_win)
    ds = s - np.mean(s)
    denom = float(np.sum(ds * ds))
    if denom < 1e-12:
        return out
    global_slope = float(np.sum(ds * (z_fill - np.mean(z_fill))) / denom)
    if win < 5:
        out[:] = -global_slope
        return out
    poly = min(3, win - 1)
    dzds = savgol_filter(z_fill, window_length=win, polyorder=poly, deriv=1, delta=step, mode="interp")
    return -np.asarray(dzds, dtype=float)


def sample_thalweg_profile(
    centerline: LineString,
    dem,
    sample_step: float = 25.0,
    thalweg_half: float = 80.0,
    channel_polygon=None,
    azimuth_line: Optional[LineString] = None,
    transect_step: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Thalweg z(s): minimum DEM elevation in a short perpendicular strip."""
    az_line = azimuth_line if azimuth_line is not None else centerline
    length = float(centerline.length)
    if length <= 0:
        return np.array([]), np.array([])
    step = max(float(sample_step), 1.0)
    s = np.arange(0.0, length + 0.5 * step, step)
    if s.size == 0 or s[-1] < length - 1e-6:
        s = np.append(s, length)
    half = max(float(thalweg_half), 1.0)
    tstep = max(float(transect_step), 1.0)
    offsets = np.arange(-half, half + 0.5 * tstep, tstep)
    n_off = int(offsets.size)
    tangent_half = min(50.0, max(length * 0.05, 1.0))
    coords: list[tuple[float, float]] = []
    for si in s:
        p = centerline.interpolate(float(si))
        s_az = float(az_line.project(p))
        az = tangent_azimuth_at_chainage(az_line, s_az, tangent_half)
        if not math.isfinite(az):
            az = 0.0
        px, py = -math.sin(az), math.cos(az)
        for d in offsets:
            coords.append((p.x + float(d) * px, p.y + float(d) * py))
    z_all = np.array([e[0] for e in dem.sample(coords)], dtype=float)
    z_all = mask_nodata(z_all, getattr(dem, "nodata", None))
    z_all = z_all.reshape(s.size, n_off)
    if channel_polygon is not None and not getattr(channel_polygon, "is_empty", True):
        poly = prep(channel_polygon)
        inside = np.fromiter(
            (poly.contains(Point(xy)) for xy in coords),
            dtype=bool,
            count=len(coords),
        )
        z_all = np.where(inside.reshape(s.size, n_off), z_all, np.nan)
    with np.errstate(all="ignore"):
        filled = np.where(np.isfinite(z_all), z_all, np.inf)
        z = np.min(filled, axis=1)
        z = np.where(np.isfinite(z_all).any(axis=1), z, np.nan)
    return s, z


def thalweg_signed_slopes_at_points(
    points: Sequence[Point],
    centerline: LineString,
    dem,
    sample_step: float = 25.0,
    smooth_window: float = 15000.0,
    thalweg_half: float = 80.0,
    channel_polygon=None,
    azimuth_line: Optional[LineString] = None,
    transect_step: float = 5.0,
) -> np.ndarray:
    """Signed thalweg Sm at each station (positive downhill downstream)."""
    s, z = sample_thalweg_profile(
        centerline,
        dem,
        sample_step=sample_step,
        thalweg_half=thalweg_half,
        channel_polygon=channel_polygon,
        azimuth_line=azimuth_line,
        transect_step=transect_step,
    )
    sm_prof = signed_slope_from_profile(s, z, smooth_window)
    out = np.full(len(points), np.nan)
    if s.size < 2 or not np.isfinite(sm_prof).any():
        return out
    ok = np.isfinite(s) & np.isfinite(sm_prof)
    if ok.sum() < 2:
        return out
    s_pt = np.array([float(centerline.project(p)) for p in points], dtype=float)
    out[:] = np.interp(s_pt, s[ok], sm_prof[ok])
    return out


def slope_from_dem(point: Point, centerline: LineString, dem, lb: float, slope_min: float) -> float:
    """Least-squares |dz/ds| over [s - Lb/2, s + Lb/2] along the centerline.

    Kept for compatibility. Extraction uses ``thalweg_signed_slopes_at_points``.
    """
    try:
        s = float(centerline.project(point))
        line_length = float(centerline.length)
        lb = float(lb)
        if lb <= 0 or line_length <= 0:
            return float("nan")

        half = 0.5 * lb
        s_start = max(0.0, s - half)
        s_end = min(line_length, s + half)
        if s_end - s_start < 1e-3:
            return float("nan")

        span = s_end - s_start
        step = max(10.0, min(50.0, span / 40.0))
        distances = np.arange(s_start, s_end + 1e-9, step)
        if len(distances) < 2:
            distances = np.linspace(s_start, s_end, max(8, int(span / 5.0) + 1))

        sz_pairs = []
        for s_i in distances:
            p = centerline.interpolate(float(s_i))
            try:
                elev = list(dem.sample([(p.x, p.y)]))[0][0]
            except Exception:
                continue
            if elevation_valid(elev, dem.nodata):
                sz_pairs.append((float(s_i), float(elev)))

        if len(sz_pairs) < 2:
            return float("nan")

        sz_pairs.sort(key=lambda t: t[0])
        s_arr = np.array([t[0] for t in sz_pairs], dtype=float)
        z_arr = np.array([t[1] for t in sz_pairs], dtype=float)
        s_mean = np.mean(s_arr)
        z_mean = np.mean(z_arr)
        ds = s_arr - s_mean
        denom = np.sum(ds * ds)
        if denom < 1e-12:
            return float("nan")
        slope = abs(float(np.sum(ds * (z_arr - z_mean)) / denom))
        if slope < slope_min:
            slope = slope_min
        return slope
    except Exception:
        return float("nan")


def construct_cross_section_line(
    pt: Point,
    perp_azimuth: float,
    half_plus: float,
    half_minus: Optional[float] = None,
) -> LineString:
    """LineString through pt; plus/minus halves may differ after valley-wall clip."""
    if half_minus is None:
        half_minus = half_plus
    x1 = pt.x + half_plus * math.cos(perp_azimuth)
    y1 = pt.y + half_plus * math.sin(perp_azimuth)
    x2 = pt.x - half_minus * math.cos(perp_azimuth)
    y2 = pt.y - half_minus * math.sin(perp_azimuth)
    return LineString([(x1, y1), (pt.x, pt.y), (x2, y2)])


def sample_half_transect(dem, pt: Point, dx: float, dy: float, half: float, step: float):
    """Elevations from the node outward along (dx, dy). dist starts at 0."""
    n = max(2, int(np.floor(half / step)) + 1)
    dist = np.linspace(0.0, half, n)
    coords = [(pt.x + d * dx, pt.y + d * dy) for d in dist]
    elev = np.array([e[0] for e in dem.sample(coords)], dtype=float)
    elev = mask_nodata(elev, dem.nodata)
    return dist, elev


def first_valley_wall(
    dist: np.ndarray,
    elev: np.ndarray,
    zref: float,
    dz: float,
    persist: float,
    skip: float,
) -> float:
    """Distance of the first sustained rise above zref+dz, else the last sample.

    ``dz`` is native vertical units. ``dist``, ``persist``, and ``skip`` are
    horizontal working-CRS units.
    """
    if dist.size < 2:
        return float(dist[-1]) if dist.size else float("nan")
    step = float(np.median(np.diff(dist)))
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    need = max(1, int(round(persist / step)))
    above = np.isfinite(elev) & (elev > zref + dz) & (dist >= skip)
    run = 0
    for i, flag in enumerate(above):
        if flag:
            run += 1
            if run >= need:
                return float(dist[i - need + 1])
        else:
            run = 0
    return float(dist[-1])


def half_lengths_to_valley_walls(
    dem,
    pt: Point,
    perp_azimuth: float,
    max_half: float,
    min_half: float,
    dz: float,
    persist: float,
    skip: float,
    keep: float,
    channel_window: float,
    probe_step: float = 5.0,
) -> tuple[float, float]:
    """Return (half_plus, half_minus) clipped at each valley wall."""
    dx = math.cos(perp_azimuth)
    dy = math.sin(perp_azimuth)
    d_plus, z_plus = sample_half_transect(dem, pt, dx, dy, max_half, probe_step)
    d_minus, z_minus = sample_half_transect(dem, pt, -dx, -dy, max_half, probe_step)

    near = []
    for d, z in ((d_plus, z_plus), (d_minus, z_minus)):
        near.append(z[d <= channel_window])
    near = np.concatenate(near) if near else np.array([])
    near = near[np.isfinite(near)]
    if near.size == 0:
        return float(min_half), float(min_half)
    zref = float(np.median(near))

    def clip_half(dist, elev) -> float:
        wall = first_valley_wall(dist, elev, zref, dz, persist, skip)
        if not np.isfinite(wall):
            wall = max_half
        return float(np.clip(wall + keep, min_half, max_half))

    return clip_half(d_plus, z_plus), clip_half(d_minus, z_minus)


def tangent_azimuth_at_chainage(centerline: LineString, s: float, half_delta: float) -> float:
    """Flow-parallel tangent azimuth (radians, math.atan2 convention)."""
    length = float(centerline.length)
    if length <= 0:
        return float("nan")
    hd = min(float(half_delta), max(length * 0.5 * 0.99, 1e-6))
    s1 = max(0.0, float(s) - hd)
    s2 = min(length, float(s) + hd)
    if s2 - s1 < 1e-9:
        return float("nan")
    p1 = centerline.interpolate(s1)
    p2 = centerline.interpolate(s2)
    return math.atan2(p2.y - p1.y, p2.x - p1.x)


def circular_mean_azimuth(angles: list) -> float:
    """Mean direction for azimuths in radians (handles wrap at ±π)."""
    if not angles:
        return float("nan")
    ca = sum(math.cos(a) for a in angles) / len(angles)
    sa = sum(math.sin(a) for a in angles) / len(angles)
    if abs(ca) < 1e-15 and abs(sa) < 1e-15:
        return float(angles[0])
    return math.atan2(sa, ca)


def width_from_polygon(line: LineString, polygon_union) -> float:
    """Total intersection length between the transect and a channel polygon."""
    if polygon_union is None or polygon_union.is_empty:
        return float("nan")
    intersection = line.intersection(polygon_union)
    if intersection.is_empty:
        return float("nan")
    if intersection.geom_type == "LineString":
        return intersection.length
    if intersection.geom_type == "MultiLineString":
        return sum(segment.length for segment in intersection.geoms)
    return float("nan")


def perpendicular_azimuths(
    points: list,
    river: LineString,
    river_azimuth: LineString,
    tangent_half_delta: float,
) -> list:
    """Perpendicular (transect) azimuth at each node."""
    n_pts = len(points)
    perps: list[float] = []

    def azimuth_at_point(pt: Point) -> float:
        s_az = float(river_azimuth.project(pt))
        return tangent_azimuth_at_chainage(river_azimuth, s_az, tangent_half_delta)

    if n_pts == 0:
        return perps
    if n_pts == 1:
        azimuth = azimuth_at_point(points[0])
        if math.isnan(azimuth):
            azimuth = 0.0
        perps.append(azimuth + math.pi / 2)
        return perps

    chainages = np.array([float(river.project(p)) for p in points], dtype=float)
    order = np.argsort(chainages, kind="mergesort")
    pos_of = np.empty(n_pts, dtype=int)
    for k in range(n_pts):
        pos_of[int(order[k])] = k

    for i in range(n_pts):
        k = int(pos_of[i])
        neighbor_idxs: list[int] = []
        if k > 0:
            neighbor_idxs.append(int(order[k - 1]))
        neighbor_idxs.append(i)
        if k < n_pts - 1:
            neighbor_idxs.append(int(order[k + 1]))

        tangents: list[float] = []
        for j in neighbor_idxs:
            t = azimuth_at_point(points[j])
            if not math.isnan(t):
                tangents.append(t)

        azimuth = circular_mean_azimuth(tangents)
        if math.isnan(azimuth) and tangents:
            azimuth = tangents[0]
        if math.isnan(azimuth):
            azimuth = 0.0
        perps.append(azimuth + math.pi / 2)
    return perps
