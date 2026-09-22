import geopandas as gpd
import rasterio
import numpy as np
import math
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from scipy.signal import savgol_filter

# === USER PARAMETERS ===
river_centerline_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack_data/GIS/Centerline_flowaccumulation/Centerline_flowaccumulation.shp"
centerline_points_path = None  # set to None to auto-sample with spacing parameter, or provide path to existing points shapefile
# centerline_points_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Nooksack_research/Qgis/Crosssections/Modern_Nooksack_Centerline_Points_400m.shp"  # commented out to use auto-sampling
channel_polygon_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Nooksack_research/Qgis/Crosssections/Modern_Nooksack_channel.shp"  # optional polygon delineating channel boundaries
dem_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
spacing = 200        # distance between points along the river (m) when auto-sampling
cross_half_length = 800  # minimum half-length of each cross section (m)
sample_step = 1         # spacing of samples along each cross section (m)
# Probe this far on each side, then clip at the valley wall (see below).
max_half_length_m = 4000.0
# First sustained rise of this many metres above the near-channel elevation
# is treated as the valley wall. Levees on the Nooksack are typically <20 m.
valley_wall_dz_m = 30.0
valley_wall_persist_m = 50.0   # must stay above threshold for this distance
valley_wall_skip_channel_m = 200.0  # ignore bars/levees near the channel
valley_wall_keep_m = 100.0     # keep a short toe of the wall past the trigger
# Mask DEM samples more than this many meters above the near-channel elevation
# (avoids leftover hills dominating the profile y-axis). Set to None to disable.
max_elev_above_channel_m = 40.0
# Window (m) around centerline used to estimate the channel reference elevation
channel_elev_window_m = 50.0
# Backwater length scale Lb (m): total centerline reach used for slope (Lb/2 upstream and Lb/2 downstream of each cross-section).
# 1 km is too short on the lower Nooksack (local bars dominate). 5 km matches the
# FlowFM WSE window and the ~0.0034 reach slope reported near Everson.
backwater_length_m = 5000.0
# Savitzky–Golay window (m) for the azimuth centerline. Raster flow-accumulation
# lines stair-step; nodes stay on the original line, but transect orientation
# uses this smoothed copy so cross-sections are not perpendicular to pixel edges.
centerline_smooth_window_m = 400.0
centerline_smooth_sample_m = 10.0
# Half-span (m) along the (smoothed) centerline for finite-difference tangent
flow_tangent_half_delta_m = 100.0
output_cross_sections = "cross_sections.shp"
output_points = "centerline_points_from_xs.shp"  # node points at XS midpoints (for lambda)
output_smoothed_centerline = "centerline_smoothed_for_azimuth.shp"  # QC overlay in QGIS
output_profiles_folder = "cross_section_profiles"  # folder for CSV outputs

import os
os.makedirs(output_profiles_folder, exist_ok=True)


def _horizontal_crs(crs):
    """Drop a compound vertical CRS so shapefile outputs are 2D projected metres."""
    if crs is None:
        return crs
    try:
        if crs.is_compound and getattr(crs, "sub_crs_list", None):
            return crs.sub_crs_list[0]
    except Exception:
        pass
    return crs


def _as_single_linestring(geom) -> LineString:
    """Collapse MultiLineString parts into one LineString (longest part if unmergeable)."""
    if geom is None or geom.is_empty:
        raise ValueError("Centerline geometry is empty")
    if geom.geom_type == "LineString":
        return geom
    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else list(geom.geoms)
        return max(parts, key=lambda g: g.length)
    raise ValueError(f"Centerline must be a line, got {geom.geom_type}")


def _smooth_centerline_for_azimuth(
    line: LineString,
    window_m: float,
    sample_step_m: float = 10.0,
) -> LineString:
    """Savitzky–Golay smooth of x(s), y(s). Used only for transect orientation."""
    if window_m is None or window_m <= 0:
        return line
    length = float(line.length)
    if length <= 0:
        return line
    step = max(float(sample_step_m), 1.0)
    n = max(int(np.ceil(length / step)) + 1, 5)
    distances = np.linspace(0.0, length, n)
    coords = np.array(
        [[line.interpolate(float(s)).x, line.interpolate(float(s)).y] for s in distances],
        dtype=float,
    )
    win = int(round(float(window_m) / step))
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


# === STEP 1: Load data ===
gdf = gpd.read_file(river_centerline_path)
if gdf.empty:
    raise ValueError(f"No features found in {river_centerline_path}")
gdf = gdf.set_crs(_horizontal_crs(gdf.crs), allow_override=True)
crs = gdf.crs
if len(gdf) == 1:
    river = _as_single_linestring(gdf.geometry.iloc[0])
else:
    river = _as_single_linestring(linemerge(gdf.geometry.unary_union))

river_azimuth = _smooth_centerline_for_azimuth(
    river,
    centerline_smooth_window_m,
    centerline_smooth_sample_m,
)
if output_smoothed_centerline:
    gpd.GeoDataFrame({"id": [1]}, geometry=[river_azimuth], crs=crs).to_file(
        output_smoothed_centerline
    )
    print(
        f"ℹ️  Azimuth centerline smoothed with {centerline_smooth_window_m:.0f} m "
        f"Savitzky–Golay window → {output_smoothed_centerline}"
    )

# === STEP 2: Ingest or generate points along the centerline ===
def _snap_to_centerline(point: Point, line: LineString) -> Point:
    """Snap a point to the closest location on the centerline."""
    if point is None or point.is_empty:
        raise ValueError("Encountered empty point while snapping to centerline")
    projected_dist = line.project(point)
    return line.interpolate(projected_dist)

points = []
point_ids = []
widths = []
slopes = []
dist_outs = []
perp_azimuths = []

if centerline_points_path and os.path.exists(centerline_points_path):
    points_gdf = gpd.read_file(centerline_points_path).to_crs(crs)
    if points_gdf.empty:
        raise ValueError(f"No points found in {centerline_points_path}")
    for idx, row in points_gdf.iterrows():
        snapped_point = _snap_to_centerline(row.geometry, river)
        points.append(snapped_point)
        point_ids.append(row.get("node_id", idx + 1))
        widths.append(row.get("width"))
        slopes.append(row.get("slope"))
        along_distance = river.project(snapped_point)
        dist_outs.append(river.length - along_distance)
else:
    distance = 0
    pid = 1
    while distance <= river.length:
        snapped_point = river.interpolate(distance)
        points.append(snapped_point)
        point_ids.append(pid)
        widths.append(None)
        slopes.append(None)
        dist_outs.append(river.length - distance)
        distance += spacing
        pid += 1

# === Helper functions ===
def _is_valid_width(value) -> bool:
    try:
        return value is not None and not np.isnan(value) and value > 0
    except TypeError:
        return False

def _is_valid_slope(value) -> bool:
    """Check if slope value is valid."""
    try:
        return value is not None and not np.isnan(value) and value > 0
    except TypeError:
        return False


def _dem_elevation_valid(elev: float, nodata) -> bool:
    """True if raster sample is usable (not NaN and not nodata when nodata is defined)."""
    if elev is None or (isinstance(elev, (float, np.floating)) and np.isnan(elev)):
        return False
    if nodata is None:
        return True
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        return True
    return elev != nodata

# === STEP 3: Calculate slopes from DEM along centerline (over backwater length Lb) ===
def _calculate_slope_from_dem(
    point: Point,
    centerline: LineString,
    dem: rasterio.DatasetReader,
    lb_m: float,
) -> float:
    """
    Bed slope from DEM along the centerline over total reach Lb centered on the cross-section.

    Chainage window [s - Lb/2, s + Lb/2] along the centerline, clamped to [0, line_length]
    (end stations use a shorter asymmetric window).

    Slope is least-squares dz/ds over valid DEM samples (m/m in projected CRS).

    Args:
        point: Point along centerline
        centerline: Centerline LineString
        dem: Opened rasterio DEM
        lb_m: Total backwater reach (m); half upstream and half downstream of the station

    Returns:
        Slope magnitude (unitless m/m) or NaN if calculation fails
    """
    try:
        s = float(centerline.project(point))
        line_length = float(centerline.length)
        lb = float(lb_m)
        if lb <= 0 or line_length <= 0:
            return float("nan")

        half = 0.5 * lb
        s_start = max(0.0, s - half)
        s_end = min(line_length, s + half)
        if s_end - s_start < 1e-3:
            return float("nan")

        span = s_end - s_start
        # Enough samples along Lb without oversampling tiny steps
        step = max(10.0, min(50.0, span / 40.0))
        distances = np.arange(s_start, s_end + 1e-9, step)
        if len(distances) < 2:
            distances = np.linspace(s_start, s_end, max(8, int(span / 5.0) + 1))

        sz_pairs = []
        for s_i in distances:
            p = centerline.interpolate(float(s_i))
            coord = (p.x, p.y)
            try:
                elev = list(dem.sample([coord]))[0][0]
            except Exception:
                continue
            if _dem_elevation_valid(elev, dem.nodata):
                sz_pairs.append((float(s_i), float(elev)))

        if len(sz_pairs) < 2:
            return float("nan")

        sz_pairs.sort(key=lambda t: t[0])
        s_arr = np.array([t[0] for t in sz_pairs], dtype=float)
        z_arr = np.array([t[1] for t in sz_pairs], dtype=float)

        # Least-squares slope dz/ds along line chainage s
        s_mean = np.mean(s_arr)
        z_mean = np.mean(z_arr)
        ds = s_arr - s_mean
        denom = np.sum(ds * ds)
        if denom < 1e-12:
            return float("nan")
        slope = float(np.sum(ds * (z_arr - z_mean)) / denom)

        slope = abs(slope)
        # Minimum slope (m/m): avoids near-zero fits blowing up downstream quantities that divide by S
        slope_min = 1e-5
        if slope < slope_min:
            slope = slope_min
        return slope

    except Exception as e:
        print(f"⚠️  Warning: Could not calculate slope for point: {e}")
        return float("nan")


# Calculate slopes from DEM if available
if dem_path and os.path.exists(dem_path):
    half = 0.5 * backwater_length_m
    print(
        f"📐 Calculating slopes from DEM (reach = {backwater_length_m:.0f} m total, "
        f"±{half:.0f} m along centerline from each cross-section): {dem_path}"
    )
    dem = rasterio.open(dem_path)
    
    slopes_calculated = 0
    for idx, pt in enumerate(points):
        slope_val = _calculate_slope_from_dem(pt, river, dem, backwater_length_m)
        if _is_valid_slope(slope_val):
            slopes[idx] = slope_val
            slopes_calculated += 1
    
    dem.close()
    
    print(f"✅ Calculated {slopes_calculated} slopes from DEM")
    if slopes_calculated < len(points):
        existing = sum(1 for s in slopes if _is_valid_slope(s))
        print(f"   Total slopes: {existing} (calculated: {slopes_calculated}, from shapefile: {existing - slopes_calculated})")
        if existing < len(points):
            print(f"   Missing slopes: {len(points) - existing} (will use default: {1e-4})")
else:
    print("ℹ️  No DEM provided for slope calculation, using slopes from points shapefile or defaults")

points_gdf = gpd.GeoDataFrame({
    "node_id": point_ids,
    "dist_out": dist_outs,
    "width": widths,
    "slope": slopes
}, geometry=points, crs=crs)


def _construct_cross_section_line(
    pt: Point,
    perp_azimuth: float,
    half_plus: float,
    half_minus: float | None = None,
) -> LineString:
    """Create a LineString through pt; plus/minus halves may differ (valley-wall clip)."""
    if half_minus is None:
        half_minus = half_plus
    x1 = pt.x + half_plus * math.cos(perp_azimuth)
    y1 = pt.y + half_plus * math.sin(perp_azimuth)
    x2 = pt.x - half_minus * math.cos(perp_azimuth)
    y2 = pt.y - half_minus * math.sin(perp_azimuth)
    return LineString([(x1, y1), (pt.x, pt.y), (x2, y2)])


def _sample_half_transect(
    dem: rasterio.DatasetReader,
    pt: Point,
    dx: float,
    dy: float,
    half_m: float,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Elevations from the node outward along (dx, dy). dist starts at 0."""
    n = max(2, int(np.floor(half_m / step_m)) + 1)
    dist = np.linspace(0.0, half_m, n)
    coords = [(pt.x + d * dx, pt.y + d * dy) for d in dist]
    elev = np.array([e[0] for e in dem.sample(coords)], dtype=float)
    if dem.nodata is not None:
        elev = np.where(elev == dem.nodata, np.nan, elev)
    elev = np.where(np.isfinite(elev) & (np.abs(elev) < 1e6), elev, np.nan)
    return dist, elev


def _first_valley_wall_m(
    dist: np.ndarray,
    elev: np.ndarray,
    zref: float,
    dz: float,
    persist_m: float,
    skip_m: float,
) -> float:
    """Distance of the first sustained rise above zref+dz, else the last sample."""
    if dist.size < 2:
        return float(dist[-1]) if dist.size else float("nan")
    step = float(np.median(np.diff(dist)))
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    need = max(1, int(round(persist_m / step)))
    above = np.isfinite(elev) & (elev > zref + dz) & (dist >= skip_m)
    run = 0
    for i, flag in enumerate(above):
        if flag:
            run += 1
            if run >= need:
                return float(dist[i - need + 1])
        else:
            run = 0
    return float(dist[-1])


def _half_lengths_to_valley_walls(
    dem: rasterio.DatasetReader,
    pt: Point,
    perp_azimuth: float,
    max_half: float,
    min_half: float,
    dz: float,
    persist_m: float,
    skip_m: float,
    keep_m: float,
    channel_window_m: float,
    probe_step_m: float = 5.0,
) -> tuple[float, float]:
    """Return (half_plus, half_minus) clipped at each valley wall."""
    dx = math.cos(perp_azimuth)
    dy = math.sin(perp_azimuth)
    d_plus, z_plus = _sample_half_transect(dem, pt, dx, dy, max_half, probe_step_m)
    d_minus, z_minus = _sample_half_transect(dem, pt, -dx, -dy, max_half, probe_step_m)

    near = []
    for d, z in ((d_plus, z_plus), (d_minus, z_minus)):
        near.append(z[d <= channel_window_m])
    near = np.concatenate(near) if near else np.array([])
    near = near[np.isfinite(near)]
    if near.size == 0:
        return float(min_half), float(min_half)
    zref = float(np.median(near))

    def clip_half(dist, elev) -> float:
        wall = _first_valley_wall_m(dist, elev, zref, dz, persist_m, skip_m)
        if not np.isfinite(wall):
            wall = max_half
        return float(np.clip(wall + keep_m, min_half, max_half))

    return clip_half(d_plus, z_plus), clip_half(d_minus, z_minus)


def _tangent_azimuth_at_chainage(
    centerline: LineString,
    s: float,
    half_delta_m: float,
) -> float:
    """Flow-parallel tangent azimuth (radians, math.atan2 convention) at chainage s via centerline geometry."""
    L = float(centerline.length)
    if L <= 0:
        return float("nan")
    hd = min(float(half_delta_m), max(L * 0.5 * 0.99, 1e-6))
    s1 = max(0.0, float(s) - hd)
    s2 = min(L, float(s) + hd)
    if s2 - s1 < 1e-9:
        return float("nan")
    p1 = centerline.interpolate(s1)
    p2 = centerline.interpolate(s2)
    return math.atan2(p2.y - p1.y, p2.x - p1.x)


def _circular_mean_azimuth(angles: list) -> float:
    """Mean direction for a set of azimuths in radians (handles wrap at ±π)."""
    if not angles:
        return float("nan")
    ca = sum(math.cos(a) for a in angles) / len(angles)
    sa = sum(math.sin(a) for a in angles) / len(angles)
    if abs(ca) < 1e-15 and abs(sa) < 1e-15:
        return float(angles[0])
    return math.atan2(sa, ca)


def _calculate_width_from_polygon(
    line: LineString,
    polygon_union
) -> float:
    """Return the total intersection length between the line and channel polygon."""
    if polygon_union.is_empty:
        return float("nan")
    intersection = line.intersection(polygon_union)
    if intersection.is_empty:
        return float("nan")

    if intersection.geom_type == "LineString":
        return intersection.length
    if intersection.geom_type == "MultiLineString":
        return sum(segment.length for segment in intersection.geoms)
    return float("nan")


# === STEP 4: Flow direction and cross-section azimuths ===
# Cross sections are perpendicular to flow. Nodes stay on the original centerline;
# azimuth uses the smoothed line so raster stair-steps do not rotate the transect.
# Flow at each node is the circular mean of tangents at that node and its
# along-centerline neighbors (ordered by original chainage).
perpendicular_data = []
perp_azimuths.clear()
n_pts = len(points)


def _azimuth_at_point(pt: Point) -> float:
    s_az = float(river_azimuth.project(pt))
    return _tangent_azimuth_at_chainage(river_azimuth, s_az, flow_tangent_half_delta_m)


if n_pts == 0:
    pass
elif n_pts == 1:
    azimuth = _azimuth_at_point(points[0])
    if math.isnan(azimuth):
        azimuth = 0.0
    perp_azimuth = azimuth + math.pi / 2
    perp_azimuths.append(perp_azimuth)
    perpendicular_data.append((azimuth, perp_azimuth))
else:
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
            t = _azimuth_at_point(points[j])
            if not math.isnan(t):
                tangents.append(t)

        azimuth = _circular_mean_azimuth(tangents)
        if math.isnan(azimuth) and tangents:
            azimuth = tangents[0]
        if math.isnan(azimuth):
            azimuth = 0.0
        perp_azimuth = azimuth + math.pi / 2
        perp_azimuths.append(perp_azimuth)
        perpendicular_data.append((azimuth, perp_azimuth))

print(
    f"ℹ️  Cross-section normals from smoothed flow azimuth "
    f"(SG window {centerline_smooth_window_m:.0f} m; circular mean of node ± "
    f"along-line neighbors; tangent Δ = ±{flow_tangent_half_delta_m:.1f} m)"
)


# === STEP 5: Update widths using channel polygon if provided ===
if channel_polygon_path and os.path.exists(channel_polygon_path):
    print(f"📏 Calculating widths from channel polygon: {channel_polygon_path}")
    channel_polys = None
    last_err = None
    for attempt in range(1, 4):
        try:
            channel_polys = gpd.read_file(channel_polygon_path).to_crs(crs)
            break
        except Exception as err:
            last_err = err
            print(f"⚠️  Could not read channel polygon (attempt {attempt}/3): {err}")
            if attempt < 3:
                import time
                time.sleep(2 * attempt)
    if channel_polys is None:
        raise last_err
    channel_union = channel_polys.unary_union
    probe_half_length = max(cross_half_length, 500)
    
    widths_calculated = 0
    for idx, (pt, perp_azimuth) in enumerate(zip(points, perp_azimuths)):
        probe_line = _construct_cross_section_line(pt, perp_azimuth, probe_half_length)
        width_val = _calculate_width_from_polygon(probe_line, channel_union)
        if _is_valid_width(width_val):
            widths[idx] = width_val
            widths_calculated += 1
    
    print(f"✅ Calculated {widths_calculated} widths from channel polygon")
    if widths_calculated < len(points):
        print(f"⚠️  Warning: {len(points) - widths_calculated} points did not intersect channel polygon")
else:
    print("ℹ️  No channel polygon provided, using widths from points shapefile or defaults")

points_gdf["width"] = widths
points_gdf["slope"] = slopes  # Update slopes in points_gdf

# Save updated points with calculated widths and slopes if output path is provided
if output_points:
    points_gdf.to_file(output_points)
    print(f"✅ Saved updated points with calculated widths and slopes → {output_points}")


# === STEP 6: Create cross-section lines perpendicular to flow ===
# Probe out to max_half_length_m, then clip each side at the valley wall
# (first sustained rise of valley_wall_dz_m above the near-channel elevation).
# Falls back to min half-length when the DEM is missing or the wall is very close.
cross_sections = []
if dem_path and os.path.exists(dem_path):
    dem_walls = rasterio.open(dem_path)
    print(
        f"⛰️  Clipping transects at valley walls "
        f"(probe ±{max_half_length_m:.0f} m, Δz ≥ {valley_wall_dz_m:.0f} m, "
        f"min half {cross_half_length:.0f} m)"
    )
    n_hit = 0
    for idx, pt in enumerate(points):
        perp_azimuth = perp_azimuths[idx]
        half_plus, half_minus = _half_lengths_to_valley_walls(
            dem_walls,
            pt,
            perp_azimuth,
            max_half=max_half_length_m,
            min_half=float(cross_half_length),
            dz=valley_wall_dz_m,
            persist_m=valley_wall_persist_m,
            skip_m=valley_wall_skip_channel_m,
            keep_m=valley_wall_keep_m,
            channel_window_m=channel_elev_window_m,
        )
        if half_plus < max_half_length_m - 1.0 or half_minus < max_half_length_m - 1.0:
            n_hit += 1
        cross_sections.append(
            _construct_cross_section_line(pt, perp_azimuth, half_plus, half_minus)
        )
    dem_walls.close()
    print(f"   Valley wall hit on at least one side: {n_hit}/{len(points)}")
else:
    print("⚠️  No DEM for valley-wall clip; using min half-length")
    for idx, pt in enumerate(points):
        perp_azimuth = perp_azimuths[idx]
        cross_sections.append(
            _construct_cross_section_line(pt, perp_azimuth, float(cross_half_length))
        )

cross_gdf = gpd.GeoDataFrame({
    "node_id": point_ids,
    "dist_out": dist_outs,
    "width": widths,
    "slope": slopes
}, geometry=cross_sections, crs=crs)
cross_gdf.to_file(output_cross_sections)
print(f"✅ Created {len(cross_sections)} cross sections with widths and slopes → {output_cross_sections}")
valid_widths = [w for w in widths if _is_valid_width(w)]
valid_slopes = [s for s in slopes if _is_valid_slope(s)]
if valid_widths:
    print(f"   Width range: {min(valid_widths):.1f} - {max(valid_widths):.1f} m")
if valid_slopes:
    print(f"   Slope range: {min(valid_slopes):.6f} - {max(valid_slopes):.6f} (mean: {np.mean(valid_slopes):.6f})")

# === STEP 7: Sample DEM along each cross-section ===
# Note: DEM was already opened for slope calculation, but we need to reopen it
# for sampling along cross-sections (rasterio handles this efficiently)
dem = rasterio.open(dem_path)

masked_count = 0
for i, (line, node_id) in enumerate(zip(cross_sections, point_ids)):
    # sample points along the line (distance_m = 0 at the river node)
    num = max(2, int(line.length / sample_step))
    distances = np.linspace(0, line.length, num)
    coords_line = list(line.coords)
    if len(coords_line) >= 3:
        s_channel = float(line.project(Point(coords_line[1])))
    else:
        s_channel = line.length / 2.0
    distances_centered = distances - s_channel
    points_along = [line.interpolate(d) for d in distances]

    # sample DEM elevations
    coords = [(p.x, p.y) for p in points_along]
    elev = np.array([e[0] for e in dem.sample(coords)], dtype=float)
    if dem.nodata is not None:
        elev = np.where(elev == dem.nodata, np.nan, elev)

    # Mask valley-wall elevations relative to near-channel reference
    if max_elev_above_channel_m is not None:
        near = elev[np.abs(distances_centered) <= channel_elev_window_m]
        near_valid = near[np.isfinite(near)]
        if near_valid.size > 0:
            channel_ref = float(np.median(near_valid))
            ceiling = channel_ref + max_elev_above_channel_m
            too_high = np.isfinite(elev) & (elev > ceiling)
            masked_count += int(too_high.sum())
            elev = np.where(too_high, np.nan, elev)

    # save to CSV
    df = pd.DataFrame({
        "distance_m": distances_centered,
        "elevation_m": elev
    })
    csv_path = os.path.join(output_profiles_folder, f"cross_section_{node_id:03d}.csv")
    df.to_csv(csv_path, index=False)

print(f"✅ Saved {len(cross_sections)} elevation profiles in '{output_profiles_folder}/'")
if max_elev_above_channel_m is not None:
    print(
        f"   Masked {masked_count} samples > {max_elev_above_channel_m:.0f} m "
        f"above near-channel elevation (±{channel_elev_window_m:.0f} m window)"
    )
