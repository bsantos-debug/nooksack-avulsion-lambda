import geopandas as gpd
import rasterio
import numpy as np
import math
import pandas as pd
from shapely.geometry import LineString, Point

# === USER PARAMETERS ===
river_centerline_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Nooksack/Qgis/Crosssections/Modern_Nooksack_centerline_clean.shp"
centerline_points_path = None  # set to None to auto-sample with spacing parameter, or provide path to existing points shapefile
# centerline_points_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Nooksack/Qgis/Crosssections/Modern_Nooksack_Centerline_Points_400m.shp"  # commented out to use auto-sampling
channel_polygon_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Nooksack/Qgis/Crosssections/Modern_Nooksack_channel.shp"  # optional polygon delineating channel boundaries
dem_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/Washington/Nooksack/wa_dem/WA_DEM_m.tif"
spacing = 900        # distance between points along the river (m) when auto-sampling
cross_half_length = 400  # fallback half-length of each cross section (m) if width unavailable
sample_step = 1         # spacing of samples along each cross section (m)
output_cross_sections = "cross_sections.shp"
output_points = None  # Optional: path to save updated points with calculated widths (default: None = don't save)
output_profiles_folder = "cross_section_profiles"  # folder for CSV outputs

import os
os.makedirs(output_profiles_folder, exist_ok=True)

# === STEP 1: Load data ===
gdf = gpd.read_file(river_centerline_path)
if gdf.empty:
    raise ValueError(f"No features found in {river_centerline_path}")
river = gdf.geometry.iloc[0]  # assuming a single line feature
crs = gdf.crs

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

# === STEP 3: Calculate slopes from DEM along centerline ===
def _calculate_slope_from_dem(
    point: Point,
    centerline: LineString,
    dem: rasterio.DatasetReader,
    window_size: int = 3
) -> float:
    """
    Calculate slope from DEM along the centerline.
    
    Args:
        point: Point along centerline
        centerline: Centerline LineString
        dem: Opened rasterio DEM
        window_size: Number of points before/after to use for slope calculation
        
    Returns:
        Slope value (unitless: m/m) or NaN if calculation fails
    """
    try:
        # Get position along centerline
        s = centerline.project(point)
        line_length = centerline.length
        
        # Create points before and after for slope calculation
        # Use a reasonable spacing based on point density
        spacing = min(100.0, line_length / 20)  # ~100m or 1/20th of line length
        
        # Sample points before and after
        s_before = max(0, s - spacing * window_size)
        s_after = min(line_length, s + spacing * window_size)
        
        points_sample = []
        distances = []
        
        # Sample at regular intervals
        num_samples = window_size * 2 + 1
        for i in range(num_samples):
            s_i = s_before + (s_after - s_before) * i / (num_samples - 1) if num_samples > 1 else s
            p = centerline.interpolate(s_i)
            points_sample.append(p)
            distances.append(s_i)
        
        # Sample DEM at these points
        coords = [(p.x, p.y) for p in points_sample]
        elevations = []
        for coord in coords:
            try:
                elev = list(dem.sample([coord]))[0][0]
                if elev != dem.nodata and not np.isnan(elev):
                    elevations.append(elev)
                else:
                    return float("nan")
            except Exception:
                return float("nan")
        
        if len(elevations) < 2:
            return float("nan")
        
        # Calculate slope: change in elevation / change in distance
        # Distance along centerline
        dist_along = [d - distances[0] for d in distances]
        
        # Linear regression to get slope
        if len(elevations) >= 2:
            # Simple slope: (elev_end - elev_start) / (dist_end - dist_start)
            elev_diff = elevations[-1] - elevations[0]
            dist_diff = dist_along[-1] - dist_along[0]
            
            if abs(dist_diff) < 1e-6:  # Too close, use a minimum distance
                return float("nan")
            
            slope = elev_diff / dist_diff
            
            # Ensure slope is positive (downstream direction)
            # Negative slopes would indicate flow going uphill
            if slope < 0:
                slope = abs(slope)
            
            return float(slope)
        else:
            return float("nan")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not calculate slope for point: {e}")
        return float("nan")


# Calculate slopes from DEM if available
if dem_path and os.path.exists(dem_path):
    print(f"📐 Calculating slopes from DEM: {dem_path}")
    dem = rasterio.open(dem_path)
    
    slopes_calculated = 0
    for idx, pt in enumerate(points):
        # Only calculate if slope is not already provided
        if not _is_valid_slope(slopes[idx]):
            slope_val = _calculate_slope_from_dem(pt, river, dem)
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


def _construct_cross_section_line(pt: Point, perp_azimuth: float, half_length: float) -> LineString:
    """Create a LineString centered on pt with the given half length."""
    x1 = pt.x + half_length * math.cos(perp_azimuth)
    y1 = pt.y + half_length * math.sin(perp_azimuth)
    x2 = pt.x - half_length * math.cos(perp_azimuth)
    y2 = pt.y - half_length * math.sin(perp_azimuth)
    return LineString([(x1, y1), (x2, y2)])


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


# === STEP 4: Calculate local azimuths and perpendicular directions ===
perpendicular_data = []
for pt in points:
    s = river.project(pt)
    d = 0.01  # small step along the line
    p1 = river.interpolate(max(s - d, 0))
    p2 = river.interpolate(min(s + d, river.length))

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    azimuth = math.atan2(dy, dx)
    perp_azimuth = azimuth + math.pi / 2
    perp_azimuths.append(perp_azimuth)
    perpendicular_data.append((azimuth, perp_azimuth))


# === STEP 5: Update widths using channel polygon if provided ===
if channel_polygon_path and os.path.exists(channel_polygon_path):
    print(f"📏 Calculating widths from channel polygon: {channel_polygon_path}")
    channel_polys = gpd.read_file(channel_polygon_path).to_crs(crs)
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
# Make cross-sections 4x the channel width (2x width on each side of centerline)
# Falls back to cross_half_length if width is not available
cross_sections = []
for idx, pt in enumerate(points):
    perp_azimuth = perp_azimuths[idx]
    if _is_valid_width(widths[idx]):
        # Cross-section is 4x the width (2x width on each side)
        half_length = 3.5 * widths[idx]
    else:
        half_length = cross_half_length
    cross_sections.append(_construct_cross_section_line(pt, perp_azimuth, half_length))

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

for i, (line, node_id) in enumerate(zip(cross_sections, point_ids)):
    # sample points along the line
    num = int(line.length / sample_step)
    distances = np.linspace(0, line.length, num)
    points_along = [line.interpolate(d) for d in distances]

    # sample DEM elevations
    coords = [(p.x, p.y) for p in points_along]
    elev = list(dem.sample(coords))
    elev = [e[0] if e[0] != dem.nodata else np.nan for e in elev]

    # save to CSV
    df = pd.DataFrame({
        "distance_m": distances - line.length / 2,  # center = 0
        "elevation_m": elev
    })
    csv_path = os.path.join(output_profiles_folder, f"cross_section_{node_id:03d}.csv")
    df.to_csv(csv_path, index=False)

print(f"✅ Saved {len(cross_sections)} elevation profiles in '{output_profiles_folder}/'")
