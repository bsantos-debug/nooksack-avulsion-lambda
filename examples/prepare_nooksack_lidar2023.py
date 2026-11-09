"""Build the Nooksack centerline and a river-corridor copy of the lidar DEM.

Through the 2025 bankfull extent the centerline is the middle of that polygon:
the shortest path along its medial axis, smoothed so boundary notches do not
wiggle the line. Downstream of the extent it is the modern Nooksack centerline.
Transects are perpendicular to that path.

The source DEM is a ~10 GB Washington South (ftUS) mosaic. This writes a 3 m
UTM corridor so extraction does not have to reproject the whole county grid.
Elevations stay in US survey feet.
"""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.prepared import prep

from avulsionprecursors.geometry.transects import smooth_centerline_for_azimuth

ROOT = Path(__file__).resolve().parents[1]
MODERN = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Nooksack_research/"
    "Qgis/Crosssections/Modern_Nooksack_centerline_clean.shp"
)
BANKFULL = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/Data/Washington/Nooksack/"
    "Model_outputs/Bankfull_flow_extent_2025.shp"
)
DEM_SRC = Path("/Volumes/Extreme SSD/Nooksack_data/DEMs/nooksack_lidar2023_bathy2024.tif")
OUT_DIR = ROOT / "outputs" / "nooksack_lidar2023" / "inputs"
CENTERLINE_OUT = OUT_DIR / "centerline_bankfull_and_lower.shp"
DEM_OUT = Path(
    "/Volumes/Extreme SSD/Nooksack_data/DEMs/nooksack_lidar2023_corridor_utm10_3m.tif"
)

CORRIDOR_BUFFER_M = 4200.0
CORRIDOR_RES_M = 3.0
# Boundary sample used to build the medial axis. Edges closer to the bank than
# this are corner spurs, not the channel middle.
MEDIAL_STEP_M = 20.0
MEDIAL_MIN_CLEARANCE_M = 15.0
MEDIAL_SMOOTH_M = 400.0


def _medial_axis(polygon):
    """Shortest interior path along the polygon's medial axis.

    The modern thalweg is the guide for which end is upstream. The axis itself
    does not follow that line, so it stays in the middle of wide bends.
    """
    ring = LineString(polygon.exterior.coords)
    sites = np.array(
        [
            ring.interpolate(float(s)).coords[0]
            for s in np.arange(0.0, ring.length, MEDIAL_STEP_M)
        ]
    )
    vor = Voronoi(sites)
    prepared = prep(polygon)
    minx, miny, maxx, maxy = polygon.bounds
    inside = np.zeros(len(vor.vertices), dtype=bool)
    for i, (x, y) in enumerate(vor.vertices):
        if minx <= x <= maxx and miny <= y <= maxy and prepared.covers(Point(x, y)):
            inside[i] = True

    boundary = polygon.boundary
    graph: dict[int, list[tuple[int, float]]] = {}
    for (a, b) in vor.ridge_vertices:
        if a < 0 or b < 0 or not inside[a] or not inside[b]:
            continue
        seg = LineString([vor.vertices[a], vor.vertices[b]])
        if not prepared.covers(seg):
            continue
        if seg.interpolate(0.5, normalized=True).distance(boundary) < MEDIAL_MIN_CLEARANCE_M:
            continue
        weight = float(seg.length)
        graph.setdefault(a, []).append((b, weight))
        graph.setdefault(b, []).append((a, weight))
    if not graph:
        raise RuntimeError("Bankfull polygon has no interior medial axis.")
    return graph, vor.vertices


def _shortest_path(graph, vertices, start: int, goal: int) -> LineString:
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    queue = [(0.0, start)]
    while queue:
        travelled, node = heapq.heappop(queue)
        if node == goal:
            break
        if travelled != dist.get(node):
            continue
        for nxt, weight in graph.get(node, []):
            cand = travelled + weight
            if cand < dist.get(nxt, 1e300):
                dist[nxt] = cand
                prev[nxt] = node
                heapq.heappush(queue, (cand, nxt))
    if goal not in dist:
        raise RuntimeError("No path through the middle of the bankfull polygon.")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return LineString(vertices[path])


def _nearest_node(graph, vertices, point: Point) -> int:
    ids = np.fromiter(graph.keys(), dtype=int)
    xy = vertices[ids]
    d2 = (xy[:, 0] - point.x) ** 2 + (xy[:, 1] - point.y) ** 2
    return int(ids[int(np.argmin(d2))])


def build_centerline(modern_path: Path, bankfull_path: Path) -> LineString:
    modern = gpd.read_file(modern_path)
    bankfull = gpd.read_file(bankfull_path)
    if modern.crs != bankfull.crs:
        bankfull = bankfull.to_crs(modern.crs)
    thalweg = modern.geometry.iloc[0]
    polygon = bankfull.geometry.iloc[0]
    print(f"Modern centerline {thalweg.length/1000:.2f} km, {len(thalweg.coords)} vertices")

    chain = np.arange(0.0, thalweg.length + 1e-6, 25.0)
    inside = np.array(
        [polygon.covers(thalweg.interpolate(float(s))) for s in chain],
        dtype=bool,
    )
    if not inside.any():
        raise RuntimeError("The modern centerline does not meet the bankfull polygon.")
    entry = float(chain[int(np.argmax(inside))])
    exit_s = float(chain[int(len(inside) - 1 - np.argmax(inside[::-1]))])
    print(f"Bankfull coverage along the modern line: {entry/1000:.2f}–{exit_s/1000:.2f} km")
    print(f"Upstream of the polygon (not used): {entry/1000:.2f} km")
    print(f"Downstream of the polygon: {(thalweg.length - exit_s)/1000:.2f} km")

    graph, vertices = _medial_axis(polygon)
    start = _nearest_node(graph, vertices, thalweg.interpolate(entry))
    goal = _nearest_node(graph, vertices, thalweg.interpolate(exit_s))
    axis = smooth_centerline_for_azimuth(
        _shortest_path(graph, vertices, start, goal),
        MEDIAL_SMOOTH_M,
        MEDIAL_STEP_M,
    )
    if not axis.is_simple:
        raise RuntimeError("Bankfull centerline crosses itself.")
    inside_axis = sum(
        polygon.covers(axis.interpolate(float(d)))
        for d in np.linspace(0.0, axis.length, 80)
    )
    print(
        f"Bankfull middle {axis.length/1000:.2f} km, "
        f"{inside_axis}/80 samples inside the polygon, simple={axis.is_simple}"
    )
    if inside_axis < 70:
        raise RuntimeError("The centerline left the bankfull polygon.")

    downstream = substring(thalweg, exit_s, thalweg.length)
    coords = list(axis.coords)
    down_coords = list(downstream.coords)
    if Point(coords[-1]).distance(Point(down_coords[0])) < 1.0:
        merged = coords + down_coords[1:]
    else:
        merged = coords + down_coords
    line = LineString(merged)
    if not line.is_simple:
        raise RuntimeError("Composite centerline crosses itself.")
    print(f"Composite centerline {line.length/1000:.2f} km, simple={line.is_simple}")
    return line


def write_corridor(line: LineString, dem_src: Path, dem_out: Path) -> None:
    corridor = line.buffer(CORRIDOR_BUFFER_M)
    minx, miny, maxx, maxy = corridor.bounds
    width = int(math.ceil((maxx - minx) / CORRIDOR_RES_M))
    height = int(math.ceil((maxy - miny) / CORRIDOR_RES_M))
    print(
        f"Corridor grid {width} × {height} at {CORRIDOR_RES_M:g} m "
        f"({width * height / 1e6:.1f} million pixels)"
    )
    if width * height > 2.5e8:
        raise RuntimeError("Corridor raster is too large to build in memory.")
    transform = from_origin(minx, maxy, CORRIDOR_RES_M, CORRIDOR_RES_M)
    destination = np.full((height, width), -9999.0, dtype=np.float32)
    dem_out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_src) as src:
        print(f"Warping {src.crs} {src.width}×{src.height} → EPSG:26910")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:26910",
            src_nodata=src.nodata,
            dst_nodata=-9999.0,
            resampling=Resampling.average,
        )
    finite = np.isfinite(destination) & (destination != -9999.0)
    print(
        f"Corridor elevations (ft): min {np.min(destination[finite]):.1f} "
        f"max {np.max(destination[finite]):.1f} "
        f"finite {100 * finite.mean():.1f}%"
    )
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:26910",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(dem_out, "w", **profile) as dst:
        dst.write(destination, 1)
        dst.update_tags(
            ELEVATION_UNITS="US survey feet",
            VERTICAL_DATUM="NAVD88",
            SOURCE=str(dem_src),
        )
    print(f"Wrote {dem_out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bankfull medial-axis centerline and an optional 3 m UTM corridor."
    )
    parser.add_argument("--modern", type=Path, default=MODERN, help="Modern thalweg shapefile")
    parser.add_argument("--bankfull", type=Path, default=BANKFULL, help="Bankfull polygon shapefile")
    parser.add_argument("--dem", type=Path, default=DEM_SRC, help="Source DEM to warp into the corridor")
    parser.add_argument("--centerline-out", type=Path, default=CENTERLINE_OUT)
    parser.add_argument("--dem-out", type=Path, default=DEM_OUT)
    parser.add_argument(
        "--centerline-only",
        action="store_true",
        help="Write the centerline and skip the DEM warp",
    )
    args = parser.parse_args()
    args.centerline_out.parent.mkdir(parents=True, exist_ok=True)
    line = build_centerline(args.modern, args.bankfull)
    gpd.GeoDataFrame({"id": [1]}, geometry=[line], crs="EPSG:26910").to_file(args.centerline_out)
    print(f"Wrote {args.centerline_out}")
    if args.centerline_only:
        return
    write_corridor(line, args.dem, args.dem_out)


if __name__ == "__main__":
    main()
