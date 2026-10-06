"""Build a river reach from a centerline-points shapefile."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from avulsionprecursors.geometry.nodes import RiverNode, RiverReach


def build_reach_from_shapefiles(
    centerline_path: str,
    points_path: str,
    reach_id: int = 1,
) -> RiverReach:
    """Build a RiverReach from a centerline points shapefile.

    ``centerline_path`` is accepted for API compatibility; node locations come
    from ``points_path``. Cross-section LineStrings are attached later.
    """
    _ = centerline_path
    pts = gpd.read_file(points_path)
    if pts.empty:
        raise ValueError(f"No points found in {points_path}")

    nodes: list[RiverNode] = []
    for idx, row in pts.iterrows():
        nid_raw = row.get("node_id", idx)
        try:
            nid = int(nid_raw)
        except (TypeError, ValueError):
            nid = int(idx) + 1

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        dist_out = float(row["dist_out"]) if "dist_out" in row and pd.notna(row.get("dist_out")) else 0.0
        width = float(row["width"]) if "width" in row and pd.notna(row.get("width")) else None
        slope = float(row["slope"]) if "slope" in row and pd.notna(row.get("slope")) else None
        elev = None
        for col in ("elevation", "elev", "z"):
            if col in row and pd.notna(row.get(col)):
                elev = float(row[col])
                break

        nodes.append(
            RiverNode(
                node_id=nid,
                reach_id=reach_id,
                dist_out=dist_out,
                width=width,
                slope=slope,
                elevation=elev,
                geometry=geom,
                cross_section=None,
            )
        )

    nodes.sort(key=lambda n: n.node_id)
    return RiverReach(reach_id=reach_id, nodes=nodes)
