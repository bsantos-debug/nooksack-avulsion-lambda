"""River node / reach containers used by extraction and lambda."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from shapely.geometry import LineString, Point


@dataclass
class RiverNode:
    """One station along a river centerline.

    Units
    -----
    dist_out : distance from outlet, working CRS horizontal units
    width : channel width, working CRS horizontal units
    slope : |dz/ds| using native DEM elevations over horizontal CRS units
    elevation : native DEM vertical units (optional)
    """

    node_id: int
    reach_id: int
    dist_out: float
    width: Optional[float] = None
    slope: Optional[float] = None
    elevation: Optional[float] = None
    geometry: Optional[Point] = None
    cross_section: Optional[LineString] = None


@dataclass
class RiverReach:
    """Ordered collection of river nodes."""

    reach_id: int
    nodes: List[RiverNode] = field(default_factory=list)


# Backwards-compatible names from the original SWORD-based pipeline
SwordNode = RiverNode
SwordReach = RiverReach
