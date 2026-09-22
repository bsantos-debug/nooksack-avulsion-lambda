from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from shapely.geometry import LineString, Point


@dataclass
class SwordNode:
    node_id: int
    reach_id: int
    dist_out: float
    width: Optional[float] = None
    slope: Optional[float] = None
    elevation: Optional[float] = None
    geometry: Optional[Point] = None
    cross_section: Optional[LineString] = None


@dataclass
class SwordReach:
    reach_id: int
    nodes: List[SwordNode] = field(default_factory=list)
