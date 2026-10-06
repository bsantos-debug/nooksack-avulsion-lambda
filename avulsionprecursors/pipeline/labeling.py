"""Pipeline for cross-section labeling (file-based workflow)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from ..gui.config import GUIConfig
from ..gui.labeler import CrossSectionLabeler


def _label_file_has_labels(path: Path, required: Sequence[str]) -> bool:
    """True if the CSV has channel and at least one complete ridge+floodplain bank."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(path, comment="#")
    except Exception:
        return False
    if df.empty or "label" not in df.columns:
        return False
    have = set(df["label"].astype(str))
    if "channel" in required and "channel" not in have:
        return False
    wants_ridges = any(lab.startswith("ridge") for lab in required)
    if not wants_ridges:
        return all(lab in have for lab in required)
    bank1 = "ridge1" in have and "floodplain1" in have
    bank2 = "ridge2" in have and "floodplain2" in have
    return bank1 or bank2


def _existing_pick_distances(path: Path) -> Optional[dict]:
    """Map existing label names to dist_along for the faded overlay."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, comment="#")
    except Exception:
        return None
    if df.empty or "label" not in df.columns or "dist_along" not in df.columns:
        return None
    out = {}
    for _, row in df.iterrows():
        lab = str(row["label"])
        dist = row["dist_along"]
        if pd.notna(dist):
            out[lab] = float(dist)
    return out or None


class FileLabelingPipeline:
    """Label cross-sections generated from shapefiles + per-node profile CSVs."""

    def __init__(
        self,
        cross_sections_path: Path,
        profiles_dir: Path,
        river_name: str,
        output_dir: Optional[Path] = None,
        labels_dir: Optional[Path] = None,
        gui_config: Optional[GUIConfig] = None,
        dem_path: Optional[Path] = None,
        show_satellite: bool = True,
        show_dem: bool = True,
        overlay_path: Optional[Path] = None,
        levee_gdf: Optional[gpd.GeoDataFrame] = None,
        overlay_existing_picks: bool = True,
        levee_buffer_m: Optional[float] = None,
    ):
        self.cross_sections_path = Path(cross_sections_path)
        self.profiles_dir = Path(profiles_dir)
        self.river_name = river_name
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        self.labels_dir = Path(labels_dir) if labels_dir else self.output_dir / "labels"
        self.gui_config = gui_config or GUIConfig()
        self.dem_path = Path(dem_path) if dem_path else None
        self.show_satellite = show_satellite
        self.show_dem = show_dem
        self.overlay_existing_picks = overlay_existing_picks
        if levee_buffer_m is not None:
            self.gui_config.levee_buffer_m = float(levee_buffer_m)

        self.levee_gdf = levee_gdf
        if self.levee_gdf is None and overlay_path is not None:
            overlay = Path(overlay_path)
            if not overlay.exists():
                raise FileNotFoundError(f"Levee overlay not found: {overlay}")
            self.levee_gdf = gpd.read_file(overlay)
            print(
                f"🗺️  Levee overlay: {len(self.levee_gdf)} features from {overlay}"
            )

        if not self.cross_sections_path.exists():
            raise FileNotFoundError(f"Cross-section file not found: {self.cross_sections_path}")
        if not self.profiles_dir.exists():
            raise FileNotFoundError(f"Profiles directory not found: {self.profiles_dir}")

        self.cross_sections = gpd.read_file(self.cross_sections_path)
        if self.cross_sections.empty:
            raise ValueError(f"No cross sections found in {self.cross_sections_path}")

    def run(
        self,
        node_ids: Optional[Sequence[int]] = None,
        skip_existing: bool = True,
        overlay_existing_picks: Optional[bool] = None,
    ) -> None:
        """Launch the labeling tool for each cross-section."""
        show_old = (
            self.overlay_existing_picks
            if overlay_existing_picks is None
            else overlay_existing_picks
        )
        selection = self._filter_cross_sections(node_ids)
        required = list(self.gui_config.labels)
        pending = []
        n_skip = 0
        for _, row in selection.iterrows():
            node_id = int(row.get("node_id", row.name))
            output_file = self._label_output_path(node_id)
            if skip_existing and _label_file_has_labels(output_file, required):
                n_skip += 1
                continue
            pending.append((node_id, row, output_file))

        print(f"📍 Pick order: {' → '.join(required)}")
        if skip_existing and n_skip:
            print(
                f"⏭️  Skipping {n_skip} nodes that already have "
                f"{', '.join(required)}"
            )
        print(
            f"🎯 Opening {len(pending)} nodes"
            + (f": {[nid for nid, _, _ in pending]}" if pending else "")
        )
        if not pending:
            return

        for node_id, row, output_file in pending:
            line: LineString = row.geometry
            profile_path = self._profile_path(node_id)
            profile_gdf = self._load_profile(line, profile_path, node_id)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            labeler = CrossSectionLabeler(
                config=self.gui_config,
                output_dir=output_file.parent,
                river_name=f"{self.river_name}_node_{node_id}",
                dem_path=self.dem_path,
                show_satellite=self.show_satellite,
                show_dem=self.show_dem,
                node_id=node_id,
                levee_gdf=self.levee_gdf,
            )

            predicted = _existing_pick_distances(output_file) if show_old else None
            print(f"\nLabeling cross-section for node {node_id}")
            labeler.label_cross_section(profile_gdf, predicted_points=predicted)

    def _filter_cross_sections(self, node_ids: Optional[Sequence[int]]) -> gpd.GeoDataFrame:
        if node_ids is None:
            return self.cross_sections
        node_ids_set = {int(nid) for nid in node_ids}
        filtered = self.cross_sections[self.cross_sections["node_id"].astype(int).isin(node_ids_set)]
        if filtered.empty:
            raise ValueError(f"No cross sections found for node IDs: {node_ids}")
        return filtered

    def _profile_path(self, node_id: int) -> Path:
        candidates = [
            self.profiles_dir / f"cross_section_{node_id:03d}.csv",
            self.profiles_dir / f"{node_id}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Profile CSV not found for node {node_id}. "
            f"Checked: {', '.join(str(c) for c in candidates)}"
        )

    def _label_output_path(self, node_id: int) -> Path:
        return self.labels_dir / f"{self.river_name}_node_{node_id}_labels.csv"

    def _load_profile(self, line: LineString, profile_path: Path, node_id: int) -> gpd.GeoDataFrame:
        df = pd.read_csv(profile_path)
        dist_col = "distance" if "distance" in df.columns else "distance_m"
        elev_col = "elevation" if "elevation" in df.columns else "elevation_m"
        if dist_col not in df.columns or elev_col not in df.columns:
            raise ValueError(
                f"Expected columns 'distance'/'elevation' (or 'distance_m'/'elevation_m') in {profile_path}"
            )

        half_length = line.length / 2.0
        coords = list(line.coords)
        if len(coords) >= 3:
            s_channel = float(line.project(Point(coords[1])))
        else:
            s_channel = half_length
        distances = df[dist_col].values + s_channel

        def clamp(value: float) -> float:
            return min(max(value, 0.0), line.length)

        points = [line.interpolate(clamp(distance)) for distance in distances]

        return gpd.GeoDataFrame(
            {
                "node_id": node_id,
                "dist_along": distances,
                "distance_centered": df[dist_col],
                "elevation": df[elev_col],
            },
            geometry=points,
            crs=self.cross_sections.crs,
        )
