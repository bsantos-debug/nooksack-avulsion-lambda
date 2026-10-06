"""Interactive labeling of channel, ridges, and floodplains."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.exceptions import InputValidationError
from avulsionprecursors.gui.config import GUIConfig
from avulsionprecursors.pipeline.labeling import FileLabelingPipeline


def run_labeler(
    cfg: WorkflowConfig,
    node_ids: Optional[Sequence[int]] = None,
    skip_existing: Optional[bool] = None,
    overlay_existing: Optional[bool] = None,
) -> None:
    """Launch the GUI for each cross-section profile.

    Required inputs
    ---------------
    Cross-section shapefile and profile CSVs from the extract step.
    DEM is optional (background only).
    """
    xs = cfg.cross_sections_path()
    profiles = cfg.profiles_dir()
    if not xs.exists():
        raise InputValidationError(
            f"Cross-sections not found:\n  {xs}\nRun the extract step first."
        )
    if not profiles.exists():
        raise InputValidationError(
            f"Profiles folder not found:\n  {profiles}\nRun the extract step first."
        )

    dem_path = cfg.prepared_dem_path()
    if not dem_path.exists():
        dem_path = cfg.paths.dem if Path(cfg.paths.dem).exists() else None

    skip = cfg.labeler.skip_existing if skip_existing is None else skip_existing
    show_old = cfg.labeler.overlay_existing if overlay_existing is None else overlay_existing
    gui = GUIConfig()
    gui.levee_buffer_m = float(cfg.labeler.levee_buffer)
    pipeline = FileLabelingPipeline(
        cross_sections_path=xs,
        profiles_dir=profiles,
        river_name=cfg.study_name,
        output_dir=cfg.output_dir(),
        labels_dir=cfg.labels_dir(),
        gui_config=gui,
        dem_path=dem_path,
        show_satellite=cfg.labeler.show_satellite,
        show_dem=cfg.labeler.show_dem and dem_path is not None,
        overlay_path=cfg.paths.levee,
        overlay_existing_picks=show_old,
        levee_buffer_m=float(cfg.labeler.levee_buffer),
    )
    pipeline.run(node_ids=node_ids, skip_existing=skip)
