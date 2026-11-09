"""A short readme saved next to the label CSVs for each picking run."""

from __future__ import annotations

from pathlib import Path

from avulsionprecursors.config import WorkflowConfig

_NOTES_HEADING = "## Notes"
_UNIT_NAMES = {
    "us_ft": "US survey feet",
    "ft": "feet",
    "m": "metres",
    "native": "native DEM units",
}


def write_label_readme(cfg: WorkflowConfig) -> Path:
    """Write ``labels/README.md`` and keep any notes already under ``## Notes``."""
    path = cfg.labels_dir() / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = _notes_body(path.read_text() if path.exists() else "")
    path.write_text(_render(cfg, kept))
    return path


def _notes_body(text: str) -> str:
    if _NOTES_HEADING not in text:
        return ""
    return text.split(_NOTES_HEADING, 1)[1].strip()


def _render(cfg: WorkflowConfig, notes: str) -> str:
    ext = cfg.extract
    unit = _UNIT_NAMES.get(cfg.vertical_unit, cfg.vertical_unit)
    polygon = cfg.paths.channel_polygon
    lines = [
        f"# {cfg.study_name}",
        "",
        "Picks for this run are the CSV files in this folder.",
        "Elevations in those files are the DEM's own vertical units.",
        "",
        "## Setup",
        "",
        f"- Study: {cfg.study_name}",
        f"- DEM: {cfg.paths.dem}",
        f"- Centerline: {cfg.paths.centerline}",
        f"- Bankfull polygon: {polygon if polygon is not None else 'none'}",
        f"- Vertical units: {unit}",
        f"- Working CRS: {cfg.crs.target}",
        f"- Station spacing: {ext.spacing:g} (horizontal CRS units)",
        (
            "- Transects: perpendicular to the local centerline, "
            f"half-length {ext.cross_half_length:g} to {ext.max_half_length:g}, "
            f"stopping at a sustained rise of {ext.valley_wall_dz:g} {unit}"
        ),
        f"- Profile sample step: {ext.sample_step if ext.sample_step is not None else 'DEM cell size'}",
        "",
        "## What this run is",
        "",
        (cfg.labeler.note or "_No note was set in the config (`labeler.note`)._"),
        "",
        _NOTES_HEADING,
        "",
        notes
        or "_Anything else to remember about this session. This section is kept when the picker starts again._",
        "",
    ]
    return "\n".join(lines)
