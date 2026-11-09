"""The picking run leaves a readme beside the label CSVs."""

from pathlib import Path

from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.pipeline.label_note import write_label_readme


def test_readme_records_the_note_and_keeps_extra_notes(tmp_path: Path):
    cfg = WorkflowConfig(study_name="demo")
    cfg.paths.output_dir = tmp_path
    cfg.paths.dem = Path("dem.tif")
    cfg.paths.centerline = Path("line.shp")
    cfg.labeler.note = "First pass on the lidar."

    path = write_label_readme(cfg)
    text = path.read_text()
    assert path == tmp_path / "labels" / "README.md"
    assert "First pass on the lidar." in text
    assert "dem.tif" in text

    path.write_text(
        text.replace(
            "_Anything else to remember about this session. This section is kept when the picker starts again._",
            "Skipped the oxbow.",
        )
    )
    cfg.labeler.note = "Second pass."
    write_label_readme(cfg)
    updated = path.read_text()
    assert "Second pass." in updated
    assert "Skipped the oxbow." in updated
    assert "First pass on the lidar." not in updated
