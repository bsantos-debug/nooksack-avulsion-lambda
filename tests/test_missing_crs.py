"""Create a DEM with no CRS and confirm a clear error."""

from __future__ import annotations

from pathlib import Path

import pytest
import rasterio
import yaml

from avulsionprecursors.config import load_config
from avulsionprecursors.exceptions import CRSError
from avulsionprecursors.io.prepare import prepare_inputs
from avulsionprecursors.io.synthetic import write_synthetic_valley


def test_dem_missing_crs_has_clear_error(tmp_path: Path):
    write_synthetic_valley(tmp_path)
    dem_path = tmp_path / "nocrs.tif"
    with rasterio.open(tmp_path / "dem.tif") as src:
        arr = src.read(1)
        meta = src.meta.copy()
    meta["crs"] = None
    with rasterio.open(dem_path, "w", **meta) as dst:
        dst.write(arr, 1)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "crs": {"target": "EPSG:32610"},
                "paths": {
                    "dem": "nocrs.tif",
                    "centerline": "centerline.shp",
                    "output_dir": "outputs",
                },
            }
        )
    )
    cfg = load_config(cfg_path)
    with pytest.raises(CRSError, match="no coordinate reference system"):
        prepare_inputs(cfg)
