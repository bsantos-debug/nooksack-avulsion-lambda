"""Fill empty cells in the 2022 bathymetry corridor with upstream 2023 lidar.

The 2022 corridor stays on top wherever it already has an elevation. Only
nodata cells are taken from the upstream VRT. Elevations stay in US survey
feet. The unfilled corridor is not overwritten.

Do not point this at the full county mosaic. Warp into the existing 3 m
corridor grid so the county raster is never loaded into memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

NODATA = -9999.0


def _missing(values: np.ndarray, nodata: float) -> np.ndarray:
    return ~np.isfinite(values) | (values == nodata) | (values < -1000.0)


def fill_corridor(corridor: Path, upstream: Path, output: Path) -> None:
    if output.resolve() == corridor.resolve():
        raise SystemExit("Refusing to overwrite the unfilled corridor. Choose a new --output.")
    if output.exists():
        raise SystemExit(f"{output} already exists. Delete it first if you mean to rebuild it.")

    with rasterio.open(corridor) as base:
        kept = base.read(1).astype(np.float32)
        profile = base.profile.copy()
        transform = base.transform
        crs = base.crs
        nodata = float(base.nodata) if base.nodata is not None else NODATA

    incoming = np.full(kept.shape, nodata, dtype=np.float32)
    with rasterio.open(upstream) as src:
        print(f"Warping {upstream.name} ({src.width}×{src.height}) into the corridor grid")
        reproject(
            source=rasterio.band(src, 1),
            destination=incoming,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            src_nodata=src.nodata,
            dst_nodata=nodata,
            resampling=Resampling.average,
        )

    holes = _missing(kept, nodata)
    usable = ~_missing(incoming, nodata)
    filled = kept.copy()
    filled[holes & usable] = incoming[holes & usable]
    n_filled = int((holes & usable).sum())
    print(f"Filled {n_filled} empty cells ({100 * n_filled / filled.size:.1f}% of the grid)")
    print("Existing 2022 elevations were left unchanged.")

    profile.update(dtype="float32", nodata=nodata, compress="lzw", tiled=True, BIGTIFF="IF_SAFER")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(filled, 1)
        dst.update_tags(
            ELEVATION_UNITS="US survey feet",
            VERTICAL_DATUM="NAVD88",
            SOURCE=str(corridor),
            UPSTREAM_FILL=str(upstream),
            FILL_RULE="kept 2022 elevations where present; filled nodata from the upstream VRT",
        )
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corridor", type=Path, required=True, help="Unfilled 3 m UTM 2022 corridor")
    parser.add_argument("--upstream", type=Path, required=True, help="2023 lidar VRT behind the 2022 mosaic")
    parser.add_argument("--output", type=Path, required=True, help="New filled corridor GeoTIFF")
    args = parser.parse_args()
    fill_corridor(args.corridor, args.upstream, args.output)


if __name__ == "__main__":
    main()
