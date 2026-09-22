#!/usr/bin/env python3
"""Build a 2013 ground DTM from NOAA 2615 COPC tiles and compare ridge/levee change.

2013 is topographic lidar only (no channel bathymetry). Primary metric is
elevation at the 2024 alluvial-ridge / levee crest XY, with floodplain
samples as a vertical-datum control (Geoid18 metres vs later NAVD88 feet).

Inputs:
  COPC tiles in LAZ_DIR (EPSG:6339, NAVD88 Geoid18 metres, class 2 = ground)
  2024 topobathy + existing cross-section profiles
  2022 NOAA DTM VRT (optional; sampled at the same crest XY)

Outputs:
  data/2013_dtm/nooksack_2013_ground_1m_epsg26910.tif
  data/Nooksack_2013_2022_2024_profile_ridges.csv
  data/Nooksack_2013_vs_2024_ridge_change_summary.json
  data/Nooksack_node_73_2013_2022_2024_xs.{png,pdf,svg}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
XS_PATH = ROOT / "cross_sections.shp"
LAMBDA_CSV = ROOT / "data" / "Nooksack_lambda_results.csv"
PROFILES_DIR = ROOT / "cross_section_profiles"
CENTERLINE = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack_data/GIS/Centerline_flowaccumulation/"
    "Centerline_flowaccumulation.shp"
)
DEM_2024 = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
)
DEM_2022_VRT = ROOT / "data" / "2022_dtm" / "WA_Nooksack_DEM_2022_m9574_EPSG-2927.vrt"
LAZ_DIR = Path(os.environ.get("NOOKSACK_2013_LAZ", "/tmp/nooksack_2013_laz"))
DTM_DIR = ROOT / "data" / "2013_dtm"
TILE_DIR = DTM_DIR / "tiles"
VRT_PATH = DTM_DIR / "nooksack_2013_ground_1m_epsg26910.vrt"
DTM_PATH = VRT_PATH
OUT_CSV = ROOT / "data" / "Nooksack_2013_2022_2024_profile_ridges.csv"
OUT_JSON = ROOT / "data" / "Nooksack_2013_vs_2024_ridge_change_summary.json"

US_FT = 0.304800609601219
RES_M = 1.0
BUFFER_M = 600.0
NODATA = -9999.0
TRUNCATED_LAZ: set[str] = set()


def xy_on_line(line: LineString, dist: float) -> tuple[float, float]:
    d = float(np.clip(dist, 0.0, float(line.length)))
    p = line.interpolate(d)
    return float(p.x), float(p.y)


def sample_nearest(src: rasterio.io.DatasetReader, coords: list[tuple[float, float]]) -> np.ndarray:
    nodata = src.nodata
    vals = []
    for v in src.sample(coords):
        z = float(v[0])
        if not np.isfinite(z):
            vals.append(np.nan)
            continue
        if nodata is not None and (z == nodata or abs(z - float(nodata)) < 1e-3):
            vals.append(np.nan)
            continue
        if z < -1e20:
            vals.append(np.nan)
            continue
        vals.append(z)
    return np.asarray(vals, dtype=float)


def pick_from_2024_profile(node_id: int, line: LineString) -> dict[str, float] | None:
    path = PROFILES_DIR / f"cross_section_{int(node_id):03d}.csv"
    if not path.exists():
        return None
    prof = pd.read_csv(path)
    if "distance_m" not in prof.columns or "elevation_m" not in prof.columns:
        return None
    d = prof["distance_m"].to_numpy(dtype=float)
    z = pd.to_numeric(prof["elevation_m"], errors="coerce").to_numpy(dtype=float)
    near = np.isfinite(z) & (np.abs(d) <= 120.0)
    if not np.any(near):
        near = np.isfinite(z) & (np.abs(d) <= 250.0)
    if not np.any(near):
        return None
    i_ch = np.where(near)[0][int(np.argmin(z[near]))]
    d_ch = float(d[i_ch])
    z_ch = float(z[i_ch])

    def side_crest(mask: np.ndarray) -> tuple[float, float]:
        if not np.any(mask):
            return np.nan, np.nan
        i = np.where(mask)[0][int(np.argmax(z[mask]))]
        return float(d[i]), float(z[i])

    def side_fp(d0: float, d1: float) -> tuple[float, float]:
        m = np.isfinite(z) & (d >= min(d0, d1)) & (d <= max(d0, d1))
        if not np.any(m):
            return np.nan, np.nan
        i = np.where(m)[0][int(np.argmin(np.abs(z[m] - np.nanpercentile(z[m], 25))))]
        return float(d[i]), float(z[i])

    d_l, z_l = side_crest(np.isfinite(z) & (d < d_ch) & (d >= d_ch - 350.0))
    d_r, z_r = side_crest(np.isfinite(z) & (d > d_ch) & (d <= d_ch + 350.0))
    d_fl, z_fl = side_fp(d_ch - 300.0, d_ch - 120.0) if np.isfinite(d_l) else (np.nan, np.nan)
    d_fr, z_fr = side_fp(d_ch + 120.0, d_ch + 300.0) if np.isfinite(d_r) else (np.nan, np.nan)
    mid = 0.5 * float(line.length)
    return {
        "channel_dist_along": d_ch + mid,
        "channel_z24_ft": z_ch,
        "ridge1_dist_along": d_l + mid if np.isfinite(d_l) else np.nan,
        "ridge1_z24_ft": z_l,
        "ridge2_dist_along": d_r + mid if np.isfinite(d_r) else np.nan,
        "ridge2_z24_ft": z_r,
        "floodplain1_dist_along": d_fl + mid if np.isfinite(d_fl) else np.nan,
        "floodplain1_z24_ft": z_fl,
        "floodplain2_dist_along": d_fr + mid if np.isfinite(d_fr) else np.nan,
        "floodplain2_z24_ft": z_fr,
    }


def summarize(a: np.ndarray) -> dict[str, float]:
    x = a[np.isfinite(a)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan, "std": np.nan}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
    }


def km_bin(dist_out_m: float) -> str:
    km = dist_out_m / 1000.0
    lo = int(np.floor(km / 5.0) * 5)
    return f"{lo}–{lo + 5}"


def rasterize_laz_tile(path: Path, transformer: Transformer) -> Path | None:
    """Mean class-2 ground DTM for one COPC tile; writes a small GeoTIFF."""
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    out = TILE_DIR / (path.stem.replace(".copc", "") + "_ground_1m.tif")
    if out.exists() and out.stat().st_size > 10_000:
        return out
    las = laspy.read(path)
    cls = np.asarray(las.classification)
    m = cls == 2
    if not np.any(m):
        return None
    x, y, z = np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]
    x2, y2 = transformer.transform(x, y)
    del las
    ok = np.isfinite(x2) & np.isfinite(y2) & np.isfinite(z)
    x2, y2, z = x2[ok], y2[ok], z[ok]
    if z.size == 0:
        return None
    minx = np.floor(x2.min() / RES_M) * RES_M
    maxy = np.ceil(y2.max() / RES_M) * RES_M
    maxx = np.ceil(x2.max() / RES_M) * RES_M
    miny = np.floor(y2.min() / RES_M) * RES_M
    width = int(np.round((maxx - minx) / RES_M))
    height = int(np.round((maxy - miny) / RES_M))
    if width < 1 or height < 1:
        return None
    zsum = np.zeros((height, width), dtype=np.float64)
    zcnt = np.zeros((height, width), dtype=np.uint16)
    ix = np.floor((x2 - minx) / RES_M).astype(np.int32)
    iy = np.floor((maxy - y2) / RES_M).astype(np.int32)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    np.add.at(zsum, (iy[inside], ix[inside]), z[inside])
    np.add.at(zcnt, (iy[inside], ix[inside]), 1)
    occupied = zcnt > 0
    grid = np.full((height, width), NODATA, dtype=np.float32)
    grid[occupied] = (zsum[occupied] / zcnt[occupied]).astype(np.float32)
    transform = from_origin(minx, maxy, RES_M, RES_M)
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:26910",
        transform=transform,
        nodata=NODATA,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(grid, 1)
    return out


def build_vrt(tile_paths: list[Path]) -> Path:
    import subprocess

    DTM_DIR.mkdir(parents=True, exist_ok=True)
    lst = DTM_DIR / "tiles.txt"
    lst.write_text("\n".join(str(p) for p in tile_paths) + "\n", encoding="utf-8")
    subprocess.run(
        ["gdalbuildvrt", "-overwrite", "-srcnodata", str(NODATA), str(VRT_PATH), "-input_file_list", str(lst)],
        check=True,
    )
    return VRT_PATH


def build_dtm() -> Path:
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:6339", "EPSG:26910", always_xy=True)

    # Drop truncated downloads so they can be re-fetched after space is freed.
    for name in TRUNCATED_LAZ:
        p = LAZ_DIR / name
        if p.exists():
            print(f"Removing truncated {name} ({p.stat().st_size/1e6:.1f} MB)")
            p.unlink()

    laz_files = sorted(LAZ_DIR.glob("*.copc.laz")) + sorted(LAZ_DIR.glob("*.laz"))
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in laz_files:
        if p.name in seen or p.stat().st_size < 50_000:
            continue
        seen.add(p.name)
        uniq.append(p)
    print(f"Rasterizing {len(uniq)} 2013 COPC tiles from {LAZ_DIR}")

    tile_paths: list[Path] = []
    for i, path in enumerate(uniq, start=1):
        try:
            tif = rasterize_laz_tile(path, transformer)
        except Exception as e:
            print(f"  [{i}/{len(uniq)}] {path.name}: ERROR {e}")
            continue
        if tif is None:
            print(f"  [{i}/{len(uniq)}] {path.name}: no ground")
        else:
            print(f"  [{i}/{len(uniq)}] {path.name} -> {tif.name} ({tif.stat().st_size/1e6:.2f} MB)")
            tile_paths.append(tif)
            try:
                path.unlink()
            except OSError:
                pass

    # Already-rasterized tiles from a previous partial run
    for tif in sorted(TILE_DIR.glob("*_ground_1m.tif")):
        if tif not in tile_paths and tif.stat().st_size > 10_000:
            tile_paths.append(tif)

    if not tile_paths:
        raise SystemExit("No 2013 ground tiles were rasterized")
    vrt = build_vrt(sorted(set(tile_paths)))
    print(f"Wrote {vrt} from {len(set(tile_paths))} tiles")
    return vrt


def compare_and_plot(dtm_path: Path) -> None:
    xs = gpd.read_file(XS_PATH)
    if xs.crs is None or str(xs.crs) != "EPSG:26910":
        xs = xs.to_crs(26910)
    xs["node_id"] = xs["node_id"].astype(int)
    xs = xs.set_index("node_id", drop=False)

    lam = pd.read_csv(LAMBDA_CSV)
    lam["node_id"] = lam["node_id"].astype(int)
    lam_idx = lam.set_index("node_id")

    to_2927 = Transformer.from_crs("EPSG:26910", "EPSG:2927", always_xy=True)

    keys = (
        ("channel", "channel_dist_along"),
        ("ridge1", "ridge1_dist_along"),
        ("ridge2", "ridge2_dist_along"),
        ("floodplain1", "floodplain1_dist_along"),
        ("floodplain2", "floodplain2_dist_along"),
    )
    z24_keys = {
        "channel": "channel_z24_ft",
        "ridge1": "ridge1_z24_ft",
        "ridge2": "ridge2_z24_ft",
        "floodplain1": "floodplain1_z24_ft",
        "floodplain2": "floodplain2_z24_ft",
    }

    rows = []
    with rasterio.open(dtm_path) as dem13, rasterio.open(DEM_2022_VRT) as dem22:
        print("2013 DTM CRS", dem13.crs, "2022 CRS", dem22.crs)
        for node_id, rec in xs.iterrows():
            line = rec.geometry
            if line is None or line.is_empty:
                continue
            picks = pick_from_2024_profile(int(node_id), line)
            if not picks:
                continue
            names, coords, dists = [], [], []
            for name, dist_key in keys:
                dist = picks[dist_key]
                if not np.isfinite(dist):
                    continue
                names.append(name)
                coords.append(xy_on_line(line, dist))
                dists.append(dist)
            if not coords:
                continue
            z13 = sample_nearest(dem13, coords)
            xs22, ys22 = to_2927.transform([c[0] for c in coords], [c[1] for c in coords])
            z22_ft = sample_nearest(dem22, list(zip(xs22, ys22)))
            sampled13 = {n: float(z) for n, z in zip(names, z13)}
            sampled22 = {n: float(z) * US_FT for n, z in zip(names, z22_ft)}
            dist_out = float(rec["dist_out"]) if pd.notna(rec.get("dist_out")) else np.nan
            lambda_val = (
                float(lam_idx.loc[int(node_id), "lambda"])
                if int(node_id) in lam_idx.index
                else np.nan
            )

            def zm(name: str) -> tuple[float, float, float]:
                z24 = picks[z24_keys[name]] * US_FT if np.isfinite(picks[z24_keys[name]]) else np.nan
                return sampled13.get(name, np.nan), sampled22.get(name, np.nan), z24

            z13_r1, z22_r1, z24_r1 = zm("ridge1")
            z13_r2, z22_r2, z24_r2 = zm("ridge2")
            z13_ch, z22_ch, z24_ch = zm("channel")
            z13_f1, z22_f1, z24_f1 = zm("floodplain1")
            z13_f2, z22_f2, z24_f2 = zm("floodplain2")
            dz13_24_r1 = z24_r1 - z13_r1
            dz13_24_r2 = z24_r2 - z13_r2
            dz13_22_r1 = z22_r1 - z13_r1
            dz13_22_r2 = z22_r2 - z13_r2
            dz22_24_r1 = z24_r1 - z22_r1
            dz22_24_r2 = z24_r2 - z22_r2
            rows.append(
                {
                    "node_id": int(node_id),
                    "dist_out_m": dist_out,
                    "lambda": lambda_val,
                    "z13_ridge1_m": z13_r1,
                    "z22_ridge1_m": z22_r1,
                    "z24_ridge1_m": z24_r1,
                    "z13_ridge2_m": z13_r2,
                    "z22_ridge2_m": z22_r2,
                    "z24_ridge2_m": z24_r2,
                    "z13_channel_m": z13_ch,
                    "z22_channel_m": z22_ch,
                    "z24_channel_m": z24_ch,
                    "z13_floodplain1_m": z13_f1,
                    "z22_floodplain1_m": z22_f1,
                    "z24_floodplain1_m": z24_f1,
                    "z13_floodplain2_m": z13_f2,
                    "z22_floodplain2_m": z22_f2,
                    "z24_floodplain2_m": z24_f2,
                    "dz13_24_ridge1_m": dz13_24_r1,
                    "dz13_24_ridge2_m": dz13_24_r2,
                    "dz13_24_ridge_mean_m": np.nanmean([dz13_24_r1, dz13_24_r2]),
                    "dz13_22_ridge_mean_m": np.nanmean([dz13_22_r1, dz13_22_r2]),
                    "dz22_24_ridge_mean_m": np.nanmean([dz22_24_r1, dz22_24_r2]),
                    "dz13_24_lower_ridge_m": np.nanmin([z24_r1, z24_r2]) - np.nanmin([z13_r1, z13_r2]),
                    "dz13_24_floodplain_mean_m": np.nanmean([z24_f1 - z13_f1, z24_f2 - z13_f2]),
                    "dz13_24_channel_m": z24_ch - z13_ch,
                }
            )

    df = pd.DataFrame(rows)
    df["km_bin"] = df["dist_out_m"].apply(lambda v: km_bin(v) if np.isfinite(v) else "NA")
    df["has_2013_ridge"] = df["dz13_24_ridge_mean_m"].notna()
    pr = df[df["has_2013_ridge"]].copy()
    bias = float(np.nanmedian(pr["dz13_24_floodplain_mean_m"]))
    df["dz13_24_ridge_mean_biascorr_m"] = df["dz13_24_ridge_mean_m"] - bias
    df["dz13_24_lower_ridge_biascorr_m"] = df["dz13_24_lower_ridge_m"] - bias
    pr = df[df["has_2013_ridge"]].copy()

    print(f"Nodes with 2013 ridge sample: {len(pr)} / {len(df)}")
    print(f"Median floodplain Δz 2024−2013 (geoid/control): {bias:+.3f} m")
    for col, label in [
        ("dz13_24_ridge_mean_m", "ridge 2013→2024 (same XY)"),
        ("dz13_24_ridge_mean_biascorr_m", "ridge 2013→2024 floodplain-corrected"),
        ("dz13_22_ridge_mean_m", "ridge 2013→2022"),
        ("dz22_24_ridge_mean_m", "ridge 2022→2024"),
        ("dz13_24_lower_ridge_m", "lower crest 2013→2024"),
        ("dz13_24_floodplain_mean_m", "floodplain 2013→2024"),
        ("dz13_24_channel_m", "channel XY 2013→2024 (2013 is water/ground, not bathy)"),
    ]:
        s = summarize(pr[col].to_numpy())
        print(
            f"{label:48s}  n={s['n']:3d}  median={s['median']:+.3f} m  "
            f"mean={s['mean']:+.3f}  p10={s['p10']:+.3f}  p90={s['p90']:+.3f}"
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("dist_out_m", ascending=False).to_csv(OUT_CSV, index=False)

    bin_order = [
        "55–60", "50–55", "45–50", "40–45", "35–40", "30–35",
        "25–30", "20–25", "15–20", "10–15", "5–10", "0–5",
    ]

    def med(sub: pd.DataFrame, col: str):
        x = sub[col].dropna()
        return None if x.empty else round(float(x.median()), 3)

    km_bins = []
    for b in bin_order:
        sub = pr[pr["km_bin"] == b]
        if sub.empty:
            continue
        km_bins.append(
            {
                "bin": b,
                "n": int(len(sub)),
                "median_dz13_24_ridge_m": med(sub, "dz13_24_ridge_mean_m"),
                "median_dz13_24_ridge_biascorr_m": med(sub, "dz13_24_ridge_mean_biascorr_m"),
                "median_dz13_22_ridge_m": med(sub, "dz13_22_ridge_mean_m"),
                "median_dz22_24_ridge_m": med(sub, "dz22_24_ridge_mean_m"),
                "median_dz13_24_floodplain_m": med(sub, "dz13_24_floodplain_mean_m"),
            }
        )

    def table_rows(frame: pd.DataFrame) -> list[dict]:
        out = []
        for _, r in frame.iterrows():
            out.append(
                {
                    "node_id": int(r["node_id"]),
                    "dist_out_km": round(float(r["dist_out_m"]) / 1000.0, 2),
                    "dz13_24_ridge_m": round(float(r["dz13_24_ridge_mean_m"]), 2),
                    "dz13_24_ridge_biascorr_m": None
                    if not np.isfinite(r["dz13_24_ridge_mean_biascorr_m"])
                    else round(float(r["dz13_24_ridge_mean_biascorr_m"]), 2),
                    "dz13_22_ridge_m": None
                    if not np.isfinite(r["dz13_22_ridge_mean_m"])
                    else round(float(r["dz13_22_ridge_mean_m"]), 2),
                    "dz22_24_ridge_m": None
                    if not np.isfinite(r["dz22_24_ridge_mean_m"])
                    else round(float(r["dz22_24_ridge_mean_m"]), 2),
                    "lambda": None if not np.isfinite(r["lambda"]) else round(float(r["lambda"]), 2),
                }
            )
        return out

    def clean(d: dict) -> dict:
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}

    ranked = pr.dropna(subset=["dz13_24_ridge_mean_m"])
    node73 = df[df["node_id"] == 73]
    node73_rec = None
    if not node73.empty:
        r = node73.iloc[0]
        node73_rec = {
            "dist_out_km": round(float(r["dist_out_m"]) / 1000.0, 2),
            "left_2013_m": None if not np.isfinite(r["z13_ridge1_m"]) else round(float(r["z13_ridge1_m"]), 2),
            "left_2022_m": None if not np.isfinite(r["z22_ridge1_m"]) else round(float(r["z22_ridge1_m"]), 2),
            "left_2024_m": None if not np.isfinite(r["z24_ridge1_m"]) else round(float(r["z24_ridge1_m"]), 2),
            "right_2013_m": None if not np.isfinite(r["z13_ridge2_m"]) else round(float(r["z13_ridge2_m"]), 2),
            "right_2022_m": None if not np.isfinite(r["z22_ridge2_m"]) else round(float(r["z22_ridge2_m"]), 2),
            "right_2024_m": None if not np.isfinite(r["z24_ridge2_m"]) else round(float(r["z24_ridge2_m"]), 2),
            "dz13_24_ridge_m": None
            if not np.isfinite(r["dz13_24_ridge_mean_m"])
            else round(float(r["dz13_24_ridge_mean_m"]), 2),
        }

    summary = {
        "n_xs": int(len(xs)),
        "n_with_2013_ridge": int(len(pr)),
        "floodplain_bias_2024_minus_2013_m": None if not np.isfinite(bias) else round(bias, 4),
        "stats_2013_to_2024": {
            "ridge": clean(summarize(pr["dz13_24_ridge_mean_m"].to_numpy())),
            "ridge_biascorr": clean(summarize(pr["dz13_24_ridge_mean_biascorr_m"].to_numpy())),
            "lower_ridge": clean(summarize(pr["dz13_24_lower_ridge_m"].to_numpy())),
            "floodplain": clean(summarize(pr["dz13_24_floodplain_mean_m"].to_numpy())),
        },
        "stats_2013_to_2022": {
            "ridge": clean(summarize(pr["dz13_22_ridge_mean_m"].to_numpy())),
        },
        "stats_2022_to_2024": {
            "ridge": clean(summarize(pr["dz22_24_ridge_mean_m"].to_numpy())),
        },
        "pct_ridge_aggrade_gt_5cm_raw": round(float(100.0 * np.nanmean(pr["dz13_24_ridge_mean_m"] > 0.05)), 1),
        "pct_ridge_aggrade_gt_5cm_biascorr": round(
            float(100.0 * np.nanmean(pr["dz13_24_ridge_mean_biascorr_m"] > 0.05)), 1
        ),
        "pct_ridge_degrade_gt_5cm_biascorr": round(
            float(100.0 * np.nanmean(pr["dz13_24_ridge_mean_biascorr_m"] < -0.05)), 1
        ),
        "km_bins": km_bins,
        "top_aggrade": table_rows(ranked.nlargest(8, "dz13_24_ridge_mean_biascorr_m")),
        "top_degrade": table_rows(ranked.nsmallest(8, "dz13_24_ridge_mean_biascorr_m")),
        "node_73": node73_rec,
        "notes": {
            "dem_2013": "NOAA 2615 PSLC Nooksack 2013, class 2 ground, NAVD88 Geoid18 metres, 1 m mean DTM in EPSG:26910",
            "dem_2022": "NOAA 9574 Feb 2022 DTM, NAVD88 Geoid12B ftUS, EPSG:2927",
            "dem_2024": "Topobathy_reprojected.tif values are NAVD88 feet; XY EPSG:26910",
            "positive_dz": "later year higher (net aggradation at that XY)",
            "ridge_definition": "2024 thalweg = min within 120 m of midpoint; crests = max within 350 m left/right; 2013/2022 sampled at those XY",
            "no_2013_bathy": "Do not interpret channel Δz; 2013 returns are ground/water surface in the wet channel",
            "biascorr": "subtract median floodplain 2024−2013 so Geoid18 vs later NAVD88 offset is not counted as levee growth",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    plot_node_73(xs, dtm_path)


def plot_node_73(xs: gpd.GeoDataFrame, dtm_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
    import matplotlib.pyplot as plt

    NODE = 73
    ZOOM_M = 400.0
    rec = xs.loc[xs["node_id"] == NODE].iloc[0]
    line = rec.geometry
    length = float(line.length)
    mid = 0.5 * length
    dist_out_km = float(rec["dist_out"]) / 1000.0 if pd.notna(rec.get("dist_out")) else np.nan

    prof = pd.read_csv(PROFILES_DIR / f"cross_section_{NODE:03d}.csv")
    d_c = prof["distance_m"].to_numpy(dtype=float)
    z24 = pd.to_numeric(prof["elevation_m"], errors="coerce").to_numpy(dtype=float) * US_FT

    picks = pick_from_2024_profile(NODE, line)
    d_ch = picks["channel_dist_along"] - mid
    step = 1.0
    dists_along = np.arange(0.0, length + step, step)
    d_from_thalweg = dists_along - picks["channel_dist_along"]
    coords = [xy_on_line(line, d) for d in dists_along]
    to_2927 = Transformer.from_crs("EPSG:26910", "EPSG:2927", always_xy=True)
    xs22, ys22 = to_2927.transform([c[0] for c in coords], [c[1] for c in coords])

    with rasterio.open(dtm_path) as dem13, rasterio.open(DEM_2022_VRT) as dem22:
        z13 = sample_nearest(dem13, coords)
        z22 = sample_nearest(dem22, list(zip(xs22, ys22))) * US_FT

    # 2024 profile is already centred on midpoint (= d_c)
    d24 = d_c  # already from midpoint; thalweg-relative:
    d24_th = d24 - d_ch

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    m13 = np.isfinite(z13) & (np.abs(d_from_thalweg) <= ZOOM_M)
    m22 = np.isfinite(z22) & (np.abs(d_from_thalweg) <= ZOOM_M)
    m24 = np.isfinite(z24) & (np.abs(d24_th) <= ZOOM_M)
    ax.plot(d_from_thalweg[m13], z13[m13], color="#b36b00", lw=1.6, label="2013 lidar (ground, no bathy)")
    ax.plot(d_from_thalweg[m22], z22[m22], color="#8a8a8a", lw=1.4, label="2022 topobathy")
    ax.plot(d24_th[m24], z24[m24], color="black", lw=1.6, label="2024 topobathy")

    def mark(dist_along, z_ft, color, marker, label, xytext):
        if not np.isfinite(dist_along) or not np.isfinite(z_ft):
            return
        x = dist_along - picks["channel_dist_along"]
        y = z_ft * US_FT
        ax.scatter([x], [y], s=36, c=color, marker=marker, zorder=5, edgecolors="white", linewidths=0.4)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=xytext, fontsize=8, color=color)

    mark(picks["ridge1_dist_along"], picks["ridge1_z24_ft"], "#1f4e79", "v", "L crest 2024", (-70, 8))
    mark(picks["ridge2_dist_along"], picks["ridge2_z24_ft"], "#1f4e79", "v", "R crest 2024", (8, 8))
    ax.axvline(0.0, color="0.7", lw=0.8, ls=":")
    ax.set_xlim(-ZOOM_M, ZOOM_M)
    ax.set_xlabel("Distance from 2024 thalweg (m)")
    ax.set_ylabel("Elevation (m NAVD88)")
    ax.set_title(f"Node {NODE}  ({dist_out_km:.1f} km from outlet)")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_base = ROOT / "data" / "Nooksack_node_73_2013_2022_2024_xs"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)
    print(f"Wrote {out_base}.png")


def main() -> None:
    dtm = build_dtm()
    compare_and_plot(dtm)


if __name__ == "__main__":
    main()
