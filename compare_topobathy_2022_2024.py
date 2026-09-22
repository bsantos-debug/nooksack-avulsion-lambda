#!/usr/bin/env python3
"""Compare 2022 NOAA Nooksack DTM vs 2024 topobathy at labeled ridge/levee points.

The folder the user pointed at only contains hillshades (uint8). This script
uses the matching NOAA Digital Coast DTM tiles (NAVD88 feet, EPSG:2927)
downloaded to data/2022_dtm/.

At each labeler pick (channel, both ridges, both floodplains):
  1. Interpolate XY along the existing cross-section
  2. Sample the 2024 topobathy (EPSG:26910)
  3. Sample the 2022 DTM after transforming XY to EPSG:2927
  4. Convert 2022 feet to metres (US survey foot)

Ridge crests can migrate, so 2022 ridge elevation is also taken as the local
maximum along the transect within ±25 m of the 2024 ridge pick.

Outputs:
  data/Nooksack_2022_vs_2024_ridge_change.csv
  data/Nooksack_2022_vs_2024_ridge_change_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent
LABELS_DIR = ROOT / "data" / "labels"
XS_PATH = ROOT / "cross_sections.shp"
LAMBDA_CSV = ROOT / "data" / "Nooksack_lambda_results.csv"
PROFILES_DIR = ROOT / "cross_section_profiles"
DEM_2024 = Path(
    "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW 2/"
    "Data/Washington/Nooksack_data/BathymetryData/Topobathy_reprojected.tif"
)
DEM_2022_VRT = ROOT / "data" / "2022_dtm" / "WA_Nooksack_DEM_2022_m9574_EPSG-2927.vrt"
OUT_CSV = ROOT / "data" / "Nooksack_2022_vs_2024_ridge_change.csv"
OUT_JSON = ROOT / "data" / "Nooksack_2022_vs_2024_ridge_change_summary.json"

US_FT = 0.304800609601219  # metres per US survey foot
RIDGE_SEARCH_M = 25.0
SAMPLE_STEP_M = 1.0
FEATURES = ("channel", "ridge1", "floodplain1", "ridge2", "floodplain2")


def load_labels(node_id: int) -> dict[str, dict[str, float]] | None:
    path = LABELS_DIR / f"Nooksack_node_{int(node_id)}_labels.csv"
    if not path.exists():
        return None
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not first or first[0].strip().startswith("# SKIPPED"):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or "label" not in df.columns:
        return None
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        lab = str(row["label"]).lower()
        if lab in FEATURES:
            out[lab] = {
                "dist_along": float(row["dist_along"]),
                "elevation": float(row["elevation"]),
            }
    if "channel" not in out:
        return None
    return out


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
        if nodata is not None and (z == nodata or abs(z - float(nodata)) < 1.0):
            vals.append(np.nan)
            continue
        # NOAA / ArcGIS nodata float32
        if z < -1e20:
            vals.append(np.nan)
            continue
        vals.append(z)
    return np.asarray(vals, dtype=float)


def local_max_along_line(
    line: LineString,
    dist_along: float,
    src: rasterio.io.DatasetReader,
    transformer: Transformer | None,
    half_window_m: float = RIDGE_SEARCH_M,
    step_m: float = SAMPLE_STEP_M,
) -> float:
    length = float(line.length)
    dists = np.arange(
        max(0.0, dist_along - half_window_m),
        min(length, dist_along + half_window_m) + step_m,
        step_m,
    )
    if dists.size == 0:
        return np.nan
    pts = [xy_on_line(line, d) for d in dists]
    if transformer is not None:
        xs, ys = zip(*pts)
        xx, yy = transformer.transform(xs, ys)
        pts = list(zip(xx, yy))
    z = sample_nearest(src, pts)
    return float(np.nanmax(z)) if np.any(np.isfinite(z)) else np.nan


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
    hi = lo + 5
    return f"{lo}–{hi}"


def pick_from_2024_profile(node_id: int, line: LineString) -> dict[str, float] | None:
    """Channel = near-center min; ridges = max within 350 m on each side.

    Distances in the profile CSVs are centred on the cross-section midpoint.
    Elevations are native 2024 raster units (NAVD88 feet).
    """
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

    def side_fp(d0: float, d1: float) -> float:
        m = np.isfinite(z) & (d >= min(d0, d1)) & (d <= max(d0, d1))
        if not np.any(m):
            return np.nan
        return float(np.nanpercentile(z[m], 25))

    d_l, z_l = side_crest(np.isfinite(z) & (d < d_ch) & (d >= d_ch - 350.0))
    d_r, z_r = side_crest(np.isfinite(z) & (d > d_ch) & (d <= d_ch + 350.0))
    z_fl = side_fp(d_ch - 350.0, d_ch - 80.0) if np.isfinite(d_l) else np.nan
    z_fr = side_fp(d_ch + 80.0, d_ch + 350.0) if np.isfinite(d_r) else np.nan
    mid = 0.5 * float(line.length)
    return {
        "channel_dist_along": d_ch + mid,
        "channel_z24_ft": z_ch,
        "ridge1_dist_along": d_l + mid if np.isfinite(d_l) else np.nan,
        "ridge1_z24_ft": z_l,
        "ridge2_dist_along": d_r + mid if np.isfinite(d_r) else np.nan,
        "ridge2_z24_ft": z_r,
        "floodplain1_z24_ft": z_fl,
        "floodplain2_z24_ft": z_fr,
    }


def main() -> None:
    xs = gpd.read_file(XS_PATH)
    if xs.crs is None or str(xs.crs) != "EPSG:26910":
        xs = xs.to_crs(26910)
    xs["node_id"] = xs["node_id"].astype(int)
    xs = xs.set_index("node_id", drop=False)

    lam = pd.read_csv(LAMBDA_CSV)
    lam["node_id"] = lam["node_id"].astype(int)
    lam_idx = lam.set_index("node_id")

    transformer = Transformer.from_crs("EPSG:26910", "EPSG:2927", always_xy=True)

    rows = []
    n_no_label = 0
    n_no_xs = 0

    with rasterio.open(DEM_2024) as dem24, rasterio.open(DEM_2022_VRT) as dem22:
        print("2024 CRS:", dem24.crs)
        print("2022 CRS:", dem22.crs)
        for node_id, rec in xs.iterrows():
            line = rec.geometry
            if line is None or line.is_empty:
                n_no_xs += 1
                continue
            labels = load_labels(int(node_id))
            if not labels:
                n_no_label += 1
                continue

            feats = {}
            coords24 = []
            feat_names = []
            for name in FEATURES:
                if name not in labels:
                    feats[name] = None
                    continue
                x, y = xy_on_line(line, labels[name]["dist_along"])
                feats[name] = (x, y, labels[name]["dist_along"], labels[name]["elevation"])
                coords24.append((x, y))
                feat_names.append(name)

            z24 = sample_nearest(dem24, coords24)
            xs22, ys22 = transformer.transform(
                [c[0] for c in coords24], [c[1] for c in coords24]
            )
            z22_ft = sample_nearest(dem22, list(zip(xs22, ys22)))

            sampled = {n: (float(a), float(b)) for n, a, b in zip(feat_names, z24, z22_ft)}

            def zpair(name: str) -> tuple[float, float]:
                return sampled.get(name, (np.nan, np.nan))

            z24_ch, z22_ch_ft = zpair("channel")
            z24_r1, z22_r1_ft = zpair("ridge1")
            z24_r2, z22_r2_ft = zpair("ridge2")
            z24_f1, z22_f1_ft = zpair("floodplain1")
            z24_f2, z22_f2_ft = zpair("floodplain2")

            z22_r1_peak_ft = np.nan
            z22_r2_peak_ft = np.nan
            if "ridge1" in labels:
                z22_r1_peak_ft = local_max_along_line(
                    line, labels["ridge1"]["dist_along"], dem22, transformer
                )
            if "ridge2" in labels:
                z22_r2_peak_ft = local_max_along_line(
                    line, labels["ridge2"]["dist_along"], dem22, transformer
                )

            dist_out = float(rec["dist_out"]) if pd.notna(rec.get("dist_out")) else np.nan
            if int(node_id) in lam_idx.index:
                dist_out = float(lam_idx.loc[int(node_id), "dist_out"])
                lambda_val = lam_idx.loc[int(node_id), "lambda"] if "lambda" in lam_idx.columns else np.nan
                ridge_bank = lam_idx.loc[int(node_id), "ridge_bank_used"] if "ridge_bank_used" in lam_idx.columns else np.nan
            else:
                lambda_val = np.nan
                ridge_bank = np.nan

            rows.append(
                {
                    "node_id": int(node_id),
                    "dist_out_m": dist_out,
                    "lambda": lambda_val,
                    "ridge_bank_used": ridge_bank,
                    "x": feats["channel"][0] if feats.get("channel") else np.nan,
                    "y": feats["channel"][1] if feats.get("channel") else np.nan,
                    "label_channel_m": labels["channel"]["elevation"],
                    "z24_channel": z24_ch,
                    "z22_channel_ft": z22_ch_ft,
                    "z24_ridge1": z24_r1,
                    "z22_ridge1_ft": z22_r1_ft,
                    "z22_ridge1_peak_ft": z22_r1_peak_ft,
                    "z24_ridge2": z24_r2,
                    "z22_ridge2_ft": z22_r2_ft,
                    "z22_ridge2_peak_ft": z22_r2_peak_ft,
                    "z24_floodplain1": z24_f1,
                    "z22_floodplain1_ft": z22_f1_ft,
                    "z24_floodplain2": z24_f2,
                    "z22_floodplain2_ft": z22_f2_ft,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Labeled nodes used: {len(df)}  (no label {n_no_label}, no XS {n_no_xs})")

    # Detect 2024 units using floodplain samples (should be relatively stable)
    fp24 = np.concatenate(
        [df["z24_floodplain1"].to_numpy(), df["z24_floodplain2"].to_numpy()]
    )
    fp22 = np.concatenate(
        [df["z22_floodplain1_ft"].to_numpy(), df["z22_floodplain2_ft"].to_numpy()]
    )
    ok = np.isfinite(fp24) & np.isfinite(fp22) & (np.abs(fp22) > 1.0)
    ratio = np.nan
    if np.any(ok):
        ratio = float(np.median(fp24[ok] / fp22[ok]))
    z24_is_feet = bool(np.isfinite(ratio) and 0.85 < ratio < 1.15)
    print(f"Median z2024 / z2022_ft at floodplains = {ratio:.3f}")
    print(f"Treat 2024 raster as {'feet (NAVD88)' if z24_is_feet else 'metres'}")

    def to_m_24(z):
        z = z.astype(float)
        return z * US_FT if z24_is_feet else z

    def to_m_22(z):
        return z.astype(float) * US_FT

    df["z24_channel_m"] = to_m_24(df["z24_channel"])
    df["z24_ridge1_m"] = to_m_24(df["z24_ridge1"])
    df["z24_ridge2_m"] = to_m_24(df["z24_ridge2"])
    df["z24_floodplain1_m"] = to_m_24(df["z24_floodplain1"])
    df["z24_floodplain2_m"] = to_m_24(df["z24_floodplain2"])
    df["z22_channel_m"] = to_m_22(df["z22_channel_ft"])
    df["z22_ridge1_m"] = to_m_22(df["z22_ridge1_ft"])
    df["z22_ridge2_m"] = to_m_22(df["z22_ridge2_ft"])
    df["z22_ridge1_peak_m"] = to_m_22(df["z22_ridge1_peak_ft"])
    df["z22_ridge2_peak_m"] = to_m_22(df["z22_ridge2_peak_ft"])
    df["z22_floodplain1_m"] = to_m_22(df["z22_floodplain1_ft"])
    df["z22_floodplain2_m"] = to_m_22(df["z22_floodplain2_ft"])

    # Point-to-point DoD at 2024 label locations
    df["dz_channel_m"] = df["z24_channel_m"] - df["z22_channel_m"]
    df["dz_ridge1_m"] = df["z24_ridge1_m"] - df["z22_ridge1_m"]
    df["dz_ridge2_m"] = df["z24_ridge2_m"] - df["z22_ridge2_m"]
    df["dz_ridge1_peak_m"] = df["z24_ridge1_m"] - df["z22_ridge1_peak_m"]
    df["dz_ridge2_peak_m"] = df["z24_ridge2_m"] - df["z22_ridge2_peak_m"]
    df["dz_floodplain1_m"] = df["z24_floodplain1_m"] - df["z22_floodplain1_m"]
    df["dz_floodplain2_m"] = df["z24_floodplain2_m"] - df["z22_floodplain2_m"]
    df["dz_floodplain_mean_m"] = np.nanmean(
        np.vstack([df["dz_floodplain1_m"], df["dz_floodplain2_m"]]), axis=0
    )
    df["dz_ridge_mean_m"] = np.nanmean(
        np.vstack([df["dz_ridge1_m"], df["dz_ridge2_m"]]), axis=0
    )
    df["dz_ridge_peak_mean_m"] = np.nanmean(
        np.vstack([df["dz_ridge1_peak_m"], df["dz_ridge2_peak_m"]]), axis=0
    )

    # Gearon lower-ridge (min of the two crests), 2022 vs 2024
    df["z24_lower_ridge_m"] = np.nanmin(
        np.vstack([df["z24_ridge1_m"], df["z24_ridge2_m"]]), axis=0
    )
    df["z22_lower_ridge_m"] = np.nanmin(
        np.vstack([df["z22_ridge1_peak_m"], df["z22_ridge2_peak_m"]]), axis=0
    )
    df["dz_lower_ridge_m"] = df["z24_lower_ridge_m"] - df["z22_lower_ridge_m"]

    df["Har24_r1_m"] = df["z24_ridge1_m"] - df["z24_floodplain1_m"]
    df["Har24_r2_m"] = df["z24_ridge2_m"] - df["z24_floodplain2_m"]
    df["Har22_r1_m"] = df["z22_ridge1_peak_m"] - df["z22_floodplain1_m"]
    df["Har22_r2_m"] = df["z22_ridge2_peak_m"] - df["z22_floodplain2_m"]
    df["dHar_r1_m"] = df["Har24_r1_m"] - df["Har22_r1_m"]
    df["dHar_r2_m"] = df["Har24_r2_m"] - df["Har22_r2_m"]
    df["dHar_mean_m"] = np.nanmean(np.vstack([df["dHar_r1_m"], df["dHar_r2_m"]]), axis=0)

    df["Hm24_r1_m"] = df["z24_ridge1_m"] - df["z24_channel_m"]
    df["Hm24_r2_m"] = df["z24_ridge2_m"] - df["z24_channel_m"]
    df["Hm22_r1_m"] = df["z22_ridge1_peak_m"] - df["z22_channel_m"]
    df["Hm22_r2_m"] = df["z22_ridge2_peak_m"] - df["z22_channel_m"]
    df["dHm_r1_m"] = df["Hm24_r1_m"] - df["Hm22_r1_m"]
    df["dHm_r2_m"] = df["Hm24_r2_m"] - df["Hm22_r2_m"]
    df["dHm_mean_m"] = np.nanmean(np.vstack([df["dHm_r1_m"], df["dHm_r2_m"]]), axis=0)

    # Coverage: 2022 DTM is river-corridor only
    df["has_2022"] = df[["z22_ridge1_m", "z22_ridge2_m", "z22_channel_m"]].notna().any(axis=1)
    df["km_bin"] = df["dist_out_m"].apply(lambda v: km_bin(v) if np.isfinite(v) else "NA")

    bias = float(np.nanmedian(df["dz_floodplain_mean_m"]))
    df["dz_ridge_mean_biascorr_m"] = df["dz_ridge_mean_m"] - bias
    df["dz_lower_ridge_biascorr_m"] = df["dz_lower_ridge_m"] - bias
    df["dz_channel_biascorr_m"] = df["dz_channel_m"] - bias

    use = df[df["has_2022"]].copy()
    print(f"Nodes with 2022 coverage: {len(use)} / {len(df)}")
    print(f"Median floodplain Δz (vertical bias proxy): {bias:+.3f} m")

    for col, label in [
        ("dz_ridge_mean_m", "mean ridge (same XY)"),
        ("dz_ridge_peak_mean_m", "mean ridge (2022 local peak)"),
        ("dz_lower_ridge_m", "lower ridge crest"),
        ("dz_channel_m", "channel bed"),
        ("dz_floodplain_mean_m", "floodplain"),
        ("dHar_mean_m", "Har (ridge − floodplain)"),
        ("dHm_mean_m", "Hm (ridge − channel)"),
    ]:
        s = summarize(use[col].to_numpy())
        print(
            f"{label:32s}  n={s['n']:3d}  median={s['median']:+.3f} m  "
            f"mean={s['mean']:+.3f}  p10={s['p10']:+.3f}  p90={s['p90']:+.3f}"
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("dist_out_m", ascending=False).to_csv(OUT_CSV, index=False)

    # --- Primary: 2024-profile crests on the current cross-sections ---
    # Most labeler CSVs were picked on the previous centerline, so dist_along
    # on the rebuilt transects is not the original levee location.
    prof_rows = []
    with rasterio.open(DEM_2022_VRT) as dem22:
        for node_id, rec in xs.iterrows():
            line = rec.geometry
            if line is None or line.is_empty:
                continue
            picks = pick_from_2024_profile(int(node_id), line)
            if not picks:
                continue
            names = []
            coords = []
            for key, dist_key in (
                ("channel", "channel_dist_along"),
                ("ridge1", "ridge1_dist_along"),
                ("ridge2", "ridge2_dist_along"),
            ):
                dist = picks[dist_key]
                if not np.isfinite(dist):
                    continue
                names.append(key)
                coords.append(xy_on_line(line, dist))
            if not coords:
                continue
            xs22, ys22 = transformer.transform(
                [c[0] for c in coords], [c[1] for c in coords]
            )
            z22_ft = sample_nearest(dem22, list(zip(xs22, ys22)))
            sampled = {n: float(z) for n, z in zip(names, z22_ft)}
            dist_out = float(rec["dist_out"]) if pd.notna(rec.get("dist_out")) else np.nan
            lambda_val = (
                float(lam_idx.loc[int(node_id), "lambda"])
                if int(node_id) in lam_idx.index
                else np.nan
            )
            z24_r1 = picks["ridge1_z24_ft"] * US_FT
            z24_r2 = picks["ridge2_z24_ft"] * US_FT
            z24_ch = picks["channel_z24_ft"] * US_FT
            z22_r1 = sampled.get("ridge1", np.nan) * US_FT
            z22_r2 = sampled.get("ridge2", np.nan) * US_FT
            z22_ch = sampled.get("channel", np.nan) * US_FT
            dz_r1 = z24_r1 - z22_r1
            dz_r2 = z24_r2 - z22_r2
            lower24 = np.nanmin([z24_r1, z24_r2])
            lower22 = np.nanmin([z22_r1, z22_r2])
            dHar_mean = np.nanmean([dz_r1, dz_r2])
            prof_rows.append(
                {
                    "node_id": int(node_id),
                    "dist_out_m": dist_out,
                    "lambda": lambda_val,
                    "dz_ridge1_m": dz_r1,
                    "dz_ridge2_m": dz_r2,
                    "dz_ridge_mean_m": np.nanmean([dz_r1, dz_r2]),
                    "dz_lower_ridge_m": lower24 - lower22,
                    "dz_channel_m": z24_ch - z22_ch,
                    "dHar_mean_m": dHar_mean,
                    "dHm_mean_m": np.nanmean(
                        [
                            (z24_r1 - z24_ch) - (z22_r1 - z22_ch),
                            (z24_r2 - z24_ch) - (z22_r2 - z22_ch),
                        ]
                    ),
                }
            )

    prof = pd.DataFrame(prof_rows)
    prof["has_ridge"] = prof["dz_ridge_mean_m"].notna()
    prof["km_bin"] = prof["dist_out_m"].apply(
        lambda v: km_bin(v) if np.isfinite(v) else "NA"
    )
    pr = prof[prof["has_ridge"]].copy()
    print(f"Profile-based nodes with 2022 ridge sample: {len(pr)} / {len(prof)}")
    for col, label in [
        ("dz_ridge_mean_m", "2024 crest, mean both banks"),
        ("dz_lower_ridge_m", "2024 lower crest"),
        ("dz_channel_m", "2024 thalweg"),
        ("dHm_mean_m", "Hm change"),
    ]:
        s = summarize(pr[col].to_numpy())
        print(
            f"{label:32s}  n={s['n']:3d}  median={s['median']:+.3f} m  "
            f"mean={s['mean']:+.3f}  p10={s['p10']:+.3f}  p90={s['p90']:+.3f}"
        )

    OUT_PROF = ROOT / "data" / "Nooksack_2022_vs_2024_profile_ridges.csv"
    prof.sort_values("dist_out_m", ascending=False).to_csv(OUT_PROF, index=False)

    # 5-km bins, high km (upstream) first to match existing canvas
    bin_order = [
        "50–55",
        "45–50",
        "40–45",
        "35–40",
        "30–35",
        "25–30",
        "20–25",
        "15–20",
        "10–15",
        "5–10",
        "0–5",
    ]

    def med(sub: pd.DataFrame, col: str):
        x = sub[col].dropna()
        return None if x.empty else round(float(x.median()), 3)

    km_bins = []
    for b in bin_order:
        sub = pr[pr["km_bin"] == b]
        km_bins.append(
            {
                "bin": b,
                "n": int(len(sub)),
                "median_dz_ridge_m": med(sub, "dz_ridge_mean_m"),
                "median_dz_channel_m": med(sub, "dz_channel_m"),
                "median_dHm_m": med(sub, "dHm_mean_m"),
                "pct_abs_gt_25cm": None
                if sub["dz_ridge_mean_m"].dropna().empty
                else round(float((sub["dz_ridge_mean_m"].abs() > 0.25).mean() * 100), 1),
            }
        )

    def table_rows(frame: pd.DataFrame) -> list[dict]:
        out = []
        for _, r in frame.iterrows():
            out.append(
                {
                    "node_id": int(r["node_id"]),
                    "dist_out_km": round(float(r["dist_out_m"]) / 1000.0, 2),
                    "dz_ridge_m": round(float(r["dz_ridge_mean_m"]), 2),
                    "dz_r1_m": None
                    if not np.isfinite(r["dz_ridge1_m"])
                    else round(float(r["dz_ridge1_m"]), 2),
                    "dz_r2_m": None
                    if not np.isfinite(r["dz_ridge2_m"])
                    else round(float(r["dz_ridge2_m"]), 2),
                    "dz_channel_m": None
                    if not np.isfinite(r["dz_channel_m"])
                    else round(float(r["dz_channel_m"]), 2),
                    "lambda": None if not np.isfinite(r["lambda"]) else round(float(r["lambda"]), 2),
                }
            )
        return out

    ranked = pr.dropna(subset=["dz_ridge_mean_m"])
    relabeled_ids = {1, 2, 3, 4, 5, 6, 7, 79, 89, 90}
    rel = df[df["node_id"].isin(relabeled_ids) & df["has_2022"]]

    def clean(d: dict) -> dict:
        return {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()
        }

    summary = {
        "z24_is_feet": z24_is_feet,
        "z24_over_z22ft_ratio": None if not np.isfinite(ratio) else round(ratio, 4),
        "n_xs": int(len(xs)),
        "n_profile_with_2022_ridge": int(len(pr)),
        "n_labels_with_2022": int(len(use)),
        "profile_stats": {
            "ridge": clean(summarize(pr["dz_ridge_mean_m"].to_numpy())),
            "lower_ridge": clean(summarize(pr["dz_lower_ridge_m"].to_numpy())),
            "channel": clean(summarize(pr["dz_channel_m"].to_numpy())),
            "dHm": clean(summarize(pr["dHm_mean_m"].to_numpy())),
        },
        "relabeled_label_stats": {
            "n": int(len(rel)),
            "ridge": clean(summarize(rel["dz_ridge_mean_m"].to_numpy())),
            "channel": clean(summarize(rel["dz_channel_m"].to_numpy())),
            "floodplain": clean(summarize(rel["dz_floodplain_mean_m"].to_numpy())),
        },
        "pct_ridge_aggrade_gt_5cm": round(
            float(100.0 * np.nanmean(pr["dz_ridge_mean_m"] > 0.05)), 1
        ),
        "pct_ridge_degrade_gt_5cm": round(
            float(100.0 * np.nanmean(pr["dz_ridge_mean_m"] < -0.05)), 1
        ),
        "pct_ridge_abs_gt_25cm": round(
            float(100.0 * np.nanmean(pr["dz_ridge_mean_m"].abs() > 0.25)), 1
        ),
        "pct_channel_degrade_gt_5cm": round(
            float(100.0 * np.nanmean(pr["dz_channel_m"] < -0.05)), 1
        ),
        "km_bins": km_bins,
        "top_aggrade": table_rows(ranked.nlargest(8, "dz_ridge_mean_m")),
        "top_degrade": table_rows(ranked.nsmallest(8, "dz_ridge_mean_m")),
        "notes": {
            "dem_2022": "NOAA Digital Coast ID 9574, 24–25 Feb 2022, NAVD88 Geoid12B ftUS, EPSG:2927, 3 ft cells",
            "dem_2024": "Topobathy_reprojected.tif values are NAVD88 feet; XY is EPSG:26910 metres",
            "positive_dz": "2024 higher than 2022 (net aggradation at that XY)",
            "ridge_definition": "On each current XS: 2024 thalweg = min within 120 m of midpoint; crests = max within 350 m left/right",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PROF}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
