#!/usr/bin/env python3
"""
Spatial structure of Λ along a reach (paper-style workflow).

Implements the analysis described for quantifying spatial autocorrelation of
lambda and locating enhanced avulsion-potential zones:

1. Experimental semivariogram of log10(Λ) along the channel
2. Robert (1988) combined model: exponential + two periodic components
   (9 parameters, two-stage fit) → LC = 3r, Lλ = l1
3. Continuous wavelet transform (supporting multi-scale view)
4. Savitzky–Golay smooth of Λ with window ≈ LC
5. LISA (Anselin, 1995) on smoothed Λ → HH / LH / HL / LL
6. Flag HH zones that are bordered by LH transitions
7. Optional: proximity-weighted Λ comparison within ±LC of avulsion events

Usage:
    poetry run python analyze_lambda_hotspots.py
    poetry run python analyze_lambda_hotspots.py --avulsion-events data/avulsion_events.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import DistanceBand
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# === defaults ===
DEFAULT_LAMBDA_CSV = "data/Nooksack_lambda_results.csv"
DEFAULT_POINTS = "centerline_points_from_xs.shp"
DEFAULT_OUTDIR = "data/hotspots"
DEFAULT_RIVER = "Nooksack"
LISA_P_THRESH = 0.05


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_lambda(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"node_id", "dist_out", "lambda"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Lambda CSV missing columns: {sorted(missing)}")
    df = df[df["lambda"].notna() & (df["lambda"] > 0)].copy()
    df["node_id"] = df["node_id"].astype(int)
    df = df.sort_values("dist_out", ascending=False).reset_index(drop=True)
    dmax = float(df["dist_out"].max())
    df["dist_down_m"] = dmax - df["dist_out"]
    df["dist_down_km"] = df["dist_down_m"] / 1000.0
    df["log_lambda"] = np.log10(df["lambda"])
    return df


# ---------------------------------------------------------------------------
# Semivariogram — Robert (1988) exponential + 2 periodic components
# ---------------------------------------------------------------------------

def empirical_variogram(
    x: np.ndarray,
    z: np.ndarray,
    n_lags: int = 40,
    max_lag: Optional[float] = None,
) -> pd.DataFrame:
    """Classical experimental semivariogram along a 1-D coordinate (m)."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    if max_lag is None:
        max_lag = 0.5 * (x.max() - x.min())

    dx = np.abs(x[:, None] - x[None, :])
    dz2 = 0.5 * (z[:, None] - z[None, :]) ** 2
    iu = np.triu_indices(len(x), k=1)
    lags_all = dx[iu]
    gamma_all = dz2[iu]

    edges = np.linspace(0.0, max_lag, n_lags + 1)
    lag_centers = 0.5 * (edges[:-1] + edges[1:])
    gamma, counts = [], []
    for i in range(n_lags):
        mask = (lags_all >= edges[i]) & (lags_all < edges[i + 1])
        n = int(mask.sum())
        counts.append(n)
        gamma.append(float(gamma_all[mask].mean()) if n else np.nan)
    return pd.DataFrame({"lag_m": lag_centers, "gamma": gamma, "n_pairs": counts})


def exponential_component(h: np.ndarray, c0: float, c: float, r: float) -> np.ndarray:
    """Nugget + exponential: c0 + c (1 − exp(−h/r)). Practical range ≈ 3r = LC."""
    h = np.asarray(h, dtype=float)
    r = max(float(r), 1e-6)
    return c0 + c * (1.0 - np.exp(-h / r))


def periodic_component(h: np.ndarray, a: float, b: float, wavelength: float) -> np.ndarray:
    """Fourier periodic term a cos(2πh/l) + b sin(2πh/l) (Robert / Wren form)."""
    h = np.asarray(h, dtype=float)
    wavelength = max(float(wavelength), 1e-6)
    omega = 2.0 * np.pi * h / wavelength
    return a * np.cos(omega) + b * np.sin(omega)


def robert_model(
    h: np.ndarray,
    c0: float,
    c: float,
    r: float,
    a1: float,
    b1: float,
    l1: float,
    a2: float,
    b2: float,
    l2: float,
) -> np.ndarray:
    """Combined exponential + two periodic components (9 parameters)."""
    return (
        exponential_component(h, c0, c, r)
        + periodic_component(h, a1, b1, l1)
        + periodic_component(h, a2, b2, l2)
    )


def fit_robert_variogram(vario: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Two-stage fit following Robert (1988) / paper Text S7:

    Stage 1 — fit nugget + exponential (c0, c, r)
    Stage 2 — fit two periodic components to residuals, then refine all 9 jointly
    """
    ok = vario["gamma"].notna() & (vario["n_pairs"] > 10)
    v = vario.loc[ok]
    if len(v) < 8:
        return None

    h = v["lag_m"].to_numpy(dtype=float)
    g = v["gamma"].to_numpy(dtype=float)
    h_max = float(h.max())
    g_max = float(np.nanmax(g))
    g_min = float(np.nanmin(g[: max(3, len(g) // 10)]))

    # --- Stage 1: exponential ---
    try:
        p_exp, _ = curve_fit(
            exponential_component,
            h,
            g,
            p0=[max(g_min, 0.0), max(g_max - g_min, 1e-4), h_max / 6.0],
            bounds=([0.0, 0.0, h[1]], [g_max * 2 + 1, g_max * 3 + 1, h_max]),
            maxfev=8000,
        )
    except Exception:
        return None

    c0, c, r = map(float, p_exp)
    resid = g - exponential_component(h, c0, c, r)

    # Initial wavelengths from residual periodogram-ish peaks (lag of max |resid| oscillation)
    # Use FFT of residuals interpolated on uniform lag grid
    h_u = np.linspace(h.min(), h.max(), 256)
    resid_u = np.interp(h_u, h, resid)
    resid_u = resid_u - resid_u.mean()
    spec = np.abs(np.fft.rfft(resid_u)) ** 2
    freqs = np.fft.rfftfreq(len(h_u), d=(h_u[1] - h_u[0]))
    # ignore zero / very long wavelengths (> 0.8 * max lag)
    valid = (freqs > 1.0 / (0.8 * h_max)) & (freqs < 1.0 / (2.0 * np.median(np.diff(h))))
    if np.any(valid):
        order = np.argsort(spec[valid])[::-1]
        f_valid = freqs[valid][order]
        l1_0 = float(1.0 / f_valid[0])
        l2_0 = float(1.0 / f_valid[1]) if len(f_valid) > 1 else l1_0 * 0.5
    else:
        l1_0, l2_0 = h_max / 3.0, h_max / 6.0

    amp0 = float(np.std(resid)) if np.std(resid) > 0 else 1e-3

    def periodic_sum(hv, a1, b1, l1, a2, b2, l2):
        return periodic_component(hv, a1, b1, l1) + periodic_component(hv, a2, b2, l2)

    # --- Stage 2a: fit periodic to residuals ---
    try:
        p_per, _ = curve_fit(
            periodic_sum,
            h,
            resid,
            p0=[amp0, 0.0, l1_0, amp0 * 0.5, 0.0, l2_0],
            bounds=(
                [-g_max * 2, -g_max * 2, h[1] * 2, -g_max * 2, -g_max * 2, h[1] * 2],
                [g_max * 2, g_max * 2, h_max, g_max * 2, g_max * 2, h_max],
            ),
            maxfev=12000,
        )
    except Exception:
        p_per = np.array([amp0, 0.0, l1_0, amp0 * 0.5, 0.0, l2_0])

    # --- Stage 2b: joint refine all 9 ---
    p0 = np.concatenate([p_exp, p_per])
    bounds = (
        [0.0, 0.0, h[1], -g_max * 2, -g_max * 2, h[1] * 2, -g_max * 2, -g_max * 2, h[1] * 2],
        [g_max * 2 + 1, g_max * 3 + 1, h_max, g_max * 2, g_max * 2, h_max, g_max * 2, g_max * 2, h_max],
    )
    try:
        popt, _ = curve_fit(robert_model, h, g, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        popt = p0

    c0, c, r, a1, b1, l1, a2, b2, l2 = map(float, popt)
    # Ensure l1 is the longer (primary) wavelength
    if l2 > l1:
        a1, b1, l1, a2, b2, l2 = a2, b2, l2, a1, b1, l1

    return {
        "c0": c0,
        "c": c,
        "r": r,
        "a1": a1,
        "b1": b1,
        "l1": l1,
        "a2": a2,
        "b2": b2,
        "l2": l2,
        "LC": 3.0 * r,  # effective correlation length
        "L_lambda": l1,  # primary periodic wavelength
        "A1": float(np.hypot(a1, b1)),
        "A2": float(np.hypot(a2, b2)),
    }


# ---------------------------------------------------------------------------
# Wavelets (supporting multi-scale view)
# ---------------------------------------------------------------------------

def _ricker(points: int, a: float) -> np.ndarray:
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi**0.25))
    vec = np.arange(0, points) - (points - 1.0) / 2.0
    xsq = vec**2
    return A * (1.0 - xsq / (a**2)) * np.exp(-xsq / (2.0 * a**2))


def _cwt(data: np.ndarray, widths: np.ndarray) -> np.ndarray:
    out = np.empty((len(widths), len(data)), dtype=float)
    for i, width in enumerate(widths):
        n = min(len(data), int(np.ceil(width * 10)))
        if n % 2 == 0:
            n += 1
        w = _ricker(n, float(width))
        out[i] = np.convolve(data, w[::-1], mode="same")
    return out


def wavelet_cwt(dist_m: np.ndarray, z: np.ndarray, n_scales: int = 40):
    order = np.argsort(dist_m)
    d, y = dist_m[order], z[order]
    dx = float(np.median(np.diff(d)))
    if not np.isfinite(dx) or dx <= 0:
        dx = 200.0
    grid = np.arange(d.min(), d.max() + dx * 0.5, dx)
    y_u = np.interp(grid, d, y)
    widths = np.geomspace(1.5, max(3.0, len(grid) / 4), n_scales)
    power = np.abs(_cwt(y_u - np.nanmean(y_u), widths)) ** 2
    scales_m = widths * dx * 4.0
    return scales_m, power, grid


# ---------------------------------------------------------------------------
# Savitzky–Golay smooth at LC scale + LISA
# ---------------------------------------------------------------------------

def savgol_smooth_lambda(df: pd.DataFrame, LC_m: float) -> pd.DataFrame:
    """Smooth Λ with Savitzky–Golay window ≈ LC (odd # of samples)."""
    out = df.copy()
    dx = float(np.median(np.diff(np.sort(out["dist_down_m"].to_numpy()))))
    if not np.isfinite(dx) or dx <= 0:
        dx = 200.0
    win = int(round(LC_m / dx))
    if win % 2 == 0:
        win += 1
    win = max(5, min(win, len(out) - (1 - len(out) % 2)))
    if win >= len(out):
        win = len(out) - 1 if len(out) % 2 == 0 else len(out)
        win = max(5, win)
    if win % 2 == 0:
        win -= 1
    poly = 2 if win > 5 else 1
    # Sort by distance for filter, then restore
    order = np.argsort(out["dist_down_m"].to_numpy())
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    y = out["lambda"].to_numpy()[order]
    y_s = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    out["lambda_smooth"] = y_s[inv]
    out["log_lambda_smooth"] = np.log10(np.clip(out["lambda_smooth"], 1e-6, None))
    out.attrs["savgol_window_samples"] = win
    out.attrs["savgol_window_m"] = win * dx
    return out


def run_lisa(df: pd.DataFrame, values: np.ndarray, band_m: float, permutations: int = 499):
    coords = np.column_stack([df["dist_down_m"].to_numpy(), np.zeros(len(df))])
    w = DistanceBand(coords, threshold=band_m, binary=True, silence_warnings=True)
    w.transform = "r"
    mi = Moran(values, w, permutations=permutations)
    lisa = Moran_Local(values, w, permutations=permutations, seed=42)

    out = df[
        ["node_id", "dist_out", "dist_down_m", "dist_down_km", "lambda", "lambda_smooth"]
    ].copy()
    out["lisa_I"] = lisa.Is
    out["lisa_p"] = lisa.p_sim
    out["lisa_q"] = lisa.q
    qmap = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    out["lisa_cluster"] = out["lisa_q"].map(qmap)
    sig = out["lisa_p"] < LISA_P_THRESH
    out["lisa_sig_cluster"] = np.where(sig, out["lisa_cluster"], "ns")
    return out, mi, lisa, w


def flag_hh_bordered_by_lh(lisa_df: pd.DataFrame, w) -> pd.DataFrame:
    """
    Mark high-avulsion candidate zones:

    1. Paper criterion: significant HH with ≥1 significant LH neighbor
    2. Fallback on smooth series: significant HH at a cluster edge
       (neighbor is LL / HL / LH / ns) — HH core abutting a transition
    """
    out = lisa_df.copy()
    hh_mask = out["lisa_sig_cluster"] == "HH"
    lh_ids = set(out.index[out["lisa_sig_cluster"] == "LH"].tolist())
    non_hh_ids = set(out.index[out["lisa_sig_cluster"] != "HH"].tolist())

    bordered_lh = []
    at_edge = []
    for idx in out.index:
        neighbors = w.neighbors.get(int(idx), [])
        if not hh_mask.loc[idx]:
            bordered_lh.append(False)
            at_edge.append(False)
            continue
        bordered_lh.append(any(n in lh_ids for n in neighbors))
        at_edge.append(any(n in non_hh_ids for n in neighbors))

    out["hh_bordered_by_lh"] = bordered_lh
    out["hh_cluster_edge"] = at_edge
    out["high_avulsion_zone"] = out["hh_bordered_by_lh"] | (
        out["hh_cluster_edge"] & hh_mask
    )
    return out


def hh_regions(lisa_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse contiguous HH nodes (by dist_down) into regions."""
    hh = lisa_df[lisa_df["lisa_sig_cluster"] == "HH"].sort_values("dist_down_m")
    if hh.empty:
        return pd.DataFrame()
    regions = []
    rid = 0
    block = [hh.iloc[0]]
    for i in range(1, len(hh)):
        prev = block[-1]
        cur = hh.iloc[i]
        # gap > 1.5 median spacing → new region
        if cur["dist_down_m"] - prev["dist_down_m"] > 500:
            regions.append(_region_row(rid, block))
            rid += 1
            block = [cur]
        else:
            block.append(cur)
    regions.append(_region_row(rid, block))
    return pd.DataFrame(regions)


def _region_row(rid: int, block: list) -> dict:
    dfb = pd.DataFrame(block)
    return {
        "region_id": rid,
        "n_nodes": len(dfb),
        "dist_down_km_min": float(dfb["dist_down_km"].min()),
        "dist_down_km_max": float(dfb["dist_down_km"].max()),
        "lambda_smooth_mean": float(dfb["lambda_smooth"].mean()),
        "lambda_smooth_max": float(dfb["lambda_smooth"].max()),
        "n_edge_nodes": int(dfb["high_avulsion_zone"].sum())
        if "high_avulsion_zone" in dfb.columns
        else 0,
    }


# ---------------------------------------------------------------------------
# Optional proximity-weighted avulsion-event analysis
# ---------------------------------------------------------------------------

def proximity_weighted_analysis(
    df: pd.DataFrame,
    events: pd.DataFrame,
    LC_m: float,
) -> Optional[pd.DataFrame]:
    """
    Compare Λ within ±LC of avulsion activity locations.

    ``events`` needs either ``dist_out`` or ``dist_down_m`` (meters).
    Weights inversely by cross-section count in each window (paper-style).
    """
    if events is None or events.empty:
        return None
    ev = events.copy()
    if "dist_down_m" not in ev.columns:
        if "dist_out" not in ev.columns:
            raise ValueError("Avulsion events need dist_out or dist_down_m")
        dmax = float(df["dist_out"].max())
        ev["dist_down_m"] = dmax - ev["dist_out"].astype(float)

    rows = []
    for i, erow in ev.iterrows():
        x0 = float(erow["dist_down_m"])
        near = df[(df["dist_down_m"] - x0).abs() <= LC_m].copy()
        if near.empty:
            continue
        up = near[near["dist_down_m"] < x0]
        down = near[near["dist_down_m"] > x0]
        n = len(near)
        w = 1.0 / n
        rows.append(
            {
                "event_id": erow.get("event_id", i),
                "event_dist_down_m": x0,
                "n_xs": n,
                "weight": w,
                "lambda_mean_window": float(near["lambda_smooth"].mean()),
                "lambda_mean_upstream": float(up["lambda_smooth"].mean()) if len(up) else np.nan,
                "lambda_mean_downstream": float(down["lambda_smooth"].mean()) if len(down) else np.nan,
                "n_hh_in_window": int((near.get("lisa_sig_cluster") == "HH").sum())
                if "lisa_sig_cluster" in near.columns
                else np.nan,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting / summary
# ---------------------------------------------------------------------------

def plot_all(
    df: pd.DataFrame,
    vario: pd.DataFrame,
    fit: Optional[dict],
    scales_m: np.ndarray,
    power: np.ndarray,
    grid_m: np.ndarray,
    lisa_df: pd.DataFrame,
    mi: Moran,
    outdir: Path,
    river_name: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # (A) raw + smoothed Λ and hotspot zones
    ax = axes[0, 0]
    ax.plot(df["dist_down_km"], df["lambda"], color="0.7", lw=0.7, label="Λ")
    ax.plot(df["dist_down_km"], df["lambda_smooth"], color="k", lw=1.2, label="Λ Savitzky–Golay")
    hh = lisa_df[lisa_df["lisa_sig_cluster"] == "HH"]
    haz = lisa_df[lisa_df["high_avulsion_zone"]]
    if not hh.empty:
        ax.scatter(hh["dist_down_km"], hh["lambda_smooth"], c="crimson", s=28, zorder=4, label="LISA HH")
    if not haz.empty:
        ax.scatter(
            haz["dist_down_km"],
            haz["lambda_smooth"],
            facecolors="none",
            edgecolors="darkred",
            s=90,
            lw=1.5,
            zorder=5,
            label="HH bordered by LH",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Distance downstream (km)")
    ax.set_ylabel("Λ")
    ax.set_title(f"{river_name}: Λ profile")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # (B) Robert variogram
    ax = axes[0, 1]
    ax.plot(vario["lag_m"] / 1000.0, vario["gamma"], "o", ms=3, color="steelblue", label="experimental")
    if fit:
        h_fit = np.linspace(0, vario["lag_m"].max(), 300)
        ax.plot(
            h_fit / 1000.0,
            robert_model(
                h_fit,
                fit["c0"], fit["c"], fit["r"],
                fit["a1"], fit["b1"], fit["l1"],
                fit["a2"], fit["b2"], fit["l2"],
            ),
            color="darkred",
            lw=1.5,
            label=(
                f"Robert model\n"
                f"LC=3r={fit['LC']/1000:.1f} km\n"
                f"Lλ=l1={fit['L_lambda']/1000:.1f} km"
            ),
        )
        ax.axvline(fit["LC"] / 1000.0, color="gray", ls="--", lw=0.8, label="LC")
        ax.axvline(fit["L_lambda"] / 1000.0, color="orange", ls=":", lw=1.0, label="Lλ")
    ax.set_xlabel("Lag (km)")
    ax.set_ylabel("Semivariance γ")
    ax.set_title("Semivariogram (log₁₀ Λ) — exponential + 2 periodic")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # (C) wavelet
    ax = axes[1, 0]
    extent = [grid_m.min() / 1000, grid_m.max() / 1000, scales_m.max() / 1000, scales_m.min() / 1000]
    im = ax.imshow(power, aspect="auto", extent=extent, cmap="magma")
    if fit:
        ax.axhline(fit["L_lambda"] / 1000.0, color="cyan", ls="--", lw=1, label="Lλ")
        ax.axhline(fit["LC"] / 1000.0, color="lime", ls="--", lw=1, label="LC")
        ax.legend(fontsize=7)
    ax.set_xlabel("Distance downstream (km)")
    ax.set_ylabel("Approx. scale (km)")
    ax.set_title("CWT power (Ricker) of log₁₀ Λ")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="power")

    # (D) LISA
    ax = axes[1, 1]
    colors = {"HH": "crimson", "LL": "royalblue", "HL": "orange", "LH": "cyan", "ns": "0.85"}
    for lab, g in lisa_df.groupby("lisa_sig_cluster"):
        ax.scatter(g["dist_down_km"], g["lisa_I"], c=colors.get(lab, "0.5"), s=18, label=lab, alpha=0.9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Distance downstream (km)")
    ax.set_ylabel("Local Moran's I")
    ax.set_title(f"LISA on SG-smoothed Λ  |  global I={mi.I:.3f}, p={mi.p_sim:.3f}")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{river_name} — spatial structure of Λ & avulsion-potential hotspots",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = outdir / f"{river_name}_lambda_hotspot_analysis.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_map(lisa_df: pd.DataFrame, points_path: Path, outdir: Path, river_name: str) -> Optional[Path]:
    if not points_path.exists():
        return None
    pts = gpd.read_file(points_path)
    if "node_id" not in pts.columns:
        return None
    pts["node_id"] = pts["node_id"].astype(int)
    g = pts.merge(lisa_df, on="node_id", how="inner")
    if g.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    g.plot(ax=ax, color="0.8", markersize=6, label="nodes")
    hh = g[g["lisa_sig_cluster"] == "HH"]
    haz = g[g["high_avulsion_zone"]]
    lh = g[g["lisa_sig_cluster"] == "LH"]
    if not lh.empty:
        lh.plot(ax=ax, color="cyan", markersize=20, label=f"LH (n={len(lh)})")
    if not hh.empty:
        hh.plot(ax=ax, color="crimson", markersize=28, label=f"HH (n={len(hh)})")
    if not haz.empty:
        haz.plot(
            ax=ax,
            facecolor="none",
            edgecolor="darkred",
            markersize=70,
            label=f"HH∩LH border (n={len(haz)})",
        )
    ax.set_title(f"{river_name}: LISA hotspots (HH bordered by LH)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    path = outdir / f"{river_name}_lisa_hotspots_map.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    # shorten column names for shapefile
    gdf = g.copy()
    rename = {
        "lisa_sig_cluster": "lisa_sig",
        "high_avulsion_zone": "hi_avul",
        "hh_bordered_by_lh": "hh_lh",
        "lambda_smooth": "lam_sm",
        "dist_down_m": "dist_dn",
        "dist_down_km": "dist_dnkm",
    }
    gdf = gdf.rename(columns={k: v for k, v in rename.items() if k in gdf.columns})
    gdf.to_file(outdir / f"{river_name}_lisa_clusters.shp")
    return path


def summarize(lisa_df, fit, mi, df) -> str:
    lines: List[str] = []
    lines.append("=== Global spatial autocorrelation (smoothed Λ) ===")
    lines.append(f"  Moran's I = {mi.I:.4f}  (p = {mi.p_sim:.4f}, z = {mi.z_sim:.2f})")
    if fit:
        lines.append("=== Robert (1988) semivariogram model ===")
        lines.append(f"  c0 (nugget)     = {fit['c0']:.4g}")
        lines.append(f"  c  (partial sill)= {fit['c']:.4g}")
        lines.append(f"  r  (range)      = {fit['r']:.0f} m")
        lines.append(f"  LC = 3r         = {fit['LC']:.0f} m ({fit['LC']/1000:.2f} km)")
        lines.append(f"  Lλ = l1         = {fit['L_lambda']:.0f} m ({fit['L_lambda']/1000:.2f} km)")
        lines.append(f"  l2              = {fit['l2']:.0f} m")
        lines.append(f"  |A1|, |A2|      = {fit['A1']:.4g}, {fit['A2']:.4g}")
        lines.append(f"  Savitzky–Golay window ≈ {df.attrs.get('savgol_window_m', float('nan')):.0f} m "
                     f"({df.attrs.get('savgol_window_samples', '?')} samples)")
    lines.append(f"=== LISA significant clusters (p < {LISA_P_THRESH}) ===")
    for lab in ["HH", "LL", "HL", "LH"]:
        lines.append(f"  {lab}: n = {(lisa_df['lisa_sig_cluster'] == lab).sum()}")
    n_haz = int(lisa_df["high_avulsion_zone"].sum())
    lines.append(f"=== High avulsion-potential zones ===")
    lines.append(f"  HH bordered by LH (paper): n = {int(lisa_df['hh_bordered_by_lh'].sum())}")
    lines.append(f"  HH cluster-edge nodes:     n = {n_haz}")
    regions = hh_regions(lisa_df)
    if not regions.empty:
        lines.append(f"=== Contiguous HH regions: n = {len(regions)} ===")
        for _, r in regions.sort_values("lambda_smooth_max", ascending=False).iterrows():
            lines.append(
                f"  region {int(r['region_id'])}: "
                f"{r['dist_down_km_min']:.1f}–{r['dist_down_km_max']:.1f} km  "
                f"n={int(r['n_nodes'])}  "
                f"Λ_smooth mean={r['lambda_smooth_mean']:.1f} max={r['lambda_smooth_max']:.1f}"
            )
    haz = lisa_df[lisa_df["high_avulsion_zone"]].sort_values("lambda_smooth", ascending=False)
    if not haz.empty:
        lines.append("=== Edge / transition HH nodes (top) ===")
        for _, r in haz.head(20).iterrows():
            lines.append(
                f"  node {int(r['node_id']):3d}  dist_down={r['dist_down_km']:6.2f} km  "
                f"Λ_smooth={r['lambda_smooth']:.2f}  I={r['lisa_I']:.3f}  p={r['lisa_p']:.3f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-style Λ spatial structure analysis")
    parser.add_argument("--lambda-csv", default=DEFAULT_LAMBDA_CSV)
    parser.add_argument("--points", default=DEFAULT_POINTS)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--river-name", default=DEFAULT_RIVER)
    parser.add_argument(
        "--avulsion-events",
        default=None,
        help="Optional CSV with avulsion activity locations (dist_out or dist_down_m)",
    )
    parser.add_argument(
        "--lisa-band-m",
        type=float,
        default=None,
        help="LISA neighbor band (m). Default: LC from variogram fit.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    def proj(p: str) -> Path:
        path = Path(p)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    csv_path = proj(args.lambda_csv)
    points_path = proj(args.points)
    outdir = proj(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Loading {csv_path}")
    df = load_lambda(csv_path)
    print(f"✅ {len(df)} nodes  |  Λ median={df['lambda'].median():.2f}")

    x = df["dist_down_m"].to_numpy()
    z = df["log_lambda"].to_numpy()

    # --- Semivariogram (Robert model) ---
    print("\n📈 Experimental semivariogram + Robert (1988) fit...")
    vario = empirical_variogram(x, z, n_lags=40)
    fit = fit_robert_variogram(vario)
    if fit is None:
        print("❌ Variogram fit failed")
        return 1
    print(f"   LC = 3r = {fit['LC']:.0f} m ({fit['LC']/1000:.2f} km)")
    print(f"   Lλ = l1 = {fit['L_lambda']:.0f} m ({fit['L_lambda']/1000:.2f} km)")
    vario.to_csv(outdir / f"{args.river_name}_semivariogram.csv", index=False)
    pd.Series(fit).to_csv(outdir / f"{args.river_name}_variogram_fit.csv")

    # --- Wavelets ---
    print("🌊 CWT...")
    scales_m, power, grid = wavelet_cwt(x, z)

    # --- Smooth at LC ---
    print(f"🧹 Savitzky–Golay smooth (window ≈ LC = {fit['LC']:.0f} m)...")
    df = savgol_smooth_lambda(df, fit["LC"])
    print(f"   window = {df.attrs['savgol_window_samples']} samples "
          f"(≈ {df.attrs['savgol_window_m']:.0f} m)")

    # --- LISA ---
    # Smooth at LC (paper). For LISA weights use a local band so HH/LH edges
    # remain detectable on the smoothed series (default ~ few node spacings,
    # or --lisa-band-m / Lλ if set explicitly).
    spacing = float(np.median(np.diff(np.sort(x))))
    if args.lisa_band_m is not None:
        band = float(args.lisa_band_m)
    else:
        # Local contiguity scale: ~3 km or 8 spacings, not full LC
        band = float(min(max(8.0 * spacing, 1500.0), fit["L_lambda"]))
    band = max(band, 2.5 * spacing)
    print(f"🗺️  LISA (DistanceBand = {band:.0f} m) on LC-smoothed Λ...")
    lisa_df, mi, lisa, w = run_lisa(df, df["log_lambda_smooth"].to_numpy(), band_m=band)
    lisa_df = flag_hh_bordered_by_lh(lisa_df, w)
    n_hh = int((lisa_df["lisa_sig_cluster"] == "HH").sum())
    n_haz = int(lisa_df["high_avulsion_zone"].sum())
    print(f"   Global I={mi.I:.3f} (p={mi.p_sim:.3f})  |  HH={n_hh}  |  HH-edge zones={n_haz}")

    lisa_csv = outdir / f"{args.river_name}_lisa_results.csv"
    lisa_df.to_csv(lisa_csv, index=False)
    regions = hh_regions(lisa_df)
    if not regions.empty:
        regions.to_csv(outdir / f"{args.river_name}_hh_regions.csv", index=False)
        print(f"   Contiguous HH regions: {len(regions)}")

    # --- Optional avulsion proximity ---
    if args.avulsion_events:
        ev_path = proj(args.avulsion_events)
        if ev_path.exists():
            events = pd.read_csv(ev_path)
            # merge LISA class onto df for window counts
            df_m = df.merge(
                lisa_df[["node_id", "lisa_sig_cluster"]], on="node_id", how="left"
            )
            prox = proximity_weighted_analysis(df_m, events, fit["LC"])
            if prox is not None:
                prox_path = outdir / f"{args.river_name}_avulsion_proximity.csv"
                prox.to_csv(prox_path, index=False)
                print(f"📍 Proximity analysis → {prox_path}")
                wsum = prox["weight"].sum()
                wmean = (prox["lambda_mean_window"] * prox["weight"]).sum() / wsum
                print(f"   inverse-n weighted mean Λ within ±LC of events: {wmean:.2f}")
        else:
            print(f"⚠️  Avulsion events file not found: {ev_path}")
    else:
        print("ℹ️  No --avulsion-events provided; skipping proximity-weighted comparison.")

    fig_path = plot_all(df, vario, fit, scales_m, power, grid, lisa_df, mi, outdir, args.river_name)
    map_path = plot_map(lisa_df, points_path, outdir, args.river_name)

    summary = summarize(lisa_df, fit, mi, df)
    summary_path = outdir / f"{args.river_name}_hotspot_summary.txt"
    summary_path.write_text(summary + "\n")
    print("\n" + summary)
    print(f"\n💾 {summary_path}")
    print(f"📊 {fig_path}")
    if map_path:
        print(f"🗺️  {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
