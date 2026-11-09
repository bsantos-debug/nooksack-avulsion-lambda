"""Downstream figures for a finished Nooksack lambda table.

Reads ``<study>_lambda_results.csv`` and writes:

- superelevation, gradient advantage, and lambda
- superelevation and gradient advantage on one plot with two y axes
- ridge half-width (ridge crest to floodplain pick, lower ridge)

Pass ``--map`` to also draw the picks on satellite imagery. That download
needs a network connection.
"""

from __future__ import annotations

import argparse
import io
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW = 11


def _downstream_km(df: pd.DataFrame, cross_sections: Path | None) -> pd.Series:
    upstream = float(df["dist_out"].max())
    if cross_sections is not None and cross_sections.exists():
        import geopandas as gpd

        xs = gpd.read_file(cross_sections)
        if "dist_out" in xs.columns and xs["dist_out"].notna().any():
            upstream = float(xs["dist_out"].max())
    return (upstream - df["dist_out"]) / 1000.0


def _load(results: Path, cross_sections: Path | None) -> pd.DataFrame:
    df = pd.read_csv(results)
    use1 = df["ridge_bank_used"].astype(int) == 1
    df["half_width_m"] = np.where(
        use1,
        np.abs(df["ridge1_dist_along"] - df["floodplain1_dist_along"]),
        np.abs(df["ridge2_dist_along"] - df["floodplain2_dist_along"]),
    )
    df["down_km"] = _downstream_km(df, cross_sections)
    df = df.sort_values("down_km")
    for src, dest in (
        ("superelevation_mean", "beta_med"),
        ("gamma_mean", "gamma_med"),
        ("lambda", "lambda_med"),
        ("half_width_m", "half_med"),
    ):
        df[dest] = df[src].rolling(WINDOW, center=True, min_periods=5).median()
    return df


def _style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")


def plot_metrics(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.2), dpi=200, sharex=True)
    specs = [
        ("superelevation_mean", "beta_med", r"Superelevation $\beta = H_{ar}/H_m$", False, "#1f4e79"),
        ("gamma_mean", "gamma_med", r"Gradient advantage $\gamma = S_{AR}/S_m$", True, "#b85c38"),
        ("lambda", "lambda_med", r"Avulsion potential $\Lambda = \gamma \times \beta$", False, "#1f4e79"),
    ]
    for ax, (col, med, ylabel, logy, color) in zip(axes, specs):
        ax.scatter(df["down_km"], df[col], s=16, color=color, alpha=0.65, linewidths=0, zorder=2, label="Transect")
        ax.plot(df["down_km"], df[med], color="#111111", lw=1.6, zorder=3, label="Median of 11 transects")
        if logy:
            ax.set_yscale("log")
        _style(ax, ylabel)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[0].axhline(0, color="#666666", lw=0.8, zorder=1)
    axes[2].axhline(2, color="#8c2f2f", lw=1.4, ls="--", zorder=1, label=r"$\Lambda = 2$")
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Distance downstream (km)", fontsize=12)
    axes[-1].set_xlim(df["down_km"].min() - 0.4, df["down_km"].max() + 0.4)
    fig.suptitle("Nooksack, lower ridge, 2022 bathymetry picks", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_beta_gamma(df: pd.DataFrame, out: Path) -> None:
    fig, ax_b = plt.subplots(figsize=(8.8, 5.2), dpi=200)
    ax_g = ax_b.twinx()
    b_pts = ax_b.scatter(df["down_km"], df["superelevation_mean"], s=16, color="#1f4e79", alpha=0.65, linewidths=0, zorder=3, label="Superelevation")
    b_line, = ax_b.plot(df["down_km"], df["beta_med"], color="#1f4e79", lw=1.8, zorder=4, label="Superelevation, median of 11")
    g_pts = ax_g.scatter(df["down_km"], df["gamma_mean"], s=16, color="#b85c38", alpha=0.65, linewidths=0, zorder=2, label="Gradient advantage")
    g_line, = ax_g.plot(df["down_km"], df["gamma_med"], color="#b85c38", lw=1.8, zorder=4, label="Gradient advantage, median of 11")
    ax_g.set_yscale("log")
    ax_b.axhline(0, color="#666666", lw=0.8, zorder=1)
    ax_b.set_xlabel("Distance downstream (km)", fontsize=12)
    ax_b.set_ylabel(r"Superelevation $\beta = H_{ar}/H_m$", fontsize=11, color="#1f4e79")
    ax_g.set_ylabel(r"Gradient advantage $\gamma = S_{AR}/S_m$", fontsize=11, color="#b85c38")
    ax_b.tick_params(axis="y", colors="#1f4e79")
    ax_g.tick_params(axis="y", colors="#b85c38")
    ax_b.set_xlim(df["down_km"].min() - 0.4, df["down_km"].max() + 0.4)
    ax_b.set_ylim(-0.9, 1.4)
    ax_b.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax_b.set_title("Nooksack, lower ridge, 2022 bathymetry picks", fontsize=13)
    ax_b.legend([b_pts, b_line, g_pts, g_line], [h.get_label() for h in (b_pts, b_line, g_pts, g_line)], loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_half_width(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=200)
    ax.scatter(df["down_km"], df["half_width_m"], s=16, color="#1f4e79", alpha=0.7, linewidths=0, zorder=2, label="Lower ridge")
    ax.plot(df["down_km"], df["half_med"], color="#111111", lw=1.8, zorder=3, label="Median of 11 transects")
    ax.set_xlabel("Distance downstream (km)", fontsize=12)
    ax.set_ylabel("Ridge half-width, ridge to floodplain (m)", fontsize=11)
    ax.set_xlim(df["down_km"].min() - 0.4, df["down_km"].max() + 0.4)
    ax.set_ylim(0, float(df["half_width_m"].max()) * 1.08)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Nooksack, lower ridge, 2022 bathymetry picks", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_map(cross_sections: Path, labels_dir: Path, out: Path) -> None:
    import geopandas as gpd
    import pyproj
    from PIL import Image
    from shapely.geometry import LineString

    xs = gpd.read_file(cross_sections)
    rows = []
    segments = []
    for _, row in xs.iterrows():
        node = int(row.node_id)
        path = labels_dir / f"nooksack_bathy2022_node_{node}_labels.csv"
        if not path.exists() or path.stat().st_size < 10:
            continue
        try:
            lab = pd.read_csv(path)
        except Exception:
            continue
        if lab.empty or "label" not in lab.columns:
            continue
        line = row.geometry
        placed = []
        for rec in lab.itertuples(index=False):
            label = str(rec.label)
            if label not in {"channel", "ridge1", "ridge2", "floodplain1", "floodplain2"}:
                continue
            dist = float(np.clip(rec.dist_along, 0.0, line.length))
            pt = line.interpolate(dist)
            rows.append({"label": label, "x": pt.x, "y": pt.y})
            placed.append(dist)
        if len(placed) >= 2:
            lo, hi = min(placed), max(placed)
            segments.append(LineString([line.interpolate(lo), line.interpolate(hi)]))
    pts = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]), crs=xs.crs).to_crs(3857)
    segs = gpd.GeoDataFrame(geometry=segments, crs=xs.crs).to_crs(3857)
    minx, miny, maxx, maxy = pts.total_bounds
    pad_x = (maxx - minx) * 0.04
    pad_y = (maxy - miny) * 0.04
    west, south, east, north = minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y
    req_w = 2600
    req_h = max(2, int(round(req_w * (north - south) / (east - west))))
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{req_w},{req_h}",
        "format": "png",
        "f": "image",
    }
    url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "avulsionprecursors"})
    with urllib.request.urlopen(request, timeout=120) as response:
        image = np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"))
    fig_w = 10.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * image.shape[0] / image.shape[1]), dpi=140)
    ax.imshow(image, extent=(west, east, south, north), origin="upper", interpolation="bilinear", zorder=0)
    for geom in segs.geometry:
        x, y = geom.xy
        ax.plot(x, y, color="white", lw=0.8, ls=(0, (1.4, 1.5)), solid_capstyle="round", zorder=2)

    def scatter(mask, **kwargs):
        sub = pts.loc[mask]
        ax.scatter(sub.geometry.x, sub.geometry.y, zorder=3, **kwargs)

    scatter(pts["label"].str.startswith("floodplain"), marker="s", s=16, c="#2f6fdb", edgecolors="black", linewidths=0.35, label="Floodplain")
    scatter(pts["label"].str.startswith("ridge"), marker="x", s=26, c="#e10600", linewidths=1.0, label="Alluvial ridge")
    scatter(pts["label"].eq("channel"), marker="o", s=18, c="#ffe14a", edgecolors="black", linewidths=0.4, label="Channel")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")
    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    midy = (south + north) / 2
    midx = (west + east) / 2
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _pos: f"{transformer.transform(x, midy)[0]:.2f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: f"{transformer.transform(midx, y)[1]:.2f}"))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Nooksack River")
    legend = ax.legend(loc="upper right", frameon=True, fontsize=8, markerscale=1.5)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.92)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/nooksack_bathy2022/nooksack_bathy2022_lambda_results.csv"),
    )
    parser.add_argument(
        "--cross-sections",
        type=Path,
        default=Path("outputs/nooksack_bathy2022/cross_sections.shp"),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("outputs/nooksack_bathy2022/labels"),
    )
    parser.add_argument("--map", action="store_true", help="Also draw picks on Esri World Imagery")
    args = parser.parse_args()
    df = _load(args.results, args.cross_sections)
    out_dir = args.results.parent
    names = {
        "nooksack_bathy2022_downstream_metrics.png": plot_metrics,
        "nooksack_bathy2022_superelevation_gradient.png": plot_beta_gamma,
        "nooksack_bathy2022_ridge_half_width.png": plot_half_width,
    }
    written = []
    for name, fn in names.items():
        path = out_dir / name
        fn(df, path)
        written.append(path)
    if args.map:
        path = out_dir / "ridge_floodplain_points_map.png"
        plot_map(args.cross_sections, args.labels_dir, path)
        written.append(path)
    if args.labels_dir.exists():
        for path in written:
            target = args.labels_dir / path.name
            target.write_bytes(path.read_bytes())
            print(f"Copied {target}")


if __name__ == "__main__":
    main()
