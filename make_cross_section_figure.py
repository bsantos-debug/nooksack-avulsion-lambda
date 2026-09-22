#!/usr/bin/env python3
"""Smoothed cross-section SVG/PDF/PNG matching the unlabeled XS 151 style."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

root = Path(__file__).resolve().parent
csv_path = root / "data" / "custom_cross_section.csv"
out_stem = root / "data" / "cross_section_figure"

raw = csv_path.read_text().strip().splitlines()
if raw and raw[0].lower().startswith(("distance", "elevation", "x")):
    raw = raw[1:]
rows = []
for line in raw:
    parts = line.replace(",", " ").split()
    if not parts:
        continue
    rows.append([float(v) for v in parts])
xy = np.asarray(rows, dtype=float)
if xy.shape[1] == 1:
    z = xy[:, 0]
    x = np.arange(len(z), dtype=float) * 6.845037707524786
else:
    x, z = xy[:, 0], xy[:, 1]
m = np.isfinite(x) & np.isfinite(z)
x, z = x[m], z[m]
o = np.argsort(x)
x, z = x[o], z[o]

win = min(25, len(z) if len(z) % 2 == 1 else len(z) - 1)
win = max(5, win if win % 2 == 1 else win - 1)
zs = savgol_filter(z, win, 2, mode="interp")

i_ch = int(np.argmin(zs))
xc, zc = float(x[i_ch]), float(zs[i_ch])
xrel = x - xc


def bench(a, b):
    sel = (xrel >= a) & (xrel <= b)
    if not np.any(sel):
        return np.nan
    return float(np.nanpercentile(zs[sel], 25))


z_fp = np.nanmean([bench(-800, -400), bench(400, 800)])
if not np.isfinite(z_fp):
    z_fp = float(np.nanpercentile(zs[np.abs(xrel) > 400], 25))
h = zs - z_fp
h_ch = zc - z_fp

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 9,
})

fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor="none")
ax.set_facecolor("none")
ax.plot(xrel, h, color="#1c2833", lw=1.8, solid_capstyle="round", zorder=2)
ax.set_xlim(float(xrel.min()) - 40, float(xrel.max()) + 40)
pad = 0.08 * (float(h.max()) - float(h.min()))
ax.set_ylim(float(h.min()) - pad, float(h.max()) + pad)
ax.set_xlabel("Distance from channel (m)")
ax.set_ylabel("Height above floodplain (m)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

for ext in (".png", ".pdf", ".svg"):
    fig.savefig(out_stem.with_suffix(ext), dpi=300, transparent=True, bbox_inches="tight")

print(f"n={len(z)}  channel_x={xc:.2f}  z_fp={z_fp:.3f}  h_ch={h_ch:.2f}")
print("wrote", out_stem.with_suffix(".svg"))
