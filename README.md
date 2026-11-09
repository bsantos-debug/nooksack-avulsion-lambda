# avulsionprecursors

Tools for extracting river cross-sections from a DEM and computing avulsion
potential **Λ (lambda)** from alluvial-ridge geometry.

The workflow is meant to be reused on new DEMs. Site-specific paths, CRS,
resolution, and thresholds belong in a YAML config (or CLI flags), not in the
processing functions.

## Attribution

Adapted from Jake Gearon's [avulsionprecursors](https://github.com/jameshgrn/avulsionprecursors/tree/first), companion code to Gearon and Edmonds (2025), *Geophysical Research Letters*, https://doi.org/10.1029/2024GL114047.

This version reworks that code into a config-driven DEM workflow and extends the analysis for a new study site. The original scientific definitions of Λ are unchanged.

---

## What the tool does

Given a DEM and a river centerline, it:

1. Reprojects (and optionally clips) the DEM and vectors into one projected CRS
2. Places stations along the centerline
3. Draws cross-sections perpendicular to flow and clips them at valley walls
4. Samples elevation along each transect
5. Uses labeled channel / ridge / floodplain points (or optional auto-picks)
6. Computes **Λ** from alluvial-ridge geometry

It does **not** convert vertical units. Elevations stay as stored in the DEM.
You set elevation thresholds in those same units.

---

## Processing framework and rules

These definitions are the scientific core (Gearon et al.) and are not
site-specific:

| Symbol | Meaning | Definition |
|--------|---------|------------|
| \(H_m\) | Channel depth | ridge crest − channel bed |
| \(H_{ar}\) | Alluvial-ridge height | ridge crest − floodplain elevation |
| \(S_{AR}\) | Ridge aspect slope | \(\lvert H_{ar}\rvert\) / horizontal ridge-to-floodplain distance |
| \(S_m\) | Channel slope | signed downhill slope of the SG-smoothed thalweg long profile |
| \(\gamma\) | | \(S_{AR} / S_m\) |
| \(\beta\) | Superelevation | \(H_{ar} / H_m\) |
| \(\Lambda\) | Avulsion potential | \(\gamma \times \beta\) |

**Lower-ridge rule:** use the bank with the lower crest. If only one ridge is
labeled, use that bank.

**Cross-section geometry:**

- Station points stay on the original centerline
- A Savitzky–Golay smoothed copy of the centerline is used **only** for flow
  azimuth, so raster stair-steps do not rotate the transects
- Transects are perpendicular to the circular-mean azimuth of a node and its
  along-stream neighbors
- Each side is probed to `max_half_length`, then clipped at the first
  *sustained* rise of `valley_wall_dz` above the near-channel elevation

Default `valley_wall_dz` and `max_elev_above_channel` are **30 and 40 metres**,
matching the original workflow. If your DEM is in feet, change those values.

---

## Required software and packages

- Python 3.10+
- GDAL (via rasterio) for raster I/O
- Poetry is the project environment (see `pyproject.toml`)

Main packages: `geopandas`, `rasterio`, `numpy`, `scipy`, `pandas`,
`matplotlib`, `pyyaml`, `pyproj`. The labeling GUI also uses `pyqt5` and
`cartopy`.

---

## Installation

```bash
git clone <your-fork-url>
cd avulsionprecursors
poetry install
```

Or with a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## How to supply a new DEM

1. Copy `config/example.yaml` to a new file (for example `config/my_river.yaml`).
2. Set `paths.dem` to your raster (GeoTIFF `.tif` is typical; other
   GDAL-readable formats such as `.vrt` also work).
3. Set `paths.centerline` to a line shapefile or GeoPackage of the river.
4. Optional: `channel_polygon` (for width), `clip_boundary` (mask),
   `centerline_points` (pre-placed stations), `levee` (mapped artificial
   levees drawn in the picker).
5. Set `crs.target` to `UTM` (auto zone from the data centroid) or an EPSG
   code such as `EPSG:32610`.
6. If a file has **no CRS**, set `crs.dem_source` or `crs.centerline_source`.
7. Set `units.vertical` to `native`, `m`, `ft`, or `us_ft`. This documents the
   DEM; values are **not** converted.
8. Adjust extract thresholds so they match **your** DEM units and reach scale.

Relative paths are resolved from the folder that contains the YAML file.

---

## How to change parameters

Edit the YAML file, or override on the command line:

```bash
python -m avulsionprecursors extract -c config/my_river.yaml --spacing 100 --crs EPSG:32610
```

Common knobs:

- `extract.spacing` — distance between stations
- `extract.cross_half_length` / `max_half_length` — transect length
- `extract.valley_wall_dz` — valley-wall rise (native vertical units)
- `extract.backwater_length` — window for \(S_m\)
- `extract.max_elev_above_channel` — mask high samples, or `null` to disable

---

## How to run the workflow

```bash
python -m avulsionprecursors extract -c config/my_river.yaml
python -m avulsionprecursors label   -c config/my_river.yaml
python -m avulsionprecursors lambda  -c config/my_river.yaml
```

`run` does extract then lambda and skips the GUI (you must already have labels,
or pass `--auto-minmax`):

```bash
python -m avulsionprecursors run -c config/my_river.yaml
```

`lambda` uses the thalweg slope. To keep that result and also compute Λ with
the slope of the most frequent avulsion-routing paths, run:

```bash
python -m avulsionprecursors lambda-routing -c config/my_river.yaml
```

That command starts Gearon et al.'s softmax random walk at the crest of the
lower ridge on each transect, for one station interval (`extract.spacing`, or
`routing.path_length`). The slope used in Λ is the visit-weighted map slope of
steps that land in cells with at least `routing.frequency_threshold` of the
peak visit count. Har, SAR, and the lower-ridge rule are unchanged, so
γ = SAR / S_routing. Results go to `<study>_lambda_routing.csv` and do not
replace `<study>_lambda_results.csv`. Steps back toward the channel are blocked
until the walk has moved past the crest onto the outer bank, or downstream
at least as far as the crest is from the channel. The channel itself stays
closed for the whole walk: the centerline is buffered by half the measured
width, and those cells are treated as nodata. The walk scores slope as rise per pixel, as Gearon's code does
on a ~30 m elevation model. A finer DEM is averaged to `routing.cell_m`
(default 30 m) before the walk, so furrows and lidar noise do not outrank
the valley slope. Set `cell_m` to 0 to walk the native grid. The walk is adapted from
[rulesofriveravulsion](https://github.com/jameshgrn/rulesofriveravulsion).

The labeler asks you to click, in order: channel, ridge1, floodplain1, ridge2,
floodplain2. If `paths.levee` is set, mapped artificial levees are drawn in
magenta on the DEM/satellite maps and as bands on the profile.

`--auto-minmax` sets channel = lowest point and ridges = highest point on each
side. If valley walls are taller than levees, those hills are labeled as
ridges. Auto-minmax does **not** pick floodplains; Λ still needs floodplain
elevations for \(H_{ar}\).

A tiny synthetic example (no research data) is in `examples/synthetic_valley/`.

The November 2026 Nooksack run (2022 bathymetry, 2023 lidar, bankfull
centerline, and the labeler notes) is written up so it can be repeated on
another computer that has the same data files:

`studies/nooksack_bathy2022/README.md`

---

## Outputs

Under `paths.output_dir` (default `outputs/`):

| File | Contents |
|------|----------|
| `prepared/dem.tif` | Reprojected, optionally clipped DEM |
| `prepared/centerline.shp` | Centerline in the working CRS |
| `cross_sections.shp` | Transect lines, with width and slope |
| `centerline_points.shp` | Station points |
| `profiles/cross_section_NNN.csv` | Distance and native elevation along each transect |
| `labels/<study>_node_<id>_labels.csv` | GUI picks |
| `<study>_lambda_results.csv` | \(H_m\), \(H_{ar}\), \(\gamma\), \(\beta\), \(\Lambda\), quality flags |
| `<study>_lambda_plot.png` | Λ vs distance from outlet |

---

## Important assumptions and limitations

- The working CRS must be **projected** (metres or feet), not longitude/latitude.
- Elevations are **not** converted. If XY is metres and Z is feet, \(S_m\) and
  \(S_{AR}\) share that mixed unit and Λ (a ratio) is unchanged, but
  \(H_m\) / \(H_{ar}\) stay in native Z.
- Valley-wall clipping is a geometric heuristic, not a mapped geologic contact.
- The lower-ridge rule is always applied; there is no “mean of both banks” option.
- \(S_m\) is the signed downhill slope of a smoothed thalweg profile. Adverse
  (negative) reaches are flagged and omitted from Λ; values are not floored
  at `slope_min`.
- Discharge, hydraulic models (FlowFM), and multi-date lidar comparison are
  **not** part of this core package.
- Results depend on label quality. Auto-picked ridges can be valley walls.

---

## Tests

```bash
poetry run pytest tests
```

Tests use a small synthetic valley DEM, not research rasters.

---

## Citation

If you use the Λ definitions in research, cite the paper:

Gearon, J. H., & Edmonds, D. A. (2025). River avulsion precursors encoded in alluvial ridge geometry. *Geophysical Research Letters, 52*(8). https://doi.org/10.1029/2024GL114047

If you use this workflow, also cite the code it was adapted from:

Gearon, J. H. avulsionprecursors. https://github.com/jameshgrn/avulsionprecursors/tree/first

If you use the routing-slope lambda, also cite:

Gearon, J. H., Martin, H. K., DeLisle, C., Barefoot, E. A., Mohrig, D., Paola, C., & Edmonds, D. A. (2024). Rules of river avulsion change downstream. *Nature, 634*, 91–95. https://doi.org/10.1038/s41586-024-07964-2

Gearon, J. H. rulesofriveravulsion. https://github.com/jameshgrn/rulesofriveravulsion

and this repository:

https://github.com/bsantos-debug/nooksack-avulsion-lambda

---

## License

MIT. See `LICENSE`.
