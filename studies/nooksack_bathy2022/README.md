# Nooksack 2022 bathymetry workflow

This is the November 2026 run: lambda from the 2023 lidar plus 2022 river
bathymetry. Elevations stay in NAVD88 US survey feet. Horizontal coordinates
are metres in EPSG:26910 (NAD83 / UTM zone 10N).

The DEMs, the bankfull polygon, the modern centerline, and the label CSVs are
local data. They are not in git. This repository has the code, the config, the
centerline that was used, and the commands below. On another computer, copy
the same data files and point the commands at those copies.

## Data to copy

Keep the file names. The directories can differ.

| File | Role |
|------|------|
| `nooksack_lidar2023_bathy2022.tif` | 2022 topobathy mosaic. Bathymetry replaces lidar where the mosaic is valid. Feet, NAVD88. |
| `nooksack_lidar2023_bathy2022_with_upstream.vrt` | Same mosaic with 2023 lidar tiles behind it, for gaps upstream of the 2022 mosaic. |
| `Bankfull_flow_extent_2025.shp` | Bankfull polygon. Guides the centerline and is drawn in cyan on the picker. |
| `Modern_Nooksack_centerline_clean.shp` | Modern thalweg. Used only downstream of the bankfull polygon, and to choose which end of the polygon is upstream. |
| `studies/nooksack_bathy2022/centerline/centerline_bankfull_and_lower.shp` | Centerline used for the transects. Already in this repo. |

On the machine where this run was made, those rasters and shapefiles were:

- `/Volumes/Extreme SSD/Nooksack_data/DEMs/nooksack_lidar2023_bathy2022.tif`
- `/Volumes/Extreme SSD/Nooksack_data/DEMs/nooksack_lidar2023_bathy2022_with_upstream.vrt`
- `.../Data/Washington/Nooksack/Model_outputs/Bankfull_flow_extent_2025.shp`
- `.../Nooksack_research/Qgis/Crosssections/Modern_Nooksack_centerline_clean.shp`

`config/nooksack_bathy2022.yaml` still contains those paths for `paths.dem`
and `paths.channel_polygon`. Change those two paths on the other computer.
Leave the elevations in feet. Do not set `max_elev_above_channel`.

## Install

```bash
git clone https://github.com/bsantos-debug/nooksack-avulsion-lambda.git
cd nooksack-avulsion-lambda
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`. The labeler needs a GUI
(`MPLBACKEND=Qt5Agg` is set in the command below).

## 1. Centerline

The committed centerline is the one the picks were made on. Use it. Rebuild
it only if the bankfull polygon or the modern centerline changes:

```bash
python examples/prepare_nooksack_lidar2023.py \
  --modern /path/to/Modern_Nooksack_centerline_clean.shp \
  --bankfull /path/to/Bankfull_flow_extent_2025.shp \
  --centerline-out studies/nooksack_bathy2022/centerline/centerline_bankfull_and_lower.shp \
  --centerline-only
```

Through the bankfull polygon the line is the polygon medial axis (boundary
sampled every 20 m, edges within 15 m of the bank dropped, shortest path
between the thalweg entry and exit, then smoothed over 400 m). Downstream of
the polygon it follows the modern thalweg to the mouth. The upstream part of
the modern thalweg, before it meets the polygon, is not used. The line must
not cross itself.

## 2. 2022 corridor

Warp the 2022 mosaic to a 3 m UTM corridor. Do not warp the full county grid
into memory.

```bash
python examples/prepare_nooksack_lidar2023.py \
  --modern /path/to/Modern_Nooksack_centerline_clean.shp \
  --bankfull /path/to/Bankfull_flow_extent_2025.shp \
  --dem /path/to/nooksack_lidar2023_bathy2022.tif \
  --centerline-out studies/nooksack_bathy2022/centerline/centerline_bankfull_and_lower.shp \
  --dem-out /path/to/nooksack_lidar2023_bathy2022_corridor_utm10_3m.tif
```

This overwrites the centerline with the same construction as step 1, then
writes the corridor. The corridor is buffered 4200 m from the centerline.
Cells are 3 m. Resampling is average. Nodata is -9999. Elevations stay in
US survey feet.

## 3. Fill upstream gaps

Put the 2023 lidar behind the 2022 corridor. Where the 2022 corridor already
has an elevation, that value is kept. Only empty cells are filled. In the
upstream VRT the 2022 mosaic is the last source, so bathymetry stays on top
wherever it is valid.

```bash
python examples/fill_nooksack_upstream.py \
  --corridor /path/to/nooksack_lidar2023_bathy2022_corridor_utm10_3m.tif \
  --upstream /path/to/nooksack_lidar2023_bathy2022_with_upstream.vrt \
  --output /path/to/nooksack_lidar2023_bathy2022_corridor_utm10_3m_upstream_fill.tif
```

Point `paths.dem` in `config/nooksack_bathy2022.yaml` at that filled file.

## 4. Extract transects

```bash
python -m avulsionprecursors extract -c config/nooksack_bathy2022.yaml
```

Stations are every 200 m. Each transect is perpendicular to the station, not
to a circular mean of neighboring stations (`azimuth_include_neighbors: false`).
Each side is probed to 4000 m and stops where the ground stays 98 ft above the
near-channel elevation for 50 m. The first 200 m are ignored so near-channel
bars are not treated as valley walls, and 100 m is kept past the rise. The
shortest half-length is 800 m. `max_elev_above_channel` stays null so the
banks are not blanked. Profiles are sampled every 5 m. Thalweg slope uses a
15000 m window.

Products land in `outputs/nooksack_bathy2022/` (gitignored):

- `cross_sections.shp`
- `profiles/cross_section_NNN.csv`
- `prepared/dem.tif`

## 5. Pick channel, ridges, and floodplains

```bash
MPLBACKEND=Qt5Agg python -m avulsionprecursors label -c config/nooksack_bathy2022.yaml --no-skip
```

Click in order: channel, ridge1, floodplain1, ridge2, floodplain2.

- `s` skips the current pick. Skipping a ridge also skips its floodplain.
- `u` undoes the last click.
- `d` saves the node and opens the next one. Closing the window does not save.
- Cyan is the bankfull polygon on the map and where that polygon crosses the profile.
- One ridge and its floodplain are enough. Leave the other bank blank if it is not there.

How the 5 November 2026 points were chosen:

- Elevations are the 2022 bathymetry, the 2022 lidar, and the 2023 upstream lidar.
- The bankfull extent was used to guide the ridge points.
- The ridge point is the highest part of the alluvial ridge, including levees.
- The floodplain point is the area with the more uniform slope.
- Near glacial outwash channels, the alluvial ridges were difficult to choose.

Labels are `outputs/nooksack_bathy2022/labels/nooksack_bathy2022_node_<id>_labels.csv`.
The picker also writes `labels/README.md`. Text under `## Notes` is kept if
the picker is opened again.

## 6. Lambda

Run this after the labels exist:

```bash
python -m avulsionprecursors lambda -c config/nooksack_bathy2022.yaml
```

The lower ridge is the bank with the lower crest. A missing bank is ignored.

| Symbol | Meaning |
|--------|---------|
| \(H_m\) | ridge crest − channel bed (feet) |
| \(H_{ar}\) | ridge crest − floodplain (feet) |
| \(S_{AR}\) | \(\lvert H_{ar}\rvert\) / ridge-to-floodplain distance |
| \(S_m\) | downhill slope of the smoothed thalweg |
| \(\gamma\) | gradient advantage, \(S_{AR} / S_m\) |
| \(\beta\) | superelevation, \(H_{ar} / H_m\) |
| \(\Lambda\) | \(\gamma \times \beta\) |

\(S_{AR}\) and \(S_m\) are both feet per metre, so \(\gamma\) and \(\Lambda\) are
dimensionless. Nodes with no channel point, or without at least one ridge and
its floodplain, are omitted. Empty label files are omitted.

The table is `outputs/nooksack_bathy2022/nooksack_bathy2022_lambda_results.csv`.

## 7. Figures

```bash
python examples/plot_nooksack_downstream.py
python examples/plot_nooksack_downstream.py --map
```

`--map` downloads Esri World Imagery in Web Mercator (EPSG:3857) and draws the
picks in that same coordinate system. A yellow circle is the channel, a red
cross is a ridge, and a blue square is a floodplain. The white dashed segment
runs between the outermost picks on that transect.

Figures are written next to the lambda table and copied into `labels/`.

## What stays on the local disk

`outputs/` and the GeoTIFFs are gitignored. A fresh clone has the code and the
centerline, not the DEM, the profiles, or the picks. Copy
`outputs/nooksack_bathy2022/labels/` yourself if the other computer should
start from these picks instead of labeling again. Do not copy the older
`labels_looped_centerline/` folder into `labels/`. Those picks belong to a
centerline that crossed itself.
