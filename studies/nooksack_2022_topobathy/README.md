# Nooksack 2022 lidar + bathymetry — labeler picks (GUI)

Brooke picks channel / ridge / floodplain points **interactively** in the
labeler GUI. This folder is only for **versioned archives** of those picks and
graphs so a new no-levee pass does not overwrite the old one.

There is no auto-picker here.

## Where the picking window is

Open the interactive labeler GUI (`CrossSectionLabeler`):

```bash
# Legacy Nooksack working-tree paths (usual on Brooke’s machine)
poetry run python run_labeler.py

# Config-driven package form
python -m avulsionprecursors label -c config/nooksack_labeler_with_levee.yaml
```

Click **channel → ridge1 → floodplain1 → ridge2 → floodplain2** on each
profile. Magenta marks (if a levee overlay is set) are mapped artificial
levees — for the no-levee pass, skip those crests and pick the broader
alluvial-ridge scale instead.

## Where old 2022 labels / graphs live

They are **local / gitignored**, not in the public repo:

| Artifact | Typical path on Brooke’s machine |
|----------|----------------------------------|
| Label CSVs | `data/labels/Nooksack_node_<id>_labels.csv` |
| Profiles used by the GUI | `cross_section_profiles/cross_section_*.csv` |
| Cross-sections | `cross_sections.shp` (repo root, gitignored) |
| Lambda / DoD tables & plots | `data/Nooksack_lambda_*.csv`, `data/Nooksack_*.png`, Everson `data/Nooksack_Everson_*` |
| 2022 DTM (for later sampling) | `data/2022_dtm/…` |
| DEM shown in the labeler | 2024 topobathy GeoTIFF (OneDrive `…/BathymetryData/Topobathy_reprojected.tif`) |

`data/`, `cross_section_profiles/`, and root `cross_sections.*` are in
`.gitignore`.

## Preserve old picks before a new pass (required)

Copy — do not move or delete — then open the GUI against a **new** study name
and labels directory:

```bash
# From the repo root, with your local data/ present:
python studies/nooksack_2022_topobathy/archive_local_v1.py

# Equivalent manual commands:
mkdir -p studies/nooksack_2022_topobathy/v1_with_levee/labels \
         studies/nooksack_2022_topobathy/v1_with_levee/figures
cp -n data/labels/Nooksack_node_*_labels.csv \
      studies/nooksack_2022_topobathy/v1_with_levee/labels/
cp -n data/Nooksack_*.png data/Nooksack_Everson_* \
      studies/nooksack_2022_topobathy/v1_with_levee/figures/ 2>/dev/null || true

# Also keep a full working copy next to the live labels dir:
cp -a data/labels data/labels_with_levee_v1
```

`cp -n` / the archive script will not overwrite an existing archive file unless
you pass `--force`.

## Open the labeler for a new no-levee pass

Same cross-sections and profiles; **new** river name + labels directory:

```bash
poetry run python run_labeler.py \
  --river-name Nooksack_no_levee \
  --labels-dir data/labels_no_levee \
  --levee /path/to/mapped_levees.shp
```

Or via config:

```bash
python -m avulsionprecursors label -c config/nooksack_labeler_no_levee.yaml
```

That writes `data/labels_no_levee/Nooksack_no_levee_node_<id>_labels.csv`.
Old `data/labels/Nooksack_node_*_labels.csv` and the v1 archive stay untouched.

Optional: copy finished no-levee CSVs into
`studies/nooksack_2022_topobathy/v2_no_levee/labels/` for a checked-in-style
snapshot beside v1. After labeling, run lambda / figure scripts with the
no-levee labels dir and a parallel output name so graphs also coexist.
