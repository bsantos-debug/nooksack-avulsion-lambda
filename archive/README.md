# Archived files

Not used by the current Nooksack topobathy → lambda workflow.

## Current workflow (repo root)

1. `Crosssections.py` — build cross-sections + elevation profiles
2. `run_labeler.py` — optional interactive labeling
3. `calculate_lambda_bathymetry.py` — lambda from profile min/max + topobathy
4. `plot_lambda_on_dem.py` — map lambda on DEM

Supporting package: `avulsionprecursors/`
Outputs: `data/`, `cross_sections.*`, `centerline_points_from_xs.*`, `cross_section_profiles/`

## What’s in this archive

| path | contents |
|------|----------|
| `legacy/` | Old XGBoost lambda, SWORD/GEE `main.py`, SWORD importers, d50/shear scripts, older XS tools |
| `docs/` | BASED requirements, SWORD import guide + product PDF |
| `data/` | Older bathymetry raster (`bathy_15_ft_int.tif`) |
