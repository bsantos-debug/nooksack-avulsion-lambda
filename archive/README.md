# Archived files

These files are **not** used by the reusable DEM → lambda package. They remain so the
pre-generalization Nooksack workflow can be recovered.

A git snapshot of the working tree before refactoring is commit `2f2764c`.

## Reusable workflow (current)

```bash
python -m avulsionprecursors extract -c config/example.yaml
python -m avulsionprecursors label   -c config/example.yaml
python -m avulsionprecursors lambda  -c config/example.yaml
```

## What’s in this archive

| path | contents |
|------|----------|
| `working_snapshot/` | Root scripts and analysis modules as of the pre-refactor snapshot |
| `legacy/` | Older XGBoost lambda, SWORD/GEE `main.py`, SWORD importers, d50/shear scripts |
| `docs/` | BASED requirements, SWORD import guide + product PDF |

Nooksack-only lidar comparison scripts also live in `local/` on this machine
(gitignored; not for the public repository).

## 2022 lidar + bathy labeler (interactive)

Versioned pick archives and GUI instructions:
`studies/nooksack_2022_topobathy/`.

Configs:

- `config/nooksack_labeler_with_levee.yaml` — original labels dir (`data/labels`)
- `config/nooksack_labeler_no_levee.yaml` — parallel no-levee pass (`data/labels_no_levee`)
