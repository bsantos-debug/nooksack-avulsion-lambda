# v2 — no-levee (new GUI pass)

After archiving v1, open:

```bash
poetry run python run_labeler.py \
  --river-name Nooksack_no_levee \
  --labels-dir data/labels_no_levee
```

Live labels write to `data/labels_no_levee/Nooksack_no_levee_node_*_labels.csv`.
Copy them here when you want a versioned snapshot beside v1:

```bash
cp -n data/labels_no_levee/*_labels.csv \
  studies/nooksack_2022_topobathy/v2_no_levee/labels/
```

Pick broader alluvial-ridge / natural-bank features; skip magenta artificial
levee marks when the levee overlay is configured.
