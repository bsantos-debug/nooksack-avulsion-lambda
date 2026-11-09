#!/usr/bin/env python3
"""Copy existing local Nooksack label/graph artifacts into v1_with_levee/.

Does not delete or move sources. Skips destinations that already exist unless
--force is passed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
V1 = STUDY / "v1_with_levee"


def _copy_file(src: Path, dst: Path, force: bool) -> str:
    if not src.exists():
        return f"missing  {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return f"kept     {dst.relative_to(STUDY)} (already present)"
    shutil.copy2(src, dst)
    return f"copied   {src} → {dst.relative_to(STUDY)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Default: <data-dir>/labels")
    parser.add_argument(
        "--also-sidecar",
        action="store_true",
        help="Also copy data/labels → data/labels_with_levee_v1 (full dir clone)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data = args.data_dir
    labels_src = args.labels_dir or (data / "labels")
    reports: list[str] = []

    if labels_src.is_dir():
        matched = sorted(labels_src.glob("*_labels.csv"))
        if not matched:
            reports.append(f"empty    {labels_src} (no *_labels.csv)")
        for src in matched:
            reports.append(_copy_file(src, V1 / "labels" / src.name, args.force))
    else:
        reports.append(f"missing  {labels_src}")

    for pattern in (
        "Nooksack_2022_vs_2024_ridge_change.csv",
        "Nooksack_2022_vs_2024_ridge_change_summary.json",
        "Nooksack_2022_vs_2024_profile_ridges.csv",
        "Nooksack_lambda_results.csv",
        "Nooksack_lambda_plot.png",
    ):
        reports.append(_copy_file(data / pattern, V1 / pattern, args.force))

    for src in sorted(data.glob("Nooksack_*xs*.png")) + sorted(data.glob("Nooksack_*xs*.pdf")):
        reports.append(_copy_file(src, V1 / "figures" / src.name, args.force))
    for src in sorted(data.glob("Nooksack_Everson_*")):
        if src.is_file():
            reports.append(_copy_file(src, V1 / "figures" / src.name, args.force))

    if args.also_sidecar and labels_src.is_dir():
        sidecar = data / "labels_with_levee_v1"
        if sidecar.exists() and not args.force:
            reports.append(f"kept     {sidecar} (already present)")
        else:
            if sidecar.exists() and args.force:
                shutil.rmtree(sidecar)
            shutil.copytree(labels_src, sidecar)
            reports.append(f"copied   {labels_src} → {sidecar}")

    print("v1 archive report:")
    for line in reports:
        print(" ", line)
    print()
    print("Next: open the no-levee labeler (writes to a separate labels dir):")
    print("  python -m avulsionprecursors label -c config/nooksack_labeler_no_levee.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
