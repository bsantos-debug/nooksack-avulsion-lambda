#!/usr/bin/env python3
"""Write example labels after extract, using the known synthetic levee geometry."""

from pathlib import Path

from avulsionprecursors.config import load_config
from avulsionprecursors.io.synthetic import write_synthetic_labels

HERE = Path(__file__).resolve().parent


def main() -> None:
    cfg = load_config(HERE / "config.yaml")
    n = write_synthetic_labels(cfg.profiles_dir(), cfg.study_name, cfg.labels_dir())
    print(f"Wrote {n} label files in {cfg.labels_dir()}")


if __name__ == "__main__":
    main()
