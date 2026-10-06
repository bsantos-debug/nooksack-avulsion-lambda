#!/usr/bin/env python3
"""Create the tiny synthetic DEM, centerline, and this folder's config targets."""

from pathlib import Path

from avulsionprecursors.io.synthetic import write_synthetic_valley

HERE = Path(__file__).resolve().parent


def main() -> None:
    paths = write_synthetic_valley(HERE)
    print(f"Wrote {paths['dem']}")
    print(f"Wrote {paths['centerline']}")
    print("Next:")
    print("  python -m avulsionprecursors extract -c examples/synthetic_valley/config.yaml")
    print("  python examples/synthetic_valley/write_labels.py")
    print("  python -m avulsionprecursors lambda  -c examples/synthetic_valley/config.yaml")


if __name__ == "__main__":
    main()
