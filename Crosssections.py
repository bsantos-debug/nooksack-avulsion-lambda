#!/usr/bin/env python3
"""Backward-compatible wrapper. Prefer:

    python -m avulsionprecursors extract -c config/example.yaml

The original Nooksack script is at archive/working_snapshot/Crosssections.py
"""

import sys

from avulsionprecursors.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(1)
    raise SystemExit(main(["extract", *sys.argv[1:]]))
