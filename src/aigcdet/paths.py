"""Canonical project paths.

Defined once so that moving the package does not silently break data
resolution -- every other module imports from here rather than computing
``__file__``-relative offsets of its own.
"""

from __future__ import annotations

from pathlib import Path

# src/aigcdet/paths.py -> src/aigcdet -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"

NTIRE_ROOT = DATA_ROOT / "ntire"
FEATURE_CACHE = DATA_ROOT / "features"


def ensure(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
