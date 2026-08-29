"""Central data locations.

All heavy artifacts (datasets, mirrors, HF cache) live on F:/techjam so they
never touch the system drive. Override with SEER_DATA_ROOT.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SEER_DATA_ROOT", "F:/techjam"))


def setup() -> Path:
    """Create the data root (if the drive exists) and point the HF cache at
    it. Returns the effective data root."""
    global DATA_ROOT
    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        return DATA_ROOT  # no such drive - fall back to HF defaults elsewhere
    os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf_cache"))
    return DATA_ROOT


def mirrors_dir() -> Path:
    return DATA_ROOT / "mirrors"


def synthbuster_dir() -> Path:
    return DATA_ROOT / "synthbuster"
