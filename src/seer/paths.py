"""Central data locations.

Override with SEER_DATA_ROOT. Defaults to the RunPod network volume
(``/workspace/data``) when that mount exists, otherwise ``F:/techjam``.
"""

import os
from pathlib import Path


def _default_root() -> Path:
    env = os.environ.get("SEER_DATA_ROOT")
    if env:
        return Path(env)
    workspace = Path("/workspace")
    if workspace.is_dir() and os.access(workspace, os.W_OK):
        return workspace / "data"
    return Path("F:/techjam")


DATA_ROOT = _default_root()


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


def ntire_root() -> Path:
    """NTIRE 2026 train/val/test, as fetched by ``get_datasets.py``."""
    return DATA_ROOT / "ntire"


def dda_train_dir() -> Path:
    """DDA-Training-Set after join+extract (fake/ and optionally real/)."""
    return DATA_ROOT / "dda-train"


def sid_set_dir() -> Path:
    return DATA_ROOT / "sid-set"


def gs_images_dir(version: str) -> Path:
    """GAS-Station dump root (``gs-images-v3`` / ``gs-images-v4``)."""
    key = version if version.startswith("gs-images-") else f"gs-images-{version}"
    return DATA_ROOT / key
