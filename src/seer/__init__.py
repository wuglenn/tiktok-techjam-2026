"""Seer: a sub-2B-parameter AI-generated image detector.

Backbone: self-supervised ViT (DINOv3 / DINOv2) fully fine-tuned for
AI-image detection (continuation training), with a dual head:

  * global head  - image-level real/AI logit
  * local head   - per-patch logits (heatmaps, composite training)

TikTok TechJam 2026.
"""

__version__ = "0.1.0"

PARAM_BUDGET = 2_000_000_000  # challenge constraint: < 2B parameters

from . import paths as _paths  # noqa: E402

try:
    _paths.setup()  # data + HF cache on F:/techjam when available
except Exception:
    pass

# Model imports stay lazy so acquisition scripts (get_datasets.py,
# dataset_stats.py) do not pull torch just to read the registry.


def __getattr__(name):
    if name in {"SeerDetector", "EMA", "detection_loss", "load_checkpoint"}:
        from .model import SeerDetector, EMA, detection_loss, load_checkpoint

        exports = {
            "SeerDetector": SeerDetector,
            "EMA": EMA,
            "detection_loss": detection_loss,
            "load_checkpoint": load_checkpoint,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
