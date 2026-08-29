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

from .model import SeerDetector, EMA, detection_loss, load_checkpoint  # noqa: E402,F401
