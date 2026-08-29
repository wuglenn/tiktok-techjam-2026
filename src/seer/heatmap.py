"""Patch-level heatmaps: which parts of an image look AI-generated?

The local head produces one logit per patch; we sigmoid them, reshape to the
patch grid, and bilinearly upsample for a smooth overlay. This mirrors the
composite-training signal (localize AI regions) and is what makes mixed
real/AI images interpretable.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .augment import IMAGENET_MEAN, IMAGENET_STD, eval_transform


@torch.no_grad()
def predict_and_explain(
    model: "torch.nn.Module",
    image: Image.Image,
    res: int,
    device: Optional[torch.device] = None,
) -> Tuple[float, np.ndarray]:
    """Returns (prob_ai, heatmap HxW in [0,1]) - heatmap is None if the
    checkpoint has no per-patch head."""
    device = device or next(model.parameters()).device
    model.eval()
    x = eval_transform(image, res)[None].to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(x)
    prob = torch.sigmoid(out["logits"][0].float()).item()
    patch = out.get("patch_logits")
    if patch is None:
        return prob, None
    patch = torch.sigmoid(patch[0].float())
    G = int(round(np.sqrt(patch.numel())))
    grid = patch.reshape(G, G)[None, None]
    heat = torch.nn.functional.interpolate(
        grid, size=(res, res), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    return prob, heat


def save_heatmap(
    out_path: str,
    image: Image.Image,
    heatmap: np.ndarray,
    prob_ai: float,
    res: int,
    title: Optional[str] = None,
):
    """Render the image with its AI heatmap overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    disp = image.convert("RGB").resize((res, res), Image.BICUBIC)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(disp)
    axes[0].set_title(f"input  (P(AI) = {prob_ai:.3f})")
    axes[0].axis("off")
    axes[1].imshow(disp)
    axes[1].imshow(heatmap, cmap="turbo", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1].set_title(title or "AI heatmap (per-patch)")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
