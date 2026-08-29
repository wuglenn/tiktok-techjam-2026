"""Patch-level heatmaps: which parts of an image look AI-generated?

The local head produces one logit per patch; we sigmoid them, reshape to the
patch grid, and bilinearly upsample for a smooth overlay. This mirrors the
composite-training signal (localize AI regions) and is what makes mixed
real/AI images interpretable.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .augment import IMAGENET_MEAN, IMAGENET_STD, eval_transform


def denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalize. `images` is (B, 3, H, W) or (3, H, W)."""
    squeeze = images.dim() == 3
    if squeeze:
        images = images.unsqueeze(0)
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = (images * std + mean).clamp(0.0, 1.0)
    return out[0] if squeeze else out


def patch_logits_to_heat(patch_logits: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """(B, P) or (P,) logits -> (B, H, W) heatmap in [0, 1]."""
    if patch_logits.dim() == 1:
        patch_logits = patch_logits.unsqueeze(0)
    patch = torch.sigmoid(patch_logits.float())
    G = int(round(patch.shape[-1] ** 0.5))
    grid = patch.view(patch.shape[0], 1, G, G)
    return torch.nn.functional.interpolate(
        grid, size=size, mode="bilinear", align_corners=False
    )[:, 0]


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


def _pick_indices(labels: torch.Tensor, n: int) -> list:
    """Prefer a real/fake mix so the grid is not one class."""
    y = labels.detach().float().cpu().view(-1)
    real = (y < 0.5).nonzero(as_tuple=True)[0].tolist()
    fake = (y >= 0.5).nonzero(as_tuple=True)[0].tolist()
    n_each = max(1, n // 2)
    idx = real[:n_each] + fake[:n_each]
    if len(idx) < n:
        seen = set(idx)
        idx.extend(i for i in range(len(y)) if i not in seen)
    return idx[:n]


def save_batch_heatmaps(
    out_path: str,
    images: torch.Tensor,
    patch_logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    logits: Optional[torch.Tensor] = None,
    max_n: int = 8,
    title: Optional[str] = None,
) -> str:
    """Grid of training/eval images with predicted AI heatmaps overlaid.

    `images` are ImageNet-normalized (B, 3, H, W). Writes one PNG and
    returns its path. Also writes per-image `{stem}_{i:02d}_{real|fake}.png`
    next to the grid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if patch_logits is None:
        raise ValueError("checkpoint has no patch logits")

    B = images.shape[0]
    n = min(max_n, B)
    idx = _pick_indices(labels if labels is not None else torch.zeros(B), n)
    H, W = int(images.shape[-2]), int(images.shape[-1])
    disp = denormalize_imagenet(images.detach().float().cpu()[idx])
    heat = patch_logits_to_heat(patch_logits.detach().cpu()[idx], (H, W)).numpy()
    probs = (
        torch.sigmoid(logits.detach().float().cpu()[idx]).tolist()
        if logits is not None
        else [float("nan")] * n
    )
    y = (
        labels.detach().float().cpu().view(-1)[idx].tolist()
        if labels is not None
        else [float("nan")] * n
    )

    ncols = 2
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 2.2 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        rgb = disp[i].permute(1, 2, 0).numpy()
        tag = "fake" if y[i] >= 0.5 else "real"
        p = probs[i]
        p_s = f"P(AI)={p:.3f}" if p == p else ""
        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f"{tag}  {p_s}".strip())
        axes[i][0].axis("off")
        axes[i][1].imshow(rgb)
        axes[i][1].imshow(heat[i], cmap="turbo", alpha=0.55, vmin=0.0, vmax=1.0)
        axes[i][1].set_title("AI heatmap")
        axes[i][1].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    stem = Path(out_path)
    for i in range(n):
        rgb = (disp[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        pil = Image.fromarray(rgb)
        tag = "fake" if y[i] >= 0.5 else "real"
        save_heatmap(
            str(stem.with_name(f"{stem.stem}_{i:02d}_{tag}.png")),
            pil,
            heat[i],
            float(probs[i]) if probs[i] == probs[i] else 0.0,
            H,
        )
    return out_path
