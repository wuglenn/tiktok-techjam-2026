"""Training loop with live augmentation and a degradation-consistency loss.

Two losses, following the third-place NTIRE 2026 entry:

``BCE`` on the degraded view, so the classifier is optimised on the
distribution it will actually be scored on.

``consistency`` pulling the representation of a degraded view toward the
representation of its own clean counterpart. This is the part that buys
robustness *without* trading away clean accuracy, which plain heavy
augmentation does not manage on its own.

The backbone can be frozen (fast, cheap, good baseline) or partially unfrozen
(the last N blocks). Unfreezing is only advisable on bias-controlled data --
on biased data it reliably finds a shortcut and overwrites the pretrained
forensic prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import AIGCDataset, collate


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 0.01
    consistency_weight: float = 0.25    # 0 disables the consistency term
    label_smoothing: float = 0.05
    unfreeze_last_n: int = 0            # 0 = fully frozen backbone
    num_workers: int = 8
    amp: bool = True
    grad_clip: float = 1.0
    log_every: int = 20
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


class DetectorHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def smoothed_bce(logits: torch.Tensor, targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon > 0:
        targets = targets * (1.0 - epsilon) + 0.5 * epsilon
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def set_trainable_blocks(backbone: nn.Module, last_n: int) -> list[nn.Parameter]:
    """Unfreeze the final ``last_n`` transformer blocks plus the final norm."""
    for param in backbone.parameters():
        param.requires_grad_(False)
    if last_n <= 0:
        return []

    blocks = getattr(backbone, "blocks", None)
    trainable: list[nn.Parameter] = []
    if blocks is not None:
        for block in list(blocks)[-last_n:]:
            for param in block.parameters():
                param.requires_grad_(True)
                trainable.append(param)
    for name in ("norm", "fc_norm"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad_(True)
                trainable.append(param)
    return trainable


def train(
    backbone_wrapper,
    head: nn.Module,
    dataset: AIGCDataset,
    config: TrainConfig | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> nn.Module:
    """Train ``head`` (and optionally the last backbone blocks) on ``dataset``."""
    cfg = config or TrainConfig()
    device = torch.device(cfg.device)
    backbone = backbone_wrapper.model.to(device)
    head = head.to(device)

    backbone_params = set_trainable_blocks(backbone, cfg.unfreeze_last_n)
    backbone.train(mode=bool(backbone_params))

    groups = [{"params": head.parameters(), "lr": cfg.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    optimiser = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    steps = cfg.epochs * max(1, len(loader))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    use_pairs = dataset.return_pair and cfg.consistency_weight > 0

    def embed(images: torch.Tensor, grad: bool) -> torch.Tensor:
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            tokens = backbone.forward_features(images)
            return backbone_wrapper._pool(tokens)

    for epoch in range(cfg.epochs):
        dataset.set_epoch(epoch)
        totals = {"loss": 0.0, "bce": 0.0, "consistency": 0.0, "n": 0}

        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                features = embed(images, grad=True)
                logits = head(features)
                bce = smoothed_bce(logits, labels, cfg.label_smoothing)

                consistency = torch.zeros((), device=device)
                if use_pairs:
                    clean = batch["clean"].to(device, non_blocking=True)
                    # The clean view is a target, not a second training signal:
                    # detached so the model moves degraded features toward clean
                    # ones rather than meeting somewhere in the middle.
                    clean_features = embed(clean, grad=bool(backbone_params)).detach()
                    # Cosine distance, not MSE on normalised vectors. The latter
                    # averages over the feature dimension, so for a 1024-d
                    # embedding it sits around 1e-4 and the term is inert next
                    # to a BCE of order 0.5 -- it silently does nothing.
                    consistency = (
                        1.0 - nn.functional.cosine_similarity(features, clean_features, dim=-1)
                    ).mean()

                loss = bce + cfg.consistency_weight * consistency

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    [p for group in groups for p in group["params"]], cfg.grad_clip
                )
            scaler.step(optimiser)
            scaler.update()
            schedule.step()

            batch_size = labels.numel()
            totals["loss"] += float(loss) * batch_size
            totals["bce"] += float(bce) * batch_size
            totals["consistency"] += float(consistency) * batch_size
            totals["n"] += batch_size

            if cfg.log_every and step % cfg.log_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(loader)}  "
                    f"loss={float(loss):.4f} bce={float(bce):.4f} cons={float(consistency):.4f}",
                    flush=True,
                )

        stats = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "n"}
        print(f"epoch {epoch}: " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()), flush=True)
        if on_epoch_end:
            on_epoch_end(epoch, stats)

    head.eval()
    backbone.eval()
    return head


@torch.inference_mode()
def predict(backbone_wrapper, head: nn.Module, images, batch_size: int = 32) -> np.ndarray:
    """Probability of being AI-generated, for a list of PIL images."""
    device = next(head.parameters()).device
    scores: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        batch = backbone_wrapper._to_tensor(chunk).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            tokens = backbone_wrapper.model.forward_features(batch)
            logits = head(backbone_wrapper._pool(tokens))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(scores)
