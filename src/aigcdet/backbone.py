"""Frozen vision-foundation-model feature extraction.

The backbone is frozen, so features are computed once and cached. Every
downstream experiment -- probe architecture, loss weighting, calibration,
degradation-invariance -- then trains in seconds on the cached matrix instead
of re-running the GPU. That is what makes a wide ablation affordable.

DINOv3 weights are pulled from the ungated ``timm`` mirror, which Meta
explicitly authorised, so no HuggingFace gating approval is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import timm
import torch
from PIL import Image
from torch import nn

# Web-pretrained DINOv3. The `.sat493m` satellite variants share the naming
# convention but are useless here -- they score at chance on fake detection,
# because the forensic signal comes from pre-training exposure to web imagery.
DEFAULT_MODEL = "vit_large_patch16_dinov3.lvd1689m"

FORBIDDEN_SUBSTRINGS = ("sat493m",)


@dataclass
class BackboneConfig:
    model_name: str = DEFAULT_MODEL
    image_size: int = 256
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    # Pool patch tokens rather than using CLS: measured +3.5 points in the
    # FGTS ablation (patch 74.0 vs CLS 70.5), and register tokens are worst.
    pooling: str = "patch_mean"


class FrozenBackbone:
    """Wraps a frozen timm vision transformer as a feature extractor."""

    def __init__(self, config: BackboneConfig | None = None) -> None:
        self.config = config or BackboneConfig()
        if any(bad in self.config.model_name for bad in FORBIDDEN_SUBSTRINGS):
            raise ValueError(
                f"{self.config.model_name!r} is satellite-pretrained and cannot detect "
                "AI-generated images (0.121 accuracy on fakes). Use an lvd1689m variant."
            )

        self.model = timm.create_model(
            self.config.model_name,
            pretrained=True,
            num_classes=0,
        )
        self.model.eval().to(self.config.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        data_config = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(data_config["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(data_config["std"]).view(1, 3, 1, 1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _to_tensor(self, images: Sequence[Image.Image]) -> torch.Tensor:
        size = self.config.image_size
        arrays = [
            np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)
            for img in images
        ]
        batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float().div_(255.0)
        return (batch - self.mean) / self.std

    @torch.inference_mode()
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one embedding per image, as float32 ``(N, D)``."""
        use_amp = self.config.device.startswith("cuda") and self.config.dtype != torch.float32
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), self.config.batch_size):
            chunk = images[start : start + self.config.batch_size]
            batch = self._to_tensor(chunk).to(self.config.device)
            with torch.autocast(device_type="cuda", dtype=self.config.dtype, enabled=use_amp):
                tokens = self.model.forward_features(batch)
            outputs.append(self._pool(tokens).float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:  # already pooled by the model
            return tokens
        num_prefix = getattr(self.model, "num_prefix_tokens", 1)
        if self.config.pooling == "patch_mean":
            return tokens[:, num_prefix:].mean(dim=1)
        if self.config.pooling == "cls":
            return tokens[:, 0]
        if self.config.pooling == "cls_patch":
            return torch.cat([tokens[:, 0], tokens[:, num_prefix:].mean(dim=1)], dim=-1)
        raise ValueError(f"unknown pooling {self.config.pooling!r}")


class LinearProbe(nn.Module):
    """Lightweight head over cached frozen features."""

    def __init__(self, in_features: int, hidden: int | None = 512, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def cache_path(root: Path, tag: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{tag}.npz"


def save_features(path: Path, features: np.ndarray, labels: np.ndarray, names: Iterable[str]) -> None:
    np.savez_compressed(path, features=features, labels=labels, names=np.array(list(names)))


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["features"], data["labels"], data["names"]
