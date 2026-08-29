"""Torch dataset with on-the-fly degradation augmentation.

Augmenting inside the training loop rather than precomputing fixed views
matters for two reasons. Each image is seen under a *different* random
degradation every epoch, which is far more diverse than a handful of cached
views for the same storage cost. And it is the only option once the backbone
is being fine-tuned, because cached features are frozen by construction.

Two properties are load-bearing and easy to get wrong:

*Pair synchronisation.* In paired mode the clean and degraded views come from
the same source image and share the same geometric base transform, so the only
difference between them is the degradation. If real and fake images received
independently drawn augmentation, the augmentation itself would leak class
information and the model would learn that instead.

*Degrade before resize.* Degradations are applied at native resolution and the
resize to the backbone's input size happens afterwards. Reversing that order
would destroy the artifacts being modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import Sample
from .degradations import compound_chain

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class AugmentConfig:
    """Ranges are deliberately wider than the evaluation grid.

    The published recipes we build on bottom out well short of what we are
    scored on -- DDA never trains below JPEG Q55 and Community Forensics never
    below Q75, and neither covers Gaussian noise at all, while the target grid
    goes to Q30 and sigma 0.10. Training strictly harder than the test
    distribution is also what the NTIRE top entries did.
    """

    enabled: bool = True
    probability: float = 0.9        # fraction of samples that get degraded
    min_ops: int = 1
    max_ops: int = 4
    families: tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "crop", "tone")


class AIGCDataset(Dataset):
    """Yields normalised tensors, optionally with a paired clean view."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 256,
        augment: AugmentConfig | None = None,
        return_pair: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        seed: int = 0,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment if augment is not None else AugmentConfig()
        self.return_pair = return_pair
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seed so every epoch draws different degradations."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng((self.seed, self._epoch, index))

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
        array = np.asarray(resized, dtype=np.uint8)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self.mean) / self.std

    def _degrade(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        cfg = self.augment
        if not cfg.enabled or rng.random() > cfg.probability:
            return image
        n_ops = int(rng.integers(cfg.min_ops, cfg.max_ops + 1))
        return compound_chain(image, n_ops, rng, cfg.families)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rng = self._rng(index)
        original = sample.load()

        degraded = self._degrade(original, rng)
        item = {
            "image": self._to_tensor(degraded),
            "label": torch.tensor(float(sample.label)),
            "index": index,
        }
        if self.return_pair:
            # Same source image, no degradation: the consistency target.
            item["clean"] = self._to_tensor(original)
        return item


def collate(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch]),
    }
    if "clean" in batch[0]:
        out["clean"] = torch.stack([b["clean"] for b in batch])
    return out
