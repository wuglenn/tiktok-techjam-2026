"""Loading the NTIRE 2026 Robust AI-Generated Image Detection corpus.

Chosen as the primary source because it is the only public dataset that is
simultaneously (a) drawn from 42 generators spanning 2022-2026, (b) label-
matched on resolution, aspect ratio and JPEG quality between real and fake,
and (c) shipped with per-image ground truth for the degradations applied.

That third property is unusual and valuable: the validation labels record
which distortions hit each image and at what severity, so per-degradation
error analysis needs no extra instrumentation.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from .paths import NTIRE_ROOT

TRAIN_DIR = NTIRE_ROOT / "NTIRE-RobustAIGenDetection-train"
VAL_DIR = NTIRE_ROOT / "NTIRE-RobustAIGenDetection-val"
TEST_DIR = NTIRE_ROOT / "NTIRE-RobustAIGenDetection-test-public"


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int                       # 1 = AI-generated, 0 = real
    distortions: tuple[str, ...] = ()
    distortion_scales: tuple[float, ...] = ()
    is_distorted: bool = False

    def load(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


def _parse_list(raw: str) -> tuple:
    if not raw or raw in ("[]", '""'):
        return ()
    try:
        return tuple(ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return ()


def read_labelled_split(images_dir: Path, labels_csv: Path) -> list[Sample]:
    """Read an NTIRE split whose CSV carries labels and degradation metadata."""
    if not labels_csv.exists():
        raise FileNotFoundError(f"missing labels: {labels_csv}")

    # The zip may extract either flat or into a nested directory.
    roots = [images_dir, *(p for p in images_dir.iterdir() if p.is_dir())] if images_dir.exists() else []
    index: dict[str, Path] = {}
    for root in roots:
        for path in root.glob("*.jpg"):
            index.setdefault(path.name, path)

    samples: list[Sample] = []
    with labels_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["image_name"]
            path = index.get(name)
            if path is None:
                continue
            samples.append(
                Sample(
                    path=path,
                    label=int(row["label"]),
                    distortions=_parse_list(row.get("distortions", "")),
                    distortion_scales=_parse_list(row.get("distortion_scales", "")),
                    is_distorted=bool(int(row.get("is_distorted", 0) or 0)),
                )
            )
    if not samples:
        raise RuntimeError(f"no images matched between {images_dir} and {labels_csv}")
    return samples


def load_val(hard: bool = False) -> list[Sample]:
    suffix = "_hard" if hard else ""
    return read_labelled_split(VAL_DIR / f"val_images{suffix}", VAL_DIR / f"val{suffix}_labels.csv")


def load_test() -> list[Sample]:
    return read_labelled_split(TEST_DIR / "test_images", TEST_DIR / "test_labels.csv")


def load_train_shard(shard: int = 0) -> list[Sample]:
    """Shards are distribution-matched, so one shard is a valid training set."""
    base = TRAIN_DIR / f"shard_{shard}"
    inner = base / f"shard_{shard}"
    root = inner if inner.exists() else base
    labels_csv = root / "labels.csv"
    images_dir = root / "images"
    if not labels_csv.exists():
        found = list(base.rglob("labels.csv"))
        if not found:
            raise FileNotFoundError(f"no labels.csv under {base}")
        labels_csv = found[0]
        images_dir = labels_csv.parent / "images"
    return read_labelled_split(images_dir, labels_csv)


def stratified_subset(
    samples: Sequence[Sample],
    n: int,
    rng: np.random.Generator,
    clean_only: bool = False,
) -> list[Sample]:
    """Class-balanced subset of size ``n`` (or as close as the data allows)."""
    pool = [s for s in samples if not s.is_distorted] if clean_only else list(samples)
    reals = [s for s in pool if s.label == 0]
    fakes = [s for s in pool if s.label == 1]
    per_class = min(n // 2, len(reals), len(fakes))
    picked = [
        *(reals[i] for i in rng.choice(len(reals), per_class, replace=False)),
        *(fakes[i] for i in rng.choice(len(fakes), per_class, replace=False)),
    ]
    rng.shuffle(picked)
    return picked


def iter_images(samples: Sequence[Sample]) -> Iterator[Image.Image]:
    for sample in samples:
        yield sample.load()


def labels_of(samples: Sequence[Sample]) -> np.ndarray:
    return np.array([s.label for s in samples], dtype=np.int64)
