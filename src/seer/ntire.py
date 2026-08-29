"""Loading the NTIRE 2026 Robust AI-Generated Image Detection corpus.

Chosen as the primary source because it is the only public dataset that is
simultaneously (a) drawn from 42 generators spanning 2022-2026, (b) label-
matched on resolution, aspect ratio and JPEG quality between real and fake,
and (c) shipped with per-image ground truth for the degradations applied.

That third property is unusual and valuable: the validation labels record
which distortions hit each image and at what severity, so per-degradation
error analysis needs no extra instrumentation.

Download with ``python get_datasets.py --tier 1`` (or ``--only ntire-train``).
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image

from .paths import ntire_root


def _train_dir() -> Path:
    return ntire_root() / "NTIRE-RobustAIGenDetection-train"


def _val_dir() -> Path:
    return ntire_root() / "NTIRE-RobustAIGenDetection-val"


def _test_dir() -> Path:
    return ntire_root() / "NTIRE-RobustAIGenDetection-test-public"


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
    return read_labelled_split(_val_dir() / f"val_images{suffix}", _val_dir() / f"val{suffix}_labels.csv")


def load_test() -> list[Sample]:
    return read_labelled_split(_test_dir() / "test_images", _test_dir() / "test_labels.csv")


def load_train_shard(shard: int = 0) -> list[Sample]:
    """Shards are distribution-matched, so one shard is a valid training set."""
    base = _train_dir() / f"shard_{shard}"
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


def list_train_shards() -> list[int]:
    """Shard indices present under the NTIRE train root (e.g. 0..5)."""
    root = _train_dir()
    if not root.exists():
        return []
    out: list[int] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("shard_"):
            continue
        suffix = path.name[len("shard_") :]
        if suffix.isdigit():
            out.append(int(suffix))
    return out


def load_train(shard: int = -1) -> list[Sample]:
    """One shard, or every downloaded train shard when ``shard < 0``."""
    if shard >= 0:
        return load_train_shard(shard)
    shards = list_train_shards()
    if not shards:
        raise FileNotFoundError(f"no train shards under {_train_dir()}")
    samples: list[Sample] = []
    for index in shards:
        samples.extend(load_train_shard(index))
    return samples


def load_split(split: str = "train", shard: int = 0, hard: bool = False) -> list[Sample]:
    """Dispatch to the matching NTIRE split loader.

    ``shard < 0`` (or ``split='train_all'``) concatenates every train shard.
    """
    key = split.replace("-", "_")
    if key in ("val", "validation"):
        return load_val(hard=hard)
    if key in ("val_hard", "hard"):
        return load_val(hard=True)
    if key == "test":
        return load_test()
    if key in ("train_all", "all"):
        return load_train(-1)
    return load_train(shard)


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
