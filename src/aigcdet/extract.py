"""Threaded feature extraction with on-disk caching.

Measured on an RTX 3080: naive sequential loading runs at ~21 img/s while the
GPU sustains ~84 img/s on in-memory images, so the pipeline was JPEG-decode
bound, not compute bound. Decoding and degrading on a thread pool (PIL releases
the GIL during decode and resize) closes that gap.

Because the backbone is frozen, embeddings are written to disk keyed by
(model, resolution, pooling, degradation, sample set). Re-running an experiment
with a different head, loss or calibration is then a cache hit and costs
seconds rather than a GPU pass.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .backbone import FrozenBackbone
from .data import Sample
from .degradations import Degradation
from .paths import FEATURE_CACHE as CACHE_ROOT


def _fingerprint(samples: Sequence[Sample]) -> str:
    """Stable id for a sample set, from filenames and order."""
    digest = hashlib.sha1()
    digest.update(str(len(samples)).encode())
    for sample in samples:
        digest.update(sample.path.name.encode())
    return digest.hexdigest()[:16]


def cache_key(
    backbone: FrozenBackbone,
    samples: Sequence[Sample],
    degradation: Degradation | None,
) -> str:
    config = backbone.config
    parts = [
        config.model_name.replace("/", "_"),
        f"r{config.image_size}",
        config.pooling,
        degradation.name if degradation else "clean",
        _fingerprint(samples),
    ]
    return "__".join(parts)


def _prepare(sample: Sample, degradation: Degradation | None, seed: int) -> Image.Image:
    image = sample.load()
    if degradation is not None:
        image = degradation(image, np.random.default_rng(seed))
    return image


def extract(
    backbone: FrozenBackbone,
    samples: Sequence[Sample],
    degradation: Degradation | None = None,
    workers: int = 8,
    chunk: int = 256,
    seed: int = 0,
    use_cache: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """Embed ``samples`` (optionally degraded first), with disk caching."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"{cache_key(backbone, samples, degradation)}.npy"
    if use_cache and path.exists():
        return np.load(path)

    vectors: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(samples), chunk):
            batch = samples[start : start + chunk]
            images = list(
                pool.map(
                    lambda item: _prepare(item[1], degradation, seed + start + item[0]),
                    enumerate(batch),
                )
            )
            vectors.append(backbone.embed(images))
            if progress:
                done = min(start + chunk, len(samples))
                print(f"    {done}/{len(samples)}", end="\r", flush=True)
    if progress:
        print(" " * 32, end="\r")

    features = np.concatenate(vectors, axis=0)
    if use_cache:
        np.save(path, features)
    return features


def extract_many(
    backbone: FrozenBackbone,
    samples: Sequence[Sample],
    degradations: Iterable[Degradation | None],
    **kwargs,
) -> dict[str, np.ndarray]:
    """Embed the same samples under several degradations, keyed by name."""
    out: dict[str, np.ndarray] = {}
    for degradation in degradations:
        name = degradation.name if degradation else "clean"
        print(f"  [{name}]", flush=True)
        out[name] = extract(backbone, samples, degradation, **kwargs)
    return out


def extract_views(
    backbone: FrozenBackbone,
    samples: Sequence[Sample],
    degradations: Sequence[Degradation | None],
    workers: int = 16,
    chunk: int = 128,
    seed: int = 0,
    use_cache: bool = True,
    progress: bool = True,
) -> dict[str, np.ndarray]:
    """Embed every sample under every degradation, decoding each file once.

    ``extract`` re-reads the source image for each degradation, so decode cost
    scales with (images x degradations). Here each file is decoded once and all
    degraded views are derived from the in-memory copy, which moves the
    bottleneck back onto the GPU where it belongs.

    Degradations are still applied at native resolution before the resize to
    the backbone's input size -- doing it the other way round would destroy the
    very artifacts being measured.
    """
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    names = [d.name if d else "clean" for d in degradations]
    paths = {n: CACHE_ROOT / f"{cache_key(backbone, samples, d)}.npy" for n, d in zip(names, degradations)}

    if use_cache and all(p.exists() for p in paths.values()):
        return {name: np.load(path) for name, path in paths.items()}

    buffers: dict[str, list[np.ndarray]] = {name: [] for name in names}

    def decode(sample: Sample) -> Image.Image:
        return sample.load()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(samples), chunk):
            batch = samples[start : start + chunk]
            originals = list(pool.map(decode, batch))

            for name, degradation in zip(names, degradations):
                if degradation is None:
                    views = originals
                else:
                    views = list(
                        pool.map(
                            lambda item: degradation(item[1], np.random.default_rng(seed + start + item[0])),
                            enumerate(originals),
                        )
                    )
                buffers[name].append(backbone.embed(views))

            if progress:
                done = min(start + chunk, len(samples))
                print(f"    {done}/{len(samples)} x {len(names)} views", end="\r", flush=True)
    if progress:
        print(" " * 48, end="\r")

    out = {name: np.concatenate(chunks, axis=0) for name, chunks in buffers.items()}
    if use_cache:
        for name, features in out.items():
            np.save(paths[name], features)
    return out
