"""Image degradations for robustness evaluation and training augmentation.

Two grids are defined here:

``MANDATED_GRID``
    The exact transformation list from the Track 5 problem statement, at the
    exact severities specified. This is what the robustness summary reports.

``EXTENDED_GRID``
    Degradations the NTIRE 2026 organisers found actually break detectors --
    stacked heterogeneous compression, tone curves, local contrast, geometric
    warps. The mandated six correspond closely to the tier they discarded as
    non-damaging, so the extended grid is where detectors are separated.

Every degradation is a pure ``PIL.Image -> PIL.Image`` function operating on
RGB, so the same objects drive evaluation and training. Randomised
degradations take an explicit ``rng`` to keep the evaluation grid reproducible.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RGB = "RGB"


# --------------------------------------------------------------------------
# Primitive operations
# --------------------------------------------------------------------------

def jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode through a real JPEG codec so DCT quantisation actually happens."""
    buffer = io.BytesIO()
    image.convert(RGB).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert(RGB)


def webp(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert(RGB).save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert(RGB)


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(
    image: Image.Image,
    scale: float,
    down_resample: int = Image.BICUBIC,
    up_resample: int = Image.BICUBIC,
) -> Image.Image:
    """Downscale by ``scale`` then restore the original size.

    Simulates thumbnail generation. Downsampling permanently truncates the
    spectrum, which is why this is one of the more destructive operations.
    """
    width, height = image.size
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(small, down_resample).resize((width, height), up_resample)


def gaussian_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Additive Gaussian noise; ``sigma`` is on the [0, 1] intensity scale."""
    array = np.asarray(image.convert(RGB), dtype=np.float32) / 255.0
    noisy = array + rng.normal(0.0, sigma, array.shape).astype(np.float32)
    return Image.fromarray((np.clip(noisy, 0.0, 1.0) * 255.0).round().astype(np.uint8), RGB)


def colour_jitter(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> Image.Image:
    """Multiplicative brightness/contrast/saturation. 1.0 is a no-op."""
    out = image.convert(RGB)
    for enhancer, factor in (
        (ImageEnhance.Brightness, brightness),
        (ImageEnhance.Contrast, contrast),
        (ImageEnhance.Color, saturation),
    ):
        if factor != 1.0:
            out = enhancer(out).enhance(factor)
    return out


def centre_crop(image: Image.Image, keep: float) -> Image.Image:
    """Keep the central ``keep`` fraction of width and height."""
    width, height = image.size
    new_w, new_h = max(1, round(width * keep)), max(1, round(height * keep))
    left, top = (width - new_w) // 2, (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


def tone_curve(image: Image.Image, amount: float) -> Image.Image:
    """S-curve on luminance. Approximates auto-enhance / filter behaviour."""
    array = np.asarray(image.convert(RGB), dtype=np.float32) / 255.0
    curved = array + amount * np.sin(2.0 * np.pi * array) / (2.0 * np.pi)
    return Image.fromarray((np.clip(curved, 0.0, 1.0) * 255.0).round().astype(np.uint8), RGB)


# --------------------------------------------------------------------------
# Grid definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Degradation:
    """A named, reproducible image degradation."""

    name: str
    family: str
    apply: Callable[[Image.Image, np.random.Generator], Image.Image]
    params: dict = field(default_factory=dict)

    def __call__(self, image: Image.Image, rng: np.random.Generator | None = None) -> Image.Image:
        return self.apply(image, rng if rng is not None else np.random.default_rng(0))


def _d(name: str, family: str, fn: Callable[..., Image.Image], **params) -> Degradation:
    """Wrap a deterministic primitive as a Degradation."""
    return Degradation(name, family, lambda img, _rng: fn(img, **params), params)


IDENTITY = Degradation("clean", "clean", lambda img, _rng: img)

# The six mandated families at the exact severities in the problem statement.
MANDATED_GRID: tuple[Degradation, ...] = (
    IDENTITY,
    # 1. JPEG compression -- social re-encoding, messaging apps.
    *(_d(f"jpeg_q{q}", "jpeg", jpeg, quality=q) for q in (90, 70, 50, 30)),
    # 2. Gaussian blur -- out-of-focus capture.
    *(_d(f"blur_s{s}", "blur", gaussian_blur, sigma=s) for s in (0.5, 1.0, 2.0)),
    # 3. Resize down-then-up -- thumbnail generation.
    *(_d(f"resize_{s}x", "resize", resize_roundtrip, scale=s) for s in (0.5, 0.25)),
    # 4. Gaussian noise -- low-light sensor noise.
    *(
        Degradation(f"noise_s{s}", "noise", (lambda sig: lambda img, rng: gaussian_noise(img, sig, rng))(s), {"sigma": s})
        for s in (0.02, 0.05, 0.10)
    ),
    # 5. Colour jitter +/-20% -- photo filters, auto-enhancement.
    _d("jitter_bright_0.8", "jitter", colour_jitter, brightness=0.8),
    _d("jitter_bright_1.2", "jitter", colour_jitter, brightness=1.2),
    _d("jitter_contrast_0.8", "jitter", colour_jitter, contrast=0.8),
    _d("jitter_contrast_1.2", "jitter", colour_jitter, contrast=1.2),
    _d("jitter_sat_0.8", "jitter", colour_jitter, saturation=0.8),
    _d("jitter_sat_1.2", "jitter", colour_jitter, saturation=1.2),
    _d("jitter_all_0.8", "jitter", colour_jitter, brightness=0.8, contrast=0.8, saturation=0.8),
    _d("jitter_all_1.2", "jitter", colour_jitter, brightness=1.2, contrast=1.2, saturation=1.2),
    # 6. Centre crop 80% -- profile-picture framing.
    _d("crop_80", "crop", centre_crop, keep=0.80),
)

# Degradations the NTIRE 2026 organisers escalated because they broke detectors.
EXTENDED_GRID: tuple[Degradation, ...] = (
    # Stacked heterogeneous compression: the single most damaging family they found.
    _d("jpeg_q70_then_q40", "stacked", lambda img: jpeg(jpeg(img, 70), 40)),
    _d("jpeg_q50_then_webp_q50", "stacked", lambda img: webp(jpeg(img, 50), 50)),
    _d("webp_q60_then_jpeg_q40", "stacked", lambda img: jpeg(webp(img, 60), 40)),
    # Resample through a thumbnail, then recompress -- the true social-media path.
    _d("resize_0.5x_then_q40", "stacked", lambda img: jpeg(resize_roundtrip(img, 0.5), 40)),
    # Tone curve and aggressive jitter beyond the mandated +/-20%.
    _d("tone_curve_0.3", "tone", tone_curve, amount=0.3),
    _d("jitter_all_0.6", "jitter_hard", colour_jitter, brightness=0.6, contrast=1.4, saturation=1.4),
    # Harder crops and downscales.
    _d("crop_50", "crop_hard", centre_crop, keep=0.50),
    _d("resize_0.125x", "resize_hard", resize_roundtrip, scale=0.125),
)

FULL_GRID: tuple[Degradation, ...] = MANDATED_GRID + EXTENDED_GRID


# --------------------------------------------------------------------------
# Compound chains
# --------------------------------------------------------------------------

_CHAIN_FAMILIES: dict[str, Sequence[Callable[[Image.Image, np.random.Generator], Image.Image]]] = {
    "jpeg": [lambda img, rng: jpeg(img, int(rng.integers(25, 96)))],
    "blur": [lambda img, rng: gaussian_blur(img, float(rng.uniform(0.3, 2.5)))],
    "resize": [lambda img, rng: resize_roundtrip(img, float(rng.uniform(0.2, 0.9)))],
    "noise": [lambda img, rng: gaussian_noise(img, float(rng.uniform(0.0, 0.12)), rng)],
    "jitter": [
        lambda img, rng: colour_jitter(
            img,
            brightness=float(rng.uniform(0.7, 1.3)),
            contrast=float(rng.uniform(0.7, 1.3)),
            saturation=float(rng.uniform(0.7, 1.3)),
        )
    ],
    "crop": [lambda img, rng: centre_crop(img, float(rng.uniform(0.6, 0.95)))],
    "tone": [lambda img, rng: tone_curve(img, float(rng.uniform(-0.4, 0.4)))],
}


def compound_chain(
    image: Image.Image,
    n_ops: int,
    rng: np.random.Generator,
    families: Iterable[str] | None = None,
) -> Image.Image:
    """Apply ``n_ops`` degradations drawn from *distinct* families, in random order.

    Mirrors the NTIRE 2026 evaluation pipeline, where chained multi-family
    degradation -- not any single operation -- is what collapses detectors.
    """
    pool = list(families) if families is not None else list(_CHAIN_FAMILIES)
    chosen = rng.choice(pool, size=min(n_ops, len(pool)), replace=False)
    out = image
    for family in chosen:
        op = _CHAIN_FAMILIES[str(family)][0]
        out = op(out, rng)
    return out


def compound_grid(max_ops: int = 5, seed: int = 0) -> tuple[Degradation, ...]:
    """One Degradation per chain length, each with a fixed seed for reproducibility."""

    def make(n: int) -> Degradation:
        def apply(img: Image.Image, _rng: np.random.Generator) -> Image.Image:
            return compound_chain(img, n, np.random.default_rng(seed + n))

        return Degradation(f"compound_{n}op", "compound", apply, {"n_ops": n})

    return tuple(make(n) for n in range(1, max_ops + 1))


def training_views(
    n_views: int,
    max_ops: int = 4,
    seed: int = 1234,
) -> tuple[Degradation, ...]:
    """Random compound degradations for *training* augmentation.

    Deliberately not the evaluation grid. Training on the exact severities we
    report would be teaching to the test; the published evidence is also that
    training strictly harder than the evaluation distribution is what wins
    (the NTIRE top teams chained up to 6 operations against a 5-operation
    test pipeline).

    Each view varies per image, because the chain is seeded by view index and
    image index together.
    """

    def make(view: int) -> Degradation:
        def apply(img: Image.Image, rng: np.random.Generator) -> Image.Image:
            # rng is seeded per-image by the extractor; mix in the view index
            # so the same image gets different chains across views.
            local = np.random.default_rng(int(rng.integers(0, 2**31)) + seed * (view + 1))
            n_ops = int(local.integers(1, max_ops + 1))
            return compound_chain(img, n_ops, local)

        return Degradation(f"train_view{view}", "train_aug", apply, {"max_ops": max_ops})

    return tuple(make(v) for v in range(n_views))


def grid_by_family(grid: Iterable[Degradation]) -> dict[str, list[Degradation]]:
    grouped: dict[str, list[Degradation]] = {}
    for degradation in grid:
        grouped.setdefault(degradation.family, []).append(degradation)
    return grouped
