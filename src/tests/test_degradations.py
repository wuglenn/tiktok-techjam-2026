"""Sanity checks for the degradation pipeline.

These guard the properties the robustness evaluation depends on: every
degradation must run, preserve RGB, and actually change the image (a silent
no-op would inflate our robustness numbers).
"""

import numpy as np
import pytest
from PIL import Image

from aigcdet.degradations import (
    EXTENDED_GRID,
    IDENTITY,
    MANDATED_GRID,
    centre_crop,
    compound_chain,
    compound_grid,
    grid_by_family,
    jpeg,
    resize_roundtrip,
)


@pytest.fixture
def sample() -> Image.Image:
    """A photo-like image: smooth gradients, hard edges, and fine texture.

    All three matter. A purely smooth image is unaffected by mild blur, and a
    purely noisy one is unaffected by anything else -- either would make the
    degradations look like no-ops.
    """
    height, width = 256, 320
    rng = np.random.default_rng(0)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gradient = (xx / width) * 160.0 + (yy / height) * 60.0

    image = np.stack([gradient, gradient * 0.8 + 30.0, gradient * 0.6 + 60.0], axis=-1)
    image[64:192, 80:240] += 55.0                      # hard-edged block
    image += 18.0 * np.sin(xx / 3.0)[..., None]        # fine high-frequency texture
    image += rng.normal(0.0, 6.0, image.shape)         # sensor-like grain

    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    if a.size != b.size:
        b = b.resize(a.size)
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())


@pytest.mark.parametrize("degradation", MANDATED_GRID + EXTENDED_GRID, ids=lambda d: d.name)
def test_runs_and_preserves_rgb(sample, degradation):
    out = degradation(sample, np.random.default_rng(0))
    assert out.mode == "RGB"
    assert out.size[0] > 0 and out.size[1] > 0


@pytest.mark.parametrize(
    "degradation",
    [d for d in MANDATED_GRID if d.name != "clean"],
    ids=lambda d: d.name,
)
def test_actually_degrades(sample, degradation):
    """A degradation that silently no-ops would inflate robustness scores."""
    out = degradation(sample, np.random.default_rng(0))
    assert _mean_abs_diff(sample, out) > 0.01, f"{degradation.name} did not change the image"


def test_identity_is_a_no_op(sample):
    assert _mean_abs_diff(sample, IDENTITY(sample)) == 0.0


def test_mandated_grid_covers_all_six_families():
    families = set(grid_by_family(MANDATED_GRID)) - {"clean"}
    assert families == {"jpeg", "blur", "resize", "noise", "jitter", "crop"}


def test_jpeg_quality_is_monotonic(sample):
    """Lower quality must distort more, or the codec is not being exercised."""
    diffs = [_mean_abs_diff(sample, jpeg(sample, q)) for q in (90, 70, 50, 30)]
    assert diffs == sorted(diffs), f"JPEG distortion not monotonic in quality: {diffs}"


def test_resize_roundtrip_restores_size(sample):
    assert resize_roundtrip(sample, 0.25).size == sample.size


def test_centre_crop_keeps_fraction(sample):
    out = centre_crop(sample, 0.8)
    assert out.size == (round(sample.size[0] * 0.8), round(sample.size[1] * 0.8))


def test_compound_chains_are_deterministic(sample):
    for degradation in compound_grid(max_ops=3):
        first = degradation(sample)
        second = degradation(sample)
        assert _mean_abs_diff(first, second) == 0.0, f"{degradation.name} is not reproducible"


def test_compound_chains_escalate(sample):
    """More chained operations means more distortion, averaged over seeds.

    Crop is excluded here only because it changes the output size, which
    dominates a pixel-difference metric and swamps the effect being measured.
    A single seed is far too noisy: one heavy op can out-distort five mild ones.
    """
    families = ["jpeg", "blur", "resize", "noise", "jitter", "tone"]

    def mean_distortion(n_ops: int) -> float:
        return float(
            np.mean([
                _mean_abs_diff(sample, compound_chain(sample, n_ops, np.random.default_rng(s), families))
                for s in range(12)
            ])
        )

    assert mean_distortion(4) > mean_distortion(1)
