"""Save composited training-data samples as viewable PNGs.

Renders exactly what BatchBuilder produces during composite training: each
sample becomes a two-panel figure (image the model sees | per-patch labels),
so the four top-on-base pairings (fake/real over real/fake), the cropped
overlays and the stacked multi-overlay composites can be inspected by eye.

Synthetic images stand in for the dataset: 'fake' sources are smooth
low-frequency blobs (diffusion-like), 'real' sources are busy textured
patterns with hard-edged shapes, so the class of every region inside a
composite is identifiable by eye. The same inputs are reused for every run
to make the runs directly comparable.

Examples:
  uv run scripts/save_samples.py
  uv run scripts/save_samples.py --res 512 --n 12
  uv run scripts/save_samples.py --out runs/samples --batches 2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from seer.augment import IMAGENET_MEAN, IMAGENET_STD
from seer.config import load_config
from seer.data import BatchBuilder

PATCH_SIZE = 16  # ViT-S/16 ... ViT-L/16 patch grid

# (folder, description, composite config overrides)
RUNS = [
    ("plain", "no composites (reference: plain augmented samples)",
     ["composite.prob=0.0"]),
    ("fake_on_real", "fake crops layered over a real base (localized labels)",
     ["composite.prob=1.0", "composite.real_real_fraction=0.0",
      "composite.fake_on_real=1.0", "composite.real_on_fake=0.0",
      "composite.fake_on_fake=0.0"]),
    ("real_on_fake", "real crops layered over a fake base (inverted labels: only the pasted region is real)",
     ["composite.prob=1.0", "composite.real_real_fraction=0.0",
      "composite.fake_on_real=0.0", "composite.real_on_fake=1.0",
      "composite.fake_on_fake=0.0"]),
    ("fake_on_fake", "fake crops layered over a fake base (all patches stay AI)",
     ["composite.prob=1.0", "composite.real_real_fraction=0.0",
      "composite.fake_on_real=0.0", "composite.real_on_fake=0.0",
      "composite.fake_on_fake=1.0"]),
    ("real_on_real", "real crops layered over a real base (label stays real)",
     ["composite.prob=1.0", "composite.real_real_fraction=1.0",
      "composite.fake_on_real=0.0", "composite.real_on_fake=0.0",
      "composite.fake_on_fake=0.0"]),
    ("mix", "the default training distribution (all pairings, weights 0.5/0.25/0.25)",
     ["composite.prob=1.0", "composite.real_real_fraction=1.0"]),
]


def _synth_image(label: int, rng: np.random.RandomState, size: int = 640) -> Image.Image:
    """Synthetic stand-in for a dataset image."""
    if label == 1:  # fake: smooth low-frequency blobs
        base = rng.rand(size // 32, size // 32, 3) * 200 + 20
        return Image.fromarray(base.astype(np.uint8)).resize((size, size), Image.BICUBIC)
    base = rng.rand(size // 8, size // 8, 3) * 160 + 40  # real: busy texture
    img = np.asarray(Image.fromarray(base.astype(np.uint8)).resize((size, size), Image.BICUBIC))
    img = np.clip(img.astype(np.float32) + rng.randn(size, size, 3) * 10, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(6):
        x, y = int(rng.randint(0, size)), int(rng.randint(0, size))
        r = int(rng.randint(20, 120))
        color = tuple(int(v) for v in rng.rand(3) * 255)
        if rng.rand() < 0.5:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
        else:
            draw.rectangle([x - r, y - r, x + r, y + r], fill=color)
    return pil


def _render(path: Path, img: torch.Tensor, patch_labels: torch.Tensor,
            label: float, G: int, res: int):
    """Two-panel figure: denormalized image | patch-label overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    disp = (img * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    grid = patch_labels.reshape(G, G)[None, None].float()
    over = torch.nn.functional.interpolate(
        grid, size=(res, res), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(disp)
    axes[0].set_title(f"training sample (label = {int(label)})")
    axes[0].axis("off")
    axes[1].imshow(disp)
    axes[1].imshow(over, cmap="turbo", alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1].set_title("patch labels (warm = AI)")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="samples", help="output directory")
    p.add_argument("--res", type=int, default=224, help="training resolution")
    p.add_argument("--n", type=int, default=8, help="samples per batch (even = class-balanced)")
    p.add_argument("--batches", type=int, default=1, help="batches per run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert args.res % PATCH_SIZE == 0, f"--res must be a multiple of {PATCH_SIZE}"
    G = args.res // PATCH_SIZE

    # one fixed, class-balanced sample set, reused by every run
    samples = []
    for i in range(args.n):
        rng = np.random.RandomState(args.seed * 1009 + i)
        label = i % 2
        samples.append({
            "image": _synth_image(label, rng),
            "label": label,
            "generator": "synthetic",
            "architecture": "Synthetic",
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    readme = [
        "# Composite training-data samples",
        "",
        "Two panels per sample: the image the model sees (left) and its",
        "per-patch labels (right, warm = AI). Overlays are crops of the source",
        "image (own scale / aspect / flip), stacks of up to `max_overlays`.",
        "Generated by `scripts/save_samples.py`.",
        "",
    ]
    for name, desc, overrides in RUNS:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        cfg = load_config(overrides=[f"res={args.res}"] + overrides)
        builder = BatchBuilder(cfg, train=True, patch_grid=G, seed=args.seed)
        for bi in range(args.batches):
            batch = builder(samples)
            for si in range(batch["images"].shape[0]):
                _render(
                    d / f"{name}_{bi:02d}_{si:02d}_label{int(batch['labels'][si])}.png",
                    batch["images"][si], batch["patch_labels"][si],
                    batch["labels"][si], G, args.res,
                )
        readme.append(f"- `{name}/` - {desc}")
        print(f"{name}: saved {args.batches * args.n} samples")

    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
