"""Augmentation pipeline.

Two families:

1. train_transform - "wild simulation". Distortions applied *symmetrically* to
   real and fake images (JPEG/WebP re-encode, resize, crop, blur, noise, ...),
   so the detector cannot shortcut on the augmentation itself and must learn
   generator fingerprints. Mirrors how images degrade when shared online.

2. pangram_augment / eval_tensor - the augmented evaluation protocol from the
   Pangram Image technical blog: downscale to 1024x1024 and re-encode as
   JPEG quality 50 ("a higher degree of compression than most real-world
   images"), i.e. a worst-case robustness check.
"""

import io
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------- PIL helpers


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def jpeg_recompress(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(round(quality)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def webp_recompress(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=int(round(quality)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_resized_crop(img: Image.Image, res: int, rng: random.Random,
                        scale=(0.35, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)) -> Image.Image:
    """Torchvision-style RandomResizedCrop on PIL images."""
    W, H = img.size
    area = W * H
    for _ in range(10):
        target = area * rng.uniform(*scale)
        ar = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w = int(round(math.sqrt(target * ar)))
        h = int(round(math.sqrt(target / ar)))
        if 0 < w <= W and 0 < h <= H:
            x0 = rng.randint(0, W - w)
            y0 = rng.randint(0, H - h)
            box = (x0, y0, x0 + w, y0 + h)
            return img.resize((res, res), Image.BICUBIC, box=box)
    # fallback: center crop
    w, h = min(W, H), min(W, H)
    x0, y0 = (W - w) // 2, (H - h) // 2
    return img.resize((res, res), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))


def center_crop(img: Image.Image, scale: float = 0.8) -> Image.Image:
    """Center crop keeping `scale` of each dimension (benchmark 'crop 80%')."""
    W, H = img.size
    w, h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    x0, y0 = (W - w) // 2, (H - h) // 2
    return img.crop((x0, y0, x0 + w, y0 + h))


def center_crop_resize(img: Image.Image, res: int, scale: float = 0.8) -> Image.Image:
    W, H = img.size
    w, h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    x0, y0 = (W - w) // 2, (H - h) // 2
    return img.resize((res, res), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))


# ------------------------------------------------------------ train transform


def train_transform(img: Image.Image, res: int, rng: random.Random, cfg) -> torch.Tensor:
    """PIL -> augmented, normalized (C, res, res) float tensor.

    Parameter levels are drawn from the benchmark robustness protocols
    (JPEG q in {90,70,50,30}, blur sigma in {0.5,1,2}, noise sigma in
    {0.02,0.05,0.10}, resize 0.5x/0.25x, jitter +/-20%, crop 80%) plus
    WebP and grayscale for extra wild-simulation coverage.
    """
    img = img.convert("RGB")

    if rng.random() < cfg.hflip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if rng.random() < cfg.color_jitter_prob:
        # mild brightness / contrast / saturation jitter in image space
        from PIL import ImageEnhance

        j = cfg.color_jitter
        img = ImageEnhance.Brightness(img).enhance(1.0 + rng.uniform(-j, j))
        img = ImageEnhance.Contrast(img).enhance(1.0 + rng.uniform(-j, j))
        img = ImageEnhance.Color(img).enhance(1.0 + rng.uniform(-j, j))

    # crop strategy: deterministic center crop (benchmark style) or RRC
    if rng.random() < cfg.center_crop_prob:
        img = center_crop_resize(img, res, cfg.center_crop_scale)
    else:
        img = random_resized_crop(img, res, rng, scale=tuple(cfg.scale_range))

    if rng.random() < cfg.grayscale_prob:
        img = img.convert("L").convert("RGB")

    r = rng.random()
    if r < cfg.jpeg_prob:
        img = jpeg_recompress(img, rng.choice(cfg.jpeg_quality))
    elif r < cfg.jpeg_prob + cfg.webp_prob:
        img = webp_recompress(img, rng.choice(cfg.webp_quality))

    if rng.random() < cfg.blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(rng.choice(cfg.blur_sigma)))

    t = _to_tensor(img)

    # resolution loss: downscale then upscale (0.5x / 0.25x levels)
    if rng.random() < cfg.downscale_prob:
        s = rng.choice(cfg.downscale_levels)
        small = max(res // 16, int(round(res * s)))
        t = F.interpolate(t[None], size=(small, small), mode="bilinear", align_corners=False, antialias=True)
        t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False, antialias=True)[0]

    if rng.random() < cfg.noise_prob:
        t = t + torch.randn_like(t) * rng.choice(cfg.noise_levels)

    return t


# ------------------------------------------------------------- eval transforms


def eval_transform(img: Image.Image, res: int) -> torch.Tensor:
    """Deterministic: full image resized to (res, res) + normalize."""
    img = img.convert("RGB").resize((res, res), Image.BICUBIC)
    return _to_tensor(img)


def pangram_augment(img: Image.Image, size: int = 1024, quality: int = 50) -> Image.Image:
    """The 'augmented' protocol from Pangram's benchmark: downscale to
    size x size, JPEG re-encode at quality 50. Applied to both classes."""
    img = img.convert("RGB")
    W, H = img.size
    if max(W, H) > size:
        scale = size / max(W, H)
        img = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.BICUBIC)
    return jpeg_recompress(img, quality)


# ------------------------------------------------------------ perturbations


def _resize_perturb(img: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` then upscale back to the original resolution."""
    W, H = img.size
    w, h = max(1, round(W * scale)), max(1, round(H * scale))
    return img.resize((w, h), Image.BICUBIC).resize((W, H), Image.BICUBIC)


def _noise_perturb(img: Image.Image, sigma: float) -> Image.Image:
    t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    t = (t + torch.randn_like(t) * sigma).clamp(0.0, 1.0)
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def _jitter_perturb(img: Image.Image, amount: float = 0.2) -> Image.Image:
    from PIL import ImageEnhance

    img = ImageEnhance.Brightness(img).enhance(1.0 + amount)
    img = ImageEnhance.Contrast(img).enhance(1.0 + amount)
    img = ImageEnhance.Color(img).enhance(1.0 + amount)
    return img


PERTURBATIONS = {
    # name -> (fn, description); applied deterministically to BOTH classes
    "clean": (lambda im: im.convert("RGB"), "no perturbation"),
    "jpeg90": (lambda im: jpeg_recompress(im, 90), "JPEG quality 90"),
    "jpeg70": (lambda im: jpeg_recompress(im, 70), "JPEG quality 70"),
    "jpeg50": (lambda im: jpeg_recompress(im, 50), "JPEG quality 50"),
    "jpeg30": (lambda im: jpeg_recompress(im, 30), "JPEG quality 30"),
    "blur0.5": (lambda im: im.filter(ImageFilter.GaussianBlur(0.5)), "Gaussian blur sigma 0.5"),
    "blur1.0": (lambda im: im.filter(ImageFilter.GaussianBlur(1.0)), "Gaussian blur sigma 1.0"),
    "blur2.0": (lambda im: im.filter(ImageFilter.GaussianBlur(2.0)), "Gaussian blur sigma 2.0"),
    "resize0.5": (lambda im: _resize_perturb(im, 0.5), "resize 0.5x then upscale"),
    "resize0.25": (lambda im: _resize_perturb(im, 0.25), "resize 0.25x then upscale"),
    "noise0.02": (lambda im: _noise_perturb(im, 0.02), "Gaussian noise sigma 0.02"),
    "noise0.05": (lambda im: _noise_perturb(im, 0.05), "Gaussian noise sigma 0.05"),
    "noise0.10": (lambda im: _noise_perturb(im, 0.10), "Gaussian noise sigma 0.10"),
    "jitter20": (lambda im: _jitter_perturb(im, 0.2), "brightness/contrast/saturation +20%"),
    "crop80": (lambda im: center_crop(im, 0.8), "center crop 80%"),
    "pangram": (lambda im: pangram_augment(im), "Pangram augmented protocol (1024px + JPEG q50)"),
}


def apply_perturbation(img: Image.Image, name: str) -> Image.Image:
    """Apply a named benchmark perturbation (deterministic, both classes).
    Used by `main.py eval --perturbation <name>` to produce robustness
    tables like the GenImage / OmniAID / Pangram evaluations."""
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation '{name}'. Available: {list(PERTURBATIONS)}")
    return PERTURBATIONS[name][0](img.convert("RGB"))
