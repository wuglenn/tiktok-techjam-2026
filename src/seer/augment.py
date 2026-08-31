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
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------- PIL helpers


def _to_tensor(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _from_tensor(t: torch.Tensor) -> Image.Image:
    """Inverse of `_to_tensor` for a second encode pass (post-composite)."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype).view(3, 1, 1)
    x = (t.detach().float().cpu() * std + mean).clamp(0.0, 1.0)
    return Image.fromarray((x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))


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

    Parameter levels cover the official eval table and go harder: JPEG
    down to q=5, WebP, blur σ=4, noise 0.20, 0.125× resize, plus extras
    that wipe generator fingerprints (DCT grid-shift JPEG, resample
    mismatch, phase noise, chroma noise, recapture warp, surface blur).
    """
    img = img.convert("RGB")

    if rng.random() < cfg.hflip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if rng.random() < cfg.color_jitter_prob:
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

    extra_p = float(getattr(cfg, "extra_distort_prob", 0.0) or 0.0)
    extra_n = max(1, int(getattr(cfg, "extra_distort_max", 1) or 1))
    if extra_p > 0 and rng.random() < extra_p:
        for _ in range(rng.randint(1, extra_n)):
            img = _extra_train_distort(img, rng)

    # Occasional hue/WB after extras so color fingerprints don't survive
    # the rest of the stack unchanged.
    if rng.random() < 0.08:
        img = _hue_shift(img, rng.uniform(-18.0, 18.0))
    if rng.random() < 0.08:
        img = _white_balance(img, rng.uniform(0.82, 1.18), rng.uniform(0.82, 1.18))

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


# Extra kinds that keep pixels registered to the patch grid. Crop / flip /
# rotate / DCT-grid shift would move the composite silhouette off its labels.
_ALIGNED_EXTRA = (
    "jpeg", "doublejpeg", "webp", "blur", "impulse", "quantize", "motion",
    "shift", "pixelate", "bright", "chroma", "median", "unsharp",
    "gamma", "grain", "aberr", "fftlp", "autocontrast", "posterize",
    "social", "resample", "surface", "phase", "hue", "wb", "chroman",
    "equalize", "vignette", "speckle", "recode",
)


def post_stack_transform(t: torch.Tensor, res: int, rng: random.Random, cfg) -> torch.Tensor:
    """One shared wild-sim pass on an already-normalized (C, res, res) tensor.

    Same JPEG / WebP / downscale / blur / noise family as ``train_transform``,
    but no crop or flip: the draw applies to the whole image so independently
    augmented layers no longer keep mismatched codec/noise fingerprints, and
    composite patch labels stay aligned.
    """
    img = _from_tensor(t)
    if img.size != (res, res):
        img = img.resize((res, res), Image.BICUBIC)

    if rng.random() < cfg.color_jitter_prob:
        j = cfg.color_jitter
        img = ImageEnhance.Brightness(img).enhance(1.0 + rng.uniform(-j, j))
        img = ImageEnhance.Contrast(img).enhance(1.0 + rng.uniform(-j, j))
        img = ImageEnhance.Color(img).enhance(1.0 + rng.uniform(-j, j))

    if rng.random() < cfg.grayscale_prob:
        img = img.convert("L").convert("RGB")

    r = rng.random()
    if r < cfg.jpeg_prob:
        img = jpeg_recompress(img, rng.choice(cfg.jpeg_quality))
    elif r < cfg.jpeg_prob + cfg.webp_prob:
        img = webp_recompress(img, rng.choice(cfg.webp_quality))

    if rng.random() < cfg.blur_prob:
        img = img.filter(ImageFilter.GaussianBlur(rng.choice(cfg.blur_sigma)))

    extra_p = float(getattr(cfg, "extra_distort_prob", 0.0) or 0.0)
    extra_n = max(1, int(getattr(cfg, "extra_distort_max", 1) or 1))
    if extra_p > 0 and rng.random() < extra_p:
        for _ in range(rng.randint(1, extra_n)):
            img = _extra_train_distort(img, rng, kinds=_ALIGNED_EXTRA)

    if rng.random() < 0.08:
        img = _hue_shift(img, rng.uniform(-18.0, 18.0))
    if rng.random() < 0.08:
        img = _white_balance(img, rng.uniform(0.82, 1.18), rng.uniform(0.82, 1.18))

    out = _to_tensor(img)
    if out.shape[-2:] != (res, res):
        out = F.interpolate(
            out[None], size=(res, res), mode="bilinear",
            align_corners=False, antialias=True,
        )[0]

    if rng.random() < cfg.downscale_prob:
        s = rng.choice(cfg.downscale_levels)
        small = max(res // 16, int(round(res * s)))
        out = F.interpolate(
            out[None], size=(small, small), mode="bilinear",
            align_corners=False, antialias=True,
        )
        out = F.interpolate(
            out, size=(res, res), mode="bilinear",
            align_corners=False, antialias=True,
        )[0]

    if rng.random() < cfg.noise_prob:
        out = out + torch.randn_like(out) * rng.choice(cfg.noise_levels)
    return out


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


def _impulse_noise(img: Image.Image, amount: float = 0.02, rng: Optional[random.Random] = None) -> Image.Image:
    """Salt-and-pepper (NTIRE train: impulse noise)."""
    rng = rng or random
    arr = np.array(img)
    h, w = arr.shape[:2]
    n = max(1, int(round(h * w * float(amount))))
    ys = [rng.randrange(h) for _ in range(n)]
    xs = [rng.randrange(w) for _ in range(n)]
    for y, x in zip(ys, xs):
        arr[y, x] = 0 if rng.random() < 0.5 else 255
    return Image.fromarray(arr)


def _quantize(img: Image.Image, colors: int = 32) -> Image.Image:
    """Color quantization (NTIRE train)."""
    dither = getattr(Image, "FLOYDSTEINBERG", 1)
    return img.quantize(colors=max(2, int(colors)), dither=dither).convert("RGB")


def _motion_blur(img: Image.Image, length: int = 9) -> Image.Image:
    """Horizontal box motion blur (NTIRE val: motion blur)."""
    arr = np.asarray(img, dtype=np.float32)
    k = max(3, int(length))
    pad = k // 2
    p = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    cs = np.cumsum(p, axis=1, dtype=np.float32)
    left = np.zeros_like(cs)
    left[:, 1:] = cs[:, :-1]
    win = (cs[:, k - 1:] - left[:, : cs.shape[1] - (k - 1)]) / k
    out = win[:, : arr.shape[1]]
    if out.shape[1] < arr.shape[1]:
        out = np.pad(out, ((0, 0), (0, arr.shape[1] - out.shape[1]), (0, 0)), mode="edge")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _color_shift(img: Image.Image, delta: int = 20, channel: int = 0) -> Image.Image:
    """RGB channel shift (NTIRE train: color shift)."""
    arr = np.asarray(img, dtype=np.int16)
    c = int(channel) % 3
    arr[:, :, c] = np.clip(arr[:, :, c] + int(delta), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _pixelate(img: Image.Image, scale: float = 0.125) -> Image.Image:
    """Nearest-neighbor down/up (NTIRE val: pixelation)."""
    w, h = img.size
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((sw, sh), Image.NEAREST).resize((w, h), Image.NEAREST)


def _double_jpeg(img: Image.Image, q1: int = 70, q2: int = 40) -> Image.Image:
    """Two JPEG passes at different qualities — the usual repost path."""
    return jpeg_recompress(jpeg_recompress(img, q1), q2)


def _chroma_subsample(img: Image.Image, factor: int = 2) -> Image.Image:
    """4:2:0-style chroma decimation. Kills per-channel generator traces."""
    y, cb, cr = img.convert("YCbCr").split()
    w, h = img.size
    sw, sh = max(1, w // factor), max(1, h // factor)
    cb = cb.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    cr = cr.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def _median(img: Image.Image, size: int = 3) -> Image.Image:
    k = max(3, int(size) | 1)
    return img.filter(ImageFilter.MedianFilter(size=k))


def _unsharp(img: Image.Image, percent: int = 150, radius: float = 1.5) -> Image.Image:
    """Phone / social 'enhance' after compress — re-peaks frequencies."""
    return img.filter(ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent), threshold=2))


def _small_rotate(img: Image.Image, degrees: float) -> Image.Image:
    """Breaks grid-aligned spectral peaks from fixed upsamplers."""
    return img.rotate(float(degrees), resample=Image.BICUBIC, expand=False)


def _subpixel_nudge(img: Image.Image, dx: float, dy: float) -> Image.Image:
    return img.transform(
        img.size, Image.AFFINE, (1.0, 0.0, float(dx), 0.0, 1.0, float(dy)),
        resample=Image.BICUBIC,
    )


def _gamma(img: Image.Image, gamma: float) -> Image.Image:
    arr = np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)
    return Image.fromarray((np.power(arr, float(gamma)) * 255.0).astype(np.uint8))


def _film_grain(img: Image.Image, sigma: float = 0.04, scale: int = 8,
                rng: Optional[random.Random] = None) -> Image.Image:
    """Low-frequency grain (camera ISO), not white noise."""
    rng = rng or random
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    gh, gw = max(1, h // max(1, int(scale))), max(1, w // max(1, int(scale)))
    rs = np.random.RandomState(rng.randrange(1 << 31))
    grain = rs.randn(gh, gw).astype(np.float32) * (float(sigma) * 255.0)
    grain = np.array(Image.fromarray(grain, mode="F").resize((w, h), Image.BILINEAR))
    out = np.clip(arr + grain[:, :, None], 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def _chroma_aberration(img: Image.Image, pixels: int = 2) -> Image.Image:
    """Opposite R/B shift — cheap lens CA, breaks channel-aligned artifacts."""
    arr = np.array(img)
    px = int(pixels)
    if px:
        arr[:, :, 0] = np.roll(arr[:, :, 0], px, axis=1)
        arr[:, :, 2] = np.roll(arr[:, :, 2], -px, axis=1)
    return Image.fromarray(arr)


def _fft_lowpass(img: Image.Image, cutoff: float = 0.35) -> Image.Image:
    """Gaussian spectral cutoff. Diffusion/GAN fingerprints live at high f."""
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    yy = np.fft.fftfreq(h)[:, None]
    xx = np.fft.rfftfreq(w)[None, :]
    rr = np.sqrt(xx * xx + yy * yy)
    mask = np.exp(-0.5 * (rr / max(1e-6, float(cutoff))) ** 2).astype(np.float32)
    out = np.empty_like(arr)
    for c in range(3):
        f = np.fft.rfft2(arr[:, :, c])
        out[:, :, c] = np.fft.irfft2(f * mask, s=(h, w)).real
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _social_reencode(img: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """Messenger / Instagram path: 4:2:0 + harsh WebP or JPEG."""
    rng = rng or random
    img = _chroma_subsample(img, 2)
    if rng.random() < 0.5:
        return webp_recompress(img, rng.choice((20, 35, 50)))
    return jpeg_recompress(img, rng.choice((25, 35, 45)))


def _jpeg_grid_shift(img: Image.Image, dx: int = 3, dy: int = 2, quality: int = 40) -> Image.Image:
    """Shift off the 8×8 DCT grid, then JPEG. Classic anti-forensics."""
    shifted = img.transform(
        img.size, Image.AFFINE, (1.0, 0.0, float(dx), 0.0, 1.0, float(dy)),
        resample=Image.BICUBIC,
    )
    return jpeg_recompress(shifted, quality)


def _resample_mismatch(img: Image.Image, scale: float = 0.35,
                       down=None, up=None) -> Image.Image:
    """Down with one kernel, up with another — the usual 'saved from Photos' path."""
    down = down or Image.NEAREST
    up = up or Image.BICUBIC
    w, h = img.size
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((sw, sh), down).resize((w, h), up)


def _surface_blur(img: Image.Image, radius: float = 2.0, edge: float = 12.0) -> Image.Image:
    """Edge-preserving smooth. Kills periodic high-f without melting structure."""
    blur = img.filter(ImageFilter.GaussianBlur(float(radius)))
    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(blur, dtype=np.float32)
    mag = np.mean(np.abs(a - b), axis=2, keepdims=True)
    w = 1.0 / (1.0 + mag / max(1e-3, float(edge)))
    out = b * w + a * (1.0 - w)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _fft_phase_noise(img: Image.Image, amount: float = 0.35, cutoff: float = 0.18,
                     rng: Optional[random.Random] = None) -> Image.Image:
    """Jitter high-frequency phase. GAN/diffusion peaks live in the spectrum."""
    rng = rng or random
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    yy = np.fft.fftfreq(h)[:, None]
    xx = np.fft.rfftfreq(w)[None, :]
    high = (np.sqrt(xx * xx + yy * yy) > float(cutoff)).astype(np.float32)
    rs = np.random.RandomState(rng.randrange(1 << 31))
    out = np.empty_like(arr)
    for c in range(3):
        f = np.fft.rfft2(arr[:, :, c])
        noise = (rs.randn(*f.shape) + 1j * rs.randn(*f.shape)).astype(np.complex64)
        f = f * np.exp(1j * float(amount) * high * np.angle(noise + 1e-8))
        out[:, :, c] = np.fft.irfft2(f, s=(h, w)).real
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _hue_shift(img: Image.Image, degrees: float) -> Image.Image:
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(round(float(degrees) * 255.0 / 360.0))) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def _white_balance(img: Image.Image, r_gain: float = 1.1, b_gain: float = 0.9) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    arr[:, :, 0] *= float(r_gain)
    arr[:, :, 2] *= float(b_gain)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _chroma_noise(img: Image.Image, sigma: float = 0.04,
                  rng: Optional[random.Random] = None) -> Image.Image:
    """Camera-ISO-like noise on chroma only. Leaves luma fingerprints less intact."""
    rng = rng or random
    y, cb, cr = img.convert("YCbCr").split()
    rs = np.random.RandomState(rng.randrange(1 << 31))
    def _n(ch):
        a = np.asarray(ch, dtype=np.float32)
        a = np.clip(a + rs.randn(*a.shape).astype(np.float32) * (float(sigma) * 255.0), 0, 255)
        return Image.fromarray(a.astype(np.uint8))
    return Image.merge("YCbCr", (y, _n(cb), _n(cr))).convert("RGB")


def _vignette(img: Image.Image, strength: float = 0.35) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    yy = (np.linspace(-1.0, 1.0, h, dtype=np.float32))[:, None]
    xx = (np.linspace(-1.0, 1.0, w, dtype=np.float32))[None, :]
    fall = np.clip(1.0 - float(strength) * (xx * xx + yy * yy), 0.25, 1.0)
    return Image.fromarray(np.clip(arr * fall[:, :, None], 0, 255).astype(np.uint8))


def _perspective_nudge(img: Image.Image, pixels: float = 8.0,
                       rng: Optional[random.Random] = None) -> Image.Image:
    """Tiny homography — phone recapture / screenshot of a screen."""
    rng = rng or random
    w, h = img.size
    p = float(pixels)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(x + rng.uniform(-p, p), y + rng.uniform(-p, p)) for x, y in src]
    coeffs = _perspective_coeffs(src, dst)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)


def _perspective_coeffs(src, dst):
    matrix = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray([c for xy in src for c in xy], dtype=np.float64)
    try:
        coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return (1, 0, 0, 0, 1, 0, 0, 0)
    return tuple(float(x) for x in coeffs)


def _speckle(img: Image.Image, sigma: float = 0.08,
             rng: Optional[random.Random] = None) -> Image.Image:
    rng = rng or random
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rs = np.random.RandomState(rng.randrange(1 << 31))
    arr = np.clip(arr * (1.0 + rs.randn(*arr.shape).astype(np.float32) * float(sigma)), 0, 1)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _recode_stack(img: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """WhatsApp then Photos then Twitter: JPEG → WebP → JPEG at mixed Q."""
    rng = rng or random
    img = jpeg_recompress(img, rng.choice((80, 65, 50)))
    img = webp_recompress(img, rng.choice((55, 40, 25)))
    return jpeg_recompress(img, rng.choice((45, 30, 20)))


def _extra_train_distort(img: Image.Image, rng: random.Random,
                         kinds=None) -> Image.Image:
    """One op harder than the Pangram table, aimed at hiding generator cues."""
    kind = rng.choice(kinds if kinds is not None else (
        "jpeg", "doublejpeg", "webp", "blur", "impulse", "quantize", "motion",
        "shift", "pixelate", "bright", "chroma", "median", "unsharp", "rotate",
        "nudge", "gamma", "grain", "aberr", "fftlp", "autocontrast",
        "posterize", "social", "gridshift", "resample", "surface", "phase",
        "hue", "wb", "chroman", "equalize", "vignette", "perspective",
        "speckle", "recode",
    ))
    if kind == "jpeg":
        return jpeg_recompress(img, rng.choice((20, 10, 5)))
    if kind == "doublejpeg":
        return _double_jpeg(img, rng.choice((85, 70, 55)), rng.choice((45, 30, 15)))
    if kind == "webp":
        return webp_recompress(img, rng.choice((20, 35, 55)))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(rng.choice((3.0, 4.0))))
    if kind == "impulse":
        return _impulse_noise(img, rng.choice((0.01, 0.03)), rng)
    if kind == "quantize":
        return _quantize(img, rng.choice((64, 32, 16)))
    if kind == "motion":
        return _motion_blur(img, rng.choice((5, 9, 13)))
    if kind == "shift":
        return _color_shift(img, rng.choice((-40, -20, 20, 40)), rng.randrange(3))
    if kind == "pixelate":
        return _pixelate(img, rng.choice((0.125, 0.25)))
    if kind == "chroma":
        return _chroma_subsample(img, rng.choice((2, 2, 4)))
    if kind == "median":
        return _median(img, rng.choice((3, 5)))
    if kind == "unsharp":
        return _unsharp(img, rng.choice((120, 180)), rng.choice((1.0, 2.0)))
    if kind == "rotate":
        return _small_rotate(img, rng.choice((-7.0, -3.0, 2.5, 5.0, 8.0)))
    if kind == "nudge":
        return _subpixel_nudge(img, rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
    if kind == "gamma":
        return _gamma(img, rng.choice((0.65, 0.8, 1.25, 1.5)))
    if kind == "grain":
        return _film_grain(img, rng.choice((0.03, 0.06)), rng.choice((4, 8, 12)), rng)
    if kind == "aberr":
        return _chroma_aberration(img, rng.choice((1, 2, 3)))
    if kind == "fftlp":
        return _fft_lowpass(img, rng.choice((0.22, 0.32, 0.45)))
    if kind == "autocontrast":
        return ImageOps.autocontrast(img, cutoff=rng.choice((0, 1, 2)))
    if kind == "posterize":
        return ImageOps.posterize(img, rng.choice((3, 4, 5)))
    if kind == "social":
        return _social_reencode(img, rng)
    if kind == "gridshift":
        return _jpeg_grid_shift(
            img, rng.randrange(1, 8), rng.randrange(1, 8), rng.choice((55, 40, 25)),
        )
    if kind == "resample":
        box = getattr(Image, "BOX", Image.BILINEAR)
        lanczos = getattr(Image, "LANCZOS", Image.BICUBIC)
        down = rng.choice((Image.NEAREST, box, Image.BILINEAR))
        up = rng.choice((Image.BICUBIC, lanczos, Image.NEAREST))
        return _resample_mismatch(img, rng.choice((0.2, 0.35, 0.5)), down, up)
    if kind == "surface":
        return _surface_blur(img, rng.choice((1.5, 2.5, 3.5)), rng.choice((8.0, 14.0)))
    if kind == "phase":
        return _fft_phase_noise(img, rng.choice((0.2, 0.35, 0.5)), rng.choice((0.14, 0.22)), rng)
    if kind == "hue":
        return _hue_shift(img, rng.uniform(-25.0, 25.0))
    if kind == "wb":
        return _white_balance(img, rng.uniform(0.78, 1.22), rng.uniform(0.78, 1.22))
    if kind == "chroman":
        return _chroma_noise(img, rng.choice((0.03, 0.06)), rng)
    if kind == "equalize":
        return ImageOps.equalize(img)
    if kind == "vignette":
        return _vignette(img, rng.choice((0.25, 0.4, 0.55)))
    if kind == "perspective":
        return _perspective_nudge(img, rng.choice((4.0, 8.0, 12.0)), rng)
    if kind == "speckle":
        return _speckle(img, rng.choice((0.05, 0.10)), rng)
    if kind == "recode":
        return _recode_stack(img, rng)
    return ImageEnhance.Brightness(img).enhance(1.0 + rng.choice((-0.35, 0.35)))


# Official eval table (Pangram / GenImage / OmniAID). `--perturbation all`.
BENCHMARK_PERTURBATIONS = {
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
}

# Stronger / NTIRE-inspired levels (arxiv:2604.11487). `--perturbation extra`.
HARD_PERTURBATIONS = {
    "jpeg20": (lambda im: jpeg_recompress(im, 20), "JPEG quality 20"),
    "jpeg10": (lambda im: jpeg_recompress(im, 10), "JPEG quality 10"),
    "blur4.0": (lambda im: im.filter(ImageFilter.GaussianBlur(4.0)), "Gaussian blur sigma 4.0"),
    "resize0.125": (lambda im: _resize_perturb(im, 0.125), "resize 0.125x then upscale"),
    "noise0.20": (lambda im: _noise_perturb(im, 0.20), "Gaussian noise sigma 0.20"),
    "jitter40": (lambda im: _jitter_perturb(im, 0.4), "brightness/contrast/saturation +40%"),
    "crop60": (lambda im: center_crop(im, 0.6), "center crop 60%"),
    "impulse0.02": (lambda im: _impulse_noise(im, 0.02, random.Random(0)), "impulse noise 2%"),
    "quantize32": (lambda im: _quantize(im, 32), "color quantization 32"),
    "motion9": (lambda im: _motion_blur(im, 9), "motion blur length 9"),
    "shift20": (lambda im: _color_shift(im, 20, 0), "RGB channel shift +20"),
    "pixelate8": (lambda im: _pixelate(im, 0.125), "pixelate 8x"),
    "jpeg5": (lambda im: jpeg_recompress(im, 5), "JPEG quality 5"),
    "doublejpeg": (lambda im: _double_jpeg(im, 70, 35), "double JPEG 70 then 35"),
    "webp20": (lambda im: webp_recompress(im, 20), "WebP quality 20"),
    "chroma420": (lambda im: _chroma_subsample(im, 2), "4:2:0 chroma subsample"),
    "median3": (lambda im: _median(im, 3), "median filter 3"),
    "unsharp": (lambda im: _unsharp(im, 150, 1.5), "unsharp mask"),
    "rotate3": (lambda im: _small_rotate(im, 3.0), "rotate 3 degrees"),
    "nudge1": (lambda im: _subpixel_nudge(im, 0.7, -0.4), "subpixel translate"),
    "gamma07": (lambda im: _gamma(im, 0.7), "gamma 0.7"),
    "grain": (lambda im: _film_grain(im, 0.04, 8, random.Random(0)), "low-frequency film grain"),
    "aberr2": (lambda im: _chroma_aberration(im, 2), "chromatic aberration 2px"),
    "fftlp": (lambda im: _fft_lowpass(im, 0.32), "FFT Gaussian low-pass"),
    "autocontrast": (lambda im: ImageOps.autocontrast(im), "histogram autocontrast"),
    "posterize4": (lambda im: ImageOps.posterize(im, 4), "posterize 4 bits"),
    "social": (lambda im: _social_reencode(im, random.Random(0)), "4:2:0 + messenger re-encode"),
    "gridshift": (lambda im: _jpeg_grid_shift(im, 3, 5, 40), "shift off 8x8 grid then JPEG"),
    "resample": (lambda im: _resample_mismatch(im, 0.35, Image.NEAREST, Image.BICUBIC), "nearest down / bicubic up"),
    "surface": (lambda im: _surface_blur(im, 2.0, 12.0), "edge-preserving surface blur"),
    "phase": (lambda im: _fft_phase_noise(im, 0.35, 0.18, random.Random(0)), "FFT high-frequency phase noise"),
    "hue18": (lambda im: _hue_shift(im, 18.0), "hue rotate +18 deg"),
    "wb": (lambda im: _white_balance(im, 1.15, 0.88), "warm white balance"),
    "chroman": (lambda im: _chroma_noise(im, 0.04, random.Random(0)), "chroma-only noise"),
    "equalize": (lambda im: ImageOps.equalize(im), "histogram equalize"),
    "vignette": (lambda im: _vignette(im, 0.4), "radial vignette"),
    "perspective": (lambda im: _perspective_nudge(im, 8.0, random.Random(0)), "tiny perspective recapture"),
    "speckle": (lambda im: _speckle(im, 0.08, random.Random(0)), "multiplicative speckle"),
    "recode": (lambda im: _recode_stack(im, random.Random(0)), "JPEG then WebP then JPEG"),
}

PERTURBATIONS = {
    **BENCHMARK_PERTURBATIONS,
    **HARD_PERTURBATIONS,
    "pangram": (lambda im: pangram_augment(im), "Pangram augmented protocol (1024px + JPEG q50)"),
}


def perturbation_names(spec: Optional[str]) -> Optional[list]:
    """Resolve a sweep name to an ordered list of perturbation keys."""
    if spec in (None, "", "clean"):
        return None
    if spec == "all":
        return list(BENCHMARK_PERTURBATIONS)
    if spec in ("extra", "hard"):
        return list(HARD_PERTURBATIONS)
    if spec in ("all+extra", "all+hard"):
        return list(dict.fromkeys([*BENCHMARK_PERTURBATIONS, *HARD_PERTURBATIONS]))
    return None


def apply_perturbation(img: Image.Image, name: str) -> Image.Image:
    """Apply a named benchmark perturbation (deterministic, both classes).
    Used by `main.py eval --perturbation <name>` to produce robustness
    tables like the GenImage / OmniAID / Pangram evaluations."""
    if name not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation '{name}'. Available: {list(PERTURBATIONS)}")
    return PERTURBATIONS[name][0](img.convert("RGB"))
