"""Configuration for Seer training / evaluation.

Configs are plain YAML files mirroring the dataclasses below; anything left
out falls back to these defaults. CLI overrides use dotted paths, e.g.
  python main.py train --config configs/x.yaml --set max_steps=10 res=224
"""

import dataclasses
import os
import typing
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SourceSpec:
    """One component of the training data mixture.

    type:
      * "comfor"  - OwensLab Community Forensics schema (image_data bytes,
                    label, model_name, architecture). Reads local parquet
                    shards (downloaded once via scripts/fetch_data.py to
                    local_dirs) and falls back to HF hub streaming.
      * "hf"      - any HF dataset; give image_col and either label_col or a
                    fixed label (e.g. modern-generator sets, crawled AI art).
                    label_map remaps raw values onto 0=real / 1=fake;
                    keep_label drops other classes after remapping.
      * "folders" - local image dirs; real_dirs / fake_dirs (either optional).
                    Used for materialized downloads and synthetic mirrors.
      * "ntire"   - NTIRE 2026 labelled splits (via get_datasets.py).
                    split is train / val / val_hard / test; shard selects
                    a train shard (shard < 0 = every train shard). Labels
                    live in CSV, not folder names.

    weight: expected fraction of training samples drawn from this source.
    """

    name: str = "src"
    type: str = "comfor"
    weight: float = 1.0
    # comfor / hf
    dataset: str = ""
    split: str = "train"
    shuffle_buffer: int = 4096
    max_samples: Optional[int] = None
    # local parquet shards for comfor-type sources (F:/techjam/<name>)
    local_dirs: List[str] = field(default_factory=list)
    # hf-generic
    image_col: str = "image"
    label_col: Optional[str] = None
    label: Optional[int] = None  # fixed label when label_col is None
    generator_col: Optional[str] = None
    # remaps raw column values onto our 0=real / 1=fake convention
    label_map: Optional[dict] = None
    keep_label: Optional[int] = None  # drop rows whose mapped label differs
    # folders
    real_dirs: List[str] = field(default_factory=list)
    fake_dirs: List[str] = field(default_factory=list)
    # ntire (get_datasets.py --only ntire-train / ntire-val / ntire-test)
    shard: int = 0  # train shard index; < 0 loads every downloaded shard
    hard: bool = False
    clean_only: bool = False


@dataclass
class DataConfig:
    """Where training images come from.

    source:
      * "mixture"  - combine `sources` (weighted); the recommended setup
      * "comfor"  - single streaming source, OwensLab schema
      * "folders" - single local-dir source
      * "ntire"   - NTIRE 2026 labelled split (train shard / val / test)
    """

    source: str = "mixture"
    sources: List[SourceSpec] = field(default_factory=list)
    dataset: str = "OwensLab/CommunityForensics-Small"
    split: str = "train"
    local_dirs: List[str] = field(default_factory=list)  # local parquet for comfor
    real_dirs: List[str] = field(default_factory=list)
    fake_dirs: List[str] = field(default_factory=list)
    streaming: bool = True
    shuffle_buffer: int = 4096
    max_samples: Optional[int] = None  # cap streamed samples (None = all)
    val_max_samples: int = 2048
    val_seed: int = 1
    # Draw real/fake with equal probability, then pick a source that can
    # produce that class (weights still apply within a class). Stops fake-only
    # families from dominating the batch.
    balance_labels: bool = False


@dataclass
class AugmentConfig:
    """Wild-simulation augmentations.

    Symmetric across real/fake so the model cannot solve the task from the
    augmentation itself (e.g. "has JPEG artifacts") and must instead learn
    generator fingerprints. Parameter levels follow the benchmark
    robustness protocols (GenImage / OmniAID / Pangram's augmented eval):

      JPEG    q in {90, 70, 50, 30}
      blur    sigma in {0.5, 1.0, 2.0}
      resize  0.5x / 0.25x then upscale
      noise   sigma in {0.02, 0.05, 0.10}
      jitter  +/-20% brightness/contrast/saturation
      crop    center crop 80%
    """

    train: bool = True
    hflip_prob: float = 0.5
    scale_range: List[float] = field(default_factory=lambda: [0.35, 1.0])
    # deterministic benchmark-style crop, used instead of RandomResizedCrop
    center_crop_prob: float = 0.15
    center_crop_scale: float = 0.8
    # compression
    jpeg_prob: float = 0.75
    jpeg_quality: List[int] = field(default_factory=lambda: [90, 70, 50, 30])
    webp_prob: float = 0.10
    webp_quality: List[int] = field(default_factory=lambda: [50, 95])
    grayscale_prob: float = 0.05
    # resolution loss
    downscale_prob: float = 0.30
    downscale_levels: List[float] = field(default_factory=lambda: [0.5, 0.25])
    # blur / noise
    blur_prob: float = 0.10
    blur_sigma: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    noise_prob: float = 0.15
    noise_levels: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])
    # photometric
    color_jitter_prob: float = 0.30
    color_jitter: float = 0.2  # +/-20%


@dataclass
class CompositeConfig:
    """Composite training: layer image crops over a base image so the
    global head sees mixed content while the local head must localize
    which patches are AI. Produces interpretable heatmaps at inference time.

    Compositing is itself a simple discontinuity, so every top-on-base
    class pairing is trained, not just fake-over-real - otherwise the
    local head can win by detecting seams instead of AI texture:
      fake_on_real  - fake crop over a real base (localized patch labels)
      real_on_fake  - real crop over a fake base (inverted patch labels:
                      only the pasted region is real)
      fake_on_fake  - fake crop over a fake base (all patches stay fake;
                      seams alone are never a fake cue)
      real_on_real  - real crop over a real base (label stays real;
                      blending alone is never a fake cue)

    Overlays are crops of the source (own scale / aspect / flip) pasted
    into random regions, and a composited sample receives a stack of
    1..max_overlays pastes; each paste overwrites the patch labels in its
    footprint (last layer wins). The global label is derived from what
    remains visible: any patch fake -> fake.

    prob               - chance a fake sample is composited (one of the
                         three label-1 pairings above)
    real_real_fraction - chance a real sample is composited, relative to
                         `prob` (real-on-real keeps label 0, so the batch
                         class balance is unchanged)
    fake_on_real / real_on_fake / fake_on_fake
                      - weights splitting the composites on fake samples
                        between the three label-1 pairings
    max_overlays      - pastes per composited sample (uniform 1..max)
    patch_loss_weight - weight of the per-patch BCE in the total loss
    mode              - how each overlay is laid over the base:
                         "blend" soft alpha compositing (diffusion-style
                                seamless mixes)
                         "paste" hard-edged opaque overlay with a feathered
                                border (sticker / screenshot-style)
                         "mixed" randomly choose per overlay
    """

    prob: float = 0.25
    real_real_fraction: float = 0.25
    fake_on_real: float = 0.5
    real_on_fake: float = 0.25
    fake_on_fake: float = 0.25
    max_overlays: int = 3
    patch_loss_weight: float = 0.5
    mode: str = "mixed"


@dataclass
class ProbeConfig:
    """Page-level multi-layer linear probe (frozen backbone).

    An alternative to continuation fine-tuning: freeze the backbone and
    train a single linear layer on features tapped from several
    transformer blocks. Early blocks carry high-frequency / low-level
    statistics - where generator fingerprints live - while mid and late
    blocks carry increasingly semantic features, so the probe sees both.

    Each tap contributes [CLS ; mean(patch tokens)] for the page head and
    the raw patch tokens for a separate heatmap head; taps are concatenated
    and standardized (LayerNorm) before each linear map. The two heads do
    not share weights.

    layers - 0-based encoder block indices to tap; negative counts from
              the end (-1 = final block). Empty = auto: four evenly spaced
              taps from early to late (e.g. [6, 12, 18, 23] on ViT-L's 24
              blocks).
    """

    enabled: bool = False
    layers: List[int] = field(default_factory=list)


@dataclass
class TrainConfig:
    # model
    backbone: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    pretrained: bool = True
    freeze_backbone: bool = False  # ablation only; continuation training is better
    res: int = 512  # training resolution (must be a multiple of patch size)

    # optimization
    batch_size: int = 8
    grad_accum: int = 4
    epochs: float = 1.0
    max_steps: Optional[int] = None  # overrides epochs when set
    lr: float = 1.0e-5  # backbone base LR (LLRD scales down from here)
    head_lr: float = 1.0e-4
    llrd: float = 0.8  # layer-wise LR decay for the backbone
    weight_decay: float = 0.05
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    clip_grad: float = 1.0
    bf16: bool = True
    grad_checkpointing: bool = True
    ema_decay: float = 0.999  # <= 0 disables EMA

    # bookkeeping
    eval_every: int = 1000
    log_every: int = 20
    seed: int = 0
    out_dir: str = "runs/seer"
    resume: Optional[str] = None
    decode_workers: int = 8  # threads for image decode+augment in the collate
    num_workers: int = 0  # DataLoader processes; 0 = in-process (safer after CUDA)
    loader_readers: int = 1  # threads independently iterating the mixture
    prefetch_depth: int = 2  # collated batches queued ahead of the GPU

    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)


def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls, d):
    """Recursively construct nested dataclasses (incl. List[Dataclass])."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for k, v in d.items():
        hint = hints.get(k)
        origin = typing.get_origin(hint)
        if dataclasses.is_dataclass(hint) and isinstance(v, dict):
            v = _build(hint, v)
        elif origin is list:
            args = typing.get_args(hint)
            if args and dataclasses.is_dataclass(args[0]) and isinstance(v, list):
                v = [_build(args[0], item) if isinstance(item, dict) else item for item in v]
        kwargs[k] = v
    return cls(**kwargs)


def apply_overrides(d: dict, overrides: List[str]) -> dict:
    """Apply 'dotted.path=value' overrides; values parsed as YAML."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.subkey=value, got: {item!r}")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return d


def _rewrite_data_paths(obj):
    """Map Windows ``F:/techjam`` paths and ``$SEER_DATA_ROOT`` onto DATA_ROOT."""
    from .paths import DATA_ROOT

    prefix = "F:/techjam"

    def one(value):
        if not isinstance(value, str):
            return value
        expanded = os.path.expandvars(value)
        if expanded.lower().startswith(prefix.lower()):
            rest = expanded[len(prefix):].lstrip("/\\")
            return str(DATA_ROOT / rest)
        return expanded

    if isinstance(obj, dict):
        return {k: _rewrite_data_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_data_paths(v) for v in obj]
    return one(obj)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> TrainConfig:
    d = asdict(TrainConfig())
    if path:
        with open(path, "r", encoding="utf-8") as f:
            d = _deep_update(d, yaml.safe_load(f) or {})
    d = apply_overrides(d, overrides)
    return _build(TrainConfig, _rewrite_data_paths(d))


def config_to_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
