"""Data loading and the training-data mixture.

The mixture is the core of the recipe (Pangram Image's blog is explicit that
data composition mattered more than anything else they tried). Sources:

  * ComforStream   - OwensLab Community Forensics schema (streaming from HF).
                     4,803 open generators; CVPR 2025 showed generator
                     *diversity* is the main driver of generalization.
  * HFGenericStream - any HF dataset (image column + fixed/derived label) -
                     e.g. modern-generator or crawled-AI-art sets.
  * FolderPairStream - local dirs of real/fake images (materialized
                     downloads, synthetic mirrors, GenImage, Synthbuster...).
  * MixtureDataset  - weighted combination of the above, cycled forever.

All samples are normalized to dicts:
  {image: PIL.Image, label: 0|1, generator: str, architecture: str}
"""

import io
import os
import random
import re
from pathlib import Path
from typing import Iterator, List, Mapping, Optional

import torch
import torch.nn.functional as F
from PIL import Image

try:  # silence dataset download progress bars
    import datasets as hfds

    try:
        hfds.disable_progress_bars()
    except Exception:
        pass
except Exception:  # pragma: no cover
    hfds = None

from .labels import normalize_label

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def parquet_files(dirs: List[str]) -> List[str]:
    """All parquet shards under the given dirs (or the files themselves),
    naturally sorted (shard_2 before shard_10)."""
    files: List[str] = []
    for d in dirs or []:
        p = Path(d)
        if p.is_file():
            files.append(str(p))
        else:
            files.extend(str(f) for f in p.rglob("*.parquet"))
    return sorted(set(files), key=_natural_key)


def count_parquet_rows(files: List[str]) -> int:
    """Exact row count from parquet footers (fast, no data read)."""
    try:
        import pyarrow.parquet as pq

        total = 0
        for f in files:
            try:
                total += pq.ParquetFile(f).metadata.num_rows
            except Exception:
                pass
        return total
    except Exception:
        return 0


def _to_bool(v) -> bool:
    """Datasets are inconsistent: nsfw_flag is bool in -Small but a string
    like 'False' in -Eval (where bool('False') would be True)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def decode_row(row: dict) -> dict:
    img = Image.open(io.BytesIO(row["image_data"])).convert("RGB")
    return {
        "image": img,
        "label": int(row.get("label", 0)),
        "generator": row.get("model_name") or "",
        "architecture": row.get("architecture") or "",
        "image_name": row.get("image_name") or "",
        "prompt": (row.get("prompt") or "")[:512],
        "nsfw_flag": _to_bool(row.get("nsfw_flag", False)),
    }


def _as_pil(value):
    """HF datasets can hand us decoded PIL images, numpy arrays or bytes."""
    if value is None:
        raise ValueError("null image")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, dict) and ("bytes" in value or "path" in value):
        data = value.get("bytes")
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")
        return Image.open(value["path"]).convert("RGB")
    import numpy as np

    return Image.fromarray(np.asarray(value)).convert("RGB")


def load_sample_image(s: dict) -> Image.Image:
    """Materialize the PIL image of a sample, whether it carries an eager
    image, raw bytes (lazy parquet decode) or a file path."""
    if s.get("image") is not None:
        return s["image"]
    if s.get("image_bytes"):
        return Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
    if s.get("image_path"):
        return Image.open(s["image_path"]).convert("RGB")
    raise ValueError("sample has no image / image_bytes / image_path")


class ComforStream(torch.utils.data.IterableDataset):
    """Community Forensics reader.

    If local_dirs contain parquet shards (from scripts/fetch_data.py), reads
    them locally - no network, no arrow cache duplication, identical schema.
    Otherwise streams from the HF hub. Decoding is lazy by default: raw bytes
    are handed to the collate so a decode thread pool can do the PIL work in
    parallel (see BatchBuilder).
    """

    def __init__(
        self,
        dataset: str = "OwensLab/CommunityForensics-Small",
        split: str = "train",
        shuffle_buffer: int = 4096,
        max_samples: Optional[int] = None,
        seed: int = 0,
        local_dirs: Optional[List[str]] = None,
        lazy_decode: bool = True,
    ):
        super().__init__()
        if hfds is None:
            raise RuntimeError("`datasets` is required for Community Forensics streaming")
        self._dataset = dataset
        self._split = split
        self._shuffle_buffer = shuffle_buffer
        self._max_samples = max_samples
        self._seed = seed
        self._local_files = parquet_files(local_dirs) if local_dirs else None
        self._lazy_decode = lazy_decode

    def _iter_rows(self):
        if self._local_files:
            ds = hfds.load_dataset("parquet", data_files=self._local_files, split="train", streaming=True)
        else:
            ds = hfds.load_dataset(self._dataset, split=self._split, streaming=True)
        if self._shuffle_buffer > 0:
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer)
        if self._max_samples:
            ds = ds.take(self._max_samples)
        n = 0
        for row in ds:
            try:
                if self._lazy_decode:
                    yield {
                        "image": None,
                        "image_bytes": bytes(row["image_data"]),
                        "label": int(row.get("label", 0)),
                        "generator": row.get("model_name") or "",
                        "architecture": row.get("architecture") or "",
                        "image_name": row.get("image_name") or "",
                        "prompt": (row.get("prompt") or "")[:512],
                        "nsfw_flag": _to_bool(row.get("nsfw_flag", False)),
                    }
                else:
                    yield decode_row(row)
            except Exception:
                continue  # skip corrupt rows
            n += 1
            if self._max_samples and n >= self._max_samples:
                break

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        it = self._iter_rows()
        if worker is not None and worker.num_workers > 1:
            it = _shard(it, worker.num_workers, worker.id)
        yield from it


class HFGenericStream(torch.utils.data.IterableDataset):
    """Any HF dataset: read `image_col` as the image and derive the label
    from `label_col` (if given) or a fixed `label` (e.g. an all-fake set).

    ``label_map`` remaps raw values onto 0=real / 1=fake. ``keep_label``
    drops every other class after remapping — use this for fake-only
    slices of mixed sets whose ClassLabel is inverted.
    """

    def __init__(
        self,
        dataset: str,
        split: str = "train",
        image_col: str = "image",
        label_col: Optional[str] = None,
        label: Optional[int] = None,
        generator_col: Optional[str] = None,
        shuffle_buffer: int = 1024,
        max_samples: Optional[int] = None,
        seed: int = 0,
        name: str = "",
        label_map: Optional[Mapping] = None,
        keep_label: Optional[int] = None,
        local_dirs: Optional[List[str]] = None,
    ):
        super().__init__()
        if hfds is None:
            raise RuntimeError("`datasets` is required for HF streaming")
        self._dataset = dataset
        self._split = split
        self._image_col = image_col
        self._label_col = label_col
        self._label = label
        self._generator_col = generator_col
        self._shuffle_buffer = shuffle_buffer
        self._max_samples = max_samples
        self._seed = seed
        self._name = name or dataset
        self._label_map = dict(label_map) if label_map else None
        self._keep_label = keep_label
        self._local_files = parquet_files(local_dirs) if local_dirs else None

    def _label_of(self, row: dict) -> Optional[int]:
        if self._label_col is not None:
            return normalize_label(row.get(self._label_col), self._label_map)
        return int(self._label if self._label is not None else 0)

    def _iter_rows(self):
        if self._local_files:
            ds = hfds.load_dataset("parquet", data_files=self._local_files, split="train", streaming=True)
        else:
            ds = hfds.load_dataset(self._dataset, split=self._split, streaming=True)
        if self._shuffle_buffer > 0:
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer)
        n = 0
        for row in ds:
            try:
                lab = self._label_of(row)
                if lab is None:
                    continue
                if self._keep_label is not None and int(lab) != int(self._keep_label):
                    continue
                img = _as_pil(row.get(self._image_col))
                gen = str(row.get(self._generator_col)) if self._generator_col else self._name
                yield {"image": img, "label": int(lab), "generator": gen or self._name,
                       "architecture": "", "image_name": ""}
            except Exception:
                continue
            n += 1
            if self._max_samples and n >= self._max_samples:
                break

    def __iter__(self):
        yield from self._iter_rows()


def _shard(it: Iterator, num: int, index: int) -> Iterator:
    for i, x in enumerate(it):
        if i % num == index:
            yield x


class FolderDataset(torch.utils.data.Dataset):
    """Images on disk. `roots` may be a dir (scanned recursively) or a file
    listing image paths (one per line)."""

    def __init__(self, roots: List[str], label: int):
        super().__init__()
        self.files: List[str] = []
        for root in roots or []:
            p = Path(root)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    self.files.extend(line.strip() for line in f if line.strip())
            else:
                for dirpath, _, filenames in os.walk(p):
                    for fn in filenames:
                        if Path(fn).suffix.lower() in IMAGE_EXTS:
                            self.files.append(str(Path(dirpath) / fn))
        if not self.files:
            raise FileNotFoundError(f"No images found under {list(roots or [])}")
        self.label = label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i) -> dict:
        img = Image.open(self.files[i]).convert("RGB")
        gen = Path(self.files[i]).parent.name
        return {"image": img, "label": self.label, "generator": gen,
                "architecture": "", "image_name": self.files[i]}


class FolderPairStream:
    """Infinite iterator over a folders source, interleaving real and fake
    (when both are given) and reshuffling each pass."""

    def __init__(self, real_dirs: List[str], fake_dirs: List[str], seed: int = 0):
        self.real_files = self._scan(real_dirs) if real_dirs else []
        self.fake_files = self._scan(fake_dirs) if fake_dirs else []
        if not self.real_files and not self.fake_files:
            raise FileNotFoundError(
                f"No images found (real_dirs={real_dirs}, fake_dirs={fake_dirs})"
            )
        self.seed = seed

    @staticmethod
    def _scan(roots: List[str]) -> List[str]:
        files: List[str] = []
        for root in roots:
            p = Path(root)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    files.extend(line.strip() for line in f if line.strip())
            else:
                for dirpath, _, filenames in os.walk(p):
                    for fn in filenames:
                        if Path(fn).suffix.lower() in IMAGE_EXTS:
                            files.append(str(Path(dirpath) / fn))
        return files

    def _one_pass(self, rng: random.Random) -> Iterator[dict]:
        reals = list(self.real_files)
        fakes = list(self.fake_files)
        rng.shuffle(reals)
        rng.shuffle(fakes)
        # interleave: if only one side exists, just stream it
        # samples are lazy (image_path) - the collate's decode pool
        # materializes them in parallel
        if not reals:
            for f in fakes:
                yield {"image": None, "image_path": f, "label": 1,
                       "generator": Path(f).parent.name, "architecture": "", "image_name": f}
            return
        if not fakes:
            for f in reals:
                yield {"image": None, "image_path": f, "label": 0,
                       "generator": Path(f).parent.name, "architecture": "", "image_name": f}
            return
        i = j = 0
        while i < len(reals) or j < len(fakes):
            if i < len(reals):
                yield {"image": None, "image_path": reals[i], "label": 0,
                       "generator": Path(reals[i]).parent.name, "architecture": "",
                       "image_name": reals[i]}
                i += 1
            if j < len(fakes):
                yield {"image": None, "image_path": fakes[j], "label": 1,
                       "generator": Path(fakes[j]).parent.name, "architecture": "",
                       "image_name": fakes[j]}
                j += 1

    def __iter__(self):
        while True:
            rng = random.Random(self.seed)
            yield from self._one_pass(rng)
            self.seed += 1  # new shuffle order each pass


def _open(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class NtireStream:
    """NTIRE 2026 labelled split. Labels live in CSV, not folder names.

    Fetch first with ``python get_datasets.py --only ntire-train`` (or
    ``--tier 1``). Yields the same lazy dict as FolderPairStream so the
    collate decode pool can materialize images in parallel.
    """

    def __init__(
        self,
        split: str = "train",
        shard: int = 0,
        hard: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 0,
        clean_only: bool = False,
    ):
        from .ntire import load_split

        samples = load_split(split, shard=shard, hard=hard)
        if clean_only:
            samples = [s for s in samples if not s.is_distorted]
        self.samples = samples
        self.max_samples = max_samples
        self.seed = seed

    def __iter__(self):
        items = list(self.samples)
        rng = random.Random(self.seed)
        rng.shuffle(items)
        if self.max_samples:
            items = items[: self.max_samples]
        for s in items:
            yield {
                "image": None,
                "image_path": str(s.path),
                "label": int(s.label),
                "generator": "",
                "architecture": "",
                "image_name": s.path.name,
            }


def _source_iterator(spec, seed: int) -> Iterator[dict]:
    """Build a (re-creatable) iterator for one source spec."""
    if spec.type == "ntire":
        return iter(NtireStream(
            split=spec.split,
            shard=spec.shard,
            hard=spec.hard,
            max_samples=spec.max_samples,
            seed=seed,
            clean_only=spec.clean_only,
        ))
    if spec.type == "comfor":
        ds = ComforStream(
            dataset=spec.dataset or "OwensLab/CommunityForensics-Small",
            split=spec.split,
            shuffle_buffer=spec.shuffle_buffer,
            max_samples=spec.max_samples,
            seed=seed,
            local_dirs=spec.local_dirs or None,
        )
        return iter(ds)
    if spec.type == "hf":
        ds = HFGenericStream(
            dataset=spec.dataset,
            split=spec.split,
            image_col=spec.image_col,
            label_col=spec.label_col,
            label=spec.label,
            generator_col=spec.generator_col,
            shuffle_buffer=spec.shuffle_buffer,
            max_samples=spec.max_samples,
            seed=seed,
            name=spec.name,
            label_map=spec.label_map,
            keep_label=spec.keep_label,
            local_dirs=spec.local_dirs or None,
        )
        return iter(ds)
    if spec.type == "folders":
        return iter(FolderPairStream(spec.real_dirs, spec.fake_dirs, seed=seed))
    raise ValueError(f"Unknown source type '{spec.type}'")


class MixtureDataset(torch.utils.data.IterableDataset):
    """Weighted mixture over sources, cycled indefinitely.

    Each sample is drawn from source i with probability weight_i / sum(w).
    Sources that exhaust (folder passes, capped streams) are transparently
    restarted. Note: HF *streaming* sources re-download on restart - for big
    repeated passes, materialize them to folders with scripts/download_data.py.
    """

    def __init__(self, sources, seed: int = 0):
        super().__init__()
        if not sources:
            raise ValueError("MixtureDataset needs at least one source")
        self.sources = list(sources)
        self.weights = [max(0.0, float(s.weight)) for s in self.sources]
        if sum(self.weights) <= 0:
            raise ValueError("All mixture weights are zero")
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        iters = [None] * len(self.sources)

        def next_from(i):
            if iters[i] is None:
                iters[i] = _source_iterator(self.sources[i], seed=rng.randrange(1 << 30))
            try:
                return next(iters[i])
            except StopIteration:
                iters[i] = _source_iterator(self.sources[i], seed=rng.randrange(1 << 30))
                return next(iters[i])

        while True:
            i = rng.choices(range(len(self.sources)), weights=self.weights, k=1)[0]
            try:
                yield next_from(i)
            except StopIteration:  # empty source, permanently
                continue


class ConcatDataset(torch.utils.data.Dataset):
    """Index-addressable concatenation (for folder datasets)."""

    def __init__(self, parts):
        self.parts = parts
        self.lens = [len(p) for p in parts]
        self.total = sum(self.lens)

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        for part, n in zip(self.parts, self.lens):
            if i < n:
                return part[i]
            i -= n
        raise IndexError


def build_train_dataset(cfg) -> torch.utils.data.Dataset:
    d = cfg.data
    if d.source == "mixture" and d.sources:
        return MixtureDataset(d.sources, seed=cfg.seed)
    if d.source in ("mixture", "comfor"):
        # explicit comfor, or a mixture with no sources: Community Forensics
        return ComforStream(
            dataset=d.dataset,
            split=d.split,
            shuffle_buffer=d.shuffle_buffer,
            max_samples=d.max_samples,
            seed=cfg.seed,
            local_dirs=d.local_dirs or None,
        )
    if d.source == "folders":
        parts = []
        if d.real_dirs:
            parts.append(FolderDataset(d.real_dirs, 0))
        if d.fake_dirs:
            parts.append(FolderDataset(d.fake_dirs, 1))
        return ConcatDataset(parts)
    if d.source == "ntire":
        return NtireStream(
            split=d.split,
            max_samples=d.max_samples,
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown data source '{d.source}'")


def build_val_dataset(cfg) -> torch.utils.data.Dataset:
    """A held-out slice for monitoring during training (train-distribution)."""
    import dataclasses

    d = cfg.data
    if d.source == "mixture" and d.sources:
        # monitor on the streaming-friendly sources, capped (never mutate the
        # training specs: build fresh ones via dataclasses.replace)
        specs = [s for s in d.sources if s.type in ("comfor", "hf", "ntire")]
        specs = specs or list(d.sources)
        capped = [
            dataclasses.replace(s, max_samples=max(1, d.val_max_samples // len(specs)))
            for s in specs
        ]
        return MixtureDataset(capped, seed=d.val_seed)
    if d.source == "comfor":
        return ComforStream(
            dataset=d.dataset,
            split=d.split,
            shuffle_buffer=min(d.shuffle_buffer, 2048),
            max_samples=d.val_max_samples,
            seed=d.val_seed,
            local_dirs=d.local_dirs or None,
        )
    return build_train_dataset(cfg)  # folders: reuse train set for monitoring


class BatchBuilder:
    """Collate fn: decode + augmentation + composite training -> batch tensors.

    Image decode and wild-simulation augmentation are the CPU bottleneck of
    the pipeline, so they run in a small thread pool (PIL releases the GIL for
    decode/encode). Sources hand us lazy samples (raw bytes or file paths);
    everything is materialized here, in parallel.
    """

    def __init__(self, cfg, train: bool, patch_grid: int, seed: int = 0,
                 decode_workers: Optional[int] = None):
        self.cfg = cfg
        self.train = train and cfg.augment.train
        self.comp = cfg.composite if train else None
        self.res = cfg.res
        self.G = patch_grid
        self.rng = random.Random(seed)
        if decode_workers is None:
            decode_workers = getattr(cfg, "decode_workers", 0)
        self._pool = None
        if decode_workers and decode_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=int(decode_workers))

    def _process_one(self, s: dict, seed: int) -> torch.Tensor:
        from .augment import eval_transform, train_transform

        img = load_sample_image(s)
        if self.train:
            return train_transform(img, self.res, random.Random(seed), self.cfg.augment)
        return eval_transform(img, self.res)

    # -- composite ---------------------------------------------------------

    def _pick_mode(self) -> str:
        mode = getattr(self.comp, "mode", "blend")
        if mode == "mixed":
            mode = self.rng.choice(["blend", "paste"])
        return mode

    def _weighted_choice(self, weights) -> Optional[str]:
        """Draw a key by weight; None when every weight is zero."""
        total = sum(max(0.0, float(w)) for w in weights.values())
        if total <= 0.0:
            return None
        r = self.rng.random() * total
        for name, w in weights.items():
            r -= max(0.0, float(w))
            if r <= 0.0:
                return name
        return list(weights)[-1]

    def _rand_rect(self):
        """One random rectangle in patch-grid space -> (y0, x0, h, w) in cells."""
        G = self.G
        w = max(1, int(G * self.rng.uniform(0.15, 0.7)))
        h = max(1, int(G * self.rng.uniform(0.15, 0.7)))
        return self.rng.randint(0, G - h), self.rng.randint(0, G - w), h, w

    def _region_alpha(self, rect, mode):
        """Compositing weight (res x res) for one paste:
          "paste" - opaque hard-edged rect with a ~2px feathered border
          "blend" - smooth bilinear alpha in [0.75, 1.0]
        """
        y0, x0, h, w = rect
        if mode == "paste":
            c = self.res // self.G
            a = torch.zeros(self.res, self.res)
            a[y0 * c:(y0 + h) * c, x0 * c:(x0 + w) * c] = 1.0
            return F.avg_pool2d(a[None, None], 5, stride=1, padding=2)[0, 0].clamp(0.0, 1.0)
        m = torch.zeros(self.G, self.G)
        m[y0:y0 + h, x0:x0 + w] = 1.0
        m = F.avg_pool2d(m[None, None], 3, stride=1, padding=1)[0, 0]
        a = F.interpolate(m[None, None], size=(self.res, self.res), mode="bilinear",
                          align_corners=False)[0, 0].clamp(0.0, 1.0)
        return a * self.rng.uniform(0.75, 1.0)

    def _crop_for(self, src, ph, pw):
        """A crop of `src` resampled to (ph, pw).

        Overlays are crops of the source image - independent scale, aspect
        and flip - not co-registered full frames, so layering carries the
        resampling and content mismatch of a real edit.
        """
        _, H, W = src.shape
        ch = max(8, min(H, int(round(ph * self.rng.uniform(0.5, 2.0)))))
        cw = max(8, min(W, int(round(pw * self.rng.uniform(0.5, 2.0)))))
        y0 = self.rng.randint(0, H - ch)
        x0 = self.rng.randint(0, W - cw)
        crop = src[:, y0:y0 + ch, x0:x0 + cw]
        if (ch, cw) != (ph, pw):
            crop = F.interpolate(crop[None], size=(ph, pw), mode="bilinear",
                                 align_corners=False, antialias=True)[0]
        if self.rng.random() < 0.5:
            crop = torch.flip(crop, dims=[2])
        return crop

    def _paste(self, img, src):
        """Layer one cropped region of `src` over `img`, in place.

        Returns the paste's grid-space footprint (patch-label bookkeeping).
        """
        y0, x0, h, w = self._rand_rect()
        mode = self._pick_mode()
        c = self.res // self.G
        alpha = self._region_alpha((y0, x0, h, w), mode)
        # paste window: region plus the alpha's edge bleed, clipped to bounds
        margin = 2 * c if mode == "blend" else 2
        py, px, ph, pw = y0 * c, x0 * c, h * c, w * c
        wy0, wx0 = max(0, py - margin), max(0, px - margin)
        wy1 = min(self.res, py + ph + margin)
        wx1 = min(self.res, px + pw + margin)
        crop = self._crop_for(src, wy1 - wy0, wx1 - wx0)
        a = alpha[wy0:wy1, wx0:wx1]
        img[:, wy0:wy1, wx0:wx1] = img[:, wy0:wy1, wx0:wx1] * (1.0 - a) + crop * a
        foot = torch.zeros(self.G, self.G)
        foot[y0:y0 + h, x0:x0 + w] = 1.0
        return foot

    def _apply_composites(self, images, labels, patch_labels):
        """Layer cropped overlays over base images (composite training).

        Compositing is itself a simple discontinuity, so every top-on-base
        class pairing is trained, not just fake-over-real:
          fake_on_real  localized patch labels (fake region over a real base)
          real_on_fake  inverted patch labels: only the pasted region is real
          fake_on_fake  seams inside fully-fake content (all patches stay 1)
          real_on_real  label stays 0: blending alone is not a fake cue
        Each composited sample receives a stack of 1..max_overlays pastes
        with independent regions, scales and modes; every paste overwrites
        the patch labels in its footprint (last layer wins), and the global
        label follows what is actually visible.
        """
        B = images.shape[0]
        comp = self.comp
        if comp is None:
            return images, labels, patch_labels
        pool = images.clone()  # pristine sources: composites never feed composites
        orig = labels.clone()  # pristine classes of the pool
        by_cls = {c: [j for j in range(B) if float(orig[j]) == c] for c in (0.0, 1.0)}

        def pick(cls):
            cands = by_cls[cls]
            return self.rng.choice(cands) if cands else None

        def stack(i, base, first_src):
            """`base` content + a stack of pastes of `first_src`'s class."""
            img = pool[base].clone()
            pl = torch.full((self.G, self.G), float(orig[base]))
            src = first_src
            for _ in range(self.rng.randint(1, max(1, int(comp.max_overlays)))):
                if src is None:
                    src = pick(float(orig[first_src]))
                    if src is None:
                        break
                pl[self._paste(img, pool[src]) > 0.5] = float(orig[src])
                src = None
            images[i] = img
            patch_labels[i] = pl.flatten()
            labels[i] = pl.max()  # any visible AI patch -> fake

        for i in range(B):
            if float(orig[i]) > 0.5 and self.rng.random() < comp.prob:
                combo = self._weighted_choice({
                    "fake_on_real": comp.fake_on_real if by_cls[0.0] else 0.0,
                    "real_on_fake": comp.real_on_fake if by_cls[0.0] else 0.0,
                    "fake_on_fake": comp.fake_on_fake,
                })
                if combo is None:
                    continue
                if combo == "fake_on_real":
                    stack(i, self.rng.choice(by_cls[0.0]), i)
                elif combo == "real_on_fake":
                    stack(i, i, self.rng.choice(by_cls[0.0]))
                else:  # fake_on_fake
                    stack(i, i, self.rng.choice(by_cls[1.0]))
            elif float(orig[i]) < 0.5 and \
                    self.rng.random() < comp.prob * comp.real_real_fraction:
                partners = [j for j in by_cls[0.0] if j != i]
                if partners:
                    stack(i, i, self.rng.choice(partners))

        return images, labels, patch_labels

    # -- collate -------------------------------------------------------------

    def __call__(self, samples: List[dict]) -> dict:
        # seeds are drawn on the caller thread (rng safety under the pool)
        seeds = [self.rng.randrange(1 << 30) for _ in samples]
        if self._pool is not None and len(samples) > 1:
            imgs = list(self._pool.map(lambda p: self._process_one(*p), zip(samples, seeds)))
        else:
            imgs = [self._process_one(s, seed) for s, seed in zip(samples, seeds)]

        labels = torch.tensor([float(s["label"]) for s in samples], dtype=torch.float32)
        images = torch.stack(imgs)
        P = self.G * self.G
        patch_labels = labels.view(-1, 1).expand(-1, P).clone().contiguous()

        if self.comp is not None and self.comp.prob > 0:
            images, labels, patch_labels = self._apply_composites(images, labels, patch_labels)

        return {"images": images, "labels": labels, "patch_labels": patch_labels}


class Prefetcher:
    """Moves batches to the device on a background thread so JPEG decode and
    H2D copy overlap with GPU compute. Accepts any iterable of batches
    (including an infinite batch generator - never terminates)."""

    def __init__(self, batch_iterable, device, depth: int = 2):
        import queue
        import threading

        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.device = device

        def worker():
            try:
                for batch in batch_iterable:
                    self.q.put(batch)
            finally:
                self.q.put(None)

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.q.get()
        if batch is None:
            raise StopIteration
        return {k: v.to(self.device, non_blocking=False) for k, v in batch.items()}
