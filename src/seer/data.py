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
  {image: PIL.Image, label: 0|1, generator: str, architecture: str,
   source: str, dataset: str, source_type: str, ...source metadata}
"""

import dataclasses
import hashlib
import io
import os
import random
import re
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Set

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

# Image IDs reserved for the in-loop val slice. Training iterators skip them
# so FP/FN dumps are not scoring images the model just trained on.
_HOLDOUT_IDS: Set[str] = set()
_HOLD_SKIP_FALLBACK = 256

# Held-out Community Forensics Eval must never enter the training mixture.
_COMFOR_EVAL_MARKERS = (
    "communityforensics-eval",
    "comfor-eval",
    "comfor_eval",
)

# Organisers' demonstration val (not scored): COCO val2017 reals + WildFake
# DALL·E Advanced fakes. Same rule as CompEval — never train on these.
_DEMO_VAL_MARKERS = (
    "coco-val2017",
    "/val2017/",
    "wildfake-dalle",
    "wildfake/dalle",
    "hy2628982280/wildfake",
    "dall-e advanced",
    "dalle advanced",
    "dalle_advanced",
    "dalle-advanced",
    "dalle/advanced",
    "diffusion_based/dalle",
)

# OpenFake's own held-out splits: core/test is OOD generators paired with OOD
# reals, reddit/test is in-the-wild. scripts/openfake.py writes them under
# openfake/holdout_*, and only openfake/train may enter the mixture.
_OPENFAKE_EVAL_MARKERS = (
    "openfake/holdout",
    "openfake-holdout",
    "openfake/core/test",
    "openfake/reddit/test",
)

_HELD_OUT_TRAIN_MARKERS = (
    _COMFOR_EVAL_MARKERS + _DEMO_VAL_MARKERS + _OPENFAKE_EVAL_MARKERS
)


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


def _comfor_meta(row: dict) -> dict:
    return {
        "subset": row.get("subset") or "",
        "real_source": row.get("real_source") or "",
        "format": row.get("format") or "",
    }


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
        **_comfor_meta(row),
    }


def _bytes_id(raw: bytes) -> str:
    h = hashlib.sha1()
    h.update(raw[:4096])
    if len(raw) > 4096:
        h.update(raw[-4096:])
        h.update(str(len(raw)).encode())
    return h.hexdigest()[:16]


def _pil_id(img: Image.Image) -> str:
    tiny = img.resize((8, 8), Image.BILINEAR)
    return hashlib.sha1(tiny.tobytes()).hexdigest()[:16]


def _row_image_name(row: dict, image_col: str) -> str:
    for key in ("image_name", "filename", "file_name", "id", "image_id", "path"):
        if key == image_col:
            continue
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def dataset_id(spec) -> str:
    """Stable dataset name for provenance (HF repo, ntire split, or source)."""
    if getattr(spec, "dataset", ""):
        return spec.dataset
    if getattr(spec, "type", "") == "ntire":
        return f"ntire-{getattr(spec, 'split', None) or 'train'}"
    return getattr(spec, "name", "") or "src"


def attach_source(sample: dict, spec) -> dict:
    """Stamp mixture-source identity onto a sample (does not overwrite extras)."""
    sample["source"] = spec.name
    sample["source_type"] = spec.type
    sample["dataset"] = dataset_id(spec)
    return sample


def sample_id(sample: dict) -> str:
    """Stable identity used to hold val images out of the train mixture."""
    src = str(sample.get("source") or "")
    path = sample.get("image_path")
    if path:
        return f"{src}|path|{os.path.normpath(str(path))}"
    name = sample.get("image_name")
    if name:
        return f"{src}|name|{name}"
    digest = sample.get("content_id")
    if digest:
        return f"{src}|hash|{digest}"
    raw = sample.get("image_bytes")
    if raw:
        return f"{src}|hash|{_bytes_id(raw)}"
    img = sample.get("image")
    if isinstance(img, Image.Image):
        return f"{src}|hash|{_pil_id(img)}"
    return f"{src}|meta|{sample.get('generator', '')}|{sample.get('label')}"


def set_holdout(samples: List[dict]) -> Set[str]:
    """Register val-slice IDs so later training iterators skip those images."""
    global _HOLDOUT_IDS
    _HOLDOUT_IDS = {sample_id(s) for s in samples}
    return _HOLDOUT_IDS


def clear_holdout() -> None:
    global _HOLDOUT_IDS
    _HOLDOUT_IDS = set()


def is_held_out(sample: dict) -> bool:
    return bool(_HOLDOUT_IDS) and sample_id(sample) in _HOLDOUT_IDS


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
            print(
                f"[data] comfor: {len(self._local_files)} local parquet, "
                f"shuffle={self._shuffle_buffer}",
                flush=True,
            )
            ds = hfds.load_dataset("parquet", data_files=self._local_files, split="train", streaming=True)
        else:
            print(f"[data] comfor: streaming {self._dataset}/{self._split}", flush=True)
            ds = hfds.load_dataset(self._dataset, split=self._split, streaming=True)
        if self._shuffle_buffer > 0:
            print(f"[data] comfor: filling shuffle buffer ({self._shuffle_buffer})", flush=True)
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer)
        if self._max_samples:
            ds = ds.take(self._max_samples)
        n = 0
        first = True
        for row in ds:
            try:
                if self._lazy_decode:
                    sample = {
                        "image": None,
                        "image_bytes": bytes(row["image_data"]),
                        "label": int(row.get("label", 0)),
                        "generator": row.get("model_name") or "",
                        "architecture": row.get("architecture") or "",
                        "image_name": row.get("image_name") or "",
                        "prompt": (row.get("prompt") or "")[:512],
                        "nsfw_flag": _to_bool(row.get("nsfw_flag", False)),
                        **_comfor_meta(row),
                    }
                else:
                    sample = decode_row(row)
            except Exception:
                continue  # skip corrupt rows
            if first:
                print(f"[data] comfor: first row label={sample['label']}", flush=True)
                first = False
            yield sample
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
        name = self._name or self._dataset
        if self._local_files:
            print(
                f"[data] {name}: {len(self._local_files)} local parquet, "
                f"shuffle={self._shuffle_buffer}",
                flush=True,
            )
            ds = hfds.load_dataset("parquet", data_files=self._local_files, split="train", streaming=True)
        else:
            print(f"[data] {name}: streaming {self._dataset}/{self._split}", flush=True)
            ds = hfds.load_dataset(self._dataset, split=self._split, streaming=True)
        if self._shuffle_buffer > 0:
            print(f"[data] {name}: filling shuffle buffer ({self._shuffle_buffer})", flush=True)
            ds = ds.shuffle(seed=self._seed, buffer_size=self._shuffle_buffer)
        n = 0
        first = True
        for row in ds:
            try:
                lab = self._label_of(row)
                if lab is None:
                    continue
                if self._keep_label is not None and int(lab) != int(self._keep_label):
                    continue
                img = _as_pil(row.get(self._image_col))
                gen = str(row.get(self._generator_col)) if self._generator_col else self._name
                name = _row_image_name(row, self._image_col)
                sample = {"image": img, "label": int(lab), "generator": gen or self._name,
                          "architecture": "", "image_name": name}
                if not name:
                    sample["content_id"] = _pil_id(img)
            except Exception:
                continue
            if first:
                print(f"[data] {name}: first row label={sample['label']}", flush=True)
                first = False
            yield sample
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
    listing image paths (one per line).

    Held-out paths are dropped by default, which is what keeps a stray
    ``--fake-dir`` out of training. Eval sets that *live* at a held-out path
    (the OpenFake test splits) must opt in with ``allow_held_out=True``.
    """

    def __init__(self, roots: List[str], label: int, allow_held_out: bool = False):
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
        if not allow_held_out:
            self.files = [f for f in self.files if not is_held_out_train_ref(f)]
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
        return [f for f in files if not is_held_out_train_ref(f)]

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
        cycle: bool = True,
    ):
        from .ntire import load_split, split_is_cached

        cached = split_is_cached(split, shard, hard)
        if not cached:
            print(f"[data] ntire loading {split} shard={shard}...", flush=True)
        samples = load_split(split, shard=shard, hard=hard)
        if clean_only:
            samples = [s for s in samples if not s.is_distorted]
        self.samples = samples
        self.max_samples = max_samples
        self.seed = seed
        self.cycle = cycle
        tag = "cached" if cached else "indexed"
        print(
            f"[data] ntire {split} shard={shard}: {len(samples)} labelled images ({tag})",
            flush=True,
        )

    def __iter__(self):
        if not self.samples:
            return
        epoch = 0
        while True:
            items = list(self.samples)
            rng = random.Random(self.seed + epoch * 1009)
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
                    "distortions": s.distortions,
                    "distortion_scales": s.distortion_scales,
                    "is_distorted": bool(s.is_distorted),
                }
            if not self.cycle:
                return
            epoch += 1


def _norm_ref(value: str) -> str:
    return str(value or "").lower().replace("\\", "/")


def is_held_out_train_ref(*parts: object) -> bool:
    """True if a path / dataset / listing points at a never-train set."""
    blob = " ".join(_norm_ref(p) for p in parts if p not in (None, ""))
    return any(m in blob for m in _HELD_OUT_TRAIN_MARKERS)


def _held_out_train_reason(blob: str) -> str:
    if any(m in blob for m in _COMFOR_EVAL_MARKERS):
        return (
            "Community Forensics Eval is held-out. Do not train on "
            "OwensLab/CommunityForensics-Eval or local_dirs under comfor-eval. "
            "Use CommunityForensics-Small for training and --dataset comfor_eval for eval."
        )
    if any(m in blob for m in _OPENFAKE_EVAL_MARKERS):
        return (
            "The OpenFake test splits are held-out: core/test is unseen "
            "generators paired with unseen real sources, reddit/test is "
            "in-the-wild. Train on openfake/train (scripts/openfake.py fetch) "
            "and evaluate with --dataset openfake_test / openfake_reddit."
        )
    return (
        "The organisers' demonstration val is held-out (COCO val2017 reals, "
        "WildFake DALL·E Advanced fakes). Do not train on coco-val2017 or "
        "wildfake-dalle / DALL·E Advanced."
    )


def assert_not_held_out_train(*parts: object, spec=None) -> None:
    """Raise if a training source points at CompEval or the demo val pair."""
    extra = []
    if spec is not None:
        extra = [
            getattr(spec, "name", ""),
            getattr(spec, "dataset", ""),
            *(getattr(spec, "local_dirs", None) or []),
            *(getattr(spec, "real_dirs", None) or []),
            *(getattr(spec, "fake_dirs", None) or []),
        ]
    blob = " ".join(_norm_ref(p) for p in (*parts, *extra) if p not in (None, ""))
    if is_held_out_train_ref(blob):
        raise ValueError(_held_out_train_reason(blob))


def assert_not_comfor_eval_train(dataset: str = "", local_dirs: Optional[List[str]] = None) -> None:
    """Raise if Community Forensics Eval is being used as training data."""
    assert_not_held_out_train(dataset, *(local_dirs or []))


def _raw_source_iterator(spec, seed: int, *, cycle: bool = True) -> Iterator[dict]:
    """Build a (re-creatable) iterator for one source spec, untagged."""
    if spec.type == "ntire":
        return iter(NtireStream(
            split=spec.split,
            shard=spec.shard,
            hard=spec.hard,
            max_samples=spec.max_samples,
            seed=seed,
            clean_only=spec.clean_only,
            cycle=cycle,
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


def _source_iterator(
    spec, seed: int, *, skip_holdout: bool = True, cycle: bool = True
) -> Iterator[dict]:
    """Tagged iterator; training skips images reserved for the val slice."""
    consecutive = 0
    warned = False
    for sample in _raw_source_iterator(spec, seed, cycle=cycle):
        attach_source(sample, spec)
        if is_held_out_train_ref(sample.get("image_path"), sample.get("image_name")):
            continue
        if skip_holdout and _HOLDOUT_IDS and sample_id(sample) in _HOLDOUT_IDS:
            consecutive += 1
            if consecutive < _HOLD_SKIP_FALLBACK:
                continue
            if not warned:
                print(
                    f"[data] {spec.name}: holdout covers this source; "
                    "allowing reuse so training can proceed",
                    flush=True,
                )
                warned = True
        else:
            consecutive = 0
        yield sample


def source_classes(spec) -> List[int]:
    """Which labels a source can emit (0=real, 1=fake)."""
    if spec.type == "folders":
        out: List[int] = []
        if spec.real_dirs:
            out.append(0)
        if spec.fake_dirs:
            out.append(1)
        return out or [0, 1]
    if spec.keep_label is not None:
        return [int(spec.keep_label)]
    if spec.label is not None and spec.label_col is None:
        return [int(spec.label)]
    return [0, 1]


class MixtureDataset(torch.utils.data.IterableDataset):
    """Weighted mixture over sources, cycled indefinitely.

    Each sample is drawn from source i with probability weight_i / sum(w).
    Sources that exhaust (folder passes, capped streams) are transparently
    restarted. Note: HF *streaming* sources re-download on restart - for big
    repeated passes, materialize them to folders with scripts/download_data.py.
    """

    def __init__(self, sources, seed: int = 0, balance_labels: bool = False):
        super().__init__()
        if not sources:
            raise ValueError("MixtureDataset needs at least one source")
        for spec in sources:
            assert_not_held_out_train(spec=spec)
        self.sources = list(sources)
        self.weights = [max(0.0, float(s.weight)) for s in self.sources]
        if sum(self.weights) <= 0:
            raise ValueError("All mixture weights are zero")
        self.seed = seed
        self.balance_labels = bool(balance_labels)
        self._classes = [source_classes(s) for s in self.sources]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = self.seed if info is None else self.seed + info.id * 1009
        rng = random.Random(seed)
        iters = [None] * len(self.sources)
        dead = [False] * len(self.sources)
        seen = [False] * len(self.sources)

        def next_from(i):
            if iters[i] is None:
                print(f"[data] opening source {self.sources[i].name}", flush=True)
                iters[i] = _source_iterator(self.sources[i], seed=rng.randrange(1 << 30))
            try:
                return next(iters[i])
            except StopIteration:
                iters[i] = _source_iterator(self.sources[i], seed=rng.randrange(1 << 30))
                return next(iters[i])

        def pick(live, want=None):
            if want is None:
                cands = live
            else:
                cands = [i for i in live if want in self._classes[i]]
                if not cands:
                    cands = live
            w = [self.weights[j] for j in cands]
            return cands[rng.choices(range(len(cands)), weights=w, k=1)[0]]

        while True:
            live = [i for i, is_dead in enumerate(dead) if not is_dead]
            if not live:
                return
            want = rng.choice([0, 1]) if self.balance_labels else None
            i = pick(live, want)
            try:
                sample = next_from(i)
                if want is not None and int(sample.get("label", -1)) != want:
                    for _try in range(16):
                        i = pick(live, want)
                        sample = next_from(i)
                        if int(sample.get("label", -1)) == want:
                            break
                if not seen[i]:
                    seen[i] = True
                    print(
                        f"[data] first sample from {self.sources[i].name} "
                        f"label={sample.get('label')}",
                        flush=True,
                    )
                yield sample
            except FileNotFoundError:
                dead[i] = True
                print(f"[data] dropping {self.sources[i].name}: not found", flush=True)
            except StopIteration:
                dead[i] = True
                print(f"[data] dropping {self.sources[i].name}: empty", flush=True)


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
        return MixtureDataset(
            d.sources, seed=cfg.seed, balance_labels=d.balance_labels
        )
    if d.source in ("mixture", "comfor"):
        assert_not_held_out_train(d.dataset, *(d.local_dirs or []))
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
        assert_not_held_out_train(*(d.real_dirs or []), *(d.fake_dirs or []))
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


def _val_spec(spec):
    """Cheap shuffle for the held-out draw; we stop ourselves (no max_samples)."""
    return dataclasses.replace(
        spec,
        max_samples=None,
        shuffle_buffer=min(int(getattr(spec, "shuffle_buffer", 0) or 0), 256),
    )


def _take_source_val(spec, n: int, seed: int) -> List[dict]:
    """Up to `n` samples from one source, class-balanced when it can emit both."""
    if n <= 0:
        return []
    classes = source_classes(spec)
    it = _source_iterator(_val_spec(spec), seed, skip_holdout=False, cycle=False)
    limit = max(n * 32, 256)
    try:
        if len(classes) >= 2:
            want = {0: (n + 1) // 2, 1: n // 2}
            buckets = {0: [], 1: []}
            scanned = 0
            for sample in it:
                scanned += 1
                y = int(sample.get("label", -1))
                if y in want and len(buckets[y]) < want[y]:
                    buckets[y].append(sample)
                if len(buckets[0]) >= want[0] and len(buckets[1]) >= want[1]:
                    break
                if scanned >= limit:
                    break
            return buckets[0] + buckets[1]
        out: List[dict] = []
        for sample in it:
            out.append(sample)
            if len(out) >= n:
                break
        return out
    except FileNotFoundError:
        print(f"[data] val: skipping {spec.name}: not found", flush=True)
        return []
    except StopIteration:
        return []
    except Exception as exc:
        print(f"[data] val: skipping {spec.name}: {exc}", flush=True)
        return []


def collect_held_out_val(cfg) -> List[dict]:
    """Class-balanced slice from every training source for in-loop eval.

    These images are later registered with ``set_holdout`` so the train
    mixture does not see them.
    """
    from .config import SourceSpec

    d = cfg.data
    if d.source == "mixture" and d.sources:
        specs = list(d.sources)
        n = max(1, len(specs))
        quota = max(1, int(d.val_max_samples) // n)
        leftover = max(0, int(d.val_max_samples) - quota * n)
        out: List[dict] = []
        for i, spec in enumerate(specs):
            want = quota + (1 if i < leftover else 0)
            chunk = _take_source_val(spec, want, seed=int(d.val_seed) + i * 17)
            print(
                f"[data] val: {spec.name} held out {len(chunk)}/{want}",
                flush=True,
            )
            out.extend(chunk)
        return out
    if d.source == "comfor":
        spec = SourceSpec(
            name="comfor",
            type="comfor",
            dataset=d.dataset,
            split=d.split,
            shuffle_buffer=min(d.shuffle_buffer, 256),
            local_dirs=list(d.local_dirs or []),
        )
        return _take_source_val(spec, int(d.val_max_samples), int(d.val_seed))
    if d.source == "folders":
        spec = SourceSpec(
            name="folders",
            type="folders",
            real_dirs=list(d.real_dirs or []),
            fake_dirs=list(d.fake_dirs or []),
        )
        return _take_source_val(spec, int(d.val_max_samples), int(d.val_seed))
    if d.source == "ntire":
        spec = SourceSpec(
            name="ntire",
            type="ntire",
            split=d.split,
        )
        return _take_source_val(spec, int(d.val_max_samples), int(d.val_seed))
    return []


def build_val_dataset(cfg) -> torch.utils.data.Dataset:
    """Held-out slice covering every mixture source (see collect_held_out_val)."""
    samples = collect_held_out_val(cfg)
    sources = list(cfg.data.sources) if cfg.data.source == "mixture" and cfg.data.sources else []

    class _HeldOut(torch.utils.data.IterableDataset):
        def __init__(self, items, specs):
            super().__init__()
            self._items = items
            self.sources = specs

        def __iter__(self):
            yield from self._items

        def __len__(self):
            return len(self._items)

    return _HeldOut(samples, sources)


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
        self._decode_workers = int(decode_workers or 0)
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None and self._decode_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=self._decode_workers)
        return self._pool

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_pool"] = None
        return state

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

    def _crop_geom(self, H, W, ph, pw):
        """Crop box + flip for an overlay, shared by pixels and label maps."""
        ch = max(8, min(H, int(round(ph * self.rng.uniform(0.5, 2.0)))))
        cw = max(8, min(W, int(round(pw * self.rng.uniform(0.5, 2.0)))))
        y0 = self.rng.randint(0, H - ch)
        x0 = self.rng.randint(0, W - cw)
        return y0, x0, ch, cw, self.rng.random() < 0.5

    def _resample_crop(self, src, y0, x0, ch, cw, flip, ph, pw):
        """Apply one overlay crop. `src` is (C, H, W) — RGB or a 1-channel label."""
        crop = src[..., y0:y0 + ch, x0:x0 + cw]
        if (ch, cw) != (ph, pw):
            crop = F.interpolate(
                crop[None], size=(ph, pw), mode="bilinear",
                align_corners=False, antialias=True,
            )[0]
        if flip:
            crop = torch.flip(crop, dims=[-1])
        return crop

    def _paste(self, img, src_img, lab, src_lab):
        """Layer a crop of `src_img` / `src_lab` over `img` / `lab` in place.

        Labels travel with the same crop, scale and flip as the pixels, then
        Porter-Duff over: both the RGB and the per-pixel fake-ness.
        """
        y0, x0, h, w = self._rand_rect()
        mode = self._pick_mode()
        c = self.res // self.G
        alpha = self._region_alpha((y0, x0, h, w), mode)
        margin = 2 * c if mode == "blend" else 2
        py, px, ph, pw = y0 * c, x0 * c, h * c, w * c
        wy0, wx0 = max(0, py - margin), max(0, px - margin)
        wy1 = min(self.res, py + ph + margin)
        wx1 = min(self.res, px + pw + margin)
        _, H, W = src_img.shape
        geom = self._crop_geom(H, W, wy1 - wy0, wx1 - wx0)
        crop = self._resample_crop(src_img, *geom, wy1 - wy0, wx1 - wx0)
        lab_crop = self._resample_crop(src_lab, *geom, wy1 - wy0, wx1 - wx0)
        a = alpha[wy0:wy1, wx0:wx1]
        img[:, wy0:wy1, wx0:wx1] = img[:, wy0:wy1, wx0:wx1] * (1.0 - a) + crop * a
        lab[:, wy0:wy1, wx0:wx1] = lab[:, wy0:wy1, wx0:wx1] * (1.0 - a) + lab_crop * a

    def _apply_composites(self, images, labels, patch_labels):
        """Layer cropped overlays over base images (composite training).

        Compositing is itself a simple discontinuity, so every top-on-base
        class pairing is trained, not just fake-over-real:
          fake_on_real  localized patch labels (fake region over a real base)
          real_on_fake  inverted patch labels: only the pasted region is real
          fake_on_fake  seams inside fully-fake content (all patches stay 1)
          real_on_real  label stays 0: blending alone is not a fake cue
        Each composited sample gets n ~ Uniform{1,...,k} pastes
        (k = max_overlays) with independent regions, scales and modes.

        Per-pixel fake-ness is alpha-composited with the RGB (a 40% blend
        is a 0.4 target), then average-pooled to the patch grid. The page
        target stays binary: any visible AI → fake. Later overlays may
        read already-composited slots; their label maps travel with them.
        """
        B = images.shape[0]
        comp = self.comp
        if comp is None:
            return images, labels, patch_labels
        orig = labels.clone()
        by_cls = {c: [j for j in range(B) if float(orig[j]) == c] for c in (0.0, 1.0)}
        H, W = int(images.shape[-2]), int(images.shape[-1])
        pixel_lab = orig.view(B, 1, 1, 1).expand(B, 1, H, W).clone()
        cell = self.res // self.G

        def pick(cls):
            cands = by_cls[cls]
            return self.rng.choice(cands) if cands else None

        def stack(i, base, first_src):
            """`base` (current pixels/labels) + n pastes; first from `first_src`."""
            src_img = images[first_src].clone()
            src_lab = pixel_lab[first_src].clone()
            if i != base:
                images[i] = images[base].clone()
                pixel_lab[i] = pixel_lab[base].clone()
            k = max(1, int(comp.max_overlays))
            for n in range(self.rng.randint(1, k)):
                if n > 0:
                    src = pick(float(orig[first_src]))
                    if src is None:
                        break
                    src_img, src_lab = images[src], pixel_lab[src]
                self._paste(images[i], src_img, pixel_lab[i], src_lab)
            pl = F.avg_pool2d(pixel_lab[i][None], cell, stride=cell)[0, 0].clamp(0.0, 1.0)
            patch_labels[i] = pl.flatten()
            labels[i] = 1.0 if float(pl.max()) > 0.0 else 0.0

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
        pool = self._ensure_pool()
        if pool is not None and len(samples) > 1:
            imgs = list(pool.map(lambda p: self._process_one(*p), zip(samples, seeds)))
        else:
            imgs = [self._process_one(s, seed) for s, seed in zip(samples, seeds)]

        labels = torch.tensor([float(s["label"]) for s in samples], dtype=torch.float32)
        images = torch.stack(imgs)
        P = self.G * self.G
        patch_labels = labels.view(-1, 1).expand(-1, P).clone().contiguous()

        if self.comp is not None and self.comp.prob > 0:
            images, labels, patch_labels = self._apply_composites(images, labels, patch_labels)

        return {"images": images, "labels": labels, "patch_labels": patch_labels}


class ThreadedSampleQueue(torch.utils.data.IterableDataset):
    """Several independent mixture iterators feeding one queue.

    Parquet and folder I/O release the GIL, so extra reader threads raise
    sample yield rate and keep the decode pool from starving.
    """

    def __init__(self, make_iter, n_readers: int = 4, queue_size: int = 512):
        super().__init__()
        self._make_iter = make_iter
        self._n = max(1, int(n_readers))
        self._queue_size = int(queue_size)

    def __iter__(self):
        if self._n == 1:
            yield from self._make_iter(0)
            return
        import queue
        import threading

        q: "queue.Queue" = queue.Queue(maxsize=self._queue_size)
        sentinel = object()

        def run(idx):
            try:
                for sample in self._make_iter(idx):
                    q.put(sample)
            finally:
                q.put(sentinel)

        for i in range(self._n):
            threading.Thread(target=run, args=(i,), daemon=True).start()
        live = self._n
        while live:
            item = q.get()
            if item is sentinel:
                live -= 1
                continue
            yield item


class Prefetcher:
    """Decode/collate on a background thread, pin + H2D there too, so the
    training loop only waits if the CPU pipeline is behind."""

    def __init__(self, batch_iterable, device, depth: int = 4):
        import queue
        import threading

        self.q: "queue.Queue" = queue.Queue(maxsize=max(2, int(depth)))
        self.device = device

        def worker():
            stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
            try:
                for batch in batch_iterable:
                    if stream is not None:
                        pinned = {
                            k: v.pin_memory() if torch.is_tensor(v) else v
                            for k, v in batch.items()
                        }
                        with torch.cuda.stream(stream):
                            batch = {
                                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                                for k, v in pinned.items()
                            }
                        stream.synchronize()
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
        return batch
