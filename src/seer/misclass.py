"""Dump false-positive / false-negative images with dataset provenance."""

import json
import os
import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from .data import load_sample_image, sample_id

PROVENANCE_KEYS = (
    "source",
    "dataset",
    "source_type",
    "image_name",
    "image_path",
    "generator",
    "architecture",
    "prompt",
    "nsfw_flag",
    "subset",
    "real_source",
    "format",
    "distortions",
    "distortion_scales",
    "is_distorted",
    "content_id",
)


def provenance(sample: dict) -> dict:
    """JSON-safe metadata (no pixels / bytes)."""
    rec = {"id": sample_id(sample)}
    for key in PROVENANCE_KEYS:
        if key not in sample:
            continue
        value = sample[key]
        if value in (None, ""):
            continue
        if isinstance(value, tuple):
            value = list(value)
        rec[key] = value
    return rec


def pick_errors(
    samples: Sequence[dict],
    probs: Sequence[float],
    labels: Sequence[float],
    threshold: float = 0.5,
) -> Tuple[List[Tuple[dict, float]], List[Tuple[dict, float]]]:
    """FP (real predicted fake) and FN (fake predicted real), worst first."""
    fps, fns = [], []
    for sample, prob, label in zip(samples, probs, labels):
        pred = float(prob) >= threshold
        y = int(label)
        if y == 0 and pred:
            fps.append((sample, float(prob)))
        elif y == 1 and not pred:
            fns.append((sample, float(prob)))
    fps.sort(key=lambda t: -t[1])
    fns.sort(key=lambda t: t[1])
    return fps, fns


def _safe(text, n: int = 72) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return (out or "sample")[:n]


def _stem(sample: dict) -> str:
    path = sample.get("image_path") or sample.get("image_name") or sample.get("content_id")
    if path:
        return _safe(os.path.splitext(os.path.basename(str(path)))[0])
    return _safe(sample_id(sample).split("|")[-1])


def _save_image(sample: dict, dest: str) -> None:
    img = load_sample_image(sample).convert("RGB")
    img.save(dest, format="JPEG", quality=90)


def dump_error_lists(
    out_dir: str,
    fps: Iterable[Tuple[dict, float]],
    fns: Iterable[Tuple[dict, float]],
    *,
    step: int,
    split: str,
    max_per_kind: int = 64,
) -> dict:
    """Write JPEG copies + manifest.jsonl under ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    fps = list(fps)
    fns = list(fns)
    saved = {"fp": 0, "fn": 0}
    by_source = Counter()
    records = []

    for kind, rows in (("fp", fps), ("fn", fns)):
        dest_dir = os.path.join(out_dir, kind)
        os.makedirs(dest_dir, exist_ok=True)
        for i, (sample, prob) in enumerate(rows[: max(0, int(max_per_kind))]):
            src = _safe(sample.get("source") or sample.get("dataset") or "unk", 24)
            name = f"{i:03d}_{src}_{_stem(sample)}_p{prob:.3f}.jpg"
            rel = f"{kind}/{name}"
            path = os.path.join(out_dir, rel)
            rec = {
                "kind": kind,
                "step": int(step),
                "split": split,
                "label": 0 if kind == "fp" else 1,
                "prob": float(prob),
                "file": rel,
                **provenance(sample),
            }
            try:
                _save_image(sample, path)
                rec["saved"] = True
                saved[kind] += 1
            except Exception as exc:
                rec["saved"] = False
                rec["error"] = str(exc)
            records.append(rec)
            by_source[f"{sample.get('source') or '?'}|{kind}"] += 1

    manifest = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=str) + "\n")

    summary = {
        "step": int(step),
        "split": split,
        "n_fp": len(fps),
        "n_fn": len(fns),
        "saved_fp": saved["fp"],
        "saved_fn": saved["fn"],
        "max_per_kind": int(max_per_kind),
        "by_source": dict(by_source),
        "dir": out_dir,
        "manifest": manifest,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def dump_misclassified(
    out_dir: str,
    samples: Sequence[dict],
    probs: Sequence[float],
    labels: Sequence[float],
    *,
    step: int,
    split: str,
    max_per_kind: int = 64,
    threshold: float = 0.5,
) -> dict:
    fps, fns = pick_errors(samples, probs, labels, threshold=threshold)
    return dump_error_lists(
        out_dir, fps, fns, step=step, split=split, max_per_kind=max_per_kind
    )
