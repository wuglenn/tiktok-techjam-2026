"""Benchmark harness.

Replicates the evaluation protocol of the Pangram Image technical blog:

  * CommunityForensics-Eval ("CompEval") - the comprehensive evaluation set of
    CVPR 2025 "Community Forensics" - macro accuracy + mAP
  * "Augmented" protocol - downscale to 1024x1024 + JPEG q50, both classes
  * Robustness sweeps - the benchmark perturbation protocol (JPEG 90/70/50/30,
    blur sigma 0.5/1/2, resize 0.5x/0.25x, noise 0.02/0.05/0.10, jitter +/-20%,
    center crop 80%) via `--perturbation all`
  * False-positive-rate evals on real-only data (WikiArt etc. via folders)
  * Error analysis - the most confident false positives / false negatives are
    written out as heatmap panels (`--error-dir`), so failures can be read
    rather than guessed at

Metrics: macro (balanced) accuracy, mAP (average precision on the fake class),
AUROC, F1, precision, recall, FPR, FNR. Published reference numbers from the
Pangram blog are printed next to ours for a direct comparison.
"""

import heapq
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .augment import PERTURBATIONS, apply_perturbation, eval_transform, perturbation_names
from .data import ComforStream, FolderDataset, NtireStream, load_sample_image
from .heatmap import patch_logits_to_heat, save_heatmap
from .model import load_checkpoint

# ---------------------------------------------------------------------------
# Published numbers (Pangram Image blog, Jul 2026) for context in reports.
# (macro accuracy %, mAP %) on CommunityForensics-Eval / Synthbuster+RAISE.
# ---------------------------------------------------------------------------
PUBLISHED = {
    "CommunityForensics-Eval": [
        ("Pangram Image (2026)", 97.29, 99.70),
        ("Ours-384 (CVPR 2025)", 89.3, 98.7),
    ],
    "Synthbuster+RAISE-1K": [
        ("Pangram Image (2026)", 98.49, 99.96),
        ("B-Free (ICML 2024)", 94.9, 98.8),
    ],
}

EVAL_SPECS = {
    "comfor_eval": dict(dataset="OwensLab/CommunityForensics-Eval", split="CompEval"),
    "comfor_small": dict(dataset="OwensLab/CommunityForensics-Small", split="train"),
}

NTIRE_EVAL = {
    "ntire_val": dict(split="val"),
    "ntire_val_hard": dict(split="val_hard"),
    "ntire_test": dict(split="test"),
    "ntire_test_public": dict(split="test"),  # alias for the HF public test
}

_NTIRE_SAMPLE_CACHE: dict = {}


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs >= threshold
    pos, neg = labels == 1, labels == 0
    tp = float((pred[pos]).sum())
    fn = float((~pred[pos]).sum())
    tn = float((~pred[neg]).sum())
    fp = float((pred[neg]).sum())
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    out = {
        "n": int(labels.size),
        "n_fake": int(pos.sum()),
        "n_real": int(neg.sum()),
        "accuracy": (tp + tn) / max(1, labels.size),
        "macro_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "fpr": fp / max(1.0, fp + tn),
        "fnr": fn / max(1.0, fn + tp),
    }
    if 0 < pos.sum() < labels.size:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["auroc"] = float(roc_auc_score(labels, probs))
        out["mAP"] = float(average_precision_score(labels, probs))
    else:
        out["auroc"] = float("nan")
        out["mAP"] = float("nan")
    return out


def _group_metrics(probs, labels, keys) -> dict:
    buckets = defaultdict(lambda: ([], []))
    for p, y, k in zip(probs, labels, keys):
        buckets[str(k)][0].append(p)
        buckets[str(k)][1].append(y)
    return {
        k: compute_metrics(np.array(ps), np.array(ys))
        for k, (ps, ys) in sorted(buckets.items())
    }


def _distortion_key(sample: dict) -> str:
    d = sample.get("distortions") or ()
    if not d:
        return "none"
    return str(d[0])


class ErrorBank:
    """The most confident mistakes of an eval pass, kept for explanation.

    False positives are ranked by how confidently a real image was called AI,
    false negatives by how confidently an AI image was called real - the two
    ends of the score range are where a detector's blind spots are legible.
    Each kept sample carries the patch logits the model produced for it, so
    the dump shows *where* the model was looking, not just that it was wrong.
    """

    KINDS = {"fp": "false positive (real called AI)",
             "fn": "false negative (AI called real)"}

    def __init__(self, k: int = 4, res: int = 512, threshold: float = 0.5):
        self.k = max(0, int(k))
        self.res = int(res)
        self.threshold = float(threshold)
        self._heaps = {"fp": [], "fn": []}
        self._tie = 0

    def add(self, image, prob: float, label: int, patch_logits=None, meta=None):
        """Offer one scored sample; kept only if it beats the current worst."""
        if self.k == 0:
            return
        if label == 0 and prob >= self.threshold:
            kind, score = "fp", prob
        elif label == 1 and prob < self.threshold:
            kind, score = "fn", 1.0 - prob
        else:
            return
        heap = self._heaps[kind]
        if len(heap) >= self.k and score <= heap[0][0]:
            return  # already holding k more confident errors
        self._tie += 1
        record = {
            "prob": float(prob),
            "label": int(label),
            "image": image.convert("RGB").resize((self.res, self.res)),
            "patch_logits": None if patch_logits is None else patch_logits.clone(),
            "meta": dict(meta or {}),
        }
        heapq.heappush(heap, (score, self._tie, record))
        if len(heap) > self.k:
            heapq.heappop(heap)

    def counts(self) -> dict:
        return {k: len(v) for k, v in self._heaps.items()}

    def dump(self, out_dir: str) -> list:
        """Render every kept error as a PNG panel; returns a JSON manifest."""
        entries = []
        if not any(self._heaps.values()):
            return entries
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for kind, heap in self._heaps.items():
            ranked = sorted(heap, key=lambda item: item[0], reverse=True)
            for rank, (_score, _tie, rec) in enumerate(ranked, 1):
                meta = rec["meta"]
                stem = Path(str(meta.get("image_name") or "sample")).stem[:40] or "sample"
                path = out / f"{kind}_{rank:02d}_p{rec['prob']:.3f}_{stem}.png"
                source = meta.get("generator") or meta.get("architecture") or ""
                title = f"{self.KINDS[kind]}\n{source}" if source else self.KINDS[kind]
                if rec["patch_logits"] is None:
                    rec["image"].save(path)  # page-only checkpoint: no heatmap
                else:
                    heat = patch_logits_to_heat(
                        rec["patch_logits"], (self.res, self.res)
                    )[0].numpy()
                    save_heatmap(str(path), rec["image"], heat, rec["prob"],
                                 self.res, title=title)
                entries.append({
                    "kind": kind,
                    "rank": rank,
                    "file": str(path),
                    "prob_ai": rec["prob"],
                    "label": rec["label"],
                    "explained": rec["patch_logits"] is not None,
                    **{k: v for k, v in meta.items() if v not in (None, "", ())},
                })
        return entries


def _error_meta(sample: dict) -> dict:
    return {
        "image_name": sample.get("image_name") or "",
        "image_path": sample.get("image_path") or "",
        "source": sample.get("source") or "",
        "dataset": sample.get("dataset") or "",
        "generator": sample.get("generator") or "",
        "architecture": sample.get("architecture") or "",
        "distortions": list(sample.get("distortions") or ()),
    }


def _chunked(ds, batch_size: int, max_samples: Optional[int] = None):
    """Stream a dataset in fixed-size chunks (never materialize it all)."""
    chunk: List[dict] = []
    seen = 0
    for s in ds:
        chunk.append(s)
        seen += 1
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
        if max_samples and seen >= max_samples:
            break
    if chunk:
        yield chunk


def known_eval_datasets() -> list:
    return list(EVAL_SPECS) + list(NTIRE_EVAL) + ["folders"]


def _build_eval_dataset(dataset, real_dirs=None, fake_dirs=None):
    if dataset == "folders":
        parts = []
        if real_dirs:
            parts.append(FolderDataset(real_dirs, 0))
        if fake_dirs:
            parts.append(FolderDataset(fake_dirs, 1))
        if not parts:
            raise ValueError("folders eval needs --real-dir and/or --fake-dir")
        return torch.utils.data.ConcatDataset(parts)
    if dataset in NTIRE_EVAL:
        hit = _NTIRE_SAMPLE_CACHE.get(dataset)
        if hit is not None:
            return hit
        stream = NtireStream(split=NTIRE_EVAL[dataset]["split"], seed=0, cycle=False)
        samples = list(stream)
        for sample in samples:
            sample.setdefault("source", dataset)
            sample.setdefault("source_type", "ntire")
            sample.setdefault("dataset", f"ntire-{NTIRE_EVAL[dataset]['split']}")
        _NTIRE_SAMPLE_CACHE[dataset] = samples
        print(f"[data] cached {len(samples)} {dataset} samples for later evals", flush=True)
        return samples
    spec = EVAL_SPECS.get(dataset)
    if spec is None:
        raise ValueError(f"Unknown eval dataset '{dataset}' (try {known_eval_datasets()})")
    return ComforStream(dataset=spec["dataset"], split=spec["split"],
                         shuffle_buffer=1024, max_samples=None, seed=0)


def _tag_eval_sample(sample: dict, dataset: str) -> dict:
    sample.setdefault("source", dataset)
    sample.setdefault("dataset", dataset)
    if dataset in NTIRE_EVAL:
        sample.setdefault("source_type", "ntire")
    elif str(dataset).startswith("comfor"):
        sample.setdefault("source_type", "comfor")
    elif dataset == "folders":
        sample.setdefault("source_type", "folders")
    return sample


@torch.no_grad()
def _single_pass(model, cfg_dict, perturbation: Optional[str], augmented: bool,
                 dataset: str, batch_size: int, max_samples: Optional[int],
                 hflip_tta: bool, res: int, device,
                 dump_dir: Optional[str] = None, step: int = 0,
                 misclass_max: int = 0,
                 real_dirs: Optional[List[str]] = None,
                 fake_dirs: Optional[List[str]] = None,
                 error_bank: Optional[ErrorBank] = None) -> dict:
    ds = _build_eval_dataset(dataset, real_dirs=real_dirs, fake_dirs=fake_dirs)

    pert_name = perturbation or ("pangram" if augmented else "clean")
    perturb_fn = (lambda im: apply_perturbation(im, pert_name)) if pert_name != "clean" else None

    probs, labels, archs, distorted, dist_keys = [], [], [], [], []
    errors_fp, errors_fn = [], []
    for chunk in tqdm(_chunked(ds, batch_size, max_samples),
                      desc=f"eval[{pert_name}]", unit="img", disable=None):
        imgs = [load_sample_image(s) for s in chunk]
        if perturb_fn is not None:
            imgs = [perturb_fn(im) for im in imgs]
        x = torch.stack([eval_transform(im, res) for im in imgs]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(x)
            p = torch.sigmoid(out["logits"]).float()
            if hflip_tta:
                out_f = model(torch.flip(x, dims=[-1]))
                p = 0.5 * (p + torch.sigmoid(out_f["logits"]).float())
        if error_bank is not None:
            # the images the model actually saw (post-perturbation) are what
            # an error dump has to show
            patch = out.get("patch_logits")
            patch = None if patch is None else patch.detach().float().cpu()
            for i, s in enumerate(chunk):
                error_bank.add(
                    imgs[i], float(p[i]), int(s["label"]),
                    patch_logits=None if patch is None else patch[i],
                    meta=_error_meta(s),
                )
        p_list = p.cpu().tolist()
        y_list = [int(s["label"]) for s in chunk]
        probs.extend(p_list)
        labels.extend(y_list)
        archs.extend(s.get("architecture", "") for s in chunk)
        distorted.extend("distorted" if s.get("is_distorted") else "clean" for s in chunk)
        dist_keys.extend(_distortion_key(s) for s in chunk)
        if dump_dir and misclass_max:
            for sample, prob, y in zip(chunk, p_list, y_list):
                _tag_eval_sample(sample, dataset)
                if y == 0 and prob >= 0.5:
                    errors_fp.append((sample, float(prob)))
                elif y == 1 and prob < 0.5:
                    errors_fn.append((sample, float(prob)))

    probs = np.array(probs)
    labels = np.array(labels)
    metrics = compute_metrics(probs, labels)
    metrics["dataset"] = dataset
    metrics["perturbation"] = pert_name
    metrics["hflip_tta"] = hflip_tta
    metrics["per_architecture"] = _group_metrics(probs, labels, archs)
    if any(s != "clean" for s in distorted):
        metrics["per_distorted"] = _group_metrics(probs, labels, distorted)
        distorted_m = metrics["per_distorted"].get("distorted") or {}
        # Robust AUROC = ROC AUC on the distorted subset only (NTIRE protocol).
        metrics["robust_auroc"] = distorted_m.get("auroc", float("nan"))
        metrics["robust_mAP"] = distorted_m.get("mAP", float("nan"))
        metrics["robust_n"] = int(distorted_m.get("n") or 0)
        metrics["per_distortion"] = {
            k: v for k, v in _group_metrics(probs, labels, dist_keys).items()
            if v["n"] >= 20
        }
    if dump_dir and misclass_max:
        from .misclass import dump_error_lists

        errors_fp.sort(key=lambda t: -t[1])
        errors_fn.sort(key=lambda t: t[1])
        metrics["misclassified"] = dump_error_lists(
            dump_dir, errors_fp, errors_fn,
            step=step, split=dataset, max_per_kind=misclass_max,
        )
    return metrics


@torch.no_grad()
def eval_named_dataset(
    model,
    dataset: str,
    *,
    res: int,
    device,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    dump_dir: Optional[str] = None,
    step: int = 0,
    misclass_max: int = 0,
) -> dict:
    """Score one named eval set (used from the training loop and CLI)."""
    return _single_pass(
        model=model,
        cfg_dict={},
        perturbation=None,
        augmented=False,
        dataset=dataset,
        batch_size=batch_size,
        max_samples=max_samples,
        hflip_tta=False,
        res=res,
        device=device,
        dump_dir=dump_dir,
        step=step,
        misclass_max=misclass_max,
    )


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str,
    dataset: str = "comfor_eval",
    augmented: bool = False,
    perturbation: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    hflip_tta: bool = False,
    real_dirs: Optional[List[str]] = None,
    fake_dirs: Optional[List[str]] = None,
    out_json: Optional[str] = None,
    device: Optional[str] = None,
    res: Optional[int] = None,
    error_dir: Optional[str] = None,
    error_n: int = 4,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg_dict, ckpt = load_checkpoint(checkpoint, device=device)
    res = res or int(cfg_dict.get("res", 512))
    model.eval()

    def pass_kwargs():
        return dict(model=model, cfg_dict=cfg_dict, dataset=dataset,
                    batch_size=batch_size, max_samples=max_samples,
                    hflip_tta=hflip_tta, res=res, device=device,
                    real_dirs=real_dirs, fake_dirs=fake_dirs)

    def new_bank():
        return ErrorBank(error_n, res=res) if error_dir and error_n > 0 else None

    sweep = perturbation_names(perturbation)
    if sweep:
        results = {}
        errors = []
        for name in sweep:
            bank = new_bank()
            m = _single_pass(perturbation=name, augmented=False,
                             error_bank=bank, **pass_kwargs())
            results[name] = {k: m[k] for k in
                             ("macro_accuracy", "mAP", "auroc", "f1", "precision",
                              "recall", "accuracy", "fpr", "fnr", "n")}
            if bank is not None:
                errors.extend(_dump_errors(bank, os.path.join(error_dir, name), name))
        metrics = dict(results.get("clean") or next(iter(results.values())))
        metrics["perturbation_sweep"] = results
        if errors:
            metrics["error_analysis"] = errors
        _print_robustness_table(results)
    else:
        if perturbation is not None and perturbation not in PERTURBATIONS:
            raise ValueError(
                f"Unknown perturbation '{perturbation}'. "
                f"Available: {list(PERTURBATIONS)} or all / extra / all+extra"
            )
        bank = new_bank()
        metrics = _single_pass(perturbation=perturbation, augmented=augmented,
                               error_bank=bank, **pass_kwargs())
        if bank is not None:
            errors = _dump_errors(bank, error_dir, metrics["perturbation"])
            if errors:
                metrics["error_analysis"] = errors

    metrics["checkpoint"] = checkpoint
    _print_report(metrics, dataset)
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[seer] metrics written to {out_json}")
    return metrics


def _fmt(v, scale=100.0, nd=2):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * scale:.{nd}f}%"


def _dump_errors(bank: ErrorBank, out_dir: str, pert_name: str) -> list:
    entries = bank.dump(out_dir)
    if not entries:
        print(f"[seer] no misclassifications to dump for '{pert_name}'")
        return entries
    print(f"\n  error analysis [{pert_name}] -> {out_dir}")
    for kind, header in ErrorBank.KINDS.items():
        rows = [e for e in entries if e["kind"] == kind]
        if not rows:
            print(f"    {header}: none")
            continue
        print(f"    {header}:")
        for e in rows:
            source = e.get("generator") or e.get("architecture") or "unknown source"
            print(f"      P(AI)={e['prob_ai']:.3f}  {source:<24s} "
                  f"{Path(e['file']).name}")
    return entries


def _print_robustness_table(results: dict):
    print("\n" + "=" * 78)
    print(" Robustness sweep (benchmark perturbation protocol, both classes)")
    print("=" * 78)
    print(f"  {'perturbation':<12s} {'macro acc':>10s} {'mAP':>8s} {'AUROC':>8s} "
          f"{'F1':>8s} {'FPR':>7s} {'FNR':>7s}  n")
    for name, m in results.items():
        desc = PERTURBATIONS.get(name, ("", ""))[1]
        print(f"  {name:<12s} {_fmt(m['macro_accuracy']):>10s} {_fmt(m['mAP']):>8s} "
              f"{_fmt(m['auroc']):>8s} {_fmt(m['f1']):>8s} {_fmt(m['fpr']):>7s} "
              f"{_fmt(m['fnr']):>7s}  {m['n']}"
              f"  ({desc})")
    print("=" * 78 + "\n")


def _print_report(m: dict, dataset: str):
    print("\n" + "=" * 72)
    pert = m.get("perturbation", "clean")
    desc = PERTURBATIONS.get(pert, ("", ""))[1]
    tag = f"PERTURBED [{pert}: {desc}]" if pert != "clean" else "CLEAN"
    print(f" Seer on {dataset} [{tag}]")
    print("=" * 72)
    print(f"  macro accuracy : {_fmt(m['macro_accuracy'])}")
    print(f"  mAP            : {_fmt(m['mAP'])}")
    print(f"  AUROC          : {_fmt(m['auroc'])}")
    if m.get("robust_auroc") is not None:
        print(f"  robust AUROC   : {_fmt(m['robust_auroc'])}  (distorted n={m.get('robust_n', 0)})")
    print(f"  F1             : {_fmt(m['f1'])}")
    print(f"  precision / rec: {_fmt(m.get('precision'))} / {_fmt(m.get('recall'))}")
    print(f"  accuracy       : {_fmt(m['accuracy'])}")
    print(f"  FPR / FNR      : {_fmt(m['fpr'])} / {_fmt(m['fnr'])}  (n={m['n']})")
    per_arch = m.get("per_architecture") or {}
    if per_arch and any(a for a in per_arch):
        print("  per architecture:")
        for a, mm in per_arch.items():
            print(
                f"    {a:<12s} macro_acc={_fmt(mm['macro_accuracy']):>8s} "
                f"mAP={_fmt(mm['mAP']):>8s} F1={_fmt(mm['f1']):>8s} "
                f"AUROC={_fmt(mm['auroc']):>8s} FPR={_fmt(mm['fpr']):>8s} "
                f"(n={mm['n']})"
            )
    per_d = m.get("per_distorted") or {}
    if per_d:
        print("  clean vs distorted:")
        for a, mm in per_d.items():
            print(
                f"    {a:<12s} macro_acc={_fmt(mm['macro_accuracy']):>8s} "
                f"F1={_fmt(mm['f1']):>8s} AUROC={_fmt(mm['auroc']):>8s} "
                f"(n={mm['n']})"
            )
    per_dist = m.get("per_distortion") or {}
    if per_dist:
        print("  per first distortion (n>=20):")
        for a, mm in per_dist.items():
            print(
                f"    {a:<16s} macro_acc={_fmt(mm['macro_accuracy']):>8s} "
                f"F1={_fmt(mm['f1']):>8s} (n={mm['n']})"
            )
    if m.get("perturbation_sweep"):
        _print_robustness_table(m["perturbation_sweep"])
    pub = PUBLISHED.get(dataset if dataset in EVAL_SPECS else "", [])
    if pub:
        print("  published reference (same benchmark):")
        for name, acc, mp in pub:
            print(f"    {name:<24s} macro_acc={acc:.2f}%  mAP={mp:.2f}%")
    print("=" * 72 + "\n")


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Evaluate an Seer checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="comfor_eval",
                   help=f"one of {known_eval_datasets()}")
    p.add_argument("--real-dir", nargs="*", default=None)
    p.add_argument("--fake-dir", nargs="*", default=None)
    p.add_argument("--augmented", action="store_true",
                   help="apply the Pangram augmented protocol (1024px + JPEG q50)")
    p.add_argument("--perturbation", default=None,
                   help="one of the named perturbations, or all / extra / all+extra "
                        "(all = official JPEG/blur/resize/noise/jitter/crop table)")
    p.add_argument("--hflip-tta", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--error-dir", type=str, default=None,
                   help="write the most confident false positives / negatives "
                        "there as heatmap panels")
    p.add_argument("--error-n", type=int, default=4,
                   help="how many of each to keep (default 4)")
    args = p.parse_args(argv)
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        augmented=args.augmented,
        perturbation=args.perturbation,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        hflip_tta=args.hflip_tta,
        real_dirs=args.real_dir,
        fake_dirs=args.fake_dir,
        out_json=args.out_json,
        error_dir=args.error_dir,
        error_n=args.error_n,
    )


if __name__ == "__main__":
    main()
