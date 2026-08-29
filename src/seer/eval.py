"""Benchmark harness.

Replicates the evaluation protocol of the Pangram Image technical blog:

  * CommunityForensics-Eval ("CompEval") - the comprehensive evaluation set of
    CVPR 2025 "Community Forensics" - macro accuracy + mAP
  * "Augmented" protocol - downscale to 1024x1024 + JPEG q50, both classes
  * Robustness sweeps - the benchmark perturbation protocol (JPEG 90/70/50/30,
    blur sigma 0.5/1/2, resize 0.5x/0.25x, noise 0.02/0.05/0.10, jitter +/-20%,
    center crop 80%) via `--perturbation all`
  * False-positive-rate evals on real-only data (WikiArt etc. via folders)

Metrics: macro (balanced) accuracy, mAP (average precision on the fake class),
AUROC, FPR, FNR. Published reference numbers from the Pangram blog are printed
next to ours for a direct comparison.
"""

import json
import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .augment import PERTURBATIONS, apply_perturbation, eval_transform
from .data import ComforStream, FolderDataset, NtireStream, load_sample_image
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
}


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs >= threshold
    pos, neg = labels == 1, labels == 0
    tp = float((pred[pos]).sum())
    fn = float((~pred[pos]).sum())
    tn = float((~pred[neg]).sum())
    fp = float((pred[neg]).sum())
    out = {
        "n": int(labels.size),
        "n_fake": int(pos.sum()),
        "n_real": int(neg.sum()),
        "accuracy": (tp + tn) / max(1, labels.size),
        "macro_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp)),
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
        return NtireStream(split=NTIRE_EVAL[dataset]["split"], seed=0)
    spec = EVAL_SPECS.get(dataset)
    if spec is None:
        known = list(EVAL_SPECS) + list(NTIRE_EVAL) + ["folders"]
        raise ValueError(f"Unknown eval dataset '{dataset}' (try {known})")
    return ComforStream(dataset=spec["dataset"], split=spec["split"],
                         shuffle_buffer=1024, max_samples=None, seed=0)


@torch.no_grad()
def _single_pass(model, cfg_dict, perturbation: Optional[str], augmented: bool,
                 dataset: str, batch_size: int, max_samples: Optional[int],
                 hflip_tta: bool, res: int, device) -> dict:
    ds = _build_eval_dataset(dataset)

    pert_name = perturbation or ("pangram" if augmented else "clean")
    perturb_fn = (lambda im: apply_perturbation(im, pert_name)) if pert_name != "clean" else None

    probs, labels, archs, gens = [], [], [], []
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
        probs.extend(p.cpu().tolist())
        labels.extend(int(s["label"]) for s in chunk)
        archs.extend(s.get("architecture", "") for s in chunk)
        gens.extend(s.get("generator", "") for s in chunk)

    probs = np.array(probs)
    labels = np.array(labels)
    metrics = compute_metrics(probs, labels)
    metrics["dataset"] = dataset
    metrics["perturbation"] = pert_name
    metrics["hflip_tta"] = hflip_tta

    by_arch = defaultdict(lambda: ([], []))
    for p, y, a in zip(probs, labels, archs):
        by_arch[a][0].append(p)
        by_arch[a][1].append(y)
    metrics["per_architecture"] = {}
    for a in sorted(by_arch):
        ps, ys = np.array(by_arch[a][0]), np.array(by_arch[a][1])
        metrics["per_architecture"][a] = compute_metrics(ps, ys)
    return metrics


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
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg_dict, ckpt = load_checkpoint(checkpoint, device=device)
    res = res or int(cfg_dict.get("res", 512))
    model.eval()

    def pass_kwargs():
        return dict(model=model, cfg_dict=cfg_dict, dataset=dataset,
                    batch_size=batch_size, max_samples=max_samples,
                    hflip_tta=hflip_tta, res=res, device=device)

    if perturbation == "all":
        results = {}
        for name in PERTURBATIONS:
            m = _single_pass(perturbation=name, augmented=False, **pass_kwargs())
            results[name] = {k: m[k] for k in
                             ("macro_accuracy", "mAP", "auroc", "accuracy", "fpr", "fnr", "n")}
        metrics = dict(results["clean"])
        metrics["perturbation_sweep"] = results
        _print_robustness_table(results)
    else:
        if perturbation is not None and perturbation not in PERTURBATIONS:
            raise ValueError(f"Unknown perturbation '{perturbation}'. "
                             f"Available: {list(PERTURBATIONS)} or 'all'")
        metrics = _single_pass(perturbation=perturbation, augmented=augmented, **pass_kwargs())

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


def _print_robustness_table(results: dict):
    print("\n" + "=" * 78)
    print(" Robustness sweep (benchmark perturbation protocol, both classes)")
    print("=" * 78)
    print(f"  {'perturbation':<12s} {'macro acc':>10s} {'mAP':>8s} {'AUROC':>8s} "
          f"{'FPR':>7s} {'FNR':>7s}  n")
    for name, m in results.items():
        desc = PERTURBATIONS.get(name, ("", ""))[1]
        print(f"  {name:<12s} {_fmt(m['macro_accuracy']):>10s} {_fmt(m['mAP']):>8s} "
              f"{_fmt(m['auroc']):>8s} {_fmt(m['fpr']):>7s} {_fmt(m['fnr']):>7s}  {m['n']}"
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
    print(f"  accuracy       : {_fmt(m['accuracy'])}")
    print(f"  FPR / FNR      : {_fmt(m['fpr'])} / {_fmt(m['fnr'])}  (n={m['n']})")
    per_arch = m.get("per_architecture") or {}
    if per_arch:
        print("  per architecture:")
        for a, mm in per_arch.items():
            print(
                f"    {a:<12s} macro_acc={_fmt(mm['macro_accuracy']):>8s} "
                f"mAP={_fmt(mm['mAP']):>8s} FPR={_fmt(mm['fpr']):>8s} "
                f"(n={mm['n']})"
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
                   help=f"one of {list(EVAL_SPECS) + list(NTIRE_EVAL)} or 'folders'")
    p.add_argument("--real-dir", nargs="*", default=None)
    p.add_argument("--fake-dir", nargs="*", default=None)
    p.add_argument("--augmented", action="store_true",
                   help="apply the Pangram augmented protocol (1024px + JPEG q50)")
    p.add_argument("--perturbation", default=None,
                   help=f"benchmark perturbation: one of {list(PERTURBATIONS)} or 'all'")
    p.add_argument("--hflip-tta", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-json", type=str, default=None)
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
    )


if __name__ == "__main__":
    main()
