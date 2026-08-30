"""Rank OpenFake generators by how badly a checkpoint detects them.

Reads the probe set written by `scripts/openfake.py probe`
(`<root>/openfake/probe/{fake,real}/<model>/*.jpg`), scores every generator
with one checkpoint, and prints generators worst-first. That ranking is what
decides which OpenFake models are worth adding to the training mixture: a
generator we already catch at 99% recall under compression teaches the
detector nothing, one we catch at 40% is a hole in the mixture.

Each generator is scored clean and under the eval-table perturbations, because
a generator that is easy clean and hopeless at JPEG 30 is still a hole. The
headline column is `recall_mean`, the mean recall over all conditions at the
0.5 threshold. AUROC is computed against the pooled real scores of the same
condition, so it is comparable across generators.

  python scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
  python scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt \
      --conditions clean jpeg50 resize0.5 --per-model 80
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from seer.augment import apply_perturbation, eval_transform
from seer.model import load_checkpoint

Image.MAX_IMAGE_PIXELS = None
DEFAULT_CONDITIONS = ("clean", "jpeg50", "jpeg30", "resize0.5", "blur1.0")


def _probe_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("SEER_DATA_ROOT")
    base = Path(env) if env else Path("/workspace/data")
    return base / "openfake" / "probe"


def collect(root: Path, per_model: int, real_n: int, seed: int) -> tuple[dict, list]:
    rng = random.Random(seed)
    fakes: dict[str, list[Path]] = {}
    for d in sorted((root / "fake").glob("*")):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.jpg"))
        rng.shuffle(files)
        if files:
            fakes[d.name] = files[:per_model] if per_model else files
    reals: list[Path] = []
    for d in sorted((root / "real").glob("*")):
        if d.is_dir():
            reals.extend(d.glob("*.jpg"))
    rng.shuffle(reals)
    if real_n:
        reals = reals[:real_n]
    if not fakes or not reals:
        raise SystemExit(f"probe set incomplete under {root} "
                         f"(fake models={len(fakes)}, reals={len(reals)})")
    return fakes, reals


def _prepare(path: Path, condition: str, res: int):
    """Decode, perturb and resize one image. Runs in the decode pool."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    if condition != "clean":
        img = apply_perturbation(img, condition)
    return eval_transform(img, res)


@torch.no_grad()
def score(model, files: list[Path], condition: str, res: int, device, batch: int,
          pool: ThreadPoolExecutor) -> np.ndarray:
    """Probabilities for one file list. Decode is the bottleneck here (the
    probe JPEGs are up to 1536px and the perturbations are PIL), so it is
    farmed out to threads in bounded blocks - unbounded map would hold every
    decoded tensor in memory at once."""
    out: list[float] = []
    block = max(batch, batch * 4)
    for start in range(0, len(files), block):
        chunk = files[start:start + block]
        tensors = [t for t in pool.map(lambda f: _prepare(f, condition, res), chunk)
                   if t is not None]
        for i in range(0, len(tensors), batch):
            x = torch.stack(tensors[i:i + batch]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x)["logits"]
            out.extend(torch.sigmoid(logits).float().cpu().tolist())
    return np.asarray(out, dtype=np.float64)


def auroc(fake_probs: np.ndarray, real_probs: np.ndarray) -> float:
    if fake_probs.size == 0 or real_probs.size == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones_like(fake_probs), np.zeros_like(real_probs)])
    p = np.concatenate([fake_probs, real_probs])
    return float(roc_auc_score(y, p))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe-dir", default=None)
    p.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    p.add_argument("--per-model", type=int, default=60, help="0 = every image")
    p.add_argument("--real-n", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--workers", type=int, default=6,
                   help="decode threads; keep low, the cgroup is ~13 cpus and "
                        "a training job is usually sharing them")
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    args = p.parse_args(argv)

    root = _probe_root(args.probe_dir)
    fakes, reals = collect(root, args.per_model, args.real_n, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg_dict, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    res = args.res or int(cfg_dict.get("res", 512))

    n_fake = sum(len(v) for v in fakes.values())
    print(f"[rank] {len(fakes)} generators, {n_fake} fakes, {len(reals)} reals, "
          f"res={res}, conditions={args.conditions}", flush=True)

    per_model: dict[str, dict] = defaultdict(dict)
    real_by_cond: dict[str, np.ndarray] = {}
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for condition in args.conditions:
        real_probs = score(model, reals, condition, res, device, args.batch_size, pool)
        real_by_cond[condition] = real_probs
        fpr = float((real_probs >= 0.5).mean()) if real_probs.size else float("nan")
        print(f"[rank] {condition}: real fpr={fpr:.4f} mean_p={real_probs.mean():.4f}", flush=True)
        for name, files in fakes.items():
            probs = score(model, files, condition, res, device, args.batch_size, pool)
            per_model[name][condition] = {
                "n": int(probs.size),
                "recall": float((probs >= 0.5).mean()) if probs.size else float("nan"),
                "mean_p": float(probs.mean()) if probs.size else float("nan"),
                "auroc": auroc(probs, real_probs),
            }

    rows = []
    for name, by_cond in per_model.items():
        recalls = [v["recall"] for v in by_cond.values()]
        aurocs = [v["auroc"] for v in by_cond.values()]
        rows.append({
            "model": name,
            "n": by_cond[args.conditions[0]]["n"],
            "recall_mean": float(np.nanmean(recalls)),
            "recall_clean": by_cond.get("clean", {}).get("recall", float("nan")),
            "recall_min": float(np.nanmin(recalls)),
            "auroc_mean": float(np.nanmean(aurocs)),
            "per_condition": by_cond,
        })
    rows.sort(key=lambda r: r["recall_mean"])

    print(f"\n{'recall_mean':>11} {'clean':>7} {'worst':>7} {'auroc':>7} {'n':>5}  generator")
    for r in rows:
        print(f"{r['recall_mean']:11.3f} {r['recall_clean']:7.3f} {r['recall_min']:7.3f} "
              f"{r['auroc_mean']:7.3f} {r['n']:5d}  {r['model']}")

    payload = {
        "checkpoint": args.checkpoint,
        "res": res,
        "conditions": args.conditions,
        "n_real": len(reals),
        "real_fpr": {c: float((v >= 0.5).mean()) for c, v in real_by_cond.items()},
        "generators": rows,
    }
    out = Path(args.out_json) if args.out_json else root.parent / "rank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=1)
    print(f"\n[rank] wrote {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
