"""Batch AIGC detection over an image directory.

TikTok TechJam 2026 Track 5 deliverable: takes an image directory and writes a
JSON file with one `{"image_path", "pred"}` record per image, where `pred` is
P(AI-generated) in [0, 1].

    uv run python predict.py --image-dir ./images --checkpoint best.pt --out preds.json

    [
      {"image_path": "images/photo_001.jpg", "pred": 0.0031},
      {"image_path": "images/render_014.png", "pred": 0.9994}
    ]

`pred` is a score, not a decision: pick the operating threshold that suits the
deployment. We report metrics at 0.5.

Useful extras:

    --heatmap-dir out/heat    per-patch AI heatmap PNG next to every verdict
    --out-detailed rich.json  adds label / decode size / heatmap path per image
    --resume                  continue an interrupted run over a large directory
    --hflip-tta               average the score over the horizontal flip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

# Allow `python predict.py` from a clean checkout, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from PIL import Image, ImageFile

from seer.augment import eval_transform
from seer.heatmap import patch_logits_to_heat, save_heatmap
from seer.model import load_checkpoint

# Truncated JPEGs are common in scraped corpora; a partial decode beats a crash.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def find_images(root: str, recursive: bool = True, exts: Optional[set] = None) -> List[str]:
    """Every image under `root`, sorted so two runs agree on order."""
    exts = exts or IMAGE_EXTS
    base = Path(root)
    if base.is_file():
        return [str(base)]
    if not base.is_dir():
        raise FileNotFoundError(f"--image-dir not found: {root}")

    found: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if Path(fn).suffix.lower() in exts:
                    found.append(str(Path(dirpath) / fn))
    else:
        found = [str(p) for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(found)


def _load(path: str, res: int):
    """Decode one image to the model's input tensor.

    Returns (tensor, pil_image_or_None, error_or_None) so a single unreadable
    file cannot take down a 100k-image run.
    """
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - any decode failure is just a skip
        return None, None, f"{type(exc).__name__}: {exc}"

    # The Pangram Image model refuses < 512px inputs; we upscale instead so
    # every image in the directory gets a score. Matches src/seer/infer.py.
    w, h = img.size
    if min(w, h) < res:
        s = res / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    return eval_transform(img, res), img, None


def _batches(paths: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(paths), size):
        yield list(paths[i : i + size])


def _prior_scores(progress_path: Path, out_path: Path, threshold: float) -> dict:
    """Scores from an earlier run: the JSONL sidecar of an interrupted run, or
    the final JSON of a completed one."""
    prior: dict = {}
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a hard kill
                prior[rec["image_path"]] = rec
    elif out_path.exists():
        try:
            for rec in json.load(out_path.open("r", encoding="utf-8")):
                if rec.get("pred") is None:
                    continue  # retry anything that failed to decode
                prior[rec["image_path"]] = {
                    **rec,
                    "label": "AI" if rec["pred"] >= threshold else "REAL",
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return prior


@torch.no_grad()
def predict_dir(
    checkpoint: str,
    image_dir: str,
    out: str = "predictions.json",
    out_detailed: Optional[str] = None,
    heatmap_dir: Optional[str] = None,
    batch_size: int = 16,
    workers: int = 8,
    res: Optional[int] = None,
    device: Optional[str] = None,
    recursive: bool = True,
    limit: int = 0,
    threshold: float = 0.5,
    hflip_tta: bool = False,
    resume: bool = False,
    quiet: bool = False,
) -> List[dict]:
    paths = find_images(image_dir, recursive=recursive)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no images found under {image_dir}")

    # Scores land in a JSONL sidecar as they are produced, so an interrupted
    # run over a large directory can pick up where it stopped.
    progress_path = Path(f"{out}.progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict = _prior_scores(progress_path, Path(out), threshold) if resume else {}
    if not resume and progress_path.exists():
        progress_path.unlink()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not quiet:
        print(f"[load] {checkpoint} -> {dev}")
    model, cfg_dict, ckpt = load_checkpoint(checkpoint, device=dev)
    res = res or int(cfg_dict.get("res", 512))
    if not quiet:
        step = ckpt.get("step")
        params = ckpt.get("param_count") or sum(p.numel() for p in model.parameters())
        print(
            f"[model] {ckpt.get('backbone_name', 'unknown')} | {params:,} params"
            f"{f' | step {step:,}' if step else ''} | {res}px"
        )
        print(f"[scan] {len(paths):,} images under {image_dir}")

    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)

    # `done` is keyed by the posix form we write to JSON, so compare on that
    # rather than on the OS-native path os.walk handed us.
    todo = [p for p in paths if _posix(p) not in done]
    n_reused = len(paths) - len(todo)
    if n_reused and not quiet:
        print(f"[resume] {n_reused:,} already scored, {len(todo):,} to go")

    records: List[dict] = []
    failed: List[dict] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, progress_path.open(
        "a", encoding="utf-8"
    ) as progress:
        for batch in _batches(todo, batch_size):
            loaded = list(pool.map(lambda p: _load(p, res), batch))

            ok = [(p, t, img) for p, (t, img, err) in zip(batch, loaded) if t is not None]
            for p, (_, _, err) in zip(batch, loaded):
                if err is not None:
                    failed.append({"image_path": _posix(p), "pred": None, "error": err})

            if ok:
                x = torch.stack([t for _, t, _ in ok]).to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    out_dict = model(x)
                    if hflip_tta:
                        flipped = model(torch.flip(x, dims=[-1]))

                probs = torch.sigmoid(out_dict["logits"].float())
                if hflip_tta:
                    probs = 0.5 * (probs + torch.sigmoid(flipped["logits"].float()))
                probs = probs.cpu().tolist()

                patch = out_dict.get("patch_logits")
                heats = (
                    patch_logits_to_heat(patch.float().cpu(), (res, res)).numpy()
                    if (patch is not None and heatmap_dir)
                    else None
                )

                for i, (p, _, img) in enumerate(ok):
                    prob = float(probs[i])
                    rec = {"image_path": _posix(p), "pred": round(prob, 6)}
                    detail = {
                        **rec,
                        "label": "AI" if prob >= threshold else "REAL",
                        "width": img.width,
                        "height": img.height,
                    }
                    if heats is not None:
                        stem = Path(p).stem
                        heat_path = os.path.join(heatmap_dir, f"{stem}_p{prob:.3f}_heatmap.png")
                        save_heatmap(heat_path, img, heats[i], prob, res)
                        detail["heatmap"] = _posix(heat_path)
                    records.append(detail)
                    progress.write(json.dumps(detail) + "\n")
                progress.flush()

            if not quiet:
                n = n_reused + len(records) + len(failed)
                rate = len(records) / max(1e-9, time.perf_counter() - t0)
                line = (
                    f"[score] {n:,}/{len(paths):,}  {rate:.1f} img/s"
                    f"{f'  {len(failed)} unreadable' if failed else ''}"
                )
                print(f"\r{line:<64}", end="", flush=True)

    if not quiet:
        print()

    # Resumed records first so output order follows the directory scan.
    by_path = {r["image_path"]: r for r in list(done.values()) + records}
    ordered = [by_path[_posix(p)] for p in paths if _posix(p) in by_path] + failed

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump([{"image_path": r["image_path"], "pred": r["pred"]} for r in ordered], f, indent=2)

    if out_detailed:
        Path(out_detailed).parent.mkdir(parents=True, exist_ok=True)
        with open(out_detailed, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": checkpoint,
                    "backbone": ckpt.get("backbone_name"),
                    "step": ckpt.get("step"),
                    "param_count": ckpt.get("param_count"),
                    "res": res,
                    "threshold": threshold,
                    "hflip_tta": hflip_tta,
                    "image_dir": _posix(image_dir),
                    "n_images": len(paths),
                    "n_scored": len(ordered) - len(failed),
                    "n_failed": len(failed),
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "predictions": ordered,
                },
                f,
                indent=2,
            )

    if not quiet:
        scored = [r for r in ordered if r["pred"] is not None]
        flagged = sum(1 for r in scored if r["pred"] >= threshold)
        print(f"[done] {len(scored):,} scored in {time.perf_counter() - t0:.1f}s -> {out}")
        if scored:
            print(
                f"       {flagged:,} at or above P(AI)>={threshold} "
                f"({flagged / len(scored):.1%}), {len(scored) - flagged:,} below"
            )
        if failed:
            print(f"       {len(failed)} unreadable (pred=null): {failed[0]['error']}")
        if out_detailed:
            print(f"       detailed -> {out_detailed}")
        if heatmap_dir:
            print(f"       heatmaps -> {heatmap_dir}")

    progress_path.unlink(missing_ok=True)
    return ordered


def _posix(path: str) -> str:
    """Forward slashes so the JSON reads the same on Windows and Linux."""
    return str(path).replace("\\", "/")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Score every image in a directory for AI generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Output: JSON array of {image_path, pred} where pred = P(AI-generated).",
    )
    p.add_argument("--image-dir", required=True, help="directory of images (or a single image)")
    p.add_argument("--checkpoint", default="best.pt", help="trained Seer checkpoint")
    p.add_argument("--out", default="predictions.json", help="output JSON path")
    p.add_argument("--out-detailed", default=None, help="optional richer JSON (label, size, run metadata)")
    p.add_argument("--heatmap-dir", default=None, help="also write a per-patch AI heatmap PNG per image")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8, help="decode threads")
    p.add_argument("--res", type=int, default=None, help="override the checkpoint's input resolution")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--no-recursive", action="store_true", help="do not descend into subdirectories")
    p.add_argument("--limit", type=int, default=0, help="score only the first N images")
    p.add_argument("--threshold", type=float, default=0.5, help="reporting threshold only")
    p.add_argument("--hflip-tta", action="store_true", help="average score over the horizontal flip")
    p.add_argument("--resume", action="store_true", help="reuse scores from an interrupted run")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    predict_dir(
        checkpoint=args.checkpoint,
        image_dir=args.image_dir,
        out=args.out,
        out_detailed=args.out_detailed,
        heatmap_dir=args.heatmap_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        res=args.res,
        device=args.device,
        recursive=not args.no_recursive,
        limit=args.limit,
        threshold=args.threshold,
        hflip_tta=args.hflip_tta,
        resume=args.resume,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
