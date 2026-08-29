"""Save predicted AI heatmaps from a Seer checkpoint.

Uses held-out / folder images when they exist (NTIRE val, Open Images,
LAION reals). Falls back to a few files passed with --image.

  uv run python scripts/save_heatmaps.py --checkpoint runs/seer_probe/last.pt
  uv run python scripts/save_heatmaps.py --checkpoint runs/seer_probe/last.pt \
      --image photo.jpg --out runs/seer_probe/heatmaps
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image

from seer.heatmap import predict_and_explain, save_heatmap
from seer.model import load_checkpoint
from seer.paths import DATA_ROOT


def _first_files(root: Path, n: int) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                out.append(Path(dirpath) / fn)
                if len(out) >= n:
                    return out
    return out


def collect_images(extra: list[str], per_class: int) -> list[tuple[Path, str]]:
    picked: list[tuple[Path, str]] = []
    for path in extra:
        p = Path(path)
        if p.is_file():
            picked.append((p, "given"))
    if extra:
        return picked

    for tag, root in (
        ("real", DATA_ROOT / "open-images-v7"),
        ("real", DATA_ROOT / "laion400m-1" / "real"),
        ("real", DATA_ROOT / "coco-val2017"),
    ):
        for p in _first_files(root, per_class):
            picked.append((p, tag))
        if sum(1 for _, t in picked if t == "real") >= per_class:
            break

    try:
        from seer.ntire import load_val, stratified_subset
        import numpy as np

        samples = stratified_subset(load_val(hard=False), per_class * 2, np.random.default_rng(0))
        for s in samples:
            picked.append((s.path, "fake" if s.label == 1 else "real"))
    except Exception:
        pass

    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/seer_probe/heatmaps")
    p.add_argument("--n", type=int, default=8, help="images to dump when scanning data roots")
    p.add_argument("--image", nargs="*", default=[], help="explicit image paths")
    p.add_argument("--res", type=int, default=None)
    args = p.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(args.checkpoint, device=device)
    res = args.res or int(cfg.get("res", 512))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = collect_images(args.image, max(1, args.n // 2))
    if not items:
        raise SystemExit("no images found; pass --image or populate $SEER_DATA_ROOT")

    print(f"dumping {len(items)} heatmaps -> {out}")
    for i, (path, tag) in enumerate(items):
        img = Image.open(path).convert("RGB")
        prob, heat = predict_and_explain(model, img, res, device)
        if heat is None:
            raise SystemExit("checkpoint has no patch head")
        dest = out / f"sample_{i:02d}_{tag}_p{prob:.3f}.png"
        save_heatmap(str(dest), img, heat, prob, res)
        print(f"  {dest.name}  {path}  P(AI)={prob:.3f}")


if __name__ == "__main__":
    main()
