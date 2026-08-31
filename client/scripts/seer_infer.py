"""Seer dashboard inference bridge.

Called by the Next.js API route (client/src/app/api/analyze) — not meant for
direct human use. Prints one JSON record per image to stdout:

    [{"image": "...", "prob_ai": 0.987, "label": "AI",
      "grid": [[0.91, ...], ...], "width": 512, "height": 512}, ...]

`grid` is the raw per-patch probability grid from the local head (G x G,
e.g. 32x32 at res 512 / patch 16) — the web client upsamples and overlays it
itself, so no matplotlib rendering happens here. `grid` is null for
page-only checkpoints (frozen probes).

  uv run python client/scripts/seer_infer.py \
      --checkpoint runs/seer_vitl/best.pt --image a.jpg b.jpg

The dashboard prefers the persistent server in seer_serve.py (loads the
checkpoint once). This CLI is the fallback that loads per request.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_runtime(checkpoint: str, device: Optional[str] = None, res: Optional[int] = None):
    """Load weights on CPU, drop EMA/optimizer tensors, then move the model.

    A TechJam `best.pt` is ~4.9 GB (model + EMA + optimizer). Mapping that
    whole blob onto the GPU with `map_location=cuda` will OOM a 10 GB card.
    """
    import torch

    from seer.model import load_checkpoint

    model, cfg_dict, ckpt = load_checkpoint(checkpoint, device="cpu")
    meta = {
        "backbone": ckpt.get("backbone_name"),
        "step": ckpt.get("step"),
        "param_count": ckpt.get("param_count") or sum(p.numel() for p in model.parameters()),
        "res": int(res or cfg_dict.get("res", 512)),
    }
    for key in ("model", "ema", "optimizer", "scheduler"):
        ckpt[key] = None
    del ckpt
    gc.collect()

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    return model, dev, meta


def score_one(model, path: str, res: int, device) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image

    from seer.augment import eval_transform

    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    w, h = orig_w, orig_h
    # the commercial model refuses <512px inputs; upscale instead so
    # every image gets a verdict (mirrors seer/infer.py)
    if min(w, h) < res:
        s = res / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = img.size

    x = eval_transform(img, res)[None].to(device)
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(x)

    prob = torch.sigmoid(out["logits"][0].float()).item()
    grid = None
    patch = out.get("patch_logits")
    if patch is not None:
        probs = torch.sigmoid(patch[0].float()).cpu().numpy()
        g = int(round(float(np.sqrt(probs.size))))
        grid = [[round(float(v), 4) for v in row] for row in probs.reshape(g, g)]

    return {
        "image": path,
        "prob_ai": round(float(prob), 6),
        "label": "AI" if prob >= 0.5 else "REAL",
        "grid": grid,
        "width": orig_w,
        "height": orig_h,
    }


def score_images(model, paths: Sequence[str], res: int, device) -> List[dict[str, Any]]:
    return [score_one(model, path, res, device) for path in paths]


def main(argv=None):
    p = argparse.ArgumentParser(description="Seer -> dashboard JSON bridge")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    model, device, meta = load_runtime(args.checkpoint, args.device, args.res)
    records = score_images(model, args.image, meta["res"], device)
    json.dump(records, sys.stdout)


if __name__ == "__main__":
    main()
