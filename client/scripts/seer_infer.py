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
"""

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description="Seer -> dashboard JSON bridge")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    import numpy as np  # noqa: F401  (torch dep, keeps the heavy import local)
    import torch
    from PIL import Image

    from seer.augment import eval_transform
    from seer.model import load_checkpoint

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg_dict, _ = load_checkpoint(args.checkpoint, device=device)
    res = args.res or int(cfg_dict.get("res", 512))
    model.eval()

    records = []
    with torch.no_grad():
        for path in args.image:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            # the commercial model refuses <512px inputs; upscale instead so
            # every image gets a verdict (mirrors seer/infer.py)
            if min(w, h) < res:
                s = res / min(w, h)
                img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
                w, h = img.size

            x = eval_transform(img, res)[None].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(x)

            prob = torch.sigmoid(out["logits"][0].float()).item()
            grid = None
            patch = out.get("patch_logits")
            if patch is not None:
                probs = torch.sigmoid(patch[0].float()).cpu().numpy()
                g = int(round(float(np.sqrt(probs.size))))
                grid = [[round(float(v), 4) for v in row] for row in probs.reshape(g, g)]

            records.append(
                {
                    "image": path,
                    "prob_ai": round(prob, 6),
                    "label": "AI" if prob >= 0.5 else "REAL",
                    "grid": grid,
                    "width": w,
                    "height": h,
                }
            )

    json.dump(records, sys.stdout)


if __name__ == "__main__":
    main()
