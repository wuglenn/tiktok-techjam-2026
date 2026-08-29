"""Single-image inference CLI.

  uv run python -m seer.infer --checkpoint runs/seer/best.pt --image photo.jpg --out heatmap.png
"""

import argparse
import json
from typing import List, Optional

import torch
from PIL import Image

from .heatmap import predict_and_explain, save_heatmap
from .model import load_checkpoint


def infer(
    checkpoint: str,
    images: List[str],
    out_dir: Optional[str] = None,
    res: Optional[int] = None,
    device: Optional[str] = None,
) -> list:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg_dict, _ = load_checkpoint(checkpoint, device=device)
    res = res or int(cfg_dict.get("res", 512))

    results = []
    for path in images:
        img = Image.open(path).convert("RGB")
        # The Pangram Image model refuses < 512px inputs; we instead upscale
        # small images so every input gets a verdict.
        w, h = img.size
        if min(w, h) < res:
            s = res / min(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)

        prob, heat = predict_and_explain(model, img, res, device)
        rec = {"image": path, "prob_ai": prob, "label": "AI" if prob >= 0.5 else "REAL"}
        results.append(rec)
        print(f"{path}: P(AI)={prob:.4f} -> {rec['label']}")

        if heat is None:
            print("  (no per-patch heatmap for this checkpoint)")
        elif out_dir:
            import os

            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{base}_heatmap.png")
            save_heatmap(out_path, img, heat, prob, res)
            print(f"  heatmap -> {out_path}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Seer single-image inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--out-dir", default=None, help="directory for heatmap PNGs")
    p.add_argument("--res", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None, help="write results as JSON")
    args = p.parse_args(argv)
    results = infer(args.checkpoint, args.image, args.out_dir, args.res, args.device)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
