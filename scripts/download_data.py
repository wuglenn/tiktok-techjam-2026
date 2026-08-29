"""Materialize a slice of a Community Forensics dataset to disk.

Streaming is fine for training, but a local copy makes iteration faster and
lets you train offline. Only the shards containing the requested samples are
downloaded.

  uv run scripts/download_data.py --split train --n 5000 --out data/comfor_small
"""

import argparse
import csv
import io
import os

from PIL import Image

from seer.data import ComforStream


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="OwensLab/CommunityForensics-Small")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--out", default="data/comfor_small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=4096)
    args = p.parse_args()

    real_dir = os.path.join(args.out, "real")
    fake_dir = os.path.join(args.out, "fake")
    for d in (real_dir, fake_dir):
        os.makedirs(d, exist_ok=True)

    ds = ComforStream(
        dataset=args.dataset,
        split=args.split,
        shuffle_buffer=args.shuffle_buffer,
        max_samples=args.n,
        seed=args.seed,
    )

    meta_path = os.path.join(args.out, "meta.csv")
    n = 0
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "generator", "architecture", "prompt"])
        for s in ds:
            label = "real" if s["label"] == 0 else "fake"
            d = real_dir if label == "real" else fake_dir
            base = s.get("image_name") or f"img_{n:06d}"
            path = os.path.join(d, f"{n:06d}_{os.path.splitext(base)[0]}.png")
            s["image"].save(path)
            w.writerow([path, label, s.get("generator", ""), s.get("architecture", ""),
                        (s.get("prompt", "") or "")[:200]])
            n += 1
            if n % 500 == 0:
                print(f"[download] {n}/{args.n}")
            if n >= args.n:
                break
    print(f"[download] wrote {n} images + {meta_path}")


if __name__ == "__main__":
    main()
