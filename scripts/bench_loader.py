"""Profile the training input pipeline and identify the bottleneck.

Measures, in order:
  1. iterate  - dataset row yield (local/remote parquet, or pre-made bytes)
  2. decode   - PIL materialization of samples (single thread)
  3. collate  - decode + wild-simulation augment + composites, single
                thread vs thread pool
  4. GPU step - fwd+bwd at the target resolution (if CUDA available)

Your training throughput ceiling is min(stage 3b, stage 4).

Examples:
  uv run scripts/bench_loader.py --source synthetic
  uv run scripts/bench_loader.py --source local --parquet-dir F:/techjam/comfor-small/data
  uv run scripts/bench_loader.py --source remote --n 256
  uv run scripts/bench_loader.py --source synthetic --backbone facebook/dinov2-large --res 518
"""

import argparse
import io
import time

import numpy as np
import torch
from PIL import Image


def _synth_bytes(res, n, seed=0):
    """Pre-encode n synthetic PNG 'dataset rows' (smooth content, realistic
    JPEG/PIL cost)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        base = rng.rand(res // 16, res // 16, 3) * 200 + 20
        img = np.asarray(
            Image.fromarray(base.astype(np.uint8)).resize((res, res), Image.BICUBIC)
        ).astype(np.float32)
        img = np.clip(img + rng.randn(res, res, 3) * 6, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["local", "remote", "synthetic"], default="synthetic")
    p.add_argument("--parquet-dir", default=None)
    p.add_argument("--n", type=int, default=256, help="samples per stage")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--backbone", default="facebook/dinov2-small")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--decode-workers", type=int, default=8)
    args = p.parse_args()

    from seer.config import load_config
    from seer.data import BatchBuilder, ComforStream, load_sample_image
    from seer.model import SeerDetector, detection_loss

    n = args.n

    # ------------------------------------------------------------- dataset
    if args.source == "local":
        assert args.parquet_dir, "--parquet-dir required for local"
        ds = ComforStream(local_dirs=[args.parquet_dir], shuffle_buffer=1024, seed=0)
        ds_iter = iter(ds)
    elif args.source == "remote":
        ds = ComforStream(shuffle_buffer=256, seed=0, max_samples=n + 64)
        ds_iter = iter(ds)
    else:
        print(f"[bench] pre-encoding {n} synthetic {args.res}px images...")
        blobs = _synth_bytes(args.res, n)

        def _synth_iter():
            i = 0
            while True:
                yield {"image": None, "image_bytes": blobs[i % len(blobs)], "label": i % 2,
                       "generator": "synth", "architecture": ""}
                i += 1

        ds_iter = _synth_iter()

    def take(k):
        return [next(ds_iter) for _ in range(k)]

    # --------------------------------------------------------------- stage 1
    samples = take(min(n, 512))
    t0 = time.perf_counter()
    take(n)
    it_rate = n / (time.perf_counter() - t0)
    print(f"[bench] 1. dataset iterate ({args.source}): {it_rate:8.1f} img/s")

    # --------------------------------------------------------------- stage 2
    t0 = time.perf_counter()
    for s in samples:
        load_sample_image(s)
    dec_rate = len(samples) / (time.perf_counter() - t0)
    print(f"[bench] 2. PIL decode (1 thread):          {dec_rate:8.1f} img/s")

    # --------------------------------------------------------------- model
    model = None
    patch_size = 16
    eff_res = args.res
    if torch.cuda.is_available():
        model = SeerDetector(args.backbone, pretrained=args.pretrained)
        patch_size = model.patch_size
        eff_res = ((args.res + patch_size - 1) // patch_size) * patch_size  # e.g. 512 -> 518 for patch 14
        print(f"[bench] backbone patch {patch_size}: using res {eff_res}")

    cfg = load_config(overrides=[f"res={eff_res}", f"decode_workers={args.decode_workers}"])

    # --------------------------------------------------------------- stage 3
    batch = take(args.batch_size)
    bb_single = BatchBuilder(cfg, train=True, patch_grid=eff_res // patch_size, seed=0, decode_workers=1)
    t0 = time.perf_counter()
    b = bb_single(batch)
    single_rate = args.batch_size / (time.perf_counter() - t0)

    bb_pool = BatchBuilder(cfg, train=True, patch_grid=eff_res // patch_size, seed=0,
                           decode_workers=args.decode_workers)
    t0 = time.perf_counter()
    b = bb_pool(batch)
    pool_rate = args.batch_size / (time.perf_counter() - t0)
    print(f"[bench] 3a. full collate, 1 thread:         {single_rate:8.1f} img/s")
    print(f"[bench] 3b. full collate, {args.decode_workers} threads:        {pool_rate:8.1f} img/s")

    # --------------------------------------------------------------- stage 4
    gpu_rate = None
    if model is not None:
        model.enable_gradient_checkpointing()
        model.cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        images = b["images"].cuda()
        labels = b["labels"].cuda()
        pl = b["patch_labels"].cuda()

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(images)
                loss, _ = detection_loss(out["logits"], out["patch_logits"], labels, pl, 0.5)
            loss.backward()
            opt.step()

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        gpu_rate = (5 * args.batch_size) / (time.perf_counter() - t0)
        print(f"[bench] 4. GPU fwd+bwd ({args.backbone.split('/')[-1]}, "
              f"bs={args.batch_size}): {gpu_rate:8.1f} img/s")
    else:
        print("[bench] 4. GPU stage skipped (no CUDA or res/patch mismatch)")

    # --------------------------------------------------------------- verdict
    # the training loop always uses the pooled collate, so the single-thread
    # number is informational only
    rates = {"collate (pooled)": pool_rate}
    if gpu_rate:
        rates["GPU step"] = gpu_rate
    bottleneck = min(rates, key=rates.get)
    print(f"\n[bench] bottleneck: {bottleneck} at {rates[bottleneck]:.1f} img/s")
    ceiling = min(pool_rate, gpu_rate or float("inf"))
    print(f"[bench] training throughput ceiling: {ceiling:.1f} img/s")


if __name__ == "__main__":
    main()
