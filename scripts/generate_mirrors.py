"""Synthetic mirroring: generate AI training data locally with diffusers.

This is our open implementation of Pangram's core data technique. Instead of
paying frontier APIs, we:

  1. harvest *real-content-grounded* prompts (LAION captions that the
     Community Forensics generators were seeded with - real topics, styles,
     genres), and/or caption real images with a local VLM;
  2. generate with local checkpoints covering different generator families
     (SD1.5 / SDXL / SDXL-Turbo / FLUX.1-schnell / SD3.5-M);
  3. optionally run img2img "AI edits" over real photos (simulating edit
     tools, which regenerate the whole image even for small edits - the
     distribution Pangram flags as fully-AI).

Because the mirrors are grounded in the same content distribution as our
real negatives, the detector cannot solve the task from content priors - it
must learn generator fingerprints.

Examples:
  # 2000 SDXL fakes from harvested real-content prompts
  uv run scripts/generate_mirrors.py --generator sdxl --n 2000   # -> F:/techjam/mirrors

  # FLUX.1-schnell (needs cpu offload on 12GB GPUs)
  uv run scripts/generate_mirrors.py --generator flux-schnell --n 1000 --offload

  # "AI edited" real photos (also harvests the real bases as negatives)
  uv run scripts/generate_mirrors.py --generator sdxl --mode img2img --strength 0.45 --n 500

  # VLM-caption a real folder, then mirror from those captions
  uv run scripts/generate_mirrors.py --generator sdxl --n 500 \
      --caption-model Qwen/Qwen2.5-VL-7B-Instruct --caption-images data/my_photos

Requires the `gen` dependency group:  uv sync --group gen
"""

import argparse
import csv
import io
import os
import random
import time
from pathlib import Path

from seer.paths import DATA_ROOT

ZENODO_URL = "https://zenodo.org/records/10066460/files/synthbuster.zip"
IMAGENET_NOTE = "prompts are harvested from LAION captions of real web images"

GENERATORS = {
    "sd15": dict(
        model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        steps=30, guidance=7.5, size=512, supports_negative=True,
    ),
    "sdxl": dict(
        model="stabilityai/stable-diffusion-xl-base-1.0",
        steps=30, guidance=5.0, size=1024, supports_negative=True,
    ),
    "sdxl-turbo": dict(
        model="stabilityai/sdxl-turbo",
        steps=4, guidance=0.0, size=1024, supports_negative=True,
    ),
    "flux-schnell": dict(
        model="black-forest-labs/FLUX.1-schnell",
        steps=4, guidance=0.0, size=1024, supports_negative=False,
    ),
    "sd35-medium": dict(
        model="stabilityai/stable-diffusion-3.5-medium",
        steps=28, guidance=4.5, size=1024, supports_negative=False,
    ),
}

DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, nsfw, nude"


def harvest_prompts(n: int, seed: int = 0, pool_cap: int = 200_000) -> list:
    """Real-content-grounded prompts from the Community Forensics stream
    (the LAION captions its generators were seeded with). NSFW-filtered."""
    import datasets as hfds

    ds = hfds.load_dataset("OwensLab/CommunityForensics-Small", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=4096)
    prompts, seen = [], set()
    for row in ds:
        if len(prompts) >= n:
            break
        if row.get("label") != 1 or row.get("nsfw_flag"):
            continue
        p = (row.get("prompt") or "").strip()
        if len(p) < 8 or p.lower() in seen:
            continue
        seen.add(p.lower())
        prompts.append(p)
    if not prompts:
        raise RuntimeError("No prompts harvested - is the dataset reachable?")
    print(f"[mirror] harvested {len(prompts)} real-content prompts")
    return prompts


def harvest_real_images(n: int, out_dir: str, seed: int = 0) -> list:
    """Pull real images to use as img2img bases; they double as negatives."""
    import datasets as hfds
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    ds = hfds.load_dataset("OwensLab/CommunityForensics-Small", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=4096)
    paths = []
    for row in ds:
        if len(paths) >= n:
            break
        if row.get("label") != 0:
            continue
        try:
            img = Image.open(io.BytesIO(row["image_data"])).convert("RGB")
        except Exception:
            continue
        if min(img.size) < 256:
            continue
        path = os.path.join(out_dir, f"real_{len(paths):06d}.png")
        img.save(path)
        paths.append(path)
    print(f"[mirror] harvested {len(paths)} real images -> {out_dir}")
    return paths


def vlm_captions(image_paths: list, model_id: str, n: int) -> list:
    """Caption real images with a local VLM (full Pangram-style mirroring)."""
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import torch

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    captions = []
    for path in image_paths[:n]:
        img = Image.open(path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image as a detailed prompt for an image-generation model."},
                ],
            }
        ]
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=[img], text=[prompt], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            caption = processor.batch_decode(out, skip_special_tokens=True)[0]
            captions.append(caption.strip())
        except Exception as e:
            print(f"[mirror] VLM caption failed on {path}: {e}")
            continue
        if len(captions) % 50 == 0 and captions:
            print(f"[mirror] captioned {len(captions)}/{min(n, len(image_paths))}")
    return captions


def main():
    p = argparse.ArgumentParser(description="Generate synthetic-mirror training data")
    p.add_argument("--generator", required=True, choices=list(GENERATORS),
                   help="which local generator family to run")
    p.add_argument("--n", type=int, default=1000, help="how many images to generate")
    p.add_argument("--out", default=str(DATA_ROOT / "mirrors"),
                   help="output root (default: F:/techjam/mirrors)")
    p.add_argument("--mode", choices=["txt2img", "img2img"], default="txt2img")
    p.add_argument("--strength", type=float, default=0.45,
                   help="img2img edit strength (regeneration amount)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offload", action="store_true",
                   help="cpu offload (needed for FLUX on 12GB GPUs)")
    p.add_argument("--caption-model", default=None,
                   help="optional VLM (e.g. Qwen/Qwen2.5-VL-7B-Instruct) to caption real images")
    p.add_argument("--caption-images", default=None,
                   help="dir of real images to caption (with --caption-model)")
    p.add_argument("--i2i-prompt", default="a high quality, detailed photograph of this scene",
                   help="driving prompt for img2img mode")
    p.add_argument("--nsfw-filter", action="store_true", default=True,
                   help="filter harvested prompts by the dataset NSFW flag")
    args = p.parse_args()

    import torch
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    gen_cfg = GENERATORS[args.generator]
    steps = args.steps or gen_cfg["steps"]
    guidance = args.guidance if args.guidance is not None else gen_cfg["guidance"]
    size = args.size or gen_cfg["size"]

    out_root = Path(args.out)
    fake_dir = out_root / args.generator
    real_dir = out_root / "real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- prompts
    prompts = None
    base_images = None
    if args.mode == "txt2img":
        if args.caption_model:
            assert args.caption_images, "--caption-images required with --caption-model"
            pool = sorted(str(p) for p in Path(args.caption_images).rglob("*")
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            print(f"[mirror] captioning {min(args.n, len(pool))} images with {args.caption_model}")
            prompts = vlm_captions(pool, args.caption_model, args.n)
            if not prompts:
                raise RuntimeError("VLM captioning produced no prompts")
        else:
            prompts = harvest_prompts(args.n, seed=args.seed)
    else:
        base_images = harvest_real_images(args.n, str(real_dir), seed=args.seed + 1)
        if not base_images:
            raise RuntimeError("no real bases harvested")

    # ---------------------------------------------------------- pipeline
    pipe_cls = AutoPipelineForText2Image if args.mode == "txt2img" else AutoPipelineForImage2Image
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = pipe_cls.from_pretrained(gen_cfg["model"], torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    if args.offload or (args.generator == "flux-schnell" and not _has_vram_gb(22)):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    meta_path = out_root / f"meta_{args.generator}_{args.mode}.csv"
    rng = random.Random(args.seed)
    t0 = time.time()
    done = 0
    consecutive_failures = 0
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "generator", "mode", "prompt", "strength", "seed"])
        while done < args.n:
            seed = rng.randrange(1 << 31)
            try:
                if args.mode == "txt2img":
                    prompt = prompts[done % len(prompts)]
                    kw = dict(prompt=prompt, num_inference_steps=steps,
                              guidance_scale=guidance, height=size, width=size,
                              generator=torch.Generator("cpu").manual_seed(seed))
                    if gen_cfg["supports_negative"]:
                        kw["negative_prompt"] = DEFAULT_NEGATIVE
                    img = pipe(**kw).images[0]
                else:
                    base_path = base_images[done % len(base_images)]
                    from PIL import Image

                    base = Image.open(base_path).convert("RGB")
                    base = _fit(base, size)
                    kw = dict(prompt=args.i2i_prompt, image=base,
                              strength=args.strength, num_inference_steps=max(steps, 20),
                              guidance_scale=guidance if guidance > 0 else 1.0,
                              generator=torch.Generator("cpu").manual_seed(seed))
                    if gen_cfg["supports_negative"]:
                        kw["negative_prompt"] = DEFAULT_NEGATIVE
                    img = pipe(**kw).images[0]
            except Exception as e:
                consecutive_failures += 1
                print(f"[mirror] generation failed ({e}); retrying with new seed")
                if consecutive_failures >= 10:
                    raise RuntimeError("too many consecutive generation failures - aborting")
                continue
            consecutive_failures = 0
            path = fake_dir / f"{args.generator}_{done:06d}.png"
            img.save(path)
            w.writerow([str(path), args.generator, args.mode,
                        prompts[done % len(prompts)] if prompts else args.i2i_prompt,
                        args.strength if args.mode == "img2img" else "",
                        seed])
            done += 1
            if done % 25 == 0:
                rate = done / max(1e-9, time.time() - t0)
                eta = (args.n - done) / max(rate, 1e-9)
                print(f"[mirror] {done}/{args.n} ({rate:.2f} img/s, eta {eta / 60:.1f} min)")

    print(f"[mirror] wrote {done} images -> {fake_dir}")
    print(f"[mirror] metadata -> {meta_path}")
    print("\nAdd this to your training config's data.sources:")
    print(
        f"  - name: {args.generator}_{args.mode}\n"
        f"    type: folders\n"
        f"    real_dirs: [{str(real_dir)!r}]\n"
        f"    fake_dirs: [{str(fake_dir)!r}]\n"
        f"    weight: 0.15"
    )


def _fit(img, size):
    """Resize + center-crop a PIL image to size x size."""
    w, h = img.size
    s = min(w, h)
    x0, y0 = (w - s) // 2, (h - s) // 2
    img = img.crop((x0, y0, x0 + s, y0 + s))
    return img.resize((size, size))


def _has_vram_gb(n: float) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(0).total_memory / 1e9 >= n


if __name__ == "__main__":
    main()
