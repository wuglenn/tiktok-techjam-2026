# Seer — a sub-2B-parameter AI-generated image detector

**TikTok TechJam 2026** · Goal: rival [Pangram Image](https://www.pangram.com/blog/introducing-pangram-image-detection) with a detector under **2B parameters**.

Seer is a DINOv3 ViT-L backbone (~300M params) fully fine-tuned for AI-image
detection, with a dual head for image-level verdicts **and** patch-level
heatmaps. Total: **~302M parameters — 15% of the budget**. It is trained on a
public-data mixture that mirrors every pillar of Pangram's data strategy:
generator diversity, real-content-grounded synthetic mirroring, wild-simulation
augmentation, and composite training.

```
              ┌─────────────────────────┐
 image ──────► │  DINOv3 ViT-L (~300M)  │  continuation training:
              │  self-supervised ViT    │  the whole backbone is fine-tuned,
              └───────────┬─────────────┘  not frozen behind a probe
        CLS token          patch tokens
              │                 │
     ┌────────▼───────┐  ┌──────▼────────┐
     │  global head   │  │  local head    │
     │  (MLP)         │  │  (per-patch)   │
     └────────┬───────┘  └──────┬────────┘
              │                 │
        P(AI image)        AI heatmap
```

## State of the field (July 2026)

Pangram Image is the current commercial SOTA. Their recipe, from the
[technical blog](https://www.pangram.com/blog/introducing-pangram-image-detection):

| Pillar | Pangram Image | Seer |
|---|---|---|
| Backbone | DINOv3, **full continuation fine-tuning** ("AI detection is not an ordinary downstream task") | same — DINOv3 ViT-L, full FT with layer-wise LR decay |
| AI data | synthetic mirroring (VLM caption → regenerate) + scraped real-world AI images | mirroring via local generators (SDXL/FLUX.1-schnell/SD1.5/img2img edits) seeded by real-content captions + 4,803 open generators (Community Forensics) + 9 modern families (Synthbuster) |
| Real data | diverse web imagery; careful FPR control (WikiArt 0/2000, ReLAION 0.16% FPR) | paired reals from the mixture; WikiArt/folders FPR harness |
| Augmentation | strong "in the wild" simulation (crop, edit, compression) | symmetric wild-simulation at benchmark levels: JPEG q∈{90,70,50,30}, blur σ∈{0.5,1,2}, resize 0.5×/0.25×, noise σ∈{0.02,0.05,0.10}, jitter ±20%, crop 80%, WebP, grayscale |
| Mixed images | composite training → heatmaps | same: cropped overlays in all four real/fake pairings, stacked multi-overlay, per-patch labels |
| Scale | proprietary scrape of frontier generators (GPT Image, Nano Banana, FLUX, Midjourney, Grok) | everything above is public + locally generatable |

Reference numbers to rival (macro accuracy / mAP):

| Benchmark | Pangram Image | Previous best |
|---|---|---|
| CommunityForensics-Eval | 97.29% / 99.70% | Ours-384 (CVPR'25): 89.3% / 98.7% |
| Synthbuster + RAISE-1K | 98.49% / 99.96% | B-Free: 94.9% / 98.8% |
| Augmented (1024px + JPEG q50) | 99.03% acc | SightEngine 97.57% |
| NTIRE 2026 (AUROC) | 99.999% | MICV 99.78% |

Our eval harness (`seer/eval.py`) implements exactly these protocols, and
prints Pangram's published numbers next to ours for a direct comparison.

## The data mixture (what makes or breaks this task)

Pangram: *"the specific data composition of both human and AI-generated
imagery had the largest impact on the final accuracy of the model compared to
anything else they tried."* Their early failure mode — synthetic mirrors that
didn't match the in-the-wild distribution — is the thing our mixture is
designed around:

| Source | What it covers | How |
|---|---|---|
| **NTIRE 2026** | 42 generators spanning 2022–2026, real/fake matched on resolution, aspect ratio and JPEG quality, with per-image degradation labels. One 19 GB shard is a complete training set. | `python get_datasets.py --tier 1` → `$SEER_DATA_ROOT/ntire`. Mixture type `ntire`; eval `--dataset ntire_val` / `ntire_val_hard` / `ntire_test` |
| **CommunityForensics-Small** | 278K fakes from **4,803 generators** (LatDiff/PixDiff/GAN) + 278K paired reals. Generator *diversity* is the main driver of generalization to unseen generators (CVPR 2025). | `scripts/fetch_data.py comfor-small` → local parquet on `F:/techjam` (one-time download, then disk-speed reads) |
| **FLUX-Reason-6M** | 5.9M **FLUX.1-dev** images (all fake). Stream; the full dump is ~882 GB. | mixture `type: hf` on `LucasFang/FLUX-Reason-6M`, `label: 1`. Optional slice: `scripts/fetch_data.py flux-reason-6m --max-shards 8` |
| **Frontier fakes** | Midjourney, DALL-E, Stable Diffusion, **Nano Banana Pro** — fake class only. The Hub ClassLabel is inverted (`0=fake`, `1=real`). | `scripts/fetch_data.py frontier-fakes` (~3 GB train). Mixture remaps then `keep_label: 1` |
| **SID_Set** | 210k train images. Three classes: real / full synthetic / tampered. We keep **synthetic + tampered** as fake. | Stream `saberzl/SID_Set`. Optional slice: `scripts/fetch_data.py sid-set --max-shards 16` |
| **DDA-Training-Set** | VAE reconstructions of COCO train, format-aligned (PNG, spatial). Fake half only. | 11-part zip, not streamable. `python get_datasets.py --only dda-train` then folders `dda-train/fake` |
| **Synthbuster** | 9 modern families: DALLE2/3, Firefly, Midjourney v5, SD 1.3/1.4/2.1, SDXL | `scripts/download_synthbuster.py` (Zenodo, CC-BY) → `F:/techjam/synthbuster` |
| **Local synthetic mirrors** | frontier-family coverage + *AI-edited photos* (img2img), grounded in real-content captions so the detector can't cheat on content priors | `scripts/generate_mirrors.py` (diffusers on a single 12GB GPU) → `F:/techjam/mirrors` |
| **Your data / future generators** | anything new (GPT Image, Nano Banana, Riverflow...) drops in without code changes | any HF dataset (streamed) or local folder |

Everything lives under `F:/techjam` (override with `SEER_DATA_ROOT`); the HF
cache is redirected there too. Local parquet is read in streaming mode - no
network, no arrow cache duplication - so repeated epochs cost nothing.

Weights are per-source sampling probabilities in the config, so no family
dominates. Synthetic mirroring details (`generate_mirrors.py`):

1. **Prompt grounding** — harvest the real-content LAION captions that
   Community Forensics generators were seeded with (or caption your own
   folder with a local VLM, e.g. Qwen2.5-VL — the full Pangram technique).
2. **txt2img mirrors** — generate with SDXL / SDXL-Turbo / FLUX.1-schnell /
   SD3.5-M / SD1.5 at each model's native resolution, PNG-encoded to
   preserve generator fingerprints.
3. **img2img edits** — strength-0.2–0.7 regenerations of real photos,
   simulating edit tools (which regenerate the whole image — the reason
   Pangram flags edited images as fully-AI).
4. **Paired negatives** — the harvested real images are saved alongside and
   used as reals, keeping content distributions matched.

## Setup

```bash
uv sync                     # torch (CUDA), transformers, datasets, ...
uv sync --group gen         # optional: diffusers for synthetic mirroring
uv run pytest tests -q      # end-to-end smoke tests (no network needed)
```

DINOv3 weights are gated: accept the license at
[`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
then `hf auth login`. No account? Use open DINOv2 backbones instead
(`facebook/dinov2-large`, same ~300M class):

```bash
uv run python main.py train --config configs/seer_vitl_512.yaml --set backbone=facebook/dinov2-large res=518
```

## Usage

```bash
# 1. sanity check the model + parameter budget (random tiny backbone, offline)
uv run python main.py info --backbone tiny

# 2. quick end-to-end training run on streamed data (minutes, 12GB GPU)
uv run python main.py train --config configs/seer_vits_debug.yaml

# 3. build the data mixture (everything lands in F:/techjam, or $SEER_DATA_ROOT)
uv run python get_datasets.py --list                      # the full plan; downloads nothing
uv run python get_datasets.py --tier 1                    # NTIRE train/val/test + COCO (~25 GB)
uv run python dataset_stats.py --tier 1                   # remote metadata only, no images
uv run scripts/fetch_data.py comfor-small                 # full ~260GB; add --max-shards 30 for a slice
uv run scripts/fetch_data.py frontier-fakes               # MJ / DALL-E / SD / Nano Banana Pro (~3 GB)
# FLUX-Reason-6M is streamed (882 GB) — optional local slice:
uv run scripts/fetch_data.py flux-reason-6m --max-shards 8
uv run scripts/fetch_data.py sid-set --max-shards 16
uv run python get_datasets.py --only dda-train            # 11-part zip, ~113 GB
uv run scripts/download_synthbuster.py
uv run scripts/generate_mirrors.py --generator sdxl        --n 2000
uv run scripts/generate_mirrors.py --generator flux-schnell --n 1000 --offload
uv run scripts/generate_mirrors.py --generator sdxl --mode img2img --strength 0.45 --n 500

# 4. full training (hero config = the mixture above)
uv run python main.py train --config configs/seer_vitl_512.yaml          # A100-class
uv run python main.py train --config configs/seer_vitl_local.yaml       # single 12GB GPU

# 4b. page-level multi-layer linear probe (frozen backbone) - cheap ablation
#     against continuation training
uv run python main.py train --config configs/seer_probe.yaml
#     or on top of any config:
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set probe.enabled=true probe.layers=[3,9,15,-1] head_lr=1e-3

# 5. benchmark against Pangram's protocol
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_eval
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_eval --augmented   # 1024px + JPEG q50
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset folders \
    --real-dir data/wikiart --out-json wikiart_fpr.json                   # FPR eval

# 6. use it
uv run python main.py infer --checkpoint runs/seer_vitl/best.pt \
  --image suspect.jpg --out-dir out/                                    # verdict + heatmap PNG

# 7. inspect composite training data (image + patch labels per sample)
uv run scripts/save_samples.py                                           # writes samples/
```

## Throughput & the training bottleneck (measured, RTX 4070)

`scripts/bench_loader.py` profiles each pipeline stage independently
(`uv run scripts/bench_loader.py --source local --parquet-dir F:/techjam/comfor-small/data`):

| Stage (real data, 512px) | img/s |
|---|---|
| dataset iterate (local parquet, lazy bytes) | >10,000 |
| PIL decode, 1 thread | ~250 |
| full collate (decode + augment + composites), 1 thread | ~53 |
| full collate, 8 threads | ~110 |
| GPU fwd+bwd, DINOv2-S @518, bs8, grad ckpt | ~67 |

Conclusions:

- **On a 12GB GPU the bottleneck is the GPU itself** (ViT-L @512 runs
  ~6-10 img/s there, ~10x below the input pipeline ceiling) - use the
  `seer_vitl_local.yaml` profile and let the input pipeline wait.
- **The input pipeline becomes the bottleneck on bigger GPUs** (an A100
  at bs32 wants 60-100 img/s): the decode thread pool in `BatchBuilder`
  (8 threads) roughly doubles collate throughput; scale `decode_workers`
  with CPU cores on server runs.
- **Remote HF streaming is the slowest option** (HTTP row-group fetches
  every epoch) - used only for quick experiments; real training reads
  local parquet from `F:/techjam` via `scripts/fetch_data.py`.

## Parameter budget

| Backbone | Params (incl. heads) | % of 2B budget |
|---|---|---|
| DINOv3 ViT-S/16 | ~23M | 1.2% |
| DINOv2-S / ViT-B | ~24M / ~88M | ~4% |
| **DINOv3 ViT-L/16 (default)** | **~302M** | **15%** |
| DINOv3 ViT-H+/16 | ~843M | 42% |
| DINOv2 ViT-g/14 | ~1.14B | 57% |

Even the largest supported backbone stays under budget; the default leaves
6.6× headroom for resolution, TTA, or an ensemble.

## Training recipe (hero config)

- **Continuation training**: full backbone FT + heads, AdamW, layer-wise LR
  decay 0.8, cosine schedule, warmup 1k, EMA 0.999, bf16.
- **Dual-head objective**: image-level BCE + per-patch BCE (weight 0.5).
- **Composite training** (25% of samples): cropped overlays layered over a
  base image. Compositing is itself a discontinuity, so *all four*
  top-on-base pairings are trained — fake-over-real (localized labels),
  real-over-fake (inverted labels), fake-over-fake (label 1 everywhere),
  real-over-real (label stays real) — and patch labels come from each
  overlay's footprint in patch-grid space (last layer wins). Overlays are
  random crops of the source (own scale / aspect / flip), a sample can
  receive a stack of up to 3, and two overlay modes mix by default:
  - `blend` — smooth bilinear alpha (seamless, diffusion-style mixes)
  - `paste` — opaque hard-edged overlay with a ~2px feathered border
    (sticker / screenshot-style content drops)
- **Wild-simulation augmentation** applied symmetrically to both classes,
  with levels drawn from the benchmark robustness protocols:
  JPEG q∈{90,70,50,30} · blur σ∈{0.5,1.0,2.0} · resize 0.5×/0.25× ·
  Gaussian noise σ∈{0.02,0.05,0.10} · jitter ±20% · center crop 80% ·
  plus WebP, grayscale, hflip.
- 512×512 input (patch grid 32×32), effective batch 32, ~60k steps.

### Page-level multi-layer linear probe (ablation)

`probe` mode is the frozen-backbone alternative to continuation training:
a single linear layer on features tapped from several transformer blocks.
Early blocks carry high-frequency / low-level statistics — where generator
fingerprints live — while mid and late blocks carry increasingly semantic
features, so the probe sees both ends of the hierarchy. Each tap
contributes `[CLS ; mean(patch tokens)]`; the concatenation is standardized
(LayerNorm) and mapped to **one logit per page (image)** — there is no patch
head, so probe checkpoints produce verdicts but no heatmaps.

- `probe.layers`: 0-based block indices, negative from the end (`-1` = final
  block); empty = four evenly spaced taps (e.g. `[6, 12, 18, 23]` on
  DINOv3 ViT-L's 24 blocks).
- The backbone always stays frozen and runs without activation gradients,
  so bigger batches fit and gradient checkpointing is unnecessary
  (ViT-L probe head is ~8k parameters: `2 × 1024 × 4` features → 1).
- Use a higher head LR (~1e-3) than for fine-tuning; composite training
  still applies (the page label keeps its "contains AI content" meaning),
  the per-patch labels are simply unused.
- Expect it to land below continuation training (that gap is why the hero
  recipe fine-tunes) — but it's a fast, honest baseline and a good check on
  how linearly separable the frozen features are.

## Evaluation protocol

`main.py eval` replicates the Pangram blog's setup:

- **CommunityForensics-Eval (CompEval)** — 51.8K images, 21 generators incl.
  commercial ones; macro accuracy + mAP, per-architecture breakdown.
- **Robustness sweeps** — `--perturbation all` evaluates every benchmark
  perturbation level (JPEG 90/70/50/30, blur 0.5/1/2, resize 0.5×/0.25×,
  noise 0.02/0.05/0.10, jitter ±20%, crop 80%, Pangram protocol) and prints
  a per-level robustness table, matching how GenImage / OmniAID / the
  Pangram augmented eval report robustness.
- **NTIRE 2026** — `--dataset ntire_val` / `ntire_val_hard` / `ntire_test`
  after `python get_datasets.py --tier 1`.
- **FPR sets** — real-only folders (WikiArt etc.).
- Metrics: macro (balanced) accuracy, mAP (AP on fake class), AUROC, FPR/FNR.
  Published Pangram numbers are printed next to ours for direct comparison.

## Limitations (honest ones)

- Frontier-generator coverage (GPT Image, Nano Banana, Grok, Riverflow) is
  API-gated; our mirrors cover the *families* (diffusion, autoregressive,
  commercial) via SDXL/FLUX/SD3.5 + Synthbuster, not the exact models. Swap
  in real outputs via any folder/HF source as they become available.
- CommunityForensics-Small is SD-derivative-heavy — the weighted mixture
  mitigates, more mirror families help more.
- No deepfake/face-swap detection (Pangram's initial release doesn't either).
- Probe checkpoints (frozen backbone) are page-level only - no heatmaps.

## References

- Stajduhar & Emi, *Introducing Pangram Image Detection*, 2026 (blog)
- Park & Owens, *Community Forensics*, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125))
- Bammey, *Synthbuster*, OJSP 2023 ([Zenodo](https://zenodo.org/records/10066460))
- Simeoni et al., *DINOv3*, 2025 ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104))
- Zhu et al., *GenImage*, NeurIPS 2023 ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571))
- Gushchin et al., *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*
