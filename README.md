# Seer — a sub-2B-parameter AI-generated image detector

**TikTok TechJam 2026** · Goal: rival [Pangram Image](https://www.pangram.com/blog/introducing-pangram-image-detection) with a detector under **2B parameters**.

Seer is a DINOv3 ViT-L backbone (~300M params) fully fine-tuned for AI-image
detection, with a dual head for image-level verdicts **and** patch-level
heatmaps. Total: **~302M parameters — 15% of the budget**. It is trained on a
public-data mixture that follows Pangram's data strategy:
generator diversity, public frontier fakes, wild-simulation
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
| AI data | synthetic mirroring (VLM caption → regenerate) + scraped real-world AI images | Community Forensics (4,803 gens) + NTIRE (42 gens) + FLUX.1-dev + frontier commercial fakes + SID synthetic + GAS-Station v3/v4 |
| Real data | diverse web imagery; careful FPR control (WikiArt 0/2000, ReLAION 0.16% FPR) | Community Forensics + NTIRE matched reals + jp1924/Laion400m-1 + Open Images V7; WikiArt/folders FPR harness |
| Augmentation | strong "in the wild" simulation (crop, edit, compression) | symmetric wild-simulation at benchmark levels: JPEG q∈{90,70,50,30}, blur σ∈{0.5,1,2}, resize 0.5×/0.25×, noise σ∈{0.02,0.05,0.10}, jitter ±20%, crop 80%, WebP, grayscale |
| Mixed images | composite training → heatmaps | same: cropped overlays in all four real/fake pairings, stacked multi-overlay, per-patch labels |
| Scale | proprietary scrape of frontier generators (GPT Image, Nano Banana, FLUX, Midjourney, Grok) | everything above is public |

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
anything else they tried."* Full source list, weights, fetch commands, and
held-out sets: **[docs/DATA_MIXTURE.md](docs/DATA_MIXTURE.md)**.

The hero + probe configs (`seer_vitl_512.yaml`, `seer_probe.yaml`) share this
weighted mix. Missing folder sources are dropped at train time, not fatal.

| Source | Class | Weight | What it covers |
|---|---|---|---|
| **NTIRE 2026 train** | mixed | 0.30 | 42 gens (2022–2026), all 6 shards, real/fake matched |
| **CommunityForensics-Small** | mixed | 0.26 | 4,803 open generators + paired reals. Eval is held out |
| **GAS-Station v4 / v3** | fake | 0.11 / 0.10 | weekly open-model dumps after `wire_gasstation.py` |
| **laion400m-1** | real | 0.10 | `jp1924/Laion400m-1` images in parquet (not a URL scrape) |
| **Open Images V7** | real | 0.10 | validation + test photographs |
| **FLUX-Reason-6M** | fake | 0.05 | 5.9M FLUX.1-dev; streamed |
| **Frontier fakes** | fake | 0.08 | Midjourney / DALL-E / SD / Nano Banana Pro (label inverted) |
| **SID_Set** | fake | 0.06 | full-synthetic only (drop real + tampered) |

Roots: `$SEER_DATA_ROOT` (defaults to `/workspace/data` when that mount
exists, else `F:/techjam`). Local parquet is read in streaming mode.

## Setup

```bash
uv sync                     # torch (CUDA), transformers, datasets, ...
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

# 3. build the data mixture (see docs/DATA_MIXTURE.md)
#    everything lands in $SEER_DATA_ROOT (/workspace/data or F:/techjam)
uv run python get_datasets.py --list                      # the full plan; downloads nothing
uv run python get_datasets.py --tier 1                    # NTIRE train/val/test + COCO
uv run python dataset_stats.py --tier 1                   # remote metadata only, no images
uv run scripts/fetch_data.py comfor-small                 # full ~260GB; add --max-shards 30 for a slice
uv run scripts/fetch_data.py frontier-fakes               # MJ / DALL-E / SD / Nano Banana Pro (~3 GB)
uv run scripts/fetch_data.py flux-reason-6m --max-shards 8 # optional; full dump is streamed
uv run scripts/fetch_data.py sid-set --max-shards 16
uv run scripts/wire_gasstation.py --versions v3 v4        # unpack GAS-Station tarballs
uv run scripts/download_laion400m.py --max-shards 12 --max-images 150000 --min-side 512
uv run scripts/download_open_images.py --workers 32 --max-gb 70

# 4. full training (hero config = the mixture above)
uv run python main.py train --config configs/seer_vitl_512.yaml          # A100-class
uv run python main.py train --config configs/seer_vitl_local.yaml       # single 12GB GPU

# 4b. multi-layer linear probe (frozen backbone) - cheap ablation
#     against continuation training
uv run python main.py train --config configs/seer_probe.yaml
#     or on top of any config:
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set probe.enabled=true probe.layers=[3,9,15,-1] head_lr=1e-3

# 5. benchmark against Pangram's protocol
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_eval
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_eval --augmented   # 1024px + JPEG q50
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_test   # HF public test (2.5k)
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset folders \
    --real-dir data/wikiart --out-json wikiart_fpr.json                   # FPR eval

# 5b. error analysis: the most confident FPs / FNs, each with its heatmap
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val \
    --error-dir runs/eval/errors --error-n 6 --out-json runs/eval/ntire_val.json

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
  decay 0.8, cosine schedule, warmup 1k, EMA 0.999, bf16. The probe recipe
  uses Muon on 2D head weights instead (`optimizer: muon`).
- **Dual-head objective**: image-level BCE + per-patch BCE (weight 1.0), the
  patch term `pos_weight`-balanced by `n_real/n_fake` patches
  (`balance_patch`) so a small fake-over-real crop is not drowned out by the
  real majority around it.
- **Composite training** (60% of fake samples; ~25% of the batch is mixed
  FoR/RoF): cropped overlays layered over a base image. Compositing is
  itself a discontinuity, so *all four*
  top-on-base pairings are trained — fake-over-real (localized labels),
  real-over-fake (inverted labels), fake-over-fake (label 1 everywhere),
  real-over-real (label stays real). Per-pixel fake-ness is Porter-Duff
  composited along with the RGB — a 40% blend is a 0.4 target — then pooled
  to the patch grid, so soft mixes get soft targets; the page target stays
  binary (any visible AI → fake), and label maps travel with the crop when an
  already-composited slot is reused as a source. Overlays are
  random crops of the source (own scale / aspect / flip), a sample can
  receive n ~ Uniform{1,...,k} overlays (k = max_overlays, default 3),
  and two overlay modes mix by default:
  - `blend` — smooth bilinear alpha (seamless, diffusion-style mixes)
  - `paste` — opaque hard-edged overlay with a ~2px feathered border
    (sticker / screenshot-style content drops)
- **Wild-simulation augmentation** applied symmetrically to both classes,
  with levels drawn from the benchmark robustness protocols:
  JPEG q∈{90,70,50,30} · blur σ∈{0.5,1.0,2.0} · resize 0.5×/0.25× ·
  Gaussian noise σ∈{0.02,0.05,0.10} · jitter ±20% · center crop 80% ·
  plus WebP, grayscale, hflip.
- 512×512 input (patch grid 32×32), effective batch 108 (54 × 2 accum), ~60k steps.

### Multi-layer linear probe (ablation)

`probe` mode is the frozen-backbone alternative to continuation training:
linear heads on features tapped from several transformer blocks.
Early blocks carry high-frequency / low-level statistics — where generator
fingerprints live — while mid and late blocks carry increasingly semantic
features, so the probe sees both ends of the hierarchy. Two independent heads
are trained, each over its own LayerNorm-standardized concatenation of taps:
a **page head** on `[CLS ; mean(patch tokens)]` → one logit per image, and a
**patch head** on the raw patch tokens → one logit per patch. Probe
checkpoints therefore produce heatmaps too.

- `probe.layers`: 0-based block indices, negative from the end (`-1` = final
  block); empty = four evenly spaced taps (e.g. `[6, 12, 18, 23]` on
  DINOv3 ViT-L's 24 blocks).
- The backbone always stays frozen and runs without activation gradients,
  so bigger batches fit and gradient checkpointing is unnecessary (both ViT-L
  probe heads together are ~37k parameters: LayerNorm + linear over 8,192-d
  page features and 4,096-d patch features).
- Use a higher head LR (~1e-3) than for fine-tuning; composite training
  applies unchanged — the page head keeps the "contains AI content" label and
  the patch head trains on the composite patch targets.
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
  after `python get_datasets.py --tier 1`. `ntire_test` is the labelled
  2.5k public test from
  [`deepfakesMSU/NTIRE-RobustAIGenDetection-test-public`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-test-public)
  (clean vs distorted + per-distortion). The 512 recipe also scores it
  every `eval_every` steps (`eval_datasets: [ntire_test]`); `best.pt`
  still follows the train-distribution val slice.
- **FPR sets** — real-only folders (WikiArt etc.).
- Metrics: macro (balanced) accuracy, mAP (AP on fake class), AUROC, F1,
  precision/recall, FPR/FNR, plus per-architecture and (on NTIRE)
  clean-vs-distorted and per-distortion breakdowns.
  Published Pangram numbers are printed next to ours for direct comparison.

### Error analysis (`--error-dir`)

Aggregate metrics say *that* a detector fails, never *why*. Any eval pass can
keep its worst mistakes and write each one out as an explained panel:

```bash
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt \
    --dataset ntire_val --error-dir runs/eval/errors --error-n 6
```

Selection is by confidence, because that is where the blind spots are legible:
false positives rank by how certain the model was that a real photo was AI,
false negatives by how certain it was that a generated image was real. A
mistake at P(AI)=0.51 is noise; one at 0.99 is a lesson. `ErrorBank` keeps only
the top `--error-n` of each kind (two bounded heaps), so a 50k-image eval costs
nothing extra in memory.

```
runs/eval/errors/
  fp_01_p0.994_<image>.png   real, called AI at 99.4%
  fn_01_p0.006_<image>.png   AI, called real at 99.4%
```

Each panel is `input | input + per-patch AI heatmap`, rendered from the same
local head that produces inference heatmaps — so a false positive shows *which
region* dragged the verdict up, and a false negative shows whether the model
half-saw the generated area or missed it entirely. The images are the ones the
model actually saw, i.e. after the perturbation under test. Page-only
checkpoints dump the plain image and mark `explained: false`.

The same records land in the metrics JSON under `error_analysis`, tagged with
the generator where the corpus provides one, so failures can be counted per
generator instead of eyeballed:

```json
{"kind": "fn", "rank": 1, "file": "runs/eval/errors/fn_01_p0.006_1234.png",
 "prob_ai": 0.006, "label": 1, "explained": true,
 "generator": "flux.1-dev", "distortions": ["jpeg"]}
```

With `--perturbation all` the dump repeats per level into
`runs/eval/errors/<perturbation>/`, which answers something the summary table
cannot: are the images that fail under JPEG q30 the same ones that failed
clean, or different ones? For a quick look, run it on a single perturbation —
the full sweep writes 16 folders.

## Limitations (honest ones)

- Frontier-generator coverage (GPT Image, Nano Banana, Grok, Riverflow) is
  API-gated; the public mix covers those *families* via NTIRE, FLUX-Reason,
  frontier fakes, and GAS-Station, not the exact latest APIs. Swap in real
  outputs via any folder/HF source as they become available.
- CommunityForensics-Small is SD-derivative-heavy — the weighted mixture
  mitigates that.
- No deepfake/face-swap detection (Pangram's initial release doesn't either).
- Probe checkpoints (frozen backbone) are page-level only - no heatmaps.
- Error analysis is only as good as the eval set: `--error-dir` surfaces the
  most confident mistakes, but a corpus with no digital art in the real class
  cannot show you the false positives that class would cause.

## References

- Stajduhar & Emi, *Introducing Pangram Image Detection*, 2026 (blog)
- Park & Owens, *Community Forensics*, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125))
- Bammey, *Synthbuster*, OJSP 2023 ([Zenodo](https://zenodo.org/records/10066460))
- Simeoni et al., *DINOv3*, 2025 ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104))
- Zhu et al., *GenImage*, NeurIPS 2023 ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571))
- Gushchin et al., *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*
