# Seer — a sub-2B-parameter AI-generated image detector

**TikTok TechJam 2026 · Track 5 — AI-Generated Content Detection**

Seer takes a single image and answers two questions at once: *is this
AI-generated?* and *which parts of it are?* It is a DINOv3 ViT-L/16 backbone
fully fine-tuned for detection with a dual head — one global logit for the
image-level verdict, one logit per 16×16 patch for a spatial heatmap.

| | |
|---|---|
| Parameters | **305,233,922** (~305M) — **15.3% of the 2B budget**, 6.5× headroom |
| Input | 512×512 RGB, 32×32 patch grid |
| Outputs | `P(AI)` ∈ [0, 1] + a 1,024-cell AI heatmap |
| Held-out accuracy | **97.27%** on OpenFake `core/test` (89,225 unseen images, unseen generators *and* unseen real sources) — step 33,500 |
| False positive rate | **0.06%** on COCO val2017, **0.19%** on CommunityForensics-Eval, **0.17%** on OpenFake core reals |

---

## 1. How the solution addresses the problem statement

The task is not "classify images." The task is to stay correct when the image
has been through the internet, when it comes from a generator that did not
exist during training, and when the answer has to be defensible. We treated
those as three separate engineering problems plus one deliverable problem.

### 1.1 Generalization to unseen generators

A detector that memorizes one generator's fingerprint dies the week a new
model ships. Two decisions attack this:

**Continuation training, not a frozen probe.** The whole DINOv3 backbone is
fine-tuned (layer-wise LR decay 0.8, AdamW, EMA 0.999) rather than having
linear heads bolted onto frozen features. AI detection depends on
high-frequency statistics that self-supervised semantic pretraining actively
discards, so the backbone itself has to move. We built the frozen
multi-layer probe as an honest ablation (`configs/seer_probe.yaml`, taps
blocks 6/12/18/23) precisely so this claim is measured rather than asserted —
and it lands well below the fine-tune.

**Generator breadth as a data-mixture decision.** The training mixture spans
**~4,850 distinct generators** across 10 weighted sources: 4,782 from
Community Forensics (19 named families + 4,763 HuggingFace community
checkpoints), 42 from NTIRE 2026, 30 frontier/commercial models selected from
OpenFake, plus GAS-Station's weekly open-model dumps and FLUX.1-dev at scale.

The OpenFake selection is the part we are proudest of. OpenFake is 3.44 TB
across 645 shards and every shard interleaves all ~80 of its generators, so
you cannot fetch one selectively. More importantly *most of it is worthless*:
a generator the detector already catches at 99% recall costs bandwidth and
adds no gradient. So the mixture is chosen by **measured difficulty rather
than by name** — `scripts/openfake_rank.py` scores every generator clean and
under the full perturbation table, then `openfake.py fetch --from-rank` pulls
inversely to recall (25k images below 0.70 recall, 15k below 0.95, 10k below
0.98, **nothing above 0.98**). That selected 30 generators and skipped ~29
already-saturated ones. `nano-banana` came in at 0.20 recall, `qwen-image` at
0.41, `flux-1.1-pro` at 0.64; every `imagen-3/4`, `playground-v2.5`,
`sd-turbo` was deliberately left out.

The result is visible in the eval: on OpenFake `core/test` — 20 generators
that are *not in the mixture at all* — recall is 94.72% at a 0.17% FPR
(step 33,500).

### 1.2 Robustness to real-world transforms

Every image that matters has been re-compressed, resized, screenshotted, or
run through a messaging app. The augmentation pipeline
(`src/seer/augment.py`) applies **~35 distortion families symmetrically to
both classes** — critically, to reals too, so the model cannot learn
"compressed ⇒ fake."

Eval `--perturbation all` stays JPEG q∈{90,70,50,30}, blur σ∈{0.5,1,2},
resize 0.5×/0.25×, noise σ∈{0.02,0.05,0.10}, jitter ±20%, crop 80%, plus
clean. Train is intentionally harder: the hero YAML adds JPEG q∈{20,10,5},
blur σ=4, resize 0.125×, noise 0.20, jitter ±35%, WebP, grayscale, hflip,
and a stack of 1–4 harder ops drawn per sample:
double-JPEG, 4:2:0 chroma subsampling, "social re-encode" (messenger-style
chroma + JPEG/WebP), **8×8 DCT grid shift** (defeats detectors keyed to JPEG
block alignment), mismatched resample kernels (nearest-down/bicubic-up),
motion blur, sub-pixel nudge, FFT low-pass, FFT high-frequency phase noise,
chromatic aberration, film grain, surface blur, vignette, perspective
recapture, speckle, and a JPEG→WebP→JPEG recode stack.

The eval harness then scores the official 15-level table including clean
(`--perturbation all`: JPEG 90/70/50/30, blur 0.5/1/2, resize 0.5×/0.25×,
noise 0.02/0.05/0.10, jitter ±20%, crop 80%) plus the harder NTIRE-style
table (`--perturbation extra`) as a separate protocol, so the robustness
number is never an artifact of testing on exactly the augmentation seen
in training.

### 1.3 Low false-positive rate on genuine photographs

A detector that flags real photos is unshippable regardless of its recall.
Real mass therefore comes from five genuinely different distributions —
NTIRE's resolution/aspect/JPEG-quality-matched reals, Community Forensics'
COCO/FFHQ/LandscapesHQ/VISION pools, `jp1924/Laion400m-1` web-crawl photos,
OpenFake's Pexels + ReLAION reals, and Open Images V7 — and the held-out
FPR sets (COCO val2017, DOCCI, ImageNet, WikiArt) are never seen in
training. The loader hard-refuses any path under a held-out marker.

Measured at step 33,500: **0.06% FPR on 5,000 COCO val2017 photographs**
(3 false positives), 0.19% on CommunityForensics-Eval, 0.17% on 43,528
OpenFake reals it has never seen.

### 1.4 Interpretability and mixed real/AI images

Real-world content is rarely all-or-nothing — an AI object composited into a
photo, an inpainted face, a generated background. A page-level score cannot
express that, so the second head predicts **one logit per patch**, supervised
by **composite training**: 60% of fake samples are built by layering cropped
overlays with freeform silhouettes (rect, ellipse, polygon, star, blob,
noise) in all four pairings — fake-over-real, real-over-fake, fake-over-fake,
real-over-real — since compositing is itself a discontinuity the model must
not simply key on. Labels follow **occupancy, not blend opacity** (a 40%
alpha mix is still fake, not a 0.4 target).

This is what turns the error analysis from a number into an explanation: a
false positive shows *which region* dragged the verdict up, and a false
negative shows whether the model half-saw the generated area or missed it
entirely.

### 1.5 The required deliverable

`predict.py` walks an image directory (recursively by default) and writes the
exact required schema — a JSON array of `{"image_path", "pred"}` — with
batched bf16 inference, resumable checkpointing, and optional heatmap PNGs:

```bash
uv run python predict.py --image-dir ./images --checkpoint best.pt --out preds.json
```

```json
[
  { "image_path": "images/photo_001.jpg", "pred": 0.0031 },
  { "image_path": "images/render_014.png", "pred": 0.9994 }
]
```

`pred` is the sigmoid P(AI-generated) in [0, 1]. It is a score, not a
calibrated probability or a hard decision — the operating threshold is the
caller's choice, and 0.5 is only the default we report metrics at.

---

## 2. Development tools used

| Tool | Used for |
|---|---|
| **VS Code / Cursor** | primary editor for the whole repo |
| **uv** (Astral) | Python 3.10 env + dependency resolution; `pyproject.toml` + `uv.lock`, torch pinned to the `cu124` index |
| **RunPod** | training and full-scale eval — A100/H100-class for the hero run, RTX 4090 24 GB for the held-out suite; `/workspace` network volume for the ~2.5M-image mixture |
| **Git + GitHub** | version control |
| **Hugging Face Hub CLI** | gated-dataset auth (`hf auth login`) and shard fetching |
| **Next.js dev server** | the `client/` dashboard (`/` / `/analyze` / `/robustness` / `/errors`) — live demo when a checkpoint is loaded via `seer_serve.py` on :8765 |
| **matplotlib** | heatmap panels, error-analysis figures, robustness charts |
| **PowerShell / bash** | local (Windows) and remote (Linux pod) shells |

No Colab or Jupyter: every experiment is a CLI entry point
(`main.py train|eval|infer|info`) driven by a YAML config, so any run is
reproducible from one command and a config hash rather than from notebook
cell order.

---

## 3. Models and APIs used

### Backbone

| Model | Role |
|---|---|
| **`facebook/dinov3-vitl16-pretrain-lvd1689m`** | the backbone. DINOv3 ViT-L/16, 24 blocks, 1024-d, self-supervised on LVD-1689M. Gated — accept the licence, then `hf auth login` |
| `camenduru/dinov3-vitl16-pretrain-lvd1689m` | community mirror of the same weights, used on the pod where the gate was not accepted |
| `facebook/dinov2-large`, `facebook/dinov2-small` | ungated fallbacks (same ~300M class / debug tier) — the code path is backbone-agnostic and handles DINOv2's pos-encoding interpolation vs DINOv3's RoPE |

Attention kernels are resolved at load time with graceful degradation:
`flash_attention_4 → 3 → 2 → sdpa`.

### Heads (trained from scratch)

- **Global head** — `LayerNorm(2048) → Linear(2048,1024) → GELU → Dropout → Linear(1024,1)` over `[CLS ; mean(patch tokens)]`, giving one image-level logit.
- **Local head** — `Linear(1024,1)` applied per patch token → 1,024 logits on a 32×32 grid, sigmoid'd and bilinearly upsampled into the heatmap.
- **Probe heads** (ablation) — LayerNorm + linear over a concatenation of four block taps; ~37k parameters total.

### APIs

- **Hugging Face Hub HTTP API** — dataset and weight resolution, `repo_sha` pinning, byte-range parquet reads.
- **CVDF S3 (Open Images V7)** — real-image acquisition.
- **No third-party inference APIs.** Nothing is sent to a commercial detector or LLM at train, eval, or inference time; the model runs entirely locally. Pangram Image's published numbers are quoted from their technical blog as a comparison target only.

---

## 4. Libraries and frameworks used

### Python (`pyproject.toml`)

| Library | Version | Role |
|---|---|---|
| **PyTorch** | ≥2.6, `cu124` | training and inference; bf16 autocast, gradient checkpointing, `F.binary_cross_entropy_with_logits`, bilinear upsampling |
| **Hugging Face Transformers** | ≥4.56 | `AutoModel` / `AutoConfig` loading of DINOv3 & DINOv2, attention-implementation selection |
| **Hugging Face Datasets** | ≥3.2 | streaming + local parquet reads for Community Forensics, FLUX-Reason-6M, SID_Set, MIRAGE |
| **huggingface-hub** | ≥0.30 | auth, repo revision pinning, shard URLs |
| **Pillow** | ≥10.4 | all decode/encode and the entire augmentation stack (JPEG/WebP recompression, blur, resample, filters) |
| **NumPy** | ≥1.26 | FFT-domain distortions (low-pass, phase noise), grain/speckle, heatmap arrays |
| **scikit-learn** | ≥1.4 | `roc_auc_score` (AUROC) and `average_precision_score` (mAP) |
| **matplotlib** | ≥3.8 | heatmap overlays, error panels, robustness figures |
| **PyYAML** | ≥6 | config files with dotted `--set` overrides |
| **tqdm** | ≥4.66 | progress on long eval sweeps |
| **pyarrow** | via `datasets` | parquet footers, row-group-level reads |
| *(optional `gen` group)* | | `diffusers`, `accelerate`, `sentencepiece`, `protobuf` — only for `scripts/generate_mirrors.py` synthetic mirroring |

Deliberately **no** torchvision, timm, or albumentations: the augmentation
pipeline is hand-written on Pillow/NumPy because the distortions that matter
here (DCT grid shift, chroma subsampling, resample-kernel mismatch, FFT phase
noise) are not in any standard transform library, and the ones that are
needed to be reproducible from a seeded `random.Random` per sample.

Custom implementations rather than dependencies:

- **Muon optimizer** (`src/seer/optim.py`) — Newton–Schulz-orthogonalized momentum on 2D weights, AdamW on everything else. The dataclass default `TrainConfig.optimizer` is **muon**; the hero YAML (`seer_vitl_512.yaml`) overrides that to **adamw**. The probe recipe keeps Muon on 2D head weights.
- **EMA**, layer-wise LR decay param groups, cosine-with-warmup schedule.
- **Threaded prefetcher + decode pool** (`BatchBuilder`) — profiled with `scripts/bench_loader.py`; 8 decode threads roughly double collate throughput and are what keep an A100 fed.

### Frontend

| Library | Role |
|---|---|
| **Next.js 15** (App Router) | the dashboard, including the API routes that bridge into the Python model |
| **React 19** | UI |
| **TypeScript 5** | strict types across components and API routes |
| **Tailwind CSS 4** | styling, via `@tailwindcss/postcss` |
| **Geist** | typeface |

The dashboard's `/api/analyze` route prefers the persistent server
`client/scripts/seer_serve.py` on `http://127.0.0.1:8765`
(`SEER_INFER_URL`). If nothing is listening it falls back to spawning
`client/scripts/seer_infer.py` (`$SEER_PYTHON`, else `uv`, else the repo
`.venv`) when a checkpoint is present, and clearly labels **SIMULATED**
output when one is not. There is no `tests/` tree in this checkout.

---

## 5. Datasets and assets used

Everything is **public**. Full weights, on-disk counts, and fetch commands:
[`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md).

### Training mixture — 2,576,437 usable images, ~4,850 generators

`MixtureDataset` first draws a class 50/50 (`balance_labels`), then picks a
source within that class by weight.

| Source | HF / origin | Class | Weight | On disk |
|---|---|---|---|---|
| **NTIRE 2026 train** | `deepfakesMSU/NTIRE-RobustAIGenDetection-train` | mixed | 0.224 | 277,643 (177,643 fake / 100,000 real) — 42 generators, 2022–2026 |
| **CommunityForensics-Small** | `OwensLab/CommunityForensics-Small` | mixed | 0.176 | 556,541 (278,445 / 278,096) — 4,782 generators |
| **OpenFake (selected)** | `ComplexDataLab/OpenFake` `core/train` | mixed | 0.128 | 439,523 (309,523 / 130,000) — 30 recall-ranked generators |
| **LAION-400M** | `jp1924/Laion400m-1` (gated) | real | 0.128 | 199,998 web-crawl photographs, `min(w,h) > 512` |
| **GAS-Station v4** | `gasstation/gs-images-v4` | fake | 0.08 | 113,793 (15 model folders) |
| **GAS-Station v3** | `gasstation/gs-images-v3` | fake | 0.072 | 426,689 (19 model folders) |
| **Open Images V7** | CVDF S3 (val + test) | real | 0.072 | 167,055 photographs |
| **FLUX-Reason-6M** | `LucasFang/FLUX-Reason-6M` | fake | 0.04 | 320,000 local slice of ~5.9M FLUX.1-dev |
| **SID_Set** | `saberzl/SID_Set` | fake | 0.04 | 70,000 full-synthetic rows (real + tampered classes dropped) |
| **Frontier fakes** | `julienlucas/midjourney-dalle-sd-nanobananapro-dataset` | fake | 0.04 | 5,195 Midjourney / DALL·E / SD / Nano Banana Pro (upstream label inverted) |

Reals break down as NTIRE 31%, Community Forensics 24%, LAION 18%, OpenFake
(Pexels + ReLAION) 18%, Open Images 10% — five different capture and curation
pipelines, which is what keeps FPR flat when the real distribution shifts.

### Held-out evaluation sets — never reachable by the loader

| Set | Size | What it tests |
|---|---|---|
| **CommunityForensics-Eval** | 51,836 (25,918/25,918), 21 generators | the Pangram evaluation protocol, per-architecture breakdown |
| **OpenFake `core/test`** | 89,225 (45,697/43,528), 20 generators | unseen generators **and** unseen reals simultaneously — `gpt-image-1.5/2`, `nano-banana-pro`, `flux.2-klein-9b`, `z-image-turbo`, `midjourney-7`, `ideogram-2.0`, `recraft-v2/v3`, `sora-2`, `veo-3` vs DOCCI + ImageNet |
| **OpenFake `reddit/test`** | 36,227 (29,116/7,111) | in the wild — AI subreddits vs photography subreddits, provenance unknown |
| **MIRAGE** | 12,073 (10,682/1,391) | human-verified in-the-wild, incl. inpainting / face-swap / image-edit slices |
| **NTIRE 2026 val / val-hard / public test** | 10,000 / 2,500 / 2,500 | clean-vs-distorted and per-distortion robustness |
| **COCO val2017** | 5,000 reals | FPR-only, organisers' reference real half |
| **WikiArt & real-only folders** | — | FPR harness for digital art |

The loader refuses any path under `openfake/holdout_*`, `core/test`,
`reddit/test`, `comfor-eval`, or `coco val2017`, so a held-out image cannot
leak into training by a config mistake.

### Generated assets in this repo

- Weights are **not in git** (`*.pt` is gitignored). A TechJam `best.pt` is
  ~4.9 GB (model + EMA + optimizer). The released scoring checkpoint is
  [glennwuwu/seer](https://huggingface.co/glennwuwu/seer); `predict.py`
  downloads it on first run. Discovery: `$SEER_CHECKPOINT` → repo-root
  `best.pt` → newest `runs/*/best.pt`.
- [`client/eval/eval_step33500/`](client/eval/eval_step33500) — committed held-out
  suite at step 33,500 (`summary.json`, per-set JSONs, `run_suite.py`,
  plus a small WildFake DALL-E FN dump under `errors_dalle_advanced/`),
  bundled with the dashboard.
- [`docs/deliverables/heldout-eval-step27500.md`](docs/deliverables/heldout-eval-step27500.md)
  — earlier writeup of the same recipe at step 27,500 (metrics only; the
  pod-side JSON folder is not in this checkout).
- Heatmap PNGs rendered by `src/seer/heatmap.py` (matplotlib, `turbo`
  colormap at 0.55 alpha). The local / patch head is on both the hero
  checkpoint and `configs/seer_probe.yaml`.

### Licences

Community Forensics is CC BY-NC-SA 4.0; OpenFake is CC-BY-SA-4.0 with
non-commercial restrictions on the proprietary-generator subsets. This
project is a research/hackathon artifact and is non-commercial accordingly.
No third-party trademarks or copyrighted media are redistributed — the repo
ships code, configs, metrics, and heatmaps of publicly licensed datasets.

---

## 6. Results

Numbers below are from the committed suite at **step 33,500**
(`runs/seer_vitl/last.pt`, clean protocol, threshold 0.5):
[`client/eval/eval_step33500/`](client/eval/eval_step33500). An earlier writeup of the
same recipe at step 27,500 lives at
[`docs/deliverables/heldout-eval-step27500.md`](docs/deliverables/heldout-eval-step27500.md).
`best.pt` is a different snapshot — it follows balanced accuracy on the
*train-distribution* val slice, which saturates early, so it is not the
checkpoint behind these held-out numbers. Weights are not in git.

### Held-out suite — step 33,500, clean protocol, threshold 0.5

| Set | n (fake / real) | Macro acc | mAP | AUROC | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CommunityForensics-Eval | 51,836 (25,918 / 25,918) | **95.79%** | **99.65%** | 99.60% | 99.79% | 91.78% | **0.19%** |
| OpenFake `core/test` | 89,225 (45,697 / 43,528) | **97.27%** | 99.86% | 99.84% | 99.83% | 94.72% | **0.17%** |
| OpenFake `reddit/test` | 36,227 (29,116 / 7,111) | 88.80% | 99.19% | 96.91% | 99.36% | 79.70% | 2.11% |
| MIRAGE | 12,073 (10,682 / 1,391) | 86.47% | 99.07% | 93.34% | 99.18% | 77.90% | 4.96% |
| COCO val2017 (reals only) | 5,000 (0 / 5,000) | — | — | — | — | — | **0.06%** |
| NTIRE 2026 public test | 2,500 (1,300 / 1,200) | 91.23% | 95.69% | 96.77% | 93.33% | 89.38% | 6.92% |

Pangram Image — the commercial state of the art — reports 97.29% / 99.70%
(macro accuracy / mAP) on CommunityForensics-Eval. Seer is **1.50 points of
macro accuracy and 0.05 points of mAP behind it at 15.3% of the parameter
budget and on entirely public data**, with a lower FPR. Essentially the
whole gap is false negatives, concentrated on pixel-space diffusion and a
handful of stylized commercial generators.

Default `main.py eval` on OpenFake caps at 4,096 images
(`OPENFAKE_EVAL_MAX`); the committed OpenFake rows used `--max-samples 0`.

### Robustness — clean vs transformed (NTIRE public test, 2,500 images)

From the same step-33,500 JSON (`ntire_test.json`):

| Checkpoint | Clean acc | Distorted acc | Δ | Robust AUROC | Overall macro acc | mAP |
|---|---:|---:|---:|---:|---:|---:|
| **step 33,500** | **97.68%** | **84.64%** | **−13.04** | **92.28%** | **91.23%** | **95.69%** |

Distortion costs ~13 points of accuracy on the hardest public robustness
split while ranking quality holds (robust AUROC 92.28%) — i.e. the ordering
survives laundering better than the fixed 0.5 threshold does, which is the
signal that a threshold sweep is the cheapest remaining win.

### Error analysis — OpenFake `core/test`, step 33,500 (89,225 on-disk holdout)

Overall: **97.27%** macro acc, **99.84% AUROC**, **99.86% mAP**, 94.72%
recall at **0.17% FPR**.

**False negatives are structural, not random.** Concentrated on
stylized / illustrative generators:

| Weakest | Recall | | Strongest | Recall |
|---|---:|---|---|---:|
| `recraft-v3` (n=1,000) | 59.20% | | `illustrious` (n=6,694) | 99.79% |
| `halfmoon-4-4-25` (n=190) | 71.05% | | `seedream-v5.0` (n=372) | 99.73% |
| `ideogram-2.0` (n=282) | 75.89% | | `lumina-17-2-25` (n=543) | 99.63% |
| `frames-23-1-25` (n=250) | 78.00% | | `ernie-image-turbo` (n=687) | 99.42% |
| `midjourney-7` (n=3,586) | 84.91% | | `gpt-image-1.5` (n=5,573) | 98.55% |

`flux.2-klein-9b` is at 97.72%. The remaining hole is a stylistic family
(Recraft / Halfmoon / Frames / Ideogram), not frontier capability.

**False positives are rare.** On the held-out reals: DOCCI 0.37% FPR
(14,847) and ImageNet 0.06% FPR (28,681). Combined FPR is 0.17%.

**The trade-off we chose.** Precision is 99.83% and FPR is 0.17%; recall
is 94.72%. The model is deliberately biased toward never accusing a real
photograph, which costs recall on borderline fakes. Ranking is good
enough (mAP 99.86%) that a different operating point is one threshold
away — no retraining needed. The dashboard `/errors` page and
`main.py eval --error-dir` render each mistake with its patch heatmap.

---

## 7. Limitations and what we would improve

- **`best.pt` selection rule.** It follows balanced accuracy on the train-distribution val slice, which saturates early. The stronger held-out snapshots are later `last.pt` steps (27,500 writeup, 33,500 committed suite). Selecting on a held-out set (or a composite of NTIRE test + OpenFake test) is a one-line fix we would make first.
- **Fixed 0.5 threshold.** mAP stays ≥99% on the large held-out sets where in-the-wild accuracy is lower, meaning much of the remaining loss is threshold placement, not ranking. A per-deployment threshold sweep on `openfake_reddit` is the highest-value next step.
- **Stylized / illustrative generators.** Recraft v3 (59.20% recall at step 33,500), Halfmoon, Frames, Ideogram 2 — non-photographic aesthetics remain the blind spot. Fixing it is a data problem: rank and fetch those families the way we ranked OpenFake.
- **Image editing and face swap.** MIRAGE's inpainting / IP-OP / face-swap slices score 39–45% at the 27,500 writeup. The patch head is architecturally the right tool but was never trained on real inpainting data, only on synthetic composites.
- **Frontier API coverage.** GPT Image, Nano Banana, Grok, Riverflow are API-gated; the public mixture covers those *families* rather than the exact latest endpoints.
- **Training budget.** The hero recipe is 60,000 steps (batch 56 × 3 accum, effective 168); the committed numbers are from step 33,500 — just over halfway. The curves had not flattened.
- **Compute cost.** ViT-L at 512px is ~10 img/s on a 12 GB GPU. A distilled ViT-S/B student trained on the ViT-L's patch logits would be the obvious deployment path.
- **No tests, CI, licence file, or demo video in this checkout.** `pytest` is an optional `dev` extra in `pyproject.toml`; there is no `tests/` tree.

---

## 8. Repository map

```
main.py                  CLI: train | eval | infer | info
predict.py               Track 5 entry: image directory -> {image_path, pred} JSON
configs/                 seer_vitl_512 (hero mix) | seer_probe (same mix; patch head on)
                         seer_vitl_local (single CF stream) | seer_vits_debug (DINOv2-S @ 224)
src/seer/
  model.py               SeerDetector, dual heads, EMA, param groups, checkpoint I/O
  train.py               continuation-training loop, cosine+warmup, LLRD, bf16
  augment.py             ~35 distortion families + the benchmark perturbation tables
  data.py                MixtureDataset, BatchBuilder, composite training, prefetcher
  eval.py                metrics, per-architecture / per-distortion breakdowns, ErrorBank
  heatmap.py             patch logits -> overlay panels
  infer.py, optim.py, config.py, labels.py, paths.py, datasets_registry.py
scripts/                 dataset acquisition, OpenFake ranking, loader benchmarks
get_datasets.py          --list / --tier / --only acquisition plan
eval/eval_step33500/     committed held-out JSONs + run_suite.py
client/                  Next.js dashboard (/ /analyze /robustness /errors)
docs/                    DATA_MIXTURE.md, DELIVERABLES.md, held-out eval report
```

---

## 9. References

- Stajduhar & Emi, *Introducing Pangram Image Detection*, 2026
- Simeoni et al., *DINOv3*, 2025 — [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)
- Park & Owens, *Community Forensics*, CVPR 2025 — [arXiv:2411.04125](https://arxiv.org/abs/2411.04125)
- Gushchin et al., *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*
- Zhu et al., *GenImage*, NeurIPS 2023 — [arXiv:2306.08571](https://arxiv.org/abs/2306.08571)
- Bammey, *Synthbuster*, OJSP 2023
