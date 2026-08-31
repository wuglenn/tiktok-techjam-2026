# Held-out eval — `seer_vitl` step 27,500

Clean-protocol scores for the latest `runs/seer_vitl` snapshot
(`last.pt`, step **27,500**). NTIRE val / val-hard / public test were
skipped: those already run inside the training loop. Everything else on
disk under `/workspace/data` that the harness treats as eval is here.

Raw JSON + the suite log live next to the checkpoint:

```
runs/seer_vitl/eval_step27000/
  summary.json
  comfor_eval.json
  openfake_test.json
  openfake_reddit.json
  mirage.json
  coco_val2017.json
  suite.log
```

## Setup

| | |
|---|---|
| Checkpoint | `runs/seer_vitl/last.pt` (EMA weights) |
| Step | 27,500 / 60,000 |
| Backbone | `camenduru/dinov3-vitl16-pretrain-lvd1689m` |
| Resolution | 512 |
| Protocol | clean (no Pangram 1024/JPEG-q50, no named perturbation, no hflip TTA) |
| Threshold | 0.5 |
| Hardware | RTX 4090 24 GB, batch 32, 16 decode threads, prefetch 4 |
| Throughput | ~50 img/s CommunityForensics-Eval (parquet), ~98 img/s folder sets |
| Wall time | 43 min for 194,361 images |

Single-class buckets (one generator, or real-only COCO) have undefined
AUROC / mAP. For those rows the useful number is **recall** (fake-only) or
**FPR** (real-only), not macro accuracy.

## Headline

Precision / recall are on the fake class at threshold 0.5. Accuracy is
the raw correct rate (not class-balanced). COCO has no fakes, so
precision and recall are undefined.

| Set | n (fake / real) | Acc | Prec | Rec | Macro acc | mAP | AUROC | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CommunityForensics-Eval | 51,836 (25,918 / 25,918) | **95.65%** | **99.80%** | **91.48%** | 95.65% | 99.62% | 99.54% | 95.46% | **0.18%** | 8.52% |
| OpenFake core/test | 89,225 (45,697 / 43,528) | **97.12%** | **99.79%** | **94.58%** | 97.19% | 99.84% | 99.81% | 97.12% | **0.21%** | 5.42% |
| OpenFake reddit/test | 36,227 (29,116 / 7,111) | **83.65%** | **99.38%** | **80.16%** | 89.05% | 99.28% | 97.28% | 88.74% | 2.05% | 19.84% |
| MIRAGE | 12,073 (10,682 / 1,391) | **79.95%** | **99.08%** | **78.06%** | 86.26% | 99.02% | 93.02% | 87.32% | 5.54% | 21.94% |
| COCO val2017 (reals) | 5,000 (0 / 5,000) | **99.90%** | — | — | — | — | — | — | **0.10%** | — |

Pangram Image on CommunityForensics-Eval is **97.29% / 99.70%** (macro acc / mAP). We are
1.64 points of macro acc behind and 0.08 points of mAP behind, with a
lower FPR (0.18% vs the commercial pitch of “careful FPR control”). The
gap is almost all **false negatives**, concentrated on pixel-space diffusion
and a handful of frontier / in-the-wild generators.

## CommunityForensics-Eval

Pangram protocol: 21 generators, paired 1:1 with reals, streamed from the
local 413-shard dump at `$SEER_DATA_ROOT/comfor-eval`.

| Architecture | n | Acc | Prec | Rec | Macro acc | mAP | AUROC | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Commercial | 29,836 | 94.44% | 99.73% | 89.12% | 94.44% | 99.55% | 99.49% | 94.13% | 0.24% |
| GAN | 4,000 | 99.40% | 99.90% | 98.90% | 99.40% | 100.00% | 100.00% | 99.40% | 0.10% |
| LatDiff | 12,000 | 99.68% | 99.87% | 99.50% | 99.68% | 100.00% | 100.00% | 99.68% | 0.13% |
| Other | 2,000 | 99.90% | 100.00% | 99.80% | 99.90% | 100.00% | 100.00% | 99.90% | 0.00% |
| PixDiff | 4,000 | 86.70% | 99.93% | 73.45% | 86.70% | 98.84% | 98.66% | 84.67% | 0.05% |
| **All** | **51,836** | **95.65%** | **99.80%** | **91.48%** | **95.65%** | **99.62%** | **99.54%** | **95.46%** | **0.18%** |

Open generators (GAN / latent diffusion / Other) are essentially solved.
Commercial is the mass of the set and is where most of the 8.5% FNR sits.
Pixel-space diffusion is the weak family: ranking is still high (mAP
98.84%) but 26.6% of those fakes fall under 0.5.

## OpenFake `core/test` (`openfake_test`)

Held-out generators **and** held-out reals (DOCCI + ImageNet vs the
LAION / Pexels reals used in training). Full on-disk holdout, not the
4,096-image train-loop slice.

| | n | Acc | Prec | Rec | FPR |
|---|---:|---:|---:|---:|---:|
| Reals (DOCCI) | 14,847 | 99.64% | — | — | 0.36% |
| Reals (ImageNet) | 28,681 | 99.87% | — | — | 0.13% |
| Fakes (20 generators) | 45,697 | 94.58% | 100.00% | 94.58% | — |
| **All** | **89,225** | **97.12%** | **99.79%** | **94.58%** | **0.21%** |

Per-generator (fake-only buckets; precision is 1.0 because there are no
reals in the bucket, so accuracy equals recall):

| Generator | n | Acc | Prec | Rec | FNR |
|---|---:|---:|---:|---:|---:|
| illustrious | 6,694 | 99.88% | 100.00% | 99.88% | 0.12% |
| seedream-v5.0 | 372 | 99.73% | 100.00% | 99.73% | 0.27% |
| aurora-20-1-25 | 282 | 99.65% | 100.00% | 99.65% | 0.35% |
| lumina-17-2-25 | 543 | 99.63% | 100.00% | 99.63% | 0.37% |
| ernie-image-turbo | 687 | 99.27% | 100.00% | 99.27% | 0.73% |
| gpt-image-1.5 | 5,573 | 98.37% | 100.00% | 98.37% | 1.63% |
| flux.2-klein-9b | 8,249 | 97.36% | 100.00% | 97.36% | 2.64% |
| wan-video-2.5 | 1,174 | 97.19% | 100.00% | 97.19% | 2.81% |
| ernie-image | 315 | 96.51% | 100.00% | 96.51% | 3.49% |
| z-image-turbo | 12,634 | 95.29% | 100.00% | 95.29% | 4.71% |
| recraft-v2 | 282 | 93.62% | 100.00% | 93.62% | 6.38% |
| gpt-image-2 | 474 | 93.04% | 100.00% | 93.04% | 6.96% |
| veo-3 | 2,167 | 92.62% | 100.00% | 92.62% | 7.38% |
| nano-banana-pro | 386 | 91.71% | 100.00% | 91.71% | 8.29% |
| sora-2 | 557 | 89.77% | 100.00% | 89.77% | 10.23% |
| midjourney-7 | 3,586 | 83.32% | 100.00% | 83.32% | 16.68% |
| ideogram-2.0 | 282 | 75.89% | 100.00% | 75.89% | 24.11% |
| frames-23-1-25 | 250 | 75.60% | 100.00% | 75.60% | 24.40% |
| halfmoon-4-4-25 | 190 | 72.11% | 100.00% | 72.11% | 27.89% |
| recraft-v3 | 1,000 | **56.90%** | 100.00% | **56.90%** | **43.10%** |

The OOD real shift is cheap: 0.21% FPR on 43.5k unseen photographs. The
remaining hole is a small set of stylized / commercial generators —
Recraft v3, Halfmoon, Frames, Ideogram 2, Midjourney 7 — not the GPT
Image / FLUX.2 / Seedream mass.

## OpenFake `reddit/test` (`openfake_reddit`)

In-the-wild AI subreddits vs photography subreddits. Provenance unknown;
this is the closest thing on disk to “images people actually post.”

| | n | Acc | Prec | Rec | FPR |
|---|---:|---:|---:|---:|---:|
| Fake | 29,116 | 80.16% | 100.00% | 80.16% | — |
| Real | 7,111 | 97.95% | — | — | 2.05% |
| **All** | **36,227** | **83.65%** | **99.38%** | **80.16%** | **2.05%** |

Overall: macro acc **89.05%**, mAP **99.28%**, AUROC 97.28%, F1 88.74%.

Ranking stays high (mAP 99.28%) while the 0.5 threshold is conservative:
one in five wild fakes scores below it, and FPR rises from 0.2% on
curated reals to 2.1% on Reddit photography. A threshold sweep on this
split is the next lever if we want a deployable operating point.

## MIRAGE

Human-verified in-the-wild set (`MIRAGE-GROUP/MIRAGE` test split, 2 local
parquet shards). `source` codes are MIRAGE's own tags — they name the
construction pipeline, not a generator. Mixed-class rows (IID, OOD-R)
report both classes; the rest are fake-only so the column is recall.

| Source | n (fake / real) | Acc | Prec | Rec | Macro acc | FPR |
|---|---:|---:|---:|---:|---:|---:|
| RMG | 2,499 / 0 | 98.88% | 100.00% | 98.88% | — | — |
| PCRMG | 565 / 0 | 98.05% | 100.00% | 98.05% | — | — |
| T2I | 3,391 / 0 | 91.06% | 100.00% | 91.06% | — | — |
| IID | 883 / 798 | 83.70% | 94.85% | 72.93% | 84.27% | 4.39% |
| OOD-R | 609 / 593 | 78.20% | 90.26% | 63.88% | 78.40% | 7.08% |
| CB | 286 / 0 | 53.50% | 100.00% | 53.50% | — | — |
| TR | 427 / 0 | 47.07% | 100.00% | 47.07% | — | — |
| FS | 218 / 0 | 44.50% | 100.00% | 44.50% | — | — |
| IP/OP | 990 / 0 | 42.83% | 100.00% | 42.83% | — | — |
| IE | 814 / 0 | 38.94% | 100.00% | 38.94% | — | — |
| **All** | **10,682 / 1,391** | **79.95%** | **99.08%** | **78.06%** | **86.26%** | **5.54%** |

Tag decode (per the MIRAGE paper, arXiv:2508.13223 — the eight generation
patterns were built as 64 ComfyUI / Python pipelines over 53 models):

| Tag | Pipeline |
|---|---|
| `T2I` | vanilla text-to-image from generators held out of the ID split — CogView4-6B, Bagel, Wan2.1, HiDream, UniDiffuser (11 pipelines) |
| `RMG` | realistic model generation — full-body e-commerce model shots: a LoRA-tuned T2I model renders the person, real garments are composited in via segmentation + inpainting (16 pipelines) |
| `PCRMG` | pose-consistent model generation — RMG plus DWPose + ControlNet, so the generated model keeps the original photo's pose (16 pipelines) |
| `IP/OP` | inpainting / outpainting — segmentation- or random-box masks regenerated, or the canvas extended 1–1.5× and filled (8 pipelines) |
| `IE` | instruction-based editing — natural-language edits via Flux.1-Kontext, InstructPix2Pix, Bagel, Wanx-imageedit (4 pipelines) |
| `FS` | face swapping — faces exchanged between two real photos, then restored with GFPGAN / Real-ESRGAN (3 pipelines) |
| `CB` | background change — subject segmented out (BiRefNet / U2-Net / InSPyReNet), new background generated from the original caption (3 pipelines) |
| `TR` | virtual try-on — clothing transferred between two model photos via segmentation + local inpainting (3 pipelines) |
| `IID` | the benchmark's in-distribution test split — human-curated images plus ID-split T2I, from the sources the 20k training set drew from |
| `OOD-R` | the paper's OOD-C — expert-verified real *and* fake images curated from a platform source the ID split never saw |

Full-image synthesis is largely solved: T2I / RMG / PCRMG — 6,455 of the
10,682 fakes — recall at 91–99%. The hole is local edits: instruction
editing, inpainting/outpainting, face swap, try-on and background change
(IE / IP/OP / FS / TR / CB) sit at 39–54%, exactly the composite-like
edits the patch head is supposed to catch, but this pass is page-level
only. FPR lives on the two human-curated mixed slices — IID 4.39%,
OOD-R 7.08%.

## COCO val2017 (FPR-only)

Organisers’ demonstration-val reals. 5,000 JPEGs under
`$SEER_DATA_ROOT/coco-val2017`. No fakes, so precision, recall, AUROC,
mAP and F1 on the fake class are undefined.

| n | Acc | Prec | Rec | FPR |
|---:|---:|---:|---:|---:|
| 5,000 | **99.90%** | — | — | **0.10%** |

**5 false positives / 5,000 → 0.10% FPR.**

That matches CommunityForensics-Eval (0.18%) and OpenFake core (0.21%) and is well under
the OpenFake reddit / MIRAGE in-the-wild FPR. The detector is not
triggering on ordinary photographs.

## What this does *not* include

- **NTIRE val / val-hard / public test.** Last train-loop print at step
  26,000: `ntire_test` n=2,500, macro acc 90.97%, F1 91.12%, AUROC 96.47%,
  mAP 95.49%, clean 97.50%, distorted 84.45%, robust AUROC 91.55%. Re-run
  with `--dataset ntire_test` (and `ntire_val` / `ntire_val_hard`) if a
  frozen snapshot of those splits is needed next to this suite.
- Pangram **augmented** protocol (`--augmented`) and the
  `--perturbation all` robustness table.
- Synthbuster + RAISE, WikiArt, WildFake DALL·E — not on this volume.
- Error-panel dumps (`--error-dir`). The JSON files above are metrics
  only.

## Reproduce

```bash
export SEER_DATA_ROOT=/workspace/data
export PYTHONPATH=src

# one set at a time (main.py eval)
uv run python main.py eval --checkpoint runs/seer_vitl/last.pt \
    --dataset comfor_eval --batch-size 32 \
    --out-json runs/seer_vitl/eval_step27000/comfor_eval.json

uv run python main.py eval --checkpoint runs/seer_vitl/last.pt \
    --dataset openfake_test --max-samples 0 --batch-size 32 \
    --out-json runs/seer_vitl/eval_step27000/openfake_test.json

uv run python main.py eval --checkpoint runs/seer_vitl/last.pt \
    --dataset openfake_reddit --max-samples 0 --batch-size 32 \
    --out-json runs/seer_vitl/eval_step27000/openfake_reddit.json

uv run python main.py eval --checkpoint runs/seer_vitl/last.pt \
    --dataset mirage --batch-size 32 \
    --out-json runs/seer_vitl/eval_step27000/mirage.json

uv run python main.py eval --checkpoint runs/seer_vitl/last.pt \
    --dataset folders --real-dir /workspace/data/coco-val2017 --batch-size 32 \
    --out-json runs/seer_vitl/eval_step27000/coco_val2017.json

# or the one-load suite that produced these numbers
uv run python runs/seer_vitl/eval_step27000/run_suite.py
```

`--max-samples 0` on the OpenFake splits disables the 4,096-image
train-loop cap and scores the full holdout. CommunityForensics-Eval and MIRAGE read the
local parquet under `$SEER_DATA_ROOT/comfor-eval` and
`$SEER_DATA_ROOT/mirage`; they do not stream the Hub if those dumps are
present.
