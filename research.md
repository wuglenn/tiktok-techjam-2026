# Research Notes — Track 5: Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026. Living document.

**Last updated:** 28 Aug 2026 (v2 — major revision after deep-dive research)

---

## TL;DR — the seven findings that should drive every decision

1. **This track is a near-clone of the NTIRE 2026 CVPR challenge, and the challenge dataset is public and ungated.** 277,650 labelled images, 42 generators spanning 2022–2026, with real/fake **distribution-matched on resolution, aspect ratio and JPEG quality**. Six shards of 50k, each with the same distribution, so **one 19 GB shard is a complete training set**. See §3.
2. **Augmentation severity is the causal driver of robustness — not backbone size, not architecture.** The natural experiment inside the NTIRE results proves it decisively. See §2.3.
3. **Freeze vs fine-tune is settled, and it depends entirely on whether your data is bias-controlled.** On biased narrow data, freezing wins. On aligned diverse data, end-to-end fine-tuning wins by **26.7 balanced-accuracy points**. We have the aligned diverse data, so we fine-tune. See §4.
4. **Frequency/artifact detectors are dead for this task.** NPR drops from 91.6 to **43.3 AUC** (below chance) under resize-0.7 + JPEG-70. SAFE hits **0.3% recall on fakes** under compound degradation. See §6.
5. **Under degradation, detectors don't produce false alarms — they collapse into predicting "real."** This inverts the usual threshold-tuning intuition and is the single best insight for the Error Analysis deliverable. See §7.
6. **Our six target transformations are approximately the tier the NTIRE organisers discarded as too easy.** Only JPEG and the crop/resize family survived into their final test sets. This is an opportunity, not a problem. See §2.4.
7. **The whole thing is affordable.** The strongest hackathon-feasible published recipe (GlobalForge) trains in **9 A100-hours on a single GPU at 224×224**. See §5.2.
8. **Published augmentation ranges do not cover our test grid.** DDA never sees JPEG below **Q55**; Community Forensics never below **Q75**, and is missing Gaussian blur, Gaussian noise and colour jitter *entirely*. Copying either recipe verbatim leaves us out-of-distribution on most of the six mandated families. See §3.4.3 and §3.5.4.
9. **Pair-synchronised augmentation is the anti-shortcut mechanism most teams get wrong.** If a real and its fake receive *different* random augmentation, the augmentation itself becomes a label signal. See §3.4.2.
10. **Generator *count* buys architecture-level transfer; image count saturates at ~27K.** Going from 3 to 3,333 generators (image count held fixed) gains **+9.6 mAP on GANs** and +6.7 on pixel diffusion — architectures never seen in training — versus only +2.4 in-domain. See §3.5.1.
11. **CommunityForensics-Small has a severe, undocumented confound that we measured directly.** "PNG ⇒ fake" alone scores **71.4% balanced accuracy**, and **every single fake image is 512², 256² or 1024²**, so "non-square ⇒ real" has 100% precision over ~77% of reals. See §3.5.3. This is the centrepiece of our shortcut-audit deliverable.

---

## 0. Constraints and logistics (verified against official rules)

| Item | Value |
|---|---|
| Submission window | **29 Aug 2026 12:00 SGT → 1 Sep 2026 12:00 SGT (72 hours)** |
| Judging | 1–7 Sep · Finalists 8 Sep · Grand final @ TikTok SG 11 Sep · Winners 15 Sep |
| Model size cap | **< 2B parameters** |
| Team | 3–4 people |
| Compute | RTX 3080 (10 GB, ~8 GB free) local + rentable A100s |
| Prizes | 1st $15,000 SGD · 2nd $8,000 · 3rd $5,000 · 4th/5th $3,000 · People's Choice $500 |

**Rules details that matter**
- *"Projects must be either newly created... or, if the Project existed prior to the Submission Period, must have been significantly updated after the start."* → pre-downloading data and preparing the environment is fine; write the project code inside the window.
- Open-source use allowed with licence compliance **and** you must "enhance and build upon" it.
- *"All data used or processed should be deleted at the completion of the competition."*
- IP stays with the entrants; sponsor gets a non-exclusive licence for judging and promotion. **This is compatible with the DINOv3 licence** (see §4.3).
- Official rules list **four equally weighted** criteria: Technical Execution, Innovation & Problem Insight, Feasibility & Practicality, Impact & Relevance. Raw accuracy is a minority of the score.

**Deliverables:** Devpost description · public GitHub repo with **inference script (image dir → JSON with `image_path` + `pred`)** and comprehensive README · 3-min YouTube demo · **robustness evaluation summary** · **error analysis note**.

**Target transformations:** JPEG Q∈{90,70,50,30} · Gaussian blur σ∈{0.5,1.0,2.0} · resize 0.5×/0.25× down-then-up · Gaussian noise σ∈{0.02,0.05,0.10} · colour jitter ±20% brightness/contrast/saturation · centre crop 80%.

### Environment — VERIFIED WORKING (28 Aug 2026)

Local `.venv`: `torch 2.13.0+cu126` (CUDA OK, RTX 3080), `timm 1.0.28`.

**21 ungated DINOv3 checkpoints confirmed loadable via timm** — no HuggingFace approval needed:

```
vit_small_patch16_dinov3.lvd1689m        21 M
vit_small_plus_patch16_dinov3.lvd1689m   29 M
vit_base_patch16_dinov3.lvd1689m         86 M
vit_large_patch16_dinov3.lvd1689m        300 M   <- recommended
vit_huge_plus_patch16_dinov3.lvd1689m    840 M
vit_7b_patch16_dinov3.lvd1689m           6.7 B   <- EXCEEDS 2B CAP
convnext_{tiny,small,base,large}.dinov3_lvd1689m
vit_large_patch16_dinov3.sat493m         <- SATELLITE. Never use (see §4.2)
```

> **Correction to a claim you may encounter:** some sources state that timm does not host DINOv3 weights and that `transformers` + manual gating is the only route. That was true at release but is **no longer correct** — Meta's Patrick Labatut explicitly authorised ungated repackaging ("The model weights can be 'repackaged'... as long as the DINOv3 License is kept"), and the [timm DINOv3 collection](https://huggingface.co/collections/timm/timm-dinov3-68cb08bb0bee365973d52a4d) has been live since Sept 2025. Verified empirically on this machine.

timm note: original weights have all-zero QKV biases, so timm sets `qkv_bias=False`; `*_qkvb.*` variants keep the zero bias to match `transformers`.

---

## 1. Prior art landscape

| Source | Relevance |
|---|---|
| [Pangram Image](https://www.pangram.com/blog/introducing-pangram-image-detection) | Industry SOTA. DINOv3 backbone, **full** continuation-training, synthetic mirroring, heavy augmentation, composite training for heatmaps. 99.5% internal / 0.16% FPR on ReLAION |
| [NTIRE 2026, arXiv 2604.11487](https://arxiv.org/abs/2604.11487) | **The challenge that mirrors this track.** Dataset + 20 team recipes |
| [Simplicity Prevails, arXiv 2602.01738](https://arxiv.org/abs/2602.01738) | Frozen VFM linear probe beats specialised detectors; LoRA harms on biased data |
| [B-Free, CVPR 2025](https://grip-unina.github.io/B-Free/) | Bias-free training data; linear probe costs 26.7 bAcc |
| [DDA, NeurIPS 2025, arXiv 2505.14359](https://arxiv.org/abs/2505.14359) | Dual pixel+frequency alignment; the JPEG-restoration insight |
| [GlobalForge, arXiv 2607.14684](https://arxiv.org/html/2607.14684) | Local→global cue shift; **9 A100-hours**; RealDeg-Bench |
| [HEDGE/INTSIG, arXiv 2604.03555](https://arxiv.org/html/2604.03555) | NTIRE 4th; logit-space fusion, dual gating |
| [HiDA-Net, arXiv 2508.17346](https://arxiv.org/html/2508.17346v1) | Native-resolution tiling; JPEG-QF estimation head |
| [Community Forensics, CVPR 2025](https://arxiv.org/abs/2411.04125) | 4,803 generators |
| [Fake or JPEG?, arXiv 2403.17608](https://ar5iv.labs.arxiv.org/html/2403.17608) | JPEG/size dataset bias; +11 pp from debiasing |
| [Dissect and Prune, arXiv 2606.10309](https://arxiv.org/html/2606.10309) | Asymmetric collapse to "real" |
| [BIAS-ID, arXiv 2605.31153](https://arxiv.org/html/2605.31153v1) | Score-shift bias measurement |
| [PE-SPC, arXiv 2608.04935](https://arxiv.org/html/2608.04935v1) | Semantic prototype calibration; best published blur robustness |
| [FGTS, arXiv 2511.22471](https://arxiv.org/abs/2511.22471) | 2k images + 8k params → 92.6% GenImage; Fisher token selection |
| [What Truly Matters, arXiv 2507.10236](https://arxiv.org/pdf/2507.10236) | >1000 GPU-hour controlled ablation of every design axis |

---

## 2. NTIRE 2026 — the challenge that mirrors this track

### 2.1 Setup
108,750 real + 185,750 generated from **42 generators**, 36 transformation types, 511 registrants, 20 valid submissions. Ranked by **average Robust ROC-AUC** (AUC over the distorted half); Clean ROC-AUC reported for completeness. In each val/test split, **exactly half of the reals and half of the fakes** are degraded, so degradation carries no label signal.

AUC was chosen deliberately so that **no threshold calibration is required**. Nothing in the challenge rewarded calibrated probabilities.

### 2.2 Leaderboard — read the gap column

| Rank | Team | Avg Clean | Avg Robust | **Gap** |
|---|---|---|---|---|
| 1 | MICV | 0.9974 | **0.9723** | 0.025 |
| 2 | Ant International | 0.9972 | **0.9721** | 0.025 |
| 3 | TeleAI-TeleGuard | 0.9786 | 0.9251 | 0.054 |
| 4 | INTSIG (= HEDGE) | 0.9853 | 0.9130 | 0.072 |
| 5 | vincentlc | 0.9527 | 0.8730 | 0.080 |
| 6 | UESTC | 0.9729 | 0.8679 | 0.105 |
| 7 | Reagvis Labs | 0.9452 | 0.8603 | 0.085 |
| 8 | PSU | 0.9227 | 0.8408 | 0.082 |
| 9 | Shallow Real | **0.9953** | **0.8336** | **0.162** |

Ordering teams by the clean−robust gap almost exactly reproduces the ranking. **The winners are not better detectors; they are less-degraded detectors.**

### 2.3 The natural experiment — the most actionable finding in this document

Compare rank 9 against rank 5:

| | Shallow Real (9th) | vincentlc (5th) |
|---|---|---|
| Backbone | DINOv3-Large + LoRA r=32 α=64 | SigLIP2-giant-384 |
| Head | Multi-Aspect: CLS + 4 REG + AVG → 6144-d → MLP | **single linear layer** |
| Extras | dynamic resolution 384–1152, deep supervision on last 4 layers, supervised contrastive loss | **none** |
| Ensemble / TTA | — | **none** |
| **Augmentation** | **official pipeline, default severity** | **`distortion_prob=1.0`, up to 3 ops, `num_levels=5`** |
| Clean AUC | **0.9953** (3rd best overall) | 0.9527 |
| **Robust AUC** | **0.8336 (last)** | **0.8730** |

A vastly simpler model with *worse* clean accuracy beat a sophisticated one on robustness, and the only meaningful difference is the augmentation setting. Shallow Real had the third-best clean AUC in the entire competition and finished last of the top nine.

> **Implication for us:** spend the first hours on the degradation pipeline, not the architecture. Then note that both top teams **over-augment relative to the test distribution** — Ant's Level 4 uses a fixed **6** chained distortions when the evaluation pipeline maxes out at 5. Train harder than you test.

### 2.4 The transformations — and why our target list is the easy tier

The pipeline is a direct extension of **ARNIQA**'s `distort_images`. Composition algorithm:

```python
MEAN, STD = 0, 2.5
num_distortions = random.randint(1, max_distortions)          # challenge: 5
groups = random.sample(list(distortion_groups.keys()), num_distortions)   # DISTINCT groups
distortions = [random.choice(distortion_groups[g]) for g in groups]
probabilities = [exp(-((i - MEAN)**2) / (2 * STD**2)) for i in range(num_levels)]
probabilities = [p / sum(probabilities) for p in probabilities]
values = [np.random.choice(distortion_range[d][:num_levels], p=probabilities) for d in distortions]
# applied in order, clip(0,1) + float32 cast after each step
```

Severity comes from a **half-Gaussian over the level index**. Re-weighting that one distribution *is* the difficulty curriculum:

| Severity Gaussian | P(L1) | P(L2) | P(L3) | P(L4) | P(L5) | Used by |
|---|---|---|---|---|---|---|
| mean 0, std 2.5 | .293 | .270 | .213 | .143 | .081 | ARNIQA default; Ant "Mild" |
| mean 2.5, std 2.0 | .117 | .193 | .248 | .248 | .193 | Ant "Moderate" |
| mean 3.0 | — | — | — | — | — | TeleAI-TeleGuard |
| mean 3.5, std 1.0 | .001 | .021 | .152 | .413 | .413 | Ant "Heavy" |

**Per-level parameters** (from ARNIQA source; levels 1→5):

| Group | Transform | Parameter | Levels 1→5 |
|---|---|---|---|
| Blur | Gaussian Blur | σ, kernel `2·ceil(2σ)+1` | `0.1, 0.5, 1, 2, 5` |
| | Lens Blur | disk radius | `1, 2, 4, 6, 8` |
| | Motion Blur | radius, angle ~U(0,180) | `1, 2, 4, 6, 10` |
| Noise | White Noise | Gaussian var (RGB) | `0.001, 0.002, 0.003, 0.005, 0.01` |
| | White Noise (colour comp.) | Gaussian var (YCbCr) | `0.0001, 0.0005, 0.001, 0.002, 0.003` |
| | Impulse Noise | s&p density | `0.001, 0.005, 0.01, 0.02, 0.03` |
| | Multiplicative Noise | speckle var | `0.001, 0.005, 0.01, 0.02, 0.05` |
| Compression | JPEG | quality factor | `43, 36, 24, 7, 4` |
| | JPEG2000 | compression ratio | `16, 32, 45, 120, 170` |
| Colour | Colour Diffusion | LAB a/b blur, σ=1.5·amt+2 | `1, 3, 6, 8, 12` |
| | Colour Shift | green-channel spatial shift | `1, 3, 6, 8, 12` |
| | Colour Saturation 1 | HSV sat multiplier | `0.4, 0.2, 0.1, 0, -0.4` |
| | Colour Saturation 2 | LAB a/b multiplier | `1, 2, 3, 6, 9` |
| Brightness | Brighten | LAB-L curve, 0.5+amt/2 | `0.1, 0.2, 0.4, 0.7, 1.1` |
| | Darken | curve 0.5−amt/2 | `0.05, 0.1, 0.2, 0.4, 0.8` |
| | Mean Shift | additive constant | `0, 0.08, -0.08, 0.15, -0.15` |
| Spatial | Jitter | warp offset (imscatter, 5 iters) | `0.05, 0.1, 0.2, 0.5, 1` |
| | Non-eccentricity Patch | # 16×16 patches displaced | `20, 40, 60, 80, 100` |
| | Pixelate | strength, z=0.95−s^0.6 | `0.01, 0.05, 0.1, 0.2, 0.5` |
| | Quantization | # Otsu levels | `20, 16, 13, 10, 7` |
| | Colour Block | # random 32×32 patches | `2, 4, 6, 8, 10` |
| Sharpness/Contrast | High Sharpen | LAB-L unsharp, radius 3 | `1, 2, 3, 6, 12` |
| | Linear Contrast | curve [0.25−a/4, 0.75+a/4] | `0.0, 0.15, -0.4, 0.3, -0.6` |
| | Nonlinear Contrast | output offset | `0.4, 0.3, 0.2, 0.1, 0.05` |

JPEG level 5 is **quality 4**. Colour Saturation 1 level 5 is **negative** (hue inversion). Mean Shift L1 and Linear Contrast L1 are genuine no-ops, and level ordering is non-monotonic for several transforms.

**Which transformations actually hurt.** The paper never publishes a per-transform table, but the split composition *is* the organisers' difficulty ranking — they used their own pre-trained detectors to find the damaging ones and escalated those.

- **Dropped before the final test sets (judged too easy):** Gaussian Blur, White Noise, Colour Shift, Colour Jitter, Brightness Decrease, Linear Contrast, Colour Quantization, Multiplicative Noise, Motion Blur, Pixelation.
- **Survived into Test Private (consistently damaging):** JPEG, Lens Blur, Impulse Noise, Colour Saturation, Brightness Increase, RGB Channel Shift, Random Crop, Random Aspect Crop, Downscale.
- **Added *because* they broke the organisers' detectors:** neural compression (JPEG AI, Cheng2020), **stacked compression** (JPEG+JPEG, JPEG+JPEG AI, JPEG+JPEG2000), invisible watermark insertion, watermark-erasing adversarial attacks (WMForger), CLAHE, ISO Noise, Random Tone Curve, Perspective Transform, Shot Noise, Glass Blur.

> **This is a strategic opportunity, not a problem.** Our mandated six — JPEG, Gaussian blur, resize, Gaussian noise, colour jitter, centre crop — map almost exactly onto the *discarded* tier. Only JPEG and crop/resize survived NTIRE's cut. Practically this means **every competent submission will score well on the mandated table**, so it cannot differentiate us. Differentiation comes from going beyond it: compound chains, stacked heterogeneous compression, and neural codecs. That is exactly the "sharpness in addressing real-world degradation" the rubric rewards.

Albumentations covers ~10 of the 24 extra transforms cheaply: `RandomToneCurve`, `CLAHE`, `ISONoise`, `Perspective`, `RGBShift`, `GlassBlur`, `Downscale`, `RandomSizedCrop`, plus Poisson shot noise.

### 2.5 What the winners did

**MICV (1st, 0.9723).** Two-stream late fusion; stream 1 = committee of **four** DINOv3 backbones, stream 2 = **two**; each stream projects → MLP head → probability; final = arithmetic mean. Backbone variants never stated (the biggest reproducibility gap). Data in four tiers: open datasets (GenImage, WildFake, AIGIBench, CommunityForensics, So-Fake-Set) + self-generated from Qwen-Image/Z-Image/FLUX **covering T2I, img2img, editing and inpainting** + closed commercial APIs (Seedream, Kling, GPT-Image, Nano-banana-pro) + challenge data. Millions of samples. Hierarchical difficulty-aware augmentation, individual → multi-stage combinatorial. **Focal Loss γ=2.0 α=0.5.** AdamW, wd 0.02, LR 1e-5, linear warmup over the entire first epoch then cosine, **SWA** over final epochs. **10 epochs on 32×A100, ~8 h (~256 A100-hours).** Resolution: **random crop then resize to 512² at train; direct resize to 512² at inference** to preserve global context. No TTA, no calibration. Validation on a **fixed-seed static augmented 10k set**.

**Ant International (2nd, 0.9721).** Ablated eva02-large / eva-giant / siglip2-giant / DINOv3-7B and chose DINOv3-7B. Two fully fine-tuned **DINOv3-7B = 14B params** on B200s. **Violates our 2B cap — cite as the reason we chose differently.** Four-level offline augmentation (Clean / Mild 1–3 ops μ=0 σ=2.5 / Moderate 3–6 ops μ=2.5 σ=2.0 / Heavy fixed 6 ops μ=3.5 σ=1.0). Expert 1: levels 2–4 only, attention pooling, 512², AugMix m6-w3-d1, 1 epoch, LR 1e-6, EMA. Expert 2: levels 1–4, first-token pooling, 288², 10 epochs. **78.25 GB VRAM, 2.21 img/s on an A100.**

**TeleAI-TeleGuard (3rd, 0.9251) — the most copyable.** EVA-CLIP + LoRA on MHSA and FFN linear layers in every block. Added three distortions the organisers didn't provide after **inspecting the validation set**: Speckle Noise, Colour Cast, **Organic Moiré**; raised severity mean to 3. Core idea — **LoRA-based Pairwise Training**:

  L = L_CE(x, y) + α·L_KL(x, x̂) + β·L_MSE(f_x, f'_x̂),  **α = 0.5, β = 0.25**

Clean and distorted versions are fed **jointly in each batch**; distorted features pass through **an additional corrective feed-forward network** before MSE alignment (so it's `f_x` vs `FFN(f_x̂)`, not raw matching). Stated goal: robustness **without degrading clean performance**. 8×A800, 5 epochs, AdamW LR 2e-4 cosine. Their gap of 0.054 vs 0.080–0.162 for teams 5–9 suggests it works.

**INTSIG / HEDGE (4th, 0.9130).** Five models: M1–M3 = DINOv3-Huge @256² progressively continued (baseline → data expansion at reduced LR → **augmentation escalation from (3 ops, 3 levels) to (5,5)**), M4 = DINOv3-Huge @448², M5 = MetaCLIP2-Giant @378² with **LayerNorm + last-two-blocks only** and Focal Loss. Fusion:

  `Final = 0.7·[0.7·(0.75·M1 + 0.15·M2 + 0.1·M3) + 0.3·M4] + 0.3·M5`

**Logit-space fusion beats probability averaging by +1.08 Robust AUC and equal-weight by +2.06 Robust F1.** hflip TTA on M3/M4 only. Two hand-tuned gates (outlier suppression of M4; cross-route consensus correction at τ₁=8, τ₂=3, δ=2.5) worth only +0.04 AUC — skip. 8×H800, 20 epochs.

**vincentlc (5th, 0.8730).** SigLIP2-giant-384 → **global average pooling over final-layer patch tokens** → single linear layer. Explicitly compared CLS extraction, attention pooling, and multi-layer concatenation and found **GAP over patch tokens most robust and stable**. "Squish" resize to 384² **ignoring aspect ratio**, because RandomResizedCrop "may remove localized forensic cues." No ensemble, no TTA. **Highest value-per-line-of-code in the paper.**

**Reagvis Labs (7th, 0.8603) — the efficiency frontier.** Six branches loaded **sequentially**: CLIP-L/14+LoRA with a prototype-attention head, SigLIP-v2@384, **SRM+Bayar residual ForensicCNN**, EVA-02, EVA-02-fixed, and a SAM+CutMix continuation. Logit-space cascade fusion. **8-view degradation-aware TTA** including blur and JPEG perturbation, not just geometric. **~155 ms/image, peak VRAM < 4 GB on an H100** — 20× less memory than Ant for 0.112 less robust AUC.

**PSU (8th, 0.8408).** Seven encoders across three pre-training paradigms. VL encoders get **LayerNorm-only tuning = ~0.03% of weights**; DINOv2/ConvNeXt/EffNet fully frozen. Label-smoothed BCE ε=0.05, EMA α=0.9995. **Ensemble weights ∝ robust AUC, not clean accuracy.** Early stopping on **M = 0.7·A_robust + 0.3·A_clean** — cheap and worth adopting regardless of architecture.

### 2.6 Organisers' conclusions
Clean detection is near-saturated (>0.99). **Robustness is the differentiator.** Architectural convergence is total: every top-9 solution is an ensemble of pre-trained transformers; **no team won with a from-scratch forensic CNN or a frequency-domain method.** The only classical-forensics component anywhere is Reagvis's SRM branch at 0.15 weight. The problem "is not yet solved" — room remains in "design, training strategies and data curation."

---

## 3. Training data — the plan

### 3.1 NTIRE 2026 challenge dataset — VERIFIED PUBLIC AND UNGATED

All three repos verified `gated: False`, `private: False` on 28 Aug 2026.

| Repo | Contents | Size |
|---|---|---|
| [`deepfakesMSU/NTIRE-RobustAIGenDetection-train`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train) | **277,650 labelled images**, `shard_0.zip` … `shard_5.zip` | **114 GB** (~19 GB/shard) |
| [`deepfakesMSU/NTIRE-RobustAIGenDetection-val`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-val) | `val_images.zip`, `val_images_hard.zip`, **`val_labels.csv`, `val_hard_labels.csv`** | **4 GB** |
| [`deepfakesMSU/NTIRE-RobustAIGenDetection-test-public`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-test-public) | `test_images.zip`, **`test_labels.csv`** | **851 MB** |

> **Note:** the val and test READMEs still say "does not include labels" — that text is **stale**. `val_labels.csv`, `val_hard_labels.csv` and `test_labels.csv` are all present in the repo trees. Verify by pulling the CSVs (a few KB) before relying on them.

Shard structure: `shard_i/images/*.jpg` + `shard_i/labels.csv` (`0` = real, `1` = generated), image names are random 20-character strings. **The card states all shards have similar distribution and can be used separately** — so `shard_0` alone is a legitimate 50k training set with the full generator distribution. A ready-made PyTorch `Dataset` class is in the card.

**Why this dataset is uniquely right for us:**
- **42 generators, 2022–2026**, including Nano Banana 2, Seedream 5 Lite, GPT Image 1.5, FLUX-2 Max, Qwen-Image, Z-Image, Grok Imagine. Nothing else public covers 2026 generators.
- **Fakes are semantically paired with reals**: a VLM captions each real image, an LLM rewrites it into a T2I prompt. Kills content bias.
- **Resolution, aspect ratio and JPEG quality factor distributions are explicitly matched** between real and generated. This is the *opposite* of the confound that plagues every other benchmark (§7.1).
- Reals from CC12M, CommonPool, RedCaps — filtered from ~12M by resolution thresholding, CLIP dedup and VLM scoring (90% removed).
- Train split deliberately contains **zero proprietary and none of the best open-source generators**; the hard generators are held out to val/test. So the val/test splits are genuine unseen-generator tests.
- The **submission format is `image_name, pred`** with `pred` = probability of fake — near-identical to our required deliverable.

**Caveat: no licence tag on any of the three cards.** Flag this in the README and treat it as research use.

### 3.2 The other candidates

| Dataset | Size | Generators | Licence | Verdict |
|---|---|---|---|---|
| **NTIRE 2026 train** `shard_0` | **19 GB** | 42 (2022–26), matched | untagged | **Primary training source** |
| [CommunityForensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) | 260 GB total, **~186 parquet shards** | **4,803** | cc-by-nc-sa-4.0 | **Secondary — stream or pull ~20 shards (~28 GB)** |
| [DDA-Training-Set](https://huggingface.co/datasets/Junwei-Xi/DDA-Training-Set) | **113 GB** | 1 (SD 2.1 VAE) | **apache-2.0** | **Do not download — regenerate it in ~1 h.** See §3.4 |
| [DDA-COCO](https://huggingface.co/datasets/Junwei-Xi/DDA-COCO) (eval) | **4.3 GB** | 5 VAEs | apache-2.0 | **Get this.** The definitive "is my detector using causal features?" probe |
| [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | 140 GB, 249 parquet shards | ~1 (FLUX) | cc-by-4.0 | **Skip for training.** Worst value-per-GB |
| CIFAKE | 110 MB | 1 (SD-1.4) | check Kaggle | **Smoke test only** |
| WildFake `DALLE.zip` | 25.6 GB | DALL·E 2+3 | Apache-2.0 (re-uploader tag) | Only for the mandated validation set |
| [MIRAGE](https://huggingface.co/datasets/MIRAGE-GROUP/MIRAGE) | **1.31 GB** | many + in-the-wild | cc-by-nc-4.0 | 12,073 eval images. Great value |
| WildRF | ~1–3 GB | in-the-wild | — | Real Reddit/X/FB compression history. Best value-per-GB anywhere |
| [Synthbuster](https://zenodo.org/records/10066460) | 12.4 GB | 9 incl. DALL·E 3, Firefly, MJv5 | — | Uncompressed — *we* control degradation |
| [HiRes-50K](https://huggingface.co/datasets/Mu437/HiRes-50K) | 22.6 GB | 3 communities | cc-by-nc-4.0 | Resolution- and JPEG-matched pairs |

**Notes on the three the team asked about:**

*CommunityForensics-Small* — 278K generated from **4,803 generators** (~34× any prior dataset) + 278K real (FFHQ, VISION, COCO, Landscapes HQ). Parquet with embedded bytes and rich metadata: `model_name`, `architecture` (LatDiff/PixDiff/GAN/other), `prompt`, `real_source`, `subset`, `nsfw_flag`, `resolution`, `format`. **The per-image generator metadata is gold for the Error Analysis deliverable.** Small vs Base mean AP 0.943 vs 0.984, gap concentrated in Commercial (0.852 vs 0.985). Acknowledged bias: LAION-caption prompts skew to people; most generators are SD derivatives. **Never `load_dataset` it whole** (~600 GB with the arrow re-index) — stream, or fetch individual `data/HFCF_small_*.parquet` files.

*DDA-Training-Set* — MSCOCO reals with DDA-aligned synthetic counterparts, **all PNG on both sides**, reals cropped to multiples of 8 so VAE reconstructions align exactly. Licence **apache-2.0**, ungated. **Do not download it.** It ships as an 11-part split ZIP (`.z01`–`.z10` + `.zip`), which cannot be streamed, cannot be partially fetched, and needs ~226 GB peak disk. We can regenerate an equivalent subset in about an hour — see §3.4.

*SID_Set* — 240K published (train 210K / val 30K) at **140 GB**, reals from OpenImages V7, but the synthetic half is essentially one generator. 140 GB for ~1 generator is the worst trade on the table. Its unique asset is the 100K **tampered images with masks** for a localisation head — a stretch goal, not a day-1 dependency.

### 3.3 Recommended download plan

**Tier 1 — enough to build and validate everything (~45 GB, do this first, on the rented box):**

| Item | Size | Purpose |
|---|---|---|
| NTIRE `shard_0.zip` | ~19 GB | Primary training set, full generator distribution |
| NTIRE val (+ labels) | 4 GB | Labelled, degraded, unseen-generator validation |
| NTIRE test-public (+ labels) | 0.85 GB | Labelled held-out test |
| COCO val2017 | 0.78 GB | Mandated real half |
| WildFake `DALLE.zip` | 25.6 GB | Mandated fake half (DALL·E 3 = "Advanced") |
| MIRAGE | 1.31 GB | In-the-wild eval |
| CIFAKE | 0.11 GB | 5-minute pipeline smoke test |

**Tier 2 — generator breadth (+~50 GB):** NTIRE `shard_1`, CommunityForensics-Small (~20 streamed shards), Synthbuster + B-Free's extended Synthbuster (FLUX, SD 3.5), WildRF, HiRes-50K low-resolution buckets.

Do bulk pulls on the **rented A100 box** (multi-Gbps from HF), not the home connection.

---

## 3.4 DDA in depth — take the protocol, not the download

DDA is **NeurIPS 2025 Spotlight**. Code: <https://github.com/roy-ch/Dual-Data-Alignment>. Checkpoint: [`Junwei-Xi/Dual-Data-Alignment`](https://huggingface.co/Junwei-Xi/Dual-Data-Alignment) (`DDA_ckpt.pth`, 1.26 GB, apache-2.0, ungated — note the repo warns an earlier checkpoint was wrong, make sure you get the current one).

### 3.4.1 What DDA actually is — four operations, not three

"Pixel alignment" here does **not** mean semantic/caption alignment. It means *literal pixel-value alignment*, because the fake is a VAE reconstruction of the exact same photograph. Semantic alignment is a free by-product, not the mechanism.

| # | Operation | Domain | Probability |
|---|---|---|---|
| 1 | SD 2.1 VAE encode→decode of the real image | Pixel | always (offline) |
| 2 | JPEG-compress the **fake** at the **real's** estimated quality factor | Frequency/format | p = 0.5 |
| 3 | Full-image 2D-DCT mixup between real and fake | Frequency | p = 0.2, ratio ~ U(0, 0.8) |
| 4 | Pixel-space alpha mixup between real and fake | Pixel | p = 0.2, ratio ~ U(0, 0.8) |

Steps 3 and 4 are in the released `Training/data/datasets.py` but only step 4 appears in the paper (Eq. 3). ⚠️ There is an apparent **sign inversion between paper and code**: the paper's `r_pixel` weights the *real*, the code's `blend_factor` weights the *fake*. Under the code a "fake" sample can be up to 100% real content. Worth checking before reproducing.

### 3.4.2 The three transferable tricks — all free, all worth copying

**1. Pair-synchronised augmentation.** `ComposedTransforms` resets the RNG state so **a real and its fake receive byte-identical augmentation**. This is the whole anti-shortcut mechanism. Without it, augmentation *creates* new shortcuts (different JPEG history per class) instead of removing them. Most teams get this wrong, and it costs nothing to get right.

**2. Extreme synchronised resampling.** Every pair is additionally emitted at a **downscale factor ~ U(0.2, 1.0)** and an **upscale factor ~ U(1.0, 3.5)**, using resamplers drawn from {NEAREST, BOX, BILINEAR, HAMMING, BICUBIC, LANCZOS}. Our targets (0.5×, 0.25×) sit comfortably inside U(0.2, 1.0), and the six-kernel randomisation is what stops the model keying on one interpolator's ringing signature. **This is probably a larger contributor to DDA's post-processing robustness than the alignment itself.**

**3. Format matching as an alignment step, not an augmentation.** JPEG the fake at *its own real's* estimated quality factor. This makes JPEG history label-uninformative, so the model *cannot* learn a compression shortcut and therefore degrades gracefully when we compress at test time. The repo README is emphatic: *"This is not a generic augmentation. It is a required DDA alignment step."* Ablation shows p=0.5 beats p=1.0 (the model should see both JPEG and PNG fakes). Precomputed quality factors ship in `Training/MSCOCO_train2017.json` (118,288 entries, 3.3 MB — grab just this file).

### 3.4.3 ⚠️ Their augmentation does not cover our test grid

Analysis of the shipped quality-factor table found only **7 distinct values**: Q96 (69.9%), Q90 (29.3%), Q80 (0.8%), plus four singletons. Combined with `RandomJPEGCompression(quality 55–100, p=0.15)`, **the model never sees JPEG below Q55.**

| Target family | Our grid | DDA's training coverage | Verdict |
|---|---|---|---|
| JPEG | 90, 70, 50, **30** | Q55–100 only | **Q50 and Q30 are out-of-distribution** |
| Gaussian noise | σ 0.02, 0.05, **0.10** | single σ ≈ 0.216 (std 55 on 0–255) | **2× our hardest, nothing in between** |
| Resize | 0.5×, 0.25× | U(0.2, 1.0) sync'd, 6 kernels | ✅ well covered |
| Blur | σ 0.5, 1.0, 2.0 | GaussianBlur k=3/5, Median k=3, Motion k=5 @ p=0.2 | ~ borderline |
| Colour jitter | ±20% | `ColorJitter(0.4, 0.4, 0.4, hue=0.175)` | ✅ more aggressive than needed |
| Centre crop | 80% | `PadRandomCrop(336)` | ✅ covered |

**Fixes:** widen `RandomJPEGCompression` to `quality_lower=25`, and replace the fixed-σ noise with `σ ~ U(0.0, 0.12)` on the [0,1] scale. Two one-line changes that close two of six families.

> This generalises beyond DDA: **audit every published augmentation recipe against the actual test grid before adopting it.** This is a cheap, concrete "problem insight" point for the writeup.

### 3.4.4 Verified hyperparameters (read from code, not inferred)

DINOv2 **ViT-L/14** (~304 M), `forward_features()["x_norm_clstoken"]` → `nn.Linear(1024, 1)`. LoRA **r=8, α=1.0** on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2` (~3.15 M trainable, ~1.0%). **336×336**, `PadRandomCrop` at train / `CenterCrop` at val, **never resized**. Normalisation uses **CLIP** mean/std on a DINOv2 backbone — unusual, but keep it if using their checkpoint. AdamW lr **1e-4**, **wd 0.0**, cosine to 1e-7. Batch 16 × accum 4, but **each item yields 6 tensors** (real, fake, 2 resized reals, 2 resized fakes) → 96 images per forward. **1 epoch**, early stopping on bAcc every 10k iterations. Loss = **0.5·BCEWithLogits + 0.5·ContrastiveLoss** (pos_margin 0, neg_margin 1) — **the contrastive term is in the code but not the paper.** 8×V100.

Composition: **118,287 real (MSCOCO train2017) + 118,287 fake**, ~477 KB each. Reconstruction is 0.1792 s/image → 5.9 h for the full set on one GPU. ⚠️ arXiv **v2** of the same paper reports 11.8K/11.8K and 0.59 h with different benchmark numbers; the HF release is unambiguously the 118K version.

### 3.4.5 Regenerate instead of downloading

COCO train2017 is 19 GB of JPEG and the reconstruction is deterministic. A **20–30K-pair subset takes ~1–1.5 h on the 3080 or ~20–30 min on an A100**, versus a 113 GB download and 226 GB of disk. Given they trained for a single epoch with early stopping, 20–30K aligned pairs is plenty.

Recipe: centre-crop to the largest multiple-of-8 dimensions → normalise to [−1,1] → `vae.encode(x).latent_dist.sample()` (note: **samples** the posterior, so VAE noise is included) → ×0.18215, ÷0.18215 → `vae.decode()` → save **both** as PNG. VAE is `stabilityai/stable-diffusion-2-1` subfolder `vae`, fp16. Their ablation tested several VAEs and SD 2.1 won.

### 3.4.6 What DDA actually learns — and why it's an expiring asset

DDA-COCO per-VAE accuracy: SD2.1 (its training VAE) **99.7** → SD1.5-MSE 99.7 → SD1.5-EMA 99.3 → SDXL 95.0 → **SD3.5 68.1** → **FLUX.1 50.2 (chance)**.

> **DDA does not learn "AI-ness." It learns the transfer function of the Stable-Diffusion KL-f8 VAE decoder.**

Until ~2024 essentially every latent-diffusion image online — SD1.x, SD2.x, SDXL, every Civitai fine-tune, LCM, Turbo, ControlNet — passed through a KL-f8 decoder, so one decoder's fingerprint covered the whole in-the-wild fake distribution. That is why DDA is the only specialised detector to survive Chameleon and WildRF. It is a *narrow* skill that happened to have *broad* coverage.

The moat is eroding: FLUX, SD3.5 and most 2025+ DiT generators use 16-channel VAEs. On AIGI-Now (2025 commercial generators) **DDA averages 0.695 versus MetaCLIP2's 0.907**.

**The complementary reading is the design brief.** Frozen modern VFMs score **< 0.08** on VAE reconstruction while hitting 0.94 in-the-wild; DDA scores 0.95–0.99 on reconstruction but 0.695 on 2025 generators. **Near-orthogonal failure modes.**

### 3.4.7 The isolation ablation, and the backbone-coupling explanation

*VAE reconstruction alone* vs *+ DDA alignment* (present only in arXiv v2 — dropped from v3):

| Benchmark | VAE rec. only | + DDA | Δ |
|---|---|---|---|
| Chameleon | 62.8 | 74.3 | **+11.5** |
| EvalGEN | 84.2 | 94.0 | +9.8 |
| GenImage | 86.2 | 95.5 | +9.3 |
| WildRF | 88.3 | 95.1 | +6.8 |
| SynthWildx | 77.1 | 84.0 | +6.9 |
| DDA-COCO | 90.0 | 94.3 | +4.3 |
| DRCT-2M | 95.8 | 97.4 | +1.6 |

The alignment is worth **+5 to +11 points** on top of plain VAE reconstruction — real, but the bulk of the performance is already in "DINOv2 + LoRA on VAE reconstructions with heavy synchronised augmentation."

**Why DDA collapses on other backbones (0.53–0.75) — largely a LoRA confound, not a DDA property.** LoRA alone destroys modern VFMs on biased data: PE frozen-linear 0.959 → **0.635** at r=8. DDA *is* a LoRA method, so "DDA + PE = 0.559" versus "PE + LoRA = 0.635" means most of the gap from PE-Linear's 0.959 is attributable to unfreezing at all. Secondary factors: DDA's signal is **not linearly present even in DINOv2** (linear probe 65.0 vs LoRA 91.9), so it must be synthesised by rewriting attention/MLP; and DINOv3's Gram anchoring is explicitly designed to *smooth* dense patch features, which plausibly suppresses the sub-pixel texture residual DDA needs. Also the swap almost certainly reused DDA's DINOv2-tuned lr/rank/CLIP-normalisation.

### 3.4.8 AlignGemini — the authors' own follow-up validates a two-branch design

[arXiv 2512.06746](https://arxiv.org/abs/2512.06746), "Task-Model Alignment," same group. Two-branch **OR-fusion** of a semantically-supervised Qwen2.5-VL-7B and a **DINOv2 pixel expert trained on SD2.1 VAE reconstructions — i.e. DDA demoted to one branch of two.** In-the-wild average **91.8 ± 4.7 (+9.5 over DDA alone)**.

Their key §3 finding: **mixing semantic and pixel supervision inside a single model dilutes both; purity of supervision per branch is what works.** If we take one architectural idea from this entire body of research, this is it — and it independently validates the degradation-adaptive dual-branch design in §8.

---

## 3.5 Community Forensics in depth

CVPR 2025 ([arXiv 2411.04125](https://arxiv.org/abs/2411.04125), Park & Owens). Code: [JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics). Project page: [jespark.net](https://jespark.net/projects/2024/community_forensics/).

### 3.5.1 The generator-count scaling law — the core result

Total image count **held fixed at 100K**, only the number of distinct latent-diffusion generators varied, 10 random model subsets per point, 3K iterations. mAP:

| Generators → | 3 | 10 | ~33 | 100 | ~333 | 1000 | ~3333 | **Δ(3→3333)** |
|---|---|---|---|---|---|---|---|---|
| Latent Diffusion (in-domain) | .969 | .985 | .988 | .991 | .991 | .992 | .993 | **+0.024** |
| Commercial | .894 | .924 | .937 | .942 | .944 | .946 | .948 | **+0.054** |
| Other (Stable Cascade) | .868 | .895 | .898 | .908 | .912 | .923 | .918 | **+0.050** |
| **GAN (OOD architecture)** | .759 | .811 | .815 | .827 | .840 | .852 | .855 | **+0.096** |
| **Pixel Diffusion (OOD arch.)** | .738 | .765 | .779 | .788 | .800 | .805 | .805 | **+0.067** |

Adding more *latent-diffusion* generators improves detection of **GANs and pixel diffusion — architectures never seen in training — roughly 4× more than it improves in-domain latent diffusion.** Generator diversity buys architecture-level transfer, not just within-family coverage. Returns flatten past ~1,000 generators.

**Image count saturates far earlier than expected.** Varying image count at fixed model count:

| Images | ~1K | 3K | ~9K | 27K | ~81K | 243K |
|---|---|---|---|---|---|---|
| mAP, 1000 models | .812 | .848 | .885 | .918 | .931 | .935 |
| mAP, 10 models | .788 | .830 | .878 | .914 | .926 | .932 |
| Acc, 1000 models | .745 | .777 | .810 | .834 | .847 | .848 |
| Acc, 10 models | .732 | .765 | .793 | .820 | .823 | .826 |

Returns plateau around **27K images**. And note the accuracy gap between 1000-model and 10-model training (+0.022 at 243K) is consistently *wider* than the mAP gap (+0.003) — the authors read this as **generator diversity mainly improving threshold calibration rather than ranking quality.** Directly relevant to us: our deliverable is a confidence score and our error-analysis note is about operating points.

Architecture-family diversity (evaluated on commercial models): Systematic subset alone (1.9M images, all latent diffusion) ≈ .935 mAP / .775 Acc; Manual subset (774K images, many architectures) ≈ .930 / .822; **both together ≈ .990 / .892.** Fewer than half the images spread across diverse architectures matches 1.9M same-family images, and the two are strongly complementary.

### 3.5.2 Composition and the released detector

4,803 generators in three tiers: **Systematic** (4,763 models / 1,919,493 images, scraped from HuggingFace `library=diffusers` in descending download order, ~403 images each, all latent diffusion); **Manual** (19 models / 774,023 images — StyleGAN2/3/XL, BigGAN, GigaGAN, ProGAN, ProjectedGAN, GANsformer, SAN, CIPS, StyleSwin, GLIDE, ADM, DeepFloyd, VQ-Diffusion, DiT, Latent Flow Matching, Taming Transformers); **Commercial** (11 models / 14,918 images — DALL·E 2/3, Ideogram V1/V2, Midjourney V5/V6, Firefly 2/3, FLUX.1-dev/schnell, Imagen 3), which are **evaluation-only**.

Training-set skew by model count: **latent diffusion 99.67%**, GAN 0.25%, pixel diffusion 0.06%. By image count: 73/22/3/2.

**Released detector:** plain **ViT-S/16** CLIP-pretrained (`vit_small_patch16_384.augreg_in21k_ft_in1k`), `Linear(384→1)` head, **trained end-to-end** — freezing the backbone consistently hurts, consistent with B-Free. **21.8M params** ([`OwensLab/commfor-model-384`](https://huggingface.co/OwensLab/commfor-model-384), MIT), which is **1% of our 2B budget**. AdamW, lr 2e-5, wd 1e-2, batch 512, cosine with 20% warmup, 52K iterations (~4.9 epochs). Preprocessing: `Resize(440) → RandomCrop(384)`, ImageNet normalisation (not CLIP, despite the CLIP backbone), **no test-time augmentation**.

Headline OOD results (mean over 6 eval sets): **Ours-384 mAP 0.986 / Acc 0.923**, vs GenImage 0.934/0.862, Ojha 0.760/0.646, Wang 0.648/0.552. On Synthbuster (unseen commercial generators) 0.974 vs 0.813 for the next best.

Robustness (Fig. 7, read off plots, mAP): JPEG Q36 → **0.920**; blur σ=2.0 → **0.940**; resize fraction 0.75 (≈0.25×) → **0.925**. The **384 model beats the 224 model at every operating point**, most starkly on padding (0.940 vs 0.830). Notably blur robustness holds up *despite blur never appearing in training*, suggesting the aggressive resize/interpolation augmentation transfers.

### 3.5.3 ⚠️ Measured confounds in CF-Small — undocumented, and large

Neither the paper nor any dataset card discloses a format or resolution confound. The paper in fact claims the opposite: *"we preserve the original image format whenever possible, without any additional compression or resizing... to mitigate potential bias."* Direct measurement of all 186 parquet shards says otherwise.

**Format:**

| | JPEG | PNG | P(PNG \| class) |
|---|---|---|---|
| Fake (278,445) | 6,216 | 272,229 | **0.978** |
| Real (278,096) | 125,095 | 153,001 | **0.550** |

> A classifier that ignores pixels entirely and outputs "PNG ⇒ fake" achieves **71.4% balanced accuracy** (TPR 0.978, TNR 0.450).

Because PNG is lossless and JPEG is not, this confound lives **in pixel space**, not just the header — the 8×8 DCT blocking of the JPEG reals is a genuinely learnable cue.

**Resolution is worse.** Every single fake image is one of three square resolutions:

| Fake resolution | Images | % |
|---|---|---|
| 512 × 512 | 231,409 | 83.11% |
| 256 × 256 | 39,386 | 14.14% |
| 1024 × 1024 | 7,650 | 2.75% |

Real resolutions are dominated by non-square natural sizes (640×480, 768×512, 682×512, 640×427…); the only square entry in the real top-12 is 1024×1024 (FFHQ). **The rule "resolution ∉ {512², 256², 1024²} ⇒ real" has 100% precision and covers ~77% of reals.**

Their pipeline only partially mitigates this. `Resize(440) → Crop(384)` destroys aspect ratio, so that rule isn't directly exploitable. But `min_augs=0` means ~1/3 of samples get no JPEG recompression, and **test-time applies no JPEG augmentation at all**, so at inference the format leak is fully exposed. The resolution confound survives as a **resampling-scale** confound: 512² fakes are scaled 0.859×, **256² fakes 1.719× (upscaled)**, 1024² 0.430×. So 14% of fakes get upsampled while almost no reals do — different resample factors leave different aliasing statistics.

**This is the centrepiece of our shortcut-audit deliverable.** It is measured, reproducible, undocumented by the authors, and exactly the kind of finding the "Innovation & Problem Insight" criterion rewards. The audit script that produced it is kept at `cf_probe.py`, with per-shard measurements in `cf_small_probe.json`.

### 3.5.4 ⚠️ Their augmentation covers one of our six families

`RandomStateAugmentation` with `rsa_min_num_ops=0, rsa_max_num_ops=2`. JPEG range is **Q75–100** only.

| Our target | Covered? |
|---|---|
| JPEG Q90 | ✅ |
| JPEG **Q70 / Q50 / Q30** | ❌ entirely below the training range |
| Gaussian blur σ 0.5–2.0 | ❌ (a `RandomGaussianBlur` class exists but is **not in the op list**) |
| Resize 0.5× / 0.25× then up | ⚠️ partial — `RandomResizeWithRandomIntpl` only ever resizes *above* crop size; RRC scale bottoms at 0.9 |
| Gaussian noise σ 0.02–0.10 | ❌ **absent entirely** |
| Colour jitter ±20% | ❌ (`RandAugment_bv` exists but is an unused path) |
| Centre crop 80% | ✅ via RandomCrop + RRC + RandomPadding |

Also, `min_augs=0` means **~1/3 of training samples are completely unaugmented**. Set `min=1, max=3`.

### 3.5.5 Practical access — the shards are sorted, not shuffled

This is the trap. Shard layout, measured:

| Shards | Contents |
|---|---|
| 0 – 69 | Systematic latent diffusion — **all fake** |
| 70 – 92 | Manual set (all the GAN / PixDiff / Other data) — **all fake** |
| 93 | mixed GigaGAN + FFHQ |
| 94 – 185 | FFHQ, VISION, COCO, LandscapesHQ — **all real** |

**Taking the first N shards gives a 100%-fake dataset**, and taking only low-index fake shards gives **zero GANs and zero pixel diffusion**. The card's streaming snippet (`buffer_size=3000`) is also misleading — 3,000 rows is exactly one shard, so you stream long runs of a single label. Interleave two class-pure streams instead.

Recommended subset — **stride 3, keeping all of shards 70–92**: 79 shards, **106.5 GB, 236,378 images**, full architecture coverage, roughly balanced. Read the parquet directly rather than via `load_dataset` to skip the Arrow re-index and halve disk use.

| Scheme | Shards | Size | Rows |
|---|---|---|---|
| Everything | 186 | 259.7 GB | 556,541 |
| stride 2 (+ all 70–92) | 105 | 143.2 GB | 314,175 |
| **stride 3 (+ all 70–92)** | **79** | **106.5 GB** | **236,378** |
| stride 4 (+ all 70–92) | 65 | 85.2 GB | 194,488 |
| naive uniform stride 4 | 47 | 64.8 GB | **loses architecture coverage** |

Since image-count returns plateau at ~27K, 236K is deep into diminishing returns — **we download for generator coverage, not volume.**

### 3.5.6 Two gaps to plan around

**CF-Small contains no Commercial subset at all** — zero DALL·E, Midjourney, FLUX, Ideogram, Firefly or Imagen. This isn't on the card, and it explains the reported drop:

| Trained on | GAN | Lat.Diff | Pix.Diff | **Commercial** | Other | Mean |
|---|---|---|---|---|---|---|
| Base (2.7M) | .995 | .996 | .947 | **.985** | .998 | .984 |
| Small (278K) | .986 | .995 | .888 | **.852** | .993 | .943 |
| Δ | −.009 | −.001 | −.059 | **−.133** | −.005 | −.041 |

**If the graders test commercial-generator output — which is likely — that 0.852 is our single biggest exposure.** Cheapest fixes: pull the 14,918-image `Commercial` split from the full repo (2 shards), or lean on NTIRE's proprietary-generator coverage, which is exactly what NTIRE's val/test splits hold out.

**Licence:** CF-Small is **cc-by-nc-sa-4.0 — non-commercial, and SA is viral.** The full `OwensLab/CommunityForensics` repo is **cc-by-4.0**. If TechJam's terms are a concern, build from the base repo's Systematic+Manual splits with our own reals. TechJam's IP terms (entrants retain IP, sponsor gets a non-exclusive judging/promotion licence) are probably compatible, but this is worth ten minutes now rather than at submission.

### 3.5.7 One genuine tension with DDA and B-Free

CF's Fig. 6(b) tested whether generated images must be paired with the real dataset used to prompt them, and found **all pairings differed only marginally** — concluding that strict source alignment is *not necessary at this data scale*. That directly contradicts B-Free (semantic alignment worth ~20 bAcc) and DDA (alignment worth +5 to +11).

The likely reconciliation is scale: with 4,782 generators and 2.7M images, no single content or format shortcut generalises well enough to dominate, so the model is forced onto real artifacts anyway. At B-Free/DDA scale (~50–120K images, one generator family) a shortcut *does* dominate, so alignment is essential. **We are at the small-data end, so we should follow B-Free/DDA and align — and CF's own measured confounds (§3.5.3) suggest their conclusion should not be over-read.**

---

## 4. Freeze vs fine-tune — the contradiction, resolved

This looked contradictory across sources. It isn't.

| Source | Finding |
|---|---|
| Simplicity Prevails | Frozen DINOv3 linear probe **0.914** on Chameleon; **LoRA r=8 → 0.718**. PE: 0.959 → **0.635** |
| FGTS | Fine-tuning DINOv3-L on 720k images for 10 epochs: So-Fake-OOD 75.03 → 75.09. **No gain** |
| **B-Free** | DINOv2+reg, same data: **linear probe 68.5 bAcc vs end-to-end 95.2. −26.7 points** |
| DDA | **LoRA r=8 on DINOv2**, best in-the-wild specialist (90.7 avg over 11 benchmarks) |
| GlobalForge | LoRA r=16 α=32 on Q/K/V, all 24 layers → SOTA |
| All NTIRE top teams | Full fine-tune or LoRA. **No frozen probe in the top 9** |

**The resolving variable is training-data bias, not the adaptation method.**

> Fine-tuning searches for whatever separates the two classes. On **biased** data (GenImage SD-v1.4: one generator, JPEG-vs-PNG, content-mismatched) the easiest separator is a shortcut, so fine-tuning finds the shortcut and overwrites the pre-trained knowledge — hence the collapse. On **aligned** data (B-Free, DDA, NTIRE) no shortcut exists, so the only learnable signal is the causal artifact and adaptation helps enormously.

Simplicity Prevails trained on GenImage SD-v1.4 with **zero augmentation**. B-Free trained on semantically-aligned self-conditioned pairs with heavy content augmentation. That's the whole difference.

**Decision: fine-tune, because our data is bias-controlled.** The NTIRE corpus is resolution-, aspect- and JPEG-matched with semantically paired fakes. Under those conditions every piece of evidence favours adaptation. Concretely: **LoRA r=16–32 on Q/K/V** (GlobalForge / TeleAI / Reagvis / Shallow Real all converged here; Shallow Real specifies α=64, a 2:1 ratio), or full fine-tune at backbone LR 1e-5 to 2e-5 with head LR 5e-4 if A100 budget allows.

Keep a **frozen linear probe as the day-1 baseline** — it trains in seconds on cached features, gives an immediate number, and makes a great ablation row showing why we moved past it.

### 4.1 Backbone scaling under the 2B cap — the uncomfortable table

FGTS, So-Fake-OOD, frozen features + 1k/1k linear probe:

| Model | Params | Avg acc | Δ vs 7B |
|---|---|---|---|
| DINOv3-S/16 | 21 M | 64.6 | −22.9 |
| DINOv3-B/16 | 86 M | 70.3 | −17.2 |
| DINOv3-L/16 | 300 M | 76.7 | −10.8 |
| DINOv3-H+/16 | 840 M | 77.8 | −9.7 |
| **DINOv3-7B/16** | **6.72 B** | **87.5** | — |
| DINOv2-S→H | 21 M→632 M | 56.4 → 61.4 | (flat — DINOv3-specific effect) |

There is a large discontinuous jump between H+ (840 M) and 7B. **Under 2B we top out at DINOv3-H+ ≈ 77.8, not 87.5.** Neither published hero configuration is safely under the cap: DINOv3-Linear is 6.72 B and PE-SPC is 1.88 B (and only if you count the vision tower alone).

**Two mitigations, both cheap:**
- These numbers are *frozen-probe*. INTSIG fine-tuned DINOv3-**Huge** to 0.9897 clean / 0.9130 robust, so adaptation closes much of the scale gap. This reinforces §4.
- **Fisher-Guided Token Selection helps most at exactly the mid scales we're forced into.** Score each of the 196 spatial positions on a 1k/1k reference set with `F_i = (μ_real,i − μ_fake,i)² / (σ²_real,i + σ²_fake,i)`, keep **top-K = 10**, average those vectors. Training-free, one pass over 2k images (~3 min). Gains: **+3.6 on ViT-L, +6.1 on ViT-H**, only +1.6 on 7B. Beats Random-K at every K.

Also from FGTS: **patch-token pooling beats CLS** — patch-only 74.0 > all tokens 73.6 > CLS 70.5 > register tokens 68.3. Register tokens are worst and including them hurts. vincentlc independently reached the same conclusion (GAP over patch tokens beat CLS, attention pooling, and multi-layer concat).

### 4.2 The satellite trap
Identical DINOv3 ViT-7B architecture, different pre-training data:

| Pre-training | GenImage | Chameleon real / fake / avg |
|---|---|---|
| Web (LVD-1689M) | 0.965 | 0.933 / 0.895 / 0.914 |
| Satellite (SAT-493M) | 0.706 | 0.948 / **0.121** / 0.535 |

Satellite-pretrained DINOv3 **cannot see fakes at all**. Forensic capability is entirely a pre-training-exposure artifact — Civitai/Liblib URLs in Common Crawl went from ~0 pre-2022 to >40,000 records/snapshot by late 2025. Great slide material, and a real trap since `vit_large_patch16_dinov3.sat493m` is one autocomplete away.

### 4.3 Licences — corrected

| Backbone | Params (vision) | Licence | Gated | In-the-wild acc* |
|---|---|---|---|---|
| DINOv3 ViT-L/16 | 300 M | DINOv3 Licence | **timm: no** | **0.940** |
| **PE-Core-L14-336** | **0.32 B** | **Apache-2.0** | No | 0.899 (and **0.978 on AIGIHolmes, best overall**) |
| PE-Core-G14-448 | 1.88 B | Apache-2.0 | No | — |
| SigLIP2 SO400M / Giant | 400 M / 1.16 B | Apache-2.0 | No | 0.822 |
| MetaCLIP2 Giant | 1.84 B | **cc-by-nc-4.0 — NON-COMMERCIAL** | No | 0.842 (best blur robustness: .932 at σ=2.0) |
| ConvNeXt-V2 | 198 M | **cc-by-nc-4.0 — NON-COMMERCIAL** | No | — |
| DINOv2-L | 304 M | Apache-2.0 | No | 0.636 — do not substitute naively |
| EVA-02-L | ~300 M | MIT | No | not benchmarked |

\* linear probe on frozen features, mean over Chameleon/WildRF/SocialRF/CommunityAI.

**MetaCLIP2 and ConvNeXt-V2 are NOT permissive** — both CC-BY-NC. **PE and SigLIP2 are the genuinely permissive strong options.**

The **DINOv3 Licence permits commercial use** (non-exclusive, worldwide, royalty-free, no revenue cap). Obligations: redistribute derivatives under the same agreement, acknowledge DINOv3 in published research, no reverse engineering, trade-control and no-military clauses. TechJam's IP terms (entrants retain IP, sponsor gets a non-exclusive judging/promotion licence) do **not** conflict. Meta can amend terms unilaterally (§8) — a residual risk worth one README sentence.

---

## 5. Robustness techniques, ranked by gain × cheapness

### 5.1 The ranking
1. **Backbone choice** — DINOv2/DINOv3 fine-tuned end-to-end. Self-supervised beats vision-language for this task, consistently.
2. **Crop, don't resize; train at high resolution.** SAFE's own cross-experiment: **73.4% → 95.8%** from swapping bilinear resize for random crop.
3. **Aggressive compound degradation augmentation, staged.** +11.15 AUC for DMID, +5.59 for SPAI in a controlled ablation.
4. **Clean↔degraded consistency loss.** Cheap; two NTIRE teams use it.
5. **hflip TTA + multi-crop + small ensemble.** ~+0.5–1 point each.
6. **Do not build on NPR/FreqNet/SAFE-style frequency features.** See §6.

Controlled evidence for augmentation ([What Truly Matters](https://arxiv.org/pdf/2507.10236), >1000 GPU-hours, everything else fixed):

| Method | No aug | With aug | Δ |
|---|---|---|---|
| DMID | 78.31 | **89.46** | **+11.15** |
| SPAI | 89.66 | **95.25** | **+5.59** |
| RINE | 93.16 | 94.90 | +1.74 |
| NPR | 67.36 | 67.48 | **+0.12** |

Augmentation helps end-to-end models hugely and artifact-specific detectors not at all. **If the model is built on a fixed frequency prior, augmentation cannot save it.**

Backbone comparison from the same study (RINE architecture, all else fixed):

| Backbone | Pre-train | Avg AUC |
|---|---|---|
| **DINOv2-L/14** | 142 M | **94.90** |
| BLIP2 | 129 M | 94.15 |
| CLIP L/14 | 400 M | 91.92 |
| CLIP H/14 | 2 B | 89.60 |
| OpenCLIP L/14 | 2 B | 83.56 |

Note the **anti-correlation with pre-training scale** — the two 2B-image CLIP variants are the worst. Their explanation: image-text alignment introduces semantic shortcuts and de-emphasises low-level detail.

### 5.2 GlobalForge — the hackathon-optimal published recipe

**9 A100-hours on ONE GPU at 224×224.** Full config:

| Field | Value |
|---|---|
| Backbone | DINOv2-L (frozen) + **LoRA r=16, α=32, dropout 0, on Q/K/V of all 24 layers** |
| Input | 224×224 |
| Optimizer | AdamW (0.9/0.999, ε 1e-8), wd 0.01 decay group / 0 no-decay group |
| LR | peak 4e-4 → min 2e-5, half-cycle cosine, 1 warmup epoch, fp32, no grad clipping |
| Batch | 128/GPU × 16 accum = **effective 2048** |
| Data | 80,000/epoch (40k real + 40k fake), DDA-aligned |
| Schedule | 10 scheduled / 8 logged epochs |
| **Cost** | **≈9 GPU-hours, 1×A100-80GB** |

(The DINOv3-L variant uses LR 5e-5 → 1e-6 on 4×A100 for ~24 GPU-hours.)

**Three components:**

- **Local Information Bottleneck (LIB)** — operates on *deep features, not pixels*. Reshape patch tokens to a spatial map, depthwise-convolve with a fixed `k=3, σ=1.0` Gaussian, then blend with **one learnable scalar** β: `X̂ = (1−α)X + α·X_smooth` where `α = sigmoid(β)`. β goes in the no-weight-decay group. ~10 lines.
- **Global Structural Reasoning (GSR)** — one self-attention block with an **inverted** mask: each query may attend only to *distant* tokens. Mask where Chebyshev distance ≤ `w_gsr = 3`, applied with `p_mask = 1.0`. Residual connection.
- **Degradation-aware Contrastive Structural loss** — symmetric InfoNCE between clean and compound-degraded views of the same image, on **average-pooled final-layer patch tokens**, `τ = 0.07`, `λ_dcs = 0.01`, added to label-smoothed BCE. Augmentation for the degraded view: JPEG Q∈(20,80); blur σ∈(0.5,1.5) with 7×7 kernel; colour jitter 0.4, hue 0.06.

**Ablation (RealDeg-1Step BAcc) — the ratio that matters for triage:**

| Config | BAcc |
|---|---|
| Plain LoRA fine-tune baseline | 80.03 |
| **Data augmentation only** | **82.56** |
| Pixel-domain blur instead of LIB | 81.39 |
| Input FFT low-pass instead of LIB | 80.12 |
| Unconstrained global attention instead of GSR | 80.73 |
| **Full LIB+GSR+DCS** | **86.87** |

Augmentation alone gets +2.5; the architecture adds +4.3 more. **Input-side low-pass filtering is actively harmful** (80.12 < 82.56) — don't "pre-clean" inputs.

Component ablation: removing DCS costs **3.20 BAcc on RealDeg** but only 1.99 in-the-wild — it is specifically a degradation regulariser.

**RealDeg-Bench** (7,353 images × 13 conditions = 95,589): JPEG Q∈{90,80,70,60,40}; blur σ∈{0.5,1,2,3,5}; resize s∈{0.9,0.7,0.5,0.3,0.2}; noise var∈{0.0005,…,0.01} (σ ≈ 0.022–0.1); brightness {−0.2,−0.1,0.1,0.2}; contrast {−0.3,−0.2,0.1,0.2}; saturation {0.6,0.8,1.3,1.5}; plus compound chains N∈{1..5} sampled **with replacement**. **Their ranges bracket our spec almost exactly — lift the operator pool verbatim.**

⚠️ **Do not cite GlobalForge's JPEG row as evidence.** Every method changes by <0.2 points from clean, which is impossible at Q=40 unless the source images were already compressed. Use HEDGE's HiRes-50K curves and B-Free's Table 11 instead.

**Blur and noise are the killers, not JPEG.** GlobalForge-d2 drops from 89.31 clean to **69.95 (blur)** and **70.37 (noise)**. The DINOv3 variant holds at 84.65/82.06. **If we only test JPEG we will badly overestimate our robustness.**

### 5.3 B-Free — the calibration lesson

DINOv2 ViT-L with **4 registers**, fine-tuned end-to-end, **504×504 crops, never resized**, ADAM lr 1e-6 wd 1e-6 batch 24, **1×A100**, early stopping on val bAcc. At inference: pad after patch embedding if smaller than 504, else **average logits over multiple 504² crops**.

Data: 51,517 MS-COCO reals (largest central crop → 512²) → 309,102 fakes via **SD 2.1 inpainting with an all-zeros mask** (`put_watermark` disabled — critical). Six fake variants per real.

**Augmentation ablation:**

| Method | Aug | AUC | bAcc | NLL | ECE |
|---|---|---|---|---|---|
| paired by text | — | 93.5 | 61.9 | 1.91 | .374 |
| reconstructed | — | 94.6 | 80.7 | 0.93 | .185 |
| self-conditioned | — | 94.7 | 81.4 | 0.53 | .158 |
| self-conditioned | cutmix/mixup | 95.3 | **78.6** | 0.66 | .197 |
| self-conditioned | inpainted | 98.0 | 92.2 | 0.18 | .064 |
| self-conditioned | inpainted+ | 99.0 | 95.2 | 0.14 | .040 |
| **self-conditioned** | **inpainted++** | **99.3** | **96.4** | **0.10** | **.038** |

Three lessons: semantic alignment is worth **~20 bAcc**; **CutMix/MixUp actively hurt** (81.4 → 78.6, ECE worse); and **AUC barely moves (94.7 → 99.3) while bAcc moves 15 points and ECE improves 4×**. Augmentation's main effect is **calibration** — making a fixed threshold actually work.

Architecture ablation: **linear probe 68.5 bAcc vs end-to-end 95.2**; registers add +4.1 over plain DINOv2; SigLIP e2e 89.9.

Data ablation vs the 8M-image D³ set, same architecture: equal AUC (99.0), **+7.7 bAcc, 3× better ECE, from 26× less data.**

Weights: `https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip`. **Licence is research/nonprofit only** — risky for a sponsored hackathon; use as a comparison baseline, not a shipped component. Their **viral-images dataset** (~1,400 images with multiple re-scraped versions each) is an excellent honest holdout.

### 5.4 PE-SPC — the best published blur robustness

Reinterpret the binary head as two **category prototypes** and initialise them from text:

```python
p1 = normalize(text_encoder("AI art"))        # fake prototype
p2 = normalize(text_encoder("a real photo"))  # real prototype
head = nn.Linear(D, 2, bias=True)
head.weight.data = torch.stack([p1, p2])
head.bias.data   = torch.tensor([1.0, 0.0])
# train exactly like a linear probe but with LR = 2e-5 (not 1e-3)
```

Image features must be **L2-normalised** and in the **CLIP projection space** (1280 for PE-G), not the raw vision-tower width. Discard the text encoder afterwards. **Zero inference cost.**

Chameleon under degradation — this is the headline:

| Detector | Q65 | σ=1.5 | σ=2.0 | Avg |
|---|---|---|---|---|
| DDA | 79.0 | 79.7 | 75.8 | 81.0 |
| SigLIP2-Linear | 82.8 | 68.9 | 67.1 | 79.7 |
| MetaCLIP2-Linear | 89.8 | 93.9 | 93.2 | 92.9 |
| PE-Linear | 92.1 | 83.1 | **77.8** | 91.0 |
| DINOv3-Linear | 89.1 | 89.7 | 89.1 | 90.7 |
| **PE-SPC** | **94.0** | **96.7** | **95.9** | **95.2** |

**+18.1 over PE-Linear and +6.8 over DINOv3-Linear at σ=2.0.** SPC eliminates PE's blur fragility. Gains on *minimum* accuracy are larger than on the mean — it raises the floor.

**Hard prerequisite:** SPC only works if the backbone already maps synthetic images to "AI generated" in Top-1 text-concept matching. MetaCLIP2 and PE do; **OpenCLIP and SigLIP2 do not, and get *worse* with SPC.** Prompt wording barely matters, but class-aligned forensic semantics do: swapping the two prompts costs **−26.2**, unrelated prompts cost −8 to −9 (worse than random init).

### 5.5 Consistency losses (most underused cheap trick)
- **TeleAI:** `CE + 0.5·KL(clean‖deg) + 0.25·MSE(f_clean, FFN(f_deg))`, clean+distorted jointly in each batch. ~15 lines, 2× forward passes.
- **GlobalForge DCS:** symmetric InfoNCE, λ=0.01, τ=0.07. Even lighter (no extra FFN). Isolated gain **+3.20 RealDeg BAcc**.
- **UESTC:** feature-level self-distillation — intermediate feature maps from the epoch-2 checkpoint become dense targets in stage 2.

### 5.6 Resolution and inference
Consensus is unanimous that **resizing to 224 destroys the signal**. HiDA-Net's frequency argument: downsampling centre-truncates the DFT and permanently discards high frequencies, whereas cropping multiplies by a window function and merely convolves the spectrum.

**SAFE's cross-experiment** (GenImage, rows = train preprocessing, cols = test):

| Train ↓ / Test → | Bilinear resize | Nearest | Random crop | Centre crop | Source |
|---|---|---|---|---|---|
| Bilinear resize | 73.4 | 88.1 | 91.5 | 91.2 | 90.3 |
| Nearest resize | 73.7 | 82.4 | 89.6 | 91.1 | 93.2 |
| **Random crop** | **84.1** | **91.0** | **95.8** | **95.6** | **96.2** |

Crop-trained wins in **every** test column, including when the test set is resized. Centre-crop ≈ random-crop ≈ source at test time, so a single centre crop suffices.

**TextureCrop** (WACV 2025, sliding-window patch selection by Global Histogram Entropy, 224² window, stride 224, **top-10 patches**, logit aggregation): **+12.1 BA vs resizing**, +4.3 vs centre crop, +3.4 vs TenCrop — and it's *cheaper*, 1.05 s → 0.63 s and 17 GB → 2.08 GB on a 2048² image. But it **hurt NPR by −2.63**, so verify per model.

Resolutions actually used: B-Free 504²; HEDGE 256²+448²; MICV 512²; Ant 512²+288²; vincentlc 384²; GlobalForge 224² (compensates architecturally).

**Live disagreement worth an ablation:** vincentlc's "squish" (direct resize ignoring aspect ratio) beat INTSIG's aggressive `RandomResizedCrop(scale=(0.08,1.0))` on robustness. MICV splits the difference — crop at train, full resize at test.

### 5.7 TTA and ensembling
- **Logit space beats probability space:** +1.08 Robust AUC, +1.06 F1 (HEDGE). Softmax saturates and hides branch disagreement.
- **Weighted beats equal-weight:** +2.06 Robust F1.
- **Weight by robust AUC, not clean accuracy** (PSU).
- **Backbone diversity > more of the same backbone** — Route C (one MetaCLIP2 against three DINOv3) gave HEDGE's largest single Robust F1 gain.
- **Degradation-aware TTA** (Reagvis, 8 views): flips, multi-scale centre crops, corner crops, **plus blur and JPEG perturbation**. Unusual and specifically aimed at robust AUC.
- Sequential model loading keeps 6 branches under **4 GB peak VRAM**.

---

## 6. Frequency/artifact detectors are dead here — the evidence

**B-Free Table 11** — simulated social upload = resize ∈[0.7,1.0] + JPEG Q∈[70,100]. Note how *mild* that is:

| Method | Original AUC | Simulated AUC | Δ |
|---|---|---|---|
| **B-Free** | 99.3 | **98.5** | **−0.8** |
| DMID | 97.3 | 94.4 | −2.9 |
| CoDE | 87.5 | 82.5 | −5.0 |
| AIDE | 85.5 | 67.3 | −18.2 |
| FatFormer | 68.2 | 48.7 | −19.5 |
| LGrad | 84.4 | 60.2 | −24.2 |
| LaDeDa | 91.7 | 51.9 | −39.8 |
| **NPR** | **91.6** | **43.3** | **−48.3** |

**NPR goes below chance — label inversion — from resize-0.7 plus Q70.** Those are exactly our spec's degradation levels.

**HEDGE RealChain (compound chains):**

| Method | Real Acc | **Fake Acc** | BAcc |
|---|---|---|---|
| SAFE | 99.3 | **0.3** | 49.8 |
| AIDE | 98.8 | **1.3** | 50.0 |
| FatFormer | 98.3 | 4.1 | 51.2 |
| NPR | 73.5 | 37.9 | 55.7 |
| DDA | 79.3 | 52.4 | 65.8 |
| REM | 85.3 | 83.0 | 84.2 |
| **HEDGE** | **98.7** | **87.8** | **93.2** |

**Three qualifications worth knowing:**
1. **Blur and resize hurt frequency detectors more than JPEG.** FreqNet's JPEG response is nearly flat but its blur response *inverts* (0.35 at σ=1 → 0.15 at σ=4, slope −0.067/σ, R²=0.97), and resize is its worst case.
2. **Phase survives what magnitude doesn't.** JPEG quantisation affects magnitude (`|F_quant| ≈ |F|/Q`) while **phase is unchanged** unless the coefficient is zeroed. A phase-based branch is a genuinely open, cheap idea.
3. **Not useless — useless *alone*.** Reagvis includes an SRM+Bayar residual branch at weight **0.15**, costing 5 ms and 0.1 GB. As a small low-weight ensemble member with heavy augmentation it adds diversity.

**Meta-point:** much of the apparent success of frequency detectors is a *dataset artifact* — reals are camera-JPEG, fakes are lossless PNG. Equalising compression improved JPEG robustness by **+13.26 accuracy at Q95, +8.75 at Q80, +4.49 at Q60**. **If real and fake differ in compression history, the validation numbers are fiction.**

---

## 7. Bias, shortcuts, and the error-analysis angle

### 7.1 The mandated validation set is confounded
COCO val2017 = 4,998 photographic JPEGs at ~640×480. "DALL-E Advanced" = the **DALL·E 3** subset of WildFake's cross-version taxonomy (confirmed via `label_csv_files/dalle3.csv`, whose 1,405,003 bytes ÷ ~159 bytes/row ≈ 8,843 rows), natively 1024×1024 / 1792×1024. **Label correlates with resolution, aspect ratio, format and quantisation tables.**

Two consequences: a model that learns nothing about generative artifacts can post a high AUC, and our local score will overstate true performance. We must equalise both classes — random-resized-crop to a fixed size, then re-encode both through the same randomised JPEG/WebP pipeline — before training *and* before local evaluation.

**Planned audit (~1 hour):** train a classifier on *metadata only* (format, resolution, estimated JPEG QF, bytes-per-pixel). If it scores high AUC, the benchmark is gameable. Then show our detector's advantage survives when the confound is neutralised. Cheap, sharp, and almost nobody will do it.

Access note: DALL·E has no per-subset archive — `Images/Diffusion_based/DALLE.zip` is a monolithic **25.6 GB** containing both Typical (DALL·E 2) and Advanced (DALL·E 3), on a China-hosted 1.29 TB ModelScope mirror. Try `remotezip` first (ZIP central directory is at the end; if ModelScope honours HTTP Range you can extract only `Advanced/`). Also note `Real/wukong.zip` is **164 bytes — broken**.

### 7.2 The asymmetric failure mode
Under degradation, detectors do **not** produce false alarms. They **collapse into predicting "real"** — SAFE 99.3% real accuracy against 0.3% fake accuracy. Detectors exploit compression artifacts as a spurious indicator of *realness* and overfit fragile generator shortcuts for *fakeness*; post-processing destroys the latter, so the model defaults to the real class.

> **Operational implication, and it's counterintuitive:** if we tune the threshold for low FPR on *clean* validation data, degradation will silently push recall toward zero while FPR still looks excellent. **Fit the threshold on a degraded validation split, and always report Fake Accuracy and Real Accuracy separately — never just balanced accuracy.** Prefer a **minimax** threshold (maximise the *worst* pipeline's accuracy) over an average-optimal one.

### 7.3 Metrics to adopt
- **AUC** — the headline, and what NTIRE scored.
- **Accuracy at a fixed 0.5 threshold** — B-Free's argument is that threshold-free metrics systematically overstate deployment readiness.
- **The AUC−bAcc gap is itself a bias diagnostic**: a large gap suggests the detector is exploiting bias rather than detecting artifacts.
- **Binary ECE, 15 bins**, class-rebalanced; **balanced NLL**.
- **Worst-case across transformation families**, not just the mean (DDA reports avg *and* min; min is the operationally meaningful number).
- **BIAS-ID score shifts** — mean score displacement per transformation. A detector can hold its AUC while its scores drift systematically toward one class, which breaks any fixed threshold. "The effects of bias are not always visible from inspecting performance metrics alone."

**Nobody in this literature uses post-hoc temperature scaling.** Calibration is achieved through training-time choices: focal loss (MICV γ=2 α=0.5), label smoothing (PSU ε=0.05), EMA/SWA, and augmentation (B-Free: ECE .158 → .038 purely from augmentation).

**Honesty note on low-FPR claims:** for a target FPR *p* at relative 95% half-width *r*, you need ≈ `4/(r²p)` benign samples. Estimating 0.1% FPR at ±25% requires **~64,000 real images**. If we quote a low-FPR operating point we should state the confidence interval.

---

## 8. Recommended approach

**Two branches with pure supervision, fused adaptively.** This is validated independently by AlignGemini (§3.4.8): mixing semantic and pixel supervision inside one model dilutes both, whereas per-branch purity plus fusion gains +9.5 in-the-wild.

| | Branch A — semantic/structural | Branch B — pixel/reconstruction |
|---|---|---|
| Backbone | DINOv3 ViT-L/16 (300 M, ungated timm) | DINOv2 ViT-L/14 (304 M) |
| Adaptation | fine-tune (LoRA r=16–32, α=2r, Q/K/V) | LoRA r=8, α=1.0, qkv/proj/fc1/fc2 |
| Data | NTIRE `shard_0` + CommunityForensics-Small | self-regenerated DDA pairs (20–30K) |
| Resolution | 256–384, crop at train | 336, `PadRandomCrop`, never resize |
| Detects | broad generator artifacts, structural anomalies | KL-f8 VAE decoder fingerprint |
| Blind to | VAE reconstruction (< 0.08) | 2025+ 16-channel-VAE generators (0.695) |

~0.6 B total — comfortably under the 2B cap, and both fit on the 3080 for inference.

**Branch A: fine-tune or freeze?** Run both — a frozen probe on cached features trains in seconds and makes an honest ablation row. The evidence favours fine-tuning *because our data is bias-controlled* (§4), but the LoRA-harms result is real enough that this should be decided by measurement on the frozen degraded validation grid, not by assertion. Note that the 0.940 frozen-probe headline uses DINOv3 **ViT-7B**, 3.5× over our cap, so it is not achievable by us either way. **Do not LoRA a PE backbone** — PE drops 0.959 → 0.635 at r=8.

**Free third ensemble member / warm start.** [`OwensLab/commfor-model-384`](https://huggingface.co/OwensLab/commfor-model-384) is **21.8 M params, MIT-licensed**, and already scores 0.986 mAP / 0.923 accuracy averaged over six OOD eval sets, with 0.920 mAP at JPEG Q36 and 0.940 at blur σ=2.0. That is 1% of our parameter budget for a strong, permissively-licensed baseline. Fine-tuning from it beats training from CLIP init and collapses compute from days to hours. At minimum it is the baseline row in our robustness table; most likely it is also a cheap third ensemble member.

**Data.** NTIRE `shard_0` primary (bias-matched, 42 generators, 2022–26) + CommunityForensics-Small for generator breadth (**stride-3 subset keeping all of shards 70–92**: 79 shards, 106.5 GB, 236 K images, full architecture coverage — see §3.5.5, and beware that shards are sorted by label) + self-regenerated DDA pairs as near-boundary hard negatives. Suggested share of the fake half: **~40% CommunityForensics / ~35% NTIRE modern DiT / ~25% DDA**. Keep DDA a minority — at 118 K it would swamp the gradient with one decoder's fingerprint. Hold out NTIRE val/test, DDA-COCO, and the mandated COCO + DALL·E 3 pair.

**Known coverage hole:** CF-Small has **zero commercial-generator images** (−13.3 mAP on that slice). NTIRE's val/test splits are exactly where the proprietary generators live, so NTIRE is our patch for this — but if the graders test Midjourney/DALL·E-style output, this is our largest single exposure.

**Training.** Label-smoothed BCE or Focal γ=2 α=0.5, plus **TeleAI pairwise consistency** (`0.5·KL + 0.25·MSE` through a corrective FFN), plus an optional **QFE-style degradation-estimation head**. EMA/SWA over final epochs. Early stop on `0.7·A_robust + 0.3·A_clean`.

**Augmentation — the highest-priority work item.** ARNIQA operator pool at `distortion_prob=1.0`, up to 5 chained ops from distinct groups, `num_levels=5`, staged severity curriculum (μ = 0 → 2.5 → 3.5). Three non-negotiables:
1. **Pair-synchronised** — a real and its fake get byte-identical augmentation (§3.4.2).
2. **Format-align every source before augmenting.** SID_Set reals are OpenImages JPEG and its fakes are PNG; DDA is PNG/PNG; CommunityForensics preserves original formats. Mixed naively, the fastest gradient is "PNG ⇒ fake." Decode everything to RGB, then **re-encode both classes through one identical random JPEG-quality draw.** Single highest-leverage line in the pipeline.
3. **Cover the actual test grid** — JPEG floor at Q25, noise σ ~ U(0, 0.12) (§3.4.3).

**Inference.** Crop at train; fuse a global resized view with multi-crop scores in **logit space**. hflip plus degradation-aware TTA views. Fuse the two branches with **degradation-adaptive weights** rather than static ones.

**The differentiators** (what wins on the rubric, beyond accuracy):
1. **Degradation-adaptive fusion** — use the estimated degradation level to re-weight a low-level branch against the semantic VFM branch, and to shift the threshold. Every NTIRE ensemble used *static* weights. Directly addresses the documented collapse-to-real.
2. **The shortcut audit** (§7.1) — prove the provided benchmark is gameable, then prove we don't game it.
3. **Score-shift reporting** (BIAS-ID) alongside AUC.
4. **Free patch-token heatmaps** — apply the head to individual patch tokens. ~20 lines on a ViT; Pangram sells this.
5. **Going beyond the mandated six transformations** into compound chains and stacked codecs, since the mandated six are the easy tier.

**Build order.** Evaluation harness first (`transformations.py` implementing the mandated six exactly, plus compound chains), then a visibly-collapsing baseline (NPR-style), then the frozen probe, then the fine-tuned model, then calibration and error analysis. Deliverables #4 and #5 fall out of step 1.

**Frozen-feature caching remains the iteration trick** for the probe baseline and all head/calibration experiments even though the final model is fine-tuned.

---

## 9. Decisions and justifications for the README

| Decision | Justification |
|---|---|
| Train on NTIRE 2026, not the three suggested datasets alone | Only public corpus with 2026 generators **and** real/fake matched on resolution, aspect ratio and JPEG quality |
| Don't train on CIFAKE | 32×32. All six target transformations are meaningless at that size; a 256× area reduction low-passes away the evidence. Reals and fakes also have *different* resampling histories, so the learnable signal is an antialiasing signature. Smoke test only |
| Skip SID_Set for training | 140 GB for ~1 generator (FLUX) is the worst value-per-GB available |
| Fine-tune rather than freeze | B-Free: linear probe costs **26.7 bAcc**. Valid *because* our data is bias-controlled — see §4 |
| Don't build on frequency/artifact detectors | NPR 91.6 → 43.3 AUC (below chance) under resize-0.7 + Q70; SAFE 0.3% fake recall under chains |
| DINOv3 over PE as primary | PE leans on high-frequency traces (77.8% at σ=2.0, 54.8% on recapture); DINOv3 captures structural anomalies that survive low-pass filtering |
| DINOv3 ViT-L over ViT-H+ | 300 M vs 840 M for +1.1 point frozen; leaves budget for a second backbone within 2B |
| Not 2× DINOv3-7B like Ant International | 14 B parameters — 7× over the cap |
| Report worst-case and per-class accuracy | Degradation causes collapse-to-real, invisible in balanced accuracy |
| No CutMix/MixUp | B-Free: 81.4 → 78.6 bAcc, calibration worsens |
| No input-side denoising/low-pass | GlobalForge: 80.12 vs 82.56 — actively harmful |
| Regenerate DDA rather than download it | 113 GB split ZIP, unstreamable, 226 GB peak disk, vs ~1 h of GPU time for an equivalent 20–30K subset |
| Two branches with pure supervision, not one mixed model | AlignGemini: mixing semantic + pixel supervision in one model dilutes both; per-branch purity + fusion = +9.5 in-the-wild |
| Keep DDA data to ~25% of fakes | DDA learns the KL-f8 decoder fingerprint (FLUX reconstruction = 50.2, chance). Useful, narrow, and expiring |
| Widened JPEG/noise augmentation ranges | Published recipes bottom out at Q55 (DDA) and Q75 (CF); CF omits blur, noise and colour jitter entirely. Our grid needs Q30, σ_noise to 0.10, blur to σ=2.0 |
| Never LoRA a Perception Encoder backbone | 0.959 → 0.635 on Chameleon at r=8 |
| Download CF-Small for **generator coverage**, not volume | Image-count returns plateau at ~27 K; generator-count returns run to ~1,000. 3→3,333 generators is worth **+9.6 mAP on unseen GAN architectures** |
| Stride-sample CF shards while keeping 70–92 | Shards are **sorted by label**; naive prefix download yields a 100%-fake set with zero GANs or pixel diffusion |
| Warm-start from `commfor-model-384` | 21.8 M params, MIT, 0.986 mAP OOD. Free strong baseline; training from scratch is the one clearly wasteful option |
| Format- and resolution-align every source before training | Measured on CF-Small: "PNG ⇒ fake" alone = **71.4% balanced accuracy**; all fakes are 512²/256²/1024² |

---

## 10. Open questions

- [ ] Confirm `val_labels.csv` / `test_labels.csv` contents (READMEs claim no labels; files exist)
- [ ] NTIRE dataset licence — untagged on all three repos
- [x] ~~Community Forensics generator-count scaling curve~~ — resolved (§3.5.1)
- [x] ~~DDA backbone-coupling~~ — resolved (§3.4.7): mostly a LoRA confound, plus the signal not being linearly present
- [x] ~~Whether CF parquet shards are grouped by generator~~ — resolved (§3.5.5): shards are **sorted by label**, 0–92 fake and 94–185 real
- [ ] Confirm the TechJam IP terms are compatible with CF-Small's `cc-by-nc-sa-4.0` (the full CF repo is `cc-by-4.0` if not)
- [ ] Measure the same format/resolution confound on SID_Set and on the NTIRE corpus (NTIRE claims to be matched — verify rather than trust)
- [ ] Verify the DDA paper/code sign inversion in `pixel_blend_mix` before reproducing (§3.4.1)
- [ ] Does SPC transfer to PE-Core-L/14, or is it G-specific?
- [ ] Intermediate-layer features on PE — PE's own paper argues its best embeddings "are not at the output of the network," and nobody has tried this for AIGC detection
- [ ] Confirm how organisers count the 2B cap (vision tower only, or shipped artifact incl. text tower)
