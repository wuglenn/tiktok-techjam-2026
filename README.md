# Seer — a sub-2B-parameter AI-generated image detector

**TikTok TechJam 2026 · Track 5 — AI-Generated Content Detection**

- Writeup: [`project_description.md`](project_description.md)
- Deliverables: [`docs/DELIVERABLES.md`](docs/DELIVERABLES.md)
- Data mixture, licences, fetch commands: [`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md)

**Required README sections:**
[Introduction](#introduction) ·
[Setup](#setup) ·
[Reproducibility](#reproducibility) ·
[Limitations and next steps](#limitations-and-next-steps) ·
[Team member contributions](#team-member-contributions)

**Quick start**

```bash
uv sync --frozen

# score every image in a directory (or pass a single image path)
# writes preds.json and infer-style original|heatmap PNGs under output/
uv run python predict.py --image-dir ./images --out preds.json
```

---

## Introduction

Seer detects AI-generated images. For each image it returns
**P(AI-generated) ∈ [0, 1]** and a per-patch heatmap of where the AI
content is. Metrics are reported at threshold 0.5.

- DINOv3 ViT-L/16 backbone, fully fine-tuned (continuation training, not a
  frozen probe).
- Dual heads: a global MLP on `[CLS ; mean(patch tokens)]` for the image
  verdict, and a linear local head on every patch token for the heatmap.
- 305,233,922 parameters — 15.3% of the 2B budget.
- Trained on a weighted public mixture (~2.58M usable images) following
  Pangram's data strategy: generator diversity, frontier fakes,
  wild-simulation augmentation, composite overlays with patch labels.

```
              ┌─────────────────────────┐
 image ─────► │  DINOv3 ViT-L (~305M)   │  continuation training:
              │  self-supervised ViT    │  the whole backbone is fine-tuned,
              └───────────┬─────────────┘  not frozen behind a probe
        CLS token          patch tokens
              │                 │
     ┌────────▼───────┐  ┌──────▼────────┐
     │  global head   │  │  local head   │
     │  (MLP)         │  │  (per-patch)  │
     └────────┬───────┘  └──────┬────────┘
              │                 │
        P(AI image)        AI heatmap
```

Held-out results (step 33,500, committed in
[`eval/eval_step33500/`](eval/eval_step33500/)):

| Set | n (fake / real) | Macro acc | mAP | AUROC | FPR |
|---|---:|---:|---:|---:|---:|
| CommunityForensics-Eval | 51,836 (25,918 / 25,918) | **95.79%** | **99.65%** | 99.60% | **0.19%** |
| OpenFake `core/test` — unseen gens *and* unseen reals | 89,225 (45,697 / 43,528) | **97.27%** | 99.86% | 99.84% | **0.17%** |
| OpenFake `reddit/test` — in the wild | 36,227 (29,116 / 7,111) | 88.80% | 99.19% | 96.91% | 2.11% |
| MIRAGE — human-verified wild | 12,073 (10,682 / 1,391) | 86.47% | 99.07% | 93.34% | 4.96% |
| COCO val2017 (reals) | 5,000 (0 / 5,000) | — | — | — | **0.06%** |
| NTIRE 2026 public test | 2,500 (1,300 / 1,200) | 91.23% | 95.69% | 96.77% | 6.92% |

Reference points: Pangram Image reports 97.29% / 99.70% macro acc / mAP on
CommunityForensics-Eval; the CVPR 2025 Community Forensics baseline reports
89.3% / 98.7%. NTIRE breakdown: clean 97.68% / distorted 84.64%, robust
AUROC 92.28%.

---

## Setup

Python **≥ 3.10**. Node **≥ 20.9** for the dashboard. The Python
environment is managed by [uv](https://docs.astral.sh/uv/); [`uv.lock`](uv.lock)
pins every package version.

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# or: brew install uv
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies

```bash
uv sync --frozen                      # exact versions from uv.lock

hf auth login                         # training / data fetch only (gated datasets)
export SEER_DATA_ROOT=/path/to/data   # required before fetching or training

cd client && npm install              # dashboard only
```

`uv.lock` records two torch builds (CUDA 12.4 on Linux/Windows, macOS ARM
on Darwin); `uv sync` picks the right one per platform. Scoring needs no
Hub login — `predict.py` downloads
[glennwuwu/seer](https://huggingface.co/glennwuwu/seer) automatically.

### Environment and ports

| Variable | Default | Role |
|---|---|---|
| `SEER_DATA_ROOT` | `/workspace/data` if `/workspace` is writable | all dataset roots |
| `SEER_CHECKPOINT` | repo-root `best.pt`, else newest `runs/*/best.pt` | inference weights |
| `SEER_PYTHON` | `uv run python`, else `.venv` | interpreter for dashboard inference |
| `SEER_INFER_URL` | `http://127.0.0.1:8765` | persistent inference server |
| `HF_TOKEN` / `HF_HOME` | — / `$SEER_DATA_ROOT/hf_cache` | gated Hub access and cache |

| Port | Process |
|---|---|
| **3000** | `cd client && npm run dev` (Next.js) |
| **8765** | `client/scripts/seer_serve.py` (bound to 127.0.0.1) |

Weights are **not** in git (`*.pt` is gitignored; a TechJam `best.pt` is
~4.9 GB). The scoring checkpoint is
[glennwuwu/seer](https://huggingface.co/glennwuwu/seer); `predict.py`
downloads it on first use, or pass `--checkpoint /path/to/best.pt`. No
DINOv3 access? Train on an open backbone instead:

```bash
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set backbone=facebook/dinov2-large res=518
```

There is no `tests/` tree in this checkout.

---

## Reproducibility

### Downloading the data mixture

Full source list, licences, and commands:
[`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md). Everything lands under
`$SEER_DATA_ROOT`. The mixture is largely non-commercial (Community
Forensics is CC BY-NC-SA 4.0; some OpenFake subsets are non-commercial).

```bash
uv run python get_datasets.py --list          # full plan; downloads nothing
uv run python get_datasets.py --tier 1        # NTIRE train/val/test + COCO
uv run python get_datasets.py --only ntire-train ntire-val ntire-test coco-val2017 mirage
uv run python dataset_stats.py --tier 1       # remote metadata only, no images

uv run scripts/fetch_data.py comfor-small          # ~260 GB; --max-shards 30 for a slice
uv run scripts/fetch_data.py frontier-fakes        # MJ / DALL·E / SD / Nano Banana Pro (~3 GB)
uv run scripts/fetch_data.py flux-reason-6m --max-shards 8
uv run scripts/fetch_data.py sid-set --max-shards 16
uv run scripts/wire_gasstation.py --versions v3 v4
uv run scripts/download_laion400m.py --max-shards 20 --max-images 400000 --min-side 512
uv run scripts/download_open_images.py --workers 32 --max-gb 70
```

OpenFake is selected by measured difficulty, not name: rank every generator
by recall under the eval perturbations, then fetch inversely to recall
(defaults: 25k images below 0.70 recall, 15k below 0.95, 10k below 0.98,
nothing above).

```bash
uv run scripts/openfake.py probe --shards 3
uv run scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
uv run scripts/openfake.py fetch --from-rank $SEER_DATA_ROOT/openfake/rank.json \
    --labels fake real --tier 0.70=25000 0.95=15000 0.98=10000 \
    --cap-model pexels=80000 laion=50000
```

Held-out sets — never trained on; the loader refuses any path under
`openfake/holdout_*`:

```bash
uv run scripts/openfake.py holdout --config core      # openfake_test
uv run scripts/openfake.py holdout --config reddit    # openfake_reddit
```

Missing sources are dropped at train time, not fatal — training works on
a slice.

### Training

```bash
uv run python main.py train --config configs/seer_vitl_512.yaml
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set resume=runs/seer_vitl/last.pt               # continue a run
```

- `seer_vitl_512.yaml` is the hero config: A100/H100-class (batch 56 × 3
  accum, ViT-L @ 512, no grad checkpointing). 

### Parameters

All parameters and per-source weights:
[`configs/seer_vitl_512.yaml`](configs/seer_vitl_512.yaml).

| Parameter | Value |
|---|---|
| Backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated), pretrained |
| Resolution | 512 (32×32 patch grid) |
| Batch | 56 × 3 grad-accum = 168 effective |
| Steps | 60,000 |
| Optimizer | AdamW, lr 5.0e-5, head lr 1.0e-4, LLRD 0.8, weight decay 0.05, warmup 1,000 |
| Precision / EMA | bf16, EMA 0.999 |
| Loss | image BCE + per-patch BCE (weight 1.0, patch `pos_weight` balanced) |
| Data | 50/50 real/fake (`balance_labels`), ~2.58M usable images |
| Augmentation | wild-simulation, intentionally harder than the eval table (JPEG q5–90, WebP, blur, noise, downscale, jitter, DCT/resample/FFT extras) |
| Composites | 60% of samples, all four real/fake pairings, 1–5 overlays |

Mixture weights (per-source detail in the YAML and
[`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md)):

| Source | Class | Weight |
|---|---|---|
| NTIRE 2026 train — 42 gens 2022–2026, matched reals | mixed | 0.224 |
| CommunityForensics-Small — 4,803 open generators + paired reals | mixed | 0.176 |
| OpenFake — 30 generators selected by measured recall + LAION/Pexels reals | mixed | 0.128 |
| laion400m-1 | real | 0.128 |
| GAS-Station v4 / v3 | fake | 0.08 / 0.072 |
| Open Images V7 | real | 0.072 |
| FLUX-Reason-6M | fake | 0.04 |
| frontier-fakes — MJ / DALL·E / SD / Nano Banana Pro | fake | 0.04 |
| SID_Set — full-synthetic only | fake | 0.04 |

Override any field on the CLI: `--set key=value ...`.

### Inference

`predict.py` walks an image directory (JPEG / PNG / WebP / BMP / TIFF /
GIF, recursive) and writes a JSON array of `{image_path, pred}`:

```bash
uv run python predict.py --image-dir ./images --out preds.json
uv run python predict.py --image-dir ./images --device cpu --out preds.json  # no NVIDIA GPU
```

```json
[
  {"image_path": "images/photo_001.jpg", "pred": 0.0031},
  {"image_path": "images/render_014.png", "pred": 0.9994}
]
```

The same run also writes `main.py infer`-style two-panel PNGs (original |
per-patch AI heatmap) to `output/`; pass `--no-heatmap` for JSON only.

| Flag | Default | What it does |
|---|---|---|
| `--image-dir` | required | directory of images, or a single image |
| `--checkpoint` | auto | local `.pt`, else download [glennwuwu/seer](https://huggingface.co/glennwuwu/seer) |
| `--out` | `predictions.json` | output JSON |
| `--out-detailed` | off | richer JSON (label, size, heatmap path, run metadata) |
| `--heatmap-dir` | `output/` | infer-style original \| heatmap PNG per image (`main.py infer`) |
| `--no-heatmap` | off | JSON only; skip the panels |
| `--batch-size` | 16 | inference batch |
| `--workers` | 8 | decode threads |
| `--res` | checkpoint's | override input resolution |
| `--device` | cuda if available | `cuda` or `cpu` |
| `--no-recursive` | off | do not descend into subdirectories |
| `--limit` | 0 (all) | score only the first N images |
| `--threshold` | 0.5 | reporting threshold only |
| `--hflip-tta` | off | average the score over the horizontal flip |
| `--resume` | off | continue an interrupted directory run |
| `--quiet` | off | suppress progress |

Single image + heatmap via the training CLI:

```bash
uv run python main.py infer --checkpoint runs/seer_vitl/best.pt \
  --image suspect.jpg --out-dir out/
```

### Re-running the held-out suite

The published numbers are the committed JSONs in
[`eval/eval_step33500/`](eval/eval_step33500/), scored from the
step-33,500 `last.pt`. A local `best.pt` follows train-val balanced
accuracy — a different checkpoint; do not treat them as interchangeable.

| Artifact | What it is |
|---|---|
| [`eval/eval_step33500/`](eval/eval_step33500/) | committed suite, step-33,500 `last.pt` — **the published numbers** |
| [`docs/deliverables/heldout-eval-step27500.md`](docs/deliverables/heldout-eval-step27500.md) | earlier writeup of the same recipe, step 27,500 |
| `best.pt` | train-val snapshot; val saturates early, so not the file behind the table |

The committed driver is
[`eval/eval_step33500/run_suite.py`](eval/eval_step33500/run_suite.py);
it hardcodes the pod paths from that run. Equivalent commands:

```bash
export SEER_DATA_ROOT=/path/to/data
CKPT=runs/seer_vitl/last.pt

uv run python main.py eval --checkpoint $CKPT --dataset comfor_eval
uv run python main.py eval --checkpoint $CKPT --dataset openfake_test --max-samples 0
uv run python main.py eval --checkpoint $CKPT --dataset openfake_reddit --max-samples 0
uv run python main.py eval --checkpoint $CKPT --dataset mirage
uv run python main.py eval --checkpoint $CKPT --dataset folders \
    --real-dir $SEER_DATA_ROOT/coco-val2017 --out-json coco_fpr.json
uv run python main.py eval --checkpoint $CKPT --dataset ntire_test
```

- `openfake_test` / `openfake_reddit` default to 4,096 images; the
  committed rows used `--max-samples 0` (full set).
- Robustness sweeps: `--perturbation all` (15-level eval table: JPEG
  q∈{90,70,50,30}, blur σ∈{0.5,1,2}, resize 0.5×/0.25×, noise
  σ∈{0.02,0.05,0.10}, jitter ±20%, crop 80%, plus clean), `extra` (harder
  NTIRE-style, alias `hard`), or `all+extra`.
- Error analysis: `--error-dir runs/eval/errors --error-n 6` writes the
  most confident FP/FN panels with heatmaps, plus per-generator records
  in the metrics JSON.

---

## Dashboard

Next.js app in [`client/`](client/) — details:
[`client/README.md`](client/README.md).

```bash
uv run python client/scripts/seer_serve.py --checkpoint best.pt   # keep model in memory, :8765
cd client && npm install && npm run dev                          # http://localhost:3000
```

| Page | Shows |
|---|---|
| `/analyze` | upload images → P(AI) + per-patch heatmap; exports `seer_predictions.json` |
| `/robustness` | clean vs transformed table + charts, NTIRE leaderboard |
| `/errors` | most confident FP/FN with heatmaps |

Upload limits: 12 images / 40 MB each. Without a local or cached
checkpoint, `/analyze` runs in simulated mode.

---

## Limitations and next steps

- **No tests, CI, Docker, or LICENSE** in this checkout.
- **Checkpoint selection.** `best.pt` follows train-val balanced accuracy,
  which saturates early; the published numbers are from step-33,500
  `last.pt`. The hero run is 60,000 steps and was only just over halfway —
  the curves had not flattened.
- **Scoring weights live on Hugging Face, not in git.** Without a local or
  cached checkpoint, dashboard `/analyze` is simulated.
- **CommunityForensics PNG⇒fake confound.** Shards are sorted by label, so
  container format is a shortcut unless wild-simulation augmentation runs
  on both classes. The corpus is SD-derivative-heavy; the weighted mixture
  keeps it from dominating.
- **Still-open held-out gaps.** OpenFake `reddit/test` is 88.80%
  (in the wild, unknown provenance). Recraft v3 / Halfmoon / Frames /
  Ideogram 2 are the structural false-negative family. MIRAGE
  inpainting / IP-OP / face-swap slices were 39–45% at the 27,500 writeup
  — the patch head was trained on synthetic composites, not real edits.
- **Frontier APIs** (GPT Image, Nano Banana, Grok, Riverflow) are
  API-gated; the public mix covers those families, not the exact latest
  endpoints.
- **Licences.** The mixture is largely non-commercial — see
  [`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md) before use outside this
  hackathon.
- **Compute.** ViT-L @ 512 is ~6–10 img/s on a 12 GB GPU; the hero run is
  A100/H100-class. No distilled student in this repo.
- **Fixed 0.5 threshold.** Ranking stays strong (mAP ≥ 99% on the large
  held-out sets) where in-the-wild accuracy is lower.
- **No face-swap / deepfake-specific head.** No demo video in this
  checkout (the Devpost YouTube link is not in the repo).

Given more time: select the scoring checkpoint on held-out sets (NTIRE
public test + OpenFake `core/test`); calibrate the threshold on
`openfake_reddit`; close the stylized-generator hole by fetching
inversely to measured recall; train the patch head on real edits; finish
the 60k run; ship a scoring-only checkpoint (model + EMA, no optimizer)
plus a demo video; add a `tests/` tree and CI; distill ViT-L → ViT-S/B
for deployment.

---

## Team member contributions

| Name (git author) | Email | What the history shows |
|---|---|---|
| **Glenn Wu** | wuglenn.wg@gmail.com | Training recipe and continuation FT, data mixture / OpenFake ranking, model and dual heads, eval harness and the step-33,500 suite, dashboard polish |
| **Jovan Tan** | joovanntan@gmail.com | Early detection-pipeline commit; later dashboard / website updates for submission |
| **Ethan Sim** | ethansim123@gmail.com | Error-analysis path on `main.py eval`, README / eval documentation, gated DINOv3 backbone config updates |

---

## Repo map

```
main.py                  CLI: train | eval | infer | info
predict.py               Track 5 entry: image directory → {image_path, pred} JSON
configs/                 seer_vitl_512 (hero) | seer_vitl_local | seer_probe | seer_vits_debug
src/seer/                model, train, eval, data, augment, heatmap, infer, paths
scripts/                 fetch_data, openfake*, download_*, wire_gasstation, ...
get_datasets.py          --list / --tier / --only acquisition plan
eval/eval_step33500/     committed held-out JSONs + run_suite.py
client/                  Next.js dashboard
docs/                    DATA_MIXTURE.md, DELIVERABLES.md, deliverables/
project_description.md  Devpost writeup
```

---

## References

- Stajduhar & Emi, *Introducing Pangram Image Detection*, 2026 (blog)
- Park & Owens, *Community Forensics*, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125))
- Bammey, *Synthbuster*, OJSP 2023 ([Zenodo](https://zenodo.org/records/10066460))
- Simeoni et al., *DINOv3*, 2025 ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104))
- Zhu et al., *GenImage*, NeurIPS 2023 ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571))
- Gushchin et al., *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*
