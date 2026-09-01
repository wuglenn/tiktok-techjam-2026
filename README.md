# Seer — a sub-2B-parameter AI-generated image detector

**TikTok TechJam 2026 · Track 5 — AI-Generated Content Detection**

Hackathon writeup: [`project_description.md`](project_description.md).
Official deliverable list: [`docs/DELIVERABLES.md`](docs/DELIVERABLES.md).
Mixture, licences, and fetch commands: [`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md).

**Required README sections:**
[Project overview](#project-overview) ·
[Setup and installation](#setup-and-installation) ·
[Reproducing results](#reproducing-results) ·
[Limitations and next steps](#limitations-and-next-steps) ·
[Team member contributions](#team-member-contributions)

**Score images now:** [Quick start](#quick-start-score-a-folder-of-images) · [`predict.py`](predict.py) · [weights](https://huggingface.co/glennwuwu/seer)

---

## Quick start: score a folder of images

The official Track 5 scoring script is repo-root [`predict.py`](predict.py).
It walks an image directory and writes a JSON array of `{image_path, pred}`,
where `pred` is **P(AI-generated) ∈ [0, 1]**.

Weights live on Hugging Face: **[glennwuwu/seer](https://huggingface.co/glennwuwu/seer)**
(`best.pt`, ~4.9 GB). The first run downloads them automatically.

```bash
# 1. install (needs [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10)
uv sync

# 2. score every image in a directory (or pass a single image path)
uv run python predict.py --image-dir ./images --out preds.json
```

That is the whole getting-started path. No extra Hub login is required for
scoring: the architecture config is bundled, and the fine-tuned weights come
from `glennwuwu/seer`. Later runs reuse the Hugging Face cache, a repo-root
`best.pt`, or `$SEER_CHECKPOINT` if you set one.

```json
[
  {"image_path": "images/photo_001.jpg", "pred": 0.0031},
  {"image_path": "images/render_014.png", "pred": 0.9994}
]
```

`pred` is a score, not a hard decision. Metrics in this repo are reported at
threshold 0.5. JPEG / PNG / WebP / BMP / TIFF / GIF are scanned recursively.

| Flag | Default | What it does |
|---|---|---|
| `--image-dir` | required | directory of images, or a single image |
| `--out` | `predictions.json` | official `{image_path, pred}` JSON |
| `--checkpoint` | auto | local `.pt`, else download [glennwuwu/seer](https://huggingface.co/glennwuwu/seer) |
| `--device` | cuda if available | `cuda` or `cpu` |
| `--batch-size` | 16 | inference batch |
| `--no-recursive` | off | do not descend into subdirectories |
| `--limit` | 0 (all) | score only the first N images |
| `--heatmap-dir` | off | per-patch AI heatmap PNG next to every verdict |
| `--out-detailed` | off | richer JSON (label, size, heatmap path, run metadata) |
| `--resume` | off | continue an interrupted directory run |

CPU example: `uv run python predict.py --image-dir ./images --device cpu`.
`uv sync` pins the CUDA 12.4 torch wheel; see [Setup](#setup-and-installation)
if that install does not match your machine.

---

## Project overview

Seer is a Track 5 AI-generated image detector: given a photo, it returns
**P(AI-generated) ∈ [0, 1]** and, when a local/patch head is present, a
per-patch heatmap of *where* the AI content is. The scoring entry is
repo-root [`predict.py`](predict.py) — it walks an image directory and
writes a JSON array of `{image_path, pred}`. A Next.js dashboard in
[`client/`](client/) covers the live demo (`/analyze`), robustness
summary, and error-analysis pages.

The model is a **DINOv3 ViT-L/16** backbone fully fine-tuned (continuation
training, not a frozen probe) with **dual heads**: a global MLP on
`[CLS ; mean(patch tokens)]` for the page verdict, and a linear local
head on every patch token for the heatmap. Measured total:
**305,233,922 parameters — 15.3% of the 2B budget**. Training follows
Pangram's public-data strategy: generator diversity, frontier fakes,
wild-simulation augmentation, and composite overlays so mixed real/AI
images have patch labels.

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

`pred` is a score, not a hard decision; metrics in this repo are reported
at threshold 0.5. Headline held-out numbers (step 33,500, committed in
[`eval/eval_step33500/`](eval/eval_step33500/)): **97.27%** macro acc on
OpenFake `core/test` (89k unseen images), **95.79%** on
CommunityForensics-Eval, **0.06%** FPR on COCO val2017. Those rows, how
to regenerate them, and how they differ from `best.pt` / the step-27,500
writeup are under [Reproducing results](#reproducing-results). Deeper
recipe (mixture, architecture, eval protocol, papers) is below the
required sections.

---

## Setup and installation

Python **≥ 3.10** (`.python-version` is 3.10). Node **≥ 20.9** for
`client/`. Install [uv](https://docs.astral.sh/uv/) first.

`pyproject.toml` pins torch to the **CUDA 12.4** wheel index
(`pytorch-cu124`). That is the only supported pinned path — **macOS and
CPU-only installs are not**. A CPU `predict.py` / dashboard pass can
still run if you already have a checkpoint and a working torch, but do
not expect `uv sync` to give you a macOS/CPU build.

```bash
# 1. Python env (CUDA 12.4 torch)
uv sync
# optional, only for scripts/generate_mirrors.py:
uv sync --group gen

# 2. Hugging Face auth — only for *training / data fetch*
#    (DINOv3 and jp1924/Laion400m-1 are gated). Scoring via predict.py
#    does not need this: it downloads glennwuwu/seer automatically.
hf auth login          # or: export HF_TOKEN=...

# 3. data root (required on macOS/Linux)
export SEER_DATA_ROOT=/path/to/data

# 4. dashboard (Node)
cd client && npm install
```

**`SEER_DATA_ROOT`.** If unset, `src/seer/paths.py` uses
`/workspace/data` when a writable `/workspace` mount exists, otherwise
**`F:/techjam`**. There is no `/workspace` default on a normal laptop —
export `SEER_DATA_ROOT` before fetching or training. `HF_HOME` defaults
to `$SEER_DATA_ROOT/hf_cache` once `seer.paths.setup()` runs.

No account for DINOv3? Use an open DINOv2 backbone instead
(`facebook/dinov2-large`, same ~300M class):

```bash
uv run python main.py train --config configs/seer_vitl_512.yaml --set backbone=facebook/dinov2-large res=518
```

**Weights are not in git** (`*.pt` is gitignored; a TechJam `best.pt` is
~4.9 GB, model + EMA + optimizer). The released scoring checkpoint is
**[glennwuwu/seer](https://huggingface.co/glennwuwu/seer)**. `predict.py`
downloads `best.pt` on the first run; you can also place it at the repo
root or export `$SEER_CHECKPOINT`. Dashboard discovery: `$SEER_CHECKPOINT`,
then repo-root `best.pt`, then the newest `runs/*/best.pt` (preferring
`seer_vitl*` runs). Training from scratch is under
[Reproducing results](#reproducing-results).

**GPU / disk.** The hero recipe (`seer_vitl_512.yaml`) is an A100/H100-class
run: batch 56 × 3 accum, ViT-L @ 512, no grad checkpointing. A single 12 GB
GPU should use `seer_vitl_local.yaml` (micro-batch 4 × 8 accum) and expect
~6–10 img/s. The usable mixture is **~2.58M images**; Community
Forensics-Small alone is ~260 GB, GAS-Station and OpenFake are much larger
if you fetch them in full. Missing sources are dropped, so you can train
on a slice. This is not a terabyte dump sitting in the repo — see
[`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md) and `get_datasets.py`.

There is no `tests/` tree in this checkout. Do not run `pytest` expecting
a suite.

### Environment and ports

| Variable | Default | Role |
|---|---|---|
| `SEER_DATA_ROOT` | `/workspace/data` if `/workspace` is writable, else `F:/techjam` | all dataset roots |
| `SEER_CHECKPOINT` | (discovery, see above) | dashboard / live inference weights |
| `SEER_PYTHON` | `uv run python`, else `.venv` | interpreter for `seer_infer.py` / `seer_serve.py` |
| `SEER_INFER_URL` | `http://127.0.0.1:8765` | persistent inference server |
| `HF_TOKEN` / `HF_HOME` | — / `$SEER_DATA_ROOT/hf_cache` | gated Hub access and cache |

| Port | Process |
|---|---|
| **3000** | `cd client && npm run dev` (Next.js) |
| **8765** | `client/scripts/seer_serve.py` (bound to 127.0.0.1) |

---

## Reproducing results

Three different things live in this repo: **(1)** the official scoring
script, **(2)** a dashboard demo, **(3)** committed held-out numbers.
Weights are required for (1) and for a *live* (2). The numbers in (3)
are already checked in as JSON — you do not need a GPU to *read* them.

### Scoring / inference (`predict.py`)

See [Quick start](#quick-start-score-a-folder-of-images) for the one-command
path. `--checkpoint` is optional: if you omit it (or pass `best.pt` and the
file is missing), weights are fetched from
[glennwuwu/seer](https://huggingface.co/glennwuwu/seer).

```bash
uv run python predict.py --image-dir ./images --out preds.json
uv run python predict.py --image-dir ./images --checkpoint best.pt --out preds.json
```

```json
[
  {"image_path": "images/photo_001.jpg", "pred": 0.0031},
  {"image_path": "images/render_014.png", "pred": 0.9994}
]
```

Flags:

| Flag | Default | What it does |
|---|---|---|
| `--image-dir` | required | directory of images, or a single image |
| `--checkpoint` | auto (`glennwuwu/seer`) | trained Seer checkpoint |
| `--out` | `predictions.json` | official `{image_path, pred}` JSON |
| `--out-detailed` | off | richer JSON (label, size, heatmap path, run metadata) |
| `--heatmap-dir` | off | per-patch AI heatmap PNG next to every verdict |
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

Single-image heatmap via the training CLI:

```bash
uv run python main.py infer --checkpoint runs/seer_vitl/best.pt \
  --image suspect.jpg --out-dir out/
```

### Dashboard demo

```bash
# 1. put best.pt at the repo root (or export SEER_CHECKPOINT=/path/to/best.pt)

# 2. keep the model in memory (otherwise /api/analyze
#    respawns client/scripts/seer_infer.py on every upload)
uv run python client/scripts/seer_serve.py --checkpoint best.pt
# listens on http://127.0.0.1:8765  (override with SEER_INFER_URL)

# 3. dashboard
cd client && npm install && npm run dev
# http://localhost:3000  →  /analyze
```

**Without weights, `/analyze` is SIMULATED** — deterministic fake
verdicts, labeled as such in the UI. Live mode needs a checkpoint *and*
a Python interpreter (`$SEER_PYTHON`, else `uv`, else the repo `.venv`).
Upload limits: **12 images / 40 MB each**. `/robustness` and `/errors`
scan eval JSONs from `eval/eval_step33500/` first, then `runs/eval/` and
`runs/`, and fall back to bundled demo data when none are present.

### Published numbers (which checkpoint is which)

| Artifact | What it is | Quote it as |
|---|---|---|
| [`eval/eval_step33500/`](eval/eval_step33500/) | committed suite from `runs/seer_vitl/last.pt` at **step 33,500**, clean protocol, threshold 0.5 | **the published numbers** |
| [`docs/deliverables/heldout-eval-step27500.md`](docs/deliverables/heldout-eval-step27500.md) | earlier writeup of the **same recipe** at step 27,500 | older writeup, not the committed suite |
| `best.pt` | whichever snapshot the train loop saved on **train-distribution val** balanced accuracy | a different checkpoint — val saturates early, so this is **not** the file behind the table |

| Set | n (fake / real) | Macro acc | mAP | AUROC | FPR | vs Pangram |
|---|---:|---:|---:|---:|---:|---|
| CommunityForensics-Eval | 51,836 (25,918 / 25,918) | **95.79%** | **99.65%** | 99.60% | **0.19%** | Pangram 97.29% / 99.70% |
| OpenFake `core/test` | 89,225 (45,697 / 43,528) | **97.27%** | 99.86% | 99.84% | **0.17%** | unseen gens *and* unseen reals |
| OpenFake `reddit/test` | 36,227 (29,116 / 7,111) | 88.80% | 99.19% | 96.91% | 2.11% | in the wild |
| MIRAGE | 12,073 (10,682 / 1,391) | 86.47% | 99.07% | 93.34% | 4.96% | human-verified wild |
| COCO val2017 (reals) | 5,000 (0 / 5,000) | — | — | — | **0.06%** | organisers' real half |
| NTIRE 2026 public test | 2,500 (1,300 / 1,200) | 91.23% | 95.69% | 96.77% | 6.92% | clean 97.68% / distorted 84.64%; robust AUROC 92.28% |

### Re-running the held-out suite

The committed driver is
[`eval/eval_step33500/run_suite.py`](eval/eval_step33500/run_suite.py).
It hardcodes the pod paths used for that run
(`CKPT=/workspace/tiktok-techjam-2026/runs/seer_vitl/last.pt`,
`COCO=/workspace/data/coco-val2017`). Point those at your checkout /
`$SEER_DATA_ROOT`, or use the equivalent `main.py eval` commands:

```bash
export SEER_DATA_ROOT=/path/to/data
CKPT=runs/seer_vitl/last.pt          # the step-33,500 snapshot, not best.pt

uv run python main.py eval --checkpoint $CKPT --dataset comfor_eval
uv run python main.py eval --checkpoint $CKPT --dataset openfake_test --max-samples 0
uv run python main.py eval --checkpoint $CKPT --dataset openfake_reddit --max-samples 0
uv run python main.py eval --checkpoint $CKPT --dataset mirage
uv run python main.py eval --checkpoint $CKPT --dataset folders \
    --real-dir $SEER_DATA_ROOT/coco-val2017 --out-json coco_fpr.json
uv run python main.py eval --checkpoint $CKPT --dataset ntire_test
```

Default `main.py eval` on OpenFake caps at **4,096** images
(`OPENFAKE_EVAL_MAX`). The committed OpenFake rows used
`--max-samples 0` (full set). Datasets expect the usual folders under
`$SEER_DATA_ROOT` (Community Forensics eval is streamed; OpenFake
holdouts via `scripts/openfake.py holdout`; NTIRE / COCO / MIRAGE via
`get_datasets.py` — see [`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md)).

Perturbation / robustness flags (not used for the clean table above):

```bash
uv run python main.py eval --checkpoint $CKPT --dataset ntire_test --perturbation all
uv run python main.py eval --checkpoint $CKPT --dataset ntire_test --perturbation extra   # alias: hard
uv run python main.py eval --checkpoint $CKPT --dataset ntire_test --perturbation all+extra
uv run python main.py eval --checkpoint $CKPT --dataset ntire_test --hflip-tta
uv run python main.py eval --checkpoint $CKPT --dataset comfor_eval --augmented            # 1024px + JPEG q50
```

`--perturbation all` is the 15-level eval table (JPEG q∈{90,70,50,30},
blur σ∈{0.5,1,2}, resize 0.5×/0.25×, noise σ∈{0.02,0.05,0.10}, jitter
±20%, crop 80%, plus clean). Full CLI catalog is under [Usage](#usage).

### Training from scratch (if you have the data)

Hero config: [`configs/seer_vitl_512.yaml`](configs/seer_vitl_512.yaml).
Effective batch **56 × 3 = 168**, **60,000** steps, resume from
`runs/seer_vitl/last.pt`. Set `SEER_DATA_ROOT` first. Do not expect this
README to list every shard — fetch pointers live in
[`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md) and `get_datasets.py`.

```bash
export SEER_DATA_ROOT=/path/to/data
uv run python get_datasets.py --list                      # prints the plan; downloads nothing
uv run python get_datasets.py --tier 1                    # NTIRE train/val/test + COCO
uv run python get_datasets.py --only ntire-train ntire-val ntire-test coco-val2017 mirage

uv run python main.py train --config configs/seer_vitl_512.yaml
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set resume=runs/seer_vitl/last.pt
```

`seer_vitl_local.yaml` and `seer_vits_debug.yaml` are **not** the hero
mixture — they fall back to a Community Forensics stream. Missing folder
sources are dropped at train time, not fatal. The mixture is largely
**non-commercial** (Community Forensics is CC BY-NC-SA 4.0; OpenFake's
proprietary-generator subsets are non-commercial).

### Weights

Not in git. The scoring checkpoint is
**[glennwuwu/seer](https://huggingface.co/glennwuwu/seer)** (`best.pt`).
`predict.py` downloads it automatically; you can also pass
`--checkpoint /path/to/best.pt` or export `$SEER_CHECKPOINT`. A live
dashboard uses the same file. The published table was scored from
step-33,500 `last.pt`.

---

## Limitations and next steps

### Current limitations

- **No tests, CI, Docker, or LICENSE** in this checkout. `pytest` is an
  optional `dev` extra in `pyproject.toml`; there is no `tests/` tree.
- **Checkpoint selection vs held-out numbers.** `best.pt` follows
  train-val balanced accuracy, which saturates early. The stronger
  snapshots are later `last.pt` steps (27,500 writeup, 33,500 committed
  suite). The hero recipe is 60,000 steps; those numbers are from just
  over halfway, and the curves had not flattened.
- **Scoring weights are on Hugging Face, not in git.**
  [`predict.py`](predict.py) downloads [glennwuwu/seer](https://huggingface.co/glennwuwu/seer)
  on first use. Without a local or cached checkpoint, the dashboard
  `/analyze` path is **SIMULATED**.
- **CommunityForensics PNG⇒fake confound.** CF-Small shards are sorted by
  label, so container format is a shortcut unless JPEG/WebP
  wild-simulation runs on *both* classes. The corpus is also
  SD-derivative-heavy; the weighted mixture is what keeps it from
  dominating.
- **Held-out gaps that are still open.** OpenFake `reddit/test` is 88.80%
  macro acc (in the wild, unknown provenance). Recraft v3 / Halfmoon /
  Frames / Ideogram 2 are the structural false-negative family on
  `core/test` (Recraft v3 recall 59.20% at step 33,500). MIRAGE
  inpainting / IP-OP / face-swap slices were 39–45% at the 27,500
  writeup — the patch head was trained on synthetic composites, not
  real edits.
- **Frontier APIs.** GPT Image, Nano Banana, Grok, Riverflow are
  API-gated; the public mix covers those *families* (NTIRE, OpenFake,
  GAS-Station, frontier-fakes), not the exact latest endpoints.
- **Mixture licences.** Largely non-commercial — see
  [`docs/DATA_MIXTURE.md`](docs/DATA_MIXTURE.md) before any use outside
  this hackathon.
- **Compute.** ViT-L @ 512 is ~6–10 img/s on a 12 GB GPU; the hero run
  is A100/H100-class. A distilled student is not in this repo.
- **Fixed 0.5 threshold.** Ranking stays strong (mAP ≥99% on the large
  held-out sets) where in-the-wild accuracy is lower, so some of the
  remaining loss is operating point, not ordering.
- **No face-swap / deepfake product.** Pangram's initial release does
  not ship that either; we did not add a face-specific head.
- **No demo video in this checkout.** Track 5 asks for a public YouTube
  link on Devpost; that URL is not in the repo.

### What we would improve given more time

- **Select the scoring checkpoint on held-out sets** (a composite of
  NTIRE public test + OpenFake `core/test`), not train-val — a one-line
  change that would have shipped `last.pt` @ 33,500 (or later) as
  `best.pt`.
- **Calibrate the threshold** on `openfake_reddit` (and report a sweep)
  instead of freezing 0.5. Cheap, and it matches the mAP-vs-accuracy gap.
- **Close the stylized-generator hole** the same way we ranked OpenFake:
  measure Recraft / Ideogram / Halfmoon / Frames recall and fetch
  inversely, rather than adding more SD-family volume.
- **Train the patch head on real edits** (MIRAGE-style inpainting and
  face-swap), not only synthetic composites, then re-score those slices.
- **Finish the 60k-step hero run** and keep evaluating past 33,500 —
  the curves were still moving.
- **Ship a scoring-only checkpoint** (model + EMA, no optimizer) so the
  Hub file is smaller than the current ~4.9 GB train blob, plus a short
  end-to-end demo video. `predict.py` already downloads
  [glennwuwu/seer](https://huggingface.co/glennwuwu/seer).
- **Add a real `tests/` tree and CI** around `predict.py` JSON schema,
  path defaults, and a tiny offline backbone (`main.py info --backbone tiny`).
- **Distill ViT-L → ViT-S/B** on the teacher's patch logits for a
  deployable student; the 2B budget still has ~6.5× headroom if an
  ensemble is the better use of that space.

---

## Team member contributions

| Name (git author) | Email | What the history shows |
|---|---|---|
| **Glenn Wu** | wuglenn.wg@gmail.com | Training recipe and continuation FT, data mixture / OpenFake ranking, model and dual heads, eval harness and the step-33,500 suite, dashboard polish |
| **Jovan Tan** | joovanntan@gmail.com | Early detection-pipeline commit; later dashboard / website updates for submission |
| **Ethan Sim** | ethansim123@gmail.com | Error-analysis path on `main.py eval`, README / eval documentation, gated DINOv3 backbone config updates |

---

## State of the field (July 2026)

Pangram Image is the current commercial SOTA. Their recipe, from the
[technical blog](https://www.pangram.com/blog/introducing-pangram-image-detection):

| Pillar | Pangram Image | Seer |
|---|---|---|
| Backbone | DINOv3, **full continuation fine-tuning** ("AI detection is not an ordinary downstream task") | same — DINOv3 ViT-L, full FT with layer-wise LR decay |
| AI data | synthetic mirroring (VLM caption → regenerate) + scraped real-world AI images | Community Forensics (4,782 gens on disk) + NTIRE (42 gens) + OpenFake's frontier/community generators, selected by measured recall + FLUX.1-dev + SID synthetic + GAS-Station v3/v4 |
| Real data | diverse web imagery; careful FPR control (WikiArt 0/2000, ReLAION 0.16% FPR) | Community Forensics + NTIRE matched reals + `jp1924/Laion400m-1` + Open Images V7 + OpenFake LAION/Pexels; WikiArt/folders FPR harness |
| Augmentation | strong "in the wild" simulation (crop, edit, compression) | train is **harder than the eval table** (see recipe). Eval `--perturbation all` stays JPEG q∈{90,70,50,30}, blur σ∈{0.5,1,2}, resize 0.5×/0.25×, noise σ∈{0.02,0.05,0.10}, jitter ±20%, crop 80% |
| Mixed images | composite training → heatmaps | same: cropped overlays in all four real/fake pairings, stacked multi-overlay, per-patch labels |
| Scale | proprietary scrape of frontier generators (GPT Image, Nano Banana, FLUX, Midjourney, Grok) | everything above is public — OpenFake supplies the same families (nano-banana, GPT Image 1, Midjourney 6, Ideogram 3, FLUX.2 Klein, Grok 2, Seedream 4.5) |

Reference numbers to rival (macro accuracy / mAP):

| Benchmark | Pangram Image | Previous best |
|---|---|---|
| CommunityForensics-Eval | 97.29% / 99.70% | Ours-384 (CVPR'25): 89.3% / 98.7% |
| Synthbuster + RAISE-1K | 98.49% / 99.96% | B-Free: 94.9% / 98.8% |
| Augmented (1024px + JPEG q50) | 99.03% acc | SightEngine 97.57% |
| NTIRE 2026 (AUROC) | 99.999% | MICV 99.78% |

Our eval harness (`src/seer/eval.py`) implements these protocols and
prints Pangram's published numbers next to ours. Seer's own scores are
the step-33,500 table under [Reproducing results](#reproducing-results),
not the Pangram column here.

---

## The data mixture (what makes or breaks this task)

Pangram: *"the specific data composition of both human and AI-generated
imagery had the largest impact on the final accuracy of the model compared to
anything else they tried."* Full source list, weights, licences, fetch
commands, and held-out sets: **[docs/DATA_MIXTURE.md](docs/DATA_MIXTURE.md)**.

The hero + probe configs (`seer_vitl_512.yaml`, `seer_probe.yaml`) share
this weighted mix. Missing folder sources are dropped at train time, not
fatal. `seer_vitl_local.yaml` and `seer_vits_debug.yaml` are **not** this
mixture — they fall back to a Community Forensics stream.

The mixture is largely **non-commercial** (Community Forensics is
CC BY-NC-SA 4.0; OpenFake's proprietary-generator subsets are
non-commercial). This repo is a research/hackathon artifact accordingly.

| Source | Class | Weight | What it covers |
|---|---|---|---|
| **NTIRE 2026 train** | mixed | 0.224 | 42 gens (2022–2026), all 6 shards, real/fake matched |
| **CommunityForensics-Small** | mixed | 0.176 | 4,782 open generators + paired reals. Eval is held out. PNG⇒fake is a known confound on this corpus — wild-simulation aug is mandatory |
| **OpenFake (selected)** | mixed | 0.128 | the 30 frontier/community generators this detector measurably misses + LAION/Pexels reals |
| **GAS-Station v4 / v3** | fake | 0.08 / 0.072 | weekly open-model dumps after `wire_gasstation.py` |
| **laion400m-1** | real | 0.128 | `jp1924/Laion400m-1` images in parquet (gated; not a URL scrape) |
| **Open Images V7** | real | 0.072 | validation + test photographs |
| **FLUX-Reason-6M** | fake | 0.04 | 5.9M FLUX.1-dev; streamed |
| **Frontier fakes** | fake | 0.04 | Midjourney / DALL·E / SD / Nano Banana Pro (label inverted) |
| **SID_Set** | fake | 0.04 | full-synthetic only (drop real + tampered) |

Roots: `$SEER_DATA_ROOT` (see [Setup and installation](#setup-and-installation)).
Local parquet is read in streaming mode.

### Choosing generators by measured difficulty, not by name

OpenFake is 3.44 TB over 645 shards and every shard interleaves all ~80 of
its generators, so it cannot be fetched selectively by file. More
importantly, most of it is not worth fetching: a generator this detector
already catches at 99% recall under JPEG 30 adds cost, not signal. So the
selection is measured rather than guessed —

```bash
uv run scripts/openfake.py probe --shards 3            # per-generator sample
uv run scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
uv run scripts/openfake.py fetch --from-rank $SEER_DATA_ROOT/openfake/rank.json \
    --labels fake real --tier 0.70=25000 0.95=15000 0.98=10000 \
    --cap-model pexels=80000 laion=50000
```

`openfake_rank.py` scores every generator clean and under the eval-table
perturbations; `fetch --from-rank` then pulls **inversely to recall**.
Defaults (`DEFAULT_TIERS` in `scripts/openfake.py`): **25k** images below
0.70 recall, **15k** below 0.95, **10k** below 0.98, nothing at or above
0.98. On the step-6000 ViT-L checkpoint that selected 30 generators —
worst were `nano-banana` at 0.20 recall / 0.70 AUROC, `qwen-image` 0.41,
`flux-1.1-pro` 0.64, `sd-3.5` 0.68, with `ideogram-3.0` at 0.85 and
`flux.2-klein-4b` at 0.98 — and skipped the ~29 it already saturates.
`tiny-random-sana` is excluded on purpose: it is a HuggingFace test stub
emitting uniform RGB noise, and our augmentation puts noise on *real*
images. OpenFake's `core/test` and `reddit/test` are held out (below).

---

## Usage

Full CLI catalog. The official Track 5 entry and the judge-facing
reproduce path are under [Reproducing results](#reproducing-results).

```bash
# 1. sanity check the model + parameter budget (random tiny ViT, offline)
uv run python main.py info --backbone tiny

# 2. quick end-to-end training: DINOv2-S @ 224 on a Community Forensics
#    slice (minutes, 12GB GPU). Not the hero mixture.
uv run python main.py train --config configs/seer_vits_debug.yaml

# 3. build the data mixture (see docs/DATA_MIXTURE.md)
#    everything lands in $SEER_DATA_ROOT
export SEER_DATA_ROOT=/path/to/data
uv run python get_datasets.py --list                      # the full plan; downloads nothing
uv run python get_datasets.py --tier 1                    # NTIRE train/val/test + COCO
uv run python get_datasets.py --only ntire-train ntire-val ntire-test coco-val2017 mirage
uv run python dataset_stats.py --tier 1                   # remote metadata only, no images
uv run scripts/fetch_data.py comfor-small                 # full ~260GB; add --max-shards 30 for a slice
uv run scripts/fetch_data.py frontier-fakes               # MJ / DALL-E / SD / Nano Banana Pro (~3 GB)
uv run scripts/fetch_data.py flux-reason-6m --max-shards 8 # optional; full dump is streamed
uv run scripts/fetch_data.py sid-set --max-shards 16
uv run scripts/wire_gasstation.py --versions v3 v4        # unpack GAS-Station tarballs
uv run scripts/download_laion400m.py --max-shards 20 --max-images 400000 --min-side 512
uv run scripts/openfake.py probe --shards 3               # then rank + fetch, see above
uv run scripts/openfake.py holdout --config core          # held-out OOD eval
uv run scripts/openfake.py holdout --config reddit        # held-out in-the-wild eval
uv run scripts/download_open_images.py --workers 32 --max-gb 70

# 4. full training (hero config = the mixture above)
uv run python main.py train --config configs/seer_vitl_512.yaml          # A100-class
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set resume=runs/seer_vitl/last.pt                                  # continue a run
uv run python main.py train --config configs/seer_vitl_local.yaml       # single 12GB GPU; not the hero mix

# 4b. multi-layer linear probe (frozen backbone) — cheap ablation
#     against continuation training. Same mixture as the hero YAML.
uv run python main.py train --config configs/seer_probe.yaml
#     or on top of any config:
uv run python main.py train --config configs/seer_vitl_512.yaml \
    --set probe.enabled=true probe.layers=[3,9,15,-1] head_lr=1e-3

# 5. benchmark against Pangram's protocol
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_eval
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_small
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset comfor_eval --augmented   # 1024px + JPEG q50
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val_hard
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_test   # HF public test (2.5k)
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_test_public  # alias
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset openfake_test    # unseen gens AND unseen reals
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset openfake_reddit  # in the wild
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset mirage
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset folders \
    --real-dir $SEER_DATA_ROOT/wikiart --out-json wikiart_fpr.json                   # FPR eval
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset folders \
    --real-dir $SEER_DATA_ROOT/coco-val2017 --out-json coco_fpr.json                 # COCO val2017 FPR
#    OpenFake defaults to 4,096 images (OPENFAKE_EVAL_MAX). Full set:
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset openfake_test --max-samples 0
#    Robustness: all (15-level eval table incl. clean) / extra|hard / all+extra
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_test --perturbation all
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_test --hflip-tta

# 5b. error analysis: the most confident FPs / FNs, each with its heatmap
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt --dataset ntire_val \
    --error-dir runs/eval/errors --error-n 6 --out-json runs/eval/ntire_val.json

# 6. use it — official Track 5 entry, or a single-image heatmap
uv run python predict.py --image-dir ./images --out preds.json   # downloads glennwuwu/seer if needed
uv run python main.py infer --checkpoint runs/seer_vitl/best.pt \
  --image suspect.jpg --out-dir out/                                    # verdict + heatmap PNG

# 7. inspect composite training data (image + patch labels per sample)
uv run scripts/save_samples.py                                           # writes samples/
```

There is **no** `--dataset synthbuster`. Fetch with
`uv run scripts/download_synthbuster.py`, then score via `--dataset folders`
(`--fake-dir $SEER_DATA_ROOT/synthbuster/fake`; RAISE reals are a separate
form-gated download). Optional synthetic mirroring lives in
`scripts/generate_mirrors.py` (`uv sync --group gen`).

### Repo map

```
main.py                  CLI: train | eval | infer | info
predict.py               Track 5 entry: image directory → {image_path, pred} JSON
configs/                 seer_vitl_512 (hero mix) | seer_vitl_local | seer_probe | seer_vits_debug
src/seer/                model, train, eval, data, augment, heatmap, infer, paths, ...
scripts/                 fetch_data, openfake*, generate_mirrors, download_*, wire_gasstation
get_datasets.py          --list / --tier / --only acquisition plan
eval/eval_step33500/     committed held-out JSONs + run_suite.py
client/                  Next.js dashboard (see client/README.md)
docs/                    DATA_MIXTURE.md, DELIVERABLES.md, deliverables/held-out writeup
project_description.md   Devpost writeup
```

---

## Dashboard (`client/`)

A Next.js dashboard (dark mode, Geist, Tailwind) covering the demo,
robustness-summary, and error-analysis deliverables — with live inference
when a checkpoint is present. Details: [`client/README.md`](client/README.md).

| Page | Shows |
|---|---|
| `/` | overview — architecture, held-out table, data mixture |
| `/analyze` | upload images → P(AI) + per-patch heatmap; exports `seer_predictions.json` (`image_path` / `pred`) |
| `/robustness` | clean vs transformed table + charts, plus the NTIRE 2026 open-test leaderboard |
| `/errors` | most confident FP/FN with heatmaps, plus the trade-offs note |

```bash
cd client && npm install && npm run dev       # http://localhost:3000
```

Prefer `client/scripts/seer_serve.py` on **:8765** so `/api/analyze`
does not spawn `seer_infer.py` per upload. Upload limits: **12 images /
40 MB each**. `/robustness` and `/errors` scan eval JSONs from
`eval/eval_step33500/` first, then `runs/eval/` and `runs/`, and fall
back to bundled demo data when none are present.

---

## Throughput & the training bottleneck (measured, RTX 4070)

`scripts/bench_loader.py` profiles each pipeline stage independently
(`uv run scripts/bench_loader.py --source local --parquet-dir $SEER_DATA_ROOT/commfor-small/data`):

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
  local parquet from `$SEER_DATA_ROOT` via `scripts/fetch_data.py`.

---

## Parameter budget

| Backbone | Params (incl. heads) | % of 2B budget |
|---|---|---|
| DINOv3 ViT-S/16 | ~23M | 1.2% |
| DINOv2-S / ViT-B | ~24M / ~88M | ~4% |
| **DINOv3 ViT-L/16 (default)** | **305,233,922** | **15.3%** |
| DINOv3 ViT-H+/16 | ~843M | 42% |
| DINOv2 ViT-g/14 | ~1.14B | 57% |

Even the largest supported backbone stays under budget; the default leaves
~6.5× headroom for resolution, TTA, or an ensemble.

---

## Training recipe (hero config)

- **Continuation training**: full backbone FT + heads. The dataclass
  default `TrainConfig.optimizer` is **muon**; the hero YAML overrides
  that to **`adamw`**. Layer-wise LR decay 0.8, cosine schedule, warmup
  1k, EMA 0.999, bf16. The probe recipe (`seer_probe.yaml`) keeps Muon
  on 2D head weights.
- **Dual-head objective**: image-level BCE + per-patch BCE (weight 1.0), the
  patch term `pos_weight`-balanced by `n_real/n_fake` patches
  (`balance_patch`) so a small fake-over-real crop is not drowned out by the
  real majority around it.
- **Composite training** (60% of fake samples; FoR/RoF/FoF/RoR quota-equal,
  page labels stay 1:1): cropped overlays layered over a base image. Compositing is
  itself a discontinuity, so *all four*
  top-on-base pairings are trained — fake-over-real (localized labels),
  real-over-fake (inverted labels), fake-over-fake (label 1 everywhere),
  real-over-real (label stays real). Each overlay independently draws a
  freeform silhouette (rect, ellipse, polygon, star, blob, noise),
  alpha-blend vs hard-paste, and hard vs soft feather, so one stacked
  image can mix those combinations. RGB uses that alpha; labels
  follow occupancy, not blend opacity (a 40% mix is still fake, not a
  0.4 target). Soft-feather seams stay mixed after average-pool; the
  page target is binary (any visible AI → fake). Label maps travel with
  the crop when an already-composited slot is reused as a source.
  Overlays are large difficulty-varying crops of the source (easy ≈
  full frame, hard still a substantial semantic region) so objects/scene
  survive the shrink-to-window; a sample can receive n ~ Uniform{1,...,k}
  overlays (k = max_overlays, default 5). Later overlays on a fake page
  are independently real or fake so stacks cover every class sequence.
- **Wild-simulation augmentation** applied symmetrically to both classes.
  Train is **intentionally harder than the eval perturbation table**.
  Eval `--perturbation all` stays JPEG 90/70/50/30, blur 0.5/1/2,
  resize 0.5×/0.25×, noise 0.02/0.05/0.10, jitter ±20%, crop 80%.
  The hero YAML adds JPEG q∈{20,10,5}, blur σ=4, resize 0.125×, noise
  0.20, jitter ±35%, plus a stack of extras (DCT grid-shift JPEG,
  resample-kernel mismatch, FFT phase noise, social re-encode, …).
  Score those extras with `--perturbation extra` (alias: `hard`) or
  `all+extra` so robustness is not an artifact of testing the train aug.
- 512×512 input (patch grid 32×32), **effective batch 168** (56 × 3
  accum), **60,000** steps. Resume with
  `--set resume=runs/seer_vitl/last.pt`.

### Multi-layer linear probe (ablation)

`probe` mode is the frozen-backbone alternative to continuation training:
linear heads on features tapped from several transformer blocks.
Early blocks carry high-frequency / low-level statistics — where generator
fingerprints live — while mid and late blocks carry increasingly semantic
features, so the probe sees both ends of the hierarchy. Two independent heads
are trained, each over its own LayerNorm-standardized concatenation of taps:
a **page head** on `[CLS ; mean(patch tokens)]` → one logit per image, and a
**patch head** on the raw patch tokens → one logit per patch.
`configs/seer_probe.yaml` enables that patch head
(`composite.patch_loss_weight: 1.0`); probe checkpoints therefore produce
heatmaps too.

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

---

## Evaluation protocol

`main.py eval` replicates the Pangram blog's setup:

- **CommunityForensics-Eval** — 51.8K images, 21 generators incl.
  commercial ones; macro accuracy + mAP, per-architecture breakdown.
  `--dataset comfor_small` scores the train-split Small dump (not held out
  in the same way).
- **Robustness sweeps** — `--perturbation all` evaluates every benchmark
  perturbation level (JPEG 90/70/50/30, blur 0.5/1/2, resize 0.5×/0.25×,
  noise 0.02/0.05/0.10, jitter ±20%, crop 80%, plus clean) and prints
  a per-level robustness table. `--perturbation extra` (alias `hard`) is
  the harder NTIRE-style table; `all+extra` runs both.
- **NTIRE 2026** — `--dataset ntire_val` / `ntire_val_hard` / `ntire_test`
  (alias `ntire_test_public`) after `python get_datasets.py --tier 1` or
  `--only ntire-val ntire-test`. `ntire_test` is the labelled 2.5k public
  test from
  [`deepfakesMSU/NTIRE-RobustAIGenDetection-test-public`](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-test-public)
  (clean vs distorted + per-distortion). The 512 recipe also scores it
  every `eval_every` steps (`eval_datasets: [ntire_test, openfake_test]`);
  `best.pt` still follows the train-distribution val slice.
- **OpenFake held-out** — `--dataset openfake_test` / `openfake_reddit`
  after `scripts/openfake.py holdout`. Default cap is **4,096** images
  (`OPENFAKE_EVAL_MAX`); pass `--max-samples 0` for the full set (the
  committed step-33,500 OpenFake rows did). `openfake_test` shifts
  generators *and* real sources at once (`gpt-image-1.5/2`,
  `nano-banana-pro`, `flux.2-klein-9b`, `midjourney-7`, `ideogram-2.0`,
  `recraft-v2/v3`, `sora-2`, `veo-3` against DOCCI + ImageNet reals);
  `openfake_reddit` is naturally circulated content with unknown
  provenance. Neither can reach training: the loader refuses any path
  under `openfake/holdout_*`. Both report per generator.
- **MIRAGE** — `--dataset mirage` after `get_datasets.py --only mirage`.
- **FPR sets** — real-only `--dataset folders` (WikiArt, COCO val2017
  after `get_datasets.py --only coco-val2017`, etc.).
- **Synthbuster** — not a `--dataset` name; folders +
  `scripts/download_synthbuster.py`.
- Metrics: macro (balanced) accuracy, mAP (AP on fake class), AUROC, F1,
  precision/recall, FPR/FNR, plus per-architecture and (on NTIRE)
  clean-vs-distorted and per-distortion breakdowns.
  Published Pangram numbers are printed next to ours for direct comparison.

**Which numbers to quote.** The committed, inspectable suite is
[`eval/eval_step33500/`](eval/eval_step33500/) (step 33,500 `last.pt`).
[`docs/deliverables/heldout-eval-step27500.md`](docs/deliverables/heldout-eval-step27500.md)
is an earlier writeup of the same recipe at step 27,500. A local
`best.pt` is whichever train-val snapshot the loop saved — do not
treat it as interchangeable with either JSON.

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
`--perturbation all` writes 15 folders (the official table, including clean).

---

## References

- Stajduhar & Emi, *Introducing Pangram Image Detection*, 2026 (blog)
- Park & Owens, *Community Forensics*, CVPR 2025 ([arXiv:2411.04125](https://arxiv.org/abs/2411.04125))
- Bammey, *Synthbuster*, OJSP 2023 ([Zenodo](https://zenodo.org/records/10066460))
- Simeoni et al., *DINOv3*, 2025 ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104))
- Zhu et al., *GenImage*, NeurIPS 2023 ([arXiv:2306.08571](https://arxiv.org/abs/2306.08571))
- Gushchin et al., *NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild*
