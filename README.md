# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026, Track 5.

Detects AI-generated images and keeps working after the image has been through
the things that happen to images on the internet: recompression, blurring,
thumbnailing, sensor noise, filters and cropping.

Research notes, with sources and measured findings, are in [`research.md`](research.md).

---

## Contents

- [Setup](#setup)
- [Datasets](#datasets)
  - [Label conventions — read this first](#label-conventions--read-this-first)
  - [What is on disk](#what-is-on-disk)
  - [The full dataset plan](#the-full-dataset-plan)
  - [The degradations the organisers actually apply](#the-degradations-the-organisers-actually-apply)
- [Usage](#usage)
- [Results so far](#results-so-far)
- [Layout](#layout)

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"

pytest src/tests -q             # 65 tests
```

Requires Python ≥ 3.10 and `timm ≥ 1.0.20`.

**DINOv3 weights are not gated for us.** Meta's own HuggingFace repos require
manual approval that can take days, but the `timm` mirror is ungated with
Meta's explicit permission, and that is what `aigcdet.backbone` loads. No token
or approval is needed.

---

## Datasets

### Label conventions — read this first

Conventions differ between datasets. Getting one wrong does not crash anything,
it silently inverts training, so the convention is asserted rather than assumed.

| Dataset | Column / mechanism | Real | AI-generated | Note |
|---|---|---|---|---|
| **NTIRE 2026** | `label` in `labels.csv` | `0` | `1` | Our primary source |
| Community Forensics | `label` column | `"0"` | `"1"` | **Strings, not ints** |
| SID_Set | `label` column | `0` | `1` = fully synthetic, `2` = **tampered** | **Three classes** — must be collapsed for binary use |
| CIFAKE | directory name | `REAL/` | `FAKE/` | No label column |
| MIRAGE | `label` column | `0` | `1` | |

**Our own output follows the NTIRE convention**: `pred` is the probability that
an image is **AI-generated**, so `pred → 1.0` means fake and `pred → 0.0` means
authentic.

Two traps worth naming. SID_Set is three-class, so treating it as binary without
deciding what to do with the 100k tampered images will quietly mislabel a third
of the set. And Community Forensics stores labels as strings, so `label == 1`
is always `False`.

### What is on disk

Reproduce with `python dataset_stats.py --local`.

| Split | Total | Real | AI-generated | % fake | Degraded |
|---|---|---|---|---|---|
| `ntire-train` shard 0 | 50,000 | 17,982 | 32,018 | 64.0% | 0 |
| `ntire-val` | 10,000 | 5,000 | 5,000 | 50.0% | 5,000 |
| `ntire-val-hard` | 2,500 | 1,250 | 1,250 | 50.0% | 1,250 |
| `ntire-test` | 2,500 | 1,200 | 1,300 | 52.0% | 1,250 |

Three things follow from this table:

**Training data is not balanced** — 64% of shard 0 is AI-generated, reflecting
the corpus-wide 1 : 1.77 real : fake ratio. Sample it balanced (which
`stratified_subset` does) or weight the loss, otherwise the prior alone buys 64%
accuracy and the metric becomes meaningless.

**Evaluation splits are balanced and half-degraded.** In each of val, val-hard
and test, exactly half of *each class* is degraded and half is left clean, so
degradation itself carries no label information. That is what lets us report
clean and robust AUC on the same split without confounding them.

**Everything here is held out except shard 0.** The full training set is
277,650 images across six distribution-matched shards, so one 19 GB shard is a
legitimate training set on its own.

### The full dataset plan

`python get_datasets.py --list` prints this. Sizes are measured from the remote
repositories, not estimated.

| Key | Tier | Size | Role |
|---|---|---|---|
| `ntire-val` | 1 | 4.0 GB | Primary held-out evaluation, with per-image degradation ground truth |
| `ntire-test` | 1 | 0.9 GB | Second held-out set, unseen proprietary generators |
| `ntire-train` | 1 | 19.0 GB | Primary training source (shard 0 of 6) |
| `coco-val2017` | 1 | 0.8 GB | Real half of the organisers' reference benchmark |
| `commfor-small` | 2 | 106.5 GB | Generator breadth: 4,803 generators |
| `dda-coco` | 2 | 4.3 GB | Shortcut probe — do we use causal artifacts or dataset bias? |
| `dda-train` | 2 | — | **Regenerated, not downloaded** (see below) |
| `mirage` | 2 | 1.3 GB | In-the-wild evaluation, human-verified |
| `sid-set` | 3 | 140.0 GB | Only for tampering / localisation |
| `cifake` | 3 | 0.1 GB | Pipeline smoke test only |
| `wildfake-dalle` | 3 | 25.6 GB | Fake half of the reference benchmark (DALL·E 3) |
| `synthbuster` | 3 | 12.4 GB | 9 commercial generators, uncompressed |

Why NTIRE is the primary source: it is the only public corpus that is
simultaneously drawn from 42 generators spanning 2022–2026, **matched between
real and fake on resolution, aspect ratio and JPEG quality**, and shipped with
per-image degradation ground truth. That matching matters — most AIGC datasets
pair JPEG reals against PNG fakes, and a detector will happily learn the file
format instead of the forgery.

Three entries behave unusually and the tooling encodes why:

- **`dda-train` is deliberately not downloaded.** It is an 11-part split ZIP
  that cannot be streamed or partially fetched and needs ~226 GB of peak disk.
  An equivalent subset regenerates from COCO in about an hour.
- **`commfor-small` shards are sorted by label** — 0–92 are fake, 94–185 are
  real. A naive prefix or uniform stride silently yields a single-class set with
  no GAN or pixel-diffusion images, so the selection keeps shards 70–92 in full.
- **`cifake` is a smoke test, not training data.** At 32×32 every one of the six
  target transformations is meaningless, and its reals and fakes have different
  resampling histories.

### The degradations the organisers actually apply

The validation and test labels record which degradations hit each image and at
what severity, so this is measured, not inferred. The vocabulary escalates
sharply across splits:

| Split | Most frequent | What is newly introduced |
|---|---|---|
| `val` | `downscale` (3,708), `lincontrchange`, `jpeg`, `randomcrop` | the basic families: blur, noise, crops, colour, JPEG |
| `val-hard` | `downscale` (931), `quantization`, `jpeg` | **`jpeg_ai`** (neural codec), **`jpeg_recompression_1/2/comb`** (stacked compression), **`adv_embed_clip/resnet`** (adversarial watermark embedding), `clahe`, `randomtonecurve`, `perspective`, `isonoise` |
| `test` | `downscale` (926), **`watermark` (448)**, `brighten`, `lensblur` | **`cheng2020`** (learned compression), `jpeg2000`, **`wmforger`** (watermark forging), `shotnoise` |

Two observations that shape our training distribution. `downscale` is the single
most common operation in every split by a wide margin. And `jitter` appears 401
times in `val` but **zero times** in `val-hard` or `test` — colour jitter was
dropped as insufficiently damaging, while stacked compression and neural codecs
were added because they break detectors.

The six transformations mandated by the problem statement correspond closely to
the tier that was *discarded*. We evaluate on them because they are the spec,
but we train on a wider and harder distribution.

---

## Usage

### Get the data

```bash
python get_datasets.py --list                 # the plan; downloads nothing
python get_datasets.py --tier 1               # core, ~25 GB
python get_datasets.py --tier 2 --dry-run     # what tier 2 would cost
python get_datasets.py --only ntire-val mirage
```

Resumable and idempotent — anything already present is skipped. Kaggle and
ModelScope entries print the exact command instead of guessing at credentials.

### Inspect datasets without downloading them

```bash
python dataset_stats.py                       # all datasets, remote metadata only
python dataset_stats.py --tier 1 --detail
python dataset_stats.py --only commfor-small --audit
python dataset_stats.py --local               # class balance of what is on disk
python dataset_stats.py --json stats.json
```

Everything except `--local` uses HTTP metadata only: the HuggingFace repo tree
for exact byte sizes, and datasets-server for row counts and schema. No images
are fetched.

Where a statistic would be misleading, the tool says so instead of printing it.
For Community Forensics the datasets-server has indexed only 10,542 of ~556,000
rows and, because shards are label-sorted, that prefix is entirely one class —
so `--audit` reports the limitation rather than a plausible-looking distribution.

### Train

```bash
# frozen backbone, live augmentation, consistency loss
python src/scripts/train_live.py --train-n 4000 --epochs 4

# unfreeze the last two blocks (more VRAM)
python src/scripts/train_live.py --train-n 8000 --epochs 4 --unfreeze-last-n 2

# reference points
python src/scripts/run_baseline.py --train-n 4000     # clean-trained probe
python src/scripts/train_augmented.py --views 6       # cached-view augmentation
```

Augmentation is applied **inside the training loop**, so each image is degraded
differently every epoch rather than cycling a few cached views. Degradations
run at native resolution *before* the resize to the backbone input — reversing
that order destroys the artifacts being detected. In paired mode the clean and
degraded views come from the same source image, so the augmentation itself
cannot leak class information.

### Evaluate

```python
from aigcdet.degradations import MANDATED_GRID, EXTENDED_GRID, compound_grid
from aigcdet.evaluate import evaluate_grid, summarise, to_markdown

results = evaluate_grid(images, labels, score_fn, MANDATED_GRID)
print(to_markdown(results))
print(summarise(results))
```

`MANDATED_GRID` is the exact grid from the problem statement. `EXTENDED_GRID`
adds stacked compression and harder crops. `compound_grid()` chains 1–5
operations from distinct families, which is where detectors actually break.

Reported per degradation: AUC, accuracy, **real and fake accuracy separately**,
TPR at 1% and 0.1% FPR, and mean score shift. The per-class split matters —
under degradation detectors do not raise false alarms, they collapse into
predicting "real", and balanced accuracy hides that entirely.

---

## Results so far

Frozen DINOv3 ViT-L/16 (303M parameters, 15% of the 2B cap), logistic probe,
4,000 training images, evaluated on 800 held-out validation images.

| Metric | Clean-trained | Cached-view augmentation |
|---|---|---|
| Clean AUC | 0.9236 | 0.9129 |
| Mean degraded AUC | 0.9077 | 0.9004 |
| Worst-case AUC | 0.8338 | **0.8407** |
| Clean − worst gap | 0.0898 | **0.0722** |
| Max score drift | 0.1187 | **0.0693** |

Worst case is Gaussian noise σ=0.10 for both. Augmentation narrowed the gap and
cut score drift by 42%, at the cost of about a point of clean AUC — a smaller
effect than expected, which is consistent with the backbone being frozen: a
probe can only reweight features that already exist. Live augmentation with a
partially unfrozen backbone has not been run at scale yet.

Raw tables are in [`results/`](results/).

---

## Layout

```
get_datasets.py          acquire datasets (tiered, resumable)
dataset_stats.py         inspect datasets remotely, or class balance locally
research.md              research notes with sources and measured findings

src/aigcdet/
  degradations.py        the six mandated families, extended grid, compound chains
  dataset.py             torch Dataset with live, pair-synchronised augmentation
  backbone.py            frozen DINOv3 feature extractor
  extract.py             threaded extraction with on-disk feature caching
  train.py               training loop, consistency loss, partial unfreezing
  evaluate.py            robustness metrics
  data.py                NTIRE loading
  datasets_registry.py   single source of truth for every dataset
  paths.py               canonical project paths

src/scripts/             runnable experiments
src/tests/               65 tests
data/                    downloaded datasets and the feature cache (gitignored)
results/                 robustness tables and summaries
```

Because the backbone is frozen, `extract.py` caches embeddings to disk keyed by
model, resolution, pooling and degradation. A pass that takes 30 seconds becomes
0.02 seconds on re-run, which is what makes a wide ablation affordable.

---

## Notes and limitations

- The NTIRE repositories carry **no licence tag**. Treat as research use and
  confirm before any downstream distribution.
- Community Forensics is **CC BY-NC-SA 4.0** (non-commercial, viral share-alike).
  The full `OwensLab/CommunityForensics` repo is CC BY 4.0 if that matters.
- DINOv3 is under the DINOv3 Licence, which does permit commercial use but
  requires derivatives to carry the same licence.
- Extraction is CPU-bound on the degradations, not on decode or GPU compute —
  decoding once and fanning out to nine views gained only 1.08×. Bulk extraction
  wants a high-core machine more than a bigger GPU.
