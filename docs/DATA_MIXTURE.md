# Training data mixture

Canonical train mix for the hero recipe (`configs/seer_vitl_512.yaml`) and the
frozen-backbone probe (`configs/seer_probe.yaml`, page head **and** patch head).
Those two configs share the same `data.sources` list and weights so the
continuation vs probe comparison is honest.

`seer_vitl_local.yaml` and `seer_vits_debug.yaml` do **not** use this mixture
(they fall back to a single Community Forensics stream). The debug recipe is
DINOv2-S @ 224, not a random tiny backbone.

Roots: `$SEER_DATA_ROOT`. If unset, `src/seer/paths.py` uses `/workspace/data`
when a writable `/workspace` mount exists, otherwise **`F:/techjam`**. On
macOS/Linux that Windows default is almost certainly wrong — export
`SEER_DATA_ROOT` before fetching or training. Paths below are written as
`$SEER_DATA_ROOT/<name>`.

The mixture is largely **non-commercial** (Community Forensics is
CC BY-NC-SA 4.0; OpenFake's proprietary-generator subsets are
non-commercial). This repo is a research/hackathon artifact accordingly.

## How sampling works

`MixtureDataset` first draws a class (real vs fake) with equal probability
when `balance_labels` is on, then picks source *i* among sources that can
produce that class with probability `weight_i / sum(weights in class)`.
Weights are relative — the sampler renormalizes within the drawn class —
and sum to 1.0 across all sources.

A source that raises `FileNotFoundError` or is empty is dropped for the rest
of the run (logged as `[data] dropping <name>`). That is how unwired
GAS-Station listings and still-empty real folders fail open.

Labels after mapping: **0 = real**, **1 = fake**. Composite overlays (60% of
fake samples; FoR/RoF/FoF/RoR quota-equal, page labels stay 1:1) can change
the page/patch labels on top of this mix; they are not a separate source.

## Train sources

Yaml weights sum to **1.0**, so each weight is that source's draw share
before `balance_labels` picks the class.

| Source | Class | Weight | On-disk / stream | How to materialize |
|---|---|---|---|---|
| `comfor` | mixed (paired real + fake) | 0.176 | `$SEER_DATA_ROOT/commfor-small` (556,541: 278,445 fake / 278,096 real) | `scripts/fetch_data.py comfor-small` |
| `ntire` | mixed (matched real + fake) | 0.224 | `$SEER_DATA_ROOT/ntire` (277,643: 177,643 fake / 100,000 real) | `python get_datasets.py --only ntire-train` |
| `openfake` | mixed (selected fakes + reals) | 0.128 | `$SEER_DATA_ROOT/openfake/train` (439,523: 309,523 fake / 130,000 real) | `scripts/openfake.py fetch --from-rank …` |
| `flux-reason` | fake | 0.04 | stream `LucasFang/FLUX-Reason-6M` (320,000 local slice; ~5.9M full) | optional slice: `scripts/fetch_data.py flux-reason-6m --max-shards 8` |
| `frontier-fakes` | fake only | 0.04 | `$SEER_DATA_ROOT/frontier-fakes` (5,195 fakes used of 10,695) | `scripts/fetch_data.py frontier-fakes` |
| `sid-set` | fake only (full synthetic) | 0.04 | `$SEER_DATA_ROOT/sid-set` (70,000 class-1 of 210,000) | optional slice: `scripts/fetch_data.py sid-set --max-shards 16` |
| `gs-images-v3` | fake | 0.072 | `$SEER_DATA_ROOT/gs-images-v3/wired/images.txt` (426,689) | `scripts/wire_gasstation.py --versions v3` |
| `gs-images-v4` | fake | 0.08 | `$SEER_DATA_ROOT/gs-images-v4/wired/images.txt` (113,793) | `scripts/wire_gasstation.py --versions v4` |
| `laion400m-1` | real | 0.128 | `$SEER_DATA_ROOT/laion400m-1/real` (199,998, growing toward 400k) | `scripts/download_laion400m.py` |
| `open-images-v7` | real | 0.072 | `$SEER_DATA_ROOT/open-images-v7` (167,055) | `scripts/download_open_images.py` |

Within-class shares (what a weight actually buys, since `balance_labels`
picks the class first):

| Class | Shares |
|---|---|
| fake | ntire 28%, comfor 22%, **openfake 16%**, gs-v4 10%, gs-v3 9%, flux-reason 5%, frontier-fakes 5%, sid-set 5% |
| real | ntire 31%, comfor 24%, **laion400m-1 18%**, openfake 18%, open-images-v7 10% |

Community Forensics is roughly balanced; NTIRE train is real/fake matched per
shard but not exactly 50/50 overall. `balance_labels` then forces **50/50
real/fake** batches; the weights above only decide which source is used once
the class is chosen.

Do not add a source above ~25% share without revisiting the others. Recency
is tilted toward NTIRE (42 generators, 2022–2026), OpenFake (frontier
commercial APIs through 2026) and GAS-Station v4.

## On-disk inventory

Counts below are what this volume actually has under `$SEER_DATA_ROOT`.
Weights above decide draw probability, not how often a unique image is
seen. Streamed sources can draw past the local slice.

| Source | Fake | Real | Total | Fake generators |
|---|---:|---:|---:|---|
| `comfor` | 278,445 | 278,096 | 556,541 | 4,782 (19 named + 4,763 HF community) |
| `ntire` | 177,643 | 100,000 | 277,643 | 42 (not tagged per image) |
| `openfake` | 309,523 | 130,000 | 439,523 | 30 ranked |
| `flux-reason` | 320,000 local | 0 | 320,000 | FLUX.1-dev only (full Hub dump ~5.9M) |
| `frontier-fakes` | 5,195 | 5,500 unused | 10,695 | untagged Midjourney / DALL-E / SD / Nano Banana Pro mix |
| `sid-set` | 70,000 | 70,000 unused (+ 70k tampered unused) | 210,000 | untagged |
| `gs-images-v3` | 426,689 | 0 | 426,689 | 19 folders (`unknown` = 186,579 unlabeled) |
| `gs-images-v4` | 113,793 | 0 | 113,793 | 15 folders (`unknown` = 58,733 unlabeled) |
| `laion400m-1` | 0 | 199,998 | 199,998 | — |
| `open-images-v7` | 0 | 167,055 | 167,055 | — |
| **Usable by the mix** | **1,701,288** | **875,149** | **2,576,437** | reals exclude frontier / SID unused rows |

## Per-source notes

### `comfor` — Community Forensics Small

- Hub: `OwensLab/CommunityForensics-Small` (train only).
- **On disk** (`$SEER_DATA_ROOT/commfor-small`, 186 parquet shards):
  **556,541 images — 278,445 fake / 278,096 real**. Fakes span **4,782
  generators**: 19 canonical named families (79,152 images) plus 4,763
  HuggingFace community checkpoints (typically 30–57 images each).
  Architecture of the fakes: LatDiff 207,581, GAN 57,397, PixDiff 7,251,
  Other 6,216. Reals are four source pools: COCO 118,287, LandscapesHQ
  90,000, FFHQ 63,000, VISION 6,809.
- **Held out:** `OwensLab/CommunityForensics-Eval`. The loader refuses to
  train on that repo or any `comfor-eval` path.
- No commercial APIs (no DALL-E / Midjourney / FLUX). SD-derivative heavy;
  weight is kept below NTIRE so it does not dominate.
- Licence: CC BY-NC-SA 4.0. Shards are sorted by label (PNG⇒fake is a
  known confound); wild-simulation aug is mandatory.

  | Generator | Images |
  |---|---:|
  | `GigaGAN` | 17,612 |
  | `BigGAN` | 10,360 |
  | `tamingTransformers` | 6,216 |
  | `StyleSANXL` | 6,216 |
  | `LFM` | 6,216 |
  | `StyleGANXL` | 6,216 |
  | `glide` | 5,179 |
  | `StyleGAN2` | 4,144 |
  | `StyleSwin` | 3,108 |
  | `ProGAN` | 3,108 |
  | `StyleGAN3` | 2,279 |
  | `VQDiffusion` | 2,072 |
  | `StyleGAN2-ADA` | 1,246 |
  | `Gansformer` | 1,036 |
  | `guidedDiffusion` | 1,036 |
  | `ProjectedGAN` | 1,036 |
  | `CIPS` | 1,036 |
  | `DiT` | 1,036 |
  | `DeepFloyd` | 1,036 |
  | 4,763 HF community models | 199,293 |

### `ntire` — NTIRE 2026 train (all shards)

- Hub: `deepfakesMSU/NTIRE-RobustAIGenDetection-train`.
- **On disk** (all 6 shards): **277,643 images — 177,643 fake / 100,000
  real**. Per-shard: 50,000 each for `shard_0`–`shard_4`, 27,643 for
  `shard_5`. Upstream documents 42 generators (2022–2026); `labels.csv`
  has only `image_name,label`, so there is no per-image generator
  breakdown on disk.
- `split: train`, `shard: -1` concatenates every downloaded shard.
  Reals are matched on resolution, aspect ratio, and JPEG quality.
- Val / test are **not** in this source (see held-out below).

### `flux-reason` — FLUX.1-dev

- Every row is fake (`label: 1`). Single generator: **FLUX.1-dev**.
- **On disk:** 320,000 images (64 of 415 `Aesthetics-Part01` shards).
  Full dump is ~5.9M / ~882 GB; default training streams the rest. Do
  not snapshot the repo. Weight is kept below `sid-set` so one 2024
  family does not dominate fake draws.

### `frontier-fakes` — Midjourney / DALL-E / SD / Nano Banana Pro

- Hub: `julienlucas/midjourney-dalle-sd-nanobananapro-dataset`, train split.
- Upstream ClassLabel is inverted (`0=fake`, `1=real`). Yaml remaps then
  `keep_label: 1` so only fakes are trained. Train fakes are only ~5k;
  the 0.04 weight is already larger than that pool can sustain without
  repeats.

### `sid-set` — SID_Set full synthetic

- Hub: `saberzl/SID_Set`. Three classes: `0=real`, `1=full synthetic`,
  `2=tampered`. We keep **class 1 only** (`label_map` + `keep_label: 1`).
  Reals and tampered images are dropped. ~70k usable of ~210k train rows.

### `openfake` — OpenFake v2, selected by measured difficulty

- Hub: `ComplexDataLab/OpenFake`, config `core`, split `train` only.
- **Never snapshot this repo.** It is 3.44 TB over 645 parquet shards, and
  every shard interleaves all ~80 generators (~30 rows per generator per
  4,000-row shard). That means neither filename selection nor parquet
  row-group pushdown can isolate a model — a row group holds ~69 rows in
  ~100 MB, so fetching one generator's rows costs almost the whole shard.
  `scripts/openfake.py` therefore streams shards, filters on the `model`
  column, and deletes each shard before pulling the next. Peak footprint is
  one 5.4 GB shard plus the JPEGs kept.
- **Which generators, and why.** Breadth is not the point here; the mixture
  already has **4,782** counted generators from `comfor` (the registry / code
  comments still quote 4,803 — the Community Forensics paper figure). This
  source exists to close a measured hole. The pipeline is:

  ```bash
  uv run scripts/openfake.py probe --shards 3          # per-generator sample
  uv run scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
  uv run scripts/openfake.py fetch --from-rank $SEER_DATA_ROOT/openfake/rank.json \
      --labels fake real --tier 0.70=25000 0.95=15000 0.98=10000 \
      --cap-model pexels=80000 laion=50000
  ```

  `openfake_rank.py` scores every generator clean and under the eval-table
  perturbations, and `fetch --from-rank` turns `recall_mean` into a cap:

  | `recall_mean` | Cap | Reading |
  |---|---|---|
  | < 0.70 | 25,000 | a hole — most of this generator gets past the detector |
  | 0.70–0.95 | 15,000 | leaks under compression |
  | 0.95–0.98 | 10,000 | mostly solved, kept for recency |
  | ≥ 0.98 | not fetched | already caught; adds cost, not signal |

- **On disk** (`$SEER_DATA_ROOT/openfake/train`, `core` / `train`, all
  608 shards scanned): **439,523 images — 309,523 fake (30 generators) +
  130,000 real**. Reals are at cap: Pexels 80,000 and LAION 50,000.
  Seventeen of 30 fakes hit their cap; the rest are supply-limited in
  OpenFake (sparse commercial APIs). Ranked on the step-6000 ViT-L
  checkpoint. The worst were `nano-banana` (0.20 recall, 0.70 AUROC),
  `qwen-image` (0.41), `flux-1.1-pro` (0.64), `sd-3.5` (0.68),
  `sdxl-realvis-v5` (0.68), `chroma` (0.74); `ideogram-3.0` came in at 0.85
  and `flux.2-klein-4b` at 0.98. Everything at 1.00 — every
  `cyberrealistic-pony`, `animagine-xl`, `imagen-3/4`, `playground-v2.5`,
  `sd-turbo` — was left out on purpose.

  | Generator | Images | `recall_mean` |
  |---|---:|---:|
  | `flux-1.1-pro` | 25,000 | 0.64 |
  | `sd-3.5` | 25,000 | 0.68 |
  | `sdxl-realvis-v5` | 25,000 | 0.68 |
  | `flux-mvc5000` | 15,000 | 0.90 |
  | `flux.1-dev` | 15,000 | 0.94 |
  | `hidream-i1-full` | 15,000 | 0.91 |
  | `ideogram-3.0` | 15,000 | 0.85 |
  | `midjourney-6` | 15,000 | 0.94 |
  | `mystic` | 15,000 | 0.89 |
  | `sd-1.5-epicdream` | 15,000 | 0.84 |
  | `sdxl-epic-realism` | 15,000 | 0.93 |
  | `flux.1-schnell` | 10,000 | 0.98 |
  | `flux.2-dev` | 10,000 | 0.97 |
  | `gpt-image-1` | 10,000 | 0.97 |
  | `sd-1.5` | 10,000 | 0.97 |
  | `sdxl` | 10,000 | 0.95 |
  | `sdxl-touchofrealism` | 10,000 | 0.97 |
  | `grok-2-image-1212` | 9,303 | 0.98 |
  | `flux.2-klein-4b` | 7,459 | 0.98 |
  | `qwen-image` | 7,428 | 0.41 |
  | `chroma` | 5,643 | 0.74 |
  | `nano-banana` | 3,841 | 0.20 |
  | `stable-diffusion-xl-base-1.0` | 3,741 | 0.96 |
  | `anima-preview` | 3,698 | 0.97 |
  | `flux-amateursnapshotphotos` | 3,640 | 0.97 |
  | `seedream-v4.5` | 2,205 | 0.98 |
  | `kolors-v1.0` | 2,200 | 0.98 |
  | `nano-banana-2` | 1,985 | 0.97 |
  | `anima-preview2` | 1,880 | 0.95 |
  | `flux-realism` | 1,500 | 0.81 |
- **`tiny-random-sana` is excluded by default.** It is a HuggingFace
  tiny-random test stub whose output is uniform RGB noise, not generated
  imagery. It scored 0.000 recall, which looks like the biggest hole in the
  dataset and is in fact the opposite: our augmentation puts Gaussian noise
  on *real* images, so training on noise-as-fake is contradictory
  supervision. Rerun `fetch` with `--exclude` to change that set.
- Reals are OpenFake's LAION (ReLAION-5B, filtered to newsworthy/political)
  and Pexels (clean professional stock). Worth their 18% of real mass: the
  same checkpoint that sits at 0.7% FPR on our own val scored **6–9% FPR**
  on these, so they are a genuinely unseen real distribution, not padding.
- PNGs and oversized / non-RGB rows are re-encoded to JPEG q95 with the
  long side capped at 1536. Native JPEGs that already fit are written
  through: they are already JPEG, so they do not create the
  "PNG ⇒ fake" shortcut Community Forensics already has, and skipping
  the decode/re-encode is what keeps fetch from starving training.
- Licence: CC-BY-SA-4.0, but the proprietary-generator subsets are
  non-commercial (provider non-compete clauses).

### `gs-images-v3` / `gs-images-v4` — GAS-Station

- Hubs: `gasstation/gs-images-v3`, `gasstation/gs-images-v4`.
- Parquet has `archive_filename` + `file_path_in_archive`, **no** `image`
  column. Images sit in `archives/**/*.tar.gz`.
- `scripts/wire_gasstation.py --versions v3 v4 [--delete-archives]` unpacks
  into `wired/images/<model>/` and writes `wired/images.txt`. The mixture
  points at that listing. Until the listing exists the source is dropped.
  `unknown` means the parquet join missed (`unlabeled` in the wire log).
- **On disk v3:** 426,689 wired images (19 folders). **On disk v4:**
  113,793 (15 folders). Both listings are fake-only.

  | Generator | v3 | v4 |
  |---|---:|---:|
  | `Flux.2` | 190,278 | 44,113 |
  | `unknown` | 186,579 | 58,733 |
  | `Flux.1` | 9,576 | — |
  | `seedream-4-5` | — | 8,817 |
  | `diffusers_stable-diffusion-xl-1.0-inpainting-0.1` | 3,293 | 187 |
  | `deepseek-ai_Janus-Pro-7B` | 3,264 | 188 |
  | `Lykon_dreamshaper-8-inpainting` | 3,262 | 186 |
  | `SG161222_RealVisXL_V4.0` | 3,254 | 184 |
  | `THUDM_CogView4-6B` | 3,253 | 188 |
  | `cagliostrolab_animagine-xl-3.1` | 3,246 | 186 |
  | `lodestones_Chroma1-HD` | 3,243 | 187 |
  | `stabilityai_stable-diffusion-xl-base-1.0` | 3,242 | 184 |
  | `runwayml_stable-diffusion-v1-5-midjourney-v6` | 3,239 | 187 |
  | `prompthero_openjourney-v4` | 3,236 | 187 |
  | `DeepFloyd_IF` | 2,707 | 166 |
  | `black-forest-labs_FLUX.1-dev` | 2,011 | 100 |
  | `Image` | 1,410 | — |
  | `Adobe_Stock` | 786 | — |
  | `Gen4` | 498 | — |
  | `RunwayML` | 312 | — |

### `laion400m-1` — hosted LAION-400M images

- Hub: **`jp1924/Laion400m-1`** (gated). Images are already in the parquet
  `image` column. This is **not** a URL scrape of Re-LAION metadata.
- `scripts/download_laion400m.py` pulls a few ~10 GB shards, keeps
  `min(w,h) > 512`, writes JPEGs to `laion400m-1/real/`, deletes the shard.
  Do not snapshot the full 441-shard / ~4.4 TB dump.
- **On disk:** **199,998** JPEGs from 9 of 441 shards (~42 GB). Default
  pull is 20 shards / 400k images / 90 GB; resume continues from shard 9.
- Weight 0.128 matches `openfake`, so web-crawl reals are ~18% of real
  draws rather than an 11% footnote. This is the dedicated real-only
  source aimed at FPR on in-the-wild photographs.
- Do not use `scripts/download_relaion.py` for this source.

### `open-images-v7` — Open Images V7 reals

- `scripts/download_open_images.py` fetches the CVDF S3 **validation + test**
  JPEGs. **On disk:** **167,055** reals — 41,620 `validation/` + 125,435
  `test/`. Real-only; `_meta` CSVs are ignored.

## Held out (not in the train mix)

| Set | On disk | Role | Fetch |
|---|---|---|---|
| NTIRE val | 10,000 (5k/5k) | Primary labelled eval + per-distortion analysis | `python get_datasets.py --only ntire-val` |
| NTIRE val-hard | 2,500 (1,250/1,250) | Distorted val | same |
| NTIRE public test | 2,500 (1,200 real / 1,300 fake) | Unseen proprietary generators | `python get_datasets.py --only ntire-test` |
| OpenFake `core/test` | not pulled | Generators **and** real sources both unseen (`gpt-image-1.5/2`, `nano-banana-pro`, `flux.2-klein-9b`, `z-image-turbo`, `midjourney-7`, `ideogram-2.0`, `recraft-v2/v3`, `sora-2`, `veo-3` vs DOCCI + ImageNet reals) | `scripts/openfake.py holdout --config core` |
| OpenFake `reddit/test` | not pulled | In-the-wild: AI subreddits vs photography subreddits | `scripts/openfake.py holdout --config reddit` |
| CommunityForensics-Eval | 51,836 (25,918/25,918), 21 gens | Pangram's evaluation protocol | streamed at `--dataset comfor_eval` |
| COCO val2017 | 5,000 reals | Real half of the organisers' reference pair | `python get_datasets.py --only coco-val2017` |
| MIRAGE | 12,073 (10,682 fake / 1,391 real) | Small human-verified in-the-wild eval | `python get_datasets.py --only mirage` |
| WikiArt / other real folders | — | FPR-only harness | `--dataset folders --real-dir …` |
| Synthbuster + RAISE | — | Optional 9-family eval (not a train source) | `scripts/download_synthbuster.py` |

CommunityForensics-Eval fakes (paired 1:1 with reals under the same
`model_name`): Hourglass 2,000, MidjourneyV6_1 1,999, MidjourneyV5_2 1,993,
Firefly_Image3 1,948, Firefly_Image2 1,921, Imagen3 1,057, and 1,000 each
of DFGAN, stable_cascade, GALIP, LCM_lora_sdxl, kvikontent_midjourney_v6,
Dalle2, Dalle3, LCM_lora_sdv15, DeciDiffusionV2, FLUX-dev, FLUX-schnell,
IdeogramV2, IdeogramV1, kandinsky_2_2, LCM_lora_ssd1b.

MIRAGE `source` codes (not generator names): T2I 3,391, RMG 2,499, IID
1,681, OOD-R 1,202, IP/OP 990, IE 814, PCRMG 565, TR 427, CB 286, FS 218.

## In the registry, not in the hero mix

These exist in `src/seer/datasets_registry.py` / `get_datasets.py` but are
**not** `data.sources` entries:

- **DDA-Training-Set** — COCO VAE reconstructions; 11-part zip. `get_datasets.py`
  still prints `python src/scripts/build_dda_pairs.py`; that script is **not**
  in the repo. Do not use DDA for the hero mix.
- **Synthbuster** — eval / optional `--dataset folders` source, not a
  `--dataset synthbuster` name. Fetch with `scripts/download_synthbuster.py`.
- **WildFake DALL-E** — ModelScope; organisers' fake reference half.
- **CIFAKE** — 32×32 smoke test only. Never train a deployable detector on it.

## Acquisition cheat sheet

```bash
export SEER_DATA_ROOT=/path/to/data   # required on macOS/Linux; else F:/techjam (or /workspace/data)

python get_datasets.py --list
python get_datasets.py --only ntire-train ntire-val ntire-test coco-val2017

uv run scripts/fetch_data.py comfor-small
uv run scripts/fetch_data.py frontier-fakes
# streamed (optional local slices):
uv run scripts/fetch_data.py flux-reason-6m --max-shards 8
uv run scripts/fetch_data.py sid-set --max-shards 16

uv run scripts/wire_gasstation.py --versions v3 v4 --delete-archives
uv run scripts/download_laion400m.py --max-shards 20 --max-images 400000 --min-side 512
uv run scripts/download_open_images.py --workers 32 --max-gb 70

# OpenFake: rank first, then fetch only the generators that are still holes
uv run scripts/openfake.py probe --shards 3
uv run scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
uv run scripts/openfake.py fetch --from-rank $SEER_DATA_ROOT/openfake/rank.json \
    --labels fake real --tier 0.70=25000 0.95=15000 0.98=10000 \
    --cap-model pexels=80000 laion=50000
uv run scripts/openfake.py holdout --config core     # eval only
uv run scripts/openfake.py holdout --config reddit   # eval only
```

`openfake.py` and `openfake_rank.py` keep their worker counts low on
purpose: they are meant to run *alongside* a training job rather than
instead of one.

Inspect remote metadata without pulling images: `python dataset_stats.py`.
Edit weights only in the two yaml configs above; keep them in lockstep.
