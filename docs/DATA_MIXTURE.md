# Training data mixture

Canonical train mix for the hero recipe (`configs/seer_vitl_512.yaml`) and the
page-level probe (`configs/seer_probe.yaml`). Those two configs share the same
`data.sources` list and weights so the continuation vs probe comparison is
honest.

`seer_vitl_local.yaml` and `seer_vits_debug.yaml` do **not** use this mixture
(they fall back to a single Community Forensics stream).

Roots default to `$SEER_DATA_ROOT` (`/workspace/data` on the network volume,
otherwise `F:/techjam`). Paths below are written as `$SEER_DATA_ROOT/<name>`.

## How sampling works

`MixtureDataset` first draws a class (real vs fake) with equal probability
when `balance_labels` is on, then picks source *i* among sources that can
produce that class with probability `weight_i / sum(weights in class)`.
Weights are relative, not required to sum to 1.

A source that raises `FileNotFoundError` or is empty is dropped for the rest
of the run (logged as `[data] dropping <name>`). That is how unwired
GAS-Station listings and still-empty real folders fail open.

Labels after mapping: **0 = real**, **1 = fake**. Composite overlays (60% of
fake samples, tilted toward fake-on-real / real-on-fake) can change the
page/patch labels on top of this mix; they are not a separate source.

## Train sources

Current yaml weights sum to **1.18**. The share column is `weight / 1.18`
(source-draw share before `balance_labels`).

| Source | Class | Weight | Share | On-disk / stream | How to materialize |
|---|---|---|---|---|---|
| `comfor` | mixed (paired real + fake) | 0.22 | 18.6% | `$SEER_DATA_ROOT/commfor-small` | `scripts/fetch_data.py comfor-small` |
| `ntire` | mixed (matched real + fake) | 0.28 | 23.7% | `$SEER_DATA_ROOT/ntire` | `python get_datasets.py --only ntire-train` |
| `openfake` | mixed (selected fakes + reals) | 0.16 | 13.6% | `$SEER_DATA_ROOT/openfake/train` (~436k: 306k fake / 130k real) | `scripts/openfake.py fetch --from-rank …` |
| `flux-reason` | fake | 0.05 | 4.2% | stream `LucasFang/FLUX-Reason-6M` | optional slice: `scripts/fetch_data.py flux-reason-6m --max-shards 8` |
| `frontier-fakes` | fake only | 0.05 | 4.2% | `$SEER_DATA_ROOT/frontier-fakes` | `scripts/fetch_data.py frontier-fakes` |
| `sid-set` | fake only (full synthetic) | 0.05 | 4.2% | stream `saberzl/SID_Set` | optional slice: `scripts/fetch_data.py sid-set --max-shards 16` |
| `gs-images-v3` | fake | 0.09 | 7.6% | `$SEER_DATA_ROOT/gs-images-v3/wired/images.txt` | `scripts/wire_gasstation.py --versions v3` |
| `gs-images-v4` | fake | 0.10 | 8.5% | `$SEER_DATA_ROOT/gs-images-v4/wired/images.txt` | `scripts/wire_gasstation.py --versions v4` |
| `laion400m-1` | real | 0.09 | 7.6% | `$SEER_DATA_ROOT/laion400m-1/real` | `scripts/download_laion400m.py` |
| `open-images-v7` | real | 0.09 | 7.6% | `$SEER_DATA_ROOT/open-images-v7` | `scripts/download_open_images.py` |

Within-class shares (what a weight actually buys, since `balance_labels`
picks the class first):

| Class | Shares |
|---|---|
| fake | ntire 28%, comfor 22%, **openfake 16%**, gs-v4 10%, gs-v3 9%, flux-reason 5%, frontier-fakes 5%, sid-set 5% |
| real | ntire 33%, comfor 26%, **openfake 19%**, laion400m-1 11%, open-images-v7 11% |

Community Forensics is roughly balanced; NTIRE train is real/fake matched per
shard but not exactly 50/50 overall. `balance_labels` then forces **50/50
real/fake** batches; the weights above only decide which source is used once
the class is chosen.

Do not add a source above ~25% share without revisiting the others. Recency
is tilted toward NTIRE (42 generators, 2022–2026), OpenFake (frontier
commercial APIs through 2026) and GAS-Station v4.

## Per-source notes

### `comfor` — Community Forensics Small

- Hub: `OwensLab/CommunityForensics-Small` (train only).
- ~4,803 open generators (latent diffusion / pixel diffusion / GAN) plus
  paired reals. Strongest public driver of unseen-generator transfer.
- **Held out:** `OwensLab/CommunityForensics-Eval`. The loader refuses to
  train on that repo or any `comfor-eval` path.
- No commercial APIs (no DALL-E / Midjourney / FLUX). SD-derivative heavy;
  weight is kept below NTIRE so it does not dominate.
- Licence: CC BY-NC-SA 4.0. Shards are sorted by label (PNG⇒fake is a
  known confound); wild-simulation aug is mandatory.

### `ntire` — NTIRE 2026 train (all shards)

- Hub: `deepfakesMSU/NTIRE-RobustAIGenDetection-train`.
- `split: train`, `shard: -1` concatenates every downloaded shard (`shard_0`
  … `shard_5`). ~278k images, 42 generators, reals matched on resolution,
  aspect ratio, and JPEG quality.
- Labels live in per-shard `labels.csv`, not folder names.
- Val / test are **not** in this source (see held-out below).

### `flux-reason` — FLUX.1-dev

- Every row is fake (`label: 1`). Full dump is ~5.9M images / ~882 GB;
  default training streams it. Do not snapshot the repo. Weight is kept
  below `sid-set` so one 2024 family does not dominate fake draws.

### `frontier-fakes` — Midjourney / DALL-E / SD / Nano Banana Pro

- Hub: `julienlucas/midjourney-dalle-sd-nanobananapro-dataset`, train split.
- Upstream ClassLabel is inverted (`0=fake`, `1=real`). Yaml remaps then
  `keep_label: 1` so only fakes are trained. Train fakes are only ~5k;
  the 0.08 weight is already larger than that pool can sustain without
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
  already has 4,803 generators from `comfor`. This source exists to close a
  measured hole. The pipeline is:

  ```bash
  uv run scripts/openfake.py probe --shards 3          # per-generator sample
  uv run scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
  uv run scripts/openfake.py fetch --from-rank /workspace/data/openfake/rank.json \
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

- **On disk** (`$SEER_DATA_ROOT/openfake/train`, `core` / `train`):
  **435,553 images — 305,553 fake (30 generators) + 130,000 real**.
  Reals are at cap: Pexels 80,000 and LAION 50,000. Fakes are the ranked
  subset below; 15 of 30 hit their cap, the rest are supply-limited in
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
  | `flux.1-dev` | 15,000 | 0.94 |
  | `hidream-i1-full` | 15,000 | 0.91 |
  | `ideogram-3.0` | 15,000 | 0.85 |
  | `midjourney-6` | 15,000 | 0.94 |
  | `sd-1.5-epicdream` | 15,000 | 0.84 |
  | `sdxl-epic-realism` | 15,000 | 0.93 |
  | `flux-mvc5000` | 14,844 | 0.90 |
  | `mystic` | 14,247 | 0.89 |
  | `flux.1-schnell` | 10,000 | 0.98 |
  | `flux.2-dev` | 10,000 | 0.97 |
  | `gpt-image-1` | 10,000 | 0.97 |
  | `sd-1.5` | 10,000 | 0.97 |
  | `sdxl` | 10,000 | 0.95 |
  | `sdxl-touchofrealism` | 10,000 | 0.97 |
  | `grok-2-image-1212` | 8,759 | 0.98 |
  | `flux.2-klein-4b` | 7,057 | 0.98 |
  | `qwen-image` | 7,023 | 0.41 |
  | `chroma` | 5,337 | 0.74 |
  | `nano-banana` | 3,643 | 0.20 |
  | `stable-diffusion-xl-base-1.0` | 3,528 | 0.96 |
  | `anima-preview` | 3,461 | 0.97 |
  | `flux-amateursnapshotphotos` | 3,444 | 0.97 |
  | `seedream-v4.5` | 2,083 | 0.98 |
  | `kolors-v1.0` | 2,073 | 0.98 |
  | `nano-banana-2` | 1,847 | 0.97 |
  | `anima-preview2` | 1,780 | 0.95 |
  | `flux-realism` | 1,427 | 0.81 |
- **`tiny-random-sana` is excluded by default.** It is a HuggingFace
  tiny-random test stub whose output is uniform RGB noise, not generated
  imagery. It scored 0.000 recall, which looks like the biggest hole in the
  dataset and is in fact the opposite: our augmentation puts Gaussian noise
  on *real* images, so training on noise-as-fake is contradictory
  supervision. Rerun `fetch` with `--exclude` to change that set.
- Reals are OpenFake's LAION (ReLAION-5B, filtered to newsworthy/political)
  and Pexels (clean professional stock). Worth their 19% of real mass: the
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
  into `wired/images/` and writes `wired/images.txt`. The mixture points at
  that listing. Until the listing exists the source is dropped.

### `laion400m-1` — hosted LAION-400M images

- Hub: **`jp1924/Laion400m-1`** (gated). Images are already in the parquet
  `image` column. This is **not** a URL scrape of Re-LAION metadata.
- `scripts/download_laion400m.py` pulls a few ~10 GB shards, keeps
  `min(w,h) > 512`, writes JPEGs to `laion400m-1/real/`, deletes the shard.
  Do not snapshot the full 441-shard / ~4.4 TB dump.
- Do not use `scripts/download_relaion.py` for this source.

### `open-images-v7` — Open Images V7 reals

- `scripts/download_open_images.py` fetches the CVDF S3 **validation + test**
  JPEGs (~167k). The folders source scans `$SEER_DATA_ROOT/open-images-v7`
  recursively (`validation/`, `test/`). `_meta` CSVs are ignored.

## Held out (not in the train mix)

| Set | Role | Fetch |
|---|---|---|
| NTIRE val / val-hard | Primary labelled eval + per-distortion analysis | `python get_datasets.py --only ntire-val` |
| NTIRE public test | Unseen proprietary generators | `python get_datasets.py --only ntire-test` |
| OpenFake `core/test` | Generators **and** real sources both unseen (`gpt-image-1.5/2`, `nano-banana-pro`, `flux.2-klein-9b`, `z-image-turbo`, `midjourney-7`, `ideogram-2.0`, `recraft-v2/v3`, `sora-2`, `veo-3` vs DOCCI + ImageNet reals) | `scripts/openfake.py holdout --config core` |
| OpenFake `reddit/test` | In-the-wild: AI subreddits vs photography subreddits | `scripts/openfake.py holdout --config reddit` |
| CommunityForensics-Eval | Pangram CompEval protocol | streamed at `--dataset comfor_eval` |
| COCO val2017 | Real half of the organisers' reference pair | `python get_datasets.py --only coco-val2017` |
| MIRAGE | Small human-verified in-the-wild eval | `python get_datasets.py --only mirage` |
| WikiArt / other real folders | FPR-only harness | `--dataset folders --real-dir …` |
| Synthbuster + RAISE | Optional 9-family eval (not a train source) | `scripts/download_synthbuster.py` |

## In the registry, not in the hero mix

These exist in `src/seer/datasets_registry.py` / `get_datasets.py` but are
**not** `data.sources` entries:

- **DDA-Training-Set** — COCO VAE reconstructions; 11-part zip.
- **Synthbuster** — eval / optional folders source, not wired.
- **WildFake DALL-E** — ModelScope; organisers' fake reference half.
- **CIFAKE** — 32×32 smoke test only. Never train a deployable detector on it.

## Acquisition cheat sheet

```bash
export SEER_DATA_ROOT=/workspace/data   # or F:/techjam

python get_datasets.py --list
python get_datasets.py --only ntire-train ntire-val ntire-test coco-val2017

uv run scripts/fetch_data.py comfor-small
uv run scripts/fetch_data.py frontier-fakes
# streamed (optional local slices):
uv run scripts/fetch_data.py flux-reason-6m --max-shards 8
uv run scripts/fetch_data.py sid-set --max-shards 16

uv run scripts/wire_gasstation.py --versions v3 v4 --delete-archives
uv run scripts/download_laion400m.py --max-shards 12 --max-images 150000 --min-side 512
uv run scripts/download_open_images.py --workers 32 --max-gb 70

# OpenFake: rank first, then fetch only the generators that are still holes
uv run scripts/openfake.py probe --shards 3
uv run scripts/openfake_rank.py --checkpoint runs/seer_vitl/best.pt
uv run scripts/openfake.py fetch --from-rank /workspace/data/openfake/rank.json \
    --labels fake real --tier 0.70=25000 0.95=15000 0.98=10000 \
    --cap-model pexels=80000 laion=50000
uv run scripts/openfake.py holdout --config core     # eval only
uv run scripts/openfake.py holdout --config reddit   # eval only
```

`openfake.py` and `openfake_rank.py` both keep their worker counts low on
purpose: this container's cgroup allows ~13.6 CPUs, and they are meant to run
*alongside* a training job rather than instead of one.

Inspect remote metadata without pulling images: `python dataset_stats.py`.
Edit weights only in the two yaml configs above; keep them in lockstep.
