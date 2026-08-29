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

`MixtureDataset` draws each training example from source *i* with probability
`weight_i / sum(weights)`. Weights are relative, not required to sum to 1.

A source that raises `FileNotFoundError` or is empty is dropped for the rest
of the run (logged as `[data] dropping <name>`). That is how unwired
GAS-Station listings and still-empty real folders fail open.

Labels after mapping: **0 = real**, **1 = fake**. Composite overlays (25% of
steps) can change the page/patch labels on top of this mix; they are not a
separate source.

## Train sources

Current yaml weights sum to **1.20**. The share column is `weight / 1.20`.

| Source | Class | Weight | Share | On-disk / stream | How to materialize |
|---|---|---|---|---|---|
| `comfor` | mixed (paired real + fake) | 0.26 | 21.7% | `$SEER_DATA_ROOT/commfor-small` | `scripts/fetch_data.py comfor-small` |
| `ntire` | mixed (matched real + fake) | 0.30 | 25.0% | `$SEER_DATA_ROOT/ntire` | `python get_datasets.py --only ntire-train` |
| `flux-reason` | fake | 0.09 | 7.5% | stream `LucasFang/FLUX-Reason-6M` | optional slice: `scripts/fetch_data.py flux-reason-6m --max-shards 8` |
| `frontier-fakes` | fake only | 0.08 | 6.7% | `$SEER_DATA_ROOT/frontier-fakes` | `scripts/fetch_data.py frontier-fakes` |
| `sid-set` | fake only (full synthetic) | 0.06 | 5.0% | stream `saberzl/SID_Set` | optional slice: `scripts/fetch_data.py sid-set --max-shards 16` |
| `gs-images-v3` | fake | 0.10 | 8.3% | `$SEER_DATA_ROOT/gs-images-v3/wired/images.txt` | `scripts/wire_gasstation.py --versions v3` |
| `gs-images-v4` | fake | 0.11 | 9.2% | `$SEER_DATA_ROOT/gs-images-v4/wired/images.txt` | `scripts/wire_gasstation.py --versions v4` |
| `laion400m-1` | real | 0.10 | 8.3% | `$SEER_DATA_ROOT/laion400m-1/real` | `scripts/download_laion400m.py` |
| `open-images-v7` | real | 0.10 | 8.3% | `$SEER_DATA_ROOT/open-images-v7` | `scripts/download_open_images.py` |

Pure-fake sources are **36.7%** of draws, pure-real **16.7%**, mixed **46.7%**.
Community Forensics is roughly balanced; NTIRE train is real/fake matched per
shard but not exactly 50/50 overall. After that, expected class mass is
roughly **~40% real / ~60% fake** before composites.

Do not add a source above ~25% share without revisiting the others. Recency
is tilted toward NTIRE (42 generators, 2022–2026) and GAS-Station v4.

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
  default training streams it. Do not snapshot the repo.

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
```

Inspect remote metadata without pulling images: `python dataset_stats.py`.
Edit weights only in the two yaml configs above; keep them in lockstep.
