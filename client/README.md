# Seer dashboard

Next.js dashboard for the Seer AI-generated image detector (TikTok TechJam 2026,
Track 5). Dark-mode, Geist Sans, Tailwind CSS. Node **≥ 20.9**.

| Page | Deliverable it covers |
| --- | --- |
| `/` | overview — architecture, held-out table, data mixture |
| `/analyze` | end-to-end demo — upload images, get P(AI) verdicts + per-patch heatmaps; exports `seer_predictions.json` (`image_path` / `pred`) |
| `/robustness` | robustness evaluation summary — clean vs transformed table + charts, plus the NTIRE 2026 open-test leaderboard |
| `/errors` | error analysis note — most confident false positives / false negatives with heatmaps, plus trade-offs |

## Run it

Weights are **not in git** (`*.pt` is gitignored). The scoring checkpoint is
[glennwuwu/seer](https://huggingface.co/glennwuwu/seer); from the repo root,
`uv run python predict.py --image-dir ./images` downloads `best.pt` on first
use. On localhost with a local model available, `/api/analyze` uses it
(prefer keeping the model in memory via `seer_serve.py`; otherwise it
respawns `client/scripts/seer_infer.py` per upload). Everywhere else it
scores on the Modal deployment (`SEER_MODAL_URL`).

```bash
# 1. put best.pt at the repo root (or export SEER_CHECKPOINT=/path/to/best.pt)
#    First-run: uv run python predict.py --image-dir ./images  (downloads Hub weights)

# 2. persistent inference server (bound to 127.0.0.1)
uv run python client/scripts/seer_serve.py --checkpoint best.pt
# http://127.0.0.1:8765  (override with SEER_INFER_URL)

# 3. dashboard
cd client
npm install
npm run dev        # http://localhost:3000
```

Production: `npm run build && npm start`.

## Modal deployment (remote GPU)

[`scripts/modal_seer.py`](scripts/modal_seer.py) deploys the detector to
[Modal](https://modal.com) so the dashboard can score images with **no local
weights, GPU, or Python environment**. It pulls `best.pt` from
[glennwuwu/seer](https://huggingface.co/glennwuwu/seer) into a Modal Volume,
keeps the model in GPU memory, and serves the same `/health` + `/analyze`
contract as `seer_serve.py` (images travel as base64 instead of local paths).

```bash
pip install modal
modal setup                                    # one-time auth
modal run client/scripts/modal_seer.py         # fill the weights Volume (~4.9 GB, once)
modal deploy client/scripts/modal_seer.py      # prints the endpoint URL

# then point the dashboard at it
export SEER_MODAL_URL=https://<workspace>--seer-seer-api.modal.run
cd client && npm run dev
```

`/api/analyze` scores on this deployment whenever the dashboard is **not**
running on localhost with a working local model — any hosted deployment uses
Modal by default. There is no simulated mode: if no backend is reachable, the
request fails with an error instead of returning fake verdicts. A cold
container boots on the first request (~1–2 min including the 4.9 GB load; the
Volume pre-fetch above is what keeps it off the GPU clock).

Containers default to an **L40S** GPU. `SEER_MODAL_GPU` overrides that at
deploy time with a single Modal GPU name (`T4`, `L4`, `A10G`, `L40S`,
`A100`, `H100`, …) or a comma-separated **pool** of names — the scheduler
places each container on the first listed type with capacity, so a deploy
keeps working when one card is out of stock:

```bash
SEER_MODAL_GPU=A10G modal deploy client/scripts/modal_seer.py          # one type
SEER_MODAL_GPU=T4,A10G,L40S modal deploy client/scripts/modal_seer.py  # pool
```

| Command | What it does |
| --- | --- |
| `modal deploy client/scripts/modal_seer.py` | persistent deployment; prints the URL to export |
| `modal serve client/scripts/modal_seer.py` | ephemeral dev deployment, live-reloads on file changes |
| `modal run client/scripts/modal_seer.py --image-path a.jpg` | smoke-test one image through the deployed class |
| `SEER_MODAL_GPU=… modal deploy …` | choose GPU(s) for the container (default `L40S`) |

The endpoint URL is public — anyone holding it can score images. For
restricted access use Modal's [proxy tokens](https://modal.com/docs/guide/webhooks#authentication)
or front it with your own auth.

| Port | Process |
| --- | --- |
| **3000** | `npm run dev` (Next.js) |
| **8765** | `client/scripts/seer_serve.py` |

| Variable | Default | Role |
| --- | --- | --- |
| `SEER_CHECKPOINT` | discovery below | live-inference weights |
| `SEER_PYTHON` | `uv run python`, else repo `.venv` | interpreter for `seer_infer.py` / `seer_serve.py` |
| `SEER_INFER_URL` | `http://127.0.0.1:8765` | persistent inference server |
| `SEER_MODAL_URL` | — | Modal deployment URL ([`scripts/modal_seer.py`](scripts/modal_seer.py)); the default backend away from localhost |
| `HF_TOKEN` | — | gated Hub access if the Python side has to fetch a backbone; not required once a local `.pt` is present |
| `SEER_DATA_ROOT` | `/workspace/data` if writable, else `F:/techjam` | **not** read by the Next app; only Python training / eval / fetch scripts |

Checkpoint discovery: `$SEER_CHECKPOINT`, then repo-root `best.pt`, then
the newest `runs/*/best.pt` (preferring `seer_vitl*` runs).

Upload limits: **12 images / 40 MB each**.

## Inference backends

`/api/analyze` picks exactly one backend — there is no simulated fallback:

- **Local** — only when the dashboard itself runs on localhost *and* a model
  is available: `seer_serve.py` on `:8765`, else a discoverable checkpoint
  plus a Python interpreter (`$SEER_PYTHON`, else `uv`, else the repo
  `.venv`). Nothing listening → uploads go to `.seer-tmp/` and
  `scripts/seer_infer.py` is spawned per request. Each image returns
  `{prob_ai, label, grid}` — `grid` is the local head's raw patch
  probabilities (heatmaps work; probe checkpoints include a patch head too).
- **Modal** — everything else, whenever `SEER_MODAL_URL` is set: hosted
  deployments score remotely with no local weights, interpreter, or repo
  root. Same `{prob_ai, label, grid}` records, same heatmap rendering.
- **Error** — no reachable backend: the API answers 503 with setup
  instructions rather than fake verdicts.

`/robustness` and `/errors` scan the committed suite bundled at
`client/eval/eval_step33500/` first, then `runs/eval/` and `runs/` from
the repo root (written by `main.py eval --out-json ...`, error panels by
`--error-dir`). With none present they show bundled demo data, clearly
labeled. To populate extra runs:

```bash
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt \
  --dataset ntire_val --perturbation all \
  --error-dir runs/eval/errors --error-n 6 \
  --out-json runs/eval/ntire_val.json
```

The committed step-33,500 suite is bundled at `client/eval/eval_step33500/`
and is what the dashboard picks up first.

## Layout

```
src/
  app/
    page.tsx              # overview — architecture, benchmarks, data mixture
    analyze/page.tsx      # upload + verdicts + heatmaps
    robustness/page.tsx   # clean vs transformed tables & charts
    errors/page.tsx       # FP/FN gallery + trade-offs note
    api/status/route.ts   # mode / checkpoint / interpreter probe
    api/analyze/route.ts  # POST images -> verdicts (local or Modal backend; errors otherwise)
    api/eval/route.ts     # eval JSONs from eval/eval_step33500, then runs/
    api/eval-image/route.ts  # serves error-panel PNGs (path-checked)
  components/             # app header, heat canvas, verdict widgets, charts
  lib/                    # shared types, turbo colormap, formatting, demo data
scripts/seer_serve.py     # persistent server on :8765 (preferred)
scripts/seer_infer.py     # one-shot JSON bridge (fallback spawn)
scripts/modal_seer.py     # Modal deployment — remote GPU backend (see above)
eval/eval_step33500/      # committed held-out suite: JSONs + error panels + run_suite.py
```
