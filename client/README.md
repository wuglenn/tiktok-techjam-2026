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
use. Without a local or cached checkpoint, `/analyze` is **SIMULATED**.
Prefer keeping the model in memory; otherwise `/api/analyze` respawns
`client/scripts/seer_infer.py` on every upload.

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

| Port | Process |
| --- | --- |
| **3000** | `npm run dev` (Next.js) |
| **8765** | `client/scripts/seer_serve.py` |

| Variable | Default | Role |
| --- | --- | --- |
| `SEER_CHECKPOINT` | discovery below | live-inference weights |
| `SEER_PYTHON` | `uv run python`, else repo `.venv` | interpreter for `seer_infer.py` / `seer_serve.py` |
| `SEER_INFER_URL` | `http://127.0.0.1:8765` | persistent inference server |
| `HF_TOKEN` | — | gated Hub access if the Python side has to fetch a backbone; not required once a local `.pt` is present |
| `SEER_DATA_ROOT` | `/workspace/data` if writable, else `F:/techjam` | **not** read by the Next app; only Python training / eval / fetch scripts |

Checkpoint discovery: `$SEER_CHECKPOINT`, then repo-root `best.pt`, then
the newest `runs/*/best.pt` (preferring `seer_vitl*` runs).

Upload limits: **12 images / 40 MB each**.

## Live inference vs simulated mode

The dashboard works in two modes and labels itself honestly in either:

- **Live** — a checkpoint is found *and* a Python interpreter is found
  (`$SEER_PYTHON`, else `uv`, else the repo `.venv`). `/api/analyze`
  prefers `seer_serve.py` on `:8765`. If nothing is listening it writes
  uploads to `.seer-tmp/` and spawns `scripts/seer_infer.py` per request.
  Each image returns `{prob_ai, label, grid}` — `grid` is the local
  head's raw patch probabilities (heatmaps work; probe checkpoints
  include a patch head too).
- **Simulated** — no checkpoint (or no interpreter / no repo root).
  `/api/analyze` returns deterministic fake verdicts seeded from the file
  bytes. The UI marks every simulated result.

`/robustness` and `/errors` scan eval JSONs from `eval/eval_step33500/`
first, then `runs/eval/` and `runs/` (written by
`main.py eval --out-json ...`, error panels by `--error-dir`). With none
present they show bundled demo data, clearly labeled. To populate extra
runs:

```bash
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt \
  --dataset ntire_val --perturbation all \
  --error-dir runs/eval/errors --error-n 6 \
  --out-json runs/eval/ntire_val.json
```

The committed step-33,500 suite is already under `eval/eval_step33500/`
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
    api/analyze/route.ts  # POST images -> verdicts (live bridge or simulation)
    api/eval/route.ts     # eval JSONs from eval/eval_step33500, then runs/
    api/eval-image/route.ts  # serves error-panel PNGs (path-checked)
  components/             # app header, heat canvas, verdict widgets, charts
  lib/                    # shared types, turbo colormap, formatting, demo data
scripts/seer_serve.py     # persistent server on :8765 (preferred)
scripts/seer_infer.py     # one-shot JSON bridge (fallback spawn)
```
