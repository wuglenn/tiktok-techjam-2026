# Seer dashboard

Next.js dashboard for the Seer AI-generated image detector (TikTok TechJam 2026,
Track 5). Dark-mode, Geist Sans, Tailwind CSS.

| Page | Deliverable it covers |
| --- | --- |
| `/analyze` | end-to-end demo — upload images, get P(AI) verdicts + per-patch heatmaps; exports the required `image_path` / `pred` JSON |
| `/robustness` | robustness evaluation summary — clean vs transformed performance across the benchmark perturbation protocol |
| `/errors` | error analysis note — most confident false positives / false negatives with heatmaps, plus trade-offs |

## Run it

```bash
cd client
npm install
npm run dev        # http://localhost:3000
```

Production: `npm run build && npm start`.

## Live inference vs demo mode

The dashboard works in two modes and labels itself honestly in either:

- **Live** — when a checkpoint exists (env `SEER_CHECKPOINT`, else the newest
  `runs/*/best.pt`, preferring `seer_vitl*` runs) and a Python interpreter is
  found (`uv`, else the repo `.venv`). `/api/analyze` writes uploads to
  `.seer-tmp/`, spawns `scripts/seer_infer.py` via that interpreter with the
  repo root as cwd, and streams back `{prob_ai, label, grid}` per image —
  `grid` is the local head's raw patch probabilities.
- **Simulated** — otherwise `/api/analyze` returns deterministic simulated
  verdicts (seeded from the file bytes) so the demo still runs end-to-end. The
  UI marks every simulated result. Set `SEER_CHECKPOINT` or train a model to
  enable live mode.

`/robustness` and `/errors` read real eval JSONs from `runs/eval/*.json` and
`runs/*.json` (written by `main.py eval --out-json ...`, error panels by
`--error-dir`). With none present they show bundled demo data, clearly
labeled. To populate with real numbers:

```bash
uv run python main.py eval --checkpoint runs/seer_vitl/best.pt \
  --dataset ntire_val --perturbation all \
  --error-dir runs/eval/errors --error-n 6 \
  --out-json runs/eval/ntire_val.json
```

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
    api/eval/route.ts     # eval JSONs from runs/ (or demo data)
    api/eval-image/route.ts  # serves runs/** error panels (path-checked)
  components/             # app header, heat canvas, verdict widgets, charts
  lib/                    # shared types, turbo colormap, formatting, demo data
scripts/seer_infer.py     # JSON bridge into the seer package (spawned by the API)
```
