# TikTok TechJam 2026 Track 5 Deliverables

Official Track 5 checklist, then what this checkout actually contains.
Nothing below invents a video, licence, CI job, or test suite that is
not in the tree.

## Official requirements

1. Written Project Description (via Devpost)
- Provide a clear written description of your project that includes:
  - How your solution addresses the problem statement
  - Development tools used (e.g. VSCode, Colab, Jupyter)
  - Models or APIs used
  - Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
  - Datasets and assets used
2. Public Code/GitHub Repository
- Submit a link to a public Code/GitHub repository containing:
  - Well-structured, commented code covering all components of your solution
  - A script that takes an image directory as input and outputs a confidence score for each image, indicating the likelihood that it is AIGC-generated. The output should be a JSON file containing image_path and pred for each image.
  - A README file that includes:
    - Project overview
    - Setup and installation instructions
    - Steps to reproduce your results
    - A brief reflection on your solution's limitations and what you would improve given more time
    - Team member contributions (if applicable, i.e. team participants, non-solo participants)
3. Demo Video
- Submit a short video that:
  - Demonstrates your solution working end-to-end (e.g. inference results, dashboard, model predictions)
  - Is uploaded to YouTube and set to public visibility
  - Is linked in your Devpost description
  - Does not include third-party trademarks or copyrighted content without permission
4. Robustness Evaluation Summary
- Include a compact table or visual summary comparing performance on clean images versus transformed images.
5. Error Analysis Note
- Highlight representative false positives, false negatives, and any trade-offs in the proposed approach.

## What is in this repo

| Requirement | Status |
|---|---|
| Written project description | [`project_description.md`](../project_description.md) — Devpost writeup |
| Scoring script (`image_path` / `pred` JSON) | repo-root [`predict.py`](../predict.py) — official Track 5 entry |
| README (overview, setup, reproduce, limitations) | [`README.md`](../README.md) |
| Team member contributions | **Not in the repo.** No contributions section in the README. |
| Demo video / YouTube link | **Not in the repo.** No video file and no YouTube URL. |
| Robustness summary | Dashboard [`client/`](../client/) `/robustness` (clean vs transformed + NTIRE open-test). Committed NTIRE public-test JSON at [`eval/eval_step33500/ntire_test.json`](../eval/eval_step33500/ntire_test.json). |
| Error analysis | Dashboard `/errors` plus `--error-dir` on `main.py eval`. Committed suite includes a small WildFake DALL-E FN dump under `eval/eval_step33500/errors_dalle_advanced/`. Earlier writeup: [`deliverables/heldout-eval-step27500.md`](deliverables/heldout-eval-step27500.md). |
| Held-out scores | [`eval/eval_step33500/`](../eval/eval_step33500/) (step 33,500). Do not mix with the step-27,500 writeup. |
| Live demo | `client/` dashboard — `/`, `/analyze`, `/robustness`, `/errors` (see [`client/README.md`](../client/README.md)). Prefer `client/scripts/seer_serve.py` on **:8765**; Next is **:3000**. Upload limits: 12 images / 40 MB. Heatmaps work when a checkpoint with a local/patch head is loaded (`seer_probe.yaml` includes that head). Env: `SEER_CHECKPOINT`, `SEER_PYTHON`, `SEER_INFER_URL`, `HF_TOKEN`. `SEER_DATA_ROOT` is Python/data only, not the Next app. |
| Tests / CI / Docker / LICENSE | **None.** No `tests/`, no pytest suite, no `.github` workflows, no Dockerfile, no LICENSE file. |
| Weights | **Not in git.** Discovery: `$SEER_CHECKPOINT` → repo-root `best.pt` → newest `runs/*/best.pt`. Without a checkpoint, `/analyze` is SIMULATED. |
