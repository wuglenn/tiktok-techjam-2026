"""Score repo-root best.pt on glennwuwu/tiktok-techjam-2026-eval.

One checkpoint load, then clean + Pangram augmented + the official
benchmark perturbation table. Writes a JSON after each pass so a long
run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from seer.augment import BENCHMARK_PERTURBATIONS
from seer.eval import _print_report, _print_robustness_table, _single_pass
from seer.model import load_checkpoint

CKPT = Path(__file__).resolve().parents[2] / "best.pt"
OUT = Path(__file__).resolve().parent
# clean + Pangram protocol, then the official JPEG/blur/resize/noise/jitter/crop table.
PASSES = ["clean", "pangram"] + [k for k in BENCHMARK_PERTURBATIONS if k != "clean"]
METRIC_KEYS = (
    "n", "n_fake", "n_real", "accuracy", "macro_accuracy", "precision",
    "recall", "f1", "fpr", "fnr", "auroc", "mAP",
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(CKPT))
    p.add_argument("--dataset", default="techjam_eval")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--out-dir", default=str(OUT))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[techjam] checkpoint={args.checkpoint} device={device}", flush=True)
    model, cfg_dict, ckpt = load_checkpoint(args.checkpoint, device=device)
    cfg_dict = dict(cfg_dict)
    cfg_dict["decode_workers"] = 8
    cfg_dict["prefetch_depth"] = 2
    res = int(cfg_dict.get("res", 512))
    step = ckpt.get("step")
    for key in ("optimizer", "scheduler", "ema", "model"):
        ckpt.pop(key, None)
    import gc
    gc.collect()
    print(
        f"[techjam] loaded step={step} res={res} backbone={cfg_dict.get('backbone')} "
        f"batch={args.batch_size}",
        flush=True,
    )
    model.eval()

    summary = []
    sweep = {}
    status = 0
    for name in PASSES:
        dest = out_dir / f"{name}.json"
        if dest.exists() and args.max_samples is None:
            print(f"[techjam] skip {name} (already have {dest})", flush=True)
            with dest.open(encoding="utf-8") as f:
                metrics = json.load(f)
            sweep[name] = {k: metrics.get(k) for k in METRIC_KEYS}
            summary.append({"name": name, **sweep[name], "skipped": True})
            continue
        t0 = time.time()
        print(f"\n[techjam] start {name}", flush=True)
        try:
            metrics = _single_pass(
                model=model,
                cfg_dict=cfg_dict,
                perturbation=None if name == "clean" else name,
                augmented=False,
                dataset=args.dataset,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                hflip_tta=False,
                res=res,
                device=device,
            )
            metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
            metrics["step"] = step
            metrics["seconds"] = round(time.time() - t0, 1)
            _print_report(metrics, args.dataset)
            dest.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
            print(f"[techjam] wrote {dest} in {metrics['seconds']}s", flush=True)
            row = {k: metrics.get(k) for k in METRIC_KEYS}
            row["seconds"] = metrics["seconds"]
            sweep[name] = row
            summary.append({"name": name, **row})
        except Exception as exc:
            status = 1
            print(f"[techjam] FAIL {name}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            summary.append({"name": name, "error": str(exc)})
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "step": step,
        "dataset": "glennwuwu/tiktok-techjam-2026-eval",
        "results": summary,
        "perturbation_sweep": sweep,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    if len(sweep) > 1:
        _print_robustness_table({k: v for k, v in sweep.items() if v.get("n")})
    print(json.dumps(_jsonable(summary), indent=2), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
