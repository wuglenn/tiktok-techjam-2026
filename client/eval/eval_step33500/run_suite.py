"""One-load eval suite for the current seer_vitl last.pt (step 33500)."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from seer.eval import _print_report, _single_pass, load_checkpoint

CKPT = Path("/workspace/tiktok-techjam-2026/runs/seer_vitl/last.pt")
OUT = Path("/workspace/tiktok-techjam-2026/runs/seer_vitl/eval_step33500")
COCO = "/workspace/data/coco-val2017"

JOBS = [
    dict(name="comfor_eval", dataset="comfor_eval"),
    dict(name="openfake_test", dataset="openfake_test", max_samples=0),
    dict(name="openfake_reddit", dataset="openfake_reddit", max_samples=0),
    dict(name="mirage", dataset="mirage"),
    dict(name="coco_val2017", dataset="folders", real_dirs=[COCO]),
    dict(name="ntire_test", dataset="ntire_test"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[suite] checkpoint={CKPT} device={device}", flush=True)
    model, cfg_dict, ckpt = load_checkpoint(str(CKPT), device=device)
    cfg_dict = dict(cfg_dict)
    cfg_dict["decode_workers"] = 16
    cfg_dict["prefetch_depth"] = 4
    res = int(cfg_dict.get("res", 512))
    step = ckpt.get("step")
    print(
        f"[suite] loaded step={step} res={res} backbone={cfg_dict.get('backbone')} "
        f"decode_workers={cfg_dict['decode_workers']} batch=32",
        flush=True,
    )
    model.eval()

    summary = []
    status = 0
    for job in JOBS:
        name = job["name"]
        t0 = time.time()
        print(f"\n[suite] start {name}", flush=True)
        try:
            metrics = _single_pass(
                model=model,
                cfg_dict=cfg_dict,
                perturbation=None,
                augmented=False,
                dataset=job["dataset"],
                batch_size=32,
                max_samples=job.get("max_samples"),
                hflip_tta=False,
                res=res,
                device=device,
                real_dirs=job.get("real_dirs"),
                fake_dirs=job.get("fake_dirs"),
            )
            metrics["checkpoint"] = str(CKPT)
            metrics["step"] = step
            metrics["suite_name"] = name
            metrics["seconds"] = round(time.time() - t0, 1)
            _print_report(metrics, name)
            out_json = OUT / f"{name}.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[suite] wrote {out_json} in {metrics['seconds']}s", flush=True)
            summary.append({
                "name": name,
                "n": metrics.get("n"),
                "n_fake": metrics.get("n_fake"),
                "n_real": metrics.get("n_real"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "macro_accuracy": metrics.get("macro_accuracy"),
                "mAP": metrics.get("mAP"),
                "auroc": metrics.get("auroc"),
                "f1": metrics.get("f1"),
                "fpr": metrics.get("fpr"),
                "fnr": metrics.get("fnr"),
                "seconds": metrics["seconds"],
            })
        except Exception as exc:
            status = 1
            print(f"[suite] FAIL {name}: {exc}", flush=True)
            traceback.print_exc()
            summary.append({"name": name, "error": str(exc)})

    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(CKPT), "step": step, "results": summary}, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
