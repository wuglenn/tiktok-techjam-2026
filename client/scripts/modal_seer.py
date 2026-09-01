"""Modal deployment for the Seer AI-generated image detector.

Serves the scoring checkpoint (huggingface.co/glennwuwu/seer, `best.pt`,
~4.9 GB) from a Modal GPU container. The dashboard talks to it through the
"Score on Modal" flag on /analyze — see client/README.md.

The HTTP contract mirrors client/scripts/seer_serve.py so the dashboard's
health probe works unchanged, except images travel as base64 instead of
local file paths (remote containers cannot read the upload directory):

  GET  /health   {ok, ready, checkpoint, device, backbone, step, res, ...}
  POST /analyze  {"images": [{"name": "a.jpg", "data": "<base64>"}]}
             ->  [{"image", "prob_ai", "label", "grid", "width", "height"}, ...]

Deploy (docs: https://modal.com/docs):

  pip install modal
  modal setup                                             # one-time auth
  modal run client/scripts/modal_seer.py                 # fill the weights Volume
  modal deploy client/scripts/modal_seer.py              # prints the endpoint URL
  export SEER_MODAL_URL=<that url>                       # then start the dashboard

Dev loop: `modal serve client/scripts/modal_seer.py` runs an ephemeral app
that live-reloads on file changes. Weights are cached in the
`seer-checkpoints` Modal Volume; GPU type defaults to L40S and can be
changed at deploy time, e.g. `SEER_MODAL_GPU=A100 modal deploy ...`.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import modal

HF_REPO_ID = "glennwuwu/seer"
HF_WEIGHTS = "best.pt"  # ~4.9 GB: model + EMA + optimizer

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path("/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / HF_WEIGHTS

# L40S is Modal's cost/performance pick for inference (48 GB, bf16-capable).
# Any Modal GPU string works; evaluated when `modal deploy` runs.
GPU = os.environ.get("SEER_MODAL_GPU", "L40S")

checkpoint_volume = modal.Volume.from_name("seer-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6,<3",
        "transformers>=4.56,<6",  # DINOv3 (dinov3_vit) support
        "pillow>=10.4,<12",
        "numpy>=1.26,<3",
        "huggingface-hub>=0.30",
        "fastapi",
    )
    # the seer package: model code + the bundled DINOv3 config, so no gated
    # DINOv3 Hub access is needed to rebuild the architecture
    .add_local_dir(REPO_ROOT / "src", "/root/seer/src", copy=True)
    .env({"PYTHONPATH": "/root/seer/src", "SEER_DATA_ROOT": "/root/seer-data"})
)

app = modal.App("seer", image=image)


def _ensure_checkpoint() -> Path:
    """Fetch best.pt from the Hugging Face Hub into the Volume (idempotent)."""
    if CHECKPOINT_PATH.is_file():
        return CHECKPOINT_PATH
    from huggingface_hub import hf_hub_download

    print(
        f"[modal] downloading https://huggingface.co/{HF_REPO_ID}/{HF_WEIGHTS} "
        "(~4.9 GB, once)",
        flush=True,
    )
    hf_hub_download(repo_id=HF_REPO_ID, filename=HF_WEIGHTS, local_dir=str(CHECKPOINT_DIR))
    checkpoint_volume.commit()
    print(f"[modal] checkpoint stored on volume: {CHECKPOINT_PATH}", flush=True)
    return CHECKPOINT_PATH


@app.function(volumes={CHECKPOINT_DIR: checkpoint_volume}, timeout=30 * 60)
def download_weights() -> str:
    """Pre-fetch best.pt into the Volume so GPU containers never pay for it."""
    return str(_ensure_checkpoint())


@app.cls(
    gpu=GPU,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    timeout=15 * 60,  # first boot may download + deserialize the 4.9 GB blob
    scaledown_window=5 * 60,  # keep one container warm for demo traffic
    max_containers=3,  # cap GPU spend; extra requests queue
    cpu=4,
    memory=(8 * 1024, 16 * 1024),  # torch.load of the full blob peaks ~12 GB
)
@modal.concurrent(max_inputs=8)
class Seer:
    """Seer detector served over HTTP (same contract as seer_serve.py)."""

    @modal.enter()
    def load(self) -> None:
        import gc

        import torch

        from seer.model import load_checkpoint

        t0 = time.time()
        path = _ensure_checkpoint()
        print(f"[modal] loading {path} (CPU, then GPU)…", flush=True)
        model, cfg_dict, ckpt = load_checkpoint(path, device="cpu")
        self.meta: dict[str, Any] = {
            "backbone": ckpt.get("backbone_name"),
            "step": ckpt.get("step"),
            "param_count": ckpt.get("param_count")
            or sum(p.numel() for p in model.parameters()),
            "res": int(cfg_dict.get("res", 512)),
        }
        # A TechJam best.pt keeps EMA/optimizer alongside the live weights;
        # drop them before moving to the GPU (mirrors seer_infer.load_runtime).
        for key in ("model", "ema", "optimizer", "scheduler"):
            ckpt[key] = None
        del ckpt
        gc.collect()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.lock = threading.Lock()
        self.autocast_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

        dummy = torch.zeros(1, 3, self.meta["res"], self.meta["res"], device=self.device)
        with torch.no_grad():
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.autocast_bf16
            ):
                self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(
            f"[modal] ready in {time.time() - t0:.1f}s | {self.meta.get('backbone')} | "
            f"step {self.meta.get('step')} | {self.meta['res']}px | {self.device}",
            flush=True,
        )

    # ---------------------------------------------------------------- scoring

    def _score(self, images: list[dict]) -> list[dict[str, Any]]:
        """[{"name", "data": base64}] -> seer_serve.py-style records."""
        import numpy as np
        import torch
        from PIL import Image, ImageFile

        from seer.augment import eval_transform

        ImageFile.LOAD_TRUNCATED_IMAGES = True  # partial decodes beat crashes

        res = self.meta["res"]
        records: list[dict[str, Any]] = []
        for item in images:
            name = item.get("name") or "image"
            data = item.get("data") or ""
            if data.startswith("data:") and "," in data:  # tolerate data URLs
                data = data.split(",", 1)[1]
            try:
                raw = base64.b64decode(data, validate=False)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"{name}: invalid base64 image data ({exc})") from exc

            img = Image.open(io.BytesIO(raw)).convert("RGB")
            orig_w, orig_h = img.size
            w, h = orig_w, orig_h
            # mirror seer_infer.score_one: upscale <res inputs so every
            # image still gets a verdict
            if min(w, h) < res:
                s = res / min(w, h)
                img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)

            x = eval_transform(img, res)[None].to(self.device)
            with self.lock, torch.no_grad():
                with torch.autocast(
                    self.device.type, dtype=torch.bfloat16, enabled=self.autocast_bf16
                ):
                    out = self.model(x)

            prob = torch.sigmoid(out["logits"][0].float()).item()
            grid = None
            patch = out.get("patch_logits")
            if patch is not None:
                probs = torch.sigmoid(patch[0].float()).cpu().numpy()
                g = int(round(float(np.sqrt(probs.size))))
                grid = [[round(float(v), 4) for v in row] for row in probs.reshape(g, g)]

            records.append(
                {
                    "image": name,
                    "prob_ai": round(float(prob), 6),
                    "label": "AI" if prob >= 0.5 else "REAL",
                    "grid": grid,
                    "width": orig_w,
                    "height": orig_h,
                }
            )
        return records

    @modal.method()
    def score(self, images: list[dict]) -> list[dict[str, Any]]:
        """Programmatic entry point (also used by the smoke-test entrypoint)."""
        return self._score(images)

    # ------------------------------------------------------------------- HTTP

    @modal.asgi_app()
    def api(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel

        class ImageIn(BaseModel):
            name: str = "image"
            data: str  # base64 bytes, raw or as a data: URL

        class AnalyzeIn(BaseModel):
            images: list[ImageIn]

        web_app = FastAPI(title="Seer", docs_url=None, redoc_url=None, openapi_url=None)

        @web_app.get("/health")
        async def health() -> dict:
            return {
                "ok": True,
                "ready": True,
                "checkpoint": f"{HF_REPO_ID}/{HF_WEIGHTS} (Modal)",
                "device": str(self.device),
                "backbone": self.meta.get("backbone"),
                "step": self.meta.get("step"),
                "param_count": self.meta.get("param_count"),
                "res": self.meta.get("res"),
            }

        @web_app.post("/analyze")
        def analyze(payload: AnalyzeIn):
            try:
                t0 = time.time()
                records = self._score([im.model_dump() for im in payload.images])
                print(
                    f"[modal] scored {len(records)} image(s) in {time.time() - t0:.2f}s",
                    flush=True,
                )
                return records
            except Exception as exc:  # noqa: BLE001
                return JSONResponse(
                    status_code=500, content={"error": f"{type(exc).__name__}: {exc}"}
                )

        return web_app


@app.local_entrypoint()
def main(image_path: Optional[str] = None) -> None:
    """Smoke-test the deployment from your machine:

      modal run client/scripts/modal_seer.py                      # fill the Volume
      modal run client/scripts/modal_seer.py --image-path a.jpg   # score one image
    """
    if image_path is None:
        print(f"[smoke] checkpoint on volume: {download_weights.remote()}")
        print("[smoke] deploy with: modal deploy client/scripts/modal_seer.py")
        return
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    records = Seer().score.remote([{"name": Path(image_path).name, "data": data}])
    print(json.dumps(records, indent=2))
