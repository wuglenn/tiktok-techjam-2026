"""Modal deployment for the Seer AI-generated image detector.

Serves two checkpoints from huggingface.co/glennwuwu/seer: `best.pt`
provides the global P(AI) score and `heatmap.pt` provides the patch heatmap. The dashboard scores here by default
whenever it is not running on localhost with a working local model — see
client/README.md.

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
`seer-checkpoints` Modal Volume. The container's GPU is picked by the
SEER_MODAL_GPU env var at deploy time (default L40S): either one Modal GPU
name (T4, A10G, L40S, A100, H100, ...) or a comma-separated pool of names
that the scheduler fills from whichever type has capacity, e.g.
`SEER_MODAL_GPU=T4,A10G modal deploy ...`.
"""

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
SCORE_WEIGHTS = "best.pt"
HEATMAP_WEIGHTS = "heatmap.pt"
HF_WEIGHTS = (SCORE_WEIGHTS, HEATMAP_WEIGHTS)

def _repo_root() -> Optional[Path]:
    """Walk up from this file to the repo root (has src/seer + pyproject.toml).

    Returns None inside the Modal container: Modal copies this file to
    /root/modal_seer.py there, and the seer package is already baked into
    the image at /root/seer/src, so nothing needs bundling.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "src" / "seer" / "model.py"
        ).is_file():
            return parent
    return None


REPO_ROOT = _repo_root()
CHECKPOINT_DIR = Path("/checkpoints")
CHECKPOINT_PATHS = {
    filename: CHECKPOINT_DIR / filename for filename in HF_WEIGHTS
}


def _resolve_gpu_request() -> str | list[str]:
    """Parse SEER_MODAL_GPU into the Modal gpu= spec (evaluated at deploy).

    Defaults to T4, L4, and A10G (Modal's name for its A10-class GPU). A
    single name yields a plain str; comma-separated names yield a list, and
    Modal runs the container on the first listed type that has capacity.
    """
    raw = os.environ.get("SEER_MODAL_GPU", "T4,L4,A10G")
    specs = [s.strip() for s in raw.split(",") if s.strip()]
    if not specs:
        raise ValueError(
            "SEER_MODAL_GPU is set but empty; expected one GPU name or a list"
        )
    if any(c.isspace() for s in specs for c in s):
        raise ValueError(
            f"SEER_MODAL_GPU={raw!r}: separate GPU names with commas, "
            "e.g. SEER_MODAL_GPU=T4,L4,A10G"
        )
    return specs[0] if len(specs) == 1 else specs


GPU_REQUEST = _resolve_gpu_request()

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
    .env({"PYTHONPATH": "/root/seer/src", "SEER_DATA_ROOT": "/root/seer-data"})
)
if REPO_ROOT is not None:  # deploy-time only: bundle the seer package source
    image = image.add_local_dir(REPO_ROOT / "src", "/root/seer/src", copy=True)

app = modal.App("seer", image=image)


def _ensure_checkpoints() -> dict[str, Path]:
    """Fetch both model checkpoints into the shared Volume (idempotent)."""
    missing = [name for name, path in CHECKPOINT_PATHS.items() if not path.is_file()]
    if not missing:
        return CHECKPOINT_PATHS

    from huggingface_hub import hf_hub_download

    for filename in missing:
        print(
            f"[modal] downloading https://huggingface.co/{HF_REPO_ID}/{filename} "
            "(once)",
            flush=True,
        )
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=str(CHECKPOINT_DIR),
        )
    checkpoint_volume.commit()
    print(f"[modal] checkpoints stored on volume: {missing}", flush=True)
    return CHECKPOINT_PATHS


@app.function(volumes={CHECKPOINT_DIR: checkpoint_volume}, timeout=30 * 60)
def download_weights() -> dict[str, str]:
    """Pre-fetch both checkpoints so GPU containers never pay for downloads."""
    return {name: str(path) for name, path in _ensure_checkpoints().items()}


@app.cls(
    gpu=GPU_REQUEST,  # str = one type; list[str] = pool, first type with capacity
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    timeout=15 * 60,  # first boot may download + deserialize both checkpoints
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
        paths = _ensure_checkpoints()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, Any] = {}
        self.meta: dict[str, dict[str, Any]] = {}

        # Load one checkpoint at a time, discard its optimizer/EMA payload, and
        # move the live model to the GPU before deserializing the next ~5 GB file.
        for role, filename in (("score", SCORE_WEIGHTS), ("heatmap", HEATMAP_WEIGHTS)):
            path = paths[filename]
            print(
                f"[modal] loading {role} model from {path} (CPU, then GPU)…",
                flush=True,
            )
            model, cfg_dict, ckpt = load_checkpoint(path, device="cpu")
            self.meta[role] = {
                "checkpoint": filename,
                "backbone": ckpt.get("backbone_name"),
                "step": ckpt.get("step"),
                "param_count": ckpt.get("param_count")
                or sum(p.numel() for p in model.parameters()),
                "res": int(cfg_dict.get("res", 512)),
            }
            for key in ("model", "ema", "optimizer", "scheduler"):
                ckpt[key] = None
            del ckpt
            gc.collect()
            self.models[role] = model.to(self.device).eval()

        self.lock = threading.Lock()
        self.autocast_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

        with torch.no_grad():
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.autocast_bf16
            ):
                for role, model in self.models.items():
                    res = self.meta[role]["res"]
                    dummy = torch.zeros(1, 3, res, res, device=self.device)
                    model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(
            f"[modal] ready in {time.time() - t0:.1f}s | "
            f"score={self.meta['score']['checkpoint']}@{self.meta['score']['step']} | "
            f"heatmap={self.meta['heatmap']['checkpoint']}@{self.meta['heatmap']['step']} | "
            f"{self.device}",
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

        score_res = self.meta["score"]["res"]
        heatmap_res = self.meta["heatmap"]["res"]
        min_res = max(score_res, heatmap_res)
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
            if min(w, h) < min_res:
                s = min_res / min(w, h)
                img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)

            score_x = eval_transform(img, score_res)[None].to(self.device)
            heatmap_x = (
                score_x
                if heatmap_res == score_res
                else eval_transform(img, heatmap_res)[None].to(self.device)
            )
            with self.lock, torch.no_grad():
                with torch.autocast(
                    self.device.type, dtype=torch.bfloat16, enabled=self.autocast_bf16
                ):
                    score_out = self.models["score"](score_x)
                    heatmap_out = self.models["heatmap"](heatmap_x)

            prob = torch.sigmoid(score_out["logits"][0].float()).item()
            grid = None
            patch = heatmap_out.get("patch_logits")
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

        web_app = FastAPI(title="Seer", docs_url=None, redoc_url=None, openapi_url=None)

        @web_app.get("/health")
        async def health() -> dict:
            return {
                "ok": True,
                "ready": True,
                "revision": 3,
                "checkpoint": f"{HF_REPO_ID}/{SCORE_WEIGHTS} (score) + "
                f"{HF_REPO_ID}/{HEATMAP_WEIGHTS} (heatmap)",
                "device": str(self.device),
                "backbone": self.meta["score"].get("backbone"),
                "step": self.meta["score"].get("step"),
                "param_count": sum(m["param_count"] for m in self.meta.values()),
                "res": self.meta["score"].get("res"),
                "heatmap_res": self.meta["heatmap"].get("res"),
                "models": self.meta,
            }

        @web_app.post("/analyze")
        def analyze(payload: dict):
            # plain dict body (no pydantic models): closure-local model classes
            # are fragile under FastAPI's annotation resolution
            images = payload.get("images") or payload.get("image") or []
            if isinstance(images, dict):
                images = [images]
            if not isinstance(images, list) or not images:
                return JSONResponse(status_code=400, content={"error": "no images"})
            for im in images:
                if not isinstance(im, dict) or not isinstance(im.get("data"), str) or not im.get("data"):
                    return JSONResponse(
                        status_code=400,
                        content={"error": 'each image needs {"name", "data": "<base64>"}'},
                    )
            try:
                t0 = time.time()
                records = self._score(images)
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
        print(f"[smoke] gpu request: {GPU_REQUEST}")
        print("[smoke] deploy with: modal deploy client/scripts/modal_seer.py")
        return
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    records = Seer().score.remote([{"name": Path(image_path).name, "data": data}])
    print(json.dumps(records, indent=2))
