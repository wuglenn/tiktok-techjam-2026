"""Persistent Seer inference server for the dashboard.

Loads `best.pt` once, then keeps the model in memory and scores images over
localhost HTTP. The Next.js `/api/analyze` route prefers this over spawning
`seer_infer.py` on every upload.

  .venv/Scripts/python.exe client/scripts/seer_serve.py --checkpoint best.pt

Endpoints (bound to 127.0.0.1):

  GET  /health   {ok, checkpoint, device, res, ...}
  POST /analyze  JSON {"images": ["C:/.../a.jpg", ...]} -> same records as seer_infer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from seer_infer import load_runtime, score_images  # noqa: E402

STATE: dict[str, Any] = {
    "ready": False,
    "error": None,
    "model": None,
    "device": None,
    "meta": None,
    "checkpoint": None,
    "lock": threading.Lock(),
}


def _warmup(model, device, res: int) -> None:
    import torch

    dummy = torch.zeros(1, 3, res, res, device=device)
    with torch.no_grad():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_into_state(checkpoint: str, device: Optional[str] = None, res: Optional[int] = None) -> None:
    STATE["checkpoint"] = str(Path(checkpoint).resolve())
    STATE["ready"] = False
    STATE["error"] = None
    t0 = time.time()
    print(f"[serve] loading {STATE['checkpoint']} (CPU, then device)…", flush=True)
    try:
        model, dev, meta = load_runtime(checkpoint, device, res)
        print(f"[serve] weights on {dev} — warming up…", flush=True)
        _warmup(model, dev, meta["res"])
        STATE["model"] = model
        STATE["device"] = dev
        STATE["meta"] = meta
        STATE["ready"] = True
        extra = ""
        if dev.type == "cuda":
            import torch

            extra = f" | peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        print(
            f"[serve] ready in {time.time() - t0:.1f}s | "
            f"{meta.get('backbone')} | step {meta.get('step')} | "
            f"{meta['res']}px{extra}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface any load failure on /health
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[serve] FAILED: {STATE['error']}", flush=True)
        raise


class Handler(BaseHTTPRequestHandler):
    server_version = "SeerInfer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/", "/health", "/status"):
            self._json(404, {"error": "not found"})
            return
        if not STATE["ready"]:
            self._json(
                503,
                {
                    "ok": False,
                    "ready": False,
                    "checkpoint": STATE["checkpoint"],
                    "error": STATE["error"] or "model is still loading",
                },
            )
            return
        meta = STATE["meta"] or {}
        self._json(
            200,
            {
                "ok": True,
                "ready": True,
                "checkpoint": STATE["checkpoint"],
                "device": str(STATE["device"]),
                "backbone": meta.get("backbone"),
                "step": meta.get("step"),
                "param_count": meta.get("param_count"),
                "res": meta.get("res"),
            },
        )

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/analyze":
            self._json(404, {"error": "not found"})
            return
        if not STATE["ready"]:
            self._json(503, {"error": STATE["error"] or "model is still loading"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 8_000_000:
            self._json(400, {"error": "missing or oversized JSON body"})
            return
        try:
            payload = json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        images = payload.get("images") or payload.get("image") or []
        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list) or not images:
            self._json(400, {"error": "no images"})
            return
        missing = [p for p in images if not Path(p).is_file()]
        if missing:
            self._json(400, {"error": f"image not found: {missing[0]}"})
            return
        try:
            t0 = time.time()
            with STATE["lock"]:
                records = score_images(
                    STATE["model"], images, int(STATE["meta"]["res"]), STATE["device"]
                )
            elapsed = time.time() - t0
            print(f"[serve] scored {len(records)} image(s) in {elapsed:.2f}s", flush=True)
            self._json(200, records)
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Persistent Seer inference server")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default=None)
    p.add_argument("--res", type=int, default=None)
    args = p.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt.resolve()}")

    load_into_state(str(ckpt), args.device, args.res)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}  (GET /health, POST /analyze)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stop", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()
