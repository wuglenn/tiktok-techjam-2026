"""Download a cleaned-LAION real-photo slice from metadata URL lists.

Prefers Re-LAION research-safe parquet; falls back to other public LAION
metadata dumps. Images are fetched over HTTP (many source URLs are dead —
that is expected). Already-present files are skipped.

  python scripts/download_relaion.py
  python scripts/download_relaion.py --max-images 80000 --max-gb 40
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url

# Prefer gated Re-LAION once the Hub account has accepted the terms.
CANDIDATES = (
    "laion/relaion2B-en-research-safe",
    "UCSC-VLAA/Recap-DataComp-1B",
    "mlfoundations/datacomp_1b",
    "kakaobrain/coyo-700m",
)
UA = "seer-relaion-fetch/1.0 (research; +https://huggingface.co)"
JPEG_MAGIC = b"\xff\xd8"
PNG_MAGIC = b"\x89PNG"
URL_COLS = ("URL", "url", "IMAGE_URL", "image_url", "link")
WIDTH_COLS = ("WIDTH", "width")
HEIGHT_COLS = ("HEIGHT", "height")
UNSAFE_COLS = ("punsafe", "PUNSAFE", "nsfw")
WATER_COLS = ("pwatermark", "PWATERMARK")


def _out_root() -> Path:
    env = os.environ.get("SEER_DATA_ROOT")
    if env:
        return Path(env) / "relaion-slice"
    workspace = Path("/workspace/data")
    if workspace.is_dir():
        return workspace / "relaion-slice"
    return Path("relaion-slice")


def _parquet_names(files: list[str]) -> list[str]:
    names = [
        f
        for f in files
        if f.endswith(".parquet") and not f.startswith(".") and "preview" not in f.lower()
    ]
    names.sort()
    train = [f for f in names if "train" in f.lower()]
    return train or names


def _pick_repo(token: str | None) -> tuple[str, list[str]]:
    api = HfApi(token=token)
    last_err: Exception | None = None
    for repo in CANDIDATES:
        try:
            files = _parquet_names(api.list_repo_files(repo, repo_type="dataset"))
            if not files:
                continue
            # list_repo_files can succeed on a gated repo the token cannot download
            get_hf_file_metadata(hf_hub_url(repo, files[0], repo_type="dataset"), token=token)
            print(f"[relaion] using {repo} ({len(files)} parquet files)", flush=True)
            return repo, files
        except Exception as exc:  # gated / missing / network
            last_err = exc
            print(f"[relaion] skip {repo}: {exc}", flush=True)
    raise SystemExit(f"no usable LAION metadata repo (last error: {last_err})")


def _col(batch, names):
    for name in names:
        if name in batch.schema.names:
            return name
    return None


def _iter_urls(parquet_path: Path, min_side: int, max_unsafe: float, max_watermark: float):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    url_col = width_col = height_col = unsafe_col = water_col = None
    for batch in pf.iter_batches(batch_size=4096):
        if url_col is None:
            url_col = _col(batch, URL_COLS)
            width_col = _col(batch, WIDTH_COLS)
            height_col = _col(batch, HEIGHT_COLS)
            unsafe_col = _col(batch, UNSAFE_COLS)
            water_col = _col(batch, WATER_COLS)
            if url_col is None:
                raise SystemExit(f"no URL column in {parquet_path.name}: {batch.schema.names}")
            print(
                f"[relaion] columns url={url_col} w={width_col} h={height_col} "
                f"unsafe={unsafe_col} watermark={water_col}",
                flush=True,
            )
        urls = batch.column(url_col).to_pylist()
        widths = batch.column(width_col).to_pylist() if width_col else [None] * len(urls)
        heights = batch.column(height_col).to_pylist() if height_col else [None] * len(urls)
        unsafes = batch.column(unsafe_col).to_pylist() if unsafe_col else [None] * len(urls)
        waters = batch.column(water_col).to_pylist() if water_col else [None] * len(urls)
        for url, w, h, unsafe, water in zip(urls, widths, heights, unsafes, waters):
            if not url or not isinstance(url, str) or not url.startswith("http"):
                continue
            if w is not None and h is not None:
                try:
                    if int(w) < min_side or int(h) < min_side:
                        continue
                except (TypeError, ValueError):
                    pass
            if unsafe is not None:
                try:
                    if float(unsafe) >= max_unsafe:
                        continue
                except (TypeError, ValueError):
                    pass
            if water is not None:
                try:
                    if float(water) >= max_watermark:
                        continue
                except (TypeError, ValueError):
                    pass
            yield url


def _looks_like_image(data: bytes) -> str | None:
    if len(data) < 2048:
        return None
    if data.startswith(JPEG_MAGIC):
        return ".jpg"
    if data.startswith(PNG_MAGIC):
        return ".png"
    if data[:4] in (b"RIFF",) and b"WEBP" in data[:16]:
        return ".webp"
    return None


class Budget:
    def __init__(self, max_images: int | None, max_bytes: int | None):
        self.max_images = max_images
        self.max_bytes = max_bytes
        self.ok = 0
        self.skip = 0
        self.fail = 0
        self.bytes = 0
        self.stop = False
        self.lock = threading.Lock()

    def should_stop(self) -> bool:
        with self.lock:
            if self.max_images is not None and self.ok >= self.max_images:
                self.stop = True
            if self.max_bytes is not None and self.bytes >= self.max_bytes:
                self.stop = True
            return self.stop

    def add_ok(self, n: int) -> None:
        with self.lock:
            self.ok += 1
            self.bytes += n

    def add_skip(self) -> None:
        with self.lock:
            self.skip += 1

    def add_fail(self) -> None:
        with self.lock:
            self.fail += 1


def _stem(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8", errors="replace")).hexdigest()[:20]
    return digest


def _download_one(url: str, dest_dir: Path, budget: Budget) -> None:
    if budget.should_stop():
        return
    stem = _stem(url)
    for ext in (".jpg", ".png", ".webp"):
        existing = dest_dir / f"{stem}{ext}"
        if existing.exists() and existing.stat().st_size > 2048:
            budget.add_skip()
            return
    tmp = dest_dir / f"{stem}.part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read(12_000_000)
        ext = _looks_like_image(data)
        if not ext:
            budget.add_fail()
            return
        dest = dest_dir / f"{stem}{ext}"
        tmp.write_bytes(data)
        tmp.replace(dest)
        budget.add_ok(len(data))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        budget.add_fail()
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None)
    p.add_argument("--repo", default=None, help="override metadata repo id")
    p.add_argument("--max-parquets", type=int, default=8)
    p.add_argument("--max-images", type=int, default=120000)
    p.add_argument("--max-gb", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--min-side", type=int, default=256)
    p.add_argument("--max-unsafe", type=float, default=0.5)
    p.add_argument("--max-watermark", type=float, default=0.8)
    p.add_argument("--scan-limit", type=int, default=800000, help="URL rows to consider before giving up")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface")) / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
            os.environ["HF_TOKEN"] = token

    out = Path(args.out) if args.out else _out_root()
    dest = out / "real"
    meta = out / "_meta"
    dest.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    if args.repo:
        api = HfApi(token=token)
        files = _parquet_names(api.list_repo_files(args.repo, repo_type="dataset"))
        repo, parquet_names = args.repo, files
    else:
        repo, parquet_names = _pick_repo(token)
    wanted = parquet_names[: args.max_parquets]
    if not wanted:
        raise SystemExit(f"{repo} has no parquet files")

    local_parquets: list[Path] = []
    for name in wanted:
        print(f"[relaion] download metadata {name}", flush=True)
        path = Path(
            hf_hub_download(
                repo_id=repo,
                filename=name,
                repo_type="dataset",
                local_dir=str(meta),
                token=token,
            )
        )
        local_parquets.append(path)

    # Subsample URLs so one shard does not dump a burst of near-duplicates.
    rng = random.Random(0)
    urls: list[str] = []
    for path in local_parquets:
        for url in _iter_urls(path, args.min_side, args.max_unsafe, args.max_watermark):
            if rng.random() > 0.35:
                continue
            urls.append(url)
            if len(urls) >= args.scan_limit:
                break
        if len(urls) >= args.scan_limit:
            break
    rng.shuffle(urls)
    print(f"[relaion] {len(urls)} candidate URLs -> {dest}", flush=True)

    budget = Budget(args.max_images, int(args.max_gb * 1e9) if args.max_gb else None)
    started = time.time()
    url_iter = iter(urls)
    inflight = set()
    done = 0
    window = max(64, args.workers * 8)

    def _submit(pool) -> None:
        while len(inflight) < window and not budget.should_stop():
            try:
                url = next(url_iter)
            except StopIteration:
                return
            inflight.add(pool.submit(_download_one, url, dest, budget))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        _submit(pool)
        while inflight:
            fut = next(as_completed(inflight))
            inflight.remove(fut)
            done += 1
            if done % 2000 == 0 or budget.should_stop():
                elapsed = max(1.0, time.time() - started)
                print(
                    f"[relaion] ok={budget.ok} skip={budget.skip} fail={budget.fail} "
                    f"{budget.bytes / 1e9:.1f} GB  {budget.ok / elapsed:.1f} img/s",
                    flush=True,
                )
            if not budget.should_stop():
                _submit(pool)

    print(
        f"[relaion] done ok={budget.ok} skip={budget.skip} fail={budget.fail} "
        f"{budget.bytes / 1e9:.1f} GB -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
