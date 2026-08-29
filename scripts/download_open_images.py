"""Download an Open Images V7 real-photo slice from the public CVDF S3.

Images live at:
  https://open-images-dataset.s3.amazonaws.com/{split}/{ImageID}.jpg

Default slice is the full validation + test splits (~167k JPEGs, typically
50-70 GB). Already-present files are skipped. Caps are optional.

  python scripts/download_open_images.py
  python scripts/download_open_images.py --splits validation --max-gb 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ID_CSVS = {
    "validation": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
    "train": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
}
S3 = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
UA = "seer-open-images-fetch/1.0 (research; +https://huggingface.co)"
JPEG_MAGIC = b"\xff\xd8"


def _out_root() -> Path:
    env = os.environ.get("SEER_DATA_ROOT")
    if env:
        return Path(env) / "open-images-v7"
    workspace = Path("/workspace/data")
    if workspace.is_dir():
        return workspace / "open-images-v7"
    return Path("open-images-v7")


def _fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _image_ids(split: str, cache_dir: Path) -> list[str]:
    cache = cache_dir / f"{split}-ids.csv"
    if not cache.exists():
        print(f"[open-images] fetching {split} id list", flush=True)
        cache.write_bytes(_fetch_bytes(ID_CSVS[split], timeout=180))
    ids: list[str] = []
    with cache.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = (row.get("ImageID") or row.get("image_id") or "").strip()
            if image_id:
                ids.append(image_id)
    return ids


def _looks_like_jpeg(data: bytes) -> bool:
    return len(data) > 1024 and data.startswith(JPEG_MAGIC)


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


def _download_one(split: str, image_id: str, dest: Path, budget: Budget) -> None:
    if budget.should_stop():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        budget.add_skip()
        return
    url = S3.format(split=split, image_id=image_id)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        data = _fetch_bytes(url, timeout=45)
        if not _looks_like_jpeg(data):
            budget.add_fail()
            return
        tmp.write_bytes(data)
        tmp.replace(dest)
        budget.add_ok(len(data))
    except (urllib.error.URLError, TimeoutError, OSError):
        budget.add_fail()
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="default: $SEER_DATA_ROOT/open-images-v7")
    p.add_argument("--splits", nargs="+", default=["validation", "test"], choices=list(ID_CSVS))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--max-gb", type=float, default=70.0)
    p.add_argument("--train-limit", type=int, default=80000, help="cap if train split is included")
    args = p.parse_args()

    out = Path(args.out) if args.out else _out_root()
    meta = out / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.max_images, int(args.max_gb * 1e9) if args.max_gb else None)
    started = time.time()

    jobs: list[tuple[str, str, Path]] = []
    for split in args.splits:
        ids = _image_ids(split, meta)
        if split == "train":
            ids = ids[: args.train_limit]
        print(f"[open-images] {split}: {len(ids)} ids -> {out / split}", flush=True)
        for image_id in ids:
            jobs.append((split, image_id, out / split / f"{image_id}.jpg"))

    print(f"[open-images] {len(jobs)} files, workers={args.workers}, cap={args.max_gb} GB", flush=True)
    job_iter = iter(jobs)
    inflight = set()
    done = 0
    window = max(64, args.workers * 8)

    def _submit(pool) -> None:
        while len(inflight) < window and not budget.should_stop():
            try:
                split, image_id, dest = next(job_iter)
            except StopIteration:
                return
            inflight.add(pool.submit(_download_one, split, image_id, dest, budget))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        _submit(pool)
        while inflight:
            fut = next(as_completed(inflight))
            inflight.remove(fut)
            done += 1
            if done % 1000 == 0 or budget.should_stop():
                elapsed = max(1.0, time.time() - started)
                print(
                    f"[open-images] ok={budget.ok} skip={budget.skip} fail={budget.fail} "
                    f"{budget.bytes / 1e9:.1f} GB  {budget.ok / elapsed:.1f} img/s",
                    flush=True,
                )
            if not budget.should_stop():
                _submit(pool)

    print(
        f"[open-images] done ok={budget.ok} skip={budget.skip} fail={budget.fail} "
        f"{budget.bytes / 1e9:.1f} GB -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    # Avoid leaking a huge CSV through stdout buffering on crash
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
