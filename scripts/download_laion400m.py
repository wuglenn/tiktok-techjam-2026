"""Download a size-filtered slice of jp1924/Laion400m-1.

Images live in ~10 GB parquet shards (441 shards / ~4.4 TB). This does *not*
snapshot the repo. It pulls one shard at a time, keeps rows whose width and
height are both > --min-side, writes JPEGs, then deletes the shard.

  python scripts/download_laion400m.py
  python scripts/download_laion400m.py --max-shards 20 --max-images 400000 --min-side 512
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


REPO = "jp1924/Laion400m-1"


def _out_root() -> Path:
    env = os.environ.get("SEER_DATA_ROOT")
    if env:
        return Path(env) / "laion400m-1"
    workspace = Path("/workspace/data")
    if workspace.is_dir():
        return workspace / "laion400m-1"
    return Path("laion400m-1")


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    path = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface")) / "token"
    if path.exists():
        token = path.read_text().strip()
        os.environ["HF_TOKEN"] = token
        return token
    return None


def _parquet_names(token: str | None) -> list[str]:
    files = [
        f
        for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
        if f.endswith(".parquet") and "preview" not in f.lower()
    ]
    files.sort()
    return files


def _as_bytes(value) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        data = value.get("bytes")
        if data:
            return bytes(data)
        path = value.get("path")
        if path and Path(path).exists():
            return Path(path).read_bytes()
    return None


def _extract_shard(parquet_path: Path, dest: Path, min_side: int, remaining: int) -> dict:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    names = set(pf.schema_arrow.names)
    need = [c for c in ("id", "image", "width", "height") if c in names]
    if "image" not in need:
        raise SystemExit(f"{parquet_path.name} has no image column: {pf.schema_arrow.names}")

    stats = {"rows": 0, "kept": 0, "skip_size": 0, "skip_bad": 0, "bytes": 0}
    dest.mkdir(parents=True, exist_ok=True)

    for batch in pf.iter_batches(columns=need, batch_size=128):
        cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        stats["rows"] += n
        ids = cols.get("id") or list(range(stats["rows"] - n, stats["rows"]))
        images = cols["image"]
        widths = cols.get("width") or [None] * n
        heights = cols.get("height") or [None] * n
        for image_id, image, width, height in zip(ids, images, widths, heights):
            if remaining is not None and stats["kept"] >= remaining:
                return stats
            try:
                w = int(width) if width is not None else 0
                h = int(height) if height is not None else 0
            except (TypeError, ValueError):
                w = h = 0
            if w <= min_side or h <= min_side:
                stats["skip_size"] += 1
                continue
            raw = _as_bytes(image)
            if not raw:
                stats["skip_bad"] += 1
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                stats["skip_bad"] += 1
                continue
            if img.width <= min_side or img.height <= min_side:
                stats["skip_size"] += 1
                continue
            out = dest / f"{image_id}.jpg"
            if out.exists() and out.stat().st_size > 1024:
                stats["kept"] += 1
                stats["bytes"] += out.stat().st_size
                continue
            img.save(out, format="JPEG", quality=95)
            stats["kept"] += 1
            stats["bytes"] += out.stat().st_size
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None)
    p.add_argument("--min-side", type=int, default=512, help="keep images with width AND height > this")
    p.add_argument("--max-shards", type=int, default=20, help="do not pull the full 441-shard dump")
    p.add_argument("--max-images", type=int, default=400000)
    p.add_argument("--max-gb", type=float, default=90.0)
    p.add_argument("--start-shard", type=int, default=0)
    args = p.parse_args()

    token = _token()
    out = Path(args.out) if args.out else _out_root()
    dest = out / "real"
    stage = out / "_shards"
    meta = out / "_meta"
    dest.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    files = _parquet_names(token)
    wanted = files[args.start_shard : args.start_shard + args.max_shards]
    print(
        f"[laion400m] {REPO}: {len(wanted)}/{len(files)} shards, "
        f"keep min(w,h)>{args.min_side}, cap={args.max_images} / {args.max_gb} GB -> {dest}",
        flush=True,
    )

    kept_total = sum(1 for p in dest.glob("*.jpg") if p.stat().st_size > 1024)
    bytes_total = sum(p.stat().st_size for p in dest.glob("*.jpg") if p.stat().st_size > 1024)
    max_bytes = int(args.max_gb * 1e9) if args.max_gb else None
    if kept_total:
        print(f"[laion400m] resume: {kept_total} images already on disk", flush=True)

    for index, name in enumerate(wanted, 1):
        if kept_total >= args.max_images:
            break
        if max_bytes is not None and bytes_total >= max_bytes:
            break
        marker = meta / (Path(name).name + ".done")
        if marker.exists():
            print(f"[laion400m] skip {name} (already extracted)", flush=True)
            continue
        print(f"[laion400m] ({index}/{len(wanted)}) download {name}", flush=True)
        local = Path(
            hf_hub_download(
                repo_id=REPO,
                filename=name,
                repo_type="dataset",
                local_dir=str(stage),
                token=token,
            )
        )
        remaining = args.max_images - kept_total
        stats = _extract_shard(local, dest, args.min_side, remaining)
        kept_total += stats["kept"]
        bytes_total += stats["bytes"]
        print(
            f"[laion400m] {name}: rows={stats['rows']} kept={stats['kept']} "
            f"skip_size={stats['skip_size']} skip_bad={stats['skip_bad']} "
            f"running={kept_total} {bytes_total / 1e9:.1f} GB",
            flush=True,
        )
        marker.write_text(
            f"rows={stats['rows']} kept={stats['kept']} skip_size={stats['skip_size']}\n"
        )
        try:
            local.unlink()
        except OSError:
            pass

    print(f"[laion400m] done kept={kept_total} {bytes_total / 1e9:.1f} GB -> {dest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
