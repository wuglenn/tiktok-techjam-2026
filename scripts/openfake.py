"""Selective fetch for ComplexDataLab/OpenFake (v2).

OpenFake is 3.44 TB across 645 parquet shards and every shard interleaves all
~80 generators, so neither filename selection nor parquet row-group pushdown
can isolate one model: the only way to get a single generator is to stream
shards and filter rows. This script streams shards and deletes each one after extract so the peak
footprint is a small prefetch window plus the JPEGs we keep. The next shard
downloads while the current one is extracted.

PNGs (and oversized / non-RGB JPEGs) are re-encoded to JPEG. Native JPEGs
that already fit ``max_side`` are written through — decoding them just to
write the same container back is the slow path, and they are already JPEG
so they do not create the "PNG => fake" shortcut Community Forensics has
(see docs/DATA_MIXTURE.md).

Output layout matches the mixture's ``folders`` source type, which reads the
generator name from the parent directory:

    <root>/openfake/<subset>/fake/<model>/<shard>_<name>.jpg
    <root>/openfake/<subset>/real/<source>/<shard>_<name>.jpg

Subcommands
  index    per-generator row counts from a stride of shards (keeps no images)
  probe    per-generator sample used to rank generators by difficulty
  fetch    the training pull: chosen generators, per-generator caps
  holdout  core/test + reddit/test, kept out of the train mixture

  python scripts/openfake.py probe --shards 3
  python scripts/openfake.py fetch --models ideogram-3.0 flux.2-klein-4b --cap 6000
  python scripts/openfake.py holdout --shards 2
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Semaphore, Thread

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "ComplexDataLab/OpenFake"
META_COLS = ("label", "model", "type", "release_date")
Image.MAX_IMAGE_PIXELS = None


def _out_root() -> Path:
    env = os.environ.get("SEER_DATA_ROOT")
    if env:
        return Path(env) / "openfake"
    workspace = Path("/workspace/data")
    if workspace.is_dir():
        return workspace / "openfake"
    return Path("openfake")


def _token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for base in (os.environ.get("HF_HOME"), "/workspace/.cache/huggingface",
                 str(Path.home() / ".cache/huggingface")):
        if not base:
            continue
        path = Path(base) / "token"
        if path.exists():
            token = path.read_text().strip()
            if token:
                os.environ["HF_TOKEN"] = token
                return token
    return None


def shard_files(config: str, split: str, token: str | None = None) -> list[str]:
    """Repo-relative parquet names for one config/split, in shard order."""
    files = HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
    prefix = f"{config}/{split}-"
    names = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not names:
        raise SystemExit(f"no shards for {config}/{split} in {REPO}")
    return names


def _stride(names: list[str], limit: int | None, stride: int, start: int) -> list[str]:
    picked = names[start::max(1, stride)]
    return picked[:limit] if limit else picked


def _as_bytes(value) -> bytes | None:
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


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("_", (text or "unknown").strip()) or "unknown"


def _shard_tag(name: str) -> str:
    return _slug(Path(name).stem.replace("-of-", "_"))


def _header_ok(raw: bytes, img: Image.Image, max_side: int, min_side: int) -> int:
    """0 = bad, -1 = too small, 1 = passthrough JPEG, 2 = must re-encode."""
    if min(img.width, img.height) < min_side:
        return -1
    if (img.format == "JPEG" and img.mode in ("RGB", "L")
            and raw[:2] == b"\xff\xd8"
            and (not max_side or max(img.width, img.height) <= max_side)):
        return 1
    return 2


def _write_bytes(raw: bytes, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(raw)
    tmp.replace(dest)
    return dest.stat().st_size


def _save_jpeg(raw: bytes, dest: Path, max_side: int, quality: int, min_side: int) -> int:
    """Write one row as a JPEG.

    Returns bytes written, -1 when the image is under ``min_side`` and 0 when
    it could not be decoded. Those two are worth telling apart: ~27% of
    OpenFake's LAION reals are thumbnails below 256px and dropping them is
    intended, whereas undecodable rows would mean we are losing data.

    A native JPEG that already fits ``max_side`` is copied as-is (header
    parse only). PNG / oversized / non-RGB rows still go through PIL.
    """
    try:
        img = Image.open(io.BytesIO(raw))
        kind = _header_ok(raw, img, max_side, min_side)
    except Exception:
        return 0
    if kind <= 0:
        return kind
    if kind == 1:
        return _write_bytes(raw, dest)
    try:
        img.load()
    except Exception:
        return 0
    if img.mode != "RGB":
        img = img.convert("RGB")
    long_side = max(img.width, img.height)
    if max_side and long_side > max_side:
        scale = max_side / float(long_side)
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    img.save(tmp, format="JPEG", quality=quality, subsampling=0)
    tmp.replace(dest)
    return dest.stat().st_size


def read_meta(path: Path) -> Counter:
    """(label, model) counts for one shard, image column never touched."""
    import pyarrow.parquet as pq

    table = pq.ParquetFile(path).read(columns=list(META_COLS))
    return Counter(zip(table.column("label").to_pylist(),
                       table.column("model").to_pylist()))


def extract_shard(
    path: Path,
    shard_name: str,
    dest_root: Path,
    *,
    wanted: set[str] | None,
    labels: set[str],
    caps: dict[str, int],
    counts: Counter,
    max_side: int,
    quality: int,
    min_side: int,
    workers: int,
    counts_lock: Lock | None = None,
) -> dict:
    """Write the wanted rows of one shard as JPEGs under dest_root."""
    import pyarrow.parquet as pq

    tag = _shard_tag(shard_name)
    stats = {"rows": 0, "kept": 0, "bytes": 0, "skip_model": 0, "skip_cap": 0,
             "skip_small": 0, "skip_bad": 0, "per_model": Counter()}
    jobs: list[tuple[bytes, Path, str]] = []
    lock = counts_lock or Lock()

    def flush() -> None:
        if not jobs:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            written = list(pool.map(
                lambda job: _save_jpeg(job[0], job[1], max_side, quality, min_side), jobs))
        refund: list[str] = []
        for (_, _, model), size in zip(jobs, written):
            if size > 0:
                stats["kept"] += 1
                stats["bytes"] += size
                stats["per_model"][model] += 1
            else:
                stats["skip_small" if size < 0 else "skip_bad"] += 1
                refund.append(model)
        if refund:
            with lock:
                for model in refund:
                    counts[model] -= 1
        jobs.clear()

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["image", *META_COLS], batch_size=64):
        cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        stats["rows"] += batch.num_rows
        for i in range(batch.num_rows):
            label = str(cols["label"][i])
            model = str(cols["model"][i])
            if label not in labels:
                continue
            if wanted is not None and model not in wanted:
                stats["skip_model"] += 1
                continue
            cap = caps.get(model, caps.get("*", 0))
            with lock:
                if cap and counts[model] >= cap:
                    stats["skip_cap"] += 1
                    continue
                counts[model] += 1
            raw = _as_bytes(cols["image"][i])
            if not raw:
                stats["skip_bad"] += 1
                with lock:
                    counts[model] -= 1
                continue
            name = Path(str(cols["image"][i].get("path") or f"{i:06d}")).stem
            dest = dest_root / label / _slug(model) / f"{tag}_{_slug(name)}.jpg"
            if dest.exists() and dest.stat().st_size > 1024:
                stats["kept"] += 1
                stats["per_model"][model] += 1
                continue
            jobs.append((raw, dest, model))
        if len(jobs) >= 256:
            flush()
    flush()
    return stats


# Upper bound on `recall_mean` from scripts/openfake_rank.py -> per-model cap.
# Still skip the already-solved (>=98%) generators, but take more of the ones
# we did select — leftover shards still have them, and a 2.5k–8k cap was
# burning most rows as skip_cap.
DEFAULT_TIERS = ((0.70, 25000), (0.95, 15000), (0.98, 10000))

# Not a generator: `tiny-random-sana` is a HuggingFace tiny-random test stub
# and emits uniform RGB noise. Training on it would put pure noise in the fake
# class while our augmentation puts Gaussian noise on real images - directly
# contradictory supervision.
DEFAULT_EXCLUDE = ("tiny-random-sana",)


def parse_tiers(specs: list[str] | None) -> tuple[tuple[float, int], ...]:
    if not specs:
        return DEFAULT_TIERS
    tiers = []
    for spec in specs:
        bound, _, cap = spec.partition("=")
        tiers.append((float(bound), int(cap)))
    return tuple(sorted(tiers))


def caps_from_rank(path: Path, tiers, exclude: set[str]) -> dict[str, int]:
    """Per-model caps from a rank.json, harder generators getting more mass."""
    data = json.loads(Path(path).read_text())
    caps: dict[str, int] = {}
    for row in data["generators"]:
        name = row["model"]
        if name in exclude:
            continue
        recall = row.get("recall_mean")
        if recall is None:
            continue
        for bound, cap in tiers:
            if recall < bound:
                caps[name] = cap
                break
    if not caps:
        raise SystemExit(f"no generator in {path} is below the tier bounds {tiers}")
    return caps


def _downloaded_shards(
    names: list[str],
    done: set[str],
    stage: Path,
    token: str | None,
    prefetch: int,
    download_workers: int,
    stop: Event,
):
    """Yield ``(name, local_path, release)`` while prefetching the next shards.

    ``prefetch`` is the max number of parquet files on disk at once
    (downloading + waiting + being extracted). Call ``release`` after extract
    (or after dropping the file) so a slot opens for the next download.
    """
    todo = [n for n in names if n not in done]
    if not todo:
        return
    ready: Queue = Queue()
    slots = Semaphore(max(1, prefetch))
    errors: list[BaseException] = []
    workers = max(1, download_workers)

    def worker(chunk: list[str]) -> None:
        for name in chunk:
            if stop.is_set() or errors:
                return
            slots.acquire()
            if stop.is_set() or errors:
                slots.release()
                return
            try:
                local = Path(hf_hub_download(
                    REPO, name, repo_type="dataset",
                    local_dir=str(stage), token=token))
            except BaseException as exc:
                errors.append(exc)
                ready.put(None)
                slots.release()
                return
            if stop.is_set():
                try:
                    local.unlink()
                except OSError:
                    pass
                slots.release()
                return
            ready.put((name, local))

    threads = [
        Thread(target=worker, args=(todo[i::workers],), daemon=True)
        for i in range(workers) if todo[i::workers]
    ]
    for thread in threads:
        thread.start()

    received = 0
    try:
        while received < len(todo) and not stop.is_set():
            item = ready.get()
            if item is None:
                if errors:
                    raise errors[0]
                continue
            received += 1
            name, local = item
            yield name, local, slots.release
    finally:
        stop.set()
        while True:
            try:
                leftover = ready.get_nowait()
            except Empty:
                break
            if leftover:
                try:
                    leftover[1].unlink()
                except OSError:
                    pass


def caps_satisfied(caps: dict[str, int], counts: Counter, wanted: set[str] | None) -> bool:
    if not caps or "*" in caps:
        return False
    if wanted and not wanted.issubset(caps.keys()):
        return False
    return all(counts[m] >= c for m, c in caps.items())


def run_pull(args, config: str, split: str, subset: str) -> None:
    token = _token()
    root = (Path(args.out) if args.out else _out_root()) / subset
    stage = (Path(args.out) if args.out else _out_root()) / "_shards"
    meta_dir = root / "_meta"
    for d in (root, stage, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = _stride(shard_files(config, split, token), args.shards, args.stride, args.start)
    wanted = set(args.models) if args.models else None
    labels = set(args.labels)
    caps: dict[str, int] = {}
    if getattr(args, "from_rank", None):
        caps.update(caps_from_rank(Path(args.from_rank), parse_tiers(args.tier),
                                   set(args.exclude or DEFAULT_EXCLUDE)))
        wanted = set(caps)
    if args.cap:
        # floor: raise every selected model to at least --cap. Without a
        # named set it is a catch-all so the run cannot stop early.
        if wanted:
            for model in wanted:
                caps[model] = max(caps.get(model, 0), args.cap)
        else:
            caps["*"] = args.cap
    for entry in args.cap_model or []:
        model, _, value = entry.partition("=")
        caps[model] = int(value)
        if wanted is not None:
            wanted.add(model)

    json.dump({"config": config, "split": split, "labels": sorted(labels),
               "caps": caps, "max_side": args.max_side, "quality": args.quality},
              open(meta_dir / "plan.json", "w"), indent=1, sort_keys=True)

    state_path = meta_dir / "counts.json"
    counts = Counter(json.load(open(state_path)) if state_path.exists() else {})
    done = {line.strip() for line in (meta_dir / "done.txt").read_text().splitlines()} \
        if (meta_dir / "done.txt").exists() else set()

    print(f"[openfake] {config}/{split} -> {root}", flush=True)
    print(f"[openfake] {len(names)} shards (stride={args.stride} start={args.start}), "
          f"models={'all' if wanted is None else len(wanted)}, labels={sorted(labels)}, "
          f"cap={args.cap or 'none'}, max_side={args.max_side}", flush=True)
    if counts:
        print(f"[openfake] resume: {sum(counts.values())} images already indexed", flush=True)

    started = time.time()
    written = 0
    run_kept = 0
    index_of = {name: i for i, name in enumerate(names, 1)}
    stop = Event()
    counts_lock = Lock()
    io_lock = Lock()
    prefetch = getattr(args, "prefetch", 2)
    dl_workers = getattr(args, "download_workers", 2)
    extract_workers = max(1, getattr(args, "extract_workers", 1))
    print(f"[openfake] prefetch={prefetch} download_workers={dl_workers} "
          f"extract_workers={extract_workers} encode_workers={args.workers}",
          flush=True)

    def drop_local(local: Path, release) -> None:
        try:
            local.unlink()
        except OSError:
            pass
        release()

    def extract_one(name: str, local: Path, release):
        try:
            stats = extract_shard(
                local, name, root,
                wanted=wanted, labels=labels, caps=caps, counts=counts,
                max_side=args.max_side, quality=args.quality,
                min_side=args.min_side, workers=args.workers,
                counts_lock=counts_lock,
            )
            return name, stats
        finally:
            drop_local(local, release)

    def record(name: str, stats: dict) -> None:
        nonlocal written, run_kept
        with io_lock:
            done.add(name)
            written += stats["bytes"]
            run_kept += stats["kept"]
            (meta_dir / "done.txt").write_text("\n".join(sorted(done)) + "\n")
            with counts_lock:
                snapshot = dict(counts)
            json.dump(snapshot, open(state_path, "w"), indent=1, sort_keys=True)
            total = sum(snapshot.values())
            elapsed = max(1e-9, time.time() - started)
            print(f"[openfake] ({index_of[name]}/{len(names)}) {Path(name).name}: "
                  f"rows={stats['rows']} kept={stats['kept']} "
                  f"small={stats['skip_small']} bad={stats['skip_bad']} "
                  f"cap={stats['skip_cap']} | total={total} "
                  f"({run_kept / elapsed:.0f} img/s this run) "
                  f"{written / 1e9:.1f} GB", flush=True)

    inflight = {}
    shard_iter = _downloaded_shards(
        names, done, stage, token, prefetch, dl_workers, stop,
    )
    try:
        with ThreadPoolExecutor(max_workers=extract_workers) as pool:
            while True:
                while len(inflight) < extract_workers and not stop.is_set():
                    try:
                        name, local, release = next(shard_iter)
                    except StopIteration:
                        shard_iter = None
                        break
                    if caps_satisfied(caps, counts, wanted):
                        print("[openfake] every cap reached, stopping early",
                              flush=True)
                        drop_local(local, release)
                        stop.set()
                        break
                    fut = pool.submit(extract_one, name, local, release)
                    inflight[fut] = name
                if not inflight:
                    break
                finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in finished:
                    del inflight[fut]
                    name, stats = fut.result()
                    record(name, stats)
                    if caps_satisfied(caps, counts, wanted):
                        print("[openfake] every cap reached, stopping early",
                              flush=True)
                        stop.set()
    finally:
        stop.set()
        if shard_iter is not None:
            shard_iter.close()

    print(f"[openfake] done: {sum(counts.values())} images, "
          f"{written / 1e9:.1f} GB this run -> {root}", flush=True)
    _report(counts)


def _report(counts: Counter, limit: int = 200) -> None:
    for model, n in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]:
        print(f"  {n:7d}  {model}")


def cmd_index(args) -> None:
    token = _token()
    stage = (Path(args.out) if args.out else _out_root()) / "_shards"
    stage.mkdir(parents=True, exist_ok=True)
    names = _stride(shard_files(args.config, args.split, token), args.shards, args.stride, args.start)
    print(f"[openfake] indexing {len(names)} shards of {args.config}/{args.split}", flush=True)
    total = Counter()

    def one(name: str) -> Counter:
        local = Path(hf_hub_download(REPO, name, repo_type="dataset",
                                     local_dir=str(stage), token=token))
        try:
            return read_meta(local)
        finally:
            try:
                local.unlink()
            except OSError:
                pass

    for name, counter in zip(names, map(one, names)):
        total.update(counter)
        print(f"  {Path(name).name}: {sum(counter.values())} rows", flush=True)

    scale = len(shard_files(args.config, args.split, token)) / max(1, len(names))
    rows = [{"label": label, "model": model, "n_sampled": n,
             "n_estimated_total": int(round(n * scale))}
            for (label, model), n in sorted(total.items(), key=lambda kv: -kv[1])]
    out = Path(args.json) if args.json else _out_root() / f"index_{args.config}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": args.config, "split": args.split, "shards_sampled": len(names),
               "scale": scale, "rows": rows}, open(out, "w"), indent=1)
    print(f"[openfake] wrote {out}", flush=True)
    for r in rows:
        print(f"  {r['n_sampled']:6d} sampled  ~{r['n_estimated_total']:8d} total  "
              f"{r['label']:5s}  {r['model']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, *, shards, labels, cap):
        sp.add_argument("--out", default=None, help="root (default $SEER_DATA_ROOT/openfake)")
        sp.add_argument("--shards", type=int, default=shards, help="0 = every shard")
        sp.add_argument("--stride", type=int, default=1)
        sp.add_argument("--start", type=int, default=0)
        sp.add_argument("--models", nargs="*", default=None, help="empty = every generator")
        sp.add_argument("--labels", nargs="*", default=labels, choices=["real", "fake"])
        sp.add_argument("--cap", type=int, default=cap,
                        help="per-model cap, or a floor when used with --from-rank; 0 = uncapped")
        sp.add_argument("--cap-model", nargs="*", default=None, metavar="MODEL=N")
        sp.add_argument("--max-side", type=int, default=1536)
        sp.add_argument("--min-side", type=int, default=256)
        sp.add_argument("--quality", type=int, default=95)
        sp.add_argument("--workers", type=int, default=6,
                        help="JPEG encode threads; used only for PNG / oversized rows")
        sp.add_argument("--prefetch", type=int, default=2,
                        help="max parquet files on disk (download + extract)")
        sp.add_argument("--download-workers", type=int, default=2,
                        help="parallel shard downloads")
        sp.add_argument("--extract-workers", type=int, default=1,
                        help="shards extracted in parallel")

    sp = sub.add_parser("index", help="per-generator counts, no images kept")
    sp.add_argument("--out", default=None)
    sp.add_argument("--config", default="core")
    sp.add_argument("--split", default="validation")
    sp.add_argument("--shards", type=int, default=3)
    sp.add_argument("--stride", type=int, default=1)
    sp.add_argument("--start", type=int, default=0)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("probe", help="per-generator sample for difficulty ranking")
    common(sp, shards=3, labels=["real", "fake"], cap=0)
    sp.set_defaults(func=lambda a: run_pull(a, "core", "validation", "probe"))

    sp = sub.add_parser("fetch", help="training pull for chosen generators")
    common(sp, shards=0, labels=["fake"], cap=0)
    sp.add_argument("--from-rank", default=None,
                    help="rank.json from scripts/openfake_rank.py; caps are "
                         "derived from recall_mean via --tier")
    sp.add_argument("--tier", nargs="*", default=None, metavar="RECALL=CAP",
                    help=f"default {list(DEFAULT_TIERS)}")
    sp.add_argument("--exclude", nargs="*", default=None,
                    help=f"default {list(DEFAULT_EXCLUDE)}")
    sp.set_defaults(func=lambda a: run_pull(a, "core", "train", "train"))

    sp = sub.add_parser("holdout", help="core/test + reddit/test (never trained on)")
    common(sp, shards=2, labels=["real", "fake"], cap=0)
    sp.add_argument("--config", default="core", choices=["core", "reddit"])
    sp.set_defaults(func=lambda a: run_pull(a, a.config, "test", f"holdout_{a.config}"))

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.shards == 0:
        args.shards = None
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
