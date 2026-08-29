"""Acquire every dataset in the plan.

Resumable and selective. Nothing is fetched unless asked for, and everything
already present is skipped, so this is safe to re-run.

    python get_datasets.py --list                 # show the plan, download nothing
    python get_datasets.py --tier 1               # core: train + both eval sets + COCO
    python get_datasets.py --only ntire-val mirage
    python get_datasets.py --tier 2 --dry-run     # show what tier 2 would cost

Entries that do not download a full snapshot:

* ``flux-reason-6m`` (~882 GB) and ``sid-set`` (~140 GB) are streamed.
* ``dda-train`` is an 11-part split ZIP (~113 GB, ~226 GB peak to join+extract).
* ``cifake`` and ``wildfake-dalle`` need Kaggle / ModelScope credentials.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, "src")

from seer.datasets_registry import DatasetSpec, select  # noqa: E402
from seer.paths import DATA_ROOT  # noqa: E402


def human(gb: float) -> str:
    return f"{gb:.1f} GB" if gb else "?"


def target_dir(spec: DatasetSpec) -> Path:
    return DATA_ROOT / ("ntire" if spec.key.startswith("ntire") else spec.key)


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def fetch_hf_files(spec: DatasetSpec, dry_run: bool) -> list[Path]:
    dest = target_dir(spec)
    if spec.key.startswith("ntire"):
        dest = dest / spec.repo_id.split("/")[-1]

    wanted = list(spec.files)
    if not wanted:
        from huggingface_hub import list_repo_files

        wanted = _default_file_selection(spec, list_repo_files(spec.repo_id, repo_type="dataset"))

    print(f"  {len(wanted)} file(s) -> {dest}")
    if dry_run:
        for name in wanted[:6]:
            print(f"    would fetch {name}")
        if len(wanted) > 6:
            print(f"    ... and {len(wanted) - 6} more")
        return []

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, name in enumerate(wanted, 1):
        local = dest / name
        if local.exists():
            print(f"    [{index}/{len(wanted)}] skip {name} (present)")
            paths.append(local)
            continue
        print(f"    [{index}/{len(wanted)}] {name}", flush=True)
        started = time.time()
        path = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=name,
                repo_type="dataset",
                local_dir=str(dest),
            )
        )
        size_gb = path.stat().st_size / 1e9
        print(f"        {size_gb:.2f} GB in {time.time() - started:.0f}s")
        paths.append(path)
    return paths


def _default_file_selection(spec: DatasetSpec, available: list[str]) -> list[str]:
    """Pick files when the spec does not name them explicitly."""
    if spec.key == "commfor-small":
        return [f for f in available if f.endswith(".parquet")]
    return [f for f in available if not f.startswith(".") and f != "README.md"]


def fetch_url(spec: DatasetSpec, dry_run: bool) -> list[Path]:
    dest = target_dir(spec)
    if not spec.url.endswith(".zip"):
        print(f"  manual download page: {spec.url}")
        return []
    local = dest / Path(spec.url).name
    if local.exists():
        print(f"  skip {local.name} (present)")
        return [local]
    print(f"  {spec.url} -> {local}")
    if dry_run:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec.url, timeout=120) as response, open(local, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    print(f"  {local.stat().st_size / 1e9:.2f} GB")
    return [local]


def instruct_only(spec: DatasetSpec) -> list[Path]:
    dest = target_dir(spec)
    if spec.source == "kaggle":
        print(f"  needs Kaggle credentials (~/.kaggle/kaggle.json):")
        print(f"    kaggle datasets download -d {spec.repo_id} -p {dest} --unzip")
    elif spec.source == "modelscope":
        print("  needs the modelscope client, and is China-hosted (expect slow transfer):")
        include = f" --include '{spec.files[0]}'" if spec.files else ""
        print(f"    modelscope download --dataset {spec.repo_id}{include} --local_dir {dest}")
    elif spec.source == "generate":
        print("  intentionally not downloaded -- regenerate instead:")
        print("    python src/scripts/build_dda_pairs.py --n 25000")
    elif spec.source == "stream":
        print("  stream from the Hub; do not snapshot this repo.")
        print(f"    mixture type: hf   dataset: {spec.repo_id}")
        n = 16 if spec.key == "sid-set" else 8
        print(f"    optional slice: uv run scripts/fetch_data.py {spec.key} --max-shards {n}")
    elif spec.source == "manual":
        if spec.key == "open-images-v7":
            print("  python scripts/download_open_images.py --workers 32 --max-gb 70")
        elif spec.key == "laion400m-1":
            print("  python scripts/download_laion400m.py --max-shards 8 --max-images 80000 --min-side 512")
        elif spec.url:
            print(f"  see {spec.url}")
    return []


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def join_split_zip(zip_path: Path) -> Path:
    """Unsplit an Info-ZIP multi-disk archive (``.zip`` + ``.z01`` …)."""
    siblings = sorted(zip_path.parent.glob(zip_path.stem + ".z*"))
    if not siblings:
        return zip_path
    joined = zip_path.with_name(zip_path.stem + "-joined.zip")
    if joined.exists():
        print(f"    skip join {zip_path.name} (joined archive present)")
        return joined
    print(f"    join {1 + len(siblings)} volumes -> {joined.name}", flush=True)
    subprocess.run(
        ["zip", "-s", "0", str(zip_path), "--out", str(joined)],
        check=True,
    )
    return joined


_FAKE_DIR_HINTS = ("fake", "syn", "dda", "gen", "ai", "sd", "flux")
_REAL_DIR_HINTS = ("real", "coco", "nature", "auth", "photo", "human")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def organize_dda(extracted: Path, dest: Path) -> None:
    """Sort the DDA tree into dest/fake and dest/real from folder names."""
    fake_root = dest / "fake"
    real_root = dest / "real"
    marker = dest / ".organized"
    if marker.exists():
        return
    fake_root.mkdir(parents=True, exist_ok=True)
    real_root.mkdir(parents=True, exist_ok=True)
    counts = {"fake": 0, "real": 0, "skip": 0}
    for dirpath, _, filenames in os.walk(extracted):
        imgs = [f for f in filenames if Path(f).suffix.lower() in _IMAGE_EXTS]
        if not imgs:
            continue
        name = Path(dirpath).name.lower()
        if any(h in name for h in _REAL_DIR_HINTS) and not any(h in name for h in _FAKE_DIR_HINTS):
            target, key = real_root, "real"
        elif any(h in name for h in _FAKE_DIR_HINTS):
            target, key = fake_root, "fake"
        else:
            counts["skip"] += len(imgs)
            continue
        for filename in imgs:
            src = Path(dirpath) / filename
            dst = target / filename
            if not dst.exists():
                shutil.copy2(src, dst)
            counts[key] += 1
    print(f"    organized DDA: {counts['fake']} fake, {counts['real']} real, {counts['skip']} skipped")
    marker.touch()


def extract_archives(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix != ".zip":
            continue
        archive = join_split_zip(path) if list(path.parent.glob(path.stem + ".z*")) else path
        out_dir = path.with_suffix("")
        marker = out_dir / ".extracted"
        if not marker.exists():
            print(f"    unzip {archive.name}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(out_dir)
            marker.touch()
        if path.name.startswith("DDA-Training-Set"):
            organize_dda(out_dir, path.parent)


# --------------------------------------------------------------------------

FETCHERS = {
    "hf": fetch_hf_files,
    "hf_files": fetch_hf_files,
    "url": fetch_url,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="dataset keys")
    parser.add_argument("--tier", type=int, nargs="*", help="tiers to fetch (1=core, 2=breadth, 3=optional)")
    parser.add_argument("--list", action="store_true", help="show the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="resolve files and sizes without downloading")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    specs = select(args.only, args.tier)
    if not specs:
        print("no datasets matched; try --list")
        return

    if args.list:
        print(f"{'key':<16} {'tier':>4} {'GB':>7}  {'source':<11} role")
        print("-" * 110)
        for spec in specs:
            print(f"{spec.key:<16} {spec.tier:>4} {spec.approx_gb:>7.1f}  {spec.source:<11} {spec.role}")
        print("-" * 110)
        print(f"{'TOTAL':<16} {'':>4} {sum(s.approx_gb for s in specs):>7.1f}")
        return

    total = sum(spec.approx_gb for spec in specs)
    print(f"{len(specs)} dataset(s), approximately {human(total)}\n")

    for spec in specs:
        print(f"[{spec.key}] {spec.name}  (~{human(spec.approx_gb)}, {spec.licence})")
        for note in spec.notes:
            print(f"  ! {note}")
        fetcher = FETCHERS.get(spec.source)
        paths = fetcher(spec, args.dry_run) if fetcher else instruct_only(spec)
        if paths and not args.no_extract and not args.dry_run:
            extract_archives(paths)
        print()

    print("done")


if __name__ == "__main__":
    main()
