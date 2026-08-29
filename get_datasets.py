"""Acquire every dataset in the plan.

Resumable and selective. Nothing is fetched unless asked for, and everything
already present is skipped, so this is safe to re-run.

    python get_datasets.py --list                 # show the plan, download nothing
    python get_datasets.py --tier 1               # core: train + both eval sets + COCO
    python get_datasets.py --only ntire-val mirage
    python get_datasets.py --tier 2 --dry-run     # show what tier 2 would cost

Two entries deliberately do not download:

* ``dda-train`` is an 11-part split ZIP that cannot be streamed or partially
  fetched and needs ~226 GB of peak disk. An equivalent subset is regenerated
  from COCO in about an hour instead.
* ``cifake`` and ``wildfake-dalle`` need Kaggle / ModelScope credentials, so
  the script prints the exact command rather than guessing at your auth setup.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, "src")

from aigcdet.datasets_registry import DatasetSpec, commfor_shard_selection, select  # noqa: E402
from aigcdet.paths import DATA_ROOT  # noqa: E402


def human(gb: float) -> str:
    return f"{gb:.1f} GB" if gb else "?"


def target_dir(spec: DatasetSpec) -> Path:
    return DATA_ROOT / ("ntire" if spec.key.startswith("ntire") else spec.key)


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------

def fetch_hf_files(spec: DatasetSpec, dry_run: bool) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    dest = target_dir(spec)
    if spec.key.startswith("ntire"):
        dest = dest / spec.repo_id.split("/")[-1]

    wanted = list(spec.files)
    if not wanted:
        wanted = _default_file_selection(spec, list_repo_files(spec.repo_id, repo_type="dataset"))

    print(f"  {len(wanted)} file(s) -> {dest}")
    if dry_run:
        for name in wanted[:6]:
            print(f"    would fetch {name}")
        if len(wanted) > 6:
            print(f"    ... and {len(wanted) - 6} more")
        return []

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
    """Pick a sensible subset when the spec does not name files explicitly."""
    if spec.key == "commfor-small":
        # Shards are sorted by label and subset; a naive prefix or uniform
        # stride drops every GAN and pixel-diffusion image. See registry.
        keep = set(commfor_shard_selection())
        return [f"data/HFCF_small_{i}.parquet" for i in sorted(keep)
                if f"data/HFCF_small_{i}.parquet" in set(available)]
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
    return []


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_archives(paths: list[Path]) -> None:
    for path in paths:
        if path.suffix != ".zip":
            continue
        out_dir = path.with_suffix("")
        marker = out_dir / ".extracted"
        if marker.exists():
            continue
        print(f"    unzip {path.name}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(out_dir)
        marker.touch()


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
