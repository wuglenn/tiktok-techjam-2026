"""Fetch datasets to the data root (F:/techjam).

Downloads the raw parquet shards once via huggingface_hub - after this,
training reads local disk (fast, no re-downloads across epochs, no arrow
cache duplication). Supports partial fetches with --max-shards.

  # everything (~260GB for comfor-small)
  uv run scripts/fetch_data.py comfor-small

  # a 40GB slice (first N shards, naturally ordered) - enough for ~55k images
  uv run scripts/fetch_data.py comfor-small --max-shards 30

  # the eval set
  uv run scripts/fetch_data.py comfor-eval
"""

import argparse
import fnmatch
import re
from pathlib import Path

from seer.paths import DATA_ROOT

FETCHABLES = {
    "comfor-small": dict(repo="OwensLab/CommunityForensics-Small", patterns=["data/*.parquet"]),
    "comfor-eval": dict(repo="OwensLab/CommunityForensics-Eval", patterns=["data/*.parquet"]),
}


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", choices=list(FETCHABLES))
    p.add_argument("--out", default=None, help="default: F:/techjam/<name>")
    p.add_argument("--max-shards", type=int, default=None,
                   help="download only the first N parquet shards")
    p.add_argument("--workers", type=int, default=8, help="parallel download connections")
    args = p.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    spec = FETCHABLES[args.name]
    out = Path(args.out) if args.out else DATA_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = [
        f for f in api.list_repo_files(spec["repo"], repo_type="dataset")
        if any(fnmatch.fnmatch(f, pat) for pat in spec["patterns"])
    ]
    files.sort(key=_natural_key)
    total = len(files)
    if args.max_shards:
        files = files[: args.max_shards]
    print(f"[fetch] {args.name}: {len(files)}/{total} shards -> {out}")

    snapshot_download(
        repo_id=spec["repo"],
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=files,
        max_workers=args.workers,
    )

    print(f"[fetch] done. shards in {out}")
    print("\nPoint your training config at it:")
    print("  data:")
    print(f"    local_dirs: ['{(out / 'data').as_posix()}']")


if __name__ == "__main__":
    main()
