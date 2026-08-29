"""Unpack GAS-Station v3/v4 tarballs so the hero mixture can read them.

The Hub dumps store pixels in weekly ``archives/**/*.tar.gz`` files and
metadata (``model_name``, ``file_path_in_archive``) in sidecar parquet.
There is no ``image`` column, so ``type: hf`` cannot train on them.

This script extracts every image member, names it under the generator when
the parquet join hits, and writes ``wired/images.txt`` — a path list the
hero ``type: folders`` sources already point at.

v3 is skipped while a download is still in flight (``.incomplete`` markers)
unless you pass ``--force``. v4 on this volume is complete and can be wired
now.

  uv run scripts/wire_gasstation.py                  # v4 now; v3 when ready
  uv run scripts/wire_gasstation.py --versions v4
  uv run scripts/wire_gasstation.py --versions v3 --force   # partial v3
  uv run scripts/wire_gasstation.py --versions v4 --delete-archives
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from seer.gasstation import VERSIONS, wire


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--versions",
        nargs="+",
        default=list(VERSIONS),
        choices=list(VERSIONS),
        help="which dump(s) to wire (default: v3 v4)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="extract even if HuggingFace still has .incomplete files",
    )
    parser.add_argument("--data-root", default=None, help="override SEER_DATA_ROOT")
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="remove each tar.gz after it unpacks successfully",
    )
    args = parser.parse_args()

    if args.data_root:
        os.environ["SEER_DATA_ROOT"] = args.data_root
        import seer.paths as paths

        paths.DATA_ROOT = Path(args.data_root)

    import seer.paths as paths

    print(f"[wire] data root {paths.DATA_ROOT}", flush=True)
    for version in args.versions:
        result = wire(version, force=args.force, delete_archives=args.delete_archives)
        status = result.get("status")
        if status == "missing":
            print(f"[wire] {version}: not downloaded ({result['root']})", flush=True)
        elif status == "not-ready":
            print(
                f"[wire] {version}: still downloading ({result['reason']}) — "
                "rerun when fetch finishes, or pass --force",
                flush=True,
            )
        else:
            print(
                f"[wire] {version}: {status}  {result['images']} images  "
                f"(new={result['extracted']} skip={result['skipped']} "
                f"unlabeled={result['unlabeled']} fail={result['failed']} "
                f"deleted={result.get('removed', 0)})  "
                f"-> {result['listing']}",
                flush=True,
            )
            print(
                f"  hero yaml already points at this listing:\n"
                f"    fake_dirs: [{result['listing']}]",
                flush=True,
            )


if __name__ == "__main__":
    main()
