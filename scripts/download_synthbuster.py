"""Download + organize the Synthbuster dataset (Bammey 2023, CC-BY).

Synthbuster provides 9,000 images from modern text-to-image generators
(DALL-E 2/3, Adobe Firefly, Midjourney v5, SD 1.3/1.4/2.1, SDXL) plus the
prompt lists - a compact source of frontier-family coverage for the fake
side of our mixture. Real images are NOT bundled (Synthbuster pairs with
RAISE at eval time), so keep this source fake-only in the mixture and pair
it with a real folder of similar weight.

  uv run scripts/download_synthbuster.py            # -> F:/techjam/synthbuster
  uv run scripts/download_synthbuster.py --out data/synthbuster
"""

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from seer.paths import DATA_ROOT

ZENODO_URL = "https://zenodo.org/records/10066460/files/synthbuster.zip"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_HINTS = ("raise", "real", "flickr", "coco")


def _progress(n, total, width=30):
    if total <= 0:
        return
    k = int(width * n / total)
    sys.stdout.write("\r[" + "=" * k + " " * (width - k) + f"] {n / 1e6:.1f}/{total / 1e6:.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: str):
    print(f"[synthbuster] downloading {url}")

    def hook(blocks, bs, total):
        _progress(blocks * bs, total)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print()


def organize(src_dir: Path, out_dir: Path) -> dict:
    """Sort the extracted tree into fake/<generator>/ and real/."""
    fake_root = out_dir / "fake"
    real_root = out_dir / "real"
    fake_root.mkdir(parents=True, exist_ok=True)
    real_root.mkdir(parents=True, exist_ok=True)

    counts = {"fake": 0, "real": 0}
    moved_fake_gens = set()
    for dirpath, _, filenames in os.walk(src_dir):
        d = Path(dirpath)
        imgs = [f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue
        name = d.name.lower()
        if any(h in name for h in REAL_HINTS):
            target = real_root
            counts["real"] += len(imgs)
        else:
            gen = d.name if d.parent != src_dir else "unknown"
            target = fake_root / gen
            target.mkdir(parents=True, exist_ok=True)
            moved_fake_gens.add(gen)
            counts["fake"] += len(imgs)
        for f in imgs:
            shutil.move(str(d / f), str(target / f))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DATA_ROOT / "synthbuster"),
                   help="default: F:/techjam/synthbuster")
    p.add_argument("--url", default=ZENODO_URL)
    p.add_argument("--keep-zip", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "synthbuster.zip"

    if not zip_path.exists():
        download(args.url, str(zip_path))
    else:
        print(f"[synthbuster] using existing {zip_path}")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[synthbuster] extracting to {tmp}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        counts = organize(Path(tmp), out_dir)

    if not args.keep_zip and zip_path.exists():
        os.remove(zip_path)

    print(f"[synthbuster] organized: {counts}")
    print("\nAdd this to your training config's data.sources (fake-only source):")
    print("  - name: synthbuster")
    print("    type: folders")
    print(f"    fake_dirs: ['{(out_dir / 'fake').as_posix()}']")
    print(f"    real_dirs: ['{out_dir.as_posix()}/real']  # usually empty; pair with e.g. mirrors/real")
    print("    weight: 0.10")


if __name__ == "__main__":
    main()
