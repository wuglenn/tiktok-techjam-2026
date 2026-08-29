"""Unpack GAS-Station v3/v4 tarballs into a folder listing Seer can train on.

The Hub dumps store pixels in weekly ``archives/**/*.tar.gz`` files and
metadata (``model_name``, ``file_path_in_archive``) in sidecar parquet.
There is no ``image`` column. ``scripts/wire_gasstation.py`` is the CLI.
"""

from __future__ import annotations

import os
import re
import tarfile
from pathlib import Path
from typing import Iterable

from .paths import gs_images_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VERSIONS = ("v3", "v4")


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-+]+", "_", (name or "unknown").strip()) or "unknown"
    return cleaned[:120]


def download_ready(root: Path) -> tuple[bool, str]:
    """True when archives exist and HuggingFace left no incomplete files."""
    incompletes = list(root.rglob("*.incomplete"))
    archives = list((root / "archives").rglob("*.tar.gz")) if (root / "archives").is_dir() else []
    if incompletes:
        return False, f"{len(incompletes)} incomplete download(s), {len(archives)} archives"
    if not archives:
        return False, "no archives yet"
    return True, f"{len(archives)} archives"


def _parquet_index(root: Path) -> dict[tuple[str, str], str]:
    """(archive basename, member path) -> model_name. Empty if pyarrow missing."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    index: dict[tuple[str, str], str] = {}
    for path in root.rglob("*.parquet"):
        if ".cache" in path.parts or "wired" in path.parts:
            continue
        try:
            table = pq.ParquetFile(path).read(
                columns=["archive_filename", "file_path_in_archive", "model_name"]
            )
        except Exception:
            continue
        archives = table.column("archive_filename").to_pylist()
        members = table.column("file_path_in_archive").to_pylist()
        models = table.column("model_name").to_pylist()
        for archive, member, model in zip(archives, members, models):
            if archive and member:
                index[(str(archive), str(member))] = str(model or "unknown")
    return index


def _iter_archives(root: Path) -> Iterable[Path]:
    archive_root = root / "archives"
    if not archive_root.is_dir():
        return
    for path in sorted(archive_root.rglob("*.tar.gz")):
        yield path


def _dest_for(member: str, archive_name: str, model: str, dest_root: Path) -> Path:
    basename = Path(member).name
    stem = Path(archive_name).name.replace(".tar.gz", "")
    return dest_root / _sanitize(model) / f"{stem}_{basename}"


def extract_version(
    root: Path,
    dest_root: Path | None = None,
    delete_archives: bool = False,
) -> dict:
    """Extract every image in ``root/archives`` into ``root/wired/images``.

    Returns counts and the listing path. Idempotent: existing dest files of
    matching size are left in place. When ``delete_archives`` is set, each
    tarball is removed after it unpacks successfully so both copies are
    never kept on disk.
    """
    dest_root = dest_root or (root / "wired" / "images")
    dest_root.mkdir(parents=True, exist_ok=True)
    index = _parquet_index(root)
    written: list[str] = []
    extracted = skipped = failed = unlabeled = removed = 0
    archives = list(_iter_archives(root))
    total = len(archives)

    for i, archive in enumerate(archives, 1):
        archive_ok = True
        try:
            tf = tarfile.open(archive, "r:gz")
        except (tarfile.TarError, OSError) as exc:
            print(f"[wire] skip unreadable {archive}: {exc}", flush=True)
            failed += 1
            continue
        with tf:
            for info in tf.getmembers():
                if not info.isfile():
                    continue
                ext = Path(info.name).suffix.lower()
                if ext not in IMAGE_EXTS:
                    continue
                model = index.get((archive.name, info.name)) or index.get(
                    (archive.name, Path(info.name).name)
                )
                if model is None:
                    unlabeled += 1
                    model = "unknown"
                dest = _dest_for(info.name, archive.name, model, dest_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and dest.stat().st_size == info.size:
                    skipped += 1
                    written.append(str(dest))
                    continue
                src = tf.extractfile(info)
                if src is None:
                    failed += 1
                    archive_ok = False
                    continue
                tmp = dest.with_suffix(dest.suffix + ".part")
                try:
                    with open(tmp, "wb") as handle:
                        handle.write(src.read())
                    os.replace(tmp, dest)
                    extracted += 1
                    written.append(str(dest))
                except Exception:
                    failed += 1
                    archive_ok = False
                    if tmp.exists():
                        tmp.unlink()
        if delete_archives and archive_ok:
            try:
                archive.unlink()
                removed += 1
            except OSError as exc:
                print(f"[wire] could not delete {archive}: {exc}", flush=True)
        if i == 1 or i % 25 == 0 or i == total:
            print(
                f"[wire] {i}/{total} archives  extracted={extracted} "
                f"skip={skipped} fail={failed} deleted={removed}",
                flush=True,
            )

    listing = root / "wired" / "images.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(written))
    listing.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    return {
        "root": str(root),
        "listing": str(listing),
        "images": len(unique),
        "extracted": extracted,
        "skipped": skipped,
        "unlabeled": unlabeled,
        "failed": failed,
        "removed": removed,
    }


def wire(version: str, force: bool = False, delete_archives: bool = False) -> dict:
    root = gs_images_dir(version)
    if not root.exists():
        return {"version": version, "status": "missing", "root": str(root)}
    ready, reason = download_ready(root)
    if not ready and not force:
        return {"version": version, "status": "not-ready", "reason": reason, "root": str(root)}
    stats = extract_version(root, delete_archives=delete_archives)
    stats.update(version=version, status="wired" if ready else "wired-partial", reason=reason)
    return stats
