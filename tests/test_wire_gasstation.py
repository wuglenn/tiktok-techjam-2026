"""GAS-Station archive unpacker (no network, tiny tarball)."""

import tarfile
from pathlib import Path

from PIL import Image

from seer.config import SourceSpec
from seer.data import MixtureDataset
from seer.gasstation import download_ready, extract_version


def _write_png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)


def test_extract_version_writes_listing(tmp_path: Path):
    archive_dir = tmp_path / "archives" / "2026W19"
    archive_dir.mkdir(parents=True)
    member = tmp_path / "_pack" / "images" / "foo.png"
    _write_png(member)
    tar_path = archive_dir / "images_1.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(member, arcname="images/foo.png")

    parquet = tmp_path / "data_2026W19"
    parquet.mkdir()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "archive_filename": ["images_1.tar.gz"],
            "file_path_in_archive": ["images/foo.png"],
            "model_name": ["acme/flux"],
        })
        pq.write_table(table, parquet / "shard.parquet")
    except ImportError:
        pass

    stats = extract_version(tmp_path)
    listing = Path(stats["listing"])
    assert listing.exists()
    lines = [ln for ln in listing.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1
    dest = Path(lines[0])
    assert dest.exists()
    assert dest.suffix == ".png"
    # second run is a no-op extract
    again = extract_version(tmp_path, delete_archives=True)
    assert again["extracted"] == 0
    assert again["skipped"] == 1
    assert again["images"] == 1
    assert again["removed"] == 1
    assert not tar_path.exists()


def test_download_ready_detects_incomplete(tmp_path: Path):
    (tmp_path / "archives").mkdir()
    ready, reason = download_ready(tmp_path)
    assert ready is False
    assert "no archives" in reason
    (tmp_path / "archives" / "images_1.tar.gz").write_bytes(b"x")
    ready, _ = download_ready(tmp_path)
    assert ready is True
    (tmp_path / "foo.incomplete").write_text("partial", encoding="utf-8")
    ready, reason = download_ready(tmp_path)
    assert ready is False
    assert "incomplete" in reason


def test_mixture_skips_unwired_gasstation(tmp_path: Path):
    fake = tmp_path / "fake"
    fake.mkdir()
    _write_png(fake / "a.jpg")
    sources = [
        SourceSpec(name="ok", type="folders", fake_dirs=[str(fake)], weight=1.0),
        SourceSpec(name="gs", type="folders", fake_dirs=[str(tmp_path / "missing.txt")], weight=1.0),
    ]
    sample = next(iter(MixtureDataset(sources, seed=0)))
    assert sample["label"] == 1
    assert sample["image_path"].endswith("a.jpg")
