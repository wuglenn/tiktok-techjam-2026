"""Held-out val coverage, train exclusion, and FP/FN dumps."""

import json
from pathlib import Path

from PIL import Image

from seer.config import DataConfig, SourceSpec, TrainConfig
from seer.data import (
    FolderPairStream,
    MixtureDataset,
    clear_holdout,
    collect_held_out_val,
    sample_id,
    set_holdout,
)
from seer.misclass import dump_misclassified, pick_errors, provenance


def _write_rgb(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _folder_pair(tmp: Path, n: int = 8):
    real, fake = tmp / "real", tmp / "fake"
    for i in range(n):
        _write_rgb(real / f"r{i}.jpg", (i + 1, 0, 0))
        _write_rgb(fake / f"f{i}.jpg", (0, i + 1, 0))
    return real, fake


def test_provenance_skips_pixels():
    rec = provenance({
        "source": "comfor",
        "dataset": "OwensLab/CommunityForensics-Small",
        "source_type": "comfor",
        "image_name": "a.png",
        "generator": "SDXL",
        "architecture": "LatDiff",
        "prompt": "a cat",
        "subset": "train",
        "image": object(),
        "image_bytes": b"not-in-json",
    })
    assert rec["source"] == "comfor"
    assert rec["generator"] == "SDXL"
    assert rec["architecture"] == "LatDiff"
    assert "image" not in rec and "image_bytes" not in rec
    assert rec["id"]


def test_pick_errors_worst_first():
    samples = [{"image_name": n} for n in "abcd"]
    fps, fns = pick_errors(samples, [0.9, 0.6, 0.4, 0.1], [0, 0, 1, 1])
    assert [s["image_name"] for s, _ in fps] == ["a", "b"]
    assert fps[0][1] == 0.9
    assert [s["image_name"] for s, _ in fns] == ["d", "c"]
    assert fns[0][1] == 0.1


def test_dump_writes_images_and_manifest(tmp_path: Path):
    real = tmp_path / "r.jpg"
    fake = tmp_path / "f.jpg"
    _write_rgb(real, (10, 20, 30))
    _write_rgb(fake, (200, 10, 10))
    samples = [
        {"image": None, "image_path": str(real), "label": 0, "source": "open-images-v7",
         "dataset": "open-images-v7", "source_type": "folders", "image_name": "r.jpg",
         "generator": "open-images-v7"},
        {"image": None, "image_path": str(fake), "label": 1, "source": "comfor",
         "dataset": "OwensLab/CommunityForensics-Small", "source_type": "comfor",
         "image_name": "f.jpg", "generator": "SD3", "architecture": "MMDiT"},
    ]
    out = tmp_path / "dump"
    stats = dump_misclassified(
        str(out), samples, [0.95, 0.05], [0, 1],
        step=2000, split="val", max_per_kind=8,
    )
    assert stats["n_fp"] == 1 and stats["n_fn"] == 1
    assert stats["saved_fp"] == 1 and stats["saved_fn"] == 1
    rows = [json.loads(ln) for ln in (out / "manifest.jsonl").read_text().splitlines()]
    assert {r["kind"] for r in rows} == {"fp", "fn"}
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["fp"]["source"] == "open-images-v7"
    assert by_kind["fn"]["generator"] == "SD3"
    assert by_kind["fn"]["architecture"] == "MMDiT"
    assert (out / by_kind["fp"]["file"]).exists()
    assert (out / by_kind["fn"]["file"]).exists()


def test_holdout_covers_every_source_and_skips_train(tmp_path: Path):
    clear_holdout()
    a_real, a_fake = _folder_pair(tmp_path / "a", n=8)
    b_fake = tmp_path / "b" / "fake"
    for i in range(8):
        _write_rgb(b_fake / f"g{i}.jpg", (0, 0, i + 1))

    cfg = TrainConfig(
        data=DataConfig(
            source="mixture",
            val_max_samples=4,
            val_seed=0,
            sources=[
                SourceSpec(name="pair", type="folders", weight=0.5,
                           real_dirs=[str(a_real)], fake_dirs=[str(a_fake)]),
                SourceSpec(name="gs", type="folders", weight=0.5,
                           fake_dirs=[str(b_fake)]),
            ],
        )
    )
    val = collect_held_out_val(cfg)
    assert {s["source"] for s in val} == {"pair", "gs"}
    assert all(s.get("source_type") == "folders" for s in val)
    ids = {sample_id(s) for s in val}
    assert len(ids) == len(val)
    set_holdout(val)

    mix = MixtureDataset(cfg.data.sources, seed=1)
    it = iter(mix)
    seen = [sample_id(next(it)) for _ in range(40)]
    assert ids.isdisjoint(seen)
    clear_holdout()


def test_mixture_tags_source(tmp_path: Path):
    clear_holdout()
    real, fake = _folder_pair(tmp_path, n=4)
    mix = MixtureDataset(
        [SourceSpec(name="pair", type="folders",
                    real_dirs=[str(real)], fake_dirs=[str(fake)])],
        seed=0,
    )
    sample = next(iter(mix))
    assert sample["source"] == "pair"
    assert sample["source_type"] == "folders"
    assert sample["dataset"] == "pair"
    assert sample_id(sample).startswith("pair|path|")


def test_folder_scan_drops_demo_val_images(tmp_path: Path):
    keep = tmp_path / "open-images-v7" / "ok.jpg"
    coco = tmp_path / "coco-val2017" / "000000000139.jpg"
    advanced = tmp_path / "wildfake-dalle" / "DALLE" / "Advanced" / "a.jpg"
    _write_rgb(keep, (1, 2, 3))
    _write_rgb(coco, (4, 5, 6))
    _write_rgb(advanced, (7, 8, 9))
    stream = FolderPairStream([str(tmp_path)], [], seed=0)
    assert stream.real_files == [str(keep)]
