"""NTIRE labelled-split loader (no network, no real corpus)."""

from pathlib import Path

import numpy as np
from PIL import Image

from seer.datasets_registry import commfor_shard_selection, select
from seer.ntire import _parse_list, read_labelled_split, stratified_subset


def test_parse_list():
    assert _parse_list("") == ()
    assert _parse_list("[]") == ()
    assert _parse_list("['jpeg', 'downscale']") == ("jpeg", "downscale")
    assert _parse_list("[0.5, 1.0]") == (0.5, 1.0)


def test_read_labelled_split(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    for name, color in (("a.jpg", (10, 20, 30)), ("b.jpg", (200, 10, 10))):
        Image.new("RGB", (8, 8), color).save(images / name)

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "image_name,label,distortions,distortion_scales,is_distorted\n"
        "a.jpg,0,[],[],0\n"
        "b.jpg,1,['jpeg'],[0.4],1\n"
        "missing.jpg,1,[],[],0\n",
        encoding="utf-8",
    )

    samples = read_labelled_split(images, csv_path)
    assert [s.path.name for s in samples] == ["a.jpg", "b.jpg"]
    assert [s.label for s in samples] == [0, 1]
    assert samples[1].distortions == ("jpeg",)
    assert samples[1].is_distorted is True
    assert samples[0].load().size == (8, 8)


def test_stratified_subset(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    rows = ["image_name,label,distortions,distortion_scales,is_distorted"]
    for i, label in enumerate([0, 0, 0, 1, 1, 1]):
        name = f"{i}.jpg"
        Image.new("RGB", (4, 4)).save(images / name)
        rows.append(f"{name},{label},[],[],0")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    samples = read_labelled_split(images, csv_path)
    picked = stratified_subset(samples, n=4, rng=np.random.default_rng(0))
    assert len(picked) == 4
    assert sum(s.label == 0 for s in picked) == 2
    assert sum(s.label == 1 for s in picked) == 2


def test_registry_tiers():
    keys = {s.key for s in select(tiers=[1])}
    assert {"ntire-train", "ntire-val", "ntire-test", "coco-val2017"} <= keys
    shards = commfor_shard_selection()
    assert 70 in shards and 92 in shards
    assert shards == sorted(shards)
