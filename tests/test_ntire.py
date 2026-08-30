"""NTIRE labelled-split loader (no network, no real corpus)."""

from pathlib import Path

import numpy as np
from PIL import Image

from seer.datasets_registry import commfor_shard_selection, select
from seer.ntire import (
    _parse_list,
    list_train_shards,
    load_split,
    read_labelled_split,
    stratified_subset,
)


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


def test_load_train_all_shards(tmp_path: Path, monkeypatch):
    import seer.ntire as ntire

    for shard, names in ((0, ("a.jpg", "b.jpg")), (1, ("c.jpg",))):
        root = tmp_path / f"shard_{shard}" / f"shard_{shard}"
        images = root / "images"
        images.mkdir(parents=True)
        rows = ["image_name,label,distortions,distortion_scales,is_distorted"]
        for i, name in enumerate(names):
            Image.new("RGB", (4, 4)).save(images / name)
            rows.append(f"{name},{i % 2},[],[],0")
        (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(ntire, "_train_dir", lambda: tmp_path)
    assert list_train_shards() == [0, 1]
    assert len(load_split("train", shard=0)) == 2
    all_train = load_split("train", shard=-1)
    assert len(all_train) == 3
    assert {s.path.name for s in all_train} == {"a.jpg", "b.jpg", "c.jpg"}


def test_registry_tiers():
    keys = {s.key for s in select(tiers=[1])}
    assert {"ntire-train", "ntire-val", "ntire-test", "coco-val2017"} <= keys
    shards = commfor_shard_selection()
    assert 70 in shards and 92 in shards
    assert shards == sorted(shards)


def _write_train_shard(root: Path, shard: int, names: tuple[str, ...]) -> None:
    inner = root / f"shard_{shard}" / f"shard_{shard}"
    images = inner / "images"
    images.mkdir(parents=True)
    rows = ["image_name,label,distortions,distortion_scales,is_distorted"]
    for i, name in enumerate(names):
        Image.new("RGB", (4, 4)).save(images / name)
        rows.append(f"{name},{i % 2},[],[],0")
    (inner / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_split_caches(tmp_path: Path, monkeypatch):
    import seer.ntire as ntire

    _write_train_shard(tmp_path, 0, ("a.jpg", "b.jpg"))
    monkeypatch.setattr(ntire, "_train_dir", lambda: tmp_path)
    first = load_split("train", shard=0)
    second = load_split("train", shard=0)
    assert first is second
    assert ntire.split_is_cached("train", shard=0)


def test_ntire_stream_cycle_and_one_pass(tmp_path: Path, monkeypatch):
    import seer.ntire as ntire
    from seer.data import NtireStream

    _write_train_shard(tmp_path, 0, ("a.jpg", "b.jpg"))
    monkeypatch.setattr(ntire, "_train_dir", lambda: tmp_path)

    once = list(NtireStream(split="train", shard=0, seed=0, cycle=False))
    assert len(once) == 2
    assert {s["image_name"] for s in once} == {"a.jpg", "b.jpg"}
    assert all("is_distorted" in s and "distortions" in s for s in once)

    cycling = NtireStream(split="train", shard=0, seed=0, cycle=True)
    it = iter(cycling)
    got = [next(it) for _ in range(4)]
    assert len(got) == 4
    assert {s["image_name"] for s in got} == {"a.jpg", "b.jpg"}


def test_build_val_covers_all_sources(tmp_path: Path):
    from seer.config import DataConfig, SourceSpec, TrainConfig
    from seer.data import collect_held_out_val, clear_holdout

    clear_holdout()
    real = tmp_path / "real"
    fake = tmp_path / "fake"
    real.mkdir()
    fake.mkdir()
    for i in range(6):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(real / f"r{i}.jpg")
        Image.new("RGB", (8, 8), (0, i, 0)).save(fake / f"f{i}.jpg")

    cfg = TrainConfig(
        data=DataConfig(
            source="mixture",
            sources=[
                SourceSpec(name="reals", type="folders", weight=0.5,
                           real_dirs=[str(real)]),
                SourceSpec(name="fakes", type="folders", weight=0.5,
                           fake_dirs=[str(fake)]),
            ],
            val_max_samples=4,
            val_seed=0,
        )
    )
    samples = collect_held_out_val(cfg)
    assert {s["source"] for s in samples} == {"reals", "fakes"}
    assert {s["dataset"] for s in samples} == {"reals", "fakes"}
    assert all(s["source_type"] == "folders" for s in samples)
    by = {}
    for s in samples:
        by[s["source"]] = by.get(s["source"], 0) + 1
    assert by["reals"] >= 1 and by["fakes"] >= 1
    clear_holdout()


def test_eval_dataset_names_and_groups():
    from seer.eval import _group_metrics, compute_metrics, known_eval_datasets

    names = known_eval_datasets()
    assert "ntire_test" in names
    assert "ntire_test_public" in names
    m = _group_metrics([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0], ["clean", "distorted", "clean", "distorted"])
    assert set(m) == {"clean", "distorted"}
    assert m["clean"]["n"] == 2
    # robust AUROC is the distorted-only ROC, not the pooled one
    probs = np.array([0.01, 0.99, 0.55, 0.45])
    labels = np.array([0, 1, 0, 1])
    flags = ["clean", "clean", "distorted", "distorted"]
    grouped = _group_metrics(probs, labels, flags)
    pooled = compute_metrics(probs, labels)
    assert grouped["distorted"]["auroc"] != pooled["auroc"]
    assert grouped["distorted"]["n"] == 2


def test_cached_val_samples_reused(monkeypatch):
    from seer import train as T
    from seer.config import DataConfig, TrainConfig
    from seer.data import clear_holdout

    T._VAL_SAMPLE_CACHE.clear()
    clear_holdout()
    calls = {"n": 0}

    def fake_build(cfg):
        calls["n"] += 1
        return [{"label": 0, "image_name": "a", "source": "x"},
                {"label": 1, "image_name": "b", "source": "x"}]

    monkeypatch.setattr(T, "collect_held_out_val", fake_build)
    cfg = TrainConfig(data=DataConfig(val_max_samples=2, source="mixture"))
    first = T._cached_val_samples(cfg)
    second = T._cached_val_samples(cfg)
    assert first is second
    assert calls["n"] == 1
    assert [s["image_name"] for s in first] == ["a", "b"]
    clear_holdout()
    T._VAL_SAMPLE_CACHE.clear()
