"""OpenFake: difficulty-driven selection, mixture wiring, held-out splits."""

import importlib.util
import json
from pathlib import Path

import pytest

from seer.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("configs/seer_vitl_512.yaml", "configs/seer_probe.yaml")


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank_json(tmp_path: Path, entries) -> Path:
    path = tmp_path / "rank.json"
    path.write_text(json.dumps({
        "checkpoint": "x.pt",
        "generators": [{"model": m, "recall_mean": r} for m, r in entries],
    }))
    return path


# --------------------------------------------------------------------------
# selection: caps follow measured recall, and the noise stub is dropped
# --------------------------------------------------------------------------

def test_caps_scale_inversely_with_recall(tmp_path):
    openfake = _script("openfake")
    path = _rank_json(tmp_path, [
        ("nano-banana", 0.198),      # < 0.70  -> biggest cap
        ("ideogram-3.0", 0.849),     # 0.70-0.95
        ("flux.2-klein-4b", 0.979),  # 0.95-0.98
        ("imagen-4.0", 1.0),         # >= 0.98 -> not fetched
    ])
    caps = openfake.caps_from_rank(path, openfake.DEFAULT_TIERS, set())

    assert caps["nano-banana"] == 8000
    assert caps["ideogram-3.0"] == 5000
    assert caps["flux.2-klein-4b"] == 2500
    assert "imagen-4.0" not in caps
    assert caps["nano-banana"] > caps["ideogram-3.0"] > caps["flux.2-klein-4b"]


def test_tiny_random_sana_is_excluded_by_default(tmp_path):
    """It emits uniform RGB noise, and our augmentation puts noise on reals."""
    openfake = _script("openfake")
    assert "tiny-random-sana" in openfake.DEFAULT_EXCLUDE
    path = _rank_json(tmp_path, [("tiny-random-sana", 0.0), ("nano-banana", 0.2)])

    caps = openfake.caps_from_rank(path, openfake.DEFAULT_TIERS,
                                   set(openfake.DEFAULT_EXCLUDE))
    assert "tiny-random-sana" not in caps
    assert "nano-banana" in caps

    # ...but it is a default, not a hard rule
    assert "tiny-random-sana" in openfake.caps_from_rank(
        path, openfake.DEFAULT_TIERS, set())


def test_custom_tiers_override_defaults(tmp_path):
    openfake = _script("openfake")
    path = _rank_json(tmp_path, [("a", 0.5), ("b", 0.9)])
    caps = openfake.caps_from_rank(path, openfake.parse_tiers(["0.6=100", "0.99=7"]),
                                   set())
    assert caps == {"a": 100, "b": 7}


def test_rank_without_any_hole_is_an_error(tmp_path):
    openfake = _script("openfake")
    path = _rank_json(tmp_path, [("solved", 1.0)])
    with pytest.raises(SystemExit):
        openfake.caps_from_rank(path, openfake.DEFAULT_TIERS, set())


def test_native_jpeg_is_written_through(tmp_path):
    """Header-only path: bytes on disk equal the source JPEG."""
    from PIL import Image

    openfake = _script("openfake")
    src = tmp_path / "src.jpg"
    Image.new("RGB", (64, 48), (12, 34, 56)).save(src, format="JPEG", quality=90)
    raw = src.read_bytes()
    dest = tmp_path / "out.jpg"

    written = openfake._save_jpeg(raw, dest, max_side=1536, quality=95, min_side=32)
    assert written == dest.stat().st_size
    assert dest.read_bytes() == raw


def test_png_and_oversized_jpeg_are_reencoded(tmp_path):
    from PIL import Image

    openfake = _script("openfake")

    png = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(png, format="PNG")
    dest_png = tmp_path / "from_png.jpg"
    assert openfake._save_jpeg(png.read_bytes(), dest_png, 1536, 95, 32) > 0
    assert dest_png.read_bytes()[:2] == b"\xff\xd8"
    assert dest_png.read_bytes() != png.read_bytes()

    big = tmp_path / "big.jpg"
    Image.new("RGB", (2000, 1000), (9, 9, 9)).save(big, format="JPEG", quality=80)
    dest_big = tmp_path / "resized.jpg"
    assert openfake._save_jpeg(big.read_bytes(), dest_big, 1536, 95, 32) > 0
    out = Image.open(dest_big)
    assert max(out.size) == 1536
    assert dest_big.read_bytes() != big.read_bytes()


def test_tiny_image_is_skipped(tmp_path):
    from PIL import Image

    openfake = _script("openfake")
    src = tmp_path / "tiny.jpg"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(src, format="JPEG")
    dest = tmp_path / "nope.jpg"
    assert openfake._save_jpeg(src.read_bytes(), dest, 1536, 95, min_side=256) == -1
    assert not dest.exists()


def test_caps_satisfied_only_when_every_named_model_is_full():
    openfake = _script("openfake")
    from collections import Counter

    caps = {"a": 10, "b": 10}
    wanted = {"a", "b"}
    assert not openfake.caps_satisfied(caps, Counter({"a": 10}), wanted)
    assert openfake.caps_satisfied(caps, Counter({"a": 10, "b": 12}), wanted)
    # a catch-all cap can never be "done": more shards may still hold models
    assert not openfake.caps_satisfied({"*": 10}, Counter({"a": 99}), None)


# --------------------------------------------------------------------------
# mixture wiring
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", CONFIGS)
def test_openfake_is_a_mixed_source_pointing_at_train_only(path):
    cfg = load_config(path)
    src = next(s for s in cfg.data.sources if s.name == "openfake")
    assert src.type == "folders"
    assert src.fake_dirs and src.real_dirs
    for d in src.fake_dirs + src.real_dirs:
        assert "openfake/train/" in d.replace("\\", "/")
        assert "holdout" not in d


@pytest.mark.parametrize("path", CONFIGS)
def test_no_source_dominates_after_rebalancing(path):
    cfg = load_config(path)
    weights = {s.name: s.weight for s in cfg.data.sources}
    total = sum(weights.values())
    assert total == pytest.approx(1.0)
    assert weights["openfake"] == pytest.approx(0.128)
    assert weights["laion400m-1"] == pytest.approx(0.128)
    # the documented ceiling in docs/DATA_MIXTURE.md
    assert max(weights.values()) / total <= 0.25
    # openfake mass came out of the tiny repeated frontier pool
    assert weights["frontier-fakes"] < weights["openfake"]


def test_both_configs_share_one_mixture():
    """docs/DATA_MIXTURE.md promises the probe/continuation comparison is
    honest, which requires identical sources and weights."""
    hero, probe = (load_config(p) for p in CONFIGS)
    assert {s.name: s.weight for s in hero.data.sources} == \
           {s.name: s.weight for s in probe.data.sources}


# --------------------------------------------------------------------------
# held-out splits
# --------------------------------------------------------------------------

def test_openfake_test_splits_are_known_eval_datasets():
    from seer.eval import OPENFAKE_EVAL, known_eval_datasets

    names = known_eval_datasets()
    assert "openfake_test" in names
    assert "openfake_reddit" in names
    assert OPENFAKE_EVAL["openfake_test"] == "holdout_core"
    assert OPENFAKE_EVAL["openfake_reddit"] == "holdout_reddit"


def test_missing_holdout_is_a_filenotfound_not_a_crash(tmp_path, monkeypatch):
    """The training loop skips FileNotFoundError eval sets with a log line, so
    a config may list openfake_test before the data is fetched."""
    from seer import eval as E
    from seer import paths

    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        E._build_eval_dataset("openfake_test")
    assert "openfake.py holdout" in str(exc.value)


def test_holdout_eval_uses_a_stratified_subset(tmp_path, monkeypatch):
    """Periodic evals must not walk the full ~90k holdout."""
    from PIL import Image

    from seer import eval as E
    from seer import paths

    root = tmp_path / "openfake" / "holdout_core"
    for cls, gens, n in (
        ("fake", ("nano-banana-pro", "flux.2-klein-9b", "sora-2"), 12),
        ("real", ("docci", "imagenet"), 12),
    ):
        for gen in gens:
            d = root / cls / gen
            d.mkdir(parents=True)
            for i in range(n):
                Image.new("RGB", (32, 32), (i, 7, 7)).save(d / f"{i}.jpg")

    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    ds = E._build_eval_dataset("openfake_test", max_samples=10)
    samples = [ds[i] for i in range(len(ds))]
    assert len(samples) == 10
    assert sum(s["label"] == 0 for s in samples) == 5
    assert sum(s["label"] == 1 for s in samples) == 5
    # every generator still appears (round-robin)
    assert {"nano-banana-pro", "flux.2-klein-9b", "sora-2"} <= {
        s["generator"] for s in samples if s["label"] == 1
    }
    assert {"docci", "imagenet"} <= {s["generator"] for s in samples if s["label"] == 0}

    full = E._build_eval_dataset("openfake_test", max_samples=0)
    assert len(full) == 12 * 5


def test_holdout_eval_reports_per_generator(tmp_path, monkeypatch):
    """FolderDataset carries the generator in the parent directory; the eval
    pass must group on it, otherwise the holdout collapses to one bucket."""
    from PIL import Image

    from seer import eval as E
    from seer import paths

    root = tmp_path / "openfake" / "holdout_core"
    for cls, gens in (("fake", ("nano-banana-pro", "flux.2-klein-9b")),
                      ("real", ("docci",))):
        for gen in gens:
            d = root / cls / gen
            d.mkdir(parents=True)
            Image.new("RGB", (32, 32), (7, 7, 7)).save(d / "a.jpg")

    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    ds = E._build_eval_dataset("openfake_test")
    samples = [ds[i] for i in range(len(ds))]

    assert {s["generator"] for s in samples} == {
        "nano-banana-pro", "flux.2-klein-9b", "docci"}
    assert sorted(s["label"] for s in samples) == [0, 1, 1]

    keys = [s.get("architecture") or s.get("generator") or "" for s in samples]
    grouped = E._group_metrics([0.9, 0.1, 0.2], [1, 1, 0], keys)
    assert set(grouped) == {"nano-banana-pro", "flux.2-klein-9b", "docci"}

    # the same directory is invisible without the eval-only opt-in
    from seer.data import FolderDataset

    with pytest.raises(FileNotFoundError):
        FolderDataset([str(root / "fake")], 1)
    assert len(FolderDataset([str(root / "fake")], 1, allow_held_out=True)) == 2


def test_training_refuses_openfake_holdout_dirs():
    from seer.config import DataConfig, SourceSpec, TrainConfig
    from seer.data import assert_not_held_out_train, build_train_dataset

    spec = SourceSpec(name="openfake", type="folders",
                      fake_dirs=["/workspace/data/openfake/holdout_core/fake"])
    with pytest.raises(ValueError) as exc:
        assert_not_held_out_train(spec=spec)
    assert "held-out" in str(exc.value).lower()

    cfg = TrainConfig(data=DataConfig(source="mixture", sources=[spec]))
    with pytest.raises(ValueError):
        build_train_dataset(cfg)


def test_registry_documents_the_streaming_constraint():
    from seer.datasets_registry import BY_KEY

    spec = BY_KEY["openfake"]
    assert spec.repo_id == "ComplexDataLab/OpenFake"
    assert spec.config == "core"
    notes = " ".join(spec.notes).lower()
    assert "3.44 tb" in notes and "never snapshot" in notes
    assert "held out" in notes
    assert "tiny-random-sana" in notes
