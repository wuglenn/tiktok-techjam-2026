"""Label remapping for mixed / inverted public datasets."""

from seer.config import load_config
from seer.datasets_registry import BY_KEY, select
from seer.labels import normalize_label

# julienlucas/midjourney-dalle-sd-nanobananapro-dataset ClassLabel
FRONTIER_MAP = {0: 1, 1: 0, "fake": 1, "real": 0}


def test_normalize_strings_and_ints():
    assert normalize_label("fake") == 1
    assert normalize_label("real") == 0
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0
    assert normalize_label(None, default=1) == 1


def test_frontier_classlabel_is_inverted():
    assert normalize_label(0, FRONTIER_MAP) == 1
    assert normalize_label(1, FRONTIER_MAP) == 0
    assert normalize_label("0", FRONTIER_MAP) == 1
    assert normalize_label("fake", FRONTIER_MAP) == 1
    assert normalize_label("real", FRONTIER_MAP) == 0


SID_MAP = {0: 0, 1: 1}


def test_registry_lists_new_fake_sources():
    assert "flux-reason-6m" in BY_KEY
    assert "frontier-fakes" in BY_KEY
    flux = BY_KEY["flux-reason-6m"]
    assert flux.source == "stream"
    assert flux.repo_id == "LucasFang/FLUX-Reason-6M"
    frontier = BY_KEY["frontier-fakes"]
    assert frontier.repo_id.endswith("nanobananapro-dataset")
    assert BY_KEY["sid-set"].repo_id == "saberzl/SID_Set"
    assert BY_KEY["sid-set"].source == "stream"
    assert BY_KEY["gs-images-v3"].repo_id == "gasstation/gs-images-v3"
    assert BY_KEY["gs-images-v4"].repo_id == "gasstation/gs-images-v4"
    keys = {s.key for s in select(tiers=[2])}
    assert {"flux-reason-6m", "frontier-fakes", "sid-set", "gs-images-v3", "gs-images-v4"} <= keys


def test_sid_keeps_only_synthetic():
    assert normalize_label(0, SID_MAP) == 0
    assert normalize_label(1, SID_MAP) == 1
    # tampered is not remapped to fake; keep_label=1 drops it
    assert normalize_label(2, SID_MAP) != 1
    assert normalize_label("2", SID_MAP) != 1


def test_probe_config_keeps_sid_synthetic_only():
    cfg = load_config("configs/seer_probe.yaml")
    sid = next(s for s in cfg.data.sources if s.name == "sid-set")
    assert sid.keep_label == 1
    assert normalize_label(1, sid.label_map) == 1
    assert normalize_label(2, sid.label_map) != 1


def test_train_configs_exclude_held_out_sets():
    from seer.data import (
        assert_not_comfor_eval_train,
        assert_not_held_out_train,
        build_train_dataset,
        is_held_out_train_ref,
    )
    from seer.config import SourceSpec, TrainConfig, DataConfig

    for path in ("configs/seer_vitl_512.yaml", "configs/seer_probe.yaml"):
        cfg = load_config(path)
        for s in cfg.data.sources:
            assert "eval" not in (s.dataset or "").lower()
            assert not any("comfor-eval" in d.lower() for d in s.local_dirs)
            assert not any("coco-val2017" in d.lower() for d in (s.real_dirs + s.fake_dirs + s.local_dirs))
            assert "wildfake" not in (s.dataset or "").lower()
            assert "coco-val2017" not in (s.name or "")

    assert_not_comfor_eval_train("OwensLab/CommunityForensics-Small", ["/workspace/data/commfor-small"])
    try:
        assert_not_comfor_eval_train("OwensLab/CommunityForensics-Eval")
        raise AssertionError("eval dataset should be rejected")
    except ValueError:
        pass
    try:
        assert_not_comfor_eval_train("", ["/workspace/data/comfor-eval"])
        raise AssertionError("eval local_dirs should be rejected")
    except ValueError:
        pass

    assert is_held_out_train_ref("/workspace/data/coco-val2017/000000000139.jpg")
    assert is_held_out_train_ref("/data/val2017/000000000139.jpg")
    assert is_held_out_train_ref("/workspace/data/wildfake-dalle/Images/Diffusion_based/DALLE/Advanced/a.png")
    assert not is_held_out_train_ref("/workspace/data/open-images-v7/validation/abc.jpg")
    assert not is_held_out_train_ref("/workspace/data/frontier-fakes")

    # OpenFake: only core/train may be trained on
    assert is_held_out_train_ref("/workspace/data/openfake/holdout_core/fake/nano-banana-pro/a.jpg")
    assert is_held_out_train_ref("/workspace/data/openfake/holdout_reddit/real/reddit/a.jpg")
    assert is_held_out_train_ref(r"F:\techjam\openfake\holdout_core\fake\x.jpg")
    assert not is_held_out_train_ref("/workspace/data/openfake/train/fake/nano-banana/a.jpg")
    assert not is_held_out_train_ref("/workspace/data/openfake/train/real/pexels/a.jpg")

    try:
        assert_not_held_out_train(spec=SourceSpec(
            name="coco", type="folders", real_dirs=["/workspace/data/coco-val2017"],
        ))
        raise AssertionError("coco-val2017 should be rejected")
    except ValueError as exc:
        assert "demonstration" in str(exc) or "held-out" in str(exc)

    try:
        assert_not_held_out_train(spec=SourceSpec(
            name="dalle", type="folders",
            fake_dirs=["/workspace/data/wildfake-dalle"],
        ))
        raise AssertionError("wildfake-dalle should be rejected")
    except ValueError:
        pass

    cfg = TrainConfig(data=DataConfig(
        source="mixture",
        sources=[SourceSpec(name="bad", type="comfor", dataset="OwensLab/CommunityForensics-Eval")],
    ))
    try:
        build_train_dataset(cfg)
        raise AssertionError("build_train_dataset should reject Eval")
    except ValueError as exc:
        assert "held-out" in str(exc)


def test_hero_config_keeps_only_frontier_fakes():
    cfg = load_config("configs/seer_vitl_512.yaml")
    by_name = {s.name: s for s in cfg.data.sources}
    assert set(by_name) >= {"flux-reason", "frontier-fakes", "sid-set", "ntire", "gs-images-v3", "gs-images-v4"}
    flux = by_name["flux-reason"]
    assert flux.type == "hf" and flux.label == 1 and flux.label_col is None
    front = by_name["frontier-fakes"]
    assert front.keep_label == 1
    assert normalize_label(0, front.label_map) == 1
    assert normalize_label(1, front.label_map) == 0
    sid = by_name["sid-set"]
    assert sid.keep_label == 1
    assert normalize_label(1, sid.label_map) == 1
    assert normalize_label(2, sid.label_map) != 1
    assert normalize_label(0, sid.label_map) == 0
    ntire = by_name["ntire"]
    assert ntire.type == "ntire" and ntire.split == "train" and ntire.shard == -1
    assert cfg.eval_datasets == ["ntire_test", "openfake_test"]
    assert cfg.eval_openfake_max == 4096
    gs3 = by_name["gs-images-v3"]
    assert gs3.type == "folders" and gs3.fake_dirs
    gs4 = by_name["gs-images-v4"]
    assert gs4.type == "folders" and gs4.fake_dirs
