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


SID_MAP = {0: 0, 1: 1, 2: 1}


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
    assert BY_KEY["dda-train"].repo_id == "Junwei-Xi/DDA-Training-Set"
    assert BY_KEY["dda-train"].source == "hf_files"
    keys = {s.key for s in select(tiers=[2])}
    assert {"flux-reason-6m", "frontier-fakes", "sid-set", "dda-train"} <= keys


def test_sid_collapses_tampered_to_fake():
    assert normalize_label(0, SID_MAP) == 0
    assert normalize_label(1, SID_MAP) == 1
    assert normalize_label(2, SID_MAP) == 1
    assert normalize_label("2", SID_MAP) == 1


def test_hero_config_keeps_only_frontier_fakes():
    cfg = load_config("configs/seer_vitl_512.yaml")
    by_name = {s.name: s for s in cfg.data.sources}
    assert set(by_name) >= {"flux-reason", "frontier-fakes", "sid-set", "dda-train"}
    flux = by_name["flux-reason"]
    assert flux.type == "hf" and flux.label == 1 and flux.label_col is None
    front = by_name["frontier-fakes"]
    assert front.keep_label == 1
    assert normalize_label(0, front.label_map) == 1
    assert normalize_label(1, front.label_map) == 0
    sid = by_name["sid-set"]
    assert sid.keep_label == 1
    assert normalize_label(2, sid.label_map) == 1
    assert normalize_label(0, sid.label_map) == 0
    dda = by_name["dda-train"]
    assert dda.type == "folders" and dda.fake_dirs and not dda.real_dirs
