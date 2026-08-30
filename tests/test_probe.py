"""Multi-layer linear probe tests (offline, tiny backbone).

Verifies, without network access:
  * probe mode builds independent page and patch heads over concatenated
    multi-block features
  * layer specs resolve (auto spacing, negatives, dedup) and validate
  * training updates both heads but never the (frozen) backbone
  * checkpoints round-trip through load_checkpoint as probe models
  * the full training loop runs end to end in probe mode
"""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from seer.config import load_config
from seer.model import (
    SeerDetector,
    build_param_groups,
    detection_loss,
    load_checkpoint,
    save_checkpoint,
)
from seer.train import run


def _rand_pil(size=512, rng=random.Random(0)):
    arr = rng.randrange(256) + np.random.RandomState(rng.randrange(1 << 30)).randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_probe_build_and_forward():
    m = SeerDetector("tiny", pretrained=False, probe_layers=[0, 2, 3])
    assert m.probe
    assert m.probe_layers == [0, 2, 3]
    assert not hasattr(m, "global_head") and not hasattr(m, "local_head")
    # 3 taps * ([CLS ; mean-patch] = 2*hidden) input dim
    assert m.probe_head[-1].in_features == 3 * 2 * m.hidden_size
    assert m.probe_patch_head[-1].in_features == 3 * m.hidden_size
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out["logits"].shape == (2,)
    G = 224 // m.patch_size
    assert out["patch_logits"].shape == (2, G * G)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["patch_logits"]).all()
    # features must differ between taps (early vs late blocks)
    f = m.layer_features(x)
    assert f.shape == (2, 3 * 2 * m.hidden_size)
    per_tap = f.split(2 * m.hidden_size, dim=-1)
    assert not torch.allclose(per_tap[0], per_tap[2])
    page, patches = m.probe_features(x)
    assert page.shape == f.shape
    assert patches.shape == (2, G * G, 3 * m.hidden_size)
    print("probe model build + forward OK")


def test_probe_layer_resolution():
    m = SeerDetector("tiny", pretrained=False, probe_layers=[])  # auto
    assert m.probe_layers == [1, 2, 3]  # tiny has 4 blocks
    m = SeerDetector("tiny", pretrained=False, probe_layers=[3, -1, 0, 0])
    assert m.probe_layers == [0, 3]  # negatives + dedup
    try:
        SeerDetector("tiny", pretrained=False, probe_layers=[9])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("probe layer resolution OK")


def test_probe_train_step():
    torch.manual_seed(0)
    model = SeerDetector("tiny", pretrained=False, probe_layers=[0, 1, 3])
    model.freeze_backbone()
    backbone_before = {k: v.clone() for k, v in model.backbone.state_dict().items()}

    groups = build_param_groups(model, 1e-5, 1e-3, 0.8, 0.05)
    trainable = [p for g in groups for p in g["params"]]
    head_params = list(model.probe_head.parameters()) + list(
        model.probe_patch_head.parameters()
    )
    assert {id(p) for p in trainable} == {id(p) for p in head_params}

    opt = torch.optim.AdamW(groups)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    pl = y.view(-1, 1).expand(-1, 196).contiguous()
    loss = None
    for _ in range(2):
        out = model(x)
        assert out["logits"].requires_grad  # head runs outside the no_grad block
        assert out["patch_logits"].requires_grad
        loss, stats = detection_loss(out["logits"], out["patch_logits"], y, pl)
        opt.zero_grad()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_params)
        opt.step()
        assert torch.isfinite(loss), loss
        assert stats["loss_patch"] > 0.0

    for k, v in model.backbone.state_dict().items():
        assert torch.equal(v, backbone_before[k]), k
    print(f"probe train step OK (loss={loss.item():.4f})")


def test_probe_checkpoint_roundtrip():
    cfg = load_config(overrides=[
        "backbone=tiny", "pretrained=false", "res=224", "max_steps=1",
        "probe.enabled=true", "probe.layers=[0,2]",
    ])
    assert cfg.probe.enabled and cfg.probe.layers == [0, 2]
    model = SeerDetector("tiny", pretrained=False, probe_layers=cfg.probe.layers)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "probe.pt")
        save_checkpoint(path, model, cfg, step=1)
        m2, cfg2, _ = load_checkpoint(path)
        assert getattr(m2, "probe", False)
        assert m2.probe_layers == [0, 2]
        x = torch.randn(1, 3, 224, 224)
        o1, o2 = model(x), m2(x)
        assert torch.allclose(o1["logits"], o2["logits"], atol=1e-6)
        assert torch.allclose(o1["patch_logits"], o2["patch_logits"], atol=1e-6)
    print("probe checkpoint round-trip OK")


def test_probe_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "real").mkdir()
        (d / "fake").mkdir()
        for i in range(4):
            _rand_pil(64, random.Random(i)).save(d / "real" / f"r{i}.png")
            _rand_pil(64, random.Random(i + 100)).save(d / "fake" / f"f{i}.png")

        cfg = load_config(overrides=[
            "backbone=tiny", "pretrained=false", "res=224",
            "max_steps=2", "batch_size=2", "grad_accum=1",
            "head_lr=1.0e-3", "warmup_steps=1", "eval_every=2", "log_every=1",
            "ema_decay=0.999", "grad_checkpointing=true",
            "probe.enabled=true", "probe.layers=[0,2]",
            "heatmap_every=1", "heatmap_n=2",
            "misclass_every=2", "misclass_max=4",
            f"out_dir={d.as_posix()}/run",
        ])
        cfg.data.source = "folders"
        cfg.data.real_dirs = [str(d / "real")]
        cfg.data.fake_dirs = [str(d / "fake")]
        cfg.data.val_max_samples = 2

        run(cfg)
        last = os.path.join(d, "run", "last.pt")
        assert os.path.exists(last)
        hm = os.path.join(d, "run", "heatmaps", "step_000001.png")
        assert os.path.exists(hm)
        manifest = os.path.join(d, "run", "misclassified", "step_000002", "val", "manifest.jsonl")
        assert os.path.exists(manifest)
        m, cfg_dict, _ = load_checkpoint(last)
        assert getattr(m, "probe", False) and m.probe_layers == [0, 2]
        assert hasattr(m, "probe_patch_head")
        assert cfg_dict["probe"]["enabled"]
        x = torch.randn(1, 3, 224, 224)
        out = m(x)
        assert out["patch_logits"].shape[-1] == (224 // m.patch_size) ** 2
        from seer.data import clear_holdout
        clear_holdout()
    print("probe end-to-end training OK")


if __name__ == "__main__":
    test_probe_build_and_forward()
    test_probe_layer_resolution()
    test_probe_train_step()
    test_probe_checkpoint_roundtrip()
    test_probe_end_to_end()
    print("\nALL PROBE TESTS PASSED")
