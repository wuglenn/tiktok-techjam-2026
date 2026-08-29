"""Muon optimizer: Newton-Schulz, hybrid AdamW split, train step."""

import math

import torch
import torch.nn as nn

from seer.config import load_config
from seer.model import SeerDetector, build_param_groups, detection_loss
from seer.optim import Muon, build_optimizer, group_param_counts, newton_schulz5


def test_newton_schulz_near_orthogonal():
    torch.manual_seed(0)
    G = torch.randn(32, 32)
    Q = newton_schulz5(G, steps=10).float()
    gram = Q @ Q.T
    assert torch.isfinite(Q).all()
    # quintic NS is approximate polar factor, not exact UV^T
    assert (gram - torch.eye(32)).abs().mean() < 0.35
    tall = newton_schulz5(torch.randn(64, 16), steps=5)
    assert tall.shape == (64, 16) and torch.isfinite(tall).all()
    print("newton-schulz OK")


def test_muon_splits_2d_and_1d():
    lin = nn.Linear(16, 8)
    groups = [
        {"params": list(lin.parameters()), "lr": 1e-3, "weight_decay": 0.05, "name": "layer_0"}
    ]
    opt = Muon(groups, lr=1e-3)
    flags = {id(p): g["use_muon"] for g in opt.param_groups for p in g["params"]}
    assert flags[id(lin.weight)] is True
    assert flags[id(lin.bias)] is False
    print("muon / adamw split OK")


def test_embeddings_stay_on_adamw():
    emb = nn.Embedding(10, 8)
    groups = [{"params": list(emb.parameters()), "lr": 1e-3, "name": "embeddings"}]
    opt = Muon(groups, lr=1e-3)
    assert all(not g["use_muon"] for g in opt.param_groups)
    print("embeddings on adamw OK")


def test_muon_step_updates_params():
    torch.manual_seed(0)
    model = SeerDetector("tiny", pretrained=False)
    groups = build_param_groups(model, 1e-3, 1e-2, 0.8, 0.05)
    opt = build_optimizer(groups, load_config(overrides=["optimizer=muon", "lr=1e-3"]))
    n_muon, n_adam = group_param_counts(opt)
    assert n_muon > 0 and n_adam > 0

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0.0, 1.0])
    pl = y.view(-1, 1).expand(-1, 196).contiguous()
    out = model(x)
    loss, _ = detection_loss(out["logits"], out["patch_logits"], y, pl)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)

    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert changed, "expected some parameters to move"
    # 2D hidden weights store Muon momentum; 1D store Adam moments
    muon_states = sum(1 for s in opt.state.values() if "momentum_buffer" in s)
    adam_states = sum(1 for s in opt.state.values() if "exp_avg" in s)
    assert muon_states > 0 and adam_states > 0
    print(f"muon step OK (loss={loss.item():.4f} muon={n_muon} adamw={n_adam})")


def test_build_optimizer_adamw():
    model = SeerDetector("tiny", pretrained=False)
    groups = build_param_groups(model, 1e-4, 1e-3, 0.8, 0.05)
    opt = build_optimizer(groups, load_config(overrides=["optimizer=adamw"]))
    assert type(opt).__name__ == "AdamW"
    print("adamw fallback OK")


def test_hero_configs_optimizer():
    hero = load_config("configs/seer_vitl_512.yaml")
    assert hero.optimizer == "adamw"
    probe = load_config("configs/seer_probe.yaml")
    assert probe.optimizer == "muon"
    print("hero adamw / probe muon OK")


def test_adjust_lr_scales_with_shape():
    from seer.optim import _adjust_lr

    lr = 1e-3
    scaled = _adjust_lr(lr, (1024, 1024), "match_rms_adamw")
    assert math.isclose(scaled, lr * 0.2 * 32.0)
    orig = _adjust_lr(lr, (64, 16), "original")
    assert math.isclose(orig, lr * 2.0)
    print("lr adjustment OK")


if __name__ == "__main__":
    test_newton_schulz_near_orthogonal()
    test_muon_splits_2d_and_1d()
    test_embeddings_stay_on_adamw()
    test_muon_step_updates_params()
    test_build_optimizer_adamw()
    test_hero_configs_optimizer()
    test_adjust_lr_scales_with_shape()
    print("\nALL OPTIM TESTS PASSED")
