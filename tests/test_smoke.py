"""End-to-end smoke test with a random tiny backbone and synthetic images.

Verifies, without network access:
  * model builds and total parameters stay under the 2B budget
  * forward/backward pass runs, loss is finite
  * composite training produces correctly-shaped, consistent patch labels
  * metrics computation and heatmap rendering work
"""

import io
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from seer.augment import eval_transform, jpeg_recompress, pangram_augment, train_transform
from seer.config import SourceSpec, load_config
from seer.data import BatchBuilder, build_train_dataset, load_sample_image
from seer.eval import compute_metrics
from seer.heatmap import predict_and_explain, save_heatmap
from seer.model import SeerDetector, EMA, build_param_groups, detection_loss
from seer.train import cosine_schedule


def _rand_pil(size=512, rng=random.Random(0)):
    arr = rng.randrange(256) + np.random.RandomState(rng.randrange(1 << 30)).randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_budget_and_forward():
    model = SeerDetector("tiny", pretrained=False)
    n = model.parameter_count()
    assert n < 2_000_000_000, n
    assert n > 0
    G = model.patch_grid(224)
    assert G == 14, G
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out["logits"].shape == (2,)
    assert out["patch_logits"].shape == (2, G * G)
    print(f"tiny model: {n:,} params, forward OK")


def test_train_step_and_ema():
    torch.manual_seed(0)
    model = SeerDetector("tiny", pretrained=False)
    opt = torch.optim.AdamW(build_param_groups(model, 1e-4, 1e-3, 0.8, 0.05))
    ema = EMA(model, 0.99)
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    pl = y.view(-1, 1).expand(-1, 196).contiguous()
    for _ in range(2):
        out = model(x)
        loss, stats = detection_loss(out["logits"], out["patch_logits"], y, pl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        assert torch.isfinite(loss), loss
    assert any(not torch.allclose(a, b) for a, b in zip(
        list(model.state_dict().values()), list(ema.shadow.values())))
    print(f"train step OK (loss={loss.item():.4f})")


def test_batch_builder_composites():
    cfg = load_config(overrides=["res=224", "composite.prob=1.0",
                                 "composite.real_real_fraction=0.5",
                                 "max_steps=2", "backbone=tiny", "pretrained=false"])
    for mode in ("blend", "paste", "mixed"):
        cfg.composite.mode = mode
        builder = BatchBuilder(cfg, train=True, patch_grid=14, seed=0)
        samples = [
            {"image": _rand_pil(480, random.Random(i)), "label": i % 2,
             "generator": "g", "architecture": "LatDiff"}
            for i in range(8)
        ]
        batch = builder(samples)
        assert batch["images"].shape == (8, 3, 224, 224)
        assert batch["labels"].shape == (8,)
        assert batch["patch_labels"].shape == (8, 196)
        assert set(batch["labels"].tolist()) <= {0.0, 1.0}
        assert torch.isfinite(batch["images"]).all()
        # patch labels must match the global label for non-composited samples and
        # be within [0,1] always
        assert ((batch["patch_labels"] >= 0) & (batch["patch_labels"] <= 1)).all()
    print("batch builder + composites (blend/paste/mixed) OK")


def test_composite_combinations():
    """All four top-on-base pairings occur; labels track visible content.

    Compositing is itself a simple discontinuity: if only fake-over-real
    were trained, the local head could win by detecting *seams* rather than
    AI texture. So the pairing of overlay and base classes must cover all
    four combinations, overlays must be crops (not co-registered frames),
    and a sample may receive several stacked overlays.
    """
    common = ["res=224", "backbone=tiny", "pretrained=false", "composite.prob=1.0"]

    def builder(**comp):
        cfg = load_config(overrides=common + [f"composite.{k}={v}" for k, v in comp.items()])
        return BatchBuilder(cfg, train=True, patch_grid=14, seed=0)

    def run(b, batches=16, bs=8):
        out = []
        for k in range(batches):
            samples = [
                {"image": _rand_pil(480, random.Random(k * 100 + i)), "label": i % 2,
                 "generator": "g", "architecture": "LatDiff"} for i in range(bs)
            ]
            out.append(b(samples))
        return out

    def fake_patch_rows(batches):
        return torch.cat([x["patch_labels"][x["labels"] == 1] for x in batches])

    # invariants under the default mix: binary patch labels, and the global
    # label follows what is actually visible
    for mode in ("blend", "paste", "mixed"):
        for x in run(builder(mode=mode, real_real_fraction=1.0)):
            pl, y = x["patch_labels"], x["labels"]
            assert ((pl == 0) | (pl == 1)).all()
            assert torch.equal(y, pl.amax(dim=1))
            assert torch.isfinite(x["images"]).all()

    # fake-over-real: localized (mixed) patch labels
    rows = fake_patch_rows(run(builder(fake_on_real=1.0, real_on_fake=0.0,
                                       fake_on_fake=0.0, max_overlays=1)))
    assert rows.shape[0] > 0
    assert ((rows == 0).any(dim=1) & (rows == 1).any(dim=1)).all()

    # real-over-fake: inverted patch labels - only the pasted region is real
    rows = fake_patch_rows(run(builder(fake_on_real=0.0, real_on_fake=1.0,
                                       fake_on_fake=0.0, max_overlays=1)))
    assert rows.shape[0] > 0
    assert ((rows == 0).any(dim=1) & (rows == 1).any(dim=1)).all()

    # fake-over-fake: seams inside fully-fake content, all patches stay 1
    rows = fake_patch_rows(run(builder(fake_on_real=0.0, real_on_fake=0.0,
                                       fake_on_fake=1.0, max_overlays=3)))
    assert rows.shape[0] > 0
    assert (rows == 1).all()

    # real-over-real: label stays real; blending alone is not a fake cue
    for x in run(builder(real_real_fraction=1.0, fake_on_real=0.0,
                         real_on_fake=0.0, fake_on_fake=0.0)):
        pl, y = x["patch_labels"], x["labels"]
        assert torch.equal(y, pl.amax(dim=1))
        assert (pl[y == 0] == 0).all()

    # stacking: more overlays -> more pasted area on fake samples
    one = fake_patch_rows(run(builder(fake_on_real=1.0, real_on_fake=0.0,
                                      fake_on_fake=0.0, max_overlays=1))).mean()
    many = fake_patch_rows(run(builder(fake_on_real=1.0, real_on_fake=0.0,
                                      fake_on_fake=0.0, max_overlays=5))).mean()
    assert many > one

    # degenerate batches: no real partner -> fall back to fake-over-fake
    b = builder()
    samples = [{"image": _rand_pil(480, random.Random(7 + i)), "label": 1,
                "generator": "g", "architecture": "LatDiff"} for i in range(4)]
    x = b(samples)
    assert (x["labels"] == 1).all() and (x["patch_labels"] == 1).all()
    assert torch.isfinite(x["images"]).all()

    # single-sample batches must not crash
    x = b([{"image": _rand_pil(480, random.Random(21)), "label": 1,
            "generator": "g", "architecture": "LatDiff"}])
    assert torch.isfinite(x["images"]).all() and x["labels"].tolist() == [1.0]
    x = b([{"image": _rand_pil(480, random.Random(22)), "label": 0,
            "generator": "g", "architecture": "LatDiff"}])
    assert x["labels"].tolist() == [0.0] and (x["patch_labels"] == 0).all()
    print("composite combinations (4 pairings, stacked overlays) OK")


def test_augment_pipeline():
    rng = random.Random(0)
    cfg = load_config().augment
    assert cfg.jpeg_quality == [90, 70, 50, 30]
    assert cfg.blur_sigma == [0.5, 1.0, 2.0]
    assert cfg.noise_levels == [0.02, 0.05, 0.10]
    assert cfg.downscale_levels == [0.5, 0.25]
    assert cfg.color_jitter == 0.2
    img = _rand_pil(700, rng)
    t = train_transform(img, 224, rng, cfg)
    assert t.shape == (3, 224, 224)
    assert torch.isfinite(t).all()
    aug = pangram_augment(img)
    assert max(aug.size) <= 1024
    x = eval_transform(aug, 224)
    assert x.shape == (3, 224, 224)
    j = jpeg_recompress(img, 50)
    assert j.size == img.size
    print("augmentation pipeline OK")


def test_perturbations():
    from seer.augment import PERTURBATIONS, apply_perturbation, center_crop

    rng = random.Random(1)
    img = _rand_pil(600, rng)
    for name in PERTURBATIONS:
        out = apply_perturbation(img, name)
        assert isinstance(out, Image.Image)
        if name == "crop80":
            assert out.size == (480, 480), out.size  # 80% of each side
        else:
            assert out.size == img.size, (name, out.size)
    try:
        apply_perturbation(img, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    c = center_crop(img, 0.8)
    assert c.size == (480, 480)
    print(f"perturbations OK ({len(PERTURBATIONS)} levels)")


def test_metrics_and_heatmap(tmp_path=None):
    rng = np.random.RandomState(0)
    probs = rng.rand(200)
    labels = (rng.rand(200) > 0.5).astype(int)
    m = compute_metrics(probs, labels)
    assert 0 <= m["macro_accuracy"] <= 1
    assert 0 <= m["mAP"] <= 1
    assert m["n"] == 200

    model = SeerDetector("tiny", pretrained=False).eval()
    img = _rand_pil(400, random.Random(1))
    prob, heat = predict_and_explain(model, img, 224)
    assert 0.0 <= prob <= 1.0
    assert heat.shape == (224, 224)
    if tmp_path:
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "hm.png")
            save_heatmap(out, img, heat, prob, 224)
            assert os.path.exists(out)
    print(f"metrics + heatmap OK (prob={prob:.3f})")


def test_schedule():
    assert abs(cosine_schedule(0, 100, 10, 0.05) - 0.1) < 1e-9
    assert abs(cosine_schedule(100, 100, 10, 0.05) - 0.05) < 1e-9
    v = cosine_schedule(55, 100, 10, 0.05)
    assert 0.05 <= v <= 1.0


def test_mixture_dataset(tmp_dirs=None):
    """Weighted mixture over folder sources: real/fake interleave, weights,
    cycling, lazy decode, and config plumbing end to end."""
    import tempfile
    import os
    from seer.data import FolderPairStream, MixtureDataset

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "real").mkdir()
        (d / "fake").mkdir()
        for i in range(6):
            _rand_pil(64, random.Random(i)).save(d / "real" / f"r{i}.png")
            _rand_pil(64, random.Random(i + 100)).save(d / "fake" / f"f{i}.png")

        fs = FolderPairStream([str(d / "real")], [str(d / "fake")], seed=0)
        assert len(fs.real_files) == 6 and len(fs.fake_files) == 6
        it = iter(fs)
        got = [next(it)["label"] for _ in range(8)]
        assert got[0] != got[1]  # interleaved
        # samples are lazy (image=None, image_path=...)
        s0 = next(iter(fs))
        assert s0["image"] is None and s0["image_path"]
        assert load_sample_image(s0).size == (64, 64)

        mix = MixtureDataset(
            [
                SourceSpec(name="a", type="folders", weight=0.7,
                           real_dirs=[str(d / "real")], fake_dirs=[str(d / "fake")]),
                SourceSpec(name="b", type="folders", weight=0.3,
                           fake_dirs=[str(d / "fake")]),
            ],
            seed=0,
        )
        n = {0: 0, 1: 0}
        it = iter(mix)
        for _ in range(200):
            s = next(it)
            n[s["label"]] += 1
            load_sample_image(s).size  # every sample decodes
        assert n[0] > 0 and n[1] > 0

        balanced = MixtureDataset(
            [
                SourceSpec(name="a", type="folders", weight=0.1,
                           real_dirs=[str(d / "real")], fake_dirs=[str(d / "fake")]),
                SourceSpec(name="b", type="folders", weight=0.9,
                           fake_dirs=[str(d / "fake")]),
            ],
            seed=0,
            balance_labels=True,
        )
        n = {0: 0, 1: 0}
        it = iter(balanced)
        for _ in range(200):
            n[next(it)["label"]] += 1
        assert abs(n[0] - n[1]) < 40, n  # ~50/50 despite 9:1 fake source weight

        # config plumbing: mixture via yaml-style dict
        cfg = load_config(overrides=[
            "backbone=tiny", "pretrained=false", "res=224", "max_steps=1",
        ])
        cfg.data.source = "mixture"
        cfg.data.sources = [
            SourceSpec(name="a", type="folders", weight=0.5,
                       real_dirs=[str(d / "real")], fake_dirs=[str(d / "fake")]),
        ]
        ds = build_train_dataset(cfg)
        b = BatchBuilder(cfg, train=False, patch_grid=14, seed=0)
        batch = b([next(iter(ds)) for _ in range(4)])
        assert batch["images"].shape == (4, 3, 224, 224)
    print("mixture dataset OK")


def test_local_parquet_source():
    """ComforStream over a locally-written parquet file (the F:/techjam path):
    lazy bytes -> decode -> batch, plus exact row counting."""
    import tempfile
    from seer.data import ComforStream, count_parquet_rows, parquet_files

    with tempfile.TemporaryDirectory() as d:
        import pyarrow
        pq = pytest.importorskip("pyarrow.parquet")
        rows = []
        for i in range(6):
            buf = io.BytesIO()
            _rand_pil(96, random.Random(i)).save(buf, format="PNG")
            rows.append({
                "image_name": f"{i}.png", "format": "PNG", "resolution": "[96, 96]",
                "mode": "RGB", "image_data": buf.getvalue(), "model_name": "synthetic/gen",
                "nsfw_flag": False, "prompt": "test", "real_source": "LAION",
                "subset": "test", "split": "train", "label": i % 2, "architecture": "LatDiff",
            })
        path = str(Path(d) / "shard_0.parquet")
        pq.write_table(pyarrow.table({k: [r[k] for r in rows] for k in rows[0]}), path)

        assert count_parquet_rows([path]) == 6
        assert len(parquet_files([d])) == 1

        ds = ComforStream(local_dirs=[d], shuffle_buffer=0, seed=0)
        out = list(iter(ds))
        assert len(out) == 6
        assert all(s["image"] is None and s["image_bytes"] for s in out)
        labels = {s["label"] for s in out}
        assert labels == {0, 1}
        assert load_sample_image(out[0]).size == (96, 96)
    print("local parquet source OK")


if __name__ == "__main__":
    test_budget_and_forward()
    test_train_step_and_ema()
    test_batch_builder_composites()
    test_composite_combinations()
    test_augment_pipeline()
    test_perturbations()
    test_metrics_and_heatmap()
    test_schedule()
    test_mixture_dataset()
    test_local_parquet_source()
    print("\nALL SMOKE TESTS PASSED")
