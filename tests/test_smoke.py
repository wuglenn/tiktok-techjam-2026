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

from seer.augment import (
    eval_transform, jpeg_recompress, pangram_augment, post_stack_transform,
    train_transform,
)
from seer.config import SourceSpec, load_config
from seer.data import BatchBuilder, DecodeError, SkipBatch, build_train_dataset, load_sample_image
from seer.eval import compute_metrics
from seer.heatmap import predict_and_explain, save_heatmap
from seer.model import SeerDetector, EMA, build_param_groups, detection_loss, _patch_pos_weight
from seer.train import cosine_schedule, train_data_seed


def _rand_pil(size=512, rng=random.Random(0)):
    arr = rng.randrange(256) + np.random.RandomState(rng.randrange(1 << 30)).randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_attn_auto_falls_back_to_sdpa():
    from seer.model import resolve_attn_implementation

    assert resolve_attn_implementation("sdpa") == "sdpa"
    assert resolve_attn_implementation("auto") in {
        "flash_attention_4", "flash_attention_3", "flash_attention_2", "sdpa",
    }
    print("attn resolve OK")


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
        assert stats["patch_pos_weight"] >= 1.0
    assert any(not torch.allclose(a, b) for a, b in zip(
        list(model.state_dict().values()), list(ema.shadow.values())))
    print(f"train step OK (loss={loss.item():.4f})")


def test_batch_builder_composites():
    cfg = load_config(overrides=["res=224", "composite.prob=1.0",
                                 "composite.real_real_fraction=0.5",
                                 "max_steps=2", "backbone=tiny", "pretrained=false"])
    for mode in ("blend", "paste", "mixed"):
        for feather in ("hard", "soft", "mixed"):
            cfg.composite.mode = mode
            cfg.composite.feather = feather
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
            # patch labels in [0, 1]; soft seams may be mixed, page stays binary
            assert ((batch["patch_labels"] >= 0) & (batch["patch_labels"] <= 1)).all()
    print("batch builder + composites (blend/paste × hard/soft) OK")


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
        return torch.cat([x["patch_labels"][x["labels"] > 0] for x in batches])

    # invariants: patches in [0, 1] (soft seams mixed); page is binary (any AI → fake)
    for mode in ("blend", "paste", "mixed"):
        for x in run(builder(mode=mode, feather="mixed", real_real_fraction=1.0)):
            pl, y = x["patch_labels"], x["labels"]
            assert ((pl >= 0) & (pl <= 1)).all()
            assert set(y.tolist()) <= {0.0, 1.0}
            assert torch.equal(y, (pl.amax(dim=1) > 0).float())
            assert torch.isfinite(x["images"]).all()

    # fake-over-real: localized mixed patch labels (seam cells between 0 and 1)
    rows = fake_patch_rows(run(builder(fake_on_real=1.0, real_on_fake=0.0,
                                       fake_on_fake=0.0, max_overlays=1)))
    assert rows.shape[0] > 0
    assert ((rows < 0.5).any(dim=1) & (rows > 0).any(dim=1)).all()

    # real-over-fake: inverted — overlay cells drop below 1, fake base stays
    rows = fake_patch_rows(run(builder(fake_on_real=0.0, real_on_fake=1.0,
                                       fake_on_fake=0.0, max_overlays=1)))
    assert rows.shape[0] > 0
    assert ((rows < 1).any(dim=1) & (rows > 0.5).any(dim=1)).all()

    # fake-over-fake as a single paste: patches stay 1 (later real
    # overlays on a fake page are allowed, so stacks can mix)
    rows = fake_patch_rows(run(builder(fake_on_real=0.0, real_on_fake=0.0,
                                       fake_on_fake=1.0, max_overlays=1)))
    assert rows.shape[0] > 0
    assert torch.allclose(rows, torch.ones_like(rows))

    # real-over-real: blending reals is not a fake cue
    for x in run(builder(real_real_fraction=1.0, fake_on_real=0.0,
                         real_on_fake=0.0, fake_on_fake=0.0)):
        pl, y = x["patch_labels"], x["labels"]
        assert torch.equal(y, (pl.amax(dim=1) > 0).float())
        assert (pl[y == 0] == 0).all()

    # stacking: later overlays may be real or fake; rows stay valid FoR mixes
    many = fake_patch_rows(run(builder(fake_on_real=1.0, real_on_fake=0.0,
                                      fake_on_fake=0.0, max_overlays=5)))
    assert many.shape[0] > 0
    assert ((many < 0.5).any(dim=1) & (many > 0).any(dim=1)).all()

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

    # soft feather: seam patches are strictly fractional (spatial mix, not alpha %)
    soft_rows = fake_patch_rows(run(builder(
        feather="soft", fake_on_real=1.0, real_on_fake=0.0, fake_on_fake=0.0,
        max_overlays=1,
    )))
    assert ((soft_rows > 0) & (soft_rows < 1)).any()
    print("composite combinations (4 pairings, stacked overlays) OK")


def test_composite_pairing_balance():
    """FoR/RoF/FoF/RoR quotas keep page labels 1:1 on a balanced batch."""
    cfg = load_config(overrides=[
        "res=224", "backbone=tiny", "pretrained=false", "composite.prob=1.0",
    ])
    b = BatchBuilder(cfg, train=True, patch_grid=14, seed=0)
    n_real = n_fake = 0
    for k in range(16):
        samples = [
            {"image": _rand_pil(480, random.Random(k * 100 + i)), "label": i % 2,
             "generator": "g", "architecture": "LatDiff"} for i in range(8)
        ]
        x = b(samples)
        n_real += int((x["labels"] == 0).sum())
        n_fake += int((x["labels"] == 1).sum())
        assert torch.equal(x["labels"], (x["patch_labels"].amax(dim=1) > 0).float())
    assert n_real == n_fake
    print("composite pairing balance (page 1:1) OK")


def test_post_stack_shared_pass():
    """Composites get one aligned wild-sim pass after stacking.

    Per-layer train_transform still runs first; the shared pass then
    covers the whole page so mismatched JPEG/noise is not a shortcut.
    Patch labels stay registered (no crop).
    """
    common = [
        "res=224", "backbone=tiny", "pretrained=false",
        "composite.prob=1.0", "composite.max_overlays=1",
        "augment.jpeg_prob=1.0", "augment.webp_prob=0",
        "augment.extra_distort_prob=0", "augment.downscale_prob=0",
        "augment.blur_prob=0", "augment.noise_prob=0",
        "augment.color_jitter_prob=0", "augment.grayscale_prob=0",
        "augment.hflip_prob=0",
    ]
    samples = [
        {"image": _rand_pil(480, random.Random(i)), "label": i % 2,
         "generator": "g", "architecture": "LatDiff"}
        for i in range(8)
    ]
    off = load_config(overrides=common + ["composite.post_prob=0"])
    on = load_config(overrides=common + ["composite.post_prob=1"])
    x0 = BatchBuilder(off, train=True, patch_grid=14, seed=0)(samples)
    x1 = BatchBuilder(on, train=True, patch_grid=14, seed=0)(samples)
    assert torch.equal(x0["labels"], x1["labels"])
    assert torch.equal(x0["patch_labels"], x1["patch_labels"])
    # at least one stacked page must change; the shared JPEG is global
    assert not torch.allclose(x0["images"], x1["images"], atol=1e-5)
    assert torch.isfinite(x1["images"]).all()

    cfg = load_config().augment
    cfg.jpeg_prob = 1.0
    cfg.webp_prob = 0.0
    cfg.extra_distort_prob = 0.0
    cfg.downscale_prob = 0.0
    cfg.blur_prob = 0.0
    cfg.noise_prob = 0.0
    cfg.color_jitter_prob = 0.0
    cfg.grayscale_prob = 0.0
    t = train_transform(_rand_pil(224, random.Random(3)), 224, random.Random(4), cfg)
    out = post_stack_transform(t, 224, random.Random(5), cfg)
    assert out.shape == t.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, t, atol=1e-4)
    print("post-stack shared pass OK")


def test_overlay_shapes_not_just_rects():
    """Hard occupancy is not always a filled axis-aligned rectangle."""
    cfg = load_config(overrides=["res=224", "backbone=tiny", "pretrained=false"])
    b = BatchBuilder(cfg, train=True, patch_grid=14, seed=1)
    fills = []
    for _ in range(48):
        occ = b._rand_occupancy("hard")
        hit = occ > 0.5
        if not bool(hit.any()):
            continue
        ys, xs = torch.where(hit)
        box = (int(ys.max()) - int(ys.min()) + 1) * (int(xs.max()) - int(xs.min()) + 1)
        fills.append(float(hit.sum()) / max(box, 1))
    assert fills and min(fills) < 0.85
    print("overlay shapes not just rects OK")


def test_overlay_crop_keeps_semantics():
    """Overlays crop a large, difficulty-varying fraction of the source."""
    cfg = load_config(overrides=["res=224", "backbone=tiny", "pretrained=false"])
    b = BatchBuilder(cfg, train=True, patch_grid=14, seed=0)
    fracs = []
    for _ in range(60):
        _, _, ch, cw, _ = b._crop_geom(512, 512, 80, 80)
        fracs.append((ch / 512, cw / 512))
    flat = [s for hw in fracs for s in hw]
    # hard tier floor is 0.45; never a small texture chip
    assert min(flat) >= 0.45 - 1e-9
    assert max(flat) <= 1.0
    means = [(h + w) / 2 for h, w in fracs]
    # all three difficulty tiers should appear
    assert min(means) < 0.70 and max(means) > 0.80
    print("overlay crop keeps semantics (difficulty-varying) OK")


def test_augment_pipeline():
    rng = random.Random(0)
    cfg = load_config().augment
    assert {90, 70, 50, 30, 20, 10, 5} <= set(cfg.jpeg_quality)
    assert {0.5, 1.0, 2.0, 4.0} <= set(cfg.blur_sigma)
    assert {0.02, 0.05, 0.10, 0.20} <= set(cfg.noise_levels)
    assert {0.5, 0.25, 0.125} <= set(cfg.downscale_levels)
    assert cfg.color_jitter >= 0.2
    assert cfg.extra_distort_prob >= 0.50
    assert cfg.extra_distort_max >= 4
    assert cfg.webp_prob > 0
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
    from seer.augment import (
        BENCHMARK_PERTURBATIONS,
        HARD_PERTURBATIONS,
        PERTURBATIONS,
        apply_perturbation,
        center_crop,
        perturbation_names,
    )

    rng = random.Random(1)
    img = _rand_pil(600, rng)
    assert perturbation_names("all") == list(BENCHMARK_PERTURBATIONS)
    assert "jpeg30" in BENCHMARK_PERTURBATIONS and "jpeg10" in HARD_PERTURBATIONS
    assert "crop80" in BENCHMARK_PERTURBATIONS and "blur2.0" in BENCHMARK_PERTURBATIONS
    for name in (
        "doublejpeg", "chroma420", "fftlp", "grain", "social", "median3",
        "gridshift", "resample", "phase", "chroman", "perspective", "recode",
    ):
        assert name in HARD_PERTURBATIONS
    for name in PERTURBATIONS:
        out = apply_perturbation(img, name)
        assert isinstance(out, Image.Image)
        if name.startswith("crop"):
            assert out.size[0] < img.size[0] and out.size[1] < img.size[1]
        elif name != "pangram":
            assert out.size == img.size, (name, out.size)
    assert apply_perturbation(img, "crop80").size == (480, 480)
    try:
        apply_perturbation(img, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    c = center_crop(img, 0.8)
    assert c.size == (480, 480)
    print(f"perturbations OK ({len(PERTURBATIONS)} levels)")


def test_fingerprint_mask_ops():
    from seer.augment import (
        _chroma_aberration,
        _chroma_subsample,
        _double_jpeg,
        _extra_train_distort,
        _fft_lowpass,
        _film_grain,
        _gamma,
        _median,
        _small_rotate,
        _social_reencode,
        _speckle,
        _subpixel_nudge,
        _surface_blur,
        _unsharp,
        _fft_phase_noise,
        _hue_shift,
        _jpeg_grid_shift,
        _resample_mismatch,
        _recode_stack,
        _chroma_noise,
        _perspective_nudge,
    )

    rng = random.Random(3)
    img = _rand_pil(96, rng)
    ops = (
        _double_jpeg(img, 70, 35),
        _chroma_subsample(img, 2),
        _median(img, 3),
        _unsharp(img),
        _small_rotate(img, 4.0),
        _subpixel_nudge(img, 0.8, -0.5),
        _gamma(img, 0.7),
        _film_grain(img, 0.04, 8, rng),
        _chroma_aberration(img, 2),
        _fft_lowpass(img, 0.32),
        _social_reencode(img, rng),
        _jpeg_grid_shift(img, 3, 5, 40),
        _resample_mismatch(img, 0.35),
        _surface_blur(img),
        _fft_phase_noise(img, 0.3, 0.2, rng),
        _hue_shift(img, 15.0),
        _chroma_noise(img, 0.04, rng),
        _perspective_nudge(img, 6.0, rng),
        _speckle(img, 0.06, rng),
        _recode_stack(img, rng),
    )
    for out in ops:
        assert out.size == img.size and out.mode == "RGB"
    for _ in range(24):
        out = _extra_train_distort(img, rng)
        assert out.size == img.size and out.mode == "RGB"
    print("fingerprint-mask ops OK")


def test_patch_pos_weight():
    y = torch.zeros(10, 4)
    y[0, 0] = 1.0
    w = _patch_pos_weight(y)
    assert abs(float(w) - 39.0) < 1e-5
    assert _patch_pos_weight(torch.ones(8)) is None
    assert _patch_pos_weight(torch.zeros(8)) is None
    logits = torch.zeros(8)
    labels = torch.zeros(8)
    pl = torch.zeros(8, 4)
    pl[0, 0] = 1.0
    _, bal = detection_loss(logits, torch.zeros(8, 4), labels, pl, balance_patch=True)
    _, raw = detection_loss(logits, torch.zeros(8, 4), labels, pl, balance_patch=False)
    assert bal["patch_pos_weight"] == 31.0
    assert raw["patch_pos_weight"] == 1.0
    print("patch pos_weight OK")


def test_metrics_and_heatmap(tmp_path=None):
    rng = np.random.RandomState(0)
    probs = rng.rand(200)
    labels = (rng.rand(200) > 0.5).astype(int)
    m = compute_metrics(probs, labels)
    assert 0 <= m["macro_accuracy"] <= 1
    assert 0 <= m["mAP"] <= 1
    assert 0 <= m["f1"] <= 1
    assert 0 <= m["auroc"] <= 1
    assert m["n"] == 200
    perfect = compute_metrics(np.array([0.1, 0.9, 0.8, 0.05]), np.array([0, 1, 1, 0]))
    assert perfect["f1"] == 1.0
    assert perfect["auroc"] == 1.0

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


def test_error_bank_keeps_most_confident_mistakes():
    """The bank must keep the *worst* k errors of each kind, and only errors."""
    import tempfile
    import torch as _torch
    from seer.eval import ErrorBank

    bank = ErrorBank(k=2, res=64)
    img = _rand_pil(64, random.Random(3))
    patch = _torch.zeros(16)  # 4x4 grid

    # correct predictions are never kept
    bank.add(img, 0.02, 0, patch, {"image_name": "ok_real.png"})
    bank.add(img, 0.98, 1, patch, {"image_name": "ok_fake.png"})
    assert bank.counts() == {"fp": 0, "fn": 0}

    # false positives: reals scored above threshold, ranked by confidence
    for p in (0.55, 0.99, 0.75):
        bank.add(img, p, 0, patch, {"image_name": f"real_{p}.png"})
    # false negatives: fakes scored below threshold, ranked by 1 - p
    for p in (0.45, 0.01, 0.30):
        bank.add(img, p, 1, patch, {"image_name": f"fake_{p}.png"})
    assert bank.counts() == {"fp": 2, "fn": 2}

    with tempfile.TemporaryDirectory() as d:
        entries = bank.dump(d)
        assert len(entries) == 4
        fp = [e for e in entries if e["kind"] == "fp"]
        fn = [e for e in entries if e["kind"] == "fn"]
        assert [e["prob_ai"] for e in fp] == [0.99, 0.75]   # most confident first
        assert [e["prob_ai"] for e in fn] == [0.01, 0.30]
        for e in entries:
            assert Path(e["file"]).exists()
            assert e["explained"] is True

    # a checkpoint without a patch head still dumps the image, unexplained
    plain = ErrorBank(k=1, res=64)
    plain.add(img, 0.9, 0, None, {"image_name": "no_head.png"})
    with tempfile.TemporaryDirectory() as d:
        entries = plain.dump(d)
        assert len(entries) == 1 and entries[0]["explained"] is False
        assert Path(entries[0]["file"]).exists()
    print("error bank OK (top-k false positives / negatives + heatmaps)")


def test_eval_dumps_error_analysis():
    """End-to-end: folders eval writes metrics plus explained error panels."""
    import json
    import tempfile
    from seer.eval import evaluate_checkpoint
    from seer.model import save_checkpoint

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "real").mkdir()
        (d / "fake").mkdir()
        for i in range(4):
            _rand_pil(256, random.Random(i)).save(d / "real" / f"r{i}.png")
            _rand_pil(256, random.Random(i + 50)).save(d / "fake" / f"f{i}.png")

        cfg = load_config(overrides=["backbone=tiny", "pretrained=false",
                                     "res=224", "max_steps=1"])
        model = SeerDetector("tiny", pretrained=False)
        ckpt = str(d / "ckpt.pt")
        save_checkpoint(ckpt, model, cfg, 1)

        out_json = str(d / "metrics.json")
        m = evaluate_checkpoint(
            checkpoint=ckpt, dataset="folders",
            real_dirs=[str(d / "real")], fake_dirs=[str(d / "fake")],
            batch_size=4, device="cpu", res=224,
            error_dir=str(d / "errors"), error_n=2, out_json=out_json,
        )
        assert m["n"] == 8 and m["n_real"] == 4 and m["n_fake"] == 4
        saved = json.load(open(out_json, encoding="utf-8"))
        assert saved["n"] == 8

        # an untrained head cannot be right about both classes at threshold 0.5
        entries = m.get("error_analysis") or []
        assert entries, "expected at least one misclassification to dump"
        for e in entries:
            assert Path(e["file"]).exists()
            assert 0.0 <= e["prob_ai"] <= 1.0
            assert e["kind"] in ("fp", "fn")
    print(f"eval error analysis OK ({len(entries)} panels)")


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

        # resume offsets the mixture seed so the first samples are not the
        # same prefix that a from-scratch run would have already consumed
        assert train_data_seed(0, 0) == 0
        assert train_data_seed(0, 28500) == 28500
        specs = [
            SourceSpec(name="a", type="folders", weight=1.0,
                       real_dirs=[str(d / "real")], fake_dirs=[str(d / "fake")]),
        ]
        it = iter(MixtureDataset(specs, seed=0))
        first = [next(it)["image_path"] for _ in range(8)]
        it_resume = iter(MixtureDataset(specs, seed=train_data_seed(0, 28500)))
        resumed = [next(it_resume)["image_path"] for _ in range(8)]
        assert first != resumed
        cfg.seed = 0
        assert build_train_dataset(cfg, seed=28500).seed == 28500
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
        assert out[0]["subset"] == "test" and out[0]["real_source"] == "LAION"
        assert load_sample_image(out[0]).size == (96, 96)
    print("local parquet source OK")


def test_truncated_images_are_skipped():
    """Garbage / truncated pixels must not kill collate."""
    with pytest.raises(DecodeError):
        load_sample_image({"image_bytes": b"not-an-image", "image_name": "junk.bin"})

    img = _rand_pil(64)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    # LOAD_TRUNCATED_IMAGES=True should still recover a partial JPEG.
    recovered = load_sample_image({
        "image_bytes": raw[: max(80, len(raw) // 2)],
        "image_name": "trunc.jpg",
    })
    assert recovered.mode == "RGB" and recovered.size[0] > 0

    cfg = load_config(overrides=["res=224", "composite.prob=0",
                                 "max_steps=2", "backbone=tiny", "pretrained=false"])
    builder = BatchBuilder(cfg, train=False, patch_grid=14, seed=0)
    samples = [
        {"image": _rand_pil(96, random.Random(0)), "label": 0, "generator": "g"},
        {"image_bytes": b"xxxx", "image_name": "bad.jpg", "label": 1, "generator": "g"},
        {"image": _rand_pil(96, random.Random(1)), "label": 1, "generator": "g"},
    ]
    batch = builder(samples)
    assert batch["images"].shape == (2, 3, 224, 224)
    assert batch["labels"].tolist() == [0.0, 1.0]

    with pytest.raises(SkipBatch):
        builder([{"image_bytes": b"nope", "image_name": "x", "label": 0, "generator": "g"}])
    print("truncated / corrupt image skip OK")


if __name__ == "__main__":
    test_budget_and_forward()
    test_train_step_and_ema()
    test_batch_builder_composites()
    test_composite_combinations()
    test_composite_pairing_balance()
    test_post_stack_shared_pass()
    test_overlay_shapes_not_just_rects()
    test_overlay_crop_keeps_semantics()
    test_augment_pipeline()
    test_perturbations()
    test_fingerprint_mask_ops()
    test_patch_pos_weight()
    test_metrics_and_heatmap()
    test_schedule()
    test_mixture_dataset()
    test_local_parquet_source()
    test_truncated_images_are_skipped()
    print("\nALL SMOKE TESTS PASSED")
