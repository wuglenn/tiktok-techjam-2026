"""Training loop for Seer.

Continuation training: the whole self-supervised backbone is fine-tuned
jointly with the detection heads (Pangram Image's key recipe choice - a
frozen probe leaves several points of accuracy on the table).
"""

import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augment import eval_transform  # noqa: F401 (kept for external users)
from .config import TrainConfig, load_config
from .data import (
    BatchBuilder,
    Prefetcher,
    build_train_dataset,
    build_val_dataset,
    count_parquet_rows,
    parquet_files,
)
from .model import SeerDetector, EMA, build_param_groups, detection_loss, save_checkpoint

# Approximate number of rows in streaming datasets (for epochs -> steps).
KNOWN_SIZES = {
    "OwensLab/CommunityForensics-Small": 356_216,
    "OwensLab/CommunityForensics-Eval": 51_836,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def cosine_schedule(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, max(0.0, t))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def _source_size(spec) -> Optional[int]:
    if spec.type == "comfor":
        if spec.local_dirs:
            files = parquet_files(spec.local_dirs)
            if files:
                n = count_parquet_rows(files)
                if n > 0:  # exact count from parquet footers
                    return min(n, spec.max_samples) if spec.max_samples else n
        n = KNOWN_SIZES.get(spec.dataset)
        if n is None:
            return None
        return min(n, spec.max_samples) if spec.max_samples else n
    if spec.type == "folders":
        from .data import FolderPairStream

        try:
            fs = FolderPairStream(spec.real_dirs, spec.fake_dirs)
            return len(fs.real_files) + len(fs.fake_files)
        except FileNotFoundError:
            return None
    return None  # generic HF: unknown


def _steps_per_epoch(cfg: TrainConfig) -> Optional[int]:
    eff = cfg.batch_size * cfg.grad_accum
    if cfg.data.source == "mixture" and cfg.data.sources:
        total = 0
        for spec in cfg.data.sources:
            n = _source_size(spec)
            if n is None:
                return None
            total += n
        return max(1, total // eff)
    if cfg.data.source == "comfor":
        if cfg.data.local_dirs:
            files = parquet_files(cfg.data.local_dirs)
            n = count_parquet_rows(files) if files else None
            if not n:
                n = KNOWN_SIZES.get(cfg.data.dataset)
        else:
            n = KNOWN_SIZES.get(cfg.data.dataset)
        if n is None:
            return None
        if cfg.data.max_samples:
            n = min(n, cfg.data.max_samples)
        return max(1, n // eff)
    if cfg.data.source == "folders":
        n = len(build_train_dataset(cfg))  # cheap: just file listing
        return max(1, n // eff)
    return None


@torch.no_grad()
def quick_val(model: SeerDetector, cfg: TrainConfig, device) -> dict:
    """Balanced accuracy on a held-out streamed slice (train-distribution).
    Bounded by max_batches as a safety net - some val datasets are infinite
    iterators (folder cycles)."""
    val_ds = build_val_dataset(cfg)
    builder = BatchBuilder(cfg, train=False, patch_grid=model.patch_grid(cfg.res), seed=1234)
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=builder, num_workers=0)
    max_batches = max(2, math.ceil(cfg.data.val_max_samples / max(1, cfg.batch_size))) + 4
    probs, labels = [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        images = batch["images"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and cfg.bf16)):
            out = model(images)
        probs.extend(torch.sigmoid(out["logits"]).float().cpu().tolist())
        labels.extend(batch["labels"].tolist())
    if not probs or len(set(labels)) < 2:
        return {"val_balanced_acc": float("nan")}
    probs, labels = np.array(probs), np.array(labels)
    pred = probs >= 0.5
    tpr = float(((pred == 1) & (labels == 1)).sum() / max(1, (labels == 1).sum()))
    tnr = float(((pred == 0) & (labels == 0)).sum() / max(1, (labels == 0).sum()))
    return {"val_balanced_acc": 0.5 * (tpr + tnr), "val_tpr": tpr, "val_tnr": tnr}


def run(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------- model
    probe_layers = cfg.probe.layers if cfg.probe.enabled else None
    model = SeerDetector(cfg.backbone, pretrained=cfg.pretrained, probe_layers=probe_layers)
    if cfg.probe.enabled:
        # a linear probe never updates the backbone; no activation grads
        # through it either, so gradient checkpointing is pointless too
        model.freeze_backbone()
        head_params = sum(p.numel() for p in model.probe_head.parameters())
        print(
            f"[seer] page-level linear probe: backbone frozen, taps at blocks "
            f"{model.probe_layers} -> {len(model.probe_layers) * 2 * model.hidden_size}-d "
            f"features, {head_params:,} head params"
        )
    elif cfg.grad_checkpointing:
        model.enable_gradient_checkpointing()
    if cfg.freeze_backbone:
        model.freeze_backbone()
    model.to(device)
    n_params = model.parameter_count()
    print(f"[seer] backbone={cfg.backbone} res={cfg.res}")
    print(f"[seer] model has {model.budget_report()}")
    assert n_params < 2_000_000_000, "detector exceeds the 2B parameter budget"

    # ------------------------------------------------------------- data
    train_ds = build_train_dataset(cfg)
    collate = BatchBuilder(cfg, train=True, patch_grid=model.patch_grid(cfg.res), seed=cfg.seed)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        num_workers=0,  # streaming datasets decode in-process
        drop_last=True,
    )

    spe = _steps_per_epoch(cfg)
    total_steps = cfg.max_steps or (
        int(math.ceil(cfg.epochs * spe)) if spe else None
    )
    if total_steps is None:
        raise ValueError(
            "Streaming dataset with unknown size: set `max_steps` (or `--set max_steps=...`)."
        )
    total_steps = max(1, total_steps)

    # ------------------------------------------------------- optimization
    groups = build_param_groups(model, cfg.lr, cfg.head_lr, cfg.llrd, cfg.weight_decay)
    trainable = [p for g in groups for p in g["params"]]
    print(f"[seer] trainable tensors: {len(trainable)}")
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_schedule(s, total_steps, cfg.warmup_steps, cfg.min_lr_ratio),
    )
    ema = EMA(model, cfg.ema_decay) if 0.0 < cfg.ema_decay < 1.0 else None

    start_step = 0
    best_metric = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("ema") and ema is not None:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        start_step = ckpt.get("step", 0)
        best_metric = ckpt.get("metrics", {}).get("val_balanced_acc", -1.0)
        print(f"[seer] resumed from {cfg.resume} at step {start_step}")

    # ------------------------------------------------------------- loop
    # CUDA: decode+H2D copy overlap with compute via a background thread.
    if device.type == "cuda":
        data_iter = iter(Prefetcher(infinite(loader), device, depth=2))
    else:
        data_iter = infinite(loader)
    ema_used = ema is not None
    t0 = time.time()
    running = {"loss": 0.0, "acc": 0.0, "n": 0}

    print(f"[seer] training for {total_steps} steps "
          f"(effective batch {cfg.batch_size * cfg.grad_accum}) "
          f"-> {cfg.out_dir}")

    for step in range(start_step, total_steps):
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        accum_stats = {"loss_global": 0.0, "loss_patch": 0.0}
        for _ in range(cfg.grad_accum):
            batch = next(data_iter)
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device)
            patch_labels = batch["patch_labels"].to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda" and cfg.bf16)):
                out = model(images)
                loss, stats = detection_loss(
                    out["logits"], out["patch_logits"], labels, patch_labels,
                    patch_weight=cfg.composite.patch_loss_weight,
                )
            (loss / cfg.grad_accum).backward()
            accum_loss += loss.item() / cfg.grad_accum
            accum_stats["loss_global"] += stats["loss_global"] / cfg.grad_accum
            accum_stats["loss_patch"] += stats["loss_patch"] / cfg.grad_accum

            with torch.no_grad():
                acc = ((out["logits"] >= 0) == labels.bool()).float().mean().item()
            running["loss"] += accum_loss
            running["acc"] += acc
            running["n"] += 1

        torch.nn.utils.clip_grad_norm_(trainable, cfg.clip_grad)
        optimizer.step()
        scheduler.step()
        if ema_used:
            ema.update(model)

        # --------------------------------------------------------- logging
        if (step + 1) % cfg.log_every == 0:
            n = max(1, running["n"])
            sps = (step + 1 - start_step) / max(1e-9, time.time() - t0)
            print(
                f"[step {step + 1}/{total_steps}] "
                f"loss={running['loss'] / n:.4f} "
                f"acc={running['acc'] / n:.4f} "
                f"(g={accum_stats['loss_global']:.3f} p={accum_stats['loss_patch']:.3f}) "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"{sps:.2f} it/s"
            )
            running = {"loss": 0.0, "acc": 0.0, "n": 0}

        # ------------------------------------------------------- evaluation
        if (step + 1) % cfg.eval_every == 0 or (step + 1) == total_steps:
            eval_model = model
            if ema_used:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.shadow)
                eval_model = model
            metrics = quick_val(model, cfg, device)
            if ema_used:
                model.load_state_dict(backup)
            metrics.update({"step": step + 1})
            print(f"[eval] {metrics}")

            if metrics.get("val_balanced_acc", -1) > best_metric or math.isnan(
                metrics.get("val_balanced_acc", float("nan"))
            ):
                best_metric = metrics.get("val_balanced_acc", -1)
                save_checkpoint(
                    os.path.join(cfg.out_dir, "best.pt"),
                    model, cfg, step + 1, optimizer, scheduler, ema, metrics,
                )
            save_checkpoint(
                os.path.join(cfg.out_dir, "last.pt"),
                model, cfg, step + 1, optimizer, scheduler, ema, metrics,
            )

    save_checkpoint(
        os.path.join(cfg.out_dir, "last.pt"),
        model, cfg, total_steps, optimizer, scheduler, ema,
        {"val_balanced_acc": best_metric},
    )
    print(f"[seer] done. best val_balanced_acc={best_metric:.4f} -> {cfg.out_dir}")
    return model


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Train Seer")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="dotted config overrides, e.g. max_steps=10 res=224")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    run(cfg)


if __name__ == "__main__":
    main()
