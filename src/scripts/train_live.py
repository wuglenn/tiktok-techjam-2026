"""Train with augmentation applied live, inside the training loop.

Differs from ``train_augmented.py`` (which used a fixed set of cached views) in
that every image is degraded differently on every epoch, and the backbone can
be partially unfrozen. Cached views cannot do either.

    # frozen backbone, live augmentation, consistency loss
    python src/scripts/train_live.py --train-n 4000 --epochs 4

    # unfreeze the last 2 blocks (needs more VRAM)
    python src/scripts/train_live.py --train-n 8000 --epochs 4 --unfreeze-last-n 2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from aigcdet.backbone import BackboneConfig, FrozenBackbone
from aigcdet.data import labels_of, load_train_shard, load_val, stratified_subset
from aigcdet.dataset import AIGCDataset, AugmentConfig
from aigcdet.degradations import MANDATED_GRID
from aigcdet.evaluate import evaluate_grid, results_to_dicts, summarise, to_markdown
from aigcdet.paths import RESULTS_ROOT
from aigcdet.train import DetectorHead, TrainConfig, predict, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=4000)
    parser.add_argument("--eval-n", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--consistency", type=float, default=0.25)
    parser.add_argument("--aug-prob", type=float, default=0.9)
    parser.add_argument("--max-ops", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="live")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    train_samples = stratified_subset(load_train_shard(0), args.train_n, rng)
    eval_samples = stratified_subset(load_val(), args.eval_n, rng, clean_only=True)
    print(f"train={len(train_samples)}  eval={len(eval_samples)}")

    backbone = FrozenBackbone(BackboneConfig(image_size=args.image_size))
    trainable = "frozen" if args.unfreeze_last_n == 0 else f"last {args.unfreeze_last_n} blocks"
    print(f"backbone {backbone.num_parameters / 1e6:.1f}M params ({trainable})")

    dataset = AIGCDataset(
        train_samples,
        image_size=args.image_size,
        augment=AugmentConfig(probability=args.aug_prob, max_ops=args.max_ops),
        return_pair=args.consistency > 0,
        mean=tuple(backbone.mean.flatten().tolist()),
        std=tuple(backbone.std.flatten().tolist()),
        seed=args.seed,
    )

    feature_dim = backbone.embed([train_samples[0].load()]).shape[1]
    head = DetectorHead(feature_dim)

    head = train(
        backbone,
        head,
        dataset,
        TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            consistency_weight=args.consistency,
            unfreeze_last_n=args.unfreeze_last_n,
            num_workers=args.workers,
        ),
    )

    print(f"evaluating over {len(MANDATED_GRID)} degradations ...")
    images = [s.load() for s in eval_samples]
    results = evaluate_grid(
        images,
        labels_of(eval_samples),
        lambda batch: predict(backbone, head, batch),
        MANDATED_GRID,
    )

    table = to_markdown(results)
    stats = summarise(results)
    print("\n" + table)
    print("\n" + json.dumps(stats, indent=2))

    (RESULTS_ROOT / f"{args.tag}_robustness.md").write_text(table, encoding="utf-8")
    (RESULTS_ROOT / f"{args.tag}_robustness.json").write_text(
        json.dumps({"config": vars(args), "summary": stats, "results": results_to_dicts(results)}, indent=2),
        encoding="utf-8",
    )
    torch.save({"head": head.state_dict(), "config": vars(args)}, RESULTS_ROOT / f"{args.tag}_head.pt")
    print(f"\nwrote {RESULTS_ROOT / f'{args.tag}_robustness.md'}")


if __name__ == "__main__":
    main()
