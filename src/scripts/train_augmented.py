"""Degradation-augmented probe vs the clean-trained baseline.

Tests the single highest-leverage hypothesis in the research: that augmentation
severity, not architecture, is what buys robustness. In the NTIRE 2026 results
a plain linear head with aggressive augmentation beat a far more elaborate
DINOv3 setup that used only default augmentation.

Backbone stays frozen, so the only variable is the training distribution.
Training views are random compound chains; evaluation is the mandated grid, so
we are never training on the severities we report.

    python scripts/train_augmented.py --train-n 4000 --views 6 --eval-n 800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aigcdet.backbone import BackboneConfig, FrozenBackbone
from aigcdet.data import labels_of, load_train_shard, load_val, stratified_subset
from aigcdet.degradations import MANDATED_GRID, training_views
from aigcdet.evaluate import evaluate_grid, results_to_dicts, summarise, to_markdown
from aigcdet.extract import extract_views
from aigcdet.paths import RESULTS_ROOT as OUT_DIR


def fit(features: np.ndarray, labels: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    model = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    model.fit(scaler.transform(features), labels)
    return scaler, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=4000)
    parser.add_argument("--eval-n", type=int, default=800)
    parser.add_argument("--views", type=int, default=6, help="augmented views per training image")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_samples = stratified_subset(load_train_shard(0), args.train_n, rng)
    eval_samples = stratified_subset(load_val(), args.eval_n, rng, clean_only=True)
    y_train = labels_of(train_samples)
    y_eval = labels_of(eval_samples)

    backbone = FrozenBackbone(BackboneConfig(image_size=args.image_size))
    print(f"backbone: {backbone.num_parameters / 1e6:.1f}M params")

    views = [None, *training_views(args.views)]
    print(f"extracting {len(views)} views x {len(train_samples)} train images (cached after first run)")
    train_features = extract_views(backbone, train_samples, views, workers=16)

    def score_with(scaler, model):
        def score_fn(images):
            return model.predict_proba(scaler.transform(backbone.embed(images)))[:, 1]

        return score_fn

    report: dict[str, dict] = {}
    tables: dict[str, str] = {}

    for label, keys in (
        ("clean_trained", ["clean"]),
        ("augmented", list(train_features)),
    ):
        features = np.concatenate([train_features[k] for k in keys], axis=0)
        labels = np.tile(y_train, len(keys))
        print(f"\n=== {label}: {features.shape[0]} training rows ===")
        scaler, model = fit(features, labels, args.seed)

        results = evaluate_grid(
            [s.load() for s in eval_samples], y_eval, score_with(scaler, model), MANDATED_GRID
        )
        tables[label] = to_markdown(results)
        report[label] = {"summary": summarise(results), "results": results_to_dicts(results)}
        print(json.dumps(report[label]["summary"], indent=2))

    base = report["clean_trained"]["summary"]
    aug = report["augmented"]["summary"]
    print("\n--- clean-trained vs augmented ---")
    for key in ("clean_auc", "mean_degraded_auc", "worst_degraded_auc", "clean_minus_worst", "max_abs_score_shift"):
        print(f"{key:>22}: {base[key]:.4f} -> {aug[key]:.4f}   ({aug[key] - base[key]:+.4f})")

    (OUT_DIR / "augmented_robustness.md").write_text(
        "## Clean-trained\n\n" + tables["clean_trained"] + "\n\n## Degradation-augmented\n\n" + tables["augmented"],
        encoding="utf-8",
    )
    (OUT_DIR / "augmented_robustness.json").write_text(
        json.dumps({"config": vars(args), **report}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'augmented_robustness.md'}")


if __name__ == "__main__":
    main()
