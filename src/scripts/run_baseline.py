"""Day-1 baseline: frozen DINOv3 + logistic probe, scored on the robustness grid.

Deliberately the simplest thing that produces a real number, so that every
later change is measured against it. Trains on NTIRE shard 0 and evaluates on
the held-out validation split under the six mandated degradation families.

    python scripts/run_baseline.py --train-n 4000 --eval-n 800
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from aigcdet.backbone import BackboneConfig, FrozenBackbone
from aigcdet.data import labels_of, load_train_shard, load_val, stratified_subset
from aigcdet.degradations import MANDATED_GRID
from aigcdet.evaluate import evaluate_grid, results_to_dicts, summarise, to_markdown
from aigcdet.paths import RESULTS_ROOT as OUT_DIR


def embed_samples(backbone: FrozenBackbone, samples, chunk: int = 256) -> np.ndarray:
    vectors = []
    for start in range(0, len(samples), chunk):
        batch = [s.load() for s in samples[start : start + chunk]]
        vectors.append(backbone.embed(batch))
        done = min(start + chunk, len(samples))
        print(f"  embedded {done}/{len(samples)}", end="\r", flush=True)
    print()
    return np.concatenate(vectors, axis=0)


def fit_logistic(features: np.ndarray, labels: np.ndarray, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    model.fit(scaler.transform(features), labels)
    return scaler, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-n", type=int, default=4000)
    parser.add_argument("--eval-n", type=int, default=800)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model", type=str, default=BackboneConfig.model_name)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading NTIRE shard 0 ...", flush=True)
    train_samples = stratified_subset(load_train_shard(0), args.train_n, rng)

    print("loading NTIRE validation (clean half only, held out) ...", flush=True)
    eval_samples = stratified_subset(load_val(), args.eval_n, rng, clean_only=True)
    print(f"  train={len(train_samples)}  eval={len(eval_samples)}")

    backbone = FrozenBackbone(BackboneConfig(model_name=args.model, image_size=args.image_size))
    print(f"backbone {args.model}: {backbone.num_parameters / 1e6:.1f}M params (cap is 2000M)")

    started = time.time()
    print("extracting train features ...", flush=True)
    train_features = embed_samples(backbone, train_samples)
    print(f"  {len(train_samples) / (time.time() - started):.1f} img/s")

    scaler, probe = fit_logistic(train_features, labels_of(train_samples), args.seed)

    def score_fn(images):
        return probe.predict_proba(scaler.transform(backbone.embed(images)))[:, 1]

    print(f"evaluating over {len(MANDATED_GRID)} degradations ...", flush=True)
    images = [s.load() for s in eval_samples]
    results = evaluate_grid(images, labels_of(eval_samples), score_fn, MANDATED_GRID)

    table = to_markdown(results)
    stats = summarise(results)
    print("\n" + table + "\n")
    print(json.dumps(stats, indent=2))

    (OUT_DIR / "baseline_robustness.md").write_text(table, encoding="utf-8")
    (OUT_DIR / "baseline_robustness.json").write_text(
        json.dumps({"summary": stats, "config": vars(args), "results": results_to_dicts(results)}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_DIR / 'baseline_robustness.md'}")


if __name__ == "__main__":
    main()
