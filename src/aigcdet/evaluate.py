"""Robustness evaluation: clean vs transformed, per degradation.

Reports more than accuracy, deliberately:

* **Worst-case** across degradations, not just the mean. The mean hides the
  operating point that actually breaks in production.
* **Per-class accuracy.** Under degradation, detectors do not raise false
  alarms -- they collapse into predicting "real". Balanced accuracy alone
  conceals this; real and fake accuracy separately do not.
* **Score shift.** A detector can hold its AUC while its scores drift
  systematically toward one class, which silently breaks any fixed threshold.
  This is the BIAS-ID diagnostic and it is invisible in AUC.
* **TPR at fixed low FPR**, because trust-and-safety runs at a fixed false
  positive budget, not at 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from .degradations import Degradation, MANDATED_GRID

ScoreFn = Callable[[Sequence[Image.Image]], np.ndarray]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC via rank statistic; ties averaged. Returns NaN if single-class."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups.
    sorted_scores = scores[order]
    start = 0
    for idx in range(1, len(sorted_scores) + 1):
        if idx == len(sorted_scores) or sorted_scores[idx] != sorted_scores[start]:
            ranks[order[start:idx]] = ranks[order[start:idx]].mean()
            start = idx
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Highest TPR achievable while keeping FPR at or below ``target_fpr``.

    Computed by sweeping candidate thresholds rather than by comparing against
    a quantile. A saturating classifier puts many scores at exactly 1.0, which
    makes the quantile itself 1.0; a strict ``score > threshold`` test then
    reports 0% TPR for a model with 0.91 AUC. That is a metric artifact, not a
    model property, and it is easy to mistake for a real collapse.
    """
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")

    # Candidate thresholds: every distinct score, plus one just below the
    # minimum so that "accept everything" is representable.
    candidates = np.unique(scores)
    candidates = np.concatenate([[np.nextafter(candidates[0], -np.inf)], candidates])

    # A sample is flagged when score >= threshold, so ties resolve toward
    # detection and a fully saturated score remains reachable.
    fpr = (neg[None, :] >= candidates[:, None]).mean(axis=1)
    allowed = fpr <= target_fpr
    if not allowed.any():
        return 0.0
    tpr = (pos[None, :] >= candidates[:, None]).mean(axis=1)
    return float(tpr[allowed].max())


@dataclass
class DegradationResult:
    name: str
    family: str
    auc: float
    accuracy: float
    real_accuracy: float
    fake_accuracy: float
    tpr_at_1pct_fpr: float
    tpr_at_0p1pct_fpr: float
    mean_score: float
    score_shift: float          # mean score minus the clean mean score
    n: int


def _binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float]:
    predictions = scores > threshold
    real_acc = float((~predictions[labels == 0]).mean()) if (labels == 0).any() else float("nan")
    fake_acc = float(predictions[labels == 1].mean()) if (labels == 1).any() else float("nan")
    return float(np.nanmean([real_acc, fake_acc])), real_acc, fake_acc


def evaluate_grid(
    images: Sequence[Image.Image],
    labels: np.ndarray,
    score_fn: ScoreFn,
    grid: Iterable[Degradation] = MANDATED_GRID,
    threshold: float = 0.5,
    seed: int = 0,
) -> list[DegradationResult]:
    """Score ``images`` under every degradation in ``grid``.

    The clean entry must be present in the grid for score shifts to be
    meaningful; ``MANDATED_GRID`` includes it first.
    """
    labels = np.asarray(labels)
    results: list[DegradationResult] = []
    clean_mean: float | None = None

    for degradation in grid:
        rng = np.random.default_rng(seed)
        transformed = [degradation(img, rng) for img in images]
        scores = np.asarray(score_fn(transformed), dtype=np.float64)

        if clean_mean is None:
            clean_mean = float(scores.mean())

        accuracy, real_acc, fake_acc = _binary_metrics(labels, scores, threshold)
        results.append(
            DegradationResult(
                name=degradation.name,
                family=degradation.family,
                auc=roc_auc(labels, scores),
                accuracy=accuracy,
                real_accuracy=real_acc,
                fake_accuracy=fake_acc,
                tpr_at_1pct_fpr=tpr_at_fpr(labels, scores, 0.01),
                tpr_at_0p1pct_fpr=tpr_at_fpr(labels, scores, 0.001),
                mean_score=float(scores.mean()),
                score_shift=float(scores.mean() - clean_mean),
                n=len(labels),
            )
        )
    return results


def summarise(results: Sequence[DegradationResult]) -> dict:
    """Headline numbers. Worst-case is the honest figure for a robustness track."""
    degraded = [r for r in results if r.family != "clean"]
    clean = next((r for r in results if r.family == "clean"), None)
    aucs = [r.auc for r in degraded if not np.isnan(r.auc)]
    worst = min(degraded, key=lambda r: r.auc) if degraded else None
    return {
        "clean_auc": clean.auc if clean else float("nan"),
        "mean_degraded_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "worst_degraded_auc": worst.auc if worst else float("nan"),
        "worst_degradation": worst.name if worst else None,
        "clean_minus_worst": (clean.auc - worst.auc) if (clean and worst) else float("nan"),
        "max_abs_score_shift": float(max((abs(r.score_shift) for r in degraded), default=float("nan"))),
    }


def to_markdown(results: Sequence[DegradationResult]) -> str:
    """Robustness table for the README / submission."""
    header = (
        "| Degradation | AUC | Acc | Real Acc | Fake Acc | TPR@1%FPR | Score shift |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| `{r.name}` | {r.auc:.4f} | {r.accuracy:.3f} | {r.real_accuracy:.3f} | "
        f"{r.fake_accuracy:.3f} | {r.tpr_at_1pct_fpr:.3f} | {r.score_shift:+.4f} |"
        for r in results
    ]
    return header + "\n".join(rows)


def results_to_dicts(results: Sequence[DegradationResult]) -> list[dict]:
    return [asdict(r) for r in results]
