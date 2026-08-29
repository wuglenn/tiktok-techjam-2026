"""Metric correctness, including the saturation case that produced a false zero."""

import numpy as np

from aigcdet.evaluate import roc_auc, tpr_at_fpr


def test_roc_auc_perfect_and_inverted():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_roc_auc_all_ties_is_chance():
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_roc_auc_matches_sklearn_on_random_data():
    sklearn_metrics = __import__("sklearn.metrics", fromlist=["roc_auc_score"])
    rng = np.random.default_rng(0)
    for _ in range(5):
        labels = rng.integers(0, 2, 500)
        scores = rng.normal(labels * 0.8, 1.0)
        assert abs(roc_auc(labels, scores) - sklearn_metrics.roc_auc_score(labels, scores)) < 1e-9


def test_tpr_at_fpr_separable():
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    scores = np.concatenate([np.linspace(0.0, 0.4, 100), np.linspace(0.6, 1.0, 100)])
    assert tpr_at_fpr(labels, scores, 0.01) == 1.0


def test_tpr_at_fpr_survives_saturated_scores():
    """The regression this guards against.

    A probe trained on many rows saturates probabilities to exactly 1.0. The
    previous quantile-based implementation returned 0.0 here despite the model
    being perfectly separable on everything that is not saturated.
    """
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    scores = np.concatenate([
        np.concatenate([np.full(99, 1.0), [0.0]]),   # reals: 99 saturated at 1.0
        np.full(100, 1.0),                            # fakes: all saturated
    ])
    value = tpr_at_fpr(labels, scores, 0.01)
    assert not np.isnan(value)
    assert 0.0 <= value <= 1.0


def test_tpr_at_fpr_respects_the_budget():
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.zeros(1000), np.ones(1000)])
    scores = np.concatenate([rng.normal(0.0, 1.0, 1000), rng.normal(2.0, 1.0, 1000)])
    strict, loose = tpr_at_fpr(labels, scores, 0.001), tpr_at_fpr(labels, scores, 0.1)
    assert strict <= loose


def test_tpr_at_fpr_single_class_is_nan():
    assert np.isnan(tpr_at_fpr(np.zeros(10), np.random.rand(10), 0.01))
