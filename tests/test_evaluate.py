import numpy as np
import pytest

from evaluate import compute_metrics, parse_label


def test_metrics_for_perfect_separation() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    metrics, _, _, _ = compute_metrics(labels, scores, threshold=0.5)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.auc == pytest.approx(1.0)
    assert metrics.eer == pytest.approx(0.0)
    assert 0.2 < metrics.eer_threshold <= 0.8
    assert metrics.far == pytest.approx(0.0)
    assert metrics.frr == pytest.approx(0.0)


def test_far_and_frr_at_operating_threshold() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.7, 0.2, 0.8, 0.1])
    metrics, _, _, _ = compute_metrics(labels, scores, threshold=0.5)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.far == pytest.approx(0.5)
    assert metrics.frr == pytest.approx(0.5)


def test_human_readable_labels() -> None:
    assert parse_label("same") == 1
    assert parse_label("0") == 0
    with pytest.raises(ValueError, match="Invalid label"):
        parse_label("unknown")
