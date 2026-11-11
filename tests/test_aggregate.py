import numpy as np
import pytest

from face_verify.aggregate import (
    aggregate_similarity,
    cosine_similarity,
    l2_normalize,
    mean_embedding,
)


def test_cosine_similarity_normalizes_inputs() -> None:
    assert cosine_similarity(np.array([3.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 4.0])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-2.0, 0.0])) == pytest.approx(-1.0)


def test_mean_embedding_is_unit_length() -> None:
    result = mean_embedding([np.array([2.0, 0.0]), np.array([0.0, 3.0])])
    np.testing.assert_allclose(result, np.array([1.0, 1.0]) / np.sqrt(2.0))
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_pairwise_strategy_averages_all_cross_video_pairs() -> None:
    first = np.array([[1.0, 0.0], [0.0, 1.0]])
    second = np.array([[1.0, 0.0], [1.0, 0.0]])
    # Pairwise similarities are 1, 1, 0, 0.
    assert aggregate_similarity(first, second, "pairwise") == pytest.approx(0.5)


def test_empty_embeddings_and_zero_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        mean_embedding([])
    with pytest.raises(ValueError, match="zero-length"):
        l2_normalize(np.zeros(3))


def test_unknown_strategy_has_clear_error() -> None:
    embeddings = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_similarity(embeddings, embeddings, "median")
