import numpy as np

from face_verify.verify import compare_embedding_sets


def test_threshold_decision_is_inclusive() -> None:
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    score, same = compare_embedding_sets(embeddings, embeddings, threshold=1.0)
    assert score == 1.0
    assert same is True


def test_threshold_can_reject_different_people() -> None:
    first = np.array([[1.0, 0.0]], dtype=np.float32)
    second = np.array([[0.0, 1.0]], dtype=np.float32)
    score, same = compare_embedding_sets(first, second, threshold=0.4)
    assert score == 0.0
    assert same is False
