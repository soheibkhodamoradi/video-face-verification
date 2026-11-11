"""Embedding normalization, aggregation, and similarity functions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Embedding = np.ndarray


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a float32 unit vector, rejecting zero-length vectors."""

    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot L2-normalize a zero-length or non-finite vector.")
    return array / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compute cosine similarity without assuming inputs are normalized."""

    return float(np.dot(l2_normalize(first), l2_normalize(second)))


def as_embedding_matrix(embeddings: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Validate embeddings and L2-normalize every row."""

    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
        rows = embeddings
    else:
        if len(embeddings) == 0:
            raise ValueError("At least one embedding is required.")
        rows = np.stack(embeddings)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("Embeddings must form a non-empty 2D matrix.")
    return np.stack([l2_normalize(row) for row in rows])


def mean_embedding(embeddings: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Average normalized embeddings, then normalize the mean again."""

    matrix = as_embedding_matrix(embeddings)
    return l2_normalize(matrix.mean(axis=0))


def aggregate_similarity(
    first: Sequence[np.ndarray] | np.ndarray,
    second: Sequence[np.ndarray] | np.ndarray,
    strategy: str = "mean",
) -> float:
    """Compare two embedding sets using one of the supported strategies."""

    if strategy == "mean":
        return cosine_similarity(mean_embedding(first), mean_embedding(second))
    if strategy == "pairwise":
        first_matrix = as_embedding_matrix(first)
        second_matrix = as_embedding_matrix(second)
        # Every frame from video 1 is compared with every frame from video 2.
        return float((first_matrix @ second_matrix.T).mean())
    raise ValueError(f"Unknown aggregation strategy: {strategy!r}. Use 'mean' or 'pairwise'.")
