import numpy as np

from face_verify.detector import variance_of_laplacian


def test_blur_filter_scores_texture_above_flat_image() -> None:
    flat = np.full((64, 64, 3), 127, dtype=np.uint8)
    checkerboard = np.indices((64, 64)).sum(axis=0) % 2
    textured = (checkerboard * 255).astype(np.uint8)
    assert variance_of_laplacian(flat) == 0.0
    assert variance_of_laplacian(textured) > 1000.0


def test_blur_score_handles_empty_crop() -> None:
    assert variance_of_laplacian(np.empty((0, 0, 3), dtype=np.uint8)) == 0.0
