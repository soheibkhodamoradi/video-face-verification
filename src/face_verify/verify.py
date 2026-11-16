"""End-to-end face verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from face_verify.aggregate import aggregate_similarity
from face_verify.detector import FaceDetector
from face_verify.embedder import FaceEmbedder
from face_verify.frames import iter_sampled_frames, read_image


class NoUsableFaceError(ValueError):
    """Raised when no frame in an input passes all face-quality checks."""


@dataclass(frozen=True)
class VerificationResult:
    similarity: float
    is_same_person: bool
    usable_frames_video1: int
    usable_frames_video2: int
    threshold: float
    strategy: str

    @property
    def decision(self) -> str:
        return "same person" if self.is_same_person else "different person"


def compare_embedding_sets(
    first: np.ndarray,
    second: np.ndarray,
    strategy: str = "mean",
    threshold: float = 0.4,
) -> tuple[float, bool]:
    """Pure comparison step, separate from model inference for easy testing."""

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1 and 1.")
    similarity = aggregate_similarity(first, second, strategy)
    return similarity, similarity >= threshold


class FaceVerifier:
    """Run detection, filtering, alignment, embedding, and aggregation."""

    def __init__(
        self,
        every: int = 5,
        model_name: str = "buffalo_l",
        min_face_size: int = 80,
        min_detection_score: float = 0.5,
        min_blur_score: float = 50.0,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        if every < 1:
            raise ValueError("every must be at least 1.")
        self.every = every
        self.detector = FaceDetector(
            model_name=model_name,
            min_face_size=min_face_size,
            min_detection_score=min_detection_score,
            min_blur_score=min_blur_score,
            det_size=det_size,
        )
        self.embedder = FaceEmbedder(model_name=model_name)

    def extract_video_embeddings(self, video_path: str | Path) -> np.ndarray:
        """Extract one normalized embedding per usable sampled frame."""

        embeddings: list[np.ndarray] = []
        for frame in iter_sampled_frames(video_path, every=self.every):
            face = self.detector.detect(frame.image, frame_index=frame.index)
            if face is not None:
                embeddings.append(self.embedder.embed(frame.image, face))
        if not embeddings:
            raise NoUsableFaceError(
                f"No usable face found in video '{video_path}'. Check that one face "
                "is visible and adjust the size, detection, blur, or sampling settings."
            )
        return np.stack(embeddings)

    def extract_image_embeddings(self, image_path: str | Path) -> np.ndarray:
        """Use the same pipeline for a still image (a one-frame input)."""

        image = read_image(image_path)
        face = self.detector.detect(image)
        if face is None:
            raise NoUsableFaceError(
                f"No usable face found in image '{image_path}'. Check the face size, "
                "detection score, and blur settings."
            )
        return self.embedder.embed(image, face).reshape(1, -1)

    def verify(
        self,
        video1: str | Path,
        video2: str | Path,
        strategy: str = "mean",
        threshold: float = 0.4,
    ) -> VerificationResult:
        """Compare two videos and return the score, decision, and frame counts."""

        first = self.extract_video_embeddings(video1)
        second = self.extract_video_embeddings(video2)
        similarity, is_same = compare_embedding_sets(first, second, strategy, threshold)
        return VerificationResult(
            similarity=similarity,
            is_same_person=is_same,
            usable_frames_video1=first.shape[0],
            usable_frames_video2=second.shape[0],
            threshold=threshold,
            strategy=strategy,
        )
