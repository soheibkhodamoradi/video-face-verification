"""RetinaFace detection and frame-quality filtering."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def variance_of_laplacian(image: np.ndarray) -> float:
    """Return a blur score; smaller values generally mean a blurrier image."""

    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(frozen=True)
class DetectedFace:
    """The information needed to align and embed one accepted face."""

    frame_index: int
    bbox: np.ndarray
    landmarks: np.ndarray
    detection_score: float
    blur_score: float

    @property
    def area(self) -> float:
        width = max(0.0, float(self.bbox[2] - self.bbox[0]))
        height = max(0.0, float(self.bbox[3] - self.bbox[1]))
        return width * height


class FaceDetector:
    """Find the largest face and reject low-quality detections."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        min_face_size: int = 80,
        min_detection_score: float = 0.5,
        min_blur_score: float = 50.0,
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        if min_face_size < 1:
            raise ValueError("min_face_size must be at least 1 pixel.")
        if not 0.0 <= min_detection_score <= 1.0:
            raise ValueError("min_detection_score must be between 0 and 1.")
        if min_blur_score < 0.0:
            raise ValueError("min_blur_score cannot be negative.")

        self.min_face_size = min_face_size
        self.min_detection_score = min_detection_score
        self.min_blur_score = min_blur_score

        # Import lazily so geometry/quality unit tests do not load ONNX models.
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(
            ctx_id=-1,
            det_thresh=min_detection_score,
            det_size=det_size,
        )

    def detect(self, image: np.ndarray, frame_index: int = 0) -> DetectedFace | None:
        """Return the largest acceptable face, or ``None`` if it is unusable."""

        if image.size == 0:
            return None
        faces = self._app.get(image)
        if not faces:
            return None

        # Only the largest face is considered when several people are visible.
        face = max(faces, key=lambda item: _box_area(item.bbox))
        bbox = np.asarray(face.bbox, dtype=np.float32)
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        score = float(face.det_score)
        if min(width, height) < self.min_face_size or score < self.min_detection_score:
            return None

        face_crop = _clipped_crop(image, bbox)
        blur_score = variance_of_laplacian(face_crop)
        if blur_score < self.min_blur_score:
            return None

        return DetectedFace(
            frame_index=frame_index,
            bbox=bbox,
            landmarks=np.asarray(face.kps, dtype=np.float32),
            detection_score=score,
            blur_score=blur_score,
        )


def _box_area(bbox: np.ndarray) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(
        0.0, float(bbox[3] - bbox[1])
    )


def _clipped_crop(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop a floating-point detector box safely at image boundaries."""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    return image[top:bottom, left:right]
