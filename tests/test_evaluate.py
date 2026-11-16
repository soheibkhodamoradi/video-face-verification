import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import evaluate
from evaluate import build_parser, compute_metrics, parse_label
from face_verify.detector import DetectedFace


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


def test_parser_accepts_quality_options() -> None:
    defaults = build_parser().parse_args(["pairs.csv"])
    assert defaults.min_face_size == 80
    assert defaults.min_detection_score == pytest.approx(0.5)
    assert defaults.min_blur_score == pytest.approx(50.0)

    custom = build_parser().parse_args(
        [
            "pairs.csv",
            "--min-face-size",
            "64",
            "--min-detection-score",
            "0.65",
            "--min-blur-score",
            "10",
        ]
    )
    assert custom.min_face_size == 64
    assert custom.min_detection_score == pytest.approx(0.65)
    assert custom.min_blur_score == pytest.approx(10.0)


def test_main_passes_quality_options_through_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class StubDetector:
        def __init__(
            self,
            model_name: str = "buffalo_l",
            min_face_size: int = 80,
            min_detection_score: float = 0.5,
            min_blur_score: float = 50.0,
            det_size: tuple[int, int] = (640, 640),
        ) -> None:
            captured["model_name"] = model_name
            captured["min_face_size"] = min_face_size
            captured["min_detection_score"] = min_detection_score
            captured["min_blur_score"] = min_blur_score
            captured["det_size"] = det_size

        def detect(self, image: np.ndarray, frame_index: int = 0) -> DetectedFace:
            return DetectedFace(
                frame_index=frame_index,
                bbox=np.array([0.0, 0.0, 64.0, 64.0], dtype=np.float32),
                landmarks=np.zeros((5, 2), dtype=np.float32),
                detection_score=0.95,
                blur_score=15.0,
            )

    class StubEmbedder:
        def __init__(self, model_name: str = "buffalo_l") -> None:
            captured["embedder_model_name"] = model_name

        def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
            if float(image.mean()) > 128.0:
                return np.array([1.0, 0.0], dtype=np.float32)
            return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr("face_verify.verify.FaceDetector", StubDetector)
    monkeypatch.setattr("face_verify.verify.FaceEmbedder", StubEmbedder)

    img_a1 = tmp_path / "a1.png"
    img_a2 = tmp_path / "a2.png"
    img_b1 = tmp_path / "b1.png"
    cv2.imwrite(str(img_a1), np.full((64, 64, 3), 200, dtype=np.uint8))
    cv2.imwrite(str(img_a2), np.full((64, 64, 3), 210, dtype=np.uint8))
    cv2.imwrite(str(img_b1), np.full((64, 64, 3), 40, dtype=np.uint8))

    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "path1,path2,label\na1.png,a2.png,1\na1.png,b1.png,0\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"

    exit_code = evaluate.main(
        [
            str(csv_path),
            "--strategy",
            "mean",
            "--threshold",
            "0.4",
            "--every",
            "3",
            "--min-face-size",
            "60",
            "--min-detection-score",
            "0.65",
            "--min-blur-score",
            "10",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["min_face_size"] == 60
    assert captured["min_detection_score"] == pytest.approx(0.65)
    assert captured["min_blur_score"] == pytest.approx(10.0)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["accuracy"] == pytest.approx(1.0)
