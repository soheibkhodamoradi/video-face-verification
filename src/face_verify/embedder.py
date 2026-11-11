"""Face alignment and ArcFace embedding on the CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from face_verify.aggregate import l2_normalize
from face_verify.detector import DetectedFace


class FaceEmbedder:
    """Align a detected face with its five landmarks and embed it with ArcFace."""

    def __init__(self, model_name: str = "buffalo_l") -> None:
        # Load only recognition. FaceAnalysis requires a detector, while this
        # lower-level API lets us avoid a second, unused detector in memory.
        from insightface.model_zoo import model_zoo
        from insightface.utils import ensure_available

        model_dir = Path(ensure_available("models", model_name, root="~/.insightface"))
        self._model = None
        # ArcFace files conventionally sort last (e.g. w600k_r50.onnx), so this
        # normally loads just one ONNX file. The task check keeps it general.
        for model_path in sorted(model_dir.glob("*.onnx"), reverse=True):
            candidate = model_zoo.get_model(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            if candidate is not None and candidate.taskname == "recognition":
                self._model = candidate
                break
        if self._model is None:
            raise RuntimeError(f"No ArcFace recognition model found in {model_dir}")
        self._model.prepare(ctx_id=-1)

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return one L2-normalized embedding for an accepted detection."""

        from insightface.utils import face_align

        aligned = face_align.norm_crop(
            image,
            landmark=face.landmarks,
            image_size=self._model.input_size[0],
        )
        embedding = self._model.get_feat(aligned).reshape(-1)
        return l2_normalize(embedding)
