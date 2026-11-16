"""CPU-only face verification with RetinaFace and ArcFace."""

import os

# Disable albumentations' online version check (imported transitively by
# insightface) so offline runs do not emit network warnings.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from face_verify.aggregate import cosine_similarity
from face_verify.verify import FaceVerifier, VerificationResult

__all__ = ["FaceVerifier", "VerificationResult", "cosine_similarity"]
__version__ = "0.1.0"
