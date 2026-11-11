"""CPU-only face verification with RetinaFace and ArcFace."""

from face_verify.aggregate import cosine_similarity
from face_verify.verify import FaceVerifier, VerificationResult

__all__ = ["FaceVerifier", "VerificationResult", "cosine_similarity"]
__version__ = "0.1.0"
