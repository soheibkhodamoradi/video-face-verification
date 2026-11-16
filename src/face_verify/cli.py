"""Command-line interface for comparing two videos."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from face_verify.verify import FaceVerifier


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the verification and evaluation commands."""

    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help="sample every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="minimum box width/height in pixels (default: 80)",
    )
    parser.add_argument(
        "--min-detection-score",
        type=float,
        default=0.5,
        help="minimum detector confidence (default: 0.5)",
    )
    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=50.0,
        help="minimum variance of Laplacian (default: 50.0)",
    )
    parser.add_argument(
        "--model",
        default="buffalo_l",
        help="InsightFace model pack (default: buffalo_l)",
    )


def verifier_from_args(args: argparse.Namespace) -> FaceVerifier:
    return FaceVerifier(
        every=args.every,
        model_name=args.model,
        min_face_size=args.min_face_size,
        min_detection_score=args.min_detection_score,
        min_blur_score=args.min_blur_score,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide whether two videos show the same person (CPU only)."
    )
    parser.add_argument("video1", help="path to the first video")
    parser.add_argument("video2", help="path to the second video")
    parser.add_argument("--strategy", choices=("mean", "pairwise"), default="mean")
    parser.add_argument("--threshold", type=float, default=0.4)
    add_pipeline_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        verifier = verifier_from_args(args)
        result = verifier.verify(
            args.video1,
            args.video2,
            strategy=args.strategy,
            threshold=args.threshold,
        )
    except (ValueError, OSError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(f"Similarity: {result.similarity:.6f}")
    print(f"Decision: {result.decision} (threshold {result.threshold:.3f})")
    print(f"Usable frames: video1={result.usable_frames_video1}, video2={result.usable_frames_video2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
