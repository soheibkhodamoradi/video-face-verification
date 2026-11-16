"""Video and image input helpers.

OpenCV images are BGR arrays. The rest of this package intentionally uses the
same convention because InsightFace also expects BGR input.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class InputError(ValueError):
    """Raised when an image or video cannot be read."""


@dataclass(frozen=True)
class Frame:
    """One sampled video frame and its zero-based position in the video."""

    index: int
    image: np.ndarray


def iter_sampled_frames(video_path: str | Path, every: int = 5) -> Iterator[Frame]:
    """Yield frames 0, N, 2N, ... from a video.

    The capture is released even when processing raises an exception. A clear
    error is raised for missing files, unsupported containers, and empty videos.
    """

    if every < 1:
        raise ValueError("Frame sampling interval 'every' must be at least 1.")

    path = Path(video_path)
    if not path.is_file():
        raise InputError(f"Video file does not exist or is not a file: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise InputError(
            f"Could not open video: {path}. Check that it is a supported video file."
        )

    index = 0
    frames_read = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frames_read += 1
            if index % every == 0:
                yield Frame(index=index, image=image)
            index += 1
    finally:
        capture.release()

    if frames_read == 0:
        raise InputError(f"No readable frames found in video: {path}")


def read_image(image_path: str | Path) -> np.ndarray:
    """Read one image as a BGR array, with a useful error on failure."""

    path = Path(image_path)
    if not path.is_file():
        raise InputError(f"Image file does not exist or is not a file: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise InputError(f"Could not decode image: {path}")
    return image
