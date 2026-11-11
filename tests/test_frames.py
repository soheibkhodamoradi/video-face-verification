from pathlib import Path

import pytest

from face_verify.frames import InputError, iter_sampled_frames


def test_sampling_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(iter_sampled_frames("unused.mp4", every=0))


def test_missing_video_has_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(InputError, match="does not exist"):
        list(iter_sampled_frames(missing))
