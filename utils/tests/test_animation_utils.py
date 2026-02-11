"""Tests for visualization animation helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("moviepy")
pytest.importorskip("napari_animation")
from visualization_utils import animation_utils  # noqa: F401


def test_animation_utils_imports() -> None:
    assert hasattr(animation_utils, "mp4_to_gif")
