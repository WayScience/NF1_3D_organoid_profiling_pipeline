"""Tests for visualization animation helpers."""

from __future__ import annotations

from image_analysis_3D.visualization_utils import animation_utils  # noqa: F401


def test_animation_utils_imports() -> None:
    assert hasattr(animation_utils, "mp4_to_gif")
