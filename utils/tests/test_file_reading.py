"""Tests for file reading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from file_utils.file_reading import (
    find_files_available,
    read_in_channels,
    read_zstack_image,
)


def test_find_files_available(tmp_path: Path) -> None:
    (tmp_path / "a.tif").write_text("x")
    (tmp_path / "b.tiff").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    files = find_files_available(tmp_path)
    assert len(files) == 2


def test_read_zstack_image_roundtrip(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    img = np.zeros((2, 3, 3), dtype=np.uint16)
    file_path = tmp_path / "img_405.tif"
    tifffile.imwrite(file_path, img)
    loaded = read_zstack_image(str(file_path))
    assert loaded.shape == img.shape


def test_read_in_channels(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    img = np.zeros((2, 3, 3), dtype=np.uint16)
    file_path = tmp_path / "sample_405.tif"
    tifffile.imwrite(file_path, img)
    files = [str(file_path)]
    loaded = read_in_channels(files, channel_dict={"nuclei": "405"})
    assert loaded["nuclei"] is not None
