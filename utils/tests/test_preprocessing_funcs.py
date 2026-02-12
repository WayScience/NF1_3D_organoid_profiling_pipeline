"""Tests for preprocessing utility functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tifffile")
from file_utils.preprocessing_funcs import (
    check_well_dir_name_format,
    get_to_the_unested_dir,
    get_well_fov_dirs,
    read_2D_image_for_zstacking,
)


def test_read_2d_image_for_zstacking(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    img = np.zeros((5, 6), dtype=np.uint16)
    file_path = tmp_path / "img.tif"
    tifffile.imwrite(file_path, img)
    loaded = read_2D_image_for_zstacking(str(file_path))
    assert loaded.shape == img.shape


def test_get_well_fov_dirs(tmp_path: Path) -> None:
    (tmp_path / "A01-1").mkdir()
    (tmp_path / "A01-2").mkdir()
    dirs = get_well_fov_dirs(tmp_path)
    assert len(dirs) == 2


def test_get_to_the_unested_dir(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    result = get_to_the_unested_dir(tmp_path, 3)
    assert result == nested


def test_check_well_dir_name_format() -> None:
    assert check_well_dir_name_format("A01-1") is True
    assert check_well_dir_name_format("bad") is False
