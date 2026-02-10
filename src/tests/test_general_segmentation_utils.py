"""Tests for general segmentation helpers."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")
from segmentation_utils.general_segmentation_utils import (
    calculate_bbox_area,
    calculate_mask_iou,
    calculate_overlap,
    clean_border_objects,
    create_cytoplasm_masks,
)


def test_calculate_bbox_area() -> None:
    assert calculate_bbox_area((0, 0, 2, 3)) == 6


def test_calculate_overlap() -> None:
    overlap = calculate_overlap((0, 0, 3, 3), (1, 1, 2, 2))
    assert overlap == 1.0


def test_calculate_mask_iou() -> None:
    mask1 = np.array([[1, 0], [0, 1]], dtype=bool)
    mask2 = np.array([[1, 0], [1, 0]], dtype=bool)
    assert calculate_mask_iou(mask1, mask2) is False


def test_create_cytoplasm_masks() -> None:
    nuclei = np.zeros((1, 2, 2), dtype=np.uint8)
    cells = np.zeros((1, 2, 2), dtype=np.uint8)
    nuclei[0, 0, 0] = 1
    cells[0, :, :] = 1
    cyto = create_cytoplasm_masks(nuclei, cells)
    assert cyto[0, 0, 0] == 0
    assert cyto[0, 1, 1] == 1


def test_clean_border_objects() -> None:
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = 2
    mask[0, 2, 2] = 3
    cleaned = clean_border_objects(mask, border_width=1)
    assert cleaned[0, 0, 0] == 0
    assert cleaned[0, 2, 2] == 3
