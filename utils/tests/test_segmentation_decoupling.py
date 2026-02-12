"""Tests for segmentation decoupling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")
from file_utils.segmentation_decoupling import (
    check_coordinate_inside_box,
    euclidian_2D_distance,
    extract_unique_masks,
    get_combinations_of_indices,
    get_dimensionality,
    get_larger_bbox,
    get_number_of_unique_labels,
    merge_sets,
    merge_sets_df,
    reassemble_each_mask,
)


def test_euclidian_2d_distance() -> None:
    assert euclidian_2D_distance((0.0, 0.0), (3.0, 4.0)) == 5.0


def test_check_coordinate_inside_box_true() -> None:
    assert check_coordinate_inside_box((2.0, 2.0), (1.0, 1.0, 3.0, 3.0)) is True


def test_check_coordinate_inside_box_false() -> None:
    assert check_coordinate_inside_box((0.0, 0.0), (1.0, 1.0, 3.0, 3.0)) is False


def test_get_larger_bbox() -> None:
    bbox1 = (0.0, 0.0, 4.0, 4.0)
    bbox2 = (0.0, 0.0, 2.0, 2.0)
    assert get_larger_bbox(bbox1, bbox2) == bbox1


def test_get_dimensionality() -> None:
    assert get_dimensionality(np.zeros((2, 3, 4))) == 3


def test_get_number_of_unique_labels() -> None:
    image = np.array([[0, 1], [1, 2]])
    assert get_number_of_unique_labels(image) == 3


def test_extract_unique_masks_and_combinations() -> None:
    stack = np.zeros((2, 3, 3), dtype=np.uint8)
    stack[0, 1, 1] = 1
    stack[1, 1, 1] = 1
    df = extract_unique_masks(stack)
    assert not df.empty
    combos = get_combinations_of_indices(df, distance_threshold=5)
    assert isinstance(combos, pd.DataFrame)


def test_merge_sets() -> None:
    sets = [{0, 1}, {1, 2}, {4}]
    merged, counter = merge_sets(sets)
    assert counter >= 1
    assert any(0 in s and 2 in s for s in merged)


def test_merge_sets_df() -> None:
    data = {
        "index_comparison": ["0,1", "1,2"],
        "index1": [0, 1],
        "index2": [1, 2],
        "area": [5, 3],
        "mask_indices": [np.array([0]), np.array([1])],
        "pseudo_slice": [0, 0],
        "y_x_coords": [(0, 0), (1, 1)],
        "bbox": [(0, 0, 1, 1), (0, 0, 1, 1)],
        "merged/lone": ["merged", "merged"],
    }
    merged_df = merge_sets_df(pd.DataFrame(data))
    assert "label" in merged_df.columns


def test_reassemble_each_mask() -> None:
    df = pd.DataFrame(
        {
            "label": [0, 1],
            "mask_indices": [
                (np.array([0]), np.array([0])),
                (np.array([1]), np.array([1])),
            ],
        }
    )
    mask = reassemble_each_mask(df, (1, 3, 3))
    assert mask.shape == (3, 3)
