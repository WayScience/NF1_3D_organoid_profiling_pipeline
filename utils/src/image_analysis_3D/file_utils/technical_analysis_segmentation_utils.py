"""Utilities for comparing segmentation masks in the technical analysis notebooks."""

from __future__ import annotations

import numpy as np


def convert_indexed_mask_to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert an indexed (labeled) segmentation mask to a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        An indexed segmentation mask where each object is labeled with a
        unique non-zero integer and background is 0.

    Returns
    -------
    np.ndarray
        A binary mask with the same shape as `mask`, where foreground
        (any non-zero label) is 1 and background is 0.
    """

    return (mask > 0).astype(np.uint8)


def extract_IOU(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute the Intersection over Union (IOU) between two segmentation masks.

    Both masks are first converted to binary masks via
    `convert_indexed_mask_to_binary_mask` so that indexed masks with
    differing object labels can be compared.

    Parameters
    ----------
    mask1 : np.ndarray
        The first segmentation mask.
    mask2 : np.ndarray
        The second segmentation mask.
    eps : float, optional
        A small value added to the union to avoid division by zero, by
        default 1e-8.

    Returns
    -------
    float
        The IOU between the two masks.
    """

    binary_mask1 = convert_indexed_mask_to_binary_mask(mask1)
    binary_mask2 = convert_indexed_mask_to_binary_mask(mask2)

    intersection = np.logical_and(binary_mask1, binary_mask2).sum()
    union = np.logical_or(binary_mask1, binary_mask2).sum()

    return intersection / (union + eps)


def signed_xor_3color(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Compute a signed XOR map between two segmentation masks for visualization.

    Both masks are first converted to binary masks via
    `convert_indexed_mask_to_binary_mask`. The resulting map has three
    distinct values suitable for a 3-color colormap:
    - 1 where only `mask1` is foreground
    - -1 where only `mask2` is foreground
    - 0 where both masks agree (either both foreground or both background)

    Parameters
    ----------
    mask1 : np.ndarray
        The first segmentation mask.
    mask2 : np.ndarray
        The second segmentation mask.

    Returns
    -------
    np.ndarray
        A signed difference map with values in {-1, 0, 1}.
    """

    binary_mask1 = convert_indexed_mask_to_binary_mask(mask1)
    binary_mask2 = convert_indexed_mask_to_binary_mask(mask2)

    return binary_mask1.astype(np.int8) - binary_mask2.astype(np.int8)
