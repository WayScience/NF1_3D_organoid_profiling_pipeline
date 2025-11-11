import numpy as np


def convert_indexed_mask_to_binary_mask(indexed_mask: np.ndarray) -> np.ndarray:
    """
    Convert an indexed mask to a binary mask.

    Parameters
    ----------
    indexed_mask : np.ndarray
        An indexed mask where 0 represents the background and any positive integer represents a different object.

    Returns
    -------
    np.ndarray
        A binary mask where True represents the foreground (objects) and False represents the background.
    """
    binary_mask = np.zeros_like(indexed_mask, dtype=bool)
    binary_mask[indexed_mask > 0] = True
    return binary_mask


def extract_IOU(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        A binary mask where True represents the foreground (objects) and False represents the background.
    mask2 : np.ndarray
        A binary mask where True represents the foreground (objects) and False represents the background.

    Returns
    -------
    float
        The Intersection over Union (IoU) score between the two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-12)
    return iou_score.item()


def signed_xor_3color(mask1, mask2):
    """
    Create a 3-value difference map:
    - 2 where only mask1 is True (decon only)
    - 1 where only mask2 is True (other only)
    - 0 where both masks agree
    """
    result = np.zeros_like(mask1, dtype=int)
    result[mask1 & ~mask2] = 2  # decon only
    result[~mask1 & mask2] = 1  # other only
    return result
