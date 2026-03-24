"""
# Cell segmentation in 3D
"""

from typing import Union

import cupy
import cupyx
import cupyx.scipy.ndimage
import numpy as np
import scipy
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
from skimage.filters import sobel


def fill_holes_in_mask(
    mask: np.ndarray,
    compartment: Union[str, None] = None,
) -> np.ndarray:
    """
    This function fills holes in instance segmented mask images

    Parameters
    ----------
    mask : np.ndarray
        3D instance segmented mask image where each cell has a unique integer label and background is 0
    compartment : str, optional
        Compartment type of the mask (e.g. "cell" or "organoid"), by default None. This is used to determine the hole filling strategy.

    Errors
    ------
    ValueError
        If compartment is not specified, a ValueError is raised since the hole filling strategy depends on the compartment type.

    Returns
    -------
    np.ndarray
        3D instance segmented mask image with holes filled
    """
    if compartment is None:
        raise ValueError("Compartment must be specified for hole filling.")
    # fill enclosed holes in segmented cells for each label
    # use hybrid filling: 3D fill + per-slice 2D fill (handles z-tunnels common in microscopy)
    mask_cp = cupy.asarray(mask)
    new_mask_cp = cupy.zeros_like(mask_cp)
    structure_3d = cupyx.scipy.ndimage.generate_binary_structure(rank=3, connectivity=1)

    if compartment.lower() == "cell":
        for label in cupy.unique(mask_cp):
            label = int(label)
            if label == 0:
                continue  # skip background
            tmp_mask = mask_cp == label

            # 3D enclosed-hole fill
            tmp_mask = cupyx.scipy.ndimage.binary_fill_holes(
                tmp_mask,
                structure=structure_3d,
            )

            # 2D per-z enclosed-hole fill (captures holes open across z in 3D)
            for z in range(tmp_mask.shape[0]):
                tmp_mask[z] = cupyx.scipy.ndimage.binary_fill_holes(tmp_mask[z])
            new_mask_cp[tmp_mask] = label
        # convert back to numpy array
        mask = cupy.asnumpy(new_mask_cp).astype(mask.dtype)

    elif compartment.lower() == "organoid":
        # for organoid masks
        # 3D enclosed-hole fill
        new_mask_cp = cupyx.scipy.ndimage.binary_fill_holes(
            mask_cp,
            structure=structure_3d,
        )

        # 2D per-z enclosed-hole fill (captures holes open across z in 3D)
        for z in range(new_mask_cp.shape[0]):
            new_mask_cp[z] = cupyx.scipy.ndimage.binary_fill_holes(new_mask_cp[z])

        # convert back to numpy array
        mask = cupy.asnumpy(new_mask_cp).astype(mask.dtype)
        # define labels now
        mask, _ = scipy.ndimage.label(mask)

    return mask


def segment_cells_with_3D_watershed(
    cyto_signal: np.ndarray,
    nuclei_mask: np.ndarray,
    thresholded_signal: np.ndarray,
    connectivity: int = 1,
    compactness: float = 0,
) -> np.ndarray:
    """
    Segment cells using 3D watershed algorithm.

    Segments cells using a 3D watershed algorithm given cytoplasm signal
    (channel) and nuclei mask.

    Parameters
    ----------
    cyto_signal : np.ndarray
        3D numpy array representing the cytoplasm signal.
    nuclei_mask : np.ndarray
        3D numpy array representing the nuclei mask.
    thresholded_signal : np.ndarray
        3D numpy array representing the thresholded cytoplasm signal to be used as a mask for watershed.
    connectivity : int, optional
        Connectivity parameter for the watershed algorithm. Default is 1.
        A value of 1 means only directly adjacent pixels (6-connectivity in 3D)
        are considered connected, preventing over-segmentation.
    compactness : float, optional
        Compactness parameter controlling watershed region shape. Default is 0.
        A value of 0 means no compactness enforcement, allowing irregularly
        shaped segments to capture true cell morphology.

    Returns
    -------
    np.ndarray
        3D numpy array representing the segmented cell mask.
    """
    labels = skimage.segmentation.watershed(
        image=cyto_signal,
        markers=nuclei_mask,
        # connectivity parameter controls how pixels are connected in the watershed algorithm.
        # A value of 1 means that only directly adjacent pixels (6-connectivity in 3D) are considered connected,
        # which is appropriate for cell segmentation to prevent over-segmentation.
        connectivity=connectivity,  # keep at 1
        # compactness parameter controls the shape of the watershed regions.
        # A value of 0 means that the watershed will not enforce compactness,
        # allowing for more irregularly shaped segments,
        # which is often desirable in cell segmentation to capture the true morphology of cells.
        compactness=compactness,
        mask=thresholded_signal,
    )

    # change the largest label (by area) to 0
    # cleans up the output and sets the background properly
    unique, counts = np.unique(labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    labels[labels == largest_label] = 0
    return labels


def perform_morphology_dependent_segmentation(
    organoid_label: str,
    cyto_signal: np.ndarray,
    nuclei_mask: np.ndarray,
    min_size: int = 1_000,
    max_size: int = 10_000_000,
) -> np.ndarray:
    """
    Perform morphology-dependent cell segmentation.

    Performs morphology dependent segmentation based on the provided morphology label.

    Parameters
    ----------
    organoid_label : str
        Morphology label indicating the type of morphology.
    cyto_signal : np.ndarray
        3D numpy array representing the cytoplasm signal.
    nuclei_mask : np.ndarray
        3D numpy array representing the nuclei mask.
    min_size : int, optional
        Minimum size threshold for segmented objects. Default is 1,000 voxels.
    max_size : int, optional
        Maximum size threshold for segmented objects. Default is 10,000,000 voxels.

    Returns
    -------
    np.ndarray
        3D numpy array representing the segmented cell mask.
    """
    if organoid_label in {"failed", "blank"}:
        print("Failed/blank morphology selected, skipping cell segmentation")
        return np.zeros_like(cyto_signal, dtype=np.int32)
    # generate the low frequency elevation map
    # all morphology types use the same initial elevation map
    butterworth_filter = skimage.filters.butterworth(
        cyto_signal,
        cutoff_frequency_ratio=0.08,
        order=2,
        high_pass=False,
        squared_butterworth=False,
    )

    if organoid_label in {"globular", "cluster"}:
        # apply gaussian filter to smooth the signal for better thresholding
        elevation_map_threshold_signal = skimage.filters.gaussian(
            cyto_signal, sigma=2.5
        )
        threshold = skimage.filters.threshold_otsu(elevation_map_threshold_signal)

        elevation_map_threshold_signal[elevation_map_threshold_signal < threshold] = 0
        elevation_map_threshold_signal[elevation_map_threshold_signal > 0] = 1

        elevation_map = sobel(butterworth_filter)
        connectivity = 1
        compactness = 1

    elif organoid_label in {"small", "dissociated"}:
        elevation_map_threshold_signal = skimage.filters.gaussian(cyto_signal, sigma=3)
        threshold = skimage.filters.threshold_otsu(elevation_map_threshold_signal)
        elevation_map_threshold_signal[elevation_map_threshold_signal < threshold] = 0
        elevation_map_threshold_signal[elevation_map_threshold_signal > 0] = 1
        elevation_map_threshold_signal = skimage.morphology.dilation(
            elevation_map_threshold_signal,
            skimage.morphology.ball(1),
        )

        elevation_map = skimage.filters.gaussian(butterworth_filter, sigma=1)
        elevation_map = sobel(elevation_map)

        connectivity = 1
        compactness = 0

    elif organoid_label in {"elongated"}:
        elevation_map_threshold_signal = skimage.filters.gaussian(cyto_signal, sigma=4)
        threshold = skimage.filters.threshold_otsu(elevation_map_threshold_signal)
        elevation_map_threshold_signal[elevation_map_threshold_signal < threshold] = 0
        elevation_map_threshold_signal[elevation_map_threshold_signal > 0] = 1
        elevation_map_threshold_signal = skimage.morphology.dilation(
            elevation_map_threshold_signal,
            skimage.morphology.ball(10),
        )

        elevation_map = skimage.filters.gaussian(butterworth_filter, sigma=1)
        elevation_map = sobel(elevation_map)

        connectivity = 0
        compactness = 0
    else:
        raise ValueError(f"Unknown morphology label: {organoid_label}")

    cell_mask = segment_cells_with_3D_watershed(
        cyto_signal=elevation_map,
        nuclei_mask=nuclei_mask,
        connectivity=connectivity,
        compactness=compactness,
        thresholded_signal=elevation_map_threshold_signal,
    )

    cell_mask = fill_holes_in_mask(cell_mask, compartment="cell")

    # Remove small objects while preserving label IDs
    # we avoid using the built-in skimage function to preserve label IDs
    props = skimage.measure.regionprops(cell_mask)

    # Remove objects smaller than threshold
    for prop in props:
        if prop.area < min_size:  # min size threshold (adjust as needed)
            cell_mask[cell_mask == prop.label] = 0

    # remove large objects
    unique, counts = np.unique(cell_mask[cell_mask > 0], return_counts=True)
    for label, count in zip(unique, counts):
        if count > max_size:
            cell_mask[cell_mask == label] = 0

    return cell_mask
