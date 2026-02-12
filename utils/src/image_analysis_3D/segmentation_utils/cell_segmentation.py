"""
# Cell segmentation in 3D
"""

import numpy as np
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
from skimage.filters import sobel


def segment_cells_with_3D_watershed(
    cyto_signal: np.ndarray,
    nuclei_mask: np.ndarray,
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
        compactness=compactness,  # keep at 0
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

    Returns
    -------
    np.ndarray
        3D numpy array representing the segmented cell mask.
    """
    # generate the low frequency elevation map
    # all morphology types use the same initial elevation map
    elevation_map = skimage.filters.butterworth(
        cyto_signal,
        cutoff_frequency_ratio=0.08,
        order=2,
        high_pass=False,
        squared_butterworth=False,
    )
    # generate threshold using otsu
    threshold = skimage.filters.threshold_otsu(elevation_map)
    # generate thresholded signal
    elevation_map_threshold_signal = elevation_map.copy()
    elevation_map_threshold_signal = elevation_map_threshold_signal > threshold

    min_size = 1000  # volume in voxels 10x10x10
    max_size = (
        10_000_000  # volume in voxels ~215x215x215 (max size of a cell in this dataset)
    )

    # set connectivity and compactness parameters for watershed based on morphology
    connectivity = 1
    compactness = 0

    if organoid_label == "globular":
        elevation_map = skimage.filters.gaussian(cyto_signal, sigma=1.0)
        elevation_map = sobel(elevation_map)
        # update compactness for globular morphology to reduce oversegmentation
        connectivity = 0
        compactness = 0

    elif organoid_label == "small/dissociated":
        print("Dissociated morphology selected")
        elevation_map = skimage.morphology.binary_dilation(
            elevation_map_threshold_signal,
            skimage.morphology.ball(2),
        )
        elevation_map = sobel(elevation_map)
        elevation_map = skimage.filters.gaussian(elevation_map, sigma=3)
        connectivity = 0
        compactness = 0

    elif organoid_label == "elongated":
        elevation_map = sobel(elevation_map_threshold_signal)
        elevation_map = skimage.filters.gaussian(elevation_map, sigma=3)
        connectivity = 0
        compactness = 0
    else:
        raise ValueError(f"Unknown morphology label: {organoid_label}")

    cell_mask = segment_cells_with_3D_watershed(
        cyto_signal=elevation_map,
        nuclei_mask=nuclei_mask,
        connectivity=connectivity,
        compactness=compactness,
    )
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
