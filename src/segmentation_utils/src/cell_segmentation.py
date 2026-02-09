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
) -> np.ndarray:
    """ "
    Description
    -----------
        Segments cells using 3D watershed algorithm given cytoplasm signal and nuclei mask.
    Parameters
    ----------
        cyto_signal : np.ndarray
            3D numpy array representing the cytoplasm signal.
        nuclei_mask : np.ndarray
            3D numpy array representing the nuclei mask.
    Returns
    -------
        np.ndarray
            3D numpy array representing the segmented cell mask.
    """
    labels = skimage.segmentation.watershed(
        image=cyto_signal,
        markers=nuclei_mask,
        connectivity=0,  # keep at 1
        compactness=0,  # keep at 0
    )

    # change the largest label (by area) to 0
    # cleans up the output and sets the background properly
    unique, counts = np.unique(labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    labels[labels == largest_label] = 0
    return labels


def perform_morphology_dependent_segmentation(
    label: str,
    cyto2: np.ndarray,
    nuclei_mask: np.ndarray,
) -> np.ndarray:
    """
    Description
    -----------
        Performs morphology dependent segmentation based on the provided morphology label.
    Parameters
    ----------
        label : str
            Morphology label indicating the type of morphology.
        cyto2 : np.ndarray
            3D numpy array representing the cytoplasm signal.
        nuclei_mask : np.ndarray
            3D numpy array representing the nuclei mask.
    Returns
    -------
        np.ndarray
            3D numpy array representing the segmented cell mask.
    """
    # generate the low frequency elevation map
    # all morhology types use the same initial elevation map
    elevation_map = skimage.filters.butterworth(
        cyto2,
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

    min_size = 1000  # volume in voxels
    max_size = 10_000_000  # volume in voxels

    if label == "globular":
        elevation_map = skimage.filters.gaussian(cyto2, sigma=1.0)
        elevation_map = sobel(elevation_map)

    elif label == "small/dissociated":
        print("Dissociated morphology selected")
        elevation_map = skimage.morphology.binary_dilation(
            elevation_map_threshold_signal,
            skimage.morphology.ball(2),
        )
        elevation_map = sobel(elevation_map)
        elevation_map = skimage.filters.gaussian(elevation_map, sigma=3)

    # elif label == "small":
    #     elevation_map = sobel(elevation_map)
    #     elevation_map = skimage.filters.gaussian(elevation_map, sigma=3)
    elif label == "elongated":
        elevation_map = sobel(elevation_map_threshold_signal)
        elevation_map = skimage.filters.gaussian(elevation_map, sigma=3)
    else:
        raise ValueError(f"Unknown morphology label: {label}")

    cell_mask = segment_cells_with_3D_watershed(
        cyto_signal=elevation_map,
        nuclei_mask=nuclei_mask,
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
