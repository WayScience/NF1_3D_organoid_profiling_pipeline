from typing import Dict, Tuple

import numpy
import scipy.ndimage
import skimage.morphology
import tqdm

from .loading_classes import ObjectLoader


def _subsample_image(
    image: numpy.ndarray,
    mask: numpy.ndarray,
    subsample_factor: float,
    z_to_xy_ratio: float = 10.0,
    make_isotropic: bool = True,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Helper function: Subsample 3D image and mask efficiently.

    Uses scipy.zoom instead of map_coordinates for 100x speedup.
    Handles anisotropic data (e.g., Z=1.0 μm, XY=0.1 μm).
    """

    if subsample_factor >= 1.0:
        return image, mask

    if make_isotropic:
        # Step 1: Make isotropic
        image_iso = scipy.ndimage.zoom(image, (z_to_xy_ratio, 1.0, 1.0), order=1)
        mask_iso = (
            scipy.ndimage.zoom(mask.astype(float), (z_to_xy_ratio, 1.0, 1.0), order=0)
            > 0.5
        )

        # Step 2: Subsample uniformly
        subsampled_image = scipy.ndimage.zoom(image_iso, subsample_factor, order=1)
        subsampled_mask = (
            scipy.ndimage.zoom(mask_iso.astype(float), subsample_factor, order=0) > 0.5
        )
    else:
        zoom_z = subsample_factor * z_to_xy_ratio
        zoom_xy = subsample_factor

        subsampled_image = scipy.ndimage.zoom(
            image, (zoom_z, zoom_xy, zoom_xy), order=1
        )
        subsampled_mask = (
            scipy.ndimage.zoom(mask.astype(float), (zoom_z, zoom_xy, zoom_xy), order=0)
            > 0.5
        )

    return subsampled_image, subsampled_mask


def _apply_tophat_filter(
    pixels: numpy.ndarray,
    mask: numpy.ndarray,
    radius: int,
) -> numpy.ndarray:
    """
    Helper function: Apply morphological tophat filter.
    """

    footprint = skimage.morphology.ball(radius, dtype=bool)

    # Create masked copy
    masked_image = numpy.zeros_like(pixels)
    masked_image[mask] = pixels[mask]

    # Erosion
    eroded = skimage.morphology.erosion(masked_image, footprint=footprint)

    # Re-apply mask and dilate
    masked_eroded = numpy.zeros_like(eroded)
    masked_eroded[mask] = eroded[mask]

    filtered = skimage.morphology.dilation(masked_eroded, footprint=footprint)

    return filtered


def measure_3D_granularity(
    object_loader: ObjectLoader,
    radius: int = 10,
    granular_spectrum_length: int = 16,
    subsample_image_value: float = 0.25,
    z_to_xy_ratio: float = 10.0,
    mask_threshold: float = 0.9,
    verbose: bool = False,
) -> Dict[str, list]:
    """
    Calculate the granularity of a 3D image using morphological operations.
    """

    # Validate inputs
    if subsample_image_value <= 0 or subsample_image_value > 1:
        raise ValueError(
            f"subsample_image_value must be in (0, 1], got {subsample_image_value}"
        )

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    if granular_spectrum_length <= 0:
        raise ValueError(
            f"granular_spectrum_length must be positive, got {granular_spectrum_length}"
        )

    # Get original data
    original_pixels = object_loader.image.copy()
    original_mask = object_loader.label_image.copy()
    original_shape = original_pixels.shape

    if subsample_image_value < 1.0:
        if verbose:
            print(
                f"Subsampling image from shape {original_shape} with factor {subsample_image_value}..."
            )

        # Subsample pixels and LABEL image (not binary mask!)
        pixels_subsampled, _ = _subsample_image(
            original_pixels,
            original_mask > 0,  # Binary mask for shape calculation
            subsample_image_value,
            z_to_xy_ratio=z_to_xy_ratio,
            make_isotropic=True,
        )

        # Subsample label image separately to preserve label values
        label_subsampled = scipy.ndimage.zoom(
            original_mask.astype(float),
            (
                z_to_xy_ratio * subsample_image_value,
                subsample_image_value,
                subsample_image_value,
            ),
            order=0,  # Nearest neighbor for labels
        )

        pixels = pixels_subsampled
        labels = label_subsampled.astype(int)
        mask = labels > 0
        current_shape = pixels.shape

        if verbose:
            print(f"  Original shape: {original_shape}")
            print(f"  Subsampled shape: {current_shape}")
            print(
                f"  Unique labels in original: {len(set(original_mask[original_mask > 0]))}"
            )
            print(f"  Unique labels in subsampled: {len(set(labels[labels > 0]))}")
    else:
        pixels = original_pixels
        labels = original_mask
        mask = original_mask > 0
        current_shape = original_shape

    if verbose:
        print("Applying tophat filter...")

    background = _apply_tophat_filter(pixels, mask, radius)
    pixels = pixels - background
    pixels[pixels < 0] = 0

    # Initialize measurements
    object_measurements = {
        "object_id": [],
        "feature": [],
        "value": [],
    }

    # Set pixels outside mask to 0
    pixels[~mask] = 0

    # Calculate starting mean intensity
    masked_pixels = pixels[mask]
    if masked_pixels.size == 0:
        if verbose:
            print("ERROR: No valid pixels in mask!")
        start_mean = numpy.finfo(float).eps
    else:
        start_mean = max(numpy.mean(masked_pixels), numpy.finfo(float).eps)

    if verbose:
        print(f"Start mean: {start_mean}")

    # Initialize erosion image
    ero = pixels.copy()
    ero[~mask] = 0
    current_mean = start_mean

    # Footprint for iterative erosion/reconstruction
    footprint = skimage.morphology.ball(radius, dtype=bool)

    # Get unique object IDs in subsampled space
    unique_labels = set(labels[labels > 0])
    if verbose:
        print(f"Processing {len(unique_labels)} objects")

    objects_dict = {
        "label": [],
        "previous_mean": [],
        "current_mean": [],
        "start_mean": [],
    }
    for label in unique_labels:
        objects_dict["label"].append(label)
        objects_dict["previous_mean"].append(current_mean)
        objects_dict["current_mean"].append(current_mean)
        objects_dict["start_mean"].append(start_mean)

    # Iterate through granular spectrum scales
    for scale in tqdm.tqdm(
        range(1, granular_spectrum_length + 1),
        desc="Granularity measurement",
        position=1,
        leave=False,
    ):
        # Erosion step
        ero_masked = numpy.zeros_like(ero)
        ero_masked[mask] = ero[mask]
        ero = skimage.morphology.erosion(ero_masked, footprint=footprint)

        # Reconstruction step
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)

        # Calculate image-level mean
        rec_masked = rec[mask]
        if rec_masked.size > 0:
            current_mean = numpy.mean(rec_masked)
        else:
            current_mean = 0.0

        if verbose and scale == 1:
            print(f"Scale 1 - current_mean: {current_mean}, prev_mean: {start_mean}")

        # Calculate per-object granularity at this scale
        for index, label in enumerate(objects_dict["label"]):
            new_object_mean = (
                rec[labels == label].mean() if numpy.any(labels == label) else 0.0
            )
            current_object_mean = objects_dict["current_mean"][index]

            # Granular spectrum: how much signal was removed at this scale
            gss = (current_object_mean - new_object_mean) * 100 / start_mean

            objects_dict["current_mean"][index] = new_object_mean
            # Sanity checks
            if gss < 0 or gss > 100:
                if verbose and scale == 1:
                    print(f"WARNING: Invalid gss={gss} for label={label}")
                    print(
                        f"  current_mean={current_mean}, object_mean={current_object_mean}, start_mean={start_mean}"
                    )
                gss = max(0, min(100, gss))  # Clamp to [0, 100]

            # Record measurement
            object_measurements["object_id"].append(label)
            object_measurements["feature"].append(scale)
            object_measurements["value"].append(gss)

    if verbose:
        print(f"Total measurements: {len(object_measurements['object_id'])}")
        non_zero = sum(1 for v in object_measurements["value"] if v > 0)
        print(f"Non-zero measurements: {non_zero}")
        if non_zero > 0:
            print(
                f"Mean granularity: {numpy.mean([v for v in object_measurements['value'] if v > 0]):.2f}"
            )

    return object_measurements
