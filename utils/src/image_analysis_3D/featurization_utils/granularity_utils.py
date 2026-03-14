"""
Calculate the granularity spectrum of a 3D image.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy
import scipy.ndimage
import skimage.morphology
import tqdm
from image_analysis_3D.featurization_utils.loading_classes import ObjectLoader


def _fix_scipy_ndimage_result(result: float | list | numpy.ndarray) -> numpy.ndarray:
    """Convert scipy.ndimage aggregation results to a consistent array.

    Equivalent to centrosome.cpmorphology.fixup_scipy_ndimage_result.
    scipy.ndimage.mean/sum can return a scalar when there's one label,
    or a list otherwise. This ensures we always get a numpy array.

    Parameters
    ----------
    result : scalar, list, or numpy.ndarray
        Output from scipy.ndimage.mean or similar.

    Returns
    -------
    numpy.ndarray
        1-D array of results.
    """
    if numpy.isscalar(result):
        return numpy.array([result])
    return numpy.asarray(result)


def _subsample_3d(
    data: numpy.ndarray,
    new_shape: numpy.ndarray,
    subsample_factor: float,
    order: int = 1,
) -> numpy.ndarray:
    """Subsample a 3D array using map_coordinates, matching CellProfiler.

    CellProfiler generates coordinates for the new shape and divides by
    subsample_factor to map back into the original coordinate space.
    The same scalar factor is used for all three axes.

    Parameters
    ----------
    data : numpy.ndarray
        3D array to subsample.
    new_shape : numpy.ndarray
        Target shape as a float array (coordinate grid extent).
    subsample_factor : float
        The factor used to divide coordinates (same for all axes).
    order : int
        Interpolation order (1 for linear, 0 for nearest-neighbor).

    Returns
    -------
    numpy.ndarray
        Subsampled array.
    """
    if subsample_factor >= 1.0:
        return data.copy()

    k, i, j = (
        numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(float)
        / subsample_factor
    )
    return scipy.ndimage.map_coordinates(data, (k, i, j), order=order)


def _upsample_3d(
    data: numpy.ndarray,
    subsampled_shape: numpy.ndarray,
    original_shape: tuple,
) -> numpy.ndarray:
    """Upsample a 3D array back to original shape using map_coordinates.

    Matches CellProfiler's approach for restoring reconstructed images
    to the original label resolution.

    Parameters
    ----------
    data : numpy.ndarray
        Subsampled 3D array to upsample.
    subsampled_shape : numpy.ndarray
        Shape of the subsampled space (float array, preserves CellProfiler
        precision).
    original_shape : tuple
        Target shape to upsample to.

    Returns
    -------
    numpy.ndarray
        Upsampled array at original_shape resolution.
    """
    k, i, j = numpy.mgrid[
        0 : original_shape[0], 0 : original_shape[1], 0 : original_shape[2]
    ].astype(float)
    k *= float(subsampled_shape[0] - 1) / float(original_shape[0] - 1)
    i *= float(subsampled_shape[1] - 1) / float(original_shape[1] - 1)
    j *= float(subsampled_shape[2] - 1) / float(original_shape[2] - 1)
    return scipy.ndimage.map_coordinates(data, (k, i, j), order=1)


def measure_3D_granularity(
    object_loader: ObjectLoader,
    radius: int = 10,
    granular_spectrum_length: int = 16,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    mask_threshold: float = 0.9,
    verbose: bool = False,
    image_mask: Optional[numpy.ndarray] = None,
) -> Dict[str, list]:
    """Calculate the granularity spectrum of a 3D image.

    Follows the CellProfiler MeasureGranularity algorithm exactly for 3D:
    1. Subsample the image uniformly (same factor for Z, Y, X).
    2. Further subsample for background tophat removal.
    3. Iteratively erode with ball(1) and reconstruct, measuring
       signal lost at each scale as image-level and per-object values.

    Parameters
    ----------
    object_loader : ObjectLoader
        Loader containing the image and label arrays.
    radius : int
        Radius of the structuring element for background removal.
        Should correspond to texture radius *after* subsampling.
    granular_spectrum_length : int
        Number of granularity scales to measure.
    subsample_size : float
        Subsampling factor for the image (0, 1]. Applied uniformly to Z/Y/X.
    image_sample_size : float
        Subsampling factor for background reduction (0, 1].
        Applied relative to the already-subsampled image.
    mask_threshold : float
        Threshold for converting interpolated masks back to boolean.
    verbose : bool
        Print diagnostic information.
    image_mask : numpy.ndarray or None
        Boolean mask matching the image shape. Corresponds to CellProfiler's
        ``im.mask``. If None (default), all pixels are considered valid
        (all-True mask), matching the typical CellProfiler behavior for
        unmasked images.

    Returns
    -------
    Dict[str, list]
        Dictionary with keys 'object_id', 'feature', 'value'.
        Image-level measurements use object_id=0.
    """
    # Validate inputs
    if subsample_size <= 0 or subsample_size > 1:
        raise ValueError(f"subsample_size must be in (0, 1], got {subsample_size}")
    if image_sample_size <= 0 or image_sample_size > 1:
        raise ValueError(
            f"image_sample_size must be in (0, 1], got {image_sample_size}"
        )
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if granular_spectrum_length <= 0:
        raise ValueError(
            f"granular_spectrum_length must be positive, got {granular_spectrum_length}"
        )

    # Get original data
    original_pixels = object_loader.image
    original_labels = object_loader.label_image
    original_shape = original_pixels.shape

    # Mask: CellProfiler uses im.mask (typically all-True for unmasked images)
    if image_mask is None:
        original_mask = numpy.ones(original_shape, dtype=bool)
    else:
        original_mask = image_mask.astype(bool)

    # ------------------------------------------------------------------
    # Step 1: Subsample image and mask (uniform factor for all axes)
    # CellProfiler: new_shape = shape * subsample_size
    #   coordinates = mgrid[0:new_shape] / subsample_size
    # ------------------------------------------------------------------
    new_shape = numpy.array(original_shape, dtype=float)

    if subsample_size < 1.0:
        new_shape = new_shape * subsample_size

        pixels = _subsample_3d(
            original_pixels,
            new_shape,
            subsample_factor=subsample_size,
            order=1,
        )
        mask = (
            _subsample_3d(
                original_mask.astype(float),
                new_shape,
                subsample_factor=subsample_size,
                order=1,
            )
            > mask_threshold
        )

        if verbose:
            print(
                f"Subsampled image: {original_shape} -> {pixels.shape} "
                f"(factor={subsample_size})"
            )
    else:
        pixels = original_pixels.copy()
        mask = original_mask.copy()

    # ------------------------------------------------------------------
    # Step 2: Background removal via tophat filter
    #
    # CellProfiler 3D BUG (replicated for compatibility):
    #   The 3D branch uses new_shape for grid bounds and subsample_size
    #   for coordinate division, instead of back_shape and
    #   image_sample_size as the 2D branch does. This means:
    #   - back_pixels has the SAME shape as pixels (not smaller)
    #   - Many coordinates are out of bounds → map_coordinates returns 0
    #   We replicate this exactly to match CellProfiler output.
    # ------------------------------------------------------------------
    if image_sample_size < 1.0:
        back_shape = new_shape * image_sample_size

        # CellProfiler 3D: mgrid[0:new_shape] / subsample_size
        # (NOT mgrid[0:back_shape] / image_sample_size as 2D does)
        k, i, j = (
            numpy.mgrid[0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]].astype(
                float
            )
            / subsample_size
        )
        back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
        back_mask = (
            scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j))
            > mask_threshold
        )

        if verbose:
            print(
                f"Background subsampled: pixels {pixels.shape} -> "
                f"back_pixels {back_pixels.shape} "
                f"(image_sample_size={image_sample_size})"
            )
    else:
        back_pixels = pixels
        back_mask = mask
        back_shape = new_shape

    # Tophat filter: masked erosion + masked dilation
    footprint_bg = skimage.morphology.ball(radius, dtype=bool)

    back_pixels_masked = numpy.zeros_like(back_pixels)
    back_pixels_masked[back_mask] = back_pixels[back_mask]
    back_pixels = skimage.morphology.erosion(back_pixels_masked, footprint=footprint_bg)

    back_pixels_masked = numpy.zeros_like(back_pixels)
    back_pixels_masked[back_mask] = back_pixels[back_mask]
    back_pixels = skimage.morphology.dilation(
        back_pixels_masked, footprint=footprint_bg
    )

    # Upsample background back to subsampled image size
    if image_sample_size < 1.0:
        # CellProfiler 3D: mgrid[0:new_shape] with coords scaled by
        # (back_shape - 1) / (new_shape - 1)
        k, i, j = numpy.mgrid[
            0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
        ].astype(float)
        k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
        back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)

    # Subtract background
    pixels = pixels - back_pixels
    pixels[pixels < 0] = 0

    if verbose:
        print("Background removed via tophat filter.")

    # ------------------------------------------------------------------
    # Step 3: Object initialization
    # CellProfiler computes per-object start_mean from the ORIGINAL image
    # (im.pixel_data) using the full-resolution label image, with labels
    # masked by im.mask: labels[~im.mask] = 0.
    # ------------------------------------------------------------------
    object_measurements = {
        "object_id": [],
        "feature": [],
        "value": [],
    }

    nobjects = int(numpy.max(original_labels)) if numpy.any(original_labels > 0) else 0

    if nobjects > 0:
        label_range = numpy.arange(1, nobjects + 1)

        # CellProfiler: self.labels[~im.mask] = 0
        masked_labels = original_labels.copy()
        masked_labels[~original_mask] = 0

        per_object_current_mean = _fix_scipy_ndimage_result(
            scipy.ndimage.mean(original_pixels, masked_labels, label_range)
        )
        per_object_start_mean = numpy.maximum(
            per_object_current_mean, numpy.finfo(float).eps
        )
    else:
        label_range = numpy.array([], dtype=int)
        masked_labels = original_labels
        per_object_current_mean = numpy.array([])
        per_object_start_mean = numpy.array([])

    # ------------------------------------------------------------------
    # Step 4: Granular spectrum loop
    # CellProfiler computes startmean AFTER background subtraction but
    # BEFORE zeroing pixels outside mask (zeroing is implicit via indexing).
    # ------------------------------------------------------------------
    startmean = numpy.mean(pixels[mask])
    ero = pixels.copy()
    # Mask the test image so masked pixels have no effect during reconstruction
    ero[~mask] = 0
    currentmean = startmean
    startmean = max(startmean, numpy.finfo(float).eps)

    # CellProfiler uses ball(1) for the iterative erosion/reconstruction loop
    footprint = skimage.morphology.ball(1, dtype=bool)

    if verbose:
        print(
            f"Image startmean: {startmean:.6f}, "
            f"Processing {nobjects} objects, "
            f"Spectrum length: {granular_spectrum_length}"
        )

    for scale in tqdm.tqdm(
        range(1, granular_spectrum_length + 1),
        desc="Granularity measurement",
        position=1,
        leave=False,
    ):
        prevmean = currentmean

        # Masked erosion
        ero_masked = numpy.zeros_like(ero)
        ero_masked[mask] = ero[mask]
        ero = skimage.morphology.erosion(ero_masked, footprint=footprint)

        # Reconstruction
        rec = skimage.morphology.reconstruction(ero, pixels, footprint=footprint)

        # Image-level granularity
        currentmean = numpy.mean(rec[mask])
        gs = (prevmean - currentmean) * 100 / startmean

        if verbose and scale == 1:
            print(f"Scale 1 - gs: {gs:.4f}, currentmean: {currentmean:.6f}")

        # ----------------------------------------------------------
        # Per-object granularity: upsample rec to original shape,
        # then compute per-label means using masked_labels.
        # ----------------------------------------------------------
        if nobjects > 0:
            if subsample_size < 1.0:
                rec_full = _upsample_3d(
                    rec,
                    subsampled_shape=new_shape,
                    original_shape=original_shape,
                )
            else:
                rec_full = rec

            # Single-pass per-object mean via scipy.ndimage.mean
            new_object_means = _fix_scipy_ndimage_result(
                scipy.ndimage.mean(rec_full, masked_labels, label_range)
            )

            # Granular spectrum: (prev - new) * 100 / start, per object
            gss = (
                (per_object_current_mean - new_object_means)
                * 100
                / per_object_start_mean
            )

            per_object_current_mean = new_object_means

            # Record measurements for each object
            for idx in range(len(label_range)):
                object_measurements["object_id"].append(int(label_range[idx]))
                object_measurements["feature"].append(scale)
                object_measurements["value"].append(float(gss[idx]))

    if verbose:
        n_total = len(object_measurements["object_id"])
        non_zero = sum(1 for v in object_measurements["value"] if v > 0)
        print(f"Total measurements: {n_total}")
        print(f"Non-zero measurements: {non_zero}")
        if non_zero > 0:
            vals = [v for v in object_measurements["value"] if v > 0]
            print(f"Mean granularity: {numpy.mean(vals):.2f}")

    return object_measurements
