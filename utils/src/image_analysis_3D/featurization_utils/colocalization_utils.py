"""Colocalization feature extraction utilities for 3D image objects.

Computes per-object colocalization metrics (Pearson correlation, Manders
coefficients, overlap coefficient, K1/K2 coefficients) between pairs of
fluorescence channels using the Costes automatic thresholding method.
"""

from typing import Dict, Tuple

import numpy
import scipy.ndimage
import skimage
from image_analysis_3D.image_utils.image_utils import (
    crop_3D_image,
    new_crop_border,
    select_objects_from_label,
)


def linear_costes_threshold_calculation(
    first_image: numpy.ndarray,
    second_image: numpy.ndarray,
    scale_max: int = 255,
    fast_costes: str = "Accurate",
) -> Tuple[float, float]:
    """
    Finds the Costes Automatic Threshold for colocalization using a linear algorithm.
    Candiate thresholds are gradually decreased until Pearson R falls below 0.
    If "Fast" mode is enabled the "steps" between tested thresholds will be increased
    when Pearson R is much greater than 0. The other mode is "Accurate" which
    will always step down by the same amount.
    """
    i_step = 1 / scale_max  # Step size for the threshold as a float
    non_zero = (first_image > 0) | (second_image > 0)
    xvar = numpy.var(first_image[non_zero], axis=0, ddof=1)
    yvar = numpy.var(second_image[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(first_image[non_zero], axis=0)
    ymean = numpy.mean(second_image[non_zero], axis=0)

    z = first_image[non_zero] + second_image[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt(
        (yvar - xvar) * (yvar - xvar) + 4 * (covar * covar)
    )
    a = num / denom
    b = ymean - a * xmean

    # Start at 1 step above the maximum value
    img_max = max(first_image.max(), second_image.max())
    i = i_step * ((img_max // i_step) + 1)

    num_true = None
    first_image_max = first_image.max()
    second_image_max = second_image.max()

    # Initialise without a threshold
    costReg, _ = scipy.stats.pearsonr(first_image, second_image)
    thr_first_image_c = i
    thr_second_image_c = (a * i) + b
    while i > first_image_max and (a * i) + b > second_image_max:
        i -= i_step
    while i > i_step:
        thr_first_image_c = i
        thr_second_image_c = (a * i) + b
        combt = (first_image < thr_first_image_c) | (second_image < thr_second_image_c)
        try:
            # Only run pearsonr if the input has changed.
            if (positives := numpy.count_nonzero(combt)) != num_true:
                costReg, _ = scipy.stats.pearsonr(
                    first_image[combt], second_image[combt]
                )
                num_true = positives

            if costReg <= 0:
                break
            elif fast_costes == "Accurate" or i < i_step * 10:
                i -= i_step
            elif costReg > 0.45:
                # We're way off, step down 10x
                i -= i_step * 10
            elif costReg > 0.35:
                # Still far from 0, step 5x
                i -= i_step * 5
            elif costReg > 0.25:
                # Step 2x
                i -= i_step * 2
            else:
                i -= i_step
        except ValueError:
            break
    return thr_first_image_c, thr_second_image_c


def bisection_costes_threshold_calculation(
    first_image: numpy.ndarray, second_image: numpy.ndarray, scale_max: int = 255
) -> tuple[float, float]:
    """
    Finds the Costes Automatic Threshold for colocalization using a bisection algorithm.
    Candidate thresholds are selected from within a window of possible intensities,
    this window is narrowed based on the R value of each tested candidate.
    We're looking for the first point at 0, and R value can become highly variable
    at lower thresholds in some samples. Therefore the candidate tested in each
    loop is 1/6th of the window size below the maximum value (as opposed to the midpoint).
    """

    non_zero = (first_image > 0) | (second_image > 0)
    xvar = numpy.var(first_image[non_zero], axis=0, ddof=1)
    yvar = numpy.var(second_image[non_zero], axis=0, ddof=1)

    xmean = numpy.mean(first_image[non_zero], axis=0)
    ymean = numpy.mean(second_image[non_zero], axis=0)

    z = first_image[non_zero] + second_image[non_zero]
    zvar = numpy.var(z, axis=0, ddof=1)

    covar = 0.5 * (zvar - (xvar + yvar))

    denom = 2 * covar
    num = (yvar - xvar) + numpy.sqrt((yvar - xvar) * (yvar - xvar) + 4 * (covar**2))
    a = num / denom
    b = ymean - a * xmean

    # Initialise variables
    left = 1
    right = scale_max
    mid = ((right - left) // (6 / 5)) + left
    lastmid = 0
    # Marks the value with the last positive R value.
    valid = 1

    while lastmid != mid:
        thr_first_image_c = mid / scale_max
        thr_second_image_c = (a * thr_first_image_c) + b
        combt = (first_image < thr_first_image_c) | (second_image < thr_second_image_c)
        if numpy.count_nonzero(combt) <= 2:
            # Can't run meaningful pearson with only 2 values.
            left = mid - 1
        else:
            try:
                costReg, _ = scipy.stats.pearsonr(
                    first_image[combt], second_image[combt]
                )
                if costReg < 0:
                    left = mid - 1
                elif costReg >= 0:
                    right = mid + 1
                    valid = mid
            except ValueError:
                # Catch misc Pearson errors with low sample numbers
                left = mid - 1
        lastmid = mid
        if right - left > 6:
            mid = ((right - left) // (6 / 5)) + left
        else:
            mid = ((right - left) // 2) + left

    thr_first_image_c = (valid - 1) / scale_max
    thr_second_image_c = (a * thr_first_image_c) + b

    return thr_first_image_c, thr_second_image_c


def prepare_two_images_for_colocalization(
    label_object1: numpy.ndarray,
    label_object2: numpy.ndarray,
    image_object1: numpy.ndarray,
    image_object2: numpy.ndarray,
    object_id1: int,
    object_id2: int,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    This function prepares two images for colocalization analysis by cropping them to the bounding boxes of the specified objects.
    It selects the objects from the label images, calculates their bounding boxes, and crops the images accordingly.

    Parameters
    ----------
    label_object1 : numpy.ndarray
        The segmented label image for the first object.
    label_object2 : numpy.ndarray
        The segmented label image for the second object.
    image_object1 : numpy.ndarray
        The spectral image to crop for the first object.
    image_object2 : numpy.ndarray
        The spectral image to crop for the second object.
    object_id1 : int
        The object index to select from the label image for the first object.
    object_id2 : int
        The object index to select from the label image for the second object.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The two cropped images for colocalization analysis.
    """
    label_object1 = select_objects_from_label(label_object1, object_id1)
    label_object2 = select_objects_from_label(label_object2, object_id2)
    # get the image bbox
    props_image1 = skimage.measure.regionprops_table(label_object1, properties=["bbox"])
    bbox_image1 = (
        props_image1["bbox-0"][0],  # z min
        props_image1["bbox-1"][0],  # y min
        props_image1["bbox-2"][0],  # x min
        props_image1["bbox-3"][0],  # z max
        props_image1["bbox-4"][0],  # y max
        props_image1["bbox-5"][0],  # x max
    )

    props_image2 = skimage.measure.regionprops_table(label_object2, properties=["bbox"])
    bbox_image2 = (
        props_image2["bbox-0"][0],  # z min
        props_image2["bbox-1"][0],  # y min
        props_image2["bbox-2"][0],  # x min
        props_image2["bbox-3"][0],  # z max
        props_image2["bbox-4"][0],  # y max
        props_image2["bbox-5"][0],  # x max
    )

    new_bbox1, new_bbox2 = new_crop_border(bbox_image1, bbox_image2, image_object1)

    cropped_image_1 = crop_3D_image(image_object1, new_bbox1)
    cropped_image_2 = crop_3D_image(image_object2, new_bbox2)
    return cropped_image_1, cropped_image_2


def measure_3D_colocalization(
    cropped_image_1: numpy.ndarray,
    cropped_image_2: numpy.ndarray,
    thr: int = 15,
    fast_costes: str = "Accurate",
) -> Dict[str, float]:
    """
    This function calculates the colocalization coefficients between two images.
    It computes the correlation coefficient, Manders' coefficients, overlap coefficient,
    and Costes' coefficients. The results are returned as a dictionary.

    Parameters
    ----------
    cropped_image_1 : numpy.ndarray
        The first cropped image.
    cropped_image_2 : numpy.ndarray
        The second cropped image.
    thr : int, optional
        The threshold for the Manders' coefficients, by default 15
    fast_costes : str, optional
        The mode for Costes' threshold calculation, by default "Accurate".
        Options are "Accurate" or "Fast".
        "Accurate" uses a linear algorithm, while "Fast" uses a bisection algorithm.
        The "Fast" mode is faster but less accurate.

    Returns
    -------
    Dict[str, float]
        The output features for colocalization analysis.
    """
    results = {}
    ################################################################################################
    # Calculate the correlation coefficient between the two images
    # This is the Pearson correlation coefficient
    # Pearson correlation coeffecient = cov(X, Y) / (std(X) * std(Y))
    # where cov(X, Y) is the covariance of X and Y
    # where X and Y are the two images
    # std(X) is the standard deviation of X
    # std(Y) is the standard deviation of Y
    # cov(X, Y) = sum((X - mean(X)) * (Y - mean(Y))) / (N - 1)
    # std(X) = sqrt(sum((X - mean(X)) ** 2) / (N - 1))
    # thus N -1 cancels out in the calculation below
    ################################################################################################
    mean1 = numpy.mean(cropped_image_1)
    mean2 = numpy.mean(cropped_image_2)
    std1 = numpy.sqrt(numpy.sum((cropped_image_1 - mean1) ** 2))
    std2 = numpy.sqrt(numpy.sum((cropped_image_2 - mean2) ** 2))
    x = cropped_image_1 - mean1  # x is not the same as the x dimension here
    y = cropped_image_2 - mean2  # y is not the same as the y dimension here
    corr = numpy.sum(x * y) / (std1 * std2)

    ################################################################################################
    # Calculate the Manders' coefficients
    ################################################################################################

    # Threshold as percentage of maximum intensity of objects in each channel
    try:
        tff = (thr / 100) * numpy.max(cropped_image_1)
        tss = (thr / 100) * numpy.max(cropped_image_2)
        # Ensure thresholds are at least 1 to avoid zero thresholding
        # if an errors occurs this is probably due to empty images
        # or images where the bbox is incredibly small and inconsistent
        # or the bbox is on the border of the image
        # in which case we want to remove anyway
    except ValueError:
        M1, M2 = 0.0, 0.0
    else:
        # get the thresholds
        combined_thresh = (cropped_image_1 >= tff) & (cropped_image_2 >= tss)

        first_image_thresh = cropped_image_1[combined_thresh]
        second_image_thresh = cropped_image_2[combined_thresh]

        tot_first_image_thr = scipy.ndimage.sum(
            cropped_image_1[cropped_image_1 >= tff],
        )
        tot_second_image_thr = scipy.ndimage.sum(
            cropped_image_2[cropped_image_2 >= tss]
        )

        if tot_first_image_thr > 0 and tot_second_image_thr > 0:
            M1 = scipy.ndimage.sum(first_image_thresh) / tot_first_image_thr
            M2 = scipy.ndimage.sum(second_image_thresh) / tot_second_image_thr
        else:
            M1, M2 = 0.0, 0.0
    ################################################################################################
    # Calculate the overlap coefficient
    ################################################################################################

    fpsq = scipy.ndimage.sum(
        cropped_image_1[combined_thresh] ** 2,
    )
    spsq = scipy.ndimage.sum(
        cropped_image_2[combined_thresh] ** 2,
    )
    pdt = numpy.sqrt(numpy.array(fpsq) * numpy.array(spsq))
    overlap = (
        scipy.ndimage.sum(
            cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
        )
        / pdt
    )
    K1 = scipy.ndimage.sum(
        cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
    ) / (numpy.array(fpsq))
    K2 = scipy.ndimage.sum(
        cropped_image_1[combined_thresh] * cropped_image_2[combined_thresh],
    ) / (numpy.array(spsq))

    # first_pixels, second_pixels = flattened image arrays
    # combined_thresh = boolean mask of pixels above threshold in both channels
    # fi_thresh, si_thresh = thresholded intensities (same shape as pixels)

    # --- Rank computation ---
    # Flatten images for ranking
    img1_flat = cropped_image_1.flatten()
    img2_flat = cropped_image_2.flatten()

    # --- Rank computation ---
    sorted_idx_1 = numpy.argsort(img1_flat)
    sorted_idx_2 = numpy.argsort(img2_flat)

    # Create rank arrays
    rank_1_flat = numpy.empty_like(sorted_idx_1, dtype=float)
    rank_2_flat = numpy.empty_like(sorted_idx_2, dtype=float)
    rank_1_flat[sorted_idx_1] = numpy.arange(len(sorted_idx_1))
    rank_2_flat[sorted_idx_2] = numpy.arange(len(sorted_idx_2))

    # Reshape back to original shape
    rank_im1 = rank_1_flat.reshape(cropped_image_1.shape)
    rank_im2 = rank_2_flat.reshape(cropped_image_2.shape)

    # --- Rank difference weight ---
    R = max(rank_im1.max(), rank_im2.max()) + 1
    Di = numpy.abs(rank_im1 - rank_im2)
    weight = (R - Di) / R

    # Get weights for thresholded pixels
    weight_thresh = weight[combined_thresh]

    # Get thresholded values (no double-thresholding!)
    first_image_thresh_final = first_image_thresh
    second_image_thresh_final = second_image_thresh

    # --- Calculate weighted colocalization ---
    if numpy.any(combined_thresh) and len(first_image_thresh_final) > 0:
        weighted_sum_1 = numpy.sum(first_image_thresh_final * weight_thresh)
        weighted_sum_2 = numpy.sum(second_image_thresh_final * weight_thresh)

        total_1 = numpy.sum(first_image_thresh_final)
        total_2 = numpy.sum(second_image_thresh_final)

        RWC1 = weighted_sum_1 / total_1 if total_1 > 0 else 0.0
        RWC2 = weighted_sum_2 / total_2 if total_2 > 0 else 0.0
    else:
        RWC1, RWC2 = 0.0, 0.0
    ################################################################################################
    # Calculate the Costes' coefficient
    ################################################################################################

    # Orthogonal Regression for Costes' automated threshold
    if numpy.max(cropped_image_1) > 255 or numpy.max(cropped_image_2) > 255:
        scale = 65535
    else:
        scale = 255

    if fast_costes == "Accurate":
        thr_first_image_c, thr_second_image_c = bisection_costes_threshold_calculation(
            cropped_image_1, cropped_image_2, scale
        )
    else:
        thr_first_image_c, thr_second_image_c = linear_costes_threshold_calculation(
            cropped_image_1, cropped_image_2, scale, fast_costes
        )

    # Costes' thershold for entire image is applied to each object
    first_image_above_thr = cropped_image_1 > thr_first_image_c
    second_image_above_thr = cropped_image_2 > thr_second_image_c
    combined_thresh_c = first_image_above_thr & second_image_above_thr
    first_image_thresh_c = cropped_image_1[combined_thresh_c]
    second_image_thresh_c = cropped_image_2[combined_thresh_c]

    tot_first_image_thr_c = scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= thr_first_image_c],
    )

    tot_second_image_thr_c = scipy.ndimage.sum(
        cropped_image_2[cropped_image_2 >= thr_second_image_c],
    )
    if tot_first_image_thr_c > 0 and tot_second_image_thr_c > 0:
        C1 = scipy.ndimage.sum(first_image_thresh_c) / tot_first_image_thr_c
        C2 = scipy.ndimage.sum(second_image_thresh_c) / tot_second_image_thr_c
    else:
        C1, C2 = 0.0, 0.0
    ################################################################################################
    # write the results to the output dictionary
    ################################################################################################

    results["Correlation"] = corr
    results["MandersCoeffM1"] = M1
    results["MandersCoeffM2"] = M2
    results["OverlapCoeff"] = overlap
    results["MandersCoeffCostesM1"] = C1
    results["MandersCoeffCostesM2"] = C2
    results["RankWeightedColocalizationCoeff1"] = RWC1
    results["RankWeightedColocalizationCoeff2"] = RWC2

    return results
