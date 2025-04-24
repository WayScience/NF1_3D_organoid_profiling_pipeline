from typing import Dict, Tuple, Union

import numpy
import scipy.ndimage
import skimage


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


def select_objects_from_label(
    label_image: numpy.ndarray, object_ids: list
) -> numpy.ndarray:
    """
    Selects objects from a label image based on the provided object IDs.

    Parameters
    ----------
    label_image : numpy.ndarray
        The segmented label image.
    object_ids : list
        The object IDs to select.

    Returns
    -------
    numpy.ndarray
        The label image with only the selected objects.
    """
    label_image = label_image.copy()
    label_image[label_image != object_ids] = 0
    return label_image


def expand_box(
    min_coor: int, max_coord: int, current_min: int, current_max: int, expand_by: int
) -> Union[Tuple[int, int], ValueError]:
    """
    Expand the bounding box of an object in a 3D image.

    Parameters
    ----------
    min_coor : int
        The minimum coordinate of the image for any dimension.
    max_coord : int
        The maximum coordinate of the image for any dimension.
    current_min : int
        The current minimum coordinate of the bounding box of an object for any dimension.
    current_max : int
        The current maximum coordinate of the bounding box of an object for any dimension.
    expand_by : int
        The amount to expand the bounding box by.

    Returns
    -------
    Union[Tuple[int, int], ValueError]
        The new minimum and maximum coordinates of the bounding box.
        Raises ValueError if the expansion is not possible.
    """

    if max_coord - min_coor - (current_max - current_min) < expand_by:
        return ValueError("Cannot expand box by the requested amount")
    while expand_by > 0:
        if current_min > min_coor:
            current_min -= 1
            expand_by -= 1
        elif current_max < max_coord:
            current_max += 1
            expand_by -= 1

    return current_min, current_max


def new_crop_border(
    bbox1: Tuple[
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
    ],
    bbox2: Tuple[
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
    ],
    image: numpy.ndarray,
) -> Tuple[
    Tuple[
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
    ],
    Tuple[
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
    ],
]:
    """
    Expand the bounding boxes of two objects in a 3D image to match their sizes.

    Parameters
    ----------
    bbox1 : Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]]
        The bounding box of the first object.
    bbox2 : Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]]
        The bounding box of the second object.
    image : numpy.ndarray
        The image to crop for each of the bounding boxes.

    Returns
    -------
    Tuple[Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]], Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]]]
        The new bounding boxes of the two objects.
    Raises
    ValueError
        If the expansion is not possible.
    """
    i1z1, i1y1, i1x1, i1z2, i1y2, i1x2 = bbox1
    i2z1, i2y1, i2x1, i2z2, i2y2, i2x2 = bbox2
    z_range1 = i1z2 - i1z1
    y_range1 = i1y2 - i1y1
    x_range1 = i1x2 - i1x1
    z_range2 = i2z2 - i2z1
    y_range2 = i2y2 - i2y1
    x_range2 = i2x2 - i2x1
    z_diff = numpy.abs(z_range1 - z_range2)
    y_diff = numpy.abs(y_range1 - y_range2)
    x_diff = numpy.abs(x_range1 - x_range2)
    min_z_coord = 0
    max_z_coord = image.shape[0]
    min_y_coord = 0
    max_y_coord = image.shape[1]
    min_x_coord = 0
    max_x_coord = image.shape[2]
    if z_range1 < z_range2:
        i1z1, i1z2 = expand_box(
            min_coor=min_z_coord,
            max_coord=max_z_coord,
            current_min=i1z1,
            current_max=i1z2,
            expand_by=z_diff,
        )
    elif z_range1 > z_range2:
        i2z1, i2z2 = expand_box(
            min_coor=min_z_coord,
            max_coord=max_z_coord,
            current_min=i2z1,
            current_max=i2z2,
            expand_by=z_diff,
        )
    if y_range1 < y_range2:
        i1y1, i1y2 = expand_box(
            min_coor=min_y_coord,
            max_coord=max_y_coord,
            current_min=i1y1,
            current_max=i1y2,
            expand_by=y_diff,
        )
    elif y_range1 > y_range2:
        i2y1, i2y2 = expand_box(
            min_coor=min_y_coord,
            max_coord=max_y_coord,
            current_min=i2y1,
            current_max=i2y2,
            expand_by=y_diff,
        )
    if x_range1 < x_range2:
        i1x1, i1x2 = expand_box(
            min_coor=min_x_coord,
            max_coord=max_x_coord,
            current_min=i1x1,
            current_max=i1x2,
            expand_by=x_diff,
        )
    elif x_range1 > x_range2:
        i2x1, i2x2 = expand_box(
            min_coor=min_x_coord,
            max_coord=max_x_coord,
            current_min=i2x1,
            current_max=i2x2,
            expand_by=x_diff,
        )
    return (i1z1, i1y1, i1x1, i1z2, i1y2, i1x2), (i2z1, i2y1, i2x1, i2z2, i2y2, i2x2)


# crop the image to the bbox of the mask
def crop_3D_image(
    image: numpy.ndarray,
    bbox: Tuple[
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
        Union[int, float],
    ],
) -> numpy.ndarray:
    """
    Crop a 3D image to the bounding box of a mask.

    Parameters
    ----------
    image : numpy.ndarray
        The image to crop.
    bbox : Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]]
        The bounding box of the mask.

    Returns
    -------
    numpy.ndarray
        The cropped image.
    """
    z1, y1, x1, z2, y2, x2 = bbox
    return image[z1:z2, y1:y2, x1:x2]


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
    thr = 15
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
    mean1 = scipy.ndimage.mean(cropped_image_1, 1)
    mean2 = scipy.ndimage.mean(cropped_image_2, 1)
    std1 = numpy.sqrt(scipy.ndimage.sum((cropped_image_1 - mean1) ** 2))
    std2 = numpy.sqrt(scipy.ndimage.sum((cropped_image_2 - mean2) ** 2))
    x = cropped_image_1 - mean1  # x is not the same as the x dimension here
    y = cropped_image_2 - mean2  # y is not the same as the y dimension here
    corr = scipy.ndimage.sum(x * y / (std1 * std2))

    ################################################################################################
    # Calculate the Manders' coefficients
    ################################################################################################

    # Threshold as percentage of maximum intensity of objects in each channel
    tff = (thr / 100) * scipy.ndimage.maximum(cropped_image_1)
    tss = (thr / 100) * scipy.ndimage.maximum(cropped_image_2)

    combined_thresh = (cropped_image_1 >= tff) & (cropped_image_2 >= tss)

    first_image_thresh = cropped_image_1[combined_thresh]
    second_image_thresh = cropped_image_2[combined_thresh]
    tot_first_image_thr = scipy.ndimage.sum(
        cropped_image_1[cropped_image_1 >= tff],
    )
    tot_second_image_thr = scipy.ndimage.sum(cropped_image_2[cropped_image_2 >= tss])

    M1 = scipy.ndimage.sum(first_image_thresh) / numpy.array(tot_first_image_thr)
    M2 = scipy.ndimage.sum(second_image_thresh) / numpy.array(tot_second_image_thr)

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
    C1 = scipy.ndimage.sum(first_image_thresh_c) / numpy.array(tot_first_image_thr_c)
    C2 = scipy.ndimage.sum(second_image_thresh_c) / numpy.array(tot_second_image_thr_c)

    ################################################################################################
    # write the results to the output dictionary
    ################################################################################################

    results["MEAN.CORRELATION.COEFF"] = numpy.mean(corr)
    results["MEDIAN.CORRELATION.COEFF"] = numpy.median(corr)
    results["MIN.CORRELATION.COEFF"] = numpy.min(corr)
    results["MAX.CORRELATION.COEFF"] = numpy.max(corr)
    results["MEAN.MANDERS.COEFF.M1"] = numpy.mean(M1)
    results["MEDIAN.MANDERS.COEFF.M1"] = numpy.median(M1)
    results["MIN.MANDERS.COEFF.M1"] = numpy.min(M1)
    results["MAX.MANDERS.COEFF.M1"] = numpy.max(M1)
    results["MEAN.MANDERS.COEFF.M2"] = numpy.mean(M2)
    results["MEDIAN.MANDERS.COEFF.M2"] = numpy.median(M2)
    results["MIN.MANDERS.COEFF.M2"] = numpy.min(M2)
    results["MAX.MANDERS.COEFF.M2"] = numpy.max(M2)
    results["MEAN.OVERLAP.COEFF"] = numpy.mean(overlap)
    results["MEDIAN.OVERLAP.COEFF"] = numpy.median(overlap)
    results["MIN.OVERLAP.COEFF"] = numpy.min(overlap)
    results["MAX.OVERLAP.COEFF"] = numpy.max(overlap)
    results["MEAN.K1"] = numpy.mean(K1)
    results["MEDIAN.K1"] = numpy.median(K1)
    results["MIN.K1"] = numpy.min(K1)
    results["MAX.K1"] = numpy.max(K1)
    results["MEAN.K2"] = numpy.mean(K2)
    results["MEDIAN.K2"] = numpy.median(K2)
    results["MIN.K2"] = numpy.min(K2)
    results["MAX.K2"] = numpy.max(K2)
    results["MEAN.MANDERS.COEFF.COSTES.M1"] = numpy.mean(C1)
    results["MEDIAN.MANDERS.COEFF.COSTES.M1"] = numpy.median(C1)
    results["MIN.MANDERS.COEFF.COSTES.M1"] = numpy.min(C1)
    results["MAX.MANDERS.COEFF.COSTES.M1"] = numpy.max(C1)
    results["MEAN.MANDERS.COEFF.COSTES.M2"] = numpy.mean(C2)
    results["MEDIAN.MANDERS.COEFF.COSTES.M2"] = numpy.median(C2)
    results["MIN.MANDERS.COEFF.COSTES.M2"] = numpy.min(C2)
    results["MAX.MANDERS.COEFF.COSTES.M2"] = numpy.max(C2)
    return results
