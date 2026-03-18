from typing import Tuple, Union

import numpy


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


def single_3D_image_expand_bbox(
    image: numpy.ndarray,
    bbox: tuple[int, int, int, int, int, int],
    expand_pixels: int,
    anisotropy_factor: int,
) -> tuple[int, int, int, int, int, int]:
    """
    Expand the bbox in a way that keeps the crop within the
    confines of the image volume

    Parameters
    ----------
    image : numpy.ndarray
        3D image array from which the bbox was derived
    bbox : tuple[int, int, int, int, int, int]
        3D bbox in the format (zmin, ymin, xmin, zmax, ymax, xmax)
    expand_pixels : int
        number of pixels to expand the bbox in each direction (z, y, x)
        the coordinates become isotropic here so the expansion is the same across dimensions,
        but the anisotropy factor is used to adjust for the z dimension
    anisotropy_factor : int
        The ratio of "pixel" size in um between the z dimension and the x/y dimensions.
        This is used to adjust the expansion of the bbox in the z dimension to account
        for anisotropy in the image volume.
        For example, if the z spacing is 5um and the x/y spacing is 1um,
        then the anisotropy factor would be 5.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        Updated bbox in the format (zmin, ymin, xmin, zmax, ymax, xmax)
        after expansion and adjustment for anisotropy
    """
    z1, y1, x1, z2, y2, x2 = bbox
    zmin, ymin, xmin = 0, 0, 0
    zmax, ymax, xmax = image.shape
    # adjust the anisotropy factor for the z dimension
    z1, z2 = z1 * anisotropy_factor, z2 * anisotropy_factor
    zmax = zmax * anisotropy_factor
    # expand the bbox by the specified number of pixels in each direction
    z1_expanded = z1 - expand_pixels
    y1_expanded = y1 - expand_pixels
    x1_expanded = x1 - expand_pixels
    z2_expanded = z2 + expand_pixels
    y2_expanded = y2 + expand_pixels
    x2_expanded = x2 + expand_pixels
    # convert the expanded bbox back to the original z dimension scale
    z1_expanded = numpy.floor(z1_expanded / anisotropy_factor)
    z2_expanded = numpy.ceil(z2_expanded / anisotropy_factor)
    # ensure the expanded bbox does not go outside the image boundaries
    z1_expanded, z2_expanded = (
        max(z1_expanded, numpy.floor(zmin / anisotropy_factor)).astype(int),
        min(z2_expanded, numpy.ceil(zmax / anisotropy_factor)).astype(int),
    )
    y1_expanded, y2_expanded = max(y1_expanded, ymin), min(y2_expanded, ymax)
    x1_expanded, x2_expanded = max(x1_expanded, xmin), min(x2_expanded, xmax)

    return (
        z1_expanded,
        y1_expanded,
        x1_expanded,
        z2_expanded,
        y2_expanded,
        x2_expanded,
    )


def check_for_xy_squareness(bbox: tuple[int, int, int, int, int, int]) -> float:
    """
    This function returns the ratio of the x length to the y length
    A value of 1 indicates a square bbox is present

    Parameters
    ----------
    bbox : The bbox to check
        (z_min, y_min, x_min, z_max, y_max, x_max)
        Where each value is an int representing the pixel coordinate of the bbox in that dimension

    Returns
    -------
    float
        The ratio of the y length to the x length of the bbox. A value of 1 indicates a square bbox.
    """
    z_min, y_min, x_min, z_max, y_max, x_max = bbox
    x_length = x_max - x_min
    if x_length == 0:
        raise ValueError(
            "Cannot compute xy squareness for bbox with zero width in x dimension "
            f"(bbox={bbox})."
        )
    xy_squareness = (y_max - y_min) / x_length
    return xy_squareness


def square_off_xy_crop_bbox(
    bbox: tuple[int, int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    """
    Adjust the bbox to be square in the XY plane.

    The function computes the new bbox from the current X/Y dimensions.

    Parameters
    ----------
    bbox : tuple[int, int, int, int, int, int]
        The bbox to adjust:
        (z_min, y_min, x_min, z_max, y_max, x_max)

        Each value is an integer pixel coordinate in that dimension.

    Returns
    -------
    tuple[int, int, int, int, int, int]
        The adjusted bbox that is square in the XY plane:
        (z_min, new_y_min, new_x_min, z_max, new_y_max, new_x_max)

        Each value is an integer pixel coordinate in that dimension.
    """
    zmin, ymin, xmin, zmax, ymax, xmax = bbox
    # first find the larger dimension between x and y
    x_size = xmax - xmin
    y_size = ymax - ymin
    if x_size > y_size:
        # need to expand y dimension
        new_ymin = int(ymin - (x_size - y_size) / 2)
        new_ymax = int(ymax + (x_size - y_size) / 2)
        return (zmin, new_ymin, xmin, zmax, new_ymax, xmax)
    elif y_size > x_size:
        # need to expand x dimension
        new_xmin = int(xmin - (y_size - x_size) / 2)
        new_xmax = int(xmax + (y_size - x_size) / 2)
        return (zmin, ymin, new_xmin, zmax, ymax, new_xmax)
    else:
        # already square
        return bbox
