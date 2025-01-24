import numpy as np


def euclidian_2D_distance(coord_set_1: tuple, coord_set_2: tuple) -> float:
    """
    This function calculates the euclidian distance between two sets of coordinates (2D)

    sqrt((x1 - x2)^2 + (y1 - y2)^2)

    Parameters
    ----------
    coord_set_1 : tuple
        The first set of coordinates (x, y)
    coord_set_2 : tuple
        The second set of coordinates (x, y)

    Returns
    -------
    float
        The euclidian distance between the two sets of coordinates
    """
    return np.sqrt(
        (coord_set_1[0] - coord_set_2[0]) ** 2 + (coord_set_1[1] - coord_set_2[1]) ** 2
    )


def check_coordinate_inside_box(
    coord: tuple,
    box: tuple,
) -> bool:
    """
    This function checks if a coordinate is inside a box

    Parameters
    ----------
    coord : tuple
        The coordinate to check (y, x)
    box : tuple
        The box to check against [y_min, x_min, y_max, x_max]

    Returns
    -------
    bool
        True if the coordinate is inside the box, False otherwise
    """
    # check if coords and box are valid
    if not isinstance(coord, tuple):
        raise TypeError("coord must be a tuple")
    if not isinstance(box, tuple):
        raise TypeError("box must be a list")
    if not len(box) == 4:
        raise ValueError("box must be a list of length 4")
    if not len(coord) == 2:
        raise ValueError("coord must be a tuple of length 2")

    y_coord = coord[0]
    x_coord = coord[1]

    y_min = box[0]
    x_min = box[1]
    y_max = box[2]
    x_max = box[3]

    if x_coord >= x_min and x_coord <= x_max and y_coord >= y_min and y_coord <= y_max:
        return True
    else:
        return False


def get_larger_bbox(bbox1: tuple, bbox2: tuple) -> tuple:
    """
    This function returns the larger of two bounding boxes

    Parameters
    ----------
    bbox1 : tuple
        The first bounding box [y_min, x_min, y_max, x_max]
    bbox2 : tuple
        The second bounding box [y_min, x_min, y_max, x_max]

    Returns
    -------
    tuple
        A tuple of the larger bounding box [y_min, x_min, y_max, x_max]
    """
    # check if boxes are valid
    if not isinstance(bbox1, tuple):
        raise TypeError("bbox1 must be a tuple")
    if not isinstance(bbox2, tuple):
        raise TypeError("bbox2 must be a tuple")
    if not len(bbox1) == 4:
        raise ValueError("bbox1 must be a list of length 4")
    if not len(bbox2) == 4:
        raise ValueError("bbox2 must be a list of length 4")

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if bbox1_area >= bbox2_area:
        return bbox1
    elif bbox2_area >= bbox1_area:
        return bbox2
