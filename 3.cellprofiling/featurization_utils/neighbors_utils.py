from typing import Dict, Tuple, Union

import numpy
import skimage.measure
from loading_classes import ObjectLoader


def neighbors_expand_box(
    min_coor: Union[int, float],
    max_coord: Union[int, float],
    current_min: Union[int, float],
    current_max: Union[int, float],
    expand_by: int,
) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Expand the bounding box of the object by a specified distance in each direction.

    Parameters
    ----------
    min_coor : Union[int, float]
        The global minimum coordinate of the image.
    max_coord : Union[int, float]
        The global maximum coordinate of the image.
    current_min : Union[int, float]
        The current minimum coordinate of the object.
    current_max : Union[int, float]
        The current maximum coordinate of the object.
    expand_by : int
        The distance by which to expand the bounding box.

    Returns
    -------
    Tuple[Union[int, float], Union[int, float]]
        The new minimum and maximum coordinates of the bounding box.
    """
    if current_min - expand_by < min_coor:
        current_min = min_coor
    else:
        current_min -= expand_by
    if current_max + expand_by > max_coord:
        current_max = max_coord
    else:
        current_max += expand_by
    return current_min, current_max


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
    Crop the 3D image to the bounding box of the object.

    Parameters
    ----------
    image : numpy.ndarray
        The 3D image to be cropped.
    bbox : Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float], Union[int, float]]
        The bounding box of the object in the format (z1, y1, x1, z2, y2, x2).

    Returns
    -------
    numpy.ndarray
        The cropped 3D image.
    """
    z1, y1, x1, z2, y2, x2 = bbox
    return image[z1:z2, y1:y2, x1:x2]


def measure_3D_number_of_neighbors(
    object_loader: ObjectLoader,
    distance_threshold: int = 10,
    anisotropy_factor: int = 10,
) -> Dict[str, list]:
    """
    This function calculates the number of neighbors for each object in a 3D image.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader object that contains the image and label image.
    distance_threshold : int, optional
        The distance threshold for counting neighbors, by default 10
    anisotropy_factor : int, optional
        The anisotropy factor for the image where the anisotropy factor is the ratio of the pixel size in the z direction to the pixel size in the x and y directions, by default 10

    Returns
    -------
    Dict[str, list]
        A dictionary containing the object ID and the number of neighbors for each object.
    """
    label_object = object_loader.objects
    labels = object_loader.object_ids
    # set image global min and max coordinates
    image_global_min_coord_z = 0
    image_global_min_coord_y = 0
    image_global_min_coord_x = 0
    image_global_max_coord_z = label_object.shape[0]
    image_global_max_coord_y = label_object.shape[1]
    image_global_max_coord_x = label_object.shape[2]

    neighbors_out_dict = {
        "object_id": [],
        "Neighbors_adjacent": [],
        f"Neighbors_{distance_threshold}": [],
    }
    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_label_object[selected_label_object != label] = 0
        props_label = skimage.measure.regionprops_table(
            selected_label_object, properties=["bbox"]
        )
        # get the number of neighbors for each object
        distance_x_y = distance_threshold
        distance_z = numpy.ceil(distance_threshold / anisotropy_factor).astype(int)
        # find how many other indexes are within a specified distance of the object
        # first expand the mask image by a specified distance
        z_min, y_min, x_min, z_max, y_max, x_max = (
            props_label["bbox-0"][0],
            props_label["bbox-1"][0],
            props_label["bbox-2"][0],
            props_label["bbox-3"][0],
            props_label["bbox-4"][0],
            props_label["bbox-5"][0],
        )
        original_bbox = (z_min, y_min, x_min, z_max, y_max, x_max)

        new_z_min, new_z_max = neighbors_expand_box(
            min_coor=image_global_min_coord_z,
            max_coord=image_global_max_coord_z,
            current_min=z_min,
            current_max=z_max,
            expand_by=distance_z,
        )
        new_y_min, new_y_max = neighbors_expand_box(
            min_coor=image_global_min_coord_y,
            max_coord=image_global_max_coord_y,
            current_min=y_min,
            current_max=y_max,
            expand_by=distance_x_y,
        )
        new_x_min, new_x_max = neighbors_expand_box(
            min_coor=image_global_min_coord_x,
            max_coord=image_global_max_coord_x,
            current_min=x_min,
            current_max=x_max,
            expand_by=distance_x_y,
        )
        bbox = (new_z_min, new_y_min, new_x_min, new_z_max, new_y_max, new_x_max)
        croppped_neighbor_image = crop_3D_image(image=label_object, bbox=bbox)
        self_cropped_neighbor_image = crop_3D_image(
            image=label_object, bbox=original_bbox
        )
        # find all the unique values in the cropped image of the object of interest
        # this is the number of neighbors in the cropped image
        n_neighbors_adjacent = (
            len(
                numpy.unique(
                    self_cropped_neighbor_image[self_cropped_neighbor_image > 0]
                )
            )
            - 1
        )

        # find all the unique values in the expanded cropped image of the object of interest
        # this gives the number of neighbors in a n distance of the object
        n_neighbors_by_distance = (
            len(numpy.unique(croppped_neighbor_image[croppped_neighbor_image > 0])) - 1
        )
        neighbors_out_dict["object_id"].append(label)
        neighbors_out_dict["Neighbors_adjacent"].append(n_neighbors_adjacent)
        neighbors_out_dict[f"Neighbors_{distance_threshold}"].append(
            n_neighbors_by_distance
        )

    return neighbors_out_dict
