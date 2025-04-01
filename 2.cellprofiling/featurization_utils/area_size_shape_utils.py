import numpy
import skimage.measure
from loading_classes import ImageSetLoader, ObjectLoader


def calulate_surface_area(
    label_object: numpy.array,
    props: numpy.array,
    spacing: tuple,
) -> list:
    """
    This function calculates the surface area of each object in a 3D image using the marching cubes algorithm.
    Parameters
    ----------
    label_object : numpy.array
        This is an array of the segmented objects of a given compartment.
    props : numpy.array
        This is the output of the regionprops function, which contains information about the objects.
    spacing : tuple
        This is the spacing of the image in each dimension (z, y, x).

    Returns
    -------
    list
        A list of surface areas for each object in the image.
    """

    # this seems less elegant than you might wish, given that regionprops returns a slice,
    # but we need to expand the slice out by one voxel in each direction, or surface area freaks out
    surface_areas = []
    for index, label in enumerate(props["label"]):

        volume = label_object[
            max(props["bbox-0"][index] - 1, 0) : min(
                props["bbox-3"][index] + 1, label_object.shape[0]
            ),
            max(props["bbox-1"][index] - 1, 0) : min(
                props["bbox-4"][index] + 1, label_object.shape[1]
            ),
            max(props["bbox-2"][index] - 1, 0) : min(
                props["bbox-5"][index] + 1, label_object.shape[2]
            ),
        ]
        volume_truths = volume > 0
        verts, faces, _normals, _values = skimage.measure.marching_cubes(
            volume_truths,
            method="lewiner",
            spacing=spacing,
            level=0,
        )
        surface_areas.append(skimage.measure.mesh_surface_area(verts, faces))

    return surface_areas


def measure_3D_area_size_shape(
    image_set_loader: ImageSetLoader,
    object_loader: ObjectLoader,
) -> dict:
    """
    This function calculates the area, size, and shape of objects in a 3D image using the regionprops function.
    It uses the numpy library to perform the calculations on the CPU.

    Parameters
    ----------
    image_set_loader : ImageSetLoader
        The image set loader object that contains the image and label image.
    object_loader : ObjectLoader
        The object loader object that contains the image and label image.

    Returns
    -------
    dict
        A dictionary containing the area, size, and shape of the objects in the image.
    """
    label_object = object_loader.objects
    spacing = image_set_loader.spacing
    unique_objects = object_loader.object_ids
    desired_properties = [
        "area",
        "bbox",
        "centroid",
        "bbox_area",
        "extent",
        "euler_number",
        "equivalent_diameter",
    ]

    props = skimage.measure.regionprops_table(
        label_object, properties=desired_properties
    )
    props["label"] = unique_objects
    features_to_record = {
        "object_id": props["label"],
        "VOLUME": props["area"],
        "CENTER.X": props["centroid-2"],
        "CENTER.Y": props["centroid-1"],
        "CENTER.Z": props["centroid-0"],
        "BBOX.VOLUME": props["bbox_area"],
        "MIN.X": props["bbox-2"],
        "MAX.X": props["bbox-5"],
        "MIN.Y": props["bbox-1"],
        "MAX.Y": props["bbox-4"],
        "MIN.Z": props["bbox-0"],
        "MAX.Z": props["bbox-3"],
        "EXTENT": props["extent"],
        "EULER.NUMBER": props["euler_number"],
        "EQUIVALENT.DIAMETER": props["equivalent_diameter"],
    }
    try:
        features_to_record["SURFACE.AREA"] = calulate_surface_area(
            label_object=label_object,
            props=props,
            spacing=spacing,
        )
    except:
        features_to_record["SURFACE.AREA"] = numpy.nan
    return features_to_record
