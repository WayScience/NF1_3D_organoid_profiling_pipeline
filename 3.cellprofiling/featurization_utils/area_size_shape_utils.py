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

    volume = label_object[
        max(props["bbox-0"][0], 0) : min(props["bbox-3"][0], label_object.shape[0]),
        max(props["bbox-1"][0], 0) : min(props["bbox-4"][0], label_object.shape[1]),
        max(props["bbox-2"][0], 0) : min(props["bbox-5"][0], label_object.shape[2]),
    ]
    volume_truths = volume > 0
    verts, faces, _normals, _values = skimage.measure.marching_cubes(
        volume_truths,
        method="lewiner",
        spacing=spacing,
        level=0,
    )
    surface_area = skimage.measure.mesh_surface_area(verts, faces)

    return surface_area


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
    label_object = object_loader.label_image
    spacing = image_set_loader.anisotropy_spacing
    unique_objects = object_loader.object_ids

    features_to_record = {
        "object_id": [],
        "VOLUME": [],
        "CENTER.X": [],
        "CENTER.Y": [],
        "CENTER.Z": [],
        "BBOX.VOLUME": [],
        "MIN.X": [],
        "MAX.X": [],
        "MIN.Y": [],
        "MAX.Y": [],
        "MIN.Z": [],
        "MAX.Z": [],
        "EXTENT": [],
        "EULER.NUMBER": [],
        "EQUIVALENT.DIAMETER": [],
        "SURFACE.AREA": [],
    }

    desired_properties = [
        "area",
        "bbox",
        "centroid",
        "bbox_area",
        "extent",
        "euler_number",
        "equivalent_diameter",
    ]
    for label in unique_objects:
        if label == 0:
            continue
        subset_lab_object = label_object.copy()
        subset_lab_object[subset_lab_object != label] = 0
        props = skimage.measure.regionprops_table(
            subset_lab_object, properties=desired_properties
        )

        features_to_record["object_id"].append(label)
        features_to_record["VOLUME"].append(props["area"].item())
        features_to_record["CENTER.X"].append(props["centroid-2"].item())
        features_to_record["CENTER.Y"].append(props["centroid-1"].item())
        features_to_record["CENTER.Z"].append(props["centroid-0"].item())
        features_to_record["BBOX.VOLUME"].append(props["bbox_area"].item())
        features_to_record["MIN.X"].append(props["bbox-2"].item())
        features_to_record["MAX.X"].append(props["bbox-5"].item())
        features_to_record["MIN.Y"].append(props["bbox-1"].item())
        features_to_record["MAX.Y"].append(props["bbox-4"].item())
        features_to_record["MIN.Z"].append(props["bbox-0"].item())
        features_to_record["MAX.Z"].append(props["bbox-3"].item())
        features_to_record["EXTENT"].append(props["extent"].item())
        features_to_record["EULER.NUMBER"].append(props["euler_number"].item())
        features_to_record["EQUIVALENT.DIAMETER"].append(
            props["equivalent_diameter"].item()
        )

        try:
            features_to_record["SURFACE.AREA"].append(
                calulate_surface_area(
                    label_object=label_object,
                    props=props,
                    spacing=spacing,
                )
            )
        except:
            features_to_record["SURFACE.AREA"].append(numpy.nan)
    return features_to_record
