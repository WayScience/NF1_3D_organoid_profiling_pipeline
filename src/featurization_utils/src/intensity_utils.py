import cucim.skimage.measure
import cupy
import cupyx
import cupyx.scipy.ndimage
import numpy
import scipy.ndimage
import skimage.segmentation
from loading_classes import ObjectLoader


def get_outline(mask: numpy.ndarray) -> numpy.ndarray:
    """
    Get the outline of a 3D mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The input mask.

    Returns
    -------
    numpy.ndarray
        The outline of the mask.
    """
    outline = numpy.zeros_like(mask)
    for z in range(mask.shape[0]):
        outline[z] = skimage.segmentation.find_boundaries(mask[z])
    return outline


def measure_3D_intensity_CPU(
    object_loader: ObjectLoader,
) -> dict:
    """
    Measure the intensity of objects in a 3D image.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader containing the image and label image.

    Returns
    -------
    dict
        A dictionary containing the measurements for each object.
        The keys are the measurement names and the values are the corresponding values.
    """
    image_object = object_loader.image
    label_object = object_loader.label_image
    labels = object_loader.object_ids
    ranges = len(labels)

    output_dict = {
        "object_id": [],
        "feature_name": [],
        "channel": [],
        "compartment": [],
        "value": [],
    }
    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_image_object = image_object.copy()

        selected_label_object[selected_label_object != label] = 0
        selected_label_object[selected_label_object > 0] = (
            1  # binarize the label for volume calcs
        )
        selected_image_object[selected_label_object != 1] = 0
        non_zero_pixels_object = selected_image_object[selected_image_object > 0]
        if non_zero_pixels_object.size == 0:
            non_zero_pixels_object = numpy.array([0], dtype=numpy.float32)
        mask_outlines = get_outline(selected_label_object)

        # Extract only coordinates where object exists
        z_indices, y_indices, x_indices = numpy.where(selected_label_object > 0)
        min_z, max_z = numpy.min(z_indices), numpy.max(z_indices)
        min_y, max_y = numpy.min(y_indices), numpy.max(y_indices)
        min_x, max_x = numpy.min(x_indices), numpy.max(x_indices)

        # Create coordinate grids only for the bounding box
        mesh_z, mesh_y, mesh_x = numpy.mgrid[
            min_z : max_z + 1,  # + 1 to include the max index
            min_y : max_y + 1,
            min_x : max_x + 1,
        ]

        # calculate the integrated intensity
        integrated_intensity = scipy.ndimage.sum(
            selected_image_object,
            selected_label_object,
        )
        # calculate the volume
        volume = numpy.sum(selected_label_object)

        # calculate the mean intensity
        mean_intensity = integrated_intensity / volume
        # calculate the standard deviation
        std_intensity = numpy.std(non_zero_pixels_object)
        # min intensity
        min_intensity = numpy.min(non_zero_pixels_object)
        # max intensity
        max_intensity = numpy.max(non_zero_pixels_object)
        # lower quartile
        lower_quartile_intensity = numpy.percentile(non_zero_pixels_object, 25)
        # upper quartile
        upper_quartile_intensity = numpy.percentile(non_zero_pixels_object, 75)
        # median intensity
        median_intensity = numpy.median(non_zero_pixels_object)
        # max intensity location
        max_z, max_y, max_x = scipy.ndimage.maximum_position(
            selected_image_object,
        )  # z, y, x
        cm_x = scipy.ndimage.mean(mesh_x)
        cm_y = scipy.ndimage.mean(mesh_y)
        cm_z = scipy.ndimage.mean(mesh_z)
        i_x = scipy.ndimage.sum(mesh_x)
        i_y = scipy.ndimage.sum(mesh_y)
        i_z = scipy.ndimage.sum(mesh_z)
        # calculate the center of mass
        cmi_x = i_x / integrated_intensity
        cmi_y = i_y / integrated_intensity
        cmi_z = i_z / integrated_intensity
        # calculate the center of mass distance
        diff_x = cm_x - cmi_x
        diff_y = cm_y - cmi_y
        diff_z = cm_z - cmi_z
        # mass displacement
        mass_displacement = numpy.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        # mean aboslute deviation
        mad_intensity = numpy.mean(numpy.abs(non_zero_pixels_object - mean_intensity))
        edge_count = scipy.ndimage.sum(mask_outlines)
        integrated_intensity_edge = numpy.sum(selected_image_object[mask_outlines > 0])
        mean_intensity_edge = integrated_intensity_edge / edge_count
        std_intensity_edge = numpy.std(selected_image_object[mask_outlines > 0])
        min_intensity_edge = numpy.min(selected_image_object[mask_outlines > 0])
        max_intensity_edge = numpy.max(selected_image_object[mask_outlines > 0])
        measurements_dict = {
            "INTEGRATED.INTENSITY": integrated_intensity,
            "VOLUME": volume,
            "MEAN.INTENSITY": mean_intensity,
            "STD.INTENSITY": std_intensity,
            "MIN.INTENSITY": min_intensity,
            "MAX.INTENSITY": max_intensity,
            "LOWER.QUARTILE.INTENSITY": lower_quartile_intensity,
            "UPPER.QUARTILE.INTENSITY": upper_quartile_intensity,
            "MEDIAN.INTENSITY": median_intensity,
            "MAX.Z": max_z,
            "MAX.Y": max_y,
            "MAX.X": max_x,
            "CM.X": cm_x,
            "CM.Y": cm_y,
            "CM.Z": cm_z,
            "I.X": i_x,
            "I.Y": i_y,
            "I.Z": i_z,
            "CMI.X": cmi_x,
            "CMI.Y": cmi_y,
            "CMI.Z": cmi_z,
            "DIFF.X": diff_x,
            "DIFF.Y": diff_y,
            "DIFF.Z": diff_z,
            "MASS.DISPLACEMENT": mass_displacement,
            "MAD.INTENSITY": mad_intensity,
            "EDGE.COUNT": edge_count,
            "INTEGRATED.INTENSITY.EDGE": integrated_intensity_edge,
            "MEAN.INTENSITY.EDGE": mean_intensity_edge,
            "STD.INTENSITY.EDGE": std_intensity_edge,
            "MIN.INTENSITY.EDGE": min_intensity_edge,
            "MAX.INTENSITY.EDGE": max_intensity_edge,
        }

        for feature_name, value in measurements_dict.items():
            if value.dtype != numpy.float32:
                value = numpy.float32(value)
            output_dict["object_id"].append(numpy.int32(label))
            output_dict["feature_name"].append(feature_name)
            output_dict["channel"].append(object_loader.channel)
            output_dict["compartment"].append(object_loader.compartment)
            output_dict["value"].append(value)
    return output_dict


def measure_3D_intensity_gpu(
    object_loader: ObjectLoader,
) -> dict:
    """
    Measure the intensity of objects in a 3D image using GPU acceleration.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader containing the image and label image.

    Returns
    -------
    dict
        A dictionary containing the measurements for each object.
    """
    image_object = object_loader.image
    label_object = object_loader.label_image
    labels = object_loader.object_ids
    ranges = len(labels)

    image_object = cupy.asarray(image_object)
    label_object = cupy.asarray(label_object)
    labels = cupy.asarray(labels)

    output_dict = {
        "object_id": [],
        "feature_name": [],
        "channel": [],
        "compartment": [],
        "value": [],
    }
    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_image_object = image_object.copy()
        selected_label_object[selected_label_object != label] = 0
        selected_label_object[selected_label_object > 0] = (
            1  # binarize the label for volume calcs
        )
        selected_image_object[selected_label_object != 1] = 0
        non_zero_pixels_object = selected_image_object[selected_image_object > 0]
        mask_outlines = get_outline(selected_label_object.get())
        mask_outlines = cupy.asarray(mask_outlines)

        # Extract only coordinates where object exists
        z_indices, y_indices, x_indices = cupy.where(selected_label_object > 0)
        min_z, max_z = cupy.min(z_indices), cupy.max(z_indices)
        min_y, max_y = cupy.min(y_indices), cupy.max(y_indices)
        min_x, max_x = cupy.min(x_indices), cupy.max(x_indices)

        # Create coordinate grids only for the bounding box
        mesh_z, mesh_y, mesh_x = cupy.mgrid[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ]

        # calculate the integrated intensity
        integrated_intensity = cupyx.scipy.ndimage.sum(
            selected_image_object,
            selected_label_object,
        )
        # calculate the volume
        volume = cupy.sum(selected_label_object)

        # calculate the mean intensity
        mean_intensity = integrated_intensity / volume
        # calculate the standard deviation
        std_intensity = cupy.std(non_zero_pixels_object)
        # min intensity
        min_intensity = cupy.min(non_zero_pixels_object)
        # max intensity
        max_intensity = cupy.max(non_zero_pixels_object)
        # lower quartile
        lower_quartile_intensity = cupy.percentile(non_zero_pixels_object, 25)
        # upper quartile
        upper_quartile_intensity = cupy.percentile(non_zero_pixels_object, 75)
        # median intensity
        median_intensity = cupy.median(non_zero_pixels_object)
        # max intensity location
        max_z, max_y, max_x = cupyx.scipy.ndimage.maximum_position(
            selected_image_object,
        )  # z, y, x
        cm_x = cupyx.scipy.ndimage.mean(mesh_x)
        cm_y = cupyx.scipy.ndimage.mean(mesh_y)
        cm_z = cupyx.scipy.ndimage.mean(mesh_z)
        i_x = cupyx.scipy.ndimage.sum(mesh_x)
        i_y = cupyx.scipy.ndimage.sum(mesh_y)
        i_z = cupyx.scipy.ndimage.sum(mesh_z)
        # calculate the center of mass
        cmi_x = i_x / integrated_intensity
        cmi_y = i_y / integrated_intensity
        cmi_z = i_z / integrated_intensity
        # calculate the center of mass distance
        diff_x = cm_x - cmi_x
        diff_y = cm_y - cmi_y
        diff_z = cm_z - cmi_z
        # mass displacement
        mass_displacement = cupy.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        # mean aboslute deviation
        mad_intensity = cupy.mean(cupy.abs(non_zero_pixels_object - mean_intensity))
        edge_count = cupyx.scipy.ndimage.sum(mask_outlines)
        integrated_intensity_edge = cupy.sum(selected_image_object[mask_outlines > 0])
        mean_intensity_edge = integrated_intensity_edge / edge_count
        std_intensity_edge = cupy.std(selected_image_object[mask_outlines > 0])
        min_intensity_edge = cupy.min(selected_image_object[mask_outlines > 0])
        max_intensity_edge = cupy.max(selected_image_object[mask_outlines > 0])
        measurements_dict = {
            "INTEGRATED.INTENSITY": integrated_intensity.get(),
            "VOLUME": volume.get(),
            "MEAN.INTENSITY": mean_intensity.get(),
            "STD.INTENSITY": std_intensity.get(),
            "MIN.INTENSITY": min_intensity.get(),
            "MAX.INTENSITY": max_intensity.get(),
            "LOWER.QUARTILE.INTENSITY": lower_quartile_intensity.get(),
            "UPPER.QUARTILE.INTENSITY": upper_quartile_intensity.get(),
            "MEDIAN.INTENSITY": median_intensity.get(),
            "MAX.Z": max_z.item(),
            "MAX.Y": max_y.item(),
            "MAX.X": max_x.item(),
            "CM.X": cm_x.get(),
            "CM.Y": cm_y.get(),
            "CM.Z": cm_z.get(),
            "I.X": i_x.get(),
            "I.Y": i_y.get(),
            "I.Z": i_z.get(),
            "CMI.X": cmi_x.get(),
            "CMI.Y": cmi_y.get(),
            "CMI.Z": cmi_z.get(),
            "DIFF.X": diff_x.get(),
            "DIFF.Y": diff_y.get(),
            "DIFF.Z": diff_z.get(),
            "MASS.DISPLACEMENT": mass_displacement.get(),
            "MAD.INTENSITY": mad_intensity.get(),
            "EDGE.COUNT": edge_count.get(),
            "INTEGRATED.INTENSITY.EDGE": integrated_intensity_edge.get(),
            "MEAN.INTENSITY.EDGE": mean_intensity_edge.get(),
            "STD.INTENSITY.EDGE": std_intensity_edge.get(),
            "MIN.INTENSITY.EDGE": min_intensity_edge.get(),
            "MAX.INTENSITY.EDGE": max_intensity_edge.get(),
        }

        for feature_name, value in measurements_dict.items():
            output_dict["object_id"].append(numpy.int32(label.get().item()))
            output_dict["feature_name"].append(feature_name)
            output_dict["channel"].append(object_loader.channel)
            output_dict["compartment"].append(object_loader.compartment)
            output_dict["value"].append(numpy.float32(value))
    return output_dict
