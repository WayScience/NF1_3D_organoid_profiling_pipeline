"""Intensity feature extraction utilities for 3D image objects.

Provides functions to compute intensity statistics (mean, median, min, max,
standard deviation, quartiles), edge-based measurements, center-of-mass
coordinates, and mass displacement for segmented 3D objects.
"""

import numpy
import scipy.ndimage
import skimage.segmentation
from image_analysis_3D.featurization_utils.loading_classes import ObjectLoader


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

        # Crop to bounding box for efficiency
        cropped_label = selected_label_object[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ]
        cropped_image = selected_image_object[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ]

        # Create coordinate grids for the bounding box
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

        # Skip if volume is zero to avoid division by zero
        if volume == 0 or integrated_intensity == 0:
            continue

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

        # Calculate center of mass (geometric center) using cropped arrays
        object_mask = cropped_label > 0
        cm_x = numpy.mean(mesh_x[object_mask])
        cm_y = numpy.mean(mesh_y[object_mask])
        cm_z = numpy.mean(mesh_z[object_mask])

        # Calculate intensity-weighted center of mass using cropped arrays
        intensity_x_coord = cropped_image * mesh_x
        intensity_y_coord = cropped_image * mesh_y
        intensity_z_coord = cropped_image * mesh_z
        i_x = numpy.sum(intensity_x_coord[object_mask])
        i_y = numpy.sum(intensity_y_coord[object_mask])
        i_z = numpy.sum(intensity_z_coord[object_mask])
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
            "IntegratedIntensity": integrated_intensity,
            "MeanIntensity": mean_intensity,
            "StdIntensity": std_intensity,
            "MinIntensity": min_intensity,
            "MaxIntensity": max_intensity,
            "LowerQuartileIntensity": lower_quartile_intensity,
            "UpperQuartileIntensity": upper_quartile_intensity,
            "MedianIntensity": median_intensity,
            "MassDisplacement": mass_displacement,
            "MeanAbsoluteDeviationIntensity": mad_intensity,
            "IntegratedIntensityEdge": integrated_intensity_edge,
            "MeanIntensityEdge": mean_intensity_edge,
            "StdIntensityEdge": std_intensity_edge,
            "MinIntensityEdge": min_intensity_edge,
            "MaxIntensityEdge": max_intensity_edge,
            "MaxZ": max_z,
            "MaxY": max_y,
            "MaxX": max_x,
            "CMI.X": cmi_x,
            "CMI.Y": cmi_y,
            "CMI.Z": cmi_z,
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
