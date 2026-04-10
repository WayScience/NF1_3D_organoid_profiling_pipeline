import gc

import mahotas
import numpy
import tqdm
from image_analysis_3D.featurization_utils.loading_classes import ObjectLoader


def scale_image(image: numpy.ndarray, num_gray_levels: int = 256) -> numpy.ndarray:
    """
    Scale the image to a specified number of gray levels.
    Example: 1024 gray levels will be scaled to 256 gray levels if num_gray_levels=256.
    An image with a pixel value of 0 will be scaled to 0 and a pixel value of 1023 will be scaled to 255.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be scaled. Can be a ndarray of any shape.
    num_gray_levels : int, optional
        The number of gray levels to scale the image to, by default 256

    Returns
    -------
    numpy.ndarray
        The gray level scaled image of any shape.
    """
    # scale the image to the requested gray levels
    image_min = image.min()
    image_max = image.max()
    if image_max == image_min:
        return numpy.zeros_like(image, dtype=numpy.uint8)
    image = (image - image_min) / (image_max - image_min)
    image = (image * (num_gray_levels - 1)).astype(numpy.uint8)
    return image


def measure_3D_texture(
    object_loader: ObjectLoader,
    distance: int = 1,
    grayscale: int = 256,
) -> dict:
    """
    Calculate texture features for each object in the image using Haralick features.

    The features are calculated for each object separately and the mean value is returned.

    Parameters
    ----------
    object_loader : ObjectLoader
        The object loader containing the image and object information.
    distance : int, optional
        The distance parameter for Haralick features, by default 1
    grayscale : int, optional
        The number of gray levels to scale the image to, by default 256

    Returns
    -------
    dict
        A dictionary containing the object ID, texture name, and texture value with keys:
        - object_id
        - texture_name
        - texture_value

        Texture names include: Angular Second Moment, Contrast, Correlation,
        Variance, Inverse Difference Moment, Sum Average, Sum Variance,
        Sum Entropy, Entropy, and related texture measures.

        - AngularSecondMoment
        - Contrast
        - Correlation
        - Variance
        - InverseDifferenceMoment
        - SumAverage
        - SumVariance
        - SumEntropy
        - Entropy
        - DifferenceVariance
        - DifferenceEntropy
        - InformationMeasureOfCorrelation1
        - InformationMeasureOfCorrelation2

    """
    label_object = object_loader.label_image
    labels = object_loader.object_ids
    feature_names = [
        "AngularSecondMoment",
        "Contrast",
        "Correlation",
        "Variance",
        "InverseDifferenceMoment",
        "SumAverage",
        "SumVariance",
        "SumEntropy",
        "Entropy",
        "DifferenceVariance",
        "DifferenceEntropy",
        "InformationMeasureOfCorrelation1",
        "InformationMeasureOfCorrelation2",
    ]

    output_texture_dict = {
        "object_id": [],
        "texture_name": [],
        "texture_value": [],
    }
    for index, label in tqdm.tqdm(enumerate(labels)):
        selected_label_object = label_object.copy()
        selected_label_object[selected_label_object != label] = 0
        object_voxels = selected_label_object > 0
        if not numpy.any(object_voxels):
            continue

        z_indices, y_indices, x_indices = numpy.where(object_voxels)
        min_z, max_z = numpy.min(z_indices), numpy.max(z_indices)
        min_y, max_y = numpy.min(y_indices), numpy.max(y_indices)
        min_x, max_x = numpy.min(x_indices), numpy.max(x_indices)

        image_object = object_loader.image[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ].copy()
        selected_label_object = selected_label_object[
            min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1
        ]
        image_object[selected_label_object == 0] = 0
        image_object = scale_image(image_object, num_gray_levels=grayscale)
        try:
            haralick_features = mahotas.features.haralick(
                ignore_zeros=True,
                f=image_object,
                distance=distance,
                compute_14th_feature=False,
            )
            haralick_mean = haralick_features.mean(axis=0)
        except ValueError:
            haralick_mean = numpy.full(len(feature_names), numpy.nan, dtype=float)
        for i, feature_name in enumerate(feature_names):
            output_texture_dict["object_id"].append(label)
            output_texture_dict["texture_name"].append(
                f"{feature_name}-{grayscale}-{distance}"
            )
            output_texture_dict["texture_value"].append(haralick_mean[i])
        gc.collect()
    return output_texture_dict
