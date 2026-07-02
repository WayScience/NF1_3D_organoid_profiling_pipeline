import mahotas
import numpy
import skimage
from image_analysis_3D.featurization_utils.loading_classes import ObjectLoader


def scale_image(image: numpy.ndarray, num_gray_levels: int = 256) -> numpy.ndarray:
    """
    Scale the image to a specified number of gray levels.
    Example: 1024 gray levels will be scaled to 256 gray levels if
    num_gray_levels=256.
    An image with a pixel value of 0 will be scaled to 0 and a pixel value
    of 1023 will be scaled to 255.

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
    outrange_mapping = {
        256: "uint8",
        65536: "uint16",
    }
    out_range = outrange_mapping.get(num_gray_levels)
    if out_range is None:
        raise ValueError(
            f"Unsupported num_gray_levels: {num_gray_levels}. "
            f"Supported values are: {list(outrange_mapping.keys())}"
        )
    # scale the image to the requested gray levels
    return skimage.exposure.rescale_intensity(
        image,
        in_range="image",
        out_range=out_range,
    )


def measure_3D_texture(
    object_loader: ObjectLoader,
    distance: int = 1,
    grayscale: int = 256,
) -> dict:
    """
    Calculate texture features for each object in the image using Haralick features.

    The features are calculated for each object separately and the mean value
    is returned.

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
        A dictionary containing the object ID, texture name, and texture value
        with keys:
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
    # set the number of directions based on the dimensionality of the image
    n_directions = 13

    output_texture_dict = {
        "object_id": [],
        "texture_name": [],
        "texture_value": [],
    }

    if len(labels) != 0:
        # Precompute bounding boxes for labeled regions to avoid per-object full-array copies
        props = skimage.measure.regionprops_table(
            label_object, properties=["label", "bbox"]
        )
        # Map label id to bbox (z0, y0, x0, z1, y1, x1)
        label_to_bbox = {}
        labels_prop = props.get("label", [])
        for i, lbl in enumerate(labels_prop):
            label_to_bbox[int(lbl)] = (
                int(props["bbox-0"][i]),
                int(props["bbox-1"][i]),
                int(props["bbox-2"][i]),
                int(props["bbox-3"][i]),
                int(props["bbox-4"][i]),
                int(props["bbox-5"][i]),
            )

        features = numpy.full((n_directions, 13, max(labels)), numpy.nan)
        for _, label in enumerate(labels):
            bbox = label_to_bbox.get(int(label))
            if bbox is None:
                continue

            min_z, min_y, min_x, max_z, max_y, max_x = bbox

            # Crop to the object's bounding box (skimage bboxes are half-open)
            image_object = object_loader.image[
                min_z:max_z, min_y:max_y, min_x:max_x
            ].copy()
            selected_label_object = label_object[min_z:max_z, min_y:max_y, min_x:max_x]
            object_mask = selected_label_object == label
            if not numpy.any(object_mask):
                continue
            image_object[~object_mask] = 0
            n_features = len(feature_names)
            image_object = scale_image(image_object, num_gray_levels=grayscale)

            try:
                features[:, :, label - 1] = mahotas.features.haralick(
                    ignore_zeros=True,
                    f=image_object,
                    distance=distance,
                    compute_14th_feature=False,
                )
            except ValueError:
                features[:, :, label - 1] = numpy.full(
                    (n_directions, n_features), numpy.nan
                )
    else:
        features = numpy.zeros((n_directions, 13, 0))

    for direction, direction_features in enumerate(features):
        direction_str = f"{direction:02d}"
        for feature_name, feature in zip(feature_names, direction_features):
            for object_id, feature_value in zip(labels, feature):
                output_texture_dict["object_id"].append(object_id)
                output_texture_dict["texture_name"].append(
                    f"{feature_name}-{distance}-{direction_str}-{grayscale}"
                )
                output_texture_dict["texture_value"].append(feature_value)

    return output_texture_dict
