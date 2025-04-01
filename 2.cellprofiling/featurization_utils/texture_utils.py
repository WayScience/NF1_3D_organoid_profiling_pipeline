import gc

import mahotas
import tqdm
from loading_classes import ObjectLoader


def measure_3D_texture(
    object_loader: ObjectLoader,
    distance: int = 1,
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

    Returns
    -------
    dict
        A dictionary containing the object ID, texture name, and texture value.
    """
    label_object = object_loader.objects
    labels = object_loader.object_ids
    feature_names = [
        "Angular.Second.Moment",
        "Contrast",
        "Correlation",
        "Variance",
        "Inverse.Difference.Moment",
        "Sum.Average",
        "Sum.Variance",
        "Sum.Entropy",
        "Entropy",
        "Difference.Variance",
        "Difference.Entropy",
        "Information.Measure.of.Correlation.1",
        "Information.Measure.of.Correlation.2",
    ]

    output_texture_dict = {
        "object_id": [],
        "texture_name": [],
        "texture_value": [],
    }
    for index, label in tqdm.tqdm(enumerate(labels)):
        selected_label_object = label_object.copy()
        selected_label_object[selected_label_object != label] = 0
        image_object = object_loader.image.copy()
        image_object[selected_label_object == 0] = 0
        haralick_features = mahotas.features.haralick(
            ignore_zeros=False,
            f=image_object,
            distance=distance,
            compute_14th_feature=False,
        )
        haralick_mean = haralick_features.mean(axis=0)
        for i, feature_name in enumerate(feature_names):
            output_texture_dict["object_id"].append(label)
            output_texture_dict["texture_name"].append(feature_name)
            output_texture_dict["texture_value"].append(haralick_mean[i])
        del haralick_mean
        del haralick_features
        gc.collect()
    return output_texture_dict
