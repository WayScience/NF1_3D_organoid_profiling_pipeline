"""Functions for formatting morphology feature names in a consistent way across all morphology features."""

import pathlib

import pandas


def remove_underscores_from_string(string: str) -> str:
    """
    Remove unwanted delimiters from a string and replace them with hyphens.

    Parameters
    ----------
    string : str
        The string to remove unwanted delimiters from.

    Returns
    -------
    str
        The string with unwanted delimiters removed and replaced with hyphens.
    """
    if not isinstance(string, str):
        try:
            string = str(string)
        except Exception as e:
            raise ValueError(
                f"Input string must be a string or convertible to a string. Received input: {string} of type {type(string)}"
            ) from e
    string = string.translate(
        str.maketrans(
            {
                "_": "-",
                ".": "-",
                " ": "-",
                "/": "-",
            }
        )
    )

    return string


def format_morphology_feature_name(
    compartment: str, channel: str, feature_type: str, measurement: str
) -> str:
    """
    Format a morphology feature name in a consistent way across all morphology features.
    This format follows specification for the following:
    https://github.com/WayScience/NF1_3D_organoid_profiling_pipeline/blob/main/docs/RFC-2119-Feature-Naming-Convention.md

    Parameters
    ----------
    compartment : str
        The compartment name.
    channel : str
        The channel name.
    feature_type : str
        The feature type.
    measurement : str
        The measurement name.

    Returns
    -------
    str
        The formatted feature name.
    """

    compartment = remove_underscores_from_string(compartment)
    channel = remove_underscores_from_string(channel)
    feature_type = remove_underscores_from_string(feature_type)
    measurement = remove_underscores_from_string(measurement)

    return f"{compartment}_{channel}_{feature_type}_{measurement}"


def save_features_as_parquet(
    parent_path: pathlib.Path,
    df: pandas.DataFrame,
    compartment: str,
    channel: str,
    feature_type: str,
    cpu_or_gpu: str,
) -> pathlib.Path:
    """Save features as parquet files in a consistent way across all morphology features.

    Parameters
    ----------
    parent_path : pathlib.Path
        The parent path to save the features to.
    df : pandas.DataFrame
        The dataframe containing the features to save.
    compartment : str
        The compartment name.
    channel : str
        The channel name.
    feature_type : str
        The feature type.
    cpu_or_gpu : str
        Whether the features were generated using CPU or GPU processing.

    Returns
    -------
    pathlib.Path
    """
    save_path = (
        parent_path
        / f"{compartment}_{channel}_{feature_type}_{cpu_or_gpu}_features.parquet"
    )
    df.to_parquet(save_path, index=False)
    return save_path
