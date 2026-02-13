"""Functions for formatting morphology feature names in a consistent way across all morphology features."""


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
        raise ValueError(f"Input must be a string, but got {type(string)}")
    if "_" in string:
        string = string.replace("_", "-")
    if "." in string:
        string = string.replace(".", "-")
    if " " in string:
        string = string.replace(" ", "-")
    if "/" in string:
        string = string.replace("/", "-")
    return string


def format_morphology_feature_name(
    compartment: str, channel: str, feature_type: str, measurement: str
) -> str:
    """
    Format a morphology feature name in a consistent way across all morphology features.
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
