"""Channel mapping utilities."""

import tomllib
from typing import Any


def retrieve_channel_mapping(toml_path: str) -> dict[str, Any]:
    """
    Read in channel mapping from a TOML file.

    Parameters
    ----------
    toml_path : str
        Path to the TOML file.

    Returns
    -------
    dict
        Dictionary containing the channel mapping.
    """
    with open(toml_path, "rb") as f:
        channel_mapping = tomllib.load(f)["channel_mapping"]

    return channel_mapping
