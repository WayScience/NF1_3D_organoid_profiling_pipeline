"""Helpers for reading imaging files and channel stacks."""

import pathlib
from typing import Iterable, List

import numpy as np
import skimage
import tifffile


# ----------------------------------------------------------------------
# extensions and reads
# ----------------------------------------------------------------------
def find_files_available(
    input_dir: pathlib.Path,
    image_extensions: set[str] = {".tif", ".tiff"},
) -> List[str]:
    """List available image files in a directory.

    Parameters
    ----------
    input_dir : pathlib.Path
        Directory to scan for image files.
    image_extensions : set[str], optional
        File extensions to include, by default {".tif", ".tiff"}.

    Returns
    -------
    List[str]
        Sorted list of image file paths.
    """
    files = sorted(input_dir.glob("*"))
    files = [str(x) for x in files if x.suffix in image_extensions]
    return files


def read_in_channels(
    files: Iterable[str],
    channel_dict: dict[str, str] = {
        "nuclei": "405",
        "cyto1": "488",
        "cyto2": "555",
        "cyto3": "640",
        "brightfield": "TRANS",
    },
    channels_to_read: List[str | None] = [None],
) -> dict[str, np.ndarray | None]:
    """Read z-stack images for each channel token.

    Parameters
    ----------
    files : Iterable[str]
        File paths to search for channel tokens.
    channel_dict : dict[str, str], optional
        Mapping of channel name to filename token.
    channels_to_read : List[str | None], optional
        Reserved for channel selection; currently unused.

    Returns
    -------
    dict[str, np.ndarray | None]
        Mapping of channel name to loaded image array (or None).
    """
    loaded: dict[str, np.ndarray | None] = {}
    for channel, token in channel_dict.items():
        matches = [f for f in files if token in str(pathlib.Path(f).name)]
        if len(matches) == 0:
            loaded[channel] = None
        else:
            if len(matches) > 1:
                print(
                    f"Warning: multiple files match token '{token}' for channel '{channel}'. Using first match: {matches[0]}"
                )
            try:
                loaded[channel] = np.array(read_zstack_image(matches[0]))
            except Exception as e:
                print(f"Error loading {matches[0]} for channel '{channel}': {e}")
                loaded[channel] = None

    return loaded


def read_zstack_image(file_path: str) -> np.ndarray:
    """
    Description
    -----------
    Reads in a z-stack image from a given file path and returns it as a numpy array.

    Parameters
    ----------
    file_path : str
        The path to the z-stack image file.
    Returns
    -------
    np.ndarray
        The z-stack image as a numpy array.

    Raises
    -------
    ValueError
        If the image has less than 3 dimensions.
    """

    img = tifffile.imread(file_path)

    if len(img.shape) > 5:
        # determine in any of the dimensions is size of 1?
        img = np.squeeze(img)
    elif len(img.shape) < 3:
        raise ValueError(f"Image at {file_path} has less than 3 dimensions")

    if img.dtype != np.uint16:
        if img.dtype in [np.float32, np.float64]:
            # For float images, first rescale to 0-1 range, then convert
            img = skimage.exposure.rescale_intensity(img, out_range=(0, 1))
            img = skimage.img_as_uint(img)
        else:
            # For other integer types
            img = skimage.img_as_uint(img)

    return img
