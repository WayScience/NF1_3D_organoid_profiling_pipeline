import pathlib
import re

import numpy as np
import tifffile


def read_2D_image_for_zstacking(
    file_path: str,
) -> np.ndarray:
    """
    Description
    -----------
    Reads in a 2D image from a given file path and returns it as a numpy array.

    Parameters

    ----------
    file_path : str
        The path to the 2D image file.
    Returns
    -------
    np.ndarray
        The 2D image as a numpy array.

    Raises
    -------
    ValueError
        If the image has more than 2 dimensions.
    """

    img = tifffile.imread(file_path)

    if len(img.shape) > 2 and img.shape[2] == 3:
        # If the image has 3 channels (e.g., RGB), convert to grayscale
        img = img[:, :, 0]

    return img


def get_well_fov_dirs(parent_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Retrieve all well fov dirs in a given parent dir

    Parameters
    ----------
    parent_dir : pathlib.Path
        Patient parent dir

    Returns
    -------
    List[pathlib.Path]
        List of well fov dirs in _parent_dir
    """
    well_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    return well_dirs


def get_to_the_unested_dir(
    nested_dir: pathlib.Path, times_nested: int
) -> pathlib.Path | None:
    """
    Unest the dir given the number of time the directories are nested.

    Parameters
    ----------
    nested_dir : pathlib.Path
        The parent directory containing the nested dirs
    times_nested : int
        The number of times that a dir is nested

    Returns
    -------
    pathlib.Path | None
        The output file path of the least nested parent dir or None
    """
    for _ in range(times_nested):
        nested_dir = nested_dir.glob("*").__next__()
        nested_dir = [d for d in nested_dir.parent.iterdir() if d.is_dir()]
        if len(nested_dir) != 1:
            return None
        nested_dir = nested_dir[0]
    return nested_dir


def check_well_dir_name_format(dir_name: str) -> bool:
    """
    Check if a well directory name matches the expected format.
    Accepts formats like:
      - A1-1
      - A01-01
      - A1-1 (60X)
      - A12-34 (any trailing parenthetical/metadata)

    Parameters
    ----------
    dir_name : str
        The name of the directory to check.
    Returns
    -------
    bool
        True if the directory name matches the expected format, False otherwise.

    """
    dir_name = dir_name.strip()
    pattern = re.compile(r"^[A-Z]\d{1,2}-\d{1,2}(?:\s*\(.*\))?$")
    return bool(pattern.match(dir_name))
