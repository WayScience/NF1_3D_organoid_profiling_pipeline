import os
import pathlib
from typing import Tuple

"""
This utils module contains functions to initialize the notebook environment
and check for the existence of an external mount point for Bandicoot.
"""


def init_notebook() -> Tuple[pathlib.Path, bool]:
    """
    Description
    -----------
    Initializes the notebook environment by determining the root directory of the Git repository
    and checking if the code is running in a Jupyter notebook.

    Returns
    -------
    Tuple[pathlib.Path, bool]
        - pathlib.Path: The root directory of the Git repository.
        - bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        cfg = get_ipython().config
        in_notebook = True
    except NameError:
        in_notebook = False

    # Get the current working directory
    cwd = pathlib.Path.cwd()

    if (cwd / ".git").is_dir():
        root_dir = cwd

    else:
        root_dir = None
        for parent in cwd.parents:
            if (parent / ".git").is_dir():
                root_dir = parent
                break

    # Check if a Git root directory was found
    if root_dir is None:
        raise FileNotFoundError("No Git root directory found.")
    return root_dir, in_notebook


def bandicoot_check(
    bandicoot_mount_path: pathlib.Path, root_dir: pathlib.Path
) -> pathlib.Path:
    """
    This function determines if the external mount point for Bandicoot exists.

    Parameters
    ----------
    bandicoot_mount_path : pathlib.Path
        The path to the Bandicoot mount point.
    root_dir : pathlib.Path
        The root directory of the Git repository.

    Returns
    -------
    pathlib.Path
        The base directory for image data.
    """
    if bandicoot_mount_path.exists():
        # comment out depending on whose computer you are on
        # mike's computer
        image_base_dir = pathlib.Path(
            os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")
        ).resolve(strict=True)
    else:
        image_base_dir = root_dir
    return image_base_dir


def avoid_path_crash_bandicoot(
    bandicoot_path: pathlib.Path,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    This function avoids path crashes by checking if the Bandicoot path exists
    and setting the raw image directory and output base directory accordingly.

    Parameters
    ----------
    bandicoot_path : pathlib.Path
        The path to the Bandicoot directory.

    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        The raw image directory and output base directory.
    """
    root_dir, _ = init_notebook()
    if bandicoot_path.exists():
        # comment out depending on whose computer you are on
        # mike's computer
        bandicoot_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve()
        raw_image_dir = pathlib.Path(f"{bandicoot_path}/NF1_organoid_data/").resolve()
        output_base_dir = bandicoot_path
    else:
        raw_image_dir = pathlib.Path(f"{root_dir}/NF1_organoid_data/").resolve()
        output_base_dir = root_dir
    return raw_image_dir, output_base_dir
