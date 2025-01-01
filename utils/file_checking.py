import pathlib
from typing import Tuple


def check_number_of_files(
    directory: pathlib.Path, n_files: int
) -> Tuple[None, ValueError]:
    """
    Check if the number of files in a directory is equal to a given number.

    Parameters
    ----------
    directory : pathlib.Path
        Specified directory to check file number.
    n_files : int
        The expected number of files in the directory.

    Returns
    -------
    Tuple[None, ValueError]
        Returns None if the number of files is equal to n_files, otherwise raises a ValueError.

    Raises
    ------
    ValueError
        If the number of files in the directory is not equal to n_files.
    """
    files = list(directory.glob("*"))
    files = [f for f in files if f.is_file()]
    if len(files) != n_files:
        raise ValueError(f"Expected {n_files} files in {directory}, found {len(files)}")
