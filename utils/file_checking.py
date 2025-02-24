import pathlib


def check_number_of_files(directory: pathlib.Path, n_files: int) -> bool:
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
    bool
        True if the number of files in the directory is equal to the expected number, False otherwise.
    """
    files = list(directory.glob("*"))
    files = [f for f in files if f.is_file()]
    if len(files) != n_files:
        print(
            f"{directory.name} expected {n_files} files, but found {len(files)} files."
        )
        return False
    return True
