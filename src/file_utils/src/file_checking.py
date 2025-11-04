import pathlib


def check_number_of_files(
    directory: pathlib.Path, n_files: int, verbose: bool = False
) -> bool:
    """
    Check if the number of files in a directory is equal to a given number.

    Parameters
    ----------
    directory : pathlib.Path
        Specified directory to check file number.
    n_files : int
        The expected number of files in the directory.
    verbose : bool, optional
        If verbose is True, additional information will be printed.

    Returns
    -------
    bool
        True if the number of files in the directory is equal to the expected number, False otherwise.
    """
    files = list(directory.glob("*"))
    files = [f for f in files if f.is_file()]
    if len(files) != n_files:
        if verbose:
            print(
                f"{directory.name} expected {n_files} files, but found {len(files)} files."
            )
        return False
    return True
