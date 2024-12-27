"""
This collection of functions runs CellProfiler and will rename the .sqlite outputted to any specified name if running an analysis pipeline.
"""

# must use the annotations import as CellProfiler is restricted to Python 3.8 at this time so Optional
# by itself only works in Python 3.10
from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Optional


def rename_sqlite_file(
    sqlite_dir_path: pathlib.Path, name: str, hardcode_sqlite_name: str
):
    """Rename the .sqlite file to be {method}.sqlite as to differentiate between the files

    Args:
        sqlite_dir_path (pathlib.Path):
            path to CellProfiler_output directory
        name (str):
            new name for the SQLite file
        hardcode_sqlite_name (str):
            hardcoded name of the returned SQLite file from CellProfiler to change
    """
    try:
        # CellProfiler requires a name to be set in to pipeline, so regardless of plate or method, all sqlite files name are hardcoded
        sqlite_file_path = pathlib.Path(
            f"{sqlite_dir_path}/{hardcode_sqlite_name}.sqlite"
        )

        new_file_name = str(sqlite_file_path).replace(
            sqlite_file_path.name, f"{name}.sqlite"
        )

        # change the file name in the directory
        pathlib.Path(sqlite_file_path).rename(pathlib.Path(new_file_name))
        print(f"The file is renamed to {pathlib.Path(new_file_name).name}!")

    except FileNotFoundError as e:
        print(
            f"The {hardcode_sqlite_name}.sqlite file is not found in directory. Either the pipeline wasn't ran properly or the file is already renamed.\n"
            f"{e}"
        )


def run_cellprofiler(
    path_to_pipeline: str,
    path_to_input: str,
    path_to_output: str,
    sqlite_name: Optional[None | str] = None,
    hardcode_sqlite_name: Optional[str | None] = None,
    analysis_run: Optional[bool | False] = False,
    rename_sqlite_file_bool: Optional[bool | False] = False,
):
    """Run CellProfiler on data using LoadData CSV. It can be used for both a illumination correction pipeline and analysis pipeline.

    Args:
        path_to_pipeline (str):
            path to the CellProfiler .cppipe file with the segmentation and feature measurement modules
        path_to_input (str):
            path to the input folder with the images to be analyzed
        path_to_output (str):
            path to the output folder (the directory will be created if it doesn't already exist)
        sqlite_name (str, optional):
            string with name for SQLite file for an analysis pipeline to either be renamed and/or to check to see if the
            run has already happened (default is None)
        analysis_run (bool, optional):
            will run an analysis pipeline and can rename sqlite files if using one pipeline for multiple datasets (default is False)
        rename_sqlite_file (bool, optional):
            will rename the outputted SQLite file from the CellProfiler pipeline and rename it when using one
            pipeline for multiple datasets. If kept as default, the SQLite file will not be renamed and the sqlite_name is used to
            find if the analysis pipeline has already been ran (default is False)
    """
    # check to make sure the paths to files are correct and they exists before running CellProfiler
    if not pathlib.Path(path_to_pipeline).resolve().exists:
        raise FileNotFoundError(f"Directory '{path_to_pipeline}' does not exist")

    # make logs directory
    log_dir = pathlib.Path("./logs")
    os.makedirs(log_dir, exist_ok=True)

    if not analysis_run:

        # a log file is created for each plate or data set name that holds all outputs and errors
        with open(
            pathlib.Path(f"logs/cellprofiler_output.log"),
            "w",
        ) as cellprofiler_output_file:
            # run CellProfiler pipeline
            command = [
                "cellprofiler",
                "-c",
                "-r",
                "-p",
                path_to_pipeline,
                "-i",
                path_to_input,
                "-o",
                path_to_output,
            ]
            subprocess.run(
                command,
                stdout=cellprofiler_output_file,
                stderr=cellprofiler_output_file,
                check=True,
            )
            print(
                "The CellProfiler run has been completed with log. Please check log file for any errors."
            )

    if analysis_run:
        # runs through any files that are in the output path and checks to see if analysis pipeline was already run
        if any(
            files.name.startswith(sqlite_name)
            for files in pathlib.Path(path_to_output).iterdir()
        ):
            raise NameError(
                f"The file {sqlite_name}.sqlite has already been renamed! This means it was probably already analyzed."
            )

        # a log file is created for each plate or data set name that holds all outputs and errors
        with open(
            pathlib.Path(f"logs/cellprofiler_output.log"),
            "w",
        ) as cellprofiler_output_file:
            # run CellProfiler for an analysis run
            command = [
                "cellprofiler",
                "-c",
                "-r",
                "-p",
                path_to_pipeline,
                "-i",
                path_to_input,
                "-o",
                path_to_output,
            ]
            subprocess.run(
                command,
                stdout=cellprofiler_output_file,
                stderr=cellprofiler_output_file,
                check=True,
            )

        if rename_sqlite_file_bool:
            # rename the outputted .sqlite file to the specified sqlite name if running one analysis pipeline
            rename_sqlite_file(
                sqlite_dir_path=pathlib.Path(path_to_output),
                name=sqlite_name,
                hardcode_sqlite_name=hardcode_sqlite_name,
            )
