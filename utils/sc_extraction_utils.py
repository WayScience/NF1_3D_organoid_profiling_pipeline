"""
This file contains functions to run Pycytominer merge single cells, normalize, and feature select and add
single cell count metadata into the returned csv.gz files.
"""

# must use the annotations import as CellProfiler is restricted to Python 3.8 at this time so Optional
# by itself only works in Python 3.10
from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd
from pycytominer import feature_select, normalize
from pycytominer.cyto_utils import cells, output


def add_single_cell_count_df(
    data_df: pd.DataFrame, well_column_name: str = "Metadata_Well"
) -> pd.DataFrame:
    """
    This function adds a column with the number of singles cells per well to a pandas dataframe.

    Args:
        data_df (pd.DataFrame):
            dataframe to add number of single cells to
        well_column_name (str):
            name of column for wells to use for finding single cell count (defaults to "Metadata_Well")

    Returns:
        pd.DataFrame:
            pandas dataframe with new metadata column with single cell count
    """
    merged_data = (
        data_df.groupby([well_column_name])[well_column_name]
        .count()
        .reset_index(name="Metadata_number_of_singlecells")
    )

    data_df = data_df.merge(merged_data, on=well_column_name)
    # pop out the column from the dataframe
    singlecell_column = data_df.pop("Metadata_number_of_singlecells")
    # insert the column as the second index column in the dataframe
    data_df.insert(2, "Metadata_number_of_singlecells", singlecell_column)

    return data_df


def add_sc_count_metadata_file(
    data_path: pathlib.Path,
    well_column_name: str = "Metadata_Well",
    file_type: str = "csv.gz",
):
    """
    This function loads in the saved file from Pycytominer or CytoTable (e.g. normalized, etc.), adds the single cell counts for
    each well as metadata, and saves the file to the same place (as the same file type)

    Args:
        data_path (pathlib.Path):
            path to data file to add single cell count on
        well_column_name (str):
            name of column for wells to use for finding single cell count (defaults to "Metadata_Well")
        file_type (str, optional):
            the file type of the data (options include parquet, csv, defaults to "csv.gz")
    """
    # load in data
    if file_type == "csv.gz":
        data_df = pd.read_csv(data_path, compression="gzip")
    if file_type == "parquet":
        data_df = pd.read_parquet(data_path)
    if file_type == "csv":
        data_df = pd.read_csv(data_path, compression="gzip")

    # add single cell count as new metadata column
    data_df = add_single_cell_count_df(
        data_df=data_df, well_column_name=well_column_name
    )

    # save updated df to same path as the same file type
    if file_type == "parquet":
        data_df.to_parquet(data_path)
    if file_type == "csv.gz":
        data_df.to_csv(data_path, compression="gzip")
    if file_type == "csv":
        data_df.to_csv(data_path, compression="gzip")


def extract_single_cells(
    single_cell_file: str,
    linking_cols: dict,
    platemap_df: pd.DataFrame,
    output_dir: str,
    output_file_name: str,
    compartments: list = ["Per_Nuclei", "Per_Cells", "Per_Cytoplasm"],
    join_on_columns=dict,
    normalize_sc: Optional[bool | False] = False,
    feature_selection_sc: Optional[bool | False] = False,
    norm_feature_select: Optional[bool | False] = False,
):
    """
    Use Pycytominer SingleCells class to perform single cell extraction, normalization, and feature selection on
    CellProfiler SQLite files. Normalization and feature selection are optional processes that you can choose
    to turn on. This function will output all data into individual csv.gz files to a specified output folder. The
    individual csv.gz are then updated to contain single cell count metadata.

    Args:
        single_cell_file (str):
            string file path to SQLite file (must start with sqlite:///)
        linking_cols (dict):
            dictionary with the linking columns between compartments/tables in database
        platemap_df (pd.DataFrame):
            dataframe with the platemap metadata to merge with single cells
        output_dir (str):
            directory for csv.gz files to be saved to
        output_file_name (str):
            name for merge single cell csv.gz files
        compartments (list):
            list of compartments within the SQLite file used for merging single cells
        join_on_columns (dict):
            columns to connect the metadata from platemap file with the single cells from the SQLite file. Must look like this to work:
            {"join_on": ["Metadata_well_id", "Image_Metadata_Well"]}. Note: Even though in the platemap file the name doesn't start with metadata,
            you must add the prefix "Metadata" to the name of joining column to work.
        normalize_sc (bool, optional):
            if set to True, this will perform normalization on the raw merged single cell data (defaults to False)
        feature_selection_sc (bool, optional):
            if set to True, this will perform feature extraction on raw merged single cell data (defaults to False)
        norm_feature_select (bool, optional):
            if set to True, this will perform normalization on raw merged single cell data and feature extraction on
            the normalizated data (defaults to False)
    """
    # instantiate the SingleCells class
    sc = cells.SingleCells(
        sql_file=single_cell_file,
        compartments=compartments,
        compartment_linking_cols=linking_cols,
        image_table_name="Per_Image",
        strata=["Image_Metadata_Well", "Image_Metadata_Plate"],
        merge_cols=["ImageNumber"],
        image_cols="ImageNumber",
        load_image_data=True,
    )

    # Merge single cells across compartments based on well
    anno_kwargs = join_on_columns

    sc_df = sc.merge_single_cells(
        platemap=platemap_df,
        **anno_kwargs,
    )

    sc_output_file = pathlib.Path(f"{output_dir}/sc_{output_file_name}.csv.gz")

    # Save level 2 data as a csv
    output(sc_df, sc_output_file)
    # add single cell count to the raw single cell data
    add_sc_count_metadata_file(sc_output_file)

    # Perform normalization on the raw extracted single cell data
    if normalize_sc:
        # Normalize single cell data and write to file
        normalize_sc_df = normalize(sc_df, method="standardize")

        sc_norm_output_file = pathlib.Path(
            f"{output_dir}/sc_norm_{output_file_name}.csv.gz"
        )

        output(normalize_sc_df, sc_norm_output_file)
        # add single cell count to the normalized data
        add_sc_count_metadata_file(sc_norm_output_file)

    # Perform feature selection on the raw extracted single cell data
    if feature_selection_sc:
        # Select features that will show significant difference between genotypes
        feature_select_ops = [
            "variance_threshold",
            "correlation_threshold",
            "blocklist",
        ]

        feature_select_norm_sc_df = feature_select(sc_df, operation=feature_select_ops)

        sc_norm_fs_output_file = pathlib.Path(
            f"{output_dir}/sc_fs_{output_file_name}.csv.gz"
        )

        output(feature_select_norm_sc_df, sc_norm_fs_output_file)
        # add single cell count to the feature selected data
        add_sc_count_metadata_file(sc_norm_fs_output_file)

    # Perform normalization on raw extracted single cells and perform feature selection on the normalized data
    if norm_feature_select:
        # Normalize single cell data and write to file
        normalize_sc_df = normalize(sc_df, method="standardize")

        sc_norm_output_file = pathlib.Path(
            f"{output_dir}/sc_norm_{output_file_name}.csv.gz"
        )

        output(normalize_sc_df, sc_norm_output_file)
        # add single cell count to the normalized data
        add_sc_count_metadata_file(sc_norm_output_file)

        # Select features that will show significant difference between genotypes
        feature_select_ops = [
            "variance_threshold",
            "correlation_threshold",
            "blocklist",
        ]

        feature_select_norm_sc_df = feature_select(
            normalize_sc_df, operation=feature_select_ops
        )

        sc_norm_fs_output_file = pathlib.Path(
            f"{output_dir}/sc_norm_fs_{output_file_name}.csv.gz"
        )

        output(feature_select_norm_sc_df, sc_norm_fs_output_file)
        # add single cell count to the feature selected data
        add_sc_count_metadata_file(sc_norm_fs_output_file)
