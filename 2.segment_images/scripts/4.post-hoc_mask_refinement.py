#!/usr/bin/env python
# coding: utf-8

# # Segmentation corrections

# The goal of this notebook is to correct potential errors in the segmentation of the 3D image data.
# Potential errors can be observed in the figure below where each row is a different slice of the 3D image data and each column is a different outcome of the segmentation.
# Each segmentation of cell and nucleus is shown in a different color.
# Where cells or nuclei that are the same object id are shown in the same color.
# While cells and nuclei that are different object ids are shown in different colors.
# Some of the outcomes are not correct and need to be corrected.
# While others might be correct or incorrect but there is not logical way to determine if they are correct or not.
# These cases are not corrected.
# ![Segmentation errors](../media/3D_segmentations_correction_events.png)
#

# In[1]:


import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import skimage
import tifffile

sys.path.append("../../utils")

from segmentation_decoupling import euclidian_2D_distance

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--patient",
        type=str,
        help="The patient ID",
    )

    parser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    parser.add_argument(
        "--compartment",
        type=str,
        default="none",
        help="The compartment to segment",
    )

    args = parser.parse_args()
    well_fov = args.well_fov
    compartment = args.compartment
    patient = args.patient
else:
    print("Running in a notebook")
    well_fov = "C5-2"
    compartment = "nuclei"
    patient = "NF0014"

mask_dir = pathlib.Path(f"../../data/{patient}/processed_data/{well_fov}").resolve()


# In[3]:


if compartment == "nuclei":
    mask_path = mask_dir / "nuclei_masks_reconstructed.tiff"
    mask_output_path = mask_dir / "nuclei_masks_reconstructed_corrected.tiff"
elif compartment == "cell":
    mask_path = mask_dir / "cell_masks_watershed.tiff"
    mask_output_path = mask_dir / "cell_masks_corrected.tiff"
elif compartment == "organoid":
    mask_path = mask_dir / "organoid_masks_reconstructed.tiff"
    mask_output_path = mask_dir / "organoid_masks_reconstructed_corrected.tiff"

else:
    raise ValueError("Compartment must be either nuclei, cell or organoid")

mask = tifffile.imread(mask_path)


# ### Functions for refinement

# In[4]:


from typing import List, Tuple


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a bounding box.

    Parameters
    ----------
    bbox : Tuple[int, int, int, int]
        The bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns
    -------
    int
        The area of the bounding box.
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calculate_overlap(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
) -> float:
    # calculate the % overlap of the second bbox with the first bbox
    if calculate_bbox_area(bbox1) == 0 or calculate_bbox_area(bbox2) == 0:
        return 0.0
    if calculate_bbox_area(bbox1) >= calculate_bbox_area(bbox2):
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x_max - x_min)
        overlap_height = max(0, y_max - y_min)
        overlap_area = overlap_width * overlap_height
        bbox1_area = calculate_bbox_area(bbox1)
        bbox2_area = calculate_bbox_area(bbox2)
        overlap_percentage = overlap_area / bbox2_area if bbox2_area > 0 else 0
        return overlap_percentage
    elif calculate_bbox_area(bbox1) < calculate_bbox_area(bbox2):
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x_max - x_min)
        overlap_height = max(0, y_max - y_min)
        overlap_area = overlap_width * overlap_height
        bbox1_area = calculate_bbox_area(bbox1)
        bbox2_area = calculate_bbox_area(bbox2)
        overlap_percentage = overlap_area / bbox1_area if bbox1_area > 0 else 0
        return overlap_percentage
    else:
        print("Error: Bboxes are the same size")


def merge_sets(list_of_sets: list) -> list:
    """
    Merge sets in a list of sets if they have any intersection.

    Parameters
    ----------
    list_of_sets : list
        A list of sets to be merged.

    Returns
    -------
    list
        A list of sets after merging those that intersect.
    """
    for i, set1 in enumerate(list_of_sets):
        for j, set2 in enumerate(list_of_sets):
            if i != j and len(set1.intersection(set2)) > 0:
                set1.update(set2)
    return list_of_sets


def check_for_all_same_labels(
    object_information_df: pd.DataFrame,
) -> bool:
    """
    Check if all labels in the object information DataFrame are the same.

    Parameters
    ----------
    object_information_df : pd.DataFrame
        The DataFrame containing object information with 'label' column.

    Returns
    -------
    bool
        True if all labels are the same, False otherwise.
    """
    return object_information_df["label"].nunique() == 1


def missing_slice_check(
    object_information_df: pd.DataFrame,
    window_min: int = 0,
    window_max: int = 2,
    interpolated_rows_to_add: List[int] = [],
) -> List[pd.DataFrame]:
    """
    Check for missing slices in the object information DataFrame and add interpolated rows if necessary.

    Parameters
    ----------
    object_information_df : pd.DataFrame
        The DataFrame containing object information with 'z' and 'label' columns.
    window_min : int, optional
        The minimum window size for checking missing slices, by default 0
    window_max : int, optional
        The maximum window size for checking missing slices, by default 2
    interpolated_rows_to_add : List[int], optional
        A list to store rows to be added for interpolation, by default []

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames containing rows to be added for interpolation.
    """
    max_z = object_information_df["z"].max()
    min_z = object_information_df["z"].min()
    if max_z - min_z > 1:
        if len(object_information_df) < 3:
            # get the first row
            row = object_information_df.iloc[0]
            new_row = {
                "added_z": row["z"],
                "added_new_label": row["label"],
                "zslice_to_copy": row["z"],
            }

            # interpolate the labels to the middle most slice
            # get the middle slice
            middle_slice = int((max_z + min_z) / 2)
            # insert one slice
            z_zlice_to_copy = row["z"]

            new_row = {
                # 'index': object_information_df['index'].values[0],
                # 'index': object_max_slice_label,
                "added_z": middle_slice,
                "added_new_label": row["label"],
                "zslice_to_copy": z_zlice_to_copy,
            }
            interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
    return interpolated_rows_to_add


def add_min_max_boundry_slices(
    object_information_df: pd.DataFrame,
    global_min_z: int,
    global_max_z: int,
    interpolated_rows_to_add: List[pd.DataFrame] = [],
) -> List[pd.DataFrame]:
    """
    Add slices to the object information DataFrame that are one slice away from the global min and max z slices.

    Parameters
    ----------
    object_information_df : pd.DataFrame
        The DataFrame containing object information with 'z' and 'label' columns.
    global_min_z : int
        The global minimum z slice.
    global_max_z : int
        The global maximum z slice.
    interpolated_rows_to_add : List[pd.DataFrame], optional
        A list to store rows to be added for interpolation, by default []

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames containing rows to be added for interpolation at the min and max z slices.
    """
    # find labels that are 1 slice away from the min or max and extend the label
    for i, row in object_information_df.iterrows():
        # check if the z slice is one away from the min or max (global min and max)
        if row["z"] == global_max_z - 1:
            new_row = {
                "added_z": global_max_z,
                "added_new_label": row["label"],
                "zslice_to_copy": row["z"],
            }
            interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
        elif row["z"] == global_min_z + 1:
            new_row = {
                "added_z": global_min_z,
                "added_new_label": row["label"],
                "zslice_to_copy": row["z"],
            }
            interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
    return interpolated_rows_to_add


def add_masks_where_missing(
    new_mask_image: np.ndarray,
    interpolated_rows_to_add_df: pd.DataFrame,
) -> np.ndarray:
    """
    Add masks to the new mask image where the slices are missing based on the interpolated rows.

    Parameters
    ----------
    new_mask_image : np.ndarray
        The new mask image to which the slices will be added.
    interpolated_rows_to_add_df : pd.DataFrame
        The DataFrame containing the rows to be added for interpolation, with columns 'added_z', 'added_new_label', and 'zslice_to_copy'.

    Returns
    -------
    np.ndarray
        The new mask image with the added slices.
    """
    for slice in interpolated_rows_to_add_df["added_z"].unique():
        # get the rows that correspond to the slice
        tmp_df = interpolated_rows_to_add_df[
            interpolated_rows_to_add_df["added_z"] == slice
        ]
        if tmp_df.shape[0] == 0:
            continue
        for i, row in tmp_df.iterrows():
            # get the z slice to copy mask
            new_slice = new_mask_image[row["zslice_to_copy"].astype(int), :, :].copy()
            new_slice[new_slice != row["added_new_label"]] = 0

            old_slice = new_mask_image[row["added_z"].astype(int), :, :].copy()
            max_projected_slice = np.maximum(old_slice, new_slice)
            new_mask_image[row["added_z"].astype(int), :, :] = max_projected_slice
    return new_mask_image


# ### Set data flow objects, constants and parameters

# #### Constants

# In[5]:


sliding_window_context = 3
global_max_z = mask.shape[0]  # number of z slices
global_min_z = 0
# expand the z slices into a list  of slices between the min and max z slices
z_slices = [x for x in range(global_min_z, global_max_z)]


# ### Loop through the slices in a sliding window fashion and correct the segmentation

# In[6]:


new_mask_image = mask.copy()


# In[7]:


for z in z_slices[: -(sliding_window_context - 1)]:
    interpolated_rows_to_add = []

    final_dict = {
        "index1": [],
        "index2": [],
        "z1": [],
        "z2": [],
        "distance": [],
        "label1": [],
        "label2": [],
    }
    list_of_cell_masks = []
    for z_slice in range(0, new_mask_image.shape[0] - 1):
        compartment_df = pd.DataFrame.from_dict(
            skimage.measure.regionprops_table(
                new_mask_image[z, :, :],
                properties=["centroid", "bbox"],
            )
        )
        compartment_df["z"] = z_slice

        list_of_cell_masks.append(compartment_df)
    compartment_df = pd.concat(list_of_cell_masks)

    # get the pixel value of the organoid mask at each x,y,z coordinate
    compartment_df["label"] = new_mask_image[
        compartment_df["z"].astype(int),
        compartment_df["centroid-0"].astype(int),
        compartment_df["centroid-1"].astype(int),
    ]
    compartment_df.reset_index(drop=True, inplace=True)
    compartment_df["new_label"] = compartment_df["label"]
    # drop all labels that are 0
    compartment_df = compartment_df[compartment_df["label"] != 0]

    # Get the temporary sliding window
    tmp_window_df = compartment_df[
        (compartment_df["z"] >= z) & (compartment_df["z"] < z + sliding_window_context)
    ]

    if tmp_window_df["z"].nunique() < sliding_window_context:
        continue
    for i, row1 in tmp_window_df.iterrows():
        for j, row2 in tmp_window_df.iterrows():
            if i != j:  # Ensure you're not comparing the same row
                if row1["z"] != row2["z"]:
                    # get the first bbox

                    distance = euclidian_2D_distance(
                        (row1["centroid-0"], row1["centroid-1"]),
                        (row2["centroid-0"], row2["centroid-1"]),
                    )

                    if distance < 20:
                        final_dict["index1"].append(i)
                        final_dict["index2"].append(j)
                        final_dict["z1"].append(row1["z"])
                        final_dict["z2"].append(row2["z"])
                        final_dict["distance"].append(distance)
                        final_dict["label1"].append(row1["label"])
                        final_dict["label2"].append(row2["label"])
    final_df = pd.DataFrame.from_dict(final_dict)
    final_df["index_set"] = final_df.apply(
        lambda row: frozenset([row["index1"], row["index2"]]), axis=1
    )
    final_df["index_set"] = final_df["index_set"].apply(lambda x: tuple(sorted(x)))

    list_of_sets = final_df["index_set"].tolist()
    list_of_sets = [set(s) for s in list_of_sets]
    merged_sets = merge_sets(list_of_sets)
    # drop the duplicates
    merged_sets = list({frozenset(s): s for s in merged_sets}.values())

    # from final_df generate the z-ordered cases
    for object_set in merged_sets:
        # find rows that contain integers that are in the object_set
        rows_that_contain_object_set = final_df[
            final_df["index_set"].apply(lambda x: set(x).issubset(object_set))
        ]
        # get the index, label, and z pair
        dict_of_object_information = {"index": [], "label": [], "z": []}
        for i, row in rows_that_contain_object_set.iterrows():
            dict_of_object_information["index"].append(row["index1"])
            dict_of_object_information["label"].append(row["label1"])
            dict_of_object_information["z"].append(row["z1"])
            dict_of_object_information["index"].append(row["index2"])
            dict_of_object_information["label"].append(row["label2"])
            dict_of_object_information["z"].append(row["z2"])
        object_information_df = pd.DataFrame.from_dict(dict_of_object_information)
        object_information_df.drop_duplicates(
            subset=["index", "label", "z"], inplace=True
        )
        object_information_df.sort_values(by=["index", "z"], inplace=True)
        if check_for_all_same_labels(object_information_df):
            # if all labels are the same, skip this object
            continue
        interpolated_rows_to_add = missing_slice_check(
            object_information_df, interpolated_rows_to_add=interpolated_rows_to_add
        )
        interpolated_rows_to_add = add_min_max_boundry_slices(
            object_information_df,
            global_min_z=global_min_z,
            global_max_z=global_max_z,
            interpolated_rows_to_add=interpolated_rows_to_add,
        )

    if len(interpolated_rows_to_add) == 0:
        if z == z_slices[-1]:
            tifffile.imwrite(mask_output_path, new_mask_image)
        else:
            continue
    interpolated_rows_to_add_df = pd.concat(interpolated_rows_to_add, axis=0)
    new_mask_image = new_mask_image.copy()
    new_mask_image = add_masks_where_missing(
        new_mask_image=new_mask_image,
        interpolated_rows_to_add_df=interpolated_rows_to_add_df,
    )
    print("writing the mask for z slice", z)


# In[8]:


if not mask_output_path.exists():
    tifffile.imwrite(mask_output_path, new_mask_image)
