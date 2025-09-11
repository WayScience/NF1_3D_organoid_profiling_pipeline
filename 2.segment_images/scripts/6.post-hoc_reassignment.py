#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to reassign segmentation labels based on the objects that they are contained in.
# This will mean that the segmentation label id of the cell will match that of the nucleus that it is contained in.

# In[ ]:


import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import tifffile

cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd
else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break
sys.path.append(str(root_dir / "utils"))
from arg_parsing_utils import check_for_missing_args, parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(pathlib.Path("~/mnt/bandicoot").resolve(), root_dir)


# In[ ]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
    )
else:
    print("Running in a notebook")
    well_fov = "C4-2"
    patient = "NF0014_T1"

mask_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/segmentation_masks/{well_fov}"
).resolve()


# In[3]:


def remove_edge_cases(
    mask: np.ndarray,
    border: int = 10,
) -> np.ndarray:
    """
    Remove masks that are image edge cases
    In this case - the edge literally means the edge of the image
    This is useful to remove masks that are not fully contained within the image

    Parameters
    ----------
    mask : np.ndarray
        The mask to process, should be a 3D numpy array
    border : int, optional
        The number of pixels in width to create border to scan for edge cased, by default 10

    Returns
    -------
    np.ndarray
        The mask with edge cases removed
    """

    edge_pixels = np.concatenate(
        [
            # all of z, last n rows (y), all columns (x) - bottom edge
            mask[:, -border:, :].flatten(),
            # all of z, first n rows (y), all columns (x) - top edge
            mask[:, 0:border, :].flatten(),
            # all of z, all rows (y), first n columns (x) - left edge
            mask[:, :, 0:border:].flatten(),
            # all of z, all rows (y), last n columns (x) - right edge
            mask[:, :, -border:].flatten(),
            # each are the edges stacked for the whole volume -> no need to specify every z slice or 3D edge
        ]
    )
    # get unique edge pixel values
    edge_pixels = np.unique(edge_pixels[edge_pixels > 0])

    for edge_pixel_case in edge_pixels:
        # make the edge cases equal to zero
        mask[mask == edge_pixel_case] = 0

    # return the mask with edge cases removed
    return mask


def centroid_within_bbox_detection(
    centroid: tuple,
    bbox: tuple,
) -> bool:
    """
    Check if the centroid is within the bbox

    Parameters
    ----------
    centroid : tuple
        Centroid of the object in the order of (z, y, x)
        Order of the centroid is important
    bbox : tuple
        Where the bbox is in the order of (z_min, y_min, x_min, z_max, y_max, x_max)
        Order of the bbox is important

    Returns
    -------
    bool
        True if the centroid is within the bbox, False otherwise
    """
    z_min, y_min, x_min, z_max, y_max, x_max = bbox
    z, y, x = centroid
    # check if the centroid is within the bbox
    if (
        z >= z_min
        and z <= z_max
        and y >= y_min
        and y <= y_max
        and x >= x_min
        and x <= x_max
    ):
        return True
    else:
        return False


def check_if_centroid_within_mask(
    centroid: tuple, mask: np.ndarray, label: int
) -> bool:
    """
    Check if the centroid is within the mask

    Parameters
    ----------
    centroid : tuple
        Centroid of the object in the order of (z, y, x)
        Order of the centroid is important
    mask : np.ndarray
        The mask to check against

    Returns
    -------
    bool
        True if the centroid is within the mask, False otherwise
    """
    z, y, x = centroid
    z = np.round(z).astype(int)
    y = np.round(y).astype(int)
    x = np.round(x).astype(int)
    # check if the centroid is within the segmentation mask
    cell_label = mask[z, y, x]
    if cell_label > 0 and cell_label == label:
        return True
    else:
        return False


def mask_label_reassignment(
    mask_df: pd.DataFrame,
    mask_input: np.ndarray,
) -> np.ndarray:
    """
    Reassign the labels of the mask based on the mask_df

    Parameters
    ----------
    mask_df : pd.DataFrame
        DataFrame containing the labels and centroids of the mask
    mask_input : np.ndarray
        The input mask to reassign the labels to

    Returns
    -------
    np.ndarray
        The mask with reassigned labels
    """
    for i, row in mask_df.iterrows():
        if row["label"] == row["new_label"]:
            # if the label is already the new label, skip
            continue
        mask_input[mask_input == row["label"]] = row["new_label"]
    return mask_input


# In[4]:


# get the organoid masks
# cell_mask_path = mask_dir / "cell_masks_reconstructed_corrected.tiff"
cell_mask_path = mask_dir / "cell_masks_watershed.tiff"
nuclei_mask_path = mask_dir / "nuclei_masks_reconstructed_corrected.tiff"
nuclei_mask_output_path = mask_dir / "nuclei_masks_reassigned.tiff"

cell_mask = tifffile.imread(cell_mask_path)
nuclei_mask = tifffile.imread(nuclei_mask_path)


# In[5]:


# get the centroid and bbox of the cell mask
cell_df = pd.DataFrame.from_dict(
    skimage.measure.regionprops_table(
        cell_mask,
        properties=["centroid", "bbox"],
    )
)
cell_df["compartment"] = "cell"
cell_df["label"] = cell_mask[
    cell_df["centroid-0"].astype(int),
    cell_df["centroid-1"].astype(int),
    cell_df["centroid-2"].astype(int),
]
# remove all 0 labels
cell_df = cell_df[cell_df["label"] > 0].reset_index(drop=True)


# In[6]:


nuclei_df = pd.DataFrame.from_dict(
    skimage.measure.regionprops_table(
        nuclei_mask,
        properties=["centroid", "bbox"],
    )
)
nuclei_df["compartment"] = "nuclei"
nuclei_df["label"] = nuclei_mask[
    nuclei_df["centroid-0"].astype(int),
    nuclei_df["centroid-1"].astype(int),
    nuclei_df["centroid-2"].astype(int),
]
nuclei_df = nuclei_df[nuclei_df["label"] > 0].reset_index(drop=True)
nuclei_df["new_label"] = nuclei_df["label"].copy()


# In[7]:


nuclei_df.head()


# In[8]:


cell_df.head()


# In[9]:


print(f"Number of nuclei: {len(nuclei_df)}\nNumber of cells: {len(cell_df)}\n")


# In[10]:


# if a centroid of the nuclei is inside the cell mask,
# then make the cell retain the label of the nuclei
for i, row in nuclei_df.iterrows():
    for j, row2 in cell_df.iterrows():
        nuc_contained_in_cell_bool = check_if_centroid_within_mask(
            centroid=(
                row["centroid-0"],
                row["centroid-1"],
                row["centroid-2"],
            ),
            mask=cell_mask,
            label=row2["label"],
        )
        if nuc_contained_in_cell_bool:
            # if the centroid of the nuclei is within the cell mask,
            # then make the cell retain the label of the nuclei
            nuclei_df.at[i, "new_label"] = row2["label"]
            break
        else:
            pass


# In[11]:


# merge the dataframes
nuclei_and_cell_df = pd.merge(
    nuclei_df,
    cell_df,
    left_on="new_label",
    right_on="label",
    suffixes=("_nuclei", "_cell"),
)
nuclei_and_cell_df.head()


# In[12]:


nuclei_and_cell_df[["label_nuclei", "new_label", "label_cell"]].head()
nuclei_df


# In[13]:


print(
    f"Number of nuclei: {len(nuclei_df)}\n"
    f"Number of cells: {len(cell_df)}\n"
    f"Number of cells with nuclei: {len(nuclei_and_cell_df)}"
)


# In[14]:


# remove the edge cases
cell_mask = remove_edge_cases(
    mask=cell_mask,
    border=10,
)
nuclei_mask = remove_edge_cases(
    mask=nuclei_mask,
    border=10,
)


# In[15]:


# reassign the labels of the cell mask
nuclei_mask = mask_label_reassignment(
    mask_df=nuclei_df,
    mask_input=nuclei_mask,
)
# save the cell mask
tifffile.imwrite(
    nuclei_mask_output_path,
    nuclei_mask,
)
