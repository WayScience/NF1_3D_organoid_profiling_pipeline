#!/usr/bin/env python
# coding: utf-8

# In this notebook, we preprocess z-slice spacing experiments by downsampling a high-resolution z-stack (0.1um z-slice spacing) to simulate lower-resolution z-stacks (0.2um, 0.5um, and 1.0um z-slice spacing).
# This allows us to analyze the impact of z-slice spacing on downstream analyses.
#
# For a theorectical 10um z-slice experiment:
# | Z-slice Spacing | Number of Slices in 10um Z-Stack | conversion factor from 0.1um slices to Z-slice spacing |
# |-----------------|----------------------------------| ------------------------------|
# | 0.1 um          | 100                              | 1                            |
# | 0.2 um          | 50                               | 2                            |
# | 0.5 um          | 20                               | 5                            |
# | 1.0 um          | 10                               | 10                           |

# In[1]:


import os
import pathlib

# Import dependencies
import numpy as np
import tifffile
from image_analysis_3D.file_utils.file_reading import read_zstack_image
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


def convert_indexed_mask_to_binary_mask(indexed_mask: np.ndarray) -> np.ndarray:
    """
    Convert an indexed mask to a binary mask.

    Parameters
    ----------
    indexed_mask : np.ndarray
        An indexed mask where 0 represents the background and any positive integer represents a different object.

    Returns
    -------
    np.ndarray
        A binary mask where True represents the foreground (objects) and False represents the background.
    """
    binary_mask = np.zeros_like(indexed_mask, dtype=bool)
    binary_mask[indexed_mask > 0] = True
    return binary_mask


def extract_IOU(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        A binary mask where True represents the foreground (objects) and False represents the background.
    mask2 : np.ndarray
        A binary mask where True represents the foreground (objects) and False represents the background.

    Returns
    -------
    float
        The Intersection over Union (IoU) score between the two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-12)
    return iou_score.item()


def signed_xor_3color(mask1, mask2):
    """
    Create a 3-value difference map:
    - 2 where only mask1 is True (decon only)
    - 1 where only mask2 is True (other only)
    - 0 where both masks agree
    """
    result = np.zeros_like(mask1, dtype=int)
    result[mask1 & ~mask2] = 2  # decon only
    result[~mask1 & mask2] = 1  # other only
    return result


# In[3]:


input_output_dict = {
    "patient": [
        "NF0037_T1",
        "NF0037_T1",
        "NF0055_T1",
        "NF0055_T1",
        "NF0055_T1",
        "NF0055_T1",
        "NF0055_T1",
        "NF0055_T1",
    ],
    "well_fov": [
        "F4-2",
        "F4-3",
        "E4-1-0",
        "E4-3-0",
        "E4-5-0",
        "G5-1-0",
        "G5-2-0",
        "G5-5-0",
    ],
}


# In[4]:


NF0037_T1Z0_1_path = pathlib.Path(
    f"{image_base_dir}/data/NF0037_T1-Z-0.1/zstack_images/"
).resolve(strict=True)

# output generated data
NF0037_T1Z1_path = pathlib.Path(
    f"{image_base_dir}/data/NF0037_T1-Z-1/zstack_images/"
).resolve()
NF0037_T1Z0_5_path = pathlib.Path(
    f"{image_base_dir}/data/NF0037_T1-Z-0.5/zstack_images/"
).resolve()
NF0037_T1Z0_2_path = pathlib.Path(
    f"{image_base_dir}/data/NF0037_T1-Z-0.2/zstack_images/"
).resolve()


# In[5]:


for patient, well_fov in tqdm(
    zip(input_output_dict["patient"], input_output_dict["well_fov"]),
    desc="Processing patients and wells",
    total=len(input_output_dict["patient"]),
    leave=True,
):
    file_path_0_1 = pathlib.Path(
        f"{image_base_dir}/data/{patient}-Z-0.1/zstack_images/{well_fov}"
    ).resolve(strict=True)
    file_paths_0_1 = list(file_path_0_1.glob("*"))
    for file_01 in tqdm(
        file_paths_0_1,
        desc=f"Processing files for {patient} - {well_fov}",
        total=len(file_paths_0_1),
        leave=False,
    ):
        # Read the z-stack image
        zstack_01 = read_zstack_image(file_01)
        output_0_2_file_path = pathlib.Path(
            f"{image_base_dir}/data/{patient}-Z-0.2/zstack_images/{well_fov}/{file_01.name}"
        ).resolve()
        output_0_5_file_path = pathlib.Path(
            f"{image_base_dir}/data/{patient}-Z-0.5/zstack_images/{well_fov}/{file_01.name}"
        ).resolve()
        output_1_0_file_path = pathlib.Path(
            f"{image_base_dir}/data/{patient}-Z-1/zstack_images/{well_fov}/{file_01.name}"
        ).resolve()
        # if output_0_2_file_path.exists() and output_0_5_file_path.exists() and output_1_0_file_path.exists():
        #     continue
        for output_file_path in [
            output_0_2_file_path,
            output_0_5_file_path,
            output_1_0_file_path,
        ]:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

        zstack_0_2 = zstack_01[::2, :, :].copy()  # 0.1um to 0.2um
        zstack_0_5 = zstack_01[::5, :, :].copy()  # 0.1um to 0.5um
        zstack_1_0 = zstack_01[::10, :, :].copy()  # 0.1um to 1.0um

        zstack_0_1_z_shape = zstack_01.shape[0]
        zstack_0_2_z_shape = zstack_0_2.shape[0]
        zstack_0_5_z_shape = zstack_0_5.shape[0]
        zstack_1_0_z_shape = zstack_1_0.shape[0]

        # check that the z shapes are consistent with the downsampling factors
        # must be withing 1 slice of each other due to rounding + or -
        def check_z_shape(actual, expected, name, label, tol=1):
            # round the actual and expected values to the nearest integer
            actual = int(round(actual))
            expected = int(round(expected))
            if abs(actual - expected) > tol:
                print(
                    f"Warning: Z shape mismatch for {name} between 0.1um and {label} stacks."
                )
                print(f"0.1um Z shape: {actual}, {label} Z shape: {expected}")

        check_z_shape(
            zstack_0_1_z_shape / 2, zstack_0_2_z_shape, file_01.name, "0.2um", tol=2
        )
        check_z_shape(
            zstack_0_1_z_shape / 5, zstack_0_5_z_shape, file_01.name, "0.5um", tol=2
        )
        check_z_shape(
            zstack_0_1_z_shape / 10, zstack_1_0_z_shape, file_01.name, "1.0um", tol=2
        )

        tifffile.imwrite(output_0_2_file_path, zstack_0_2)
        tifffile.imwrite(output_0_5_file_path, zstack_0_5)
        tifffile.imwrite(output_1_0_file_path, zstack_1_0)
