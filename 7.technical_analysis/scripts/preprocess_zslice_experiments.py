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

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import pandas as pd
import skimage
import tifffile
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import read_zstack_image
from matplotlib.colors import BoundaryNorm, ListedColormap
from notebook_init_utils import bandicoot_check, init_notebook
from skimage import io
from technical_analysis_segmentation_utils import (
    convert_indexed_mask_to_binary_mask,
    extract_IOU,
    signed_xor_3color,
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


well_fovs = ["F4-1", "F4-2", "F4-3"]
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


# In[3]:


for well_fov in tqdm(well_fovs, desc="Processing well FOVs", leave=True):
    well_fov_path = pathlib.Path(f"{NF0037_T1Z0_1_path}/{well_fov}").resolve(
        strict=True
    )

    well_fov_files = list(well_fov_path.glob("*"))
    for file in tqdm(well_fov_files, desc=f"Processing {well_fov} files", leave=False):
        # ----- read in original 0.1um z-slice image -----
        channel_image_0_1 = read_zstack_image(file)

        # ----- generate and save 0.2um z-slice image -----
        channel_image_0_2 = channel_image_0_1[::2, :, :].copy()  # 0.1um to 0.2um
        output_0_2_path = pathlib.Path(
            f"{NF0037_T1Z0_2_path}/{well_fov}/{file.name}"
        ).resolve()
        output_0_2_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_0_2_path.exists():
            tifffile.imwrite(output_0_2_path, channel_image_0_2)

        # ----- generate and save 0.5um z-slice image -----
        channel_image_0_5 = channel_image_0_1[::5, :, :].copy()  # 0.1um to 0.5um
        output_0_5_path = pathlib.Path(
            f"{NF0037_T1Z0_5_path}/{well_fov}/{file.name}"
        ).resolve()
        output_0_5_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_0_5_path.exists():
            tifffile.imwrite(output_0_5_path, channel_image_0_5)

        # ----- generate and save 1.0um z-slice image -----
        channel_image_1_0 = channel_image_0_1[::10, :, :].copy()  # 0.1um to 1.0um
        output_1_0_path = pathlib.Path(
            f"{NF0037_T1Z1_path}/{well_fov}/{file.name}"
        ).resolve()
        output_1_0_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_1_0_path.exists():
            tifffile.imwrite(output_1_0_path, channel_image_1_0)
        print(
            channel_image_0_1.shape,
            channel_image_0_2.shape,
            channel_image_0_5.shape,
            channel_image_1_0.shape,
        )
