#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib
import shutil
import sys

import numpy as np
import tqdm
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

from file_checking import check_number_of_files

# In[2]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    mask_subparent_name = args["mask_subparent_name"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        input_subparent_name=input_subparent_name,
        mask_subparent_name=mask_subparent_name,
    )
else:
    patient = "NF0014_T1"
    well_fov = "C4-2"
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"


# In[3]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/{well_fov}"
).resolve(strict=True)
zstack_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}"
).resolve(strict=True)


# ## Copy files from processed dir to cellprofiler images dir

# In[4]:


# regrab the segmentation data files after renaming
segmentation_data_files = list(segmentation_data_dir.glob("*"))
print(segmentation_data_files)


# In[5]:


masks_names_to_keep_dict = {
    "cell_masks_watershed.tiff": "cell_mask.tiff",
    "cytoplasm_masks.tiff": "cytoplasm_mask.tiff",
    "nuclei_masks_reassigned.tiff": "nuclei_mask.tiff",
    "organoid_masks_reconstructed.tiff": "organoid_mask.tiff",
}


# In[ ]:


for file in tqdm.tqdm(segmentation_data_files, desc="Cleaning up segmentation masks"):
    # print(file.name)
    if file.name in masks_names_to_keep_dict.keys():
        # rename file to standard name
        new_name = masks_names_to_keep_dict[file.name]
        new_path = segmentation_data_dir / new_name
        file.rename(new_path)
# reglob the files
segmentation_data_files = list(segmentation_data_dir.glob("*"))
for file in segmentation_data_files:
    if file.name in masks_names_to_keep_dict.values():
        continue
    else:
        # delete file
        file.unlink()
