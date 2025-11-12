#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

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
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    mask_subparent_name = args["mask_subparent_name"]
    check_for_missing_args(
        patient=patient,
        input_subparent_name=input_subparent_name,
        mask_subparent_name=mask_subparent_name,
    )
else:
    patient = "NF0037_T1-Z-0.1"
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"


# In[3]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/"
).resolve(strict=True)
zstack_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/"
).resolve(strict=True)


# In[4]:


# perform checks for each directory
segmentation_data_dirs = list(segmentation_data_dir.glob("*"))
segmentation_data_dirs = [d for d in segmentation_data_dirs if d.is_dir()]
zstack_dirs = list(zstack_dir.glob("*"))
zstack_dirs = [d for d in zstack_dirs if d.is_dir()]

print("Checking segmentation data directories")
dict_of_reruns = {
    "patient": [],
    "well_fov": [],
}
total = 0
completed = 0
non_completed = 0
for dir in segmentation_data_dirs:
    total += 1
    status, dir_name = check_number_of_files(dir, 4)
    if status:
        completed += 1
    else:
        non_completed += 1
        dict_of_reruns["patient"].append(patient)
        dict_of_reruns["well_fov"].append(dir_name)

print(f"Total directories checked: {total}")
print(f"Need to rerun: {non_completed}")
print(f"{completed / total * 100:.2f}% completed successfully")


# In[5]:


dict_of_reruns["well_fov"].sort()
dict_of_reruns["well_fov"]
