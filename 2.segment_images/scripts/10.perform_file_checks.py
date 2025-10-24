#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib
import sys

import numpy as np
import tqdm

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
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[ ]:


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
    patient = "NF0014_T1"
    input_subparent_name = "deconvolved_images"
    mask_subparent_name = "deconvolved_segmentation_masks"


# In[ ]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/"
).resolve(strict=True)
zstack_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/"
).resolve(strict=True)


# In[ ]:


# perform checks for each directory
segmentation_data_dirs = list(segmentation_data_dir.glob("*"))
segmentation_data_dirs = [d for d in segmentation_data_dirs if d.is_dir()]
zstack_dirs = list(zstack_dir.glob("*"))
zstack_dirs = [d for d in zstack_dirs if d.is_dir()]

print("Checking segmentation data directories")
for dir in segmentation_data_dirs:
    check_number_of_files(dir, 4)

print("Checking zstack directories")
for dir in zstack_dirs:
    check_number_of_files(dir, 9)
