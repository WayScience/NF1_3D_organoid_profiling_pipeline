#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import shutil
import sys

import numpy as np
import tqdm

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
    # check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser(
        description="set up directories for the analysis of the data"
    )

    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="patient name, e.g. 'P01'",
    )
    argparser.add_argument(
        "--well_fov",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = argparser.parse_args()
    patient = args.patient
    well_fov = args.well_fov
else:
    patient = "NF0030"
    well_fov = "D2-1"


# In[3]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/segmentation_masks/{well_fov}"
).resolve(strict=True)
zstack_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/zstack_images/{well_fov}"
).resolve(strict=True)


# In[4]:


# perform checks for each directory
segmentation_data_files = list(segmentation_data_dir.glob("*"))


# ## Copy files from processed dir to cellprofiler images dir

# In[5]:


masks_names_to_keep_dict = {
    "cell_masks_watershed.tiff": "cell_masks.tiff",
    "cytoplasm_mask.tiff": "cytoplasm_masks.tiff",
    "nuclei_masks_reassigned.tiff": "nuclei_masks.tiff",
    "organoid_masks_reconstructed.tiff": "organoid_masks.tiff",
}


# In[6]:


# remove files that are not in the list of masks to keep
for file in tqdm.tqdm(segmentation_data_files):
    # check if the file is in the masks_names_to_keep_dict as a key or value
    if (
        file.name not in masks_names_to_keep_dict.keys()
        and file.name not in masks_names_to_keep_dict.values()
    ):
        # if not, remove the file
        file.unlink()
        print(f"Removed file: {file.name}")
    else:
        # rename the file to the new name
        if file.name in masks_names_to_keep_dict.keys():
            new_name = masks_names_to_keep_dict[file.name]
            new_file_path = segmentation_data_dir / new_name
            if not new_file_path.exists():
                file.rename(new_file_path)
                print(f"Renamed file: {file.name} to {new_name}")
            else:
                print(f"File {new_name} already exists, skipping rename.")
        else:
            print(f"File {file.name} already exists, skipping rename.")


# In[7]:


# regrab the segmentation data files after renaming
segmentation_data_files = list(segmentation_data_dir.glob("*"))


# In[8]:


# copy the masks to the zstack directory
for file in tqdm.tqdm(segmentation_data_files):
    for original_name, new_name in masks_names_to_keep_dict.items():
        if file.name == new_name:
            destination = zstack_dir / new_name
            shutil.copy(file, destination)
            print(f"Copied file: {file} to {destination}")
