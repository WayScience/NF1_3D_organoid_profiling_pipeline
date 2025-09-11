#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import shutil
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
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(pathlib.Path("~/mnt/bandicoot").resolve(), root_dir)

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

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
    patient = "NF0014_T1"
    well_fov = "D2-1"


# In[ ]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/segmentation_masks/{well_fov}"
).resolve(strict=True)
zstack_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/zstack_images/{well_fov}"
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
