#!/usr/bin/env python
# coding: utf-8

# # Copy raw images into one folder to use for CellProfiler processing
# 
# Currently, the images are located nest deep within multiple folders. 
# For best practices, we will copy the images (preserving metadata) to one folder that can be used for CellProfiler processing.

# ## Import libraries

# In[1]:


import argparse
import pathlib
import shutil

import tqdm


# ## Set paths and variables

# In[2]:


argparse = argparse.ArgumentParser(
    description="Copy files from one directory to another"
)
argparse.add_argument(
    "--HPC", type=bool, help="Type of compute to run on", required=True
)

args = argparse.parse_args()
HPC = args.HPC


# In[3]:


# Define the parent directory containing all the nested folders
if not HPC:
    # local paths
    parent_dir = pathlib.Path(
        "/home/lippincm/Desktop/18TB/Cell Painting-NF0014 Thawed3-Pilot Drug Screening/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
    ).resolve(strict=True)
else:
    parent_dir = pathlib.Path(
        "/pl/active/koala/GFF_Data/GFF-Raw/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy/"
    ).resolve(strict=True)

# Create the NF0014 folder next to the parent_dir (same level in the hierarchy)
nf0014_dir = pathlib.Path("../../data/raw_images").resolve()

nf0014_dir.mkdir(parents=True, exist_ok=True)

# Image extensions that we are looking to copy
image_extensions = {".tif", ".tiff"}


# ## Reach the nested images and copy to one folder

# Run this cell through the script

# In[ ]:


for image_file in tqdm.tqdm(parent_dir.rglob("*")):
    list_of_dirs = list(image_file.rglob("*"))
    list_of_dirs = [x for x in list_of_dirs if x.is_dir()]
    for dir in list_of_dirs:
        well_dir = nf0014_dir / dir.name
        well_dir.mkdir(parents=True, exist_ok=True)
        for image in dir.rglob("*/*"):
            if image.suffix in image_extensions:
                shutil.copy2(image, well_dir)

