#!/usr/bin/env python
# coding: utf-8

# # Copy raw images into one folder to use for CellProfiler processing
# 
# Currently, the images are located nest deep within multiple folders. 
# For best practices, we will copy the images (preserving metadata) to one folder that can be used for CellProfiler processing.
# This file is modified from its original version: https://github.com/WayScience/GFF_2D_organoid_prototyping .

# ## Import libraries

# In[1]:


import argparse
import pathlib
import shutil
import sys

import tqdm


# ## Set paths and variables

# In[2]:


argparse = argparse.ArgumentParser(
    description="Copy files from one directory to another"
)
argparse.add_argument(
    "--HPC", type=bool, default=False, help="Type of compute to run on (default: False)"
)

# Parse arguments
args = argparse.parse_args(args=sys.argv[1:] if "ipykernel" not in sys.argv[0] else [])
HPC = args.HPC

print(f"HPC: {HPC}")


# In[3]:


# Define parent and destination directories in a single dictionary
dir_mapping = {
    # "NF0014": {
    #     "parent": pathlib.Path(
    #         "/media/18tbdrive/GFF_organoid_data/Cell Painting-NF0014 Thawed3-Pilot Drug Screening/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
    #         if not HPC
    #         else "/pl/active/koala/GFF_Data/GFF-Raw/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
    #     ).resolve(strict=True),
    #     "destination": pathlib.Path("../../data/NF0014_raw_images").resolve(),
    # },
    "NF0016": {
        "parent": pathlib.Path(
            "/media/18tbdrive/GFF_organoid_data/NF0016 Cell Painting-Pilot Drug Screening-selected/NF0016-Cell Painting Images/NF0016-images copy"
            if not HPC
            else "/pl/active/koala/GFF_Data/GFF-Raw/NF0016 Cell Painting-Pilot Drug Screening-selected/NF0016-Cell Painting Images/NF0016-images copy"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/NF0016_raw_images").resolve(),
    },
    "NF0018": {
        "parent": pathlib.Path(
            "/media/18tbdrive/GFF_organoid_data/NF0018 (T6) Cell Painting-Pilot Drug Screeining-selected/NF0018-Cell Painting Images/NF0018-All Acquisitions"
            if not HPC
            else "/pl/active/koala/GFF_Data/GFF-Raw/NF0018 (T6) Cell Painting-Pilot Drug Screeining-selected/NF0018-Cell Painting Images/NF0018-All Acquisitions"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/NF0018_raw_images").resolve(),
    },
}

# Image extensions that we are looking to copy
image_extensions = {".tif", ".tiff"}


# ## Reach the nested images and copy to one folder

# Run this cell through the script

# In[4]:


# Loop through each key in the mapping to copy data from the parent to the destination
for key, paths in dir_mapping.items():
    parent_dir = paths["parent"]
    dest_dir = paths["destination"]

    print(f"Processing {key}: {parent_dir} -> {dest_dir}")

    # Ensure the destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Loop through the well-level directories and copy the images
    for image_file in tqdm.tqdm(parent_dir.rglob("*")):
        # Get all subdirectories
        list_of_dirs = list(image_file.rglob("*"))
        list_of_dirs = [x for x in list_of_dirs if x.is_dir()]

        for dir in list_of_dirs:
            # Create the corresponding well directory in the destination
            well_dir = dest_dir / dir.name
            well_dir.mkdir(parents=True, exist_ok=True)

            # Copy images that match the allowed extensions
            for image in dir.rglob("*/*"):
                if image.suffix.lower() in image_extensions:
                    shutil.copy2(image, well_dir)

    print(f"Completed processing {key}: {parent_dir} -> {dest_dir}")

