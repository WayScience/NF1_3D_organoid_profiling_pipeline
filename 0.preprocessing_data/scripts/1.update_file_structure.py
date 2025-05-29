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
import multiprocessing
import os
import pathlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import tqdm

# ## Set paths and variables

# In[2]:


argparse = argparse.ArgumentParser(
    description="Copy files from one directory to another"
)
argparse.add_argument("--HPC", action="store_true", help="Type of compute to run on")

# Parse arguments
args = argparse.parse_args(args=sys.argv[1:] if "ipykernel" not in sys.argv[0] else [])
HPC = args.HPC

print(f"HPC: {HPC}")


# In[3]:


if HPC:
    raw_image_dir_hpc = pathlib.Path("/pl/active/koala/GFF_Data/GFF-Raw/").resolve(
        strict=True
    )
else:
    # comment out depending on whose computer you are on
    # mike's computer
    raw_image_dir_local = pathlib.Path(
        "/home/lippincm/Desktop/18TB/NF1_Patient_organoids/"
    ).resolve(strict=True)
    # Jenna's computer
    # raw_image_dir_local = pathlib.Path("/media/18tbdrive/GFF_organoid_data/Cell Painting-NF0014 Thawed3-Pilot Drug Screening")


# In[ ]:


# Define parent and destination directories in a single dictionary
dir_mapping = {
    "NF0014": {
        "parent": pathlib.Path(
            f"{raw_image_dir_local}/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
            if not HPC
            else f"{raw_image_dir_hpc}/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/NF0014/raw_images").resolve(),
    },
    "NF0016": {
        "parent": pathlib.Path(
            f"{raw_image_dir_local}/NF0016 Cell Painting-Pilot Drug Screening-selected/NF0016-Cell Painting Images/NF0016-images copy"
            if not HPC
            else f"{raw_image_dir_hpc}/NF0016 Cell Painting-Pilot Drug Screening-selected/NF0016-Cell Painting Images/NF0016-images copy"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/NF0016/raw_images").resolve(),
    },
    # "NF0017": {
    #     "parent": pathlib.Path(
    #         f"{raw_image_dir_local}/NF0017-T3-P7 (AGP, Mito Parameter optimization)/Acquisition 03-07-2025"
    #         if not HPC
    #         else f"{raw_image_dir_hpc}/NF0017-T3-P7 (AGP, Mito Parameter optimization)/Acquisition 03-07-2025"  # TODO: Update this later if not correct
    #     ).resolve(strict=True),
    #     "destination": pathlib.Path(
    #         "../../data/raw_images/NF0017/raw_images"
    #     ).resolve(),
    # },
    "NF0018": {
        "parent": pathlib.Path(
            f"{raw_image_dir_local}/NF0018 (T6) Cell Painting-Pilot Drug Screeining/NF0018-Cell Painting Images/NF0018-All Acquisitions"
            if not HPC
            else f"{raw_image_dir_hpc}/NF0018 (T6) Cell Painting-Pilot Drug Screeining/NF0018-Cell Painting Images/NF0018-All Acquisitions"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/NF0018/raw_images").resolve(),
    },
    "NF0021": {
        "parent": pathlib.Path(
            f"{raw_image_dir_local}/NF0021-T1/NF0021-T1 Combined"
            if not HPC
            else f"{raw_image_dir_hpc}/NF0021-T1/NF0021-T1 Combined"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/NF0021/raw_images").resolve(),
    },
    "SACRO219": {
        "parent": pathlib.Path(
            f"{raw_image_dir_local}/SARC0219-T2 Cell Painting-selected/SARC0219-T2 Combined Cell Painting images/SARC0219-T2 Combined/"
            if not HPC
            else f"{raw_image_dir_hpc}/SARC0219-T2 Cell Painting-selected/SARC0219-T2 Combined Cell Painting images/SARC0219-T2 Combined/"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/SARCO219/raw_images").resolve(),
    },
    "SARCO361": {
        "parent": pathlib.Path(
            f"{raw_image_dir_local}/SARC0361/SARC0361 Combined/"
            if not HPC
            else f"{raw_image_dir_hpc}/SARC0361/SARC0361 Combined/"
        ).resolve(strict=True),
        "destination": pathlib.Path("../../data/SARCO361/raw_images").resolve(),
    },
}

# Image extensions that we are looking to copy
image_extensions = {".tif", ".tiff"}


# ## Reach the nested images and copy to one folder

# ### Set QC functions that determine if a well/site is of good quality to process based on file structure

# In[5]:


def has_consistent_naming(well_dir: pathlib.Path) -> bool:
    """Check that all nested folders within a well directory have the same names as the well directory itself.

    Args:
        well_dir (pathlib.Path): Path to a single well directory.

    Returns:
        bool: True if all nested folders inside this well directory have the same name as the well directory, False otherwise.
    """
    # Get the name of the well directory (this will be the expected folder name)
    well_name = well_dir.name

    # Get the immediate subdirectories in the well directory (e.g., Field_1, Field_2)
    sub_dirs = [d for d in well_dir.iterdir() if d.is_dir()]

    if not sub_dirs:
        return False  # No nested folders found, treat as inconsistent

    # Check if each subdirectory contains a nested folder with the same name as the well directory
    for sub in sub_dirs:
        nested_folders = [d.name for d in sub.iterdir() if d.is_dir()]
        if well_name not in nested_folders:
            return False  # Inconsistent folder structure found

    return True  # All subdirectories have a nested folder with the same name as the well directory


def is_image_folder_empty(nested_dir: pathlib.Path) -> bool:
    """Check if a nested directory contains any images.

    Args:
        nested_dir (pathlib.Path): Path to a directory nested within the well directory

    Returns:
        bool: Boolean indicating whether the nested directory contains any images
    """
    return not any(
        image.suffix.lower() in image_extensions for image in nested_dir.rglob("*")
    )


def has_equal_images_per_channel(
    nested_dir: pathlib.Path, channel_names: list[str]
) -> bool:
    """Check if all specified channels have the same number of images by looking for the channel name in the filenames.

    Args:
        nested_dir (pathlib.Path): Path to a directory nested within the well directory.
        channel_names (list[str]): List of strings of the channel names found in the nested directory.

    Returns:
        bool: Boolean indicating whether all specified channels have the same number of images.
    """
    # Initialize counts for each channel
    channel_counts = {channel: 0 for channel in channel_names}

    # Count images for each channel based on the channel name in the filename
    for image in nested_dir.rglob("*"):  # Search for all files recursively
        if image.suffix.lower() in image_extensions:  # Ensure it's an image file
            for channel in channel_names:
                if (
                    channel in image.name
                ):  # If the channel name is found in the image filename
                    channel_counts[channel] += 1

    # Get the unique set of image counts (if all counts are equal, there should be only one unique value)
    image_counts = set(channel_counts.values())

    # If all counts are equal and non-zero, return True; otherwise, return False
    return len(image_counts) == 1 and 0 not in image_counts


# Run this cell through the script

# In[6]:


# Function to process a single nested directory


def process_nested_dir(nested_dir, dest_well_dir, channel_names, image_extensions):
    if not nested_dir.is_dir():
        return f"Skipping {nested_dir}: Not a directory"

    if is_image_folder_empty(nested_dir):
        return f"Skipping {nested_dir}: No images found"

    if not has_equal_images_per_channel(nested_dir, channel_names):
        return f"Skipping {nested_dir}: Unequal images per channel"

    # Copy images to destination, skipping files with 'Tile' in their name
    for image in nested_dir.rglob("*"):
        if image.suffix.lower() in image_extensions and "Tile" not in image.name:
            shutil.copy2(image, dest_well_dir)

    return f"Processed {nested_dir}"


# Function to process a single well directory
def process_well_dir(well_dir, dest_dir, channel_names, image_extensions):
    if not has_consistent_naming(well_dir):
        return f"Skipping {well_dir.stem}: Inconsistent nested folder names within well"

    dest_well_dir = dest_dir / well_dir.name
    dest_well_dir.mkdir(parents=True, exist_ok=True)

    nested_dirs = list(well_dir.iterdir())
    for nested_dir in nested_dirs:
        process_nested_dir(
            nested_dir,
            dest_well_dir,
            channel_names,
            image_extensions,
        )


# Set channel names
channel_names = {"405", "488", "555", "640", "TRANS", "Merge"}

# Loop through each key in the mapping to copy data from the parent to the destination
for key, paths in dir_mapping.items():
    parent_dir = paths["parent"]
    dest_dir = paths["destination"]

    print(f"Processing {key}: {parent_dir} -> {dest_dir}")

    # Ensure the destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get all well-level directories
    well_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    if not well_dirs:
        print(f"Skipping {key}: No well directories found")
        continue
    # Process well directories in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 2) as executor:
        futures = [
            executor.submit(
                process_well_dir, well_dir, dest_dir, channel_names, image_extensions
            )
            for well_dir in well_dirs
        ]
        for future in tqdm.tqdm(futures, desc=f"Processing {key}", leave=False):
            pass

    print(f"Completed processing {key}: {parent_dir} -> {dest_dir}")


# ## NF0016 specific preprocessing

# In[7]:


parent_dir_NF0016 = pathlib.Path("../../data/NF0016/raw_images").resolve(strict=True)
# get all dirs in the parent dir
parent_dir_NF0016 = list(parent_dir_NF0016.glob("*/"))
parent_dir_NF0016 = [x for x in parent_dir_NF0016 if x.is_dir()]
# get all child files in the parent dir
file_dir_NF0016 = []
for parent_dir in parent_dir_NF0016:
    file_dir_NF0016.extend(list(parent_dir.glob("*")))


# In[8]:


# rename the files in the parent dir
for file in file_dir_NF0016:
    new_file_dir = pathlib.Path(
        f"{file.parent}/{str(file.stem).replace(' (60X)', '')}.{file.suffix}"
    )
    file.rename(new_file_dir)

# rename the parent dir
for parent_dir in parent_dir_NF0016:
    new_parent_dir = pathlib.Path(
        f"{parent_dir.parent}/{str(parent_dir.stem).replace(' (60X)', '')}"
    )
    # rename the parent dir
    os.rename(parent_dir, new_parent_dir)


# ## NF0018 specific preprocessing

# In[9]:


parent_dir_NF0018 = pathlib.Path("../../data/NF0018/raw_images").resolve(strict=True)
# get all dirs in the parent dir
parent_dir_NF0018 = list(parent_dir_NF0018.glob("*/"))
parent_dir_NF0018 = [x for x in parent_dir_NF0018 if x.is_dir()]
# get all child files in the parent dir
file_dir_NF0018 = []
for parent_dir in parent_dir_NF0018:
    file_dir_NF0018.extend(list(parent_dir.glob("*")))


# In[10]:


# rename the files in the parent dir
for file in file_dir_NF0018:
    new_file_dir = pathlib.Path(
        f"{file.parent}/{str(file.stem).replace(' (60X)', '')}{file.suffix}"
    )
    file.rename(new_file_dir)

# rename the parent dir
for parent_dir in parent_dir_NF0018:
    new_parent_dir = pathlib.Path(
        f"{parent_dir.parent}/{str(parent_dir.stem).replace(' (60X)', '')}"
    )
    # rename the parent dir
    os.rename(parent_dir, new_parent_dir)
