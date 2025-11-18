#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import argparse
import os
import pathlib
import re
import sys

import numpy as np
import tifffile
import tqdm
from notebook_init_utils import avoid_path_crash_bandicoot, init_notebook
from preprocessing_funcs import (
    check_well_dir_name_format,
    get_to_the_unested_dir,
    get_well_fov_dirs,
    read_2D_image_for_zstacking,
)

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


argparse = argparse.ArgumentParser(
    description="Copy files from one directory to another"
)
argparse.add_argument("--HPC", action="store_true", help="Type of compute to run on")
# Parse arguments
args = argparse.parse_args(args=sys.argv[1:] if "ipykernel" not in sys.argv[0] else [])
HPC = args.HPC


# In[3]:


# check if bandicoot is set
bandicoot_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve()
if not HPC and bandicoot_path.exists():
    bandicoot = True
else:
    bandicoot = False

bandicoot = True


# In[4]:


if HPC:
    raw_image_dir = pathlib.Path("/pl/active/koala/GFF_Data/GFF-Raw/").resolve(
        strict=True
    )
    output_base_dir = root_dir
elif bandicoot:
    # comment out depending on whose computer you are on
    # mike's computer
    bandicoot_path = pathlib.Path(
        os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")
    ).resolve(strict=True)
    raw_image_dir = pathlib.Path(
        os.path.expanduser("~/Desktop/20TB_A/NF1_Patient_organoids")
    ).resolve(strict=True)
    output_base_dir = bandicoot_path
else:
    # comment out depending on whose computer you are on
    # mike's computer
    raw_image_dir = pathlib.Path(
        os.path.expanduser("~/Desktop/20TB_A/NF1_Patient_organoids")
    ).resolve(strict=True)
    # Jenna's computer
    # raw_image_dir_local = pathlib.Path("/media/18tbdrive/GFF_organoid_data/")
    output_base_dir = root_dir
print(f"Raw image dir: {raw_image_dir}")
print(f"Output base dir: {output_base_dir}")


# ## Define paths

# In[5]:


# Define parent and destination directories in a single dictionary
"""
This dictionary maps patient IDs to their corresponding parent directories,
destination directories, and metadata for processing.
Nested dictionary name: patientID_tumorID
Keys:
- parent: Path to the parent directory containing raw images.
- destination: Path to the output directory for processed images.
- times_nested: Number of nested timepoint directories.
- well_position: Index of the well position in the filename when split by underscores.
- channel_position: Index of the channel position in the filename when split by underscores.
"""

dir_mapping = {
    "NF0014_T1": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0014-Thawed 3 (Raw image files)-Combined/NF0014-Thawed 3 (Raw image files)-Combined copy"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0014_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0014_T2": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0014-T2 Cell Painting/NF0014-T2 Combined/"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0014_T2/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0016_T1": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0016 Cell Painting-Pilot Drug Screening-selected/NF0016-Cell Painting Images/NF0016-images copy"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0016_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    # "NF0017": {
    #     "parent": pathlib.Path(
    #         f"{raw_image_dir}/NF0017-T3-P7 (AGP, Mito Parameter optimization)/Acquisition 03-07-2025"
    #     ).resolve(strict=True),
    #     "destination": pathlib.Path(
    #         f"{output_base_dir}/data/NF0017/zstack_images"
    #     ).resolve(),
    #     "times_nested": 2,
    #     "well_position": 0,
    #     "channel_position": 1,
    # },
    "NF0018_T6": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0018 (T6) Cell Painting-Pilot Drug Screeining/NF0018-Cell Painting Images/NF0018-All Acquisitions"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0018_T6/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0021_T1": {
        "parent": pathlib.Path(f"{raw_image_dir}/NF0021-T1/NF0021-T1 Combined").resolve(
            strict=True
        ),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0021_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0030_T1": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0030 Cell Painting/NF0030 Cell Painting/NF0030-Cell Painting Images/Combined"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0030_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0035_T1_part_I": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0035-T1-Part-1/NF0035-T1-Combined Part-1"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0035_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0035_T1_part_II": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0035-T1-Part-2/NF0035-T1-Combined Part-2"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0035_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "NF0037_T1_part_I": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0031-T1 Combined 1_2/NF0031-T1 Combined 1:2"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1/zstack_images"
        ).resolve(),
        "times_nested": 0,
        "well_position": -1,
        "channel_position": -3,
    },
    "NF0037_T1_part_II": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0031-T1 Combined 2_2/NF0031-T1 Combined 2:2"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1/zstack_images"
        ).resolve(),
        "times_nested": 0,
        "well_position": -1,
        "channel_position": -3,
    },
    "NF0037-T1-Z-1": {
        "parent": pathlib.Path(f"{raw_image_dir}/NF0037-T1-Z-1/NF0037-T1-Z-1").resolve(
            strict=True
        ),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1-Z-1/zstack_images"
        ).resolve(),
        "times_nested": 0,
        "well_position": -1,
        "channel_position": -3,
    },
    "NF0037-T1-Z-0.5": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0037-T1-Z-0.5/NF0037-T1-ZTEST2-0.5"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1-Z-0.5/zstack_images"
        ).resolve(),
        "times_nested": 0,
        "well_position": -1,
        "channel_position": -3,
    },
    "NF0037-T1-Z-0.2": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0037-T1-Z-0.2/NF0037-T1-Z-0.2/NF0037-T1-ZTEST2-0.2"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1-Z-0.2/zstack_images"
        ).resolve(),
        "times_nested": 0,
        "well_position": -1,
        "channel_position": -3,
    },
    "NF0037-T1-Z-0.1": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/NF0037-T1-Z-0.1/NF0037-T1-Z-0.1/NF0037-T1-ZTEST2"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1-Z-0.1/zstack_images"
        ).resolve(),
        "times_nested": 0,
        "well_position": -1,
        "channel_position": -3,
    },
    "NF0040_T1": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/SARC0376 (NF0040) Cell Painting/SARC0376 (NF0040) Cell Painting/SARC0376 (NF0040)-Cell Painting Images/Combined"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0040_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "SARCO219_T2": {
        "parent": pathlib.Path(
            f"{raw_image_dir}/SARC0219-T2 Cell Painting-selected/SARC0219-T2 Combined Cell Painting images/SARC0219-T2 Combined/"
        ).resolve(strict=True),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/SARCO219_T2/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
    "SARCO361_T1": {
        "parent": pathlib.Path(f"{raw_image_dir}/SARC0361/SARC0361 Combined/").resolve(
            strict=True
        ),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/SARCO361_T1/zstack_images"
        ).resolve(),
        "times_nested": 2,
        "well_position": 0,
        "channel_position": 1,
    },
}


# ## Copy and zstack images

# In[6]:


# image channel names and extensions
image_extensions = {".tif", ".tiff"}
channel_names = ["405", "488", "555", "640", "TRANS"]
# make a dictionary that contains a list for each channel name, storing both filepath and filename
channel_images = {
    channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
}


# In[7]:


# loop through each patient
for patient in tqdm.tqdm(dir_mapping.keys(), desc="Processing patients", leave=True):
    # get a list of all well directories and loop through them
    all_well_dirs = get_well_fov_dirs(parent_dir=dir_mapping[patient]["parent"])

    for well_dir in tqdm.tqdm(
        all_well_dirs, desc=f"Processing wells for patient {patient}", leave=False
    ):
        # ensure that the dir follows the alpha numeric - numeric format
        well_dir_name = well_dir.name
        if not check_well_dir_name_format(well_dir_name):
            print(f"Skipping directory with unexpected name format: {well_dir_name}")
            continue
        # step through the nested directories to get to the most branched child directory
        most_branched_child_dir = get_to_the_unested_dir(
            nested_dir=pathlib.Path(well_dir),
            times_nested=dir_mapping[patient]["times_nested"],
        )
        if most_branched_child_dir is None:
            print(f"Could not un-nest directory: {well_dir}")
            continue

        # get a list of files, sort thenm, and loop through them to categorize by channel
        files = [f for f in most_branched_child_dir.glob("*") if f.is_file()]
        files.sort()
        channel_images = {
            channel_name: {"filename": [], "filepath": []}
            for channel_name in channel_names
        }
        for file in files:
            if file.suffix in image_extensions:
                if "tile" in file.stem.lower():
                    channel = file.stem.split("_")[
                        dir_mapping[patient]["channel_position"] - 1
                    ]
                else:
                    channel = file.stem.split("_")[
                        dir_mapping[patient]["channel_position"]
                    ]
                if channel in channel_images:
                    channel_images[channel]["filename"].append(file.name)
                    channel_images[channel]["filepath"].append(str(file))

        # loop through each channel and create z-stack tiffs
        for channel_name in tqdm.tqdm(
            channel_names,
            desc=f"Processing channels for well {well_dir_name} of patient {patient}",
            leave=False,
        ):
            channel_images[channel_name]["filename"] = sorted(
                channel_images[channel_name]["filename"]
            )
            if not channel_images[channel_name]["filename"]:
                print(f"No images found for channel {channel_name}")
                continue

            # generate filename below
            filepath = channel_images[channel_name]["filepath"][0]

            cleaned_filepath = filepath
            patterns = [
                r"\s*\(60X\)\s*",
                r"\s*\(NO AUTOLEVELS\)\s*",
                # r"-NEW\b",                       # remove literal "-NEW" only (word boundary)
                r"\s*\(NO AUTOLEVELS-1\)\s*",
                r"\s*\( NO AUTOLEVELS-2\)\s*",
            ]
            for pat in patterns:
                cleaned_filepath = re.sub(
                    pat, "", cleaned_filepath, flags=re.IGNORECASE
                ).strip()

            well = pathlib.Path(cleaned_filepath).parent.stem.split("_")[
                dir_mapping[patient]["well_position"]
            ]

            output_path = (
                dir_mapping[patient]["destination"]
                / f"{well}"
                / f"{well}_{channel_name}.tif"
            )
            # generate output directory if it doesn't exist
            # and check if output file already exists
            output_path.parent.mkdir(exist_ok=True, parents=True)
            if output_path.exists():
                continue
            # create z-stack tiff by reading in each 2D image and stacking them
            images_to_stack = np.array(
                [
                    read_2D_image_for_zstacking(file)
                    for file in channel_images[channel_name]["filepath"]
                ]
            )
            # write the stacked images to a tiff file
            tifffile.imwrite(output_path, images_to_stack)
