#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import argparse
import json
import os
import pathlib
import string
import sys
from typing import Tuple

import numpy as np
import pandas as pd
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


# ## Help functions

# In[2]:


# remove leading zeros from FOVS and timepoints


def remove_leading_zeros(s: str) -> str:
    return str(int(s))


def cq1_file_name_parsing(file_stem: str) -> Tuple[str, str, str, str, str]:
    """
    Extracts readable filename metadata

    Parameters
    ----------
    file_stem : str
        The filename without extension

    Returns
    -------
    Tuple[str, str, str, str, str]
        Tuple with metadata
        - well: str
        - fov: str
        - time: str
        - z_slice: str
        - channel: str
    """
    well = file_stem.split("F")[0].strip("W")
    fov = remove_leading_zeros(file_stem.split("F")[1].split("T")[0])
    time = remove_leading_zeros(file_stem.split("T")[1].split("Z")[0])
    z_slice = remove_leading_zeros(file_stem.split("Z")[1].split("C")[0])
    channel = f"C{file_stem.split('C')[1]}"
    return (
        well,
        fov,
        time,
        z_slice,
        channel,
    )


# ## parse args and set paths

# In[3]:


argparse = argparse.ArgumentParser(
    description="Copy files from one directory to another"
)
argparse.add_argument("--HPC", action="store_true", help="Type of compute to run on")
# Parse arguments
args = argparse.parse_args(args=sys.argv[1:] if "ipykernel" not in sys.argv[0] else [])
HPC = args.HPC


# In[4]:


# check if bandicoot is set
bandicoot_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve()
if not HPC and bandicoot_path.exists():
    bandicoot = True
else:
    bandicoot = False

bandicoot = True


# In[5]:


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
        os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data/Raw_patient_files")
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

# In[6]:


# Define parent and destination directories in a single dictionary
"""
This dictionary maps patient IDs to their corresponding parent directories,
destination directories, and metadata for processing.
Nested dictionary name: patientID_tumorID
Keys:
- parent: Path to the parent directory containing raw images.
- destination: Path to the output directory for processed images.
"""

dir_mapping = {
    "NF0037_T1_CQ1": {
        "parent": pathlib.Path(f"{raw_image_dir}/NF0037_T1_CQ1-yokogawa/Image").resolve(
            strict=True
        ),
        "destination": pathlib.Path(
            f"{output_base_dir}/data/NF0037_T1_CQ1/zstack_images"
        ).resolve(),
    },
}


# ## Generate a well mapping

# In[7]:


well_map = {
    f"{i:04d}": f"{row}{col}"
    for i, (row, col) in enumerate(
        ((r, c) for r in string.ascii_uppercase[:8] for c in range(1, 13)), start=1
    )
}

# write the well map to a json file
path_to_processed_data = pathlib.Path("../mapping_utils/").resolve()
path_to_processed_data.mkdir(exist_ok=True, parents=True)
with open(path_to_processed_data / "well_map.json", "w") as f:
    json.dump(well_map, f)


# In[8]:


# set the channel dictionary
# this changes from dataset to dataset
channel_dict = {
    "C1": "405",
    "C2": "488",
    "C3": "555",  # technically 561 but using 555 for naming consistency
    "C4": "640",
    "C5": "TRANS",
}


# ## Copy and zstack images

# In[9]:


# image channel names and extensions
image_extensions = {".tif", ".tiff"}
channel_names = ["405", "488", "555", "640", "TRANS"]
channel_images = {
    channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
}


# In[10]:


# loop through each patient
patient = "NF0037_T1_CQ1"
files = list(dir_mapping[patient]["parent"].rglob("*.tif*"))


# In[11]:


df = pd.DataFrame(files, columns=["Filepath"])
df.insert(0, "Filename", df["Filepath"].apply(lambda x: x.stem))
df[["well", "fov", "time", "z_slice", "channel"]] = df["Filename"].apply(
    lambda x: pd.Series(cq1_file_name_parsing(x))
)
# map the well to the well_map
df["well"] = df["well"].map(well_map)
df["well_fov"] = df["well"] + "-" + df["fov"]
df["channel"] = df["channel"].map(channel_dict)
df.sort_values(by=["Filename"], inplace=True)
df.head()


# In[12]:


print(f"transfering {len(df['well_fov'].unique())} well_fovs")


# In[13]:


for well_fov in tqdm.tqdm(
    df["well_fov"].unique(), desc="Processing well_fovs", leave=True
):
    tmp_well_df = df[df["well_fov"] == well_fov]
    for channel in tqdm.tqdm(
        tmp_well_df["channel"].unique(), desc="Processing channels", leave=False
    ):
        tmp_well_channel_df = tmp_well_df[tmp_well_df["channel"] == channel]
        list_of_filepaths = tmp_well_channel_df["Filepath"].tolist()
        output_path = (
            dir_mapping[patient]["destination"]
            / f"{well_fov}"
            / f"{well_fov}_{channel}.tif"
        )
        if "nan" in str(output_path):
            print(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        if output_path.exists():
            continue
        images_to_stack = np.array(
            [read_2D_image_for_zstacking(file) for file in list_of_filepaths]
        )
        tifffile.imwrite(output_path, images_to_stack)
