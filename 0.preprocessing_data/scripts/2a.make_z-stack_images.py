#!/usr/bin/env python
# coding: utf-8

# # Create z-stack images from the individual z-slice images for each FOV per well

# ## Import libraries

# In[ ]:


import pathlib
import pprint
import sys

import numpy as np
import pandas as pd
import tifffile as tiff

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
sys.path.append(str(root_dir / "utils"))
from notebook_init_utils import avoid_path_crash_bandicoot, init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# ## Set input and output directories

# In[ ]:


bandicoot_path = pathlib.Path("~/mnt/bandicoot").resolve()
if bandicoot_path.exists():
    bandicoot = True
else:
    bandicoot = False


# In[ ]:


raw_image_dir, output_base_dir = avoid_path_crash_bandicoot(bandicoot_path)


# In[ ]:


# # patient_ids
# patient_id_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
#     strict=True
# )
# list_of_patients = pd.read_csv(patient_id_file_path, header=None)[0].tolist()

list_of_patients = ["NF0035_T1"]


# In[ ]:


patient_input_dict = {}
for patient in list_of_patients:
    patient_input_dict[patient] = {
        "raw_images": pathlib.Path(
            f"{raw_image_dir}/data/{patient}/raw_images"
        ).resolve(),
        "zstack_output": pathlib.Path(
            f"{raw_image_dir}/data/{patient}/zstack_images"
        ).resolve(),
    }
pprint.pprint(patient_input_dict)


# In[ ]:


# Image extensions that we are looking to copy
image_extensions = {".tif", ".tiff"}


# In[ ]:


unlisted_images = {"patient": ["NF0014_T1"], "image_set": ["F11-3"]}


# In[ ]:


image_extensions = {".tif", ".tiff"}
channel_names = ["405", "488", "555", "640", "TRANS"]
# make a dictionary that contains a list for each channel name, storing both filepath and filename
channel_images = {
    channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
}
channel_images


# ## Create list of the well-site folders

# In[ ]:


# loop through patients, well_fovs, and each channel
# outer loop through patients
for patient in tqdm.tqdm(patient_input_dict.keys(), desc="Processing patients"):
    input_dirs = [
        x for x in patient_input_dict[patient]["raw_images"].iterdir() if x.is_dir()
    ]
    input_dirs.sort()

    for well_dir in tqdm.tqdm(
        input_dirs, desc=f"Processing wells for patient {patient}", leave=False
    ):
        channel_images = {
            channel_name: {"filename": [], "filepath": []}
            for channel_name in channel_names
        }
        if (
            patient in unlisted_images["patient"]
            and well_dir.name in unlisted_images["image_set"]
        ):
            print(f"Skipping {well_dir.name} because it is not listed.")
            continue

        for filename in well_dir.glob("*"):
            if filename.suffix in image_extensions:
                for channel_name in channel_names:
                    if channel_name in filename.name:
                        channel_images[channel_name]["filepath"].append(filename)

        for channel_name in tqdm.tqdm(
            channel_names, desc=f"Processing channels in {well_dir.name}", leave=False
        ):
            channel_images[channel_name]["filepath"] = sorted(
                channel_images[channel_name]["filepath"]
            )
            if not channel_images[channel_name]["filepath"]:
                print(
                    f"No files found for channel {channel_name} in {well_dir}. Skipping..."
                )
                continue

            images_to_stack = np.array(
                [
                    tiff.imread(filepath)
                    for filepath in channel_images[channel_name]["filepath"]
                ]
            )
            filepath = channel_images[channel_name]["filepath"][0]
            well = str(filepath.parent).split("/")[-1]
            output_path = (
                patient_input_dict[patient]["zstack_output"]
                / f"{well}"
                / f"{well}_{channel_name}.tif"
            )
            output_path.parent.mkdir(exist_ok=True, parents=True)
            tiff.imwrite(output_path, images_to_stack)
