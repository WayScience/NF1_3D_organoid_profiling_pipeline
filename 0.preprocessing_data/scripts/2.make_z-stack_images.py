#!/usr/bin/env python
# coding: utf-8

# # Create z-stack images from the individual z-slice images for each FOV per well

# ## Import libraries

# In[1]:


import pathlib
import pprint

import numpy as np
import tifffile as tiff

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# ## Set input and output directories

# In[ ]:


list_of_patients = [  # will be in a separate file in the future
    "NF0014",
    "NF0016",
    # "NF0017",
    "NF0018",
    "NF0021",
    "SARCO219",
    "SARCO361",
]


# In[3]:


patient_input_dict = {}
for patient in list_of_patients:
    patient_input_dict[patient] = {
        "raw_images": pathlib.Path(f"../../data/{patient}/raw_images").resolve(),
        "zstack_output": pathlib.Path(f"../../data/{patient}/zstack_images").resolve(),
    }
pprint.pprint(patient_input_dict)


# In[4]:


# Image extensions that we are looking to copy
image_extensions = {".tif", ".tiff"}


# In[5]:


unlisted_images = {"patient": ["NF0014"], "image_set": ["F11-3"]}


# In[6]:


image_extensions = {".tif", ".tiff"}
channel_names = ["405", "488", "555", "640", "TRANS"]
# make a dictionary that contains a list for each channel name, storing both filepath and filename
channel_images = {
    channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
}
channel_images


# ## Create list of the well-site folders

# In[7]:


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
