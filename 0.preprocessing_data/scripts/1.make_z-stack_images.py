#!/usr/bin/env python
# coding: utf-8

# This notebook creats z-stack images from the individual z-slice images for each FOV of each well.

# In[1]:


import pathlib

import numpy as np
import tifffile as tiff
import tqdm

# In[2]:


input_dir = pathlib.Path("../../data/raw_images").resolve(strict=True)

output_z_stack_dir = pathlib.Path("../../data/z-stack_images").resolve()
output_z_stack_dir.mkdir(exist_ok=True, parents=True)


# In[3]:


# get a list of all dirs in the input dir
input_dirs = [x for x in input_dir.iterdir() if x.is_dir()]
print(f"There are {len(input_dirs)} directories in the input directory.")


# In[4]:


image_extensions = {".tif", ".tiff"}
channel_names = ["405", "488", "555", "640", "TRANS"]
# make a dictionary that contains a list for each channel name
channel_images = {
    channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
}
channel_images


# In[5]:


for well_dir in tqdm.tqdm(input_dirs):
    channel_images = {
        channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
    }
    # get all the images in the directory
    images_names = []
    for filename in well_dir.glob("*"):
        if filename.suffix in image_extensions:
            for channel_name in channel_names:
                if channel_name in filename.name:
                    channel_images[channel_name]["filename"].append(filename.name)
                    channel_images[channel_name]["filepath"].append(filename)
                    break

    # sort the lists of filenames and filepaths
    for channel_name in channel_names:
        channel_images[channel_name]["filename"] = sorted(
            channel_images[channel_name]["filename"]
        )
        channel_images[channel_name]["filepath"] = sorted(
            channel_images[channel_name]["filepath"]
        )
        images_to_stack = []
        for filepath in channel_images[channel_name]["filepath"]:
            images_to_stack.append(tiff.imread(filepath))
        well = str(filepath.parent).split("/")[-1]
        images_to_stack = np.array(images_to_stack)
        output_path = output_z_stack_dir / f"{well}" / f"{channel_name}.tif"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        tiff.imwrite(output_path, images_to_stack)
