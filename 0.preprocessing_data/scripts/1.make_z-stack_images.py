#!/usr/bin/env python
# coding: utf-8

# # Create z-stack images from the individual z-slice images for each FOV per well

# ## Import libraries

# In[1]:


import pathlib

import numpy as np
import tifffile as tiff
import tqdm


# ## Set input and output directories

# In[2]:


input_dir = pathlib.Path("../../data/NF0014_raw_images").resolve(strict=True)

output_z_stack_dir = pathlib.Path("../../data/NF0014_zstack_images").resolve()
output_z_stack_dir.mkdir(exist_ok=True, parents=True)


# ## Create list of the well-site folders

# In[3]:


# get a list of all dirs in the input dir
input_dirs = [x for x in input_dir.iterdir() if x.is_dir()]
input_dirs.sort()
print(f"There are {len(input_dirs)} directories in the input directory.")


# ## For the NF0014 dataset, remove the files for one of the z-slices due to missing channel (640)

# In[4]:


# Remove files containing 'ZS000' (first z-slice) in the F11-3 directory because it is missing 640 channel
for file in pathlib.Path(f'{input_dir}/F11-3').glob('*ZS000*'):
    file.unlink()
    print(f"Removed: {file}")


# ## Set the dictionary for filenames and filepaths

# In[5]:


image_extensions = {".tif", ".tiff"}
channel_names = ["405", "488", "555", "640", "TRANS"]
# make a dictionary that contains a list for each channel name, storing both filepath and filename
channel_images = {channel_name: {"filename": [], "filepath": []} for channel_name in channel_names}
channel_images


# ## Loop thorugh and create z-stack images for each FOV of each well in their respective directories.

# In[6]:


for well_dir in tqdm.tqdm(input_dirs):
    print(f"Processing well_dir: {well_dir.stem}")  # Debug
    channel_images = {channel_name: {"filepath": []} for channel_name in channel_names}
    
    # Get all the images in the directory
    for filename in well_dir.glob("*"):
        if filename.suffix in image_extensions:
            for channel_name in channel_names:
                if channel_name in filename.name:
                    channel_images[channel_name]["filepath"].append(filename)
                    break

    # Iterate through each channel
    for channel_name in channel_names:
        # Sort and check filepaths
        channel_images[channel_name]["filepath"] = sorted(
            channel_images[channel_name]["filepath"]
        )
        if not channel_images[channel_name]["filepath"]:
            print(f"No files found for channel {channel_name} in {well_dir}. Skipping...")
            continue

        # Confirm before accessing
        print(f"Stacking channel {channel_name}")  # Debug
    
        # read the image files from the sorted file paths and stack them into a numpy array
        images_to_stack = np.array(
            [
                tiff.imread(filepath)
                for filepath in channel_images[channel_name]["filepath"]
            ]
        )
        filepath = channel_images[channel_name]["filepath"][0]
        well = str(filepath.parent).split("/")[-1]
        output_path = output_z_stack_dir / f"{well}" / f"{well}_{channel_name}.tif"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        tiff.imwrite(output_path, images_to_stack)

