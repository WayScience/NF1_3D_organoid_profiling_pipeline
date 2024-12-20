#!/usr/bin/env python
# coding: utf-8

# This notebook makes gifs of the z-stack segmentaions

# In[1]:


import argparse
import pathlib

import imageio
import numpy as np
import skimage
import skimage.io as io

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    mask_input_dir = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve(
        strict=True
    )
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../../data/z-stack_images/raw_z_input/").resolve(
        strict=True
    )
    mask_input_dir = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve(
        strict=True
    )


output_path = pathlib.Path(f"../processed_data/{input_dir.stem}/gifs/").resolve()
output_path.mkdir(parents=True, exist_ok=True)
img_files = sorted(input_dir.glob("*"))
mask_files = sorted(mask_input_dir.glob("*"))


# ## Nuclei

# ### Pathing and loading images

# In[3]:


for f in img_files:
    if "405" in str(f):
        nuclei_img_path = f
for f in mask_files:
    if "nuclei" in str(f):
        nuclei_mask_path = f

# read in the nucei masks
nuclei_img = io.imread(nuclei_img_path)
nuclei_mask = io.imread(nuclei_mask_path)

# scale the images to unit8
nuclei_img = (nuclei_img / 255).astype("uint8") * 2
nuclei_mask = (nuclei_mask).astype("uint8") * 16


# ### Image visualization

# In[4]:


frames = [nuclei_img[i] for i in range(nuclei_img.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "nuclei_img_output.gif")
imageio.mimsave(output_file_path, frames, duration=0.1, loop=10)


# ### Segmentation visualization

# In[5]:


frames = [nuclei_mask[i] for i in range(nuclei_mask.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "nuclei_mask_output.gif")
imageio.mimsave(output_file_path, frames, duration=0.1, loop=10)
