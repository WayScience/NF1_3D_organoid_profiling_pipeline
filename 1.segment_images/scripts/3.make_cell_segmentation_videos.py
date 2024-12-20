#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


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


# ## Cells

# In[3]:


for f in img_files:
    if "555" in str(f):
        cell_img_path = f
for f in mask_files:
    if "cell" in str(f):
        cell_mask_path = f

# read in the cell masks
cell_img = io.imread(cell_img_path)
cell_mask = io.imread(cell_mask_path)

# scale the images to unit8
cell_img = (cell_img / 255).astype("uint8") * 8
cell_mask = (cell_mask).astype("uint8") * 16


# ### Cell image visualization

# In[4]:


frames = [cell_img[i] for i in range(cell_img.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "cell_img_output.gif")
imageio.mimsave(output_file_path, frames, duration=0.1, loop=10)


# ### Cell segmentation visualization

# In[5]:


frames = [cell_mask[i] for i in range(cell_mask.shape[0])]

# Write the frames to a GIF
output_file_path = pathlib.Path(output_path / "cell_mask_output.gif")
imageio.mimsave(
    output_file_path, frames, duration=0.1, loop=10
)  # duration is the time between frames in seconds
