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
    parser.add_argument(
        "--compartment",
        type=str,
        help="The compartment to segment the nuclei from",
        choices=["nuclei", "cell", "cytoplasm"],
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    compatment = args.compartment
    mask_input_dir = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve(
        strict=True
    )
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../../data/z-stack_images/raw_z_input/").resolve(
        strict=True
    )
    compartment = "nuclei"
    mask_input_dir = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve(
        strict=True
    )


output_path = pathlib.Path(f"../processed_data/{input_dir.stem}/gifs/").resolve()
output_path.mkdir(parents=True, exist_ok=True)

img_files = sorted(input_dir.glob("*"))
mask_files = sorted(mask_input_dir.glob("*"))


# ## Load images

# In[ ]:


for f in img_files:
    if compartment == "nuclei":
        if "405" in str(f.stem):
            img_path = f
    elif compartment == "cell":
        if "555" in str(f.stem):
            img_path = f
    elif compartment == "cytoplasm":
        if "555" in str(f.stem):
            img_path = f

for f in mask_files:

    if compartment == "nuclei":
        if "nuclei" in str(f.stem):
            mask_path = f
            output_img_file_path = pathlib.Path(output_path / "nuclei_img_output.gif")
            output_mask_file_path = pathlib.Path(output_path / "nuclei_mask_output.gif")

    elif compartment == "cell":
        if "cell" in str(f.stem):
            mask_path = f
            output_img_file_path = pathlib.Path(output_path / "cell_img_output.gif")
            output_mask_file_path = pathlib.Path(output_path / "cell_mask_output.gif")
    elif compartment == "cytoplasm":
        if "cytoplasm" in str(f.stem):
            mask_path = f
            output_img_file_path = pathlib.Path(
                output_path / "cytoplasm_img_output.gif"
            )
            output_mask_file_path = pathlib.Path(
                output_path / "cytoplasm_mask_output.gif"
            )
    else:
        raise ValueError("Invalid compartment, please choose either 'nuclei' or 'cell'")

# read in the cell masks
img = io.imread(img_path)
mask = io.imread(mask_path)

# scale the images to unit8
img = (img / 255).astype("uint8") * 8
mask = (mask).astype("uint8") * 16


# ### Cell image visualization

# In[4]:


frames = [img[i] for i in range(img.shape[0])]

# Write the frames to a GIF
imageio.mimsave(output_img_file_path, frames, duration=0.1, loop=10)


# ### Cell segmentation visualization

# In[5]:


frames = [mask[i] for i in range(mask.shape[0])]

# Write the frames to a GIF
imageio.mimsave(
    output_mask_file_path, frames, duration=0.1, loop=10
)  # duration is the time between frames in seconds
