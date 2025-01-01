#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment cells within organoids properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# ## import libraries

# In[ ]:


import argparse
import pathlib

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
import torch
from cellpose import core, models
from skimage import io

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## parse args and set paths

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
    parser.add_argument(
        "--window_size", type=int, help="Size of the window to use for the segmentation"
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        help="Clip limit for the adaptive histogram equalization",
    )

    args = parser.parse_args()
    window_size = args.window_size
    clip_limit = args.clip_limit
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../examples/raw_z_input/").resolve(strict=True)
    window_size = 3
    clip_limit = 0.05

mask_path = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve()
mask_path.mkdir(exist_ok=True, parents=True)


# ## Set up images, paths and functions

# In[3]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]


# In[4]:


# get the nuclei image
for f in files:
    if "405" in f:
        nuclei = io.imread(f)
nuclei = np.array(nuclei)
imgs = skimage.exposure.equalize_adapthist(nuclei, clip_limit=clip_limit)
original_imgs = imgs
print("Subsampled image shape:", imgs.shape)
print("number of z-slices:", imgs.shape[0])
original_z_slice_count = len(imgs)
print("number of z slices in the original image:", original_z_slice_count)


# In[5]:


# make a 2.5 D max projection image stack with a sliding window of 3 slices

image_stack_2_5D = np.empty((0, imgs.shape[1], imgs.shape[2]), dtype=imgs.dtype)
for image_index in range(imgs.shape[0]):
    image_stack_window = imgs[image_index : image_index + window_size]
    if not image_stack_window.shape[0] == window_size:
        break
    # max project the image stack
    image_stack_2_5D = np.append(
        image_stack_2_5D, np.max(image_stack_window, axis=0)[np.newaxis, :, :], axis=0
    )

imgs = np.array(image_stack_2_5D)
print("2.5D image stack shape:", image_stack_2_5D.shape)


# ## Cellpose

# In[6]:


use_GPU = torch.cuda.is_available()
# Load the model
model_name = "nuclei"
model = models.CellposeModel(gpu=use_GPU, model_type=model_name)

# Perform segmentation
labels, details, _ = model.eval(
    imgs, diameter=75, channels=[0, 0], z_axis=0, stitch_threshold=0.8
)


# <img src="../notebook_imgs/Sliding_window_unaggregate.jpg" alt="image" width="300"/>

# ## Reverse the sliding window max projection

# In[ ]:


# reverse sliding window max projection
full_mask_z_stack = []
reconstruction_dict = {index: [] for index in range(original_z_slice_count)}
print(f"Decoupling the sliding window max projection of {window_size} slices")

for z_stack_mask_index in range(len(labels)):
    z_stack_decouple = []
    [z_stack_decouple.append(labels[z_stack_mask_index]) for _ in range(window_size)]
    for z_window_index, z_stack_mask in enumerate(z_stack_decouple):
        if not (z_stack_mask_index + z_window_index) >= original_z_slice_count:
            reconstruction_dict[z_stack_mask_index + z_window_index].append(
                z_stack_mask
            )

# save the reconstruction_dict to a file for downstream decoupling
np.save(mask_path / "nuclei_reconstruction_dict.npy", reconstruction_dict)
