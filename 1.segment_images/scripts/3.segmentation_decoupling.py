#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment cells within organoids properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# ## import libraries

# In[1]:


import argparse
import itertools
import pathlib
import sys

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import pandas as pd
import skimage
import tifffile
import tqdm
from skimage import io

sys.path.append("../../utils")
from segmentation_decoupling import (
    check_coordinate_inside_box,
    compare_masks_for_merged,
    euclidian_2D_distance,
    extract_unique_masks,
    get_combinations_of_indices,
    get_dimensionality,
    get_larger_bbox,
    get_number_of_unique_labels,
    merge_sets,
    merge_sets_df,
    reassemble_each_mask,
)

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# ## parse args and set paths

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
        default="none",
        help="Compartment to segment. Options are 'nuclei', 'cell', and 'organoid'",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Window size for the median filter",
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    compartment = args.compartment
    window_size = args.window_size
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../../data/NF0014/normalized_z/C4-2/").resolve(
        strict=True
    )
    compartment = "cell"
    window_size = 3

mask_path = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve()
mask_path.mkdir(exist_ok=True, parents=True)

if compartment == "nuclei":
    reconstruction_dict_path = pathlib.Path(
        mask_path / "nuclei_reconstruction_dict.npy"
    ).resolve(strict=True)
    mask_output_file_path = pathlib.Path(
        mask_path / "nuclei_masks_decoupled.tiff"
    ).resolve()
    distance_threshold = 10
elif compartment == "cell":
    reconstruction_dict_path = pathlib.Path(
        mask_path / "cell_reconstruction_dict.npy"
    ).resolve(strict=True)
    mask_output_file_path = pathlib.Path(
        mask_path / "cell_masks_decoupled.tiff"
    ).resolve()
    distance_threshold = 10
elif compartment == "organoid":
    reconstruction_dict_path = pathlib.Path(
        mask_path / "organoid_reconstruction_dict.npy"
    ).resolve(strict=True)
    mask_output_file_path = pathlib.Path(
        mask_path / "organoid_masks_decoupled.tiff"
    ).resolve()
    distance_threshold = 40
else:
    raise ValueError(
        "Invalid compartment, please choose 'nuclei', 'cell', or 'organoid'"
    )


# ## Set up images, paths and functions

# In[3]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]
# get the nuclei image
for f in files:
    if compartment == "nuclei" and "405" in f:
        imgs = io.imread(f)
    elif compartment == "cell" and "555" in f:
        imgs = io.imread(f)
    elif compartment == "organoid" and "488" in f:
        imgs = io.imread(f)
imgs = np.array(imgs)
original_imgs = imgs.copy()
original_img_shape = imgs.shape
original_z_slice_count = len(imgs)
print("number of z slices in the original image:", original_z_slice_count)


# In[4]:


reconstruction_dict = np.load(reconstruction_dict_path, allow_pickle=True).item()


# ## Reverse the sliding window max projection

# In[5]:


masks_dict = {}
for zslice, arrays in tqdm.tqdm(enumerate(reconstruction_dict)):
    df = extract_unique_masks(reconstruction_dict[zslice])
    merged_df = get_combinations_of_indices(df, distance_threshold=distance_threshold)
    # combine dfs for each window index
    # for window_index in range(window_size + 1):
    merged_df = merge_sets_df(merged_df)
    if not merged_df.empty:
        merged_df.loc[:, "slice"] = zslice
        reassembled_masks = reassemble_each_mask(
            merged_df, original_img_shape=original_imgs.shape
        )
        masks_dict[zslice] = reassembled_masks
    else:
        print(f"Warning: merged_df is empty for zslice {zslice}")
        masks_dict[zslice] = reconstruction_dict[zslice][0]


# In[6]:


# convert the masks_dict to a numpy array
masks = np.zeros(
    (original_z_slice_count, original_img_shape[1], original_img_shape[2]),
    dtype=np.uint16,
)
masks = np.array(list(masks_dict.values()))
if get_number_of_unique_labels(masks) > 255:
    masks = masks.astype(np.uint16)
else:
    masks = masks.astype(np.uint8)

# save the masks
tifffile.imwrite(mask_output_file_path, masks)


# In[7]:


if in_notebook:
    for zslice in range(masks.shape[0]):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(original_imgs[zslice], cmap="gray")
        plt.title(f"original z slice {zslice}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(masks[zslice], cmap="nipy_spectral")
        plt.title(f"segmented z slice {zslice}")
        plt.axis("off")
        plt.show()
