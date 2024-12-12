#!/usr/bin/env python
# coding: utf-8

# This notebook focuses on trying to find a way to segment cells within organoids properly.
# The end goals is to segment cell and extract morphology features from cellprofiler.
# These masks must be imported into cellprofiler to extract features.

# In[1]:


import argparse
import pathlib

import matplotlib.pyplot as plt

# Import dependencies
import numpy as np
import skimage
import tifffile
from cellpose import core, io, models

io.logger_setup()
from cellpose.io import imread
from PIL import Image
from skimage import io

use_GPU = core.use_gpu()
print(">>> GPU activated? %d" % use_GPU)


# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

print(in_notebook)


# In[2]:


if not in_notebook:
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
    input_dir = pathlib.Path("../examples/raw_z_input/").resolve(strict=True)
    window_size = 3
    clip_limit = 0.1

mask_path = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve()
mask_path.mkdir(exist_ok=True, parents=True)


# ## Set up images, paths and functions

# In[3]:


image_extensions = {".tif", ".tiff"}
files = sorted(input_dir.glob("*"))
files = [str(x) for x in files if x.suffix in image_extensions]


# In[4]:


for f in files:
    print(f)
    if "405" in f:
        nuclei = io.imread(f)
    elif "488" in f:
        cyto1 = io.imread(f)
    elif "555" in f:
        cyto2 = io.imread(f)
    elif "640" in f:
        cyto3 = io.imread(f)
    else:
        print(f"Unknown channel: {f}")

# pick which channels to use for cellpose
cyto = skimage.exposure.equalize_adapthist(cyto2, clip_limit=clip_limit)


original_nuclei_image = nuclei.copy()
original_cyto_image = cyto.copy()

original_nuclei_z_count = nuclei.shape[0]
original_cyto_z_count = cyto.shape[0]


# In[5]:


# make a 2.5 D max projection image stack with a sliding window of 3 slices
image_stack_2_5D = np.empty((0, cyto.shape[1], cyto.shape[2]), dtype=cyto.dtype)
for image_index in range(cyto.shape[0]):
    image_stack_window = cyto[image_index : image_index + window_size]
    if not image_stack_window.shape[0] == window_size:
        break
    # max project the image stack
    image_stack_2_5D = np.append(
        image_stack_2_5D, np.max(image_stack_window, axis=0)[np.newaxis, :, :], axis=0
    )

image_stack_2_5D = np.array(image_stack_2_5D)
cyto = np.array(image_stack_2_5D)
print("2.5D cyto image stack shape:", cyto.shape)


# make a 2.5 D max projection image stack with a sliding window of 3 slices
image_stack_2_5D = np.empty((0, nuclei.shape[1], nuclei.shape[2]), dtype=nuclei.dtype)
for image_index in range(nuclei.shape[0]):
    image_stack_window = nuclei[image_index : image_index + window_size]
    if not image_stack_window.shape[0] == window_size:
        break
    # max project the image stack
    image_stack_2_5D = np.append(
        image_stack_2_5D, np.max(image_stack_window, axis=0)[np.newaxis, :, :], axis=0
    )

image_stack_2_5D = np.array(image_stack_2_5D)
nuclei = np.array(image_stack_2_5D)
print("2.5D nuclei image stack shape:", nuclei.shape)


# In[6]:


if in_notebook:
    # plot the nuclei and the cyto channels
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(nuclei[9, :, :], cmap="gray")
    plt.title("nuclei")
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(cyto[9, :, :], cmap="gray")
    plt.title("cyto")
    plt.axis("off")
    plt.show()


# In[7]:


imgs = []
# save each z-slice as an RGB png
for z in range(nuclei.shape[0]):

    nuclei_tmp = nuclei[z, :, :]
    cyto_tmp = cyto[z, :, :]
    nuclei_tmp = (nuclei_tmp / nuclei_tmp.max() * 255).astype(np.uint8)
    cyto_tmp = (cyto_tmp / cyto_tmp.max() * 255).astype(np.uint8)
    # save the image as an RGB png with nuclei in blue and cytoplasm in red
    RGB = np.stack([cyto_tmp, np.zeros_like(cyto_tmp), nuclei_tmp], axis=-1)

    # change to 8-bit
    RGB = (RGB / RGB.max() * 255).astype(np.uint8)

    rgb_image_pil = Image.fromarray(RGB)

    imgs.append(rgb_image_pil)


# ## Cellpose

# In[8]:


# model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
model = models.Cellpose(model_type="cyto3", gpu=use_GPU)

channels = [[1, 3]]  # channels=[red cells, blue nuclei]
diameter = 200

masks_all_dict = {"masks": [], "imgs": []}
imgs = np.array(imgs)

for img in imgs:
    masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels)
    masks_all_dict["masks"].append(masks)
    masks_all_dict["imgs"].append(img)
print(len(masks_all_dict))


# In[9]:


masks_all = masks_all_dict["masks"]
imgs = masks_all_dict["imgs"]


# In[10]:


# masks, flows, styles, diams
plot = plt.figure(figsize=(10, 5))
if in_notebook:
    for z in range(len(masks_all)):
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(masks_all[z], cmap="gray")
        plt.title(f"mask: {z}")
        plt.subplot(122)
        plt.imshow(imgs[z], cmap="gray")
        plt.title(f"raw: {z}")
        plt.show()


# In[11]:


# reverse sliding window max projection
full_mask_z_stack = []
reconstruction_dict = {index: [] for index in range(original_cyto_z_count)}
print(f"Decoupling the sliding window max projection of {window_size} slices")

for z_stack_mask_index in range(len(masks_all)):
    z_stack_decouple = []
    [z_stack_decouple.append(masks_all[z_stack_mask_index]) for _ in range(window_size)]
    for z_window_index, z_stack_mask in enumerate(z_stack_decouple):
        if not (z_stack_mask_index + z_window_index) >= original_cyto_z_count:
            reconstruction_dict[z_stack_mask_index + z_window_index].append(
                z_stack_mask
            )


# In[12]:


# max project each z position back to the original image
reconstructed_masks = np.empty(
    (0, masks_all[0].shape[0], masks_all[0].shape[1]), dtype=masks_all[0].dtype
)


for z_index in range(original_cyto_z_count):
    z_stack_masks = np.array(reconstruction_dict[z_index])
    z_stack_max_projected = np.max(z_stack_masks, axis=0)[np.newaxis, :, :]
    reconstructed_masks = np.append(reconstructed_masks, z_stack_max_projected, axis=0)

print(reconstructed_masks.shape)
if in_notebook:
    # show each z slice of the image and masks
    for z in range(reconstructed_masks.shape[0]):
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(131)
        plt.imshow(original_nuclei_image[z, :, :], cmap="gray")
        plt.title("Nuclei")
        plt.axis("off")
        plt.subplot(132)
        plt.imshow(original_cyto_image[z, :, :], cmap="gray")
        plt.title("Cells")
        plt.axis("off")
        plt.subplot(133)
        plt.imshow(reconstructed_masks[z], cmap="magma")
        plt.title("masks")
        plt.axis("off")
        plt.show()


# In[13]:


# # save the masks
print(reconstructed_masks.shape)
# save the masks as tiff
mask_file_path = pathlib.Path(mask_path / "cell_masks.tiff").resolve()
tifffile.imsave(mask_file_path, reconstructed_masks)
