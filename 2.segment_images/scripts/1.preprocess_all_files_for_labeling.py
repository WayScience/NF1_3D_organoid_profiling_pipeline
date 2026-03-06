#!/usr/bin/env python
# coding: utf-8

# This notebook/script downsamples the original zstack images for each well FOV.
# These are used for FOV "morphology" annotation for segmentation.

# In[1]:


import argparse
import os
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import psutil
import skimage
import skimage.io
import tifffile
import torch
import torch.nn as nn
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.file_reading import *
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from matplotlib.colors import ListedColormap

# In[2]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[3]:


root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
# save path
image_file_path = pathlib.Path("../data/compressed_images/").resolve()
image_file_path.mkdir(parents=True, exist_ok=True)


# In[4]:


cyan_lut = np.zeros((256, 3))
cyan_lut[:, 1] = np.linspace(0, 1, 256)  # Green
cyan_lut[:, 2] = np.linspace(0, 1, 256)  # Blue
cyan_lut[0] = [0, 0, 0]


# In[5]:


patients_file_path = root_dir / "data/patient_IDs.txt"
patients = pd.read_csv(patients_file_path, header=None)[0].tolist()
for patient_id in tqdm.tqdm(patients, desc=f"Processing patients"):
    input_dir = pathlib.Path(
        f"{image_base_dir}/data/{patient_id}/zstack_images/"
    ).resolve(strict=True)
    # get all well_fov names for this patient
    image_paths = input_dir.rglob("*")
    image_paths = [
        x
        for x in image_paths
        if x.is_file() and x.suffix in [".tif", ".tiff"] and "555" in x.stem
    ]
    for image_path in tqdm.tqdm(image_paths, leave=False):
        save_path = pathlib.Path(
            f"{image_file_path}/{patient_id}_{image_path.stem}_downsampled.png"
        )
        if save_path.exists():
            continue
        # read the image
        image = read_zstack_image(image_path)
        # get the middle slice of the z-stack
        middle_slice = image[image.shape[0] // 2]
        # downsample the middle slice by a factor of 10 in each dimension
        downsampled_slice = middle_slice[::4, ::4].astype(np.uint8)
        # increase the brightness more aggressively and clip to 255
        if patient_id != "NF0037_T1_CQ1":
            downsampled_slice = np.clip(downsampled_slice * 5, 0, 255)
        else:
            # downsampled_slice = np.clip(downsampled_slice * 1, 0, 255)
            pass
        # make the image with a LUT of cyan
        cmap = ListedColormap(cyan_lut)
        # conver the downsampled slice to an RGB image using the LUT
        downsampled_slice = cmap(downsampled_slice)[:, :, :3] * 255
        # save the downsampled slice as a png
        skimage.io.imsave(
            save_path, downsampled_slice.astype(np.uint8), check_contrast=False
        )
