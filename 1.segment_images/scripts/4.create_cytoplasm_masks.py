#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
import tifffile

# set import path
sys.path.append(str(pathlib.Path("../../utils/").resolve()))
from file_checking import check_number_of_files

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


output_path = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve()
output_path.mkdir(parents=True, exist_ok=True)
output_file_path = pathlib.Path(output_path / "cytoplasm_mask.tiff").resolve()

check_number_of_files(pathlib.Path(mask_input_dir), 4)


# In[3]:


# get all the masks
nuclei_masks_path = pathlib.Path(mask_input_dir / "nuclei_masks.tiff").resolve(
    strict=True
)
cell_masks_path = pathlib.Path(mask_input_dir / "cell_masks.tiff").resolve(strict=True)

nuclei_masks = io.imread(nuclei_masks_path)
cell_masks = io.imread(cell_masks_path)


# In[4]:


cytoplasm_masks = np.zeros_like(cell_masks)
for z_slice_index in range(nuclei_masks.shape[0]):
    nuclei_slice_mask = nuclei_masks[z_slice_index]
    cell_slice_mask = cell_masks[z_slice_index]
    cytoplasm_mask = cell_slice_mask.copy()
    cytoplasm_mask[nuclei_slice_mask > 0] = 0
    cytoplasm_masks[z_slice_index] = cytoplasm_mask


# In[ ]:


tifffile.imwrite(output_file_path, cytoplasm_masks)
check_number_of_files(pathlib.Path(mask_input_dir), 5)


# In[6]:


if in_notebook:
    for z in range(nuclei_masks.shape[0]):
        plt.subplot(1, 3, 1)
        plt.imshow(nuclei_masks[z])
        plt.title("Nuclei Mask")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(cell_masks[z])
        plt.title("Cell Mask")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(cytoplasm_masks[z])
        plt.title("Cytoplasm Mask")
        plt.axis("off")
        plt.show()
