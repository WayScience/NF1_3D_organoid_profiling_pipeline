#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib

import imageio
import numpy as np
import skimage
import skimage.io as io
import tifffile

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
        choices=["nuclei", "cell", "cytoplasm", "organoid", "raw"],
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    compartment = args.compartment
    mask_input_dir = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve(
        strict=True
    )
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../../data/NF0014/zstack_images/C4-2/").resolve(
        strict=True
    )
    compartment = "raw"
    mask_input_dir = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve(
        strict=True
    )


output_path = pathlib.Path(f"../processed_data/{input_dir.stem}/gifs/").resolve()
output_path.mkdir(parents=True, exist_ok=True)

img_files = sorted(input_dir.glob("*"))
mask_files = sorted(mask_input_dir.glob("*"))


# ## Load images

# In[3]:


raw_channel_dict = {
    "405": None,
    "488": None,
    "555": None,
    "640": None,
    "TRANS": None,
}
for f in img_files:
    if compartment == "raw":
        if "405" in str(f.stem):
            raw_channel_dict["405"] = f
        elif "488" in str(f.stem):
            raw_channel_dict["488"] = f
        elif "555" in str(f.stem):
            raw_channel_dict["555"] = f
        elif "640" in str(f.stem):
            raw_channel_dict["640"] = f
        elif "TRANS" in str(f.stem):
            raw_channel_dict["TRANS"] = f

for f in mask_files:

    if compartment == "nuclei":
        if "nuclei" in str(f.stem) and "mask" in str(f.stem):
            mask_input_dir = f
            output_img_file_path = pathlib.Path(output_path / "nuclei_img_output.gif")
            output_mask_file_path = pathlib.Path(output_path / "nuclei_mask_output.gif")

    elif compartment == "cell":
        if "cell" in str(f.stem) and "mask" in str(f.stem):
            mask_input_dir = f
            output_img_file_path = pathlib.Path(output_path / "cell_img_output.gif")
            output_mask_file_path = pathlib.Path(output_path / "cell_mask_output.gif")
    elif compartment == "cytoplasm":
        if "cytoplasm" in str(f.stem) and "mask" in str(f.stem):
            mask_input_dir = f
            output_img_file_path = pathlib.Path(
                output_path / "cytoplasm_img_output.gif"
            )
            output_mask_file_path = pathlib.Path(
                output_path / "cytoplasm_mask_output.gif"
            )
    elif compartment == "organoid":
        if "organoid" in str(f.stem) and "mask" in str(f.stem):
            mask_input_dir = f
            output_img_file_path = pathlib.Path(output_path / "organoid_img_output.gif")
            output_mask_file_path = pathlib.Path(
                output_path / "organoid_mask_output.gif"
            )
    elif compartment == "raw":
        pass
    else:
        raise ValueError(
            "Invalid compartment, please choose either 'nuclei','cell', 'cytoplasm', or 'organoid"
        )

# read in the cell masks
if not compartment == "raw":
    mask = io.imread(mask_input_dir)

    # increase contrast of the image for visualization

    mask = skimage.exposure.rescale_intensity(mask, out_range=(0, 255))

    if np.unique(mask).max() < 256:
        mask = mask.astype("uint8")
    else:
        mask = mask.astype("uint16")


# In[4]:


duration = 0.001
loop = 0


# ### Cell image visualization

# In[5]:


import PIL.Image as Image

# In[6]:


channel_intesnity_dict = {"405": 250, "488": 200, "555": 200, "640": 50, "TRANS": 255}


# In[7]:


if compartment == "raw":
    for img_path in raw_channel_dict.keys():
        img = io.imread(raw_channel_dict[img_path])
        img = skimage.exposure.rescale_intensity(img, out_range=(0, 255))
        # img = skimage.exposure.equalize_adapthist(img, clip_limit=0.1)
        img = img / channel_intesnity_dict[img_path] * 255
        frames = [img[i] for i in range(img.shape[0])]
        raw_output_file_path = pathlib.Path(
            output_path / f"{img_path}_output.gif"
        ).resolve()
        # Write the frames to a GIF
        imageio.mimsave(raw_output_file_path, frames, duration=duration, loop=loop)
else:
    frames = [mask[i] for i in range(mask.shape[0])]

    # Write the frames to a GIF
    imageio.mimsave(
        output_mask_file_path, frames, duration=duration, loop=loop
    )  # duration is the time between frames in seconds
