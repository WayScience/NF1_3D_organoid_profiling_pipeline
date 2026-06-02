#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import tomli
import tqdm
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from matplotlib.colors import LinearSegmentedColormap

# In[2]:


OVERWRITE_COMPOSITES = False
WHOLE_MONTAGE = True
PIXEL_SIZE_MICRONS = 0.106
SCALE_BAR_LENGTH_MICRONS = 10


# In[3]:


def add_scale_bar(
    image: np.ndarray,
    pixel_size_microns: float,
    bar_length_microns: float = 10,
    bar_thickness: int = 10,
    padding_pixels: int = 10,
):
    """
    Add a scale bar to the bottom right corner of the image.

    Parameters:
    - image: 2D numpy array representing the image
    - pixel_size_microns: size of one pixel in microns
    - bar_length_microns: desired length of the scale bar in microns (default 10 microns)
    - bar_thickness: thickness of the scale bar in pixels (default 10 pixels)

    Returns:
    - image_with_bar: copy of the input image with the scale bar added
    """
    # Calculate the length of the scale bar in pixels
    bar_length_pixels = int(bar_length_microns / pixel_size_microns)

    # Create a copy of the image to draw on
    image_with_bar = np.array(image, copy=True)

    # Define the position for the scale bar (bottom right corner with some padding)
    start_x = image.shape[1] - padding_pixels - bar_length_pixels
    start_y = image.shape[0] - padding_pixels - bar_thickness

    # Draw the scale bar as white for RGB images, or as an over-range value for grayscale images
    if image_with_bar.ndim == 2:
        image_with_bar = image_with_bar.astype(np.float32, copy=False)
        image_with_bar[
            start_y : start_y + bar_thickness, start_x : start_x + bar_length_pixels
        ] = image_with_bar.max() + 1
    else:
        image_with_bar[
            start_y : start_y + bar_thickness, start_x : start_x + bar_length_pixels
        ] = 255

    return image_with_bar


def generate_montage(
    pixel_size_microns: float = PIXEL_SIZE_MICRONS,
    bar_length_microns: float = SCALE_BAR_LENGTH_MICRONS,
    full_montage: bool = True,
    composite_only: bool = False,
    **kwargs,
) -> plt.Figure:
    """Generate a montage of the 4 channels with a scale bar added to the composite RGB image.

    Parameters
    ----------
    pixel_size_microns: float
        Size of each pixel in microns
    bar_length_microns: float
        Length of the scale bar in microns
    kwargs: dictionary containing the following keys:
        agp_image_path: pathlib.Path
            file path to the AGP channel image
        dna_image_path: pathlib.Path
            file path to the DNA channel image
        er_image_path: pathlib.Path
            file path to the ER channel image
        mito_image_path: pathlib.Path
            file path to the Mito channel image
        contrast_dict: dictionary containing contrast limits for each channel (optional, if not provided will use 99.9th percentile for each channel)


    Returns
    -------
    montage: plt.Figure
        Matplotlib figure containing the montage of the 4 channels with scale bar
    """

    if composite_only and full_montage:
        raise ValueError(
            "Cannot set both composite_only and full_montage to True. Please choose one or the other."
        )

    # load images
    agp_image = tifffile.imread(kwargs["agp_image_path"])
    dna_image = tifffile.imread(kwargs["dna_image_path"])
    er_image = tifffile.imread(kwargs["er_image_path"])
    mito_image = tifffile.imread(kwargs["mito_image_path"])
    contrast_dict = kwargs.get("contrast_dict", None)

    # define cmaps: DNA -> cyan, Mito -> magenta, AGP -> green, ER -> red
    cyan_cmap = LinearSegmentedColormap.from_list("cyan_cmap", [(0, 0, 0), (0, 1, 1)])
    magenta_cmap = LinearSegmentedColormap.from_list(
        "magenta_cmap", [(0, 0, 0), (1, 0, 1)]
    )
    green_cmap = LinearSegmentedColormap.from_list("green_cmap", [(0, 0, 0), (0, 1, 0)])
    red_cmap = LinearSegmentedColormap.from_list("red_cmap", [(0, 0, 0), (1, 0, 0)])
    # ensure any over-range pixels (scale bar) render white
    cyan_cmap.set_over("white")
    magenta_cmap.set_over("white")
    green_cmap.set_over("white")
    red_cmap.set_over("white")

    # normalize each channel for display (clip to 99.9th percentile to reduce impact of outliers, then scale to [0, 255])
    def normalize_for_display(image, percentile=99.9, min_value=0, max_value=255):
        p999 = np.percentile(image, percentile)
        image_clipped = np.clip(image, min_value, p999)
        image_normalized = (image_clipped / p999) * (max_value - min_value) + min_value
        return image_normalized.astype(np.uint8)

    if contrast_dict:
        agp_display = normalize_for_display(
            agp_image,
            percentile=99.9,
            min_value=contrast_dict.get("AGP", (0, 255))[0],
            max_value=contrast_dict.get("AGP", (0, 255))[1],
        )
        dna_display = normalize_for_display(
            dna_image,
            percentile=99.9,
            min_value=contrast_dict.get("DNA", (0, 255))[0],
            max_value=contrast_dict.get("DNA", (0, 255))[1],
        )
        er_display = normalize_for_display(
            er_image,
            percentile=99.9,
            min_value=contrast_dict.get("ER", (0, 255))[0],
            max_value=contrast_dict.get("ER", (0, 255))[1],
        )
        mito_display = normalize_for_display(
            mito_image,
            percentile=99.9,
            min_value=contrast_dict.get("Mito", (0, 255))[0],
            max_value=contrast_dict.get("Mito", (0, 255))[1],
        )
    else:
        agp_display = normalize_for_display(agp_image, percentile=99.9)
        dna_display = normalize_for_display(dna_image, percentile=99.9)
        er_display = normalize_for_display(er_image, percentile=99.9)
        mito_display = normalize_for_display(mito_image, percentile=99.9)

    # add white scale bars to the single-channel display images
    agp_display_with_scale_bar = add_scale_bar(
        agp_display,
        pixel_size_microns=PIXEL_SIZE_MICRONS,
        bar_length_microns=SCALE_BAR_LENGTH_MICRONS,
    )
    er_display_with_scale_bar = add_scale_bar(
        er_display,
        pixel_size_microns=PIXEL_SIZE_MICRONS,
        bar_length_microns=SCALE_BAR_LENGTH_MICRONS,
    )
    mito_display_with_scale_bar = add_scale_bar(
        mito_display,
        pixel_size_microns=PIXEL_SIZE_MICRONS,
        bar_length_microns=SCALE_BAR_LENGTH_MICRONS,
    )
    dna_display_with_scale_bar = add_scale_bar(
        dna_display,
        pixel_size_microns=PIXEL_SIZE_MICRONS,
        bar_length_microns=SCALE_BAR_LENGTH_MICRONS,
    )

    # now take the images from the 4 channels and combine them into a single RGB image for visualization
    # normalize each channel to the range [0, 1]
    agp_norm = agp_display.astype(np.float32) / 255
    er_norm = er_display.astype(np.float32) / 255
    mito_norm = mito_display.astype(np.float32) / 255
    dna_norm = dna_display.astype(np.float32) / 255
    # build an RGB composite mapping: AGP->green, Mito->magenta (red+blue), DNA->cyan (green+blue), ER->red (reduced)
    rgb_image = np.zeros((*agp_norm.shape, 3), dtype=np.float32)
    rgb_image[..., 0] += mito_norm * 1.0  # red from mito (magenta)
    rgb_image[..., 0] += er_norm * 0.6  # red from ER (reduced)
    rgb_image[..., 1] += agp_norm * 1.0  # green from AGP
    rgb_image[..., 1] += dna_norm * 0.8  # green from DNA (cyan)
    rgb_image[..., 2] += mito_norm * 1.0  # blue from mito (magenta)
    rgb_image[..., 2] += dna_norm * 1.0  # blue from DNA (cyan)
    # clip to valid range
    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image_with_scale_bar = add_scale_bar(
        (rgb_image * 255).astype(np.uint8),
        pixel_size_microns=PIXEL_SIZE_MICRONS,
        bar_length_microns=SCALE_BAR_LENGTH_MICRONS,
    )

    if full_montage and not composite_only:
        # show the 5 image montage with the composite image in the center and the 4 individual channels around it
        fig, axs = plt.subplots(1, 5, figsize=(20, 10))
        axs[0].imshow(agp_display_with_scale_bar, cmap=green_cmap, vmin=0, vmax=255)
        axs[0].set_title("AGP")
        axs[1].imshow(dna_display_with_scale_bar, cmap=cyan_cmap, vmin=0, vmax=255)
        axs[1].set_title("DNA")
        axs[2].imshow(er_display_with_scale_bar, cmap=red_cmap, vmin=0, vmax=255)
        axs[2].set_title("ER")
        axs[3].imshow(mito_display_with_scale_bar, cmap=magenta_cmap, vmin=0, vmax=255)
        axs[3].set_title("Mito")
        axs[4].imshow(rgb_image_with_scale_bar)
        axs[4].set_title("Composite")
        # remove axes for all subplots
        for ax in axs.flatten():
            ax.axis("off")
        plt.tight_layout()
        plt.close(fig)
        return fig
    elif composite_only and not full_montage:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image_with_scale_bar)
        ax.axis("off")
        plt.tight_layout()
        plt.close(fig)
        return fig


# ## Get the paths

# In[4]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
image_base_dir = image_base_dir / "data"

drugs_plate_map_path = pathlib.Path(
    f"{root_dir}/config/platemaps/NF0014_T1_platemap.csv"
)
drugs_df = pd.read_csv(drugs_plate_map_path)


# In[5]:


output_dict = {
    "patient": [],
    "well_fov": [],
    "AGP_file_path": [],
    "DNA_file_path": [],
    "ER_file_path": [],
    "Mito_file_path": [],
    "Trans_file_path": [],
}
patients = image_base_dir.glob("*")
patients = [
    pathlib.Path(f"{p}/2D_analysis/0a.zmax_proj") for p in patients if p.is_dir()
]
# remove if any path does not exist
patients = [p for p in patients if p.exists()]
# remove patient NF0037_T1_CQ1

for patient_path in patients:
    well_fovs = patient_path.glob("*")
    well_fovs = [f for f in well_fovs if f.is_dir()]
    for well_fov in well_fovs:
        if well_fov.name in ["run_stats"]:
            continue
        files = [x for x in well_fov.glob("*") if x.is_file()]
        for f in files:
            if "555" in f.name:
                output_dict["AGP_file_path"].append(str(f))
            elif "405" in f.name:
                output_dict["DNA_file_path"].append(str(f))
            elif "488" in f.name:
                output_dict["ER_file_path"].append(str(f))
            elif "640" in f.name:
                output_dict["Mito_file_path"].append(str(f))
            elif "TRANS" in f.name:
                output_dict["Trans_file_path"].append(str(f))

        output_dict["patient"].append(patient_path.parent.parent.name)
        output_dict["well_fov"].append(well_fov.name)
df = pd.DataFrame(output_dict)
df.insert(2, "well", df["well_fov"].apply(lambda x: x.split("-")[0]))
df = df.merge(drugs_df, left_on=["well"], right_on=["WellPosition"], how="left")
df.dropna(inplace=True)
df["Dose"] = df["Dose"].astype(int)
df["full_treatment_name"] = df["Treatment"] + " " + df["Dose"].astype(str) + df["Unit"]
df["full_treatment_name_machine_readable"] = (
    df["Treatment"] + "_" + df["Dose"].astype(str) + df["Unit"]
)
df.head()


# ## merge the parameters and the file paths

# In[42]:


config_file_path = pathlib.Path(
    f"{root_dir}/figures/montage/config/montage_config.toml"
).resolve(strict=True)
# read in the config file
montage_config = tomli.load(config_file_path.open("rb"))
list_of_dfs = []
for key, value in montage_config.items():
    list_of_dfs.append(pd.DataFrame(value))
all_examples_df = pd.concat(list_of_dfs, ignore_index=True)
sampled_df = df.merge(all_examples_df, on=["patient", "well_fov"], how="inner")
print(sampled_df.shape)
# sort by patient for easier visualization
sampled_df = sampled_df.sort_values(by="patient")


# In[43]:


# generate montages for the example well_fovs listed above with individual contrast adjustments for each channel
for idx, row in tqdm.tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
    composite_image_path = pathlib.Path(
        f"../montages/individual_composites/{row['patient']}_{row['well_fov']}_{row['full_treatment_name_machine_readable']}_composite.png"
    ).resolve()
    composite_image_path.parent.mkdir(parents=True, exist_ok=True)
    if composite_image_path.exists() and not OVERWRITE_COMPOSITES:
        continue
    elif not composite_image_path.exists() or OVERWRITE_COMPOSITES:
        fig = generate_montage(
            agp_image_path=row["AGP_file_path"],
            dna_image_path=row["DNA_file_path"],
            er_image_path=row["ER_file_path"],
            mito_image_path=row["Mito_file_path"],
            composite_only=True,
            full_montage=False,
            contrast_dict={
                "AGP": (0, row["agp_max"]),
                "DNA": (0, row["dna_max"]),
                "ER": (0, row["er_max"]),
                "Mito": (0, row["mito_max"]),
            },
        )
        fig.savefig(
            composite_image_path,
            dpi=600,
        )
