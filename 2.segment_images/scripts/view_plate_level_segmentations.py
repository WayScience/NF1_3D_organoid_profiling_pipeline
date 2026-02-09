#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import os
import pathlib
import string
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook
from skimage import exposure

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


if not in_notebook:
    args = parse_args()
    clip_limit = args["clip_limit"]
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    mask_subparent_name = args["mask_subparent_name"]
    check_for_missing_args(
        patient=patient,
        input_subparent_name=input_subparent_name,
        mask_subparent_name=mask_subparent_name,
    )
else:
    print("Running in a notebook")
    patient = "NF0014_T1"
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"


figures_path = pathlib.Path(f"../figures/platemaps/").resolve()
figures_path.mkdir(parents=True, exist_ok=True)
patients_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(patients_file_path, header=None)[0].tolist()


# In[ ]:


def plot_plate_overview(
    plate: str,
    image_sub_string_to_search: str,
    title_for_substring: str,
    available_wells: dict,
    layout: int = "96",
    skip_outer_wells: bool = True,
    image_color_map: str = "nipy_spectral",
    lut: np.ndarray | None = None,
    contrast_enhance: bool = True,
    clip_limit: float = 0.03,
) -> plt.Figure:
    """
    Generate a plate-view of images from each well

    Parameters
    ----------
    plate : str
        plate identifier
    image_sub_string_to_search : str
        Substring to search for in image filenames within each well directory
    available_wells : dict
        Dictionary mapping well positions to their corresponding directories
        Dictionary is in the following format:
            {"well_position": pathlib.Path("path/to/well_directory"), ...}
    layout : int, optional
        Plate layout, by default "96"
    image_color_map : str, optional
        Colormap for images, by default "nipy_spectral"
    lut : np.ndarray | None, optional
        Custom lookup table for coloring label images. If provided, image_color_map is ignored.
        Should be an Nx3 or Nx4 array of RGB/RGBA values (0-1 or 0-255 range).
        by default None
    contrast_enhance : bool, optional
        Whether to apply contrast enhancement to images, by default True
    clip_limit : float, optional
        Clip limit for contrast enhancement, by default 0.03
    """
    if layout == "96" and skip_outer_wells:  # only the inner 60 wells
        rows = list(string.ascii_uppercase[1:7])
        cols = list(range(2, 12))
    elif layout == "96" and not skip_outer_wells:
        rows = list(string.ascii_uppercase[0:8])
        cols = list(range(1, 13))
    elif layout == "384" and skip_outer_wells:  # only the inner 308 wells
        rows = list(string.ascii_uppercase[2:15])  # B-O
        cols = list(range(2, 24))  # 2-23
    elif layout == "384" and not skip_outer_wells:
        rows = list(string.ascii_uppercase[0:16])  # A-P
        cols = list(range(1, 25))  # 1-24
    else:
        raise ValueError(f"Unsupported plate layout: {layout}")

    # Create figure with minimal spacing
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(20, 12),  # W X H
        gridspec_kw={
            "wspace": 0.02,
            "hspace": 0.02,
        },  # Minimal spacing between subplots
    )
    fontsize = 16
    fig.suptitle(
        f"{layout}-Well Plate Overview\nPlate: {plate}\n{title_for_substring}",
        fontsize=fontsize,
        fontweight="bold",
        y=0.98,
    )

    # Show in a grid one image per plate well
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            well_position = f"{row}{col:02d}"
            well_position_no_zero = f"{row}{col}"

            # Check if this well has data
            if (
                well_position_no_zero in available_wells
                or well_position in available_wells
            ):
                well_dir = available_wells.get(
                    well_position_no_zero
                ) or available_wells.get(well_position)

                # Get the first image file from this well
                image_files = sorted(
                    list(well_dir.glob(f"*{image_sub_string_to_search}*"))
                )
                if image_files:
                    nuclei_mask = read_zstack_image(image_files[0])
                    mid_z = nuclei_mask.shape[0] // 2
                    nuclei_mask = nuclei_mask[mid_z]

                    # Enhance contrast
                    if contrast_enhance:
                        nuclei_mask = exposure.equalize_adapthist(
                            nuclei_mask, clip_limit=clip_limit
                        )

                    # Display image with custom LUT or colormap
                    if lut is not None:
                        from matplotlib.colors import ListedColormap

                        cmap = ListedColormap(lut)
                        ax.imshow(nuclei_mask, cmap=cmap)
                    else:
                        ax.imshow(nuclei_mask, cmap=image_color_map)

                    # Add bezel (border) around the image
                    for spine in ax.spines.values():
                        spine.set_edgecolor("black")
                        spine.set_linewidth(2)
                        spine.set_visible(True)
                else:
                    ax.set_facecolor("lightgray")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("gray")
                        spine.set_linewidth(1)
                        spine.set_visible(True)
            else:
                # Empty well - show gray background with lighter bezel
                ax.set_facecolor("lightgray")
                for spine in ax.spines.values():
                    spine.set_edgecolor("gray")
                    spine.set_linewidth(1)
                    spine.set_visible(True)

            # Add labels only on edges
            if i == 0:  # Top row
                ax.set_title(f"{col}", fontsize=fontsize, pad=2)
            if j == 0:  # Left column
                ax.set_ylabel(f"{row}", fontsize=fontsize, rotation=0, labelpad=10)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # Adjust layout to minimize white space
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    return fig


# In[4]:


# cyan LUT (Green + Blue)
cyan_lut = np.zeros((256, 3))
cyan_lut[:, 1] = np.linspace(0, 1, 256)  # Green
cyan_lut[:, 2] = np.linspace(0, 1, 256)  # Blue
cyan_lut[0] = [0, 0, 0]

# Magenta LUT (Red + Blue)
magenta_lut = np.zeros((256, 3))
magenta_lut[:, 0] = np.linspace(0, 1, 256)  # Red
magenta_lut[:, 2] = np.linspace(0, 1, 256)  # Blue
magenta_lut[0] = [0, 0, 0]

# Yellow LUT (Red + Green)
yellow_lut = np.zeros((256, 3))
yellow_lut[:, 0] = np.linspace(0, 1, 256)  # Red
yellow_lut[:, 1] = np.linspace(0, 1, 256)  # Green
yellow_lut[0] = [0, 0, 0]

# Green LUT
green_lut = np.zeros((256, 3))
green_lut[:, 1] = np.linspace(0, 1, 256)  # Green
green_lut[0] = [0, 0, 0]

# red LUT
red_lut = np.zeros((256, 3))
red_lut[:, 0] = np.linspace(0, 1, 256)  # Red
red_lut[0] = [0, 0, 0]


# In[5]:


patients = ["NF0014_T1"]
for patient in tqdm.tqdm(
    patients, desc="Generating platemaps for patients", unit="patient"
):
    input_dir = pathlib.Path(
        f"{image_base_dir}/data/{patient}/{input_subparent_name}/"
    ).resolve(strict=True)
    mask_path = pathlib.Path(
        f"{image_base_dir}/data/{patient}/{mask_subparent_name}/"
    ).resolve(strict=True)
    # get the well_fov paths
    well_fovs = mask_path.glob("*")
    mask_available_wells = {}
    for well_fov_path in well_fovs:
        if not well_fov_path.is_dir():
            continue
        well_fov_name = well_fov_path.stem.split("-")[0]
        if well_fov_name not in mask_available_wells:
            mask_available_wells[well_fov_name] = well_fov_path

    well_fovs = input_dir.glob("*")
    image_available_wells = {}
    for well_fov_path in well_fovs:
        if not well_fov_path.is_dir():
            continue
        well_fov_name = well_fov_path.stem.split("-")[0]
        if well_fov_name not in image_available_wells:
            image_available_wells[well_fov_name] = well_fov_path

    # plot and save the plate view
    channels_to_show = ["405", "488", "555", "640"]
    masks_to_show = ["organoid", "nuclei", "cell"]

    # for channel in tqdm.tqdm(
    #     channels_to_show, desc="Generating channel platemaps", leave=False
    # ):
    #     if channel == "405":
    #         lut = cyan_lut
    #         channel_title = "Hoechst - 405nm"

    #     elif channel == "488":
    #         lut = green_lut
    #         channel_title = "Endoplasmic Reticulum - 488nm"

    #     elif channel == "555":
    #         lut = magenta_lut
    #         channel_title = "AGP - 555nm"

    #     elif channel == "640":
    #         lut = red_lut
    #         channel_title = "Mitochondria - 640nm"

    #     else:
    #         lut = None
    #     fig = plot_plate_overview(
    #         plate=patient,
    #         image_sub_string_to_search=channel,
    #         title_for_substring=channel_title,
    #         available_wells=image_available_wells,
    #         layout="96",
    #         skip_outer_wells=True,
    #         lut=lut,
    #         contrast_enhance=True,  # Enable contrast enhancement
    #         clip_limit=0.03,  # Adjust this value (0-1)
    #     )
    # # Save using matplotlib
    # output_path = figures_path / f"{patient}_platemap_{channel}.png"
    # fig.savefig(
    #     output_path,
    #     dpi=600,
    #     bbox_inches="tight",
    #     facecolor="white",
    #     edgecolor="none",
    # )
    # plt.close(fig)
    for mask in tqdm.tqdm(masks_to_show, desc="Generating mask platemaps", leave=False):
        if mask == "organoid":
            mask_title = "Organoid Mask"
        elif mask == "nuclei":
            mask_title = "Nuclei Mask"
        elif mask == "cell":
            mask_title = "Cell Mask"
        else:
            mask_title = mask
        fig = plot_plate_overview(
            plate=patient,
            image_sub_string_to_search=mask,
            title_for_substring=mask_title,
            available_wells=mask_available_wells,
            layout="96",
            skip_outer_wells=True,
            image_color_map="nipy_spectral",
        )
        # Save using matplotlib
        output_path = figures_path / f"{patient}_platemap_{mask}.png"
        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close(fig)
