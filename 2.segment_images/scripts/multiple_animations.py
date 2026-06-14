#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import argparse
import os
import pathlib
import sys

import imageio
import napari

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.file_reading import *
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from moviepy import VideoFileClip
from napari_animation import Animation
from napari_animation.easing import Easing
from PIL import Image

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


if not in_notebook:
    args = parse_args()
    input_subparent_name = args["input_subparent_name"]
    mask_subparent_name = args["mask_subparent_name"]
    amimation_subparent_name = args["amimation_subparent_name"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        input_subparent_name=input_subparent_name,
        mask_subparent_name=mask_subparent_name,
        amimation_subparent_name=amimation_subparent_name,
    )

else:
    print("Running in a notebook")
    input_subparent_name = "deconvolved_images"
    mask_subparent_name = "segmentation_masks"
    amimation_subparent_name = "animations"

patient_list_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patient_ids = pd.read_csv(patient_list_file_path, header=None)[0].tolist()
# remove NF0014_T2 from the list of patient_ids
patient_ids.remove("NF0014_T2")
patient_ids.remove("NF0055_T1")


# In[3]:


def mp4_to_gif(input_mp4: pathlib.Path, output_gif: pathlib.Path, fps: int = 30):
    """
    Convert an mp4 file to a gif file using moviepy.
    Parameters
    ----------
    input_mp4 : pathlib.Path
        The path to the input mp4 file.
    output_gif : pathlib.Path
        The path to the output gif file.
    fps : int, optional
        The frames per second for the output gif file, by default 30.

    Returns
    -------
    None
    """
    with VideoFileClip(str(input_mp4)) as clip:
        width, height = clip.size
        side = min(width, height)

        # keep codec-friendly dimensions
        if side % 2 != 0:
            side -= 1

        x1 = int((width - side) // 2)
        y1 = int((height - side) // 2)

        square_clip = clip.cropped(
            x1=x1,
            y1=y1,
            x2=x1 + side,
            y2=y1 + side,
        )

        # Write only GIF (do not overwrite source mp4)
        square_clip.write_gif(
            str(output_gif),
            fps=fps,
            loop=0,
        )


# In[4]:


def animate_view(
    viewer: napari.Viewer,
    output_path_name: str,
    steps: int = 30,
    easing: str = "linear",
    dim: int = 3,
):
    """
    Animate a napari viewer by rotating around the y-axis and then back to the original position.
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer to animate.
    output_path_name : str
        The path to save the output mp4 file.
    steps : int, optional
        The number of steps for each keyframe, by default 30.
    easing : str, optional
        The easing style for the animation, by default "linear".
    dim : int, optional
        The number of dimensions to display, by default 3.
    Returns
    -------
    None
    """
    animation = Animation(viewer)
    if easing == "linear":
        ease_style = Easing.LINEAR
    else:
        raise ValueError(f"Invalid easing style: {easing}")

    viewer.dims.ndisplay = dim
    # rotate around the y-axis
    viewer.camera.angles = (0.0, 0.0, 0.0)  # (z, y, x) axis of rotation
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 180.0, 0.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 360.0, 0.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 360.0, 180.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 360.0, 360.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    animation.animate(output_path_name, canvas_only=True)


# In[5]:


output_path = "output.zarr"
channel_map = {
    "405": "Nuclei",
    "488": "Endoplasmic Reticulum",
    "555": "Actin, Golgi, and plasma membrane (AGP)",
    "640": "Mitochondria",
    "TRANS": "Brightfield",
}
scaling_values = [1, 0.1, 0.1]


# In[6]:


patient_ids = [
    "NF0014_T1",
]
well_fovs_to_animate = [
    "C4-2",
]

# np.random.seed(0)
for patient in tqdm.tqdm(patient_ids, desc="Processing patients"):
    # get a list of all well_fovs in the zstack_images dir
    # images_dir = pathlib.Path(
    #     f"{image_base_dir}/data/{patient}/{input_subparent_name}"
    # ).resolve(strict=True)
    # list_of_well_fovs = images_dir.glob("*")
    # list_of_well_fovs = [x for x in list_of_well_fovs if x.is_dir()]
    # list_of_well_fovs = [x.name for x in list_of_well_fovs]
    # select 5 random well_fovs to animate
    # well_fovs_to_animate = np.random.choice(list_of_well_fovs, size=5, replace=False)
    for well_fov in tqdm.tqdm(well_fovs_to_animate, desc=f"Animating {patient}"):
        try:
            print(f"Animating {patient} {well_fov}")

            image_metadata = f"{patient}_{well_fov}"
            image_dir = pathlib.Path(
                f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}/"
            ).resolve(strict=True)
            label_dir = pathlib.Path(
                f"{image_base_dir}/data/{patient}/{mask_subparent_name}/{well_fov}/"
            ).resolve(strict=True)
            mp4_file_dir = pathlib.Path(
                f"{root_dir}/data/{patient}/{amimation_subparent_name}/mp4/{well_fov}/"
            ).resolve()
            gif_file_dir = pathlib.Path(
                f"{root_dir}/data/{patient}/{amimation_subparent_name}/gif/{well_fov}/"
            ).resolve()

            mp4_file_dir.mkdir(parents=True, exist_ok=True)
            gif_file_dir.mkdir(parents=True, exist_ok=True)
            # get the file counts in the mp4 and gif directories
            mp4_file_count = len(list(mp4_file_dir.glob("*")))
            gif_file_count = len(list(gif_file_dir.glob("*")))

            if mp4_file_count == 8 and gif_file_count == 8:
                print(f"Animations already exist for {patient} {well_fov}, skipping...")
                continue

            if mp4_file_count == 8:
                continue

            image_return_dict = read_in_channels(
                find_files_available(image_dir),
                channel_dict={
                    "DNA": "405",
                    "Endoplasmic_Reticulum": "488",
                    "AGP": "555",
                    "Mitochondria": "640",
                },
                channels_to_read=[
                    "DNA",
                    "Endoplasmic_Reticulum",
                    "AGP",
                    "Mitochondria",
                ],
            )

            mask_return_dict = read_in_channels(
                find_files_available(label_dir),
                channel_dict={
                    "Nuclei_mask": "nuclei",
                    "Cell_mask": "cell",
                    "Cytoplasm_mask": "cytoplasm",
                    "Organoid_mask": "organoid",
                },
                channels_to_read=[
                    "Nuclei_mask",
                    "Cell_mask",
                    "Cytoplasm_mask",
                    "Organoid_mask",
                ],
            )

            headless = False
            viewer = napari.Viewer(ndisplay=3, show=bool(not headless))

            for image_name, image_array in image_return_dict.items():
                viewer.add_image(
                    image_array,
                    name=f"{image_metadata}_{image_name}",
                    scale=scaling_values,
                )
            for mask_name, mask_array in mask_return_dict.items():
                viewer.add_labels(
                    mask_array,
                    name=f"{image_metadata}_{mask_name}",
                    scale=scaling_values,
                )

            # make the viewer full screen
            viewer.window._qt_window.showMaximized()
            # hide the layer controls
            viewer.window._qt_viewer.dockLayerList.setVisible(False)
            # hide the layer controls
            viewer.window._qt_viewer.dockLayerControls.setVisible(False)

            # set the viewer to a set window size
            viewer.window._qt_window.resize(1000, 1000)
            viewer.camera.zoom = 5.0

            # get the layer names in the viewer
            layer_names = [layer.name for layer in viewer.layers]
            # set all layers to not visible
            for layer_name in layer_names:
                viewer.layers[layer_name].visible = False
            for layer_name in layer_names:
                viewer.layers[layer_name].visible = True
                # change the brightness and contrast for the raw signal layers
                if ".tif" in layer_name:
                    save_name = layer_name.split(".tif")[0]
                else:
                    save_name = layer_name

                # map the layer name to the channel name
                if "DNA" in layer_name:
                    save_name = "DNA"
                elif "Endoplasmic" in layer_name:
                    save_name = "ER"
                elif "AGP" in layer_name:
                    save_name = "AGP"
                elif "Mitochondria" in layer_name:
                    save_name = "mitochondria"
                else:
                    save_name = layer_name

                if "Mito" in layer_name:
                    # increase contrast for the mitochondria
                    viewer.layers[layer_name].contrast_limits = (0, 20000)
                mp4_save_path = pathlib.Path(
                    f"{mp4_file_dir}/{well_fov}_{save_name}_animation.mp4"
                )
                animate_view(viewer, mp4_save_path, steps=30, easing="linear")
                viewer.layers[layer_name].visible = False

            # close the napari instance
            viewer.close()

            if gif_file_count < 8:
                # get all gifs in the directory
                mp4_file_path = list(pathlib.Path(mp4_file_dir).rglob("*.mp4"))
                for mp4_file in mp4_file_path:
                    # change the path to the gif directory
                    mp4_file = pathlib.Path(mp4_file)
                    gif_file = pathlib.Path(gif_file_dir / f"{mp4_file.stem}.gif")
                    mp4_file = str(mp4_file)
                    gif_file = str(gif_file)
                    mp4_to_gif(mp4_file, gif_file)

        except Exception as e:
            print(f"Error processing {patient} {well_fov}: {e}")
            continue
