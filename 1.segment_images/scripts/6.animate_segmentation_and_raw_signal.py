#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import argparse
import pathlib

import imageio

# import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from napari_animation import Animation
from napari_animation.easing import Easing
from nviz.image import image_set_to_arrays
from nviz.image_meta import generate_ome_xml
from nviz.view import view_ometiff_with_napari
from PIL import Image

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
        "--image_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
        required=True,
    )

    args = parser.parse_args()
    image_dir = pathlib.Path(args.image_dir).resolve(strict=True)
    label_dir = pathlib.Path(f"../processed_data/{image_dir.name}").resolve(strict=True)
    mp4_file_dir = pathlib.Path("../animations/mp4/").resolve()
    gif_file_dir = pathlib.Path("../animations/gif/").resolve()
else:
    print("Running in a notebook")
    image_dir = pathlib.Path("../../data/NF0014/zstack_images/C4-2/").resolve(
        strict=True
    )
    label_dir = pathlib.Path(f"../processed_data/{image_dir.name}").resolve(strict=True)
    mp4_file_dir = pathlib.Path("../animations/test/mp4/").resolve()
    gif_file_dir = pathlib.Path("../animations/test/gif/").resolve()


mp4_file_dir.mkdir(parents=True, exist_ok=True)
gif_file_dir.mkdir(parents=True, exist_ok=True)
tmp_output_path = "output.zarr"


# In[3]:


def mp4_to_gif(input_mp4, output_gif, fps=10):
    clip = VideoFileClip(input_mp4)
    clip = clip.set_fps(fps)  # Reduce FPS to control file size
    clip.write_gif(output_gif, loop=0)  # loop=0 makes it loop forever
    print(f"Converted {input_mp4} to {output_gif}")


# In[4]:


def animate_view(
    viewer, output_path_name: str, steps: int = 30, easing: str = "linear", dim: int = 3
):
    animation = Animation(viewer)
    if easing == "linear":
        ease_style = Easing.LINEAR
    else:
        raise ValueError(f"Invalid easing style: {easing}")

    viewer.dims.ndisplay = dim
    # rotate around the y-axis
    viewer.camera.angles = (0.0, 0.0, 90.0)  # (z, y, x) axis of rotation
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 180.0, 90.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 360.0, 90.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 0.0, 270.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    viewer.camera.angles = (0.0, 0.0, 90.0)
    animation.capture_keyframe(steps=steps, ease=ease_style)

    animation.animate(output_path_name, canvas_only=True)


# In[5]:


channel_map = {
    "405": "Nuclei",
    "488": "Endoplasmic Reticulum",
    "555": "Actin, Golgi, and plasma membrane (AGP)",
    "640": "Mitochondria",
    "TRANS": "Brightfield",
}
scaling_values = [1, 0.1, 0.1]


# In[6]:


frame_zstacks = image_set_to_arrays(
    image_dir,
    label_dir,
    channel_map=channel_map,
)

print(frame_zstacks["images"].keys())
print(frame_zstacks["labels"].keys())


# In[7]:


images_data = []
labels_data = []
channel_names = []
label_names = []


for channel, stack in frame_zstacks["images"].items():
    dim = len(stack.shape)
    images_data.append(stack)
    channel_names.append(channel)

# Collect label data
if label_dir:
    for compartment_name, stack in frame_zstacks["labels"].items():
        if len(stack.shape) != dim:
            if len(stack.shape) == 3:
                stack = np.expand_dims(stack, axis=0)
        labels_data.append(stack)
        label_names.append(f"{compartment_name} (labels)")
# Stack the images and labels along a new axis for channels
images_data = np.stack(images_data, axis=0)


# In[8]:


if labels_data:
    labels_data = np.stack(labels_data, axis=0)
    combined_data = np.concatenate((images_data, labels_data), axis=0)
    combined_channel_names = channel_names + label_names
else:
    combined_data = images_data
    combined_channel_names = channel_names
# Generate OME-XML metadata
ome_metadata = {
    "SizeC": combined_data.shape[0],
    "SizeZ": combined_data.shape[1],
    "SizeY": combined_data.shape[2],
    "SizeX": combined_data.shape[3],
    "PhysicalSizeX": scaling_values[2],
    "PhysicalSizeY": scaling_values[1],
    "PhysicalSizeZ": scaling_values[0],
    # note: we use 7-bit ascii compatible characters below
    # due to tifffile limitations
    "PhysicalSizeXUnit": "um",
    "PhysicalSizeYUnit": "um",
    "PhysicalSizeZUnit": "um",
    "Channel": [{"Name": name} for name in combined_channel_names],
}
ome_xml = generate_ome_xml(ome_metadata)
import tifffile as tiff

# Write the combined data to a single OME-TIFF
with tiff.TiffWriter(tmp_output_path, bigtiff=True) as tif:
    tif.write(combined_data, description=ome_xml, photometric="minisblack")


# In[9]:


# import shutil
# shutil.rmtree(output_path, ignore_errors=True)
# nviz.image.tiff_to_ometiff(
#     image_dir=image_dir,
#     label_dir=label_dir,
#     output_path=output_path,
#     channel_map=channel_map,
#     scaling_values=scaling_values,
#     ignore=[],
# )


# In[10]:


viewer = view_ometiff_with_napari(
    ometiff_path=tmp_output_path,
    scaling_values=scaling_values,
    headless=True,
)
# make the viewer full screen
viewer.window._qt_window.showMaximized()
# hide the layer controls
viewer.window._qt_viewer.dockLayerList.setVisible(False)
# hide the layer controls
viewer.window._qt_viewer.dockLayerControls.setVisible(False)

# set the viewer to a set window size
viewer.window._qt_window.resize(1000, 1000)


# In[11]:


# get the layer names in the viewer
layer_names = [layer.name for layer in viewer.layers]
# set all layers to not visible
for layer_name in layer_names:
    print(f"Setting {layer_name} to not visible")
    viewer.layers[layer_name].visible = False


# In[12]:


for layer_name in layer_names:
    viewer.layers[layer_name].visible = True
    if ".tif" in layer_name:
        save_name = layer_name.split(".tif")[0]
    else:
        save_name = layer_name
    save_path = pathlib.Path(f"{mp4_file_dir}/{save_name}_animation.mp4")
    if "640" in layer_name:
        # increase contrast for the mitochondria
        viewer.layers[layer_name].contrast_limits = (0, 20000)
    animate_view(viewer, save_path, steps=30, easing="linear")
    viewer.layers[layer_name].visible = False
print("All layers animated")


# In[13]:


# get all gifs in the directory
mp4_file_path = list(pathlib.Path(mp4_file_dir).rglob("*.mp4"))
for mp4_file in mp4_file_path:

    # change the path to the gif directory
    mp4_file = pathlib.Path(mp4_file)
    gif_file = pathlib.Path(gif_file_dir / f"{mp4_file.stem}.gif")
    mp4_file = str(mp4_file)
    gif_file = str(gif_file)
    mp4_to_gif(mp4_file, gif_file)
