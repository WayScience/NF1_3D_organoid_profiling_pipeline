#!/usr/bin/env python
# coding: utf-8

# # An attempt at an elegant solution to the 2D -> 3D object relation problem.
# I will use graph theory to solve this problem.
# The idea is to create a graph where each node represents a 2D object and each edge represents a potential relation between two objects across z or an absolute relation between two objects in the same z.
# These edges will be weighted based on their z distance.
# The problem then becomes a shortest path problem where we need to find the shortest path between the start and end nodes.
# An issue that will arise is figuring out how many nodes might exist in a given path. This will vary and will be a challenge to solve.
# Some nodes will not start until a certain z level and some nodes will end at a certain z level.
#
# ### To do this we must make the following assumptions:
# 1. Nodes in the same z level are connected to each other, but will not be used in the shortest path calculation - these are separate objects.
# 2. The variablility in the distance between the X-Y coordinates across z-slices for the same 3D object is minimal.
#     - This will be a learned or assumed value - probably initialized as apriori knowledge.
# 3. The distance between the X-Y coordinates across z-slices for different 3D objects is significant.
# 4. Given the assumption of 1 and 2, we do not need to calculate the shortes path between all nodes, only the nodes that are within a certain distance of each other in the X-Y plane across z-slices.
# 5. The total number of objects in a 2D slice cannot be greater than the total number of objects in the 3D object.
# 6. The largest number of objects across all 2D slices will not necessarily be the same as the number of objects in the 3D object.
# 7. The node path has to be continuous and cannot skip nodes.

# ## Imports

# In[ ]:


import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import tifffile
from cellpose import core, models, utils
from rich.pretty import pprint

sys.path.append("../../utils")
from nviz.image import image_set_to_arrays
from nviz.image_meta import extract_z_slice_number_from_filename, generate_ome_xml
from nviz.view import view_ometiff_with_napari
from segmentation_decoupling import euclidian_2D_distance

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
    parser.add_argument(
        "--radius_constraint",
        type=int,
        default=10,
        help="The maximum radius of the x-y vector",
    )
    parser.add_argument(
        "--compartment",
        type=str,
        default="none",
        help="The compartment to segment",
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
    x_y_vector_radius_max_constaint = args.radius_constraint
    compartment = args.compartment
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../processed_data/C4-2/").resolve(strict=True)

    compartment = "organoid"

if compartment == "nuclei":
    input_image_dir = pathlib.Path(input_dir / "nuclei_masks.tiff").resolve(strict=True)
    x_y_vector_radius_max_constaint = 5  # pixels
elif compartment == "cell":
    input_image_dir = pathlib.Path(input_dir / "cell_masks.tiff").resolve(strict=True)
    x_y_vector_radius_max_constaint = 15  # pixels
elif compartment == "organoid":
    input_image_dir = pathlib.Path(input_dir / "organoid_masks.tiff").resolve(
        strict=True
    )
    x_y_vector_radius_max_constaint = 50  # pixels
else:
    raise ValueError(
        "Invalid compartment, please choose either 'nuclei', 'cell', or 'organoid'"
    )

output_image_dir = input_image_dir


# In[ ]:


image_dir = "../../data/NF0014/zstack_images/C4-2/"
label_dir = input_dir
output_path = "output.zarr"
channel_map = {
    "405": "Nuclei",
    "488": "Endoplasmic Reticulum",
    "555": "Actin, Golgi, and plasma membrane (AGP)",
    "640": "Mitochondria",
    "TRANS": "Brightfield",
}
scaling_values = [1, 0.1, 0.1]


# In[ ]:


frame_zstacks = image_set_to_arrays(
    image_dir,
    label_dir,
    channel_map=channel_map,
)

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
with tiff.TiffWriter(output_path, bigtiff=True) as tif:
    tif.write(combined_data, description=ome_xml, photometric="minisblack")


# In[ ]:


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


# In[ ]:


view_ometiff_with_napari(
    ometiff_path=output_path,
    scaling_values=scaling_values,
    headless=True,
)
