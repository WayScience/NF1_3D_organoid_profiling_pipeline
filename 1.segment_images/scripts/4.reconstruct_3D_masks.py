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

# In[1]:


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
import nviz
from nviz.image_meta import extract_z_slice_number_from_filename, generate_ome_xml
from segmentation_decoupling import euclidian_2D_distance

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

mask_dir = pathlib.Path(f"../processed_data/{str(input_dir.stem)}").resolve(strict=True)
if compartment == "nuclei":
    input_image_dir = pathlib.Path(mask_dir / "nuclei_masks.tiff").resolve(strict=True)
    x_y_vector_radius_max_constaint = 5  # pixels
elif compartment == "cell":
    input_image_dir = pathlib.Path(mask_dir / "cell_masks.tiff").resolve(strict=True)
    x_y_vector_radius_max_constaint = 15  # pixels
elif compartment == "organoid":
    input_image_dir = pathlib.Path(mask_dir / "organoid_masks.tiff").resolve(
        strict=True
    )
    x_y_vector_radius_max_constaint = 50  # pixels
else:
    raise ValueError(
        "Invalid compartment, please choose either 'nuclei', 'cell', or 'organoid'"
    )

output_image_dir = input_image_dir


# ## Extract masks and masks centers (XY coordinates) from the input image

# In[3]:


image = tifffile.imread(input_image_dir)
if np.unique(image).max() > 255:
    image = image.astype(np.uint16)
else:
    image = image.astype(np.uint8)


# In[4]:


cordinates = {
    "original_label": [],
    "slice": [],
    "centroid-0": [],
    "centroid-1": [],
}

for slice in range(image.shape[0]):
    props = skimage.measure.regionprops_table(
        image[slice, :, :], properties=["label", "centroid"]
    )

    label, centroid1, centroid2 = (
        props["label"],
        props["centroid-0"],
        props["centroid-1"],
    )
    if len(label) > 0:
        for i in range(len(label)):
            cordinates["original_label"].append(label[i])
            cordinates["slice"].append(slice)
            cordinates["centroid-0"].append(centroid1[i])
            cordinates["centroid-1"].append(centroid2[i])


coordinates_df = pd.DataFrame(cordinates)
coordinates_df["unique_id"] = coordinates_df.index
coordinates_df.head()


# In[5]:


cordinates = {
    "original_label": [],
    "slice": [],
    "centroid-0": [],
    "centroid-1": [],
}

for slice in range(image.shape[0]):
    props = skimage.measure.regionprops_table(
        image[slice, :, :], properties=["label", "centroid"]
    )

    label, centroid1, centroid2 = (
        props["label"],
        props["centroid-0"],
        props["centroid-1"],
    )
    if len(label) > 0:
        for i in range(len(label)):
            cordinates["original_label"].append(label[i])
            cordinates["slice"].append(slice)
            cordinates["centroid-0"].append(centroid1[i])
            cordinates["centroid-1"].append(centroid2[i])


coordinates_df = pd.DataFrame(cordinates)
coordinates_df["unique_id"] = coordinates_df.index


# ## Plot the coordinates of the masks in the XY plane

# In[ ]:


if in_notebook:
    # plot the data
    fig, ax = plt.subplots()
    plt.scatter(
        coordinates_df["centroid-0"],
        coordinates_df["centroid-1"],
        c=coordinates_df["unique_id"],
    )
    plt.xlabel("centroid-0")
    plt.ylabel("centroid-1")
    plt.show()


# In[7]:


# generate distance pairs for each slice
x_y_vector_radius_max_constaint = 50
distance_pairs = []
for i in range(coordinates_df.shape[0]):
    for j in range(coordinates_df.shape[0]):
        if i != j:
            coordinate_pair1 = coordinates_df.loc[
                i, ["centroid-0", "centroid-1"]
            ].values
            coordinate_pair2 = coordinates_df.loc[
                j, ["centroid-0", "centroid-1"]
            ].values
            distance = euclidian_2D_distance(coordinate_pair1, coordinate_pair2)
            if distance < x_y_vector_radius_max_constaint:
                distance_pairs.append(
                    {
                        "slice1": coordinates_df.loc[i, "slice"],
                        "slice2": coordinates_df.loc[j, "slice"],
                        "index1": i,
                        "index2": j,
                        "distance": distance,
                        "coordinates1": (coordinate_pair1[0], coordinate_pair1[1]),
                        "coordinates2": (coordinate_pair2[0], coordinate_pair2[1]),
                        "pass": True,
                        "original_label1": coordinates_df.loc[i, "original_label"],
                        "original_label2": coordinates_df.loc[j, "original_label"],
                    }
                )
            else:
                distance_pairs.append(
                    {
                        "slice1": coordinates_df.loc[i, "slice"],
                        "slice2": coordinates_df.loc[j, "slice"],
                        "index1": i,
                        "index2": j,
                        "distance": distance,
                        "coordinates1": (coordinate_pair1[0], coordinate_pair1[1]),
                        "coordinates2": (coordinate_pair2[0], coordinate_pair2[1]),
                        "pass": False,
                        "original_label1": coordinates_df.loc[i, "original_label"],
                        "original_label2": coordinates_df.loc[j, "original_label"],
                    }
                )
df = pd.DataFrame(distance_pairs)
df["indexes"] = df["index1"].astype(str) + "-" + df["index2"].astype(str)
df = df[df["pass"] == True]
df["index_comparison"] = df["index1"].astype(str) + "," + df["index2"].astype(str)
df.head()


# In[8]:


index_sets = df["index_comparison"]
list_of_sets = [set(map(int, x.split(","))) for x in index_sets]
for i in list_of_sets:
    for j in list_of_sets:
        if i != j and len(i.intersection(j)) > 0:
            i.update(j)
            list_of_sets.remove(j)

merged_sets_dict = {}
for i in range(len(list_of_sets)):
    merged_sets_dict[i] = list_of_sets[i]
for row in df.iterrows():
    for num_set in merged_sets_dict:
        if int(row[1]["index1"]) in merged_sets_dict[num_set]:
            df.at[row[0], "label"] = num_set + 1
df.head()


# In[9]:


# get the original label for each index and slice and return the new label
return_dict = {
    "slice": [],
    "index": [],
    "original_label": [],
    "new_label": [],
}
for row in df.iterrows():
    return_dict["slice"].append(row[1]["slice1"])
    return_dict["index"].append(row[1]["index1"])
    return_dict["original_label"].append(row[1]["original_label1"])
    return_dict["new_label"].append(row[1]["label"])

    return_dict["slice"].append(row[1]["slice2"])
    return_dict["index"].append(row[1]["index2"])
    return_dict["original_label"].append(row[1]["original_label2"])
    return_dict["new_label"].append(row[1]["label"])
return_df = pd.DataFrame(return_dict)
print(return_df.shape)
# drop duplicate rows
return_df = return_df.drop_duplicates()
print(return_df.shape)
return_df.head()


# In[10]:


new_mask_image = np.zeros_like(image)
# mask label reassignment
for slice in range(image.shape[0]):
    mask = image[slice, :, :]
    tmp_df = return_df[return_df["slice"] == slice]
    for i in range(tmp_df.shape[0]):
        mask[mask == tmp_df.iloc[i]["original_label"]] = tmp_df.iloc[i]["new_label"]

    new_mask_image[slice, :, :] = mask
# save the new image
tifffile.imwrite(output_image_dir, new_mask_image)


# In[ ]:


if in_notebook:
    for slice in range(new_mask_image.shape[0]):
        plt.subplot(1, 2, 1)
        plt.imshow(image[slice, :, :])
        plt.title(f"Original {slice}")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(new_mask_image[slice, :, :])
        plt.title(f"New {slice}")
        plt.axis("off")
        plt.show()
