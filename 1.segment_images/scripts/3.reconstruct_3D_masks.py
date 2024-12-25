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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skimage
import skimage.io as io
import tifffile
from cellpose import core, models, utils
from rich.pretty import pprint

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
    input_dir = pathlib.Path("../processed_data/raw_z_input/").resolve(strict=True)
    x_y_vector_radius_max_constaint = 10
    compartment = "nuclei"

mask_path = pathlib.Path(f"../processed_data/{input_dir.stem}").resolve()
mask_path.mkdir(exist_ok=True, parents=True)

if compartment == "nuclei":
    input_image_dir = pathlib.Path(mask_path / "nuclei_masks.tiff").resolve(strict=True)
    output_image_dir = pathlib.Path(mask_path / "nuclei_masks.tiff").resolve()
elif compartment == "cell":
    input_image_dir = pathlib.Path(mask_path / "cell_masks.tiff").resolve(strict=True)
    output_image_dir = pathlib.Path(mask_path / "cell_masks.tiff").resolve()
else:
    raise ValueError("Invalid compartment, please choose either 'nuclei' or 'cell'")


# ## Extract masks and masks centers (XY coordinates) from the input image

# In[3]:


image = tifffile.imread(input_image_dir)

cordinates = {
    "label": [],
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
    if len(label) > 1:
        for i in range(len(label)):
            cordinates["label"].append(label[i])
            cordinates["slice"].append(slice)
            cordinates["centroid-0"].append(centroid1[i])
            cordinates["centroid-1"].append(centroid2[i])

coordinates_df = pd.DataFrame(cordinates)
coordinates_df["unique_id"] = coordinates_df.index
coordinates_df


# ## Plot the coordinates of the masks in the XY plane

# In[4]:


# plot the data
fig, ax = plt.subplots()
plt.scatter(
    coordinates_df["centroid-0"],
    coordinates_df["centroid-1"],
    c=coordinates_df["unique_id"],
)
plt.xlabel("centroid-0")
plt.ylabel("centroid-1")
if in_notebook:
    plt.show()


# ## Create a graph where each node is a 2D object and each edge is a potential relation between two objects across z or an absolute relation between two objects in the same z.

# In[5]:


# construct a graph of nodes and edges from the df
# where the nodes are the centroids of the cells
# and the edges are the distance between the centroids

# create a graph
G = nx.Graph()

# add nodes
for index, row in coordinates_df.iterrows():
    G.add_node(row["unique_id"], pos=(row["unique_id"], row["slice"]))

# connect the nodes across slices
for slice in coordinates_df["slice"].unique():
    nodes = coordinates_df[coordinates_df["slice"] == slice]
    nodes_next_slice = coordinates_df[coordinates_df["slice"] == slice + 1]
    # get the edge between all nodes in the same slice
    for index, row in nodes.iterrows():
        for index2, row2 in nodes_next_slice.iterrows():
            # check if the edge already exists
            if G.has_edge(row["unique_id"], row2["unique_id"]):
                continue
            elif G.has_edge(row2["unique_id"], row["unique_id"]):
                continue
            else:
                distance = np.sqrt(
                    (row["centroid-0"] - row2["centroid-0"]) ** 2
                    + (row["centroid-1"] - row2["centroid-1"]) ** 2
                )
                if distance < x_y_vector_radius_max_constaint:
                    G.add_edge(row["unique_id"], row2["unique_id"], weight=distance)


# In[6]:


# Visualization of the graph for 3D data
# draw the graph
pos = nx.get_node_attributes(G, "pos")

nx.draw(G, pos, with_labels=True, connectionstyle="arc3,rad=0.2", arrows=True)
# make edges curved
edge_pos = nx.spring_layout(G)
if in_notebook:
    plt.show()


# ## Sovle the graph for the shortest path between the start and end nodes.

# In[7]:


# Assuming G is your graph and df is your original DataFrame

# Create a list to store the results
results = []

# Loop through the nodes and calculate longest paths with the shortest distances in a unidirectional way
for node1 in G.nodes:
    for node2 in G.nodes:
        if node1 != node2:
            # get the slices of the nodes
            slice1 = coordinates_df[coordinates_df["unique_id"] == node1][
                "slice"
            ].values[0]
            slice2 = coordinates_df[coordinates_df["unique_id"] == node2][
                "slice"
            ].values[0]
            if slice1 > slice2:
                continue
            try:
                path = nx.shortest_path(G, node1, node2)
                distance = nx.shortest_path_length(G, node1, node2, weight="weight")
                results.append([node1, node2, path, distance])
            except nx.NetworkXNoPath:
                results.append([node1, node2, None, None])

# Convert the results to a DataFrame
results_df = pd.DataFrame(results, columns=["node1", "node2", "path", "distance"])
results_df.head()


# In[8]:


# drop rows with no path
results_df = results_df.dropna()
print(results_df.shape)
results_df.head()


# ## Clean up the graph solve output to get the final path.

# In[9]:


# drop a row if the path is contained in another path
# this is because we are only interested in the longest path with the shortest distance
# and not the subpaths
# ex. remove path pf [0, 1] if there is a path of [0, 1, 2]
rows_to_drop = []
# iterate over each row
for index, row in results_df.iterrows():
    # iterate over each row
    for index2, row2 in results_df.iterrows():
        # check if the path is a subset of another path
        if len(row["path"]) < len(row2["path"]):
            if set(row["path"]).issubset(row2["path"]):
                rows_to_drop.append(index)
        elif len(row["path"]) > len(row2["path"]):
            if set(row2["path"]).issubset(row["path"]):
                rows_to_drop.append(index2)
        else:
            continue
results_df.drop(index=rows_to_drop, inplace=True)
print(results_df.shape)
results_df.head()


# In[10]:


# add an object number to each row
results_df.reset_index(drop=True, inplace=True)
results_df["object_number"] = results_df.index
results_df.drop(columns=["node1", "node2"], inplace=True)
# un group the path column
results_df = results_df.explode("path")
results_df.rename(columns={"path": "node"}, inplace=True)


# In[11]:


# # get the x and y coordinates of each node from the original df
results_df["coorinate-0"] = results_df["node"].apply(
    lambda x: coordinates_df[coordinates_df["unique_id"] == x]["centroid-0"].values[0]
)
results_df["coorinate-1"] = results_df["node"].apply(
    lambda x: coordinates_df[coordinates_df["unique_id"] == x]["centroid-1"].values[0]
)
results_df["slice"] = results_df["node"].apply(
    lambda x: coordinates_df[coordinates_df["unique_id"] == x]["slice"].values[0]
)
results_df["label"] = results_df["node"].apply(
    lambda x: coordinates_df[coordinates_df["unique_id"] == x]["label"].values[0]
)
print(results_df.shape)
results_df


# ## Plot paths output

# In[12]:


if in_notebook:
    # plot the data in 3D with tracks
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        results_df["coorinate-0"],
        results_df["coorinate-1"],
        results_df["slice"],
        c=results_df["object_number"],
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()


# ## Generate the new image via mask number reassignment

# In[13]:


# go back through the image and color each mask with the object number based on x and y coordinates for each slice
# create a new image with the same shape as the original image
new_image = np.zeros_like(image)
# iterate over each slice
for slice in range(image.shape[0]):
    # relate the object number to each mask
    for index, row in results_df[results_df["slice"] == slice].iterrows():
        # replace the mask label with the object number
        new_image[slice][image[slice] == row["label"]] = row["object_number"]


# In[14]:


# save the new image
tifffile.imwrite(output_image_dir, new_image)


# ## Visualize the new image per z-slice

# In[15]:


# plot the new image masks
if in_notebook:
    for slice in range(new_image.shape[0]):
        plt.subplot(1, 2, 1)
        plt.imshow(image[slice])
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(new_image[slice])
        plt.title("3D related")
        plt.show()

