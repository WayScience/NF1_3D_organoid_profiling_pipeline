#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib

import pandas as pd

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        required=True,
        help="Well and field of view to process, e.g. 'A01_1'",
    )
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
else:
    well_fov = "C4-2"
    patient = "NF0014"


# In[3]:


def centroid_within_bbox_detection(
    centroid: tuple,
    bbox: tuple,
) -> bool:
    """
    Check if the centroid is within the bbox

    Parameters
    ----------
    centroid : tuple
        Centroid of the object in the order of (z, y, x)
        Order of the centroid is important
    bbox : tuple
        Where the bbox is in the order of (z_min, y_min, x_min, z_max, y_max, x_max)
        Order of the bbox is important

    Returns
    -------
    bool
        True if the centroid is within the bbox, False otherwise
    """
    z_min, y_min, x_min, z_max, y_max, x_max = bbox
    z, y, x = centroid
    # check if the centroid is within the bbox
    if (
        z >= z_min
        and z <= z_max
        and y >= y_min
        and y <= y_max
        and x >= x_min
        and x <= x_max
    ):
        return True
    else:
        return False


# ### Pathing

# In[4]:


# input paths
sc_profile_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/{well_fov}/sc_profiles_{well_fov}.parquet"
).resolve(strict=True)
organoid_profile_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/{well_fov}/organoid_profiles_{well_fov}.parquet"
).resolve(strict=True)
# output paths
sc_profile_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/{well_fov}/sc_profiles_{well_fov}_related.parquet"
).resolve()
organoid_profile_output_path = pathlib.Path(
    f"../../data/{patient}/image_based_profiles/{well_fov}/organoid_profiles_{well_fov}_related.parquet"
).resolve()


# In[5]:


sc_profile_df = pd.read_parquet(sc_profile_path)
organoid_profile_df = pd.read_parquet(organoid_profile_path)
print(f"Single-cell profile shape: {sc_profile_df.shape}")
print(f"Organoid profile shape: {organoid_profile_df.shape}")


# In[6]:


# initialize the parent organoid column
sc_profile_df.insert(2, "parent_organoid", -1)


# In[7]:


x_y_z_sc_colnames = [
    x
    for x in sc_profile_df.columns
    if "Area" in x and "center" in x.lower() and "Nuclei" in x
]
print(
    f"The nuclei centroids in the single-cell profile are in the columns:\n{x_y_z_sc_colnames}"
)


# In[8]:


organoid_bbox_colnames = [
    x for x in organoid_profile_df.columns if "Area" in x and ("MIN" in x or "MAX" in x)
]
organoid_bbox_colnames = sorted(organoid_bbox_colnames)
print(f"The organoid bounding boxes are in the columns:\n{organoid_bbox_colnames}")


# ### Relate the single cells and organoids

# In[9]:


# loop thorugh the organoids first as there are less organoids than single-cells
for organoid_index, organoid_row in organoid_profile_df.iterrows():
    # get the organoid bbox - should be alphabetically sorted
    # define the organoid bbox in the order of:
    # (z_min, y_min, x_min, z_max, y_max, x_max)
    organoid_bbox = (
        organoid_row[organoid_bbox_colnames[5]],
        organoid_row[organoid_bbox_colnames[4]],
        organoid_row[organoid_bbox_colnames[3]],
        organoid_row[organoid_bbox_colnames[2]],
        organoid_row[organoid_bbox_colnames[1]],
        organoid_row[organoid_bbox_colnames[0]],
    )
    # loop through the single-cells and check if the centroid is within the organoid bbox
    for sc_index, sc_row in sc_profile_df.iterrows():
        # get the single-cell centroid - should be alphabetically sorted
        # define the single-cell centroid in the order of (z, y, x)
        sc_centroid = (
            sc_row[x_y_z_sc_colnames[2]],
            sc_row[x_y_z_sc_colnames[1]],
            sc_row[x_y_z_sc_colnames[0]],
        )

        if centroid_within_bbox_detection(sc_centroid, organoid_bbox):
            sc_profile_df.at[sc_index, "parent_organoid"] = organoid_row["object_id"]
        else:
            # if the centroid is not within the organoid bbox, set the parent organoid to -1
            sc_profile_df.at[sc_index, "parent_organoid"] = -1


# ### Add single-cell counts for each organoid

# In[10]:


organoid_sc_counts = (
    sc_profile_df["parent_organoid"]
    .value_counts()
    .to_frame(name="single_cell_count")
    .reset_index()
)
# filter out singleton single-cells
organoid_sc_counts = organoid_sc_counts.loc[organoid_sc_counts["parent_organoid"] > 0]
# merge the organoid profile with the single-cell counts
organoid_profile_df = pd.merge(
    organoid_profile_df,
    organoid_sc_counts,
    left_on="object_id",
    right_on="parent_organoid",
    how="left",
).drop(columns=["parent_organoid"])
sc_count = organoid_profile_df.pop("single_cell_count")
organoid_profile_df.insert(2, "single_cell_count", sc_count)


# ### Save the profiles

# In[11]:


organoid_profile_df.to_parquet(organoid_profile_output_path, index=False)
organoid_profile_df.head()


# In[12]:


sc_profile_df.to_parquet(sc_profile_output_path, index=False)
sc_profile_df.head()
