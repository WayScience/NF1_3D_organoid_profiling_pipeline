#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
)
from image_analysis_3D.featurization_utils.neighbors_utils import (
    classify_cells_into_shells,
    euclidean_distance_from_centroid,
    mahalanobis_distance_from_centroid,
)
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from tqdm import tqdm

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    well_fov = "C4-2"
    image_based_profiles_subparent_name = "image_based_profiles"


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
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/sc_profiles_{well_fov}.parquet"
).resolve(strict=True)
organoid_profile_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/organoid_profiles_{well_fov}.parquet"
).resolve(strict=True)
nucleocentric_profile_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/nucleocentric_profiles_{well_fov}.parquet"
).resolve(strict=True)
# output paths
sc_profile_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/1.related_profiles/{well_fov}/sc_profiles_{well_fov}_related.parquet"
).resolve()
organoid_profile_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/1.related_profiles/{well_fov}/organoid_profiles_{well_fov}_related.parquet"
).resolve()
nucleocentric_profile_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/1.related_profiles/{well_fov}/nucleocentric_profiles_{well_fov}_related.parquet"
).resolve()
sc_profile_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[5]:


sc_profile_df = pd.read_parquet(sc_profile_path)
nucleocentric_df = pd.read_parquet(nucleocentric_profile_path)
organoid_profile_df = pd.read_parquet(organoid_profile_path)
print(f"Single-cell profile shape: {sc_profile_df.shape}")
print(f"Nucleocentric profile shape: {nucleocentric_df.shape}")
print(f"Organoid profile shape: {organoid_profile_df.shape}")


# In[6]:


# initialize the parent organoid column
sc_profile_df.insert(2, "ParentOrganoid", -1)


# In[7]:


x_y_z_sc_colnames = [
    x
    for x in sc_profile_df.columns
    if "area" in x.lower() and "center" in x.lower() and "nuclei" in x.lower()
]
x_y_z_sc_colnames


# In[8]:


organoid_bbox_colnames = [
    x
    for x in organoid_profile_df.columns
    if "area" in x.lower() and ("min" in x.lower() or "max" in x.lower())
]
organoid_bbox_colnames = sorted(organoid_bbox_colnames)


# In[9]:


sc_centroids = sc_profile_df[
    x_y_z_sc_colnames
].values  # alphabetically sorted to be in the order of x,y,z


# In[10]:


# Initialize parent_organoid to -1
sc_profile_df["ParentOrganoid"] = -1

# Extract single-cell centroids as numpy array for faster access
sc_centroids = sc_profile_df[x_y_z_sc_colnames].values  # (N_cells, 3) array
# reshape the centroids to be in z,y,x order for easier comparison with bbox
sc_centroids = sc_centroids[:, [2, 1, 0]]  # reorder to z,y,x

# Loop through organoids with progress bar
for organoid_index, organoid_row in tqdm(
    organoid_profile_df.iterrows(),
    total=len(organoid_profile_df),
    desc="Assigning cells to organoids",
):
    # Get organoid bbox
    organoid_bbox = (
        organoid_row[organoid_bbox_colnames[5]],  # z_min
        organoid_row[organoid_bbox_colnames[4]],  # y_min
        organoid_row[organoid_bbox_colnames[3]],  # x_min
        organoid_row[organoid_bbox_colnames[2]],  # z_max
        organoid_row[organoid_bbox_colnames[1]],  # y_max
        organoid_row[organoid_bbox_colnames[0]],  # x_max
    )

    z_min, y_min, x_min, z_max, y_max, x_max = organoid_bbox

    # Vectorized bbox check - much faster!
    mask = (
        (sc_centroids[:, 0] >= z_min)  # z
        & (sc_centroids[:, 0] <= z_max)  # z
        & (sc_centroids[:, 1] >= y_min)
        & (sc_centroids[:, 1] <= y_max)
        & (sc_centroids[:, 2] >= x_min)
        & (sc_centroids[:, 2] <= x_max)
    )

    # Only assign if cell doesn't already have a parent
    unassigned_mask = sc_profile_df["ParentOrganoid"] == -1
    final_mask = mask & unassigned_mask

    # Assign parent organoid to matching cells
    sc_profile_df.loc[final_mask, "ParentOrganoid"] = organoid_row["object_id"]

print(f"Assigned {(sc_profile_df['ParentOrganoid'] != -1).sum()} cells to organoids")
print(f"Unassigned cells: {(sc_profile_df['ParentOrganoid'] == -1).sum()}")


# ### Add single-cell counts for each organoid

# In[11]:


organoid_sc_counts = (
    sc_profile_df["ParentOrganoid"]
    .value_counts()
    .to_frame(name="SingleCellCount")
    .reset_index()
)
# merge the organoid profile with the single-cell counts
organoid_profile_df = pd.merge(
    organoid_profile_df,
    organoid_sc_counts,
    left_on="object_id",
    right_on="ParentOrganoid",
    how="left",
).drop(columns=["ParentOrganoid"])
sc_count = organoid_profile_df.pop("SingleCellCount")
organoid_profile_df.insert(2, "SingleCellCount", sc_count)


# Even if the file is empty we still want to add it to the final dataframe dictionary so that we can merge on the same columns later.
# This will help with file-based checking and merging.
#

# In[12]:


# replace NaN with 0 for organoids that have no assigned cells
organoid_profile_df["SingleCellCount"] = (
    organoid_profile_df["SingleCellCount"].fillna(0).astype(int)
)
organoid_profile_df.head()


# In[13]:


if organoid_profile_df.empty:
    # add a row with 0 values
    organoid_profile_df.loc[len(organoid_profile_df)] = [0] * len(
        organoid_profile_df.columns
    )
    organoid_profile_df["image_set"] = well_fov


# In[14]:


print(f"Single-cell profile shape: {sc_profile_df.shape}")


# In[15]:


if sc_profile_df.empty:
    # add a row with Na values
    sc_profile_df.loc[len(sc_profile_df)] = [None] * len(sc_profile_df.columns)
    sc_profile_df["image_set"] = well_fov


# In[16]:


# add the parent organoid to nucleocentric features
nucleocentric_df = pd.merge(
    nucleocentric_df,
    sc_profile_df[["object_id", "image_set", "ParentOrganoid"]],
    on=["object_id", "image_set"],
    how="left",
)


# ## Get single cell and organoid relationships and spatial distributions

# In[17]:


x_y_z_organoid_centroid_colnames = [
    x
    for x in organoid_profile_df.columns
    if "area" in x.lower() and "center" in x.lower()
]
x_y_z_organoid_bbox_colnames = [
    x
    for x in organoid_profile_df.columns
    if "area" in x.lower() and ("min" in x.lower() or "max" in x.lower())
]


# In[18]:


results = []

# get the organoid id and the single-cells for each

organoid_ids = organoid_profile_df["object_id"]

# organoid_id = organoid_ids[0]

for organoid_id in organoid_ids:
    organoid_centroid = (
        organoid_profile_df.loc[
            organoid_profile_df["object_id"] == organoid_id,
            x_y_z_organoid_centroid_colnames,
        ]
        .apply(pd.to_numeric, errors="coerce")
        .iloc[0]
        .to_numpy(dtype=float)
    )
    organoid_bbox = organoid_profile_df.loc[
        organoid_profile_df["object_id"] == organoid_id, x_y_z_organoid_bbox_colnames
    ].values[0]
    single_cells_in_organoid = sc_profile_df[
        sc_profile_df["ParentOrganoid"] == organoid_id
    ]
    if single_cells_in_organoid.empty:
        print(f"No single cells assigned to organoid {organoid_id}")
        continue

    single_cells_centroids = (
        single_cells_in_organoid[x_y_z_sc_colnames]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    valid_rows = ~pd.isna(single_cells_centroids).any(axis=1)
    single_cells_centroids = single_cells_centroids[valid_rows]
    single_cells_in_organoid = single_cells_in_organoid.loc[valid_rows]

    if single_cells_centroids.shape[0] == 0:
        continue

    # convert to a dict with the key being the object_id
    # rename the centroids to z,y.x
    single_cells_centroids_dict = {
        "object_id": single_cells_in_organoid["object_id"].to_numpy(),
        "z": single_cells_in_organoid[x_y_z_sc_colnames[2]].to_numpy(dtype=float),
        "y": single_cells_in_organoid[x_y_z_sc_colnames[1]].to_numpy(dtype=float),
        "x": single_cells_in_organoid[x_y_z_sc_colnames[0]].to_numpy(dtype=float),
    }

    euclidean_distance = euclidean_distance_from_centroid(
        single_cells_centroids, organoid_centroid
    )
    mahalanobis_distance = mahalanobis_distance_from_centroid(
        single_cells_centroids, organoid_centroid
    )
    shell_classification, centroid = classify_cells_into_shells(
        coords=single_cells_centroids_dict,
        n_shells=4,
        method="mahalanobis",
        min_cells_per_shell=3,
        centroid=organoid_centroid,
    )

    shell_classification_df = pd.DataFrame(shell_classification)
    shell_classification_df["ParentOrganoid"] = organoid_id
    results.append(shell_classification_df)


# In[23]:


if results:
    df = pd.concat([pd.DataFrame(r) for r in results], ignore_index=True)

else:
    df = pd.DataFrame(columns=["object_id", "ParentOrganoid"])

# rename the columns

df.rename(
    columns={
        col: format_morphology_feature_name(
            compartment="Nuclei",
            feature_type="Neighbors",
            channel="NoChannel",
            measurement=col,
        )
        for col in df.columns
        if col not in ["object_id", "ParentOrganoid"]
    },
    inplace=True,
)


# In[24]:


# concat the shell classification with the single cell profile df to get the full single cell profile with the shell classification and the parent organoid id
sc_profile_with_shells_df = pd.merge(
    sc_profile_df,
    df,
    left_on=["object_id", "ParentOrganoid"],
    right_on=["object_id", "ParentOrganoid"],
    how="left",
)


# ### Save the profiles

# In[25]:


organoid_profile_df.to_parquet(organoid_profile_output_path, index=False)
organoid_profile_df.head()


# In[26]:


sc_profile_with_shells_df.to_parquet(sc_profile_output_path, index=False)
sc_profile_with_shells_df.head()


# In[27]:


nucleocentric_df.to_parquet(nucleocentric_profile_output_path, index=False)
nucleocentric_df.head()
