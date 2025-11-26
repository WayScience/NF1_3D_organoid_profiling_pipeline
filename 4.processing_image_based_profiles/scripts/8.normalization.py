#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile normalization.
# All profiles are normalized to the DMSO control treated profiles.

# In[1]:


import os
import pathlib

import numpy as np
import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import normalize

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[ ]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


# pathing
sc_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.annotated_profiles/sc_anno.parquet"
).resolve(strict=True)
organoid_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.annotated_profiles/organoid_anno.parquet"
).resolve(strict=True)


# output path
sc_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.normalized_profiles/sc_norm.parquet"
).resolve()
organoid_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.normalized_profiles/organoid_norm.parquet"
).resolve()

organoid_normalized_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_annotated_profiles = pd.read_parquet(sc_annotated_path)
organoid_annotated_profiles = pd.read_parquet(organoid_annotated_path)


# In[5]:


sc_annotated_profiles.head()


# ### Normalize the single-cell profiles

# In[6]:


sc_metadata_columns = [x for x in sc_annotated_profiles.columns if "Metadata" in x]

sc_metadata_columns += [
    "Area.Size.Shape_Cell_CENTER.X",
    "Area.Size.Shape_Cell_CENTER.Y",
    "Area.Size.Shape_Cell_CENTER.Z",
]
sc_features_columns = [
    col for col in sc_annotated_profiles.columns if col not in sc_metadata_columns
]


# In[7]:


# find inf values and replace with NaN
sc_annotated_profiles[sc_features_columns] = sc_annotated_profiles[
    sc_features_columns
].replace([float("inf"), -float("inf")], np.nan)


# In[8]:


# normalize the data
sc_normalized_profiles = normalize(
    sc_annotated_profiles,
    features=sc_features_columns,
    meta_features=sc_metadata_columns,
    method="standardize",
    samples="Metadata_treatment == 'DMSO'",
)
sc_normalized_profiles.to_parquet(sc_normalized_output_path, index=False)
sc_normalized_profiles.head()


# ### Normalize the organoid profiles

# In[9]:


organoid_annotated_profiles.head()


# In[10]:


organoid_metadata_columns = [
    x for x in organoid_annotated_profiles.columns if "Metadata" in x
]
organoid_metadata_columns += [
    "Area.Size.Shape_Organoid_CENTER.X",
    "Area.Size.Shape_Organoid_CENTER.Y",
    "Area.Size.Shape_Organoid_CENTER.Z",
]
organoid_features_columns = [
    col
    for col in organoid_annotated_profiles.columns
    if col not in organoid_metadata_columns
]
# normalize the data
organoid_normalized_profiles = normalize(
    organoid_annotated_profiles,
    features=organoid_features_columns,
    meta_features=organoid_metadata_columns,
    method="standardize",
    samples="Metadata_treatment == 'DMSO'",
)
organoid_normalized_profiles.to_parquet(organoid_normalized_output_path, index=False)
organoid_normalized_profiles.head()
