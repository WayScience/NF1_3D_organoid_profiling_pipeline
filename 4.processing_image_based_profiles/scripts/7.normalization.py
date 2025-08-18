#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile normalization.
# All profiles are normalized to the DMSO control treated profiles.

# In[2]:


import argparse
import pathlib

import numpy as np
import pandas as pd
from pycytominer import normalize

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[3]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    patient = args.patient

else:
    patient = "NF0014"


# In[4]:


# pathing
sc_annotated_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/2.annotated_profiles/sc_anno.parquet"
).resolve(strict=True)
organoid_annotated_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/2.annotated_profiles/organoid_anno.parquet"
).resolve(strict=True)


# output path
sc_normalized_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/3.normalized_profiles/sc_norm.parquet"
).resolve()
organoid_normalized_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/3.normalized_profiles/organoid_norm.parquet"
).resolve()

organoid_normalized_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[5]:


# read in the data
sc_annotated_profiles = pd.read_parquet(sc_annotated_path)
organoid_annotated_profiles = pd.read_parquet(organoid_annotated_path)


# ### Normalize the single-cell profiles

# In[6]:


sc_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "Target",
    "Class",
    "Therapeutic Categories",
    "image_set",
    "Well",
    "parent_organoid",
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


# In[ ]:


# normalize the data
sc_normalized_profiles = normalize(
    sc_annotated_profiles,
    features=sc_features_columns,
    meta_features=sc_metadata_columns,
    method="standardize",
    samples="treatment == 'DMSO'",
)
sc_normalized_profiles.to_parquet(sc_normalized_output_path, index=False)
sc_normalized_profiles.head()


# ### Normalize the organoid profiles

# In[8]:


organoid_annotated_profiles.head()


# In[11]:


organoid_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "Target",
    "Class",
    "Therapeutic Categories",
    "image_set",
    "Well",
    "single_cell_count",
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
    samples="treatment == 'DMSO'",
)
organoid_normalized_profiles.to_parquet(organoid_normalized_output_path, index=False)
organoid_normalized_profiles.head()
