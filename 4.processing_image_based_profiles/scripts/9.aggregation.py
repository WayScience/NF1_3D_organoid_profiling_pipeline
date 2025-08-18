#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile aggregation.

# In[1]:


import argparse
import pathlib

import pandas as pd
from pycytominer import aggregate

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


# In[2]:


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
    patient = "SARCO361"


# In[3]:


# pathing
sc_fs_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/4.feature_selected_profiles/sc_fs.parquet"
).resolve(strict=True)
organoid_fs_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/4.feature_selected_profiles/organoid_fs.parquet"
).resolve(strict=True)


# output path
sc_agg_well_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_agg_well_level.parquet"
).resolve()
sc_agg_well_parent_organoid_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_agg_parent_organoid_level.parquet"
).resolve()
sc_consensus_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/sc_consensus.parquet"
).resolve()

organoid_agg_well_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/organoid_agg_well_level.parquet"
).resolve()
organoid_consensus_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/5.aggregated_profiles/organoid_consensus.parquet"
).resolve()

organoid_consensus_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_fs = pd.read_parquet(sc_fs_path)
organoid_fs = pd.read_parquet(organoid_fs_path)


# ### Aggregate the single cell profiles
# We will aggregated with a few different stratifications:
# 1. Well
# 2. Well and parent organoid
# 3. Treatment - i.e. the consensus profile for each treatment

# In[5]:


sc_fs.head()


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
]
sc_features_columns = [col for col in sc_fs.columns if col not in sc_metadata_columns]
sc_features_df = sc_fs.drop(columns=sc_metadata_columns, errors="ignore")


# In[7]:


# stratification approach #1
sc_well_agg = aggregate(
    population_df=sc_fs,
    strata=["Well", "treatment", "Target", "Class", "Therapeutic Categories"],
    features=sc_features_columns,
    operation="median",
)
sc_well_agg.to_parquet(sc_agg_well_output_path, index=False)

# stratification approach #2
sc_well_parent_organoid_agg = aggregate(
    population_df=sc_fs,
    strata=[
        "Well",
        "parent_organoid",
        "treatment",
        "Target",
        "Class",
        "Therapeutic Categories",
    ],
    features=sc_features_columns,
    operation="median",
)
sc_well_parent_organoid_agg.to_parquet(
    sc_agg_well_parent_organoid_output_path, index=False
)
# stratification approach #3
sc_consensus = aggregate(  # a.k.a. consensus
    population_df=sc_fs,
    strata=["treatment", "Target", "Class", "Therapeutic Categories"],
    features=sc_features_columns,
    operation="median",
)
sc_consensus.to_parquet(sc_consensus_output_path, index=False)


# ### Aggregate the organoid profiles
# We will aggregated with a few different stratifications:
# 1. Well
# 2. Treatment - i.e. the consensus profile for each treatment

# In[8]:


organoid_fs.head()


# In[9]:


organoid_metadata_columns = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "image_set",
    "Target",
    "Class",
    "Therapeutic Categories",
    "Well",
    "parent_organoid",
    "MOA",
]
organoidfeatures_columns = [
    col for col in organoid_fs.columns if col not in organoid_metadata_columns
]
organoid_features_df = organoid_fs.drop(columns=sc_metadata_columns, errors="ignore")


# In[10]:


# stratification approach #1
organoid_well_agg = aggregate(
    population_df=organoid_fs,
    strata=["Well", "treatment", "Target", "Class", "Therapeutic Categories"],
    features=organoidfeatures_columns,
    operation="median",
)
organoid_well_agg.to_parquet(organoid_agg_well_output_path, index=False)

# stratification approach #2
organoid_consensus = aggregate(  # a.k.a. consensus
    population_df=organoid_fs,
    strata=["treatment", "Target", "Class", "Therapeutic Categories"],
    features=organoidfeatures_columns,
    operation="median",
)
organoid_consensus.to_parquet(organoid_consensus_output_path, index=False)


# In[11]:


organoid_well_agg.head()


# In[12]:


organoid_consensus.head()
