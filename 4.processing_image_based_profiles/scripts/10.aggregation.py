#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile aggregation.

# In[1]:


import os
import pathlib

import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import aggregate

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
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
sc_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.feature_selected_profiles/sc_fs.parquet"
).resolve(strict=True)
organoid_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.feature_selected_profiles/organoid_fs.parquet"
).resolve(strict=True)


# output path
sc_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/sc_agg_well_level.parquet"
).resolve()
sc_agg_well_parent_organoid_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/sc_agg_parent_organoid_level.parquet"
).resolve()
sc_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/sc_consensus.parquet"
).resolve()

organoid_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/organoid_agg_well_level.parquet"
).resolve()
organoid_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.aggregated_profiles/organoid_consensus.parquet"
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


sc_metadata_columns = [x for x in sc_fs.columns if "Metadata" in x]

sc_metadata_columns += [
    "Area.Size.Shape_Cell_CENTER.X",
    "Area.Size.Shape_Cell_CENTER.Y",
    "Area.Size.Shape_Cell_CENTER.Z",
]
sc_features_columns = [col for col in sc_fs.columns if col not in sc_metadata_columns]
sc_features_df = sc_fs.drop(columns=sc_metadata_columns, errors="ignore")


# In[7]:


# stratification approach #1
sc_well_agg = aggregate(
    population_df=sc_fs,
    strata=[
        "Metadata_Well",
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
    features=sc_features_columns,
    operation="median",
)
sc_well_agg.to_parquet(sc_agg_well_output_path, index=False)

# stratification approach #2
sc_well_parent_organoid_agg = aggregate(
    population_df=sc_fs,
    strata=[
        "Metadata_Well",
        "Metadata_parent_organoid",
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
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
    strata=[
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
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


organoid_metadata_columns = [x for x in organoid_fs.columns if "Metadata" in x]
organoid_metadata_columns += [
    "Area.Size.Shape_Organoid_CENTER.X",
    "Area.Size.Shape_Organoid_CENTER.Y",
    "Area.Size.Shape_Organoid_CENTER.Z",
]
organoid_features_columns = [
    col for col in organoid_fs.columns if col not in organoid_metadata_columns
]
organoid_features_columns = [
    col for col in organoid_fs.columns if col not in organoid_metadata_columns
]
organoid_features_df = organoid_fs.drop(
    columns=organoid_metadata_columns, errors="ignore"
)


# In[10]:


# stratification approach #1
organoid_well_agg = aggregate(
    population_df=organoid_fs,
    strata=[
        "Metadata_Well",
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
    features=organoid_features_columns,
    operation="median",
)
organoid_well_agg.to_parquet(organoid_agg_well_output_path, index=False)

# stratification approach #2
organoid_consensus = aggregate(  # a.k.a. consensus
    population_df=organoid_fs,
    strata=[
        "Metadata_treatment",
        "Metadata_Target",
        "Metadata_Class",
        "Metadata_Therapeutic_Categories",
    ],
    features=organoid_features_columns,
    operation="median",
)
organoid_consensus.to_parquet(organoid_consensus_output_path, index=False)


# In[11]:


organoid_well_agg.head()


# In[12]:


organoid_consensus.head()
