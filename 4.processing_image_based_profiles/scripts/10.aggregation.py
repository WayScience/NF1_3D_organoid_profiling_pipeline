#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile aggregation.

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import aggregate

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[2]:


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
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sc_fs.parquet"
).resolve(strict=True)
organoid_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/organoid_fs.parquet"
).resolve(strict=True)
sc_sammed_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sammed_sc_fs.parquet"
).resolve(strict=True)
organoid_sammed_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sammed_organoid_fs.parquet"
).resolve(strict=True)
nucleocentric_sammed_sc_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sammed_nucleocentric_fs.parquet"
).resolve(strict=True)
nucleocentric_chammi_sc_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/chammi_nucleocentric_fs.parquet"
).resolve(strict=True)


# output path
sc_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/sc_agg_well_level.parquet"
).resolve()
sc_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/sc_consensus.parquet"
).resolve()

organoid_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/organoid_agg_well_level.parquet"
).resolve()
organoid_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/organoid_consensus.parquet"
).resolve()

sc_sammed_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/sammed_sc_agg_well_level.parquet"
).resolve()
sc_sammed_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/sammed_sc_consensus.parquet"
).resolve()

organoid_sammed_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/sammed_organoid_agg_well_level.parquet"
).resolve()
organoid_sammed_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/sammed_organoid_consensus.parquet"
).resolve()

nucleocentric_sammed_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/sammed_nucleocentric_agg_well_level.parquet"
).resolve()
nucleocentric_sammed_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/sammed_nucleocentric_consensus.parquet"
).resolve()

nucleocentric_chammi_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/chammi_nucleocentric_agg_well_level.parquet"
).resolve()
nucleocentric_chammi_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/chammi_nucleocentric_consensus.parquet"
).resolve()

nucleocentric_chammi_agg_well_output_path.parent.mkdir(parents=True, exist_ok=True)
nucleocentric_chammi_consensus_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_fs = pd.read_parquet(sc_fs_path)
organoid_fs = pd.read_parquet(organoid_fs_path)
sc_sammed_fs = pd.read_parquet(sc_sammed_fs_path)
organoid_sammed_fs = pd.read_parquet(organoid_sammed_fs_path)
nucleocentric_sammed_sc_fs = pd.read_parquet(nucleocentric_sammed_sc_fs_path)
nucleocentric_chammi_sc_fs = pd.read_parquet(nucleocentric_chammi_sc_fs_path)


# In[5]:


run_dict = {
    "sc": {
        "df": sc_fs,
        "agg_well_output_path": sc_agg_well_output_path,
        "consensus_output_path": sc_consensus_output_path,
    },
    "organoid": {
        "df": organoid_fs,
        "agg_well_output_path": organoid_agg_well_output_path,
        "consensus_output_path": organoid_consensus_output_path,
    },
    "sc_sammed": {
        "df": sc_sammed_fs,
        "agg_well_output_path": sc_sammed_agg_well_output_path,
        "consensus_output_path": sc_sammed_consensus_output_path,
    },
    "organoid_sammed": {
        "df": organoid_sammed_fs,
        "agg_well_output_path": organoid_sammed_agg_well_output_path,
        "consensus_output_path": organoid_sammed_consensus_output_path,
    },
    "nucleocentric_sammed_sc": {
        "df": nucleocentric_sammed_sc_fs,
        "agg_well_output_path": nucleocentric_sammed_agg_well_output_path,
        "consensus_output_path": nucleocentric_sammed_consensus_output_path,
    },
    "nucleocentric_chammi_sc": {
        "df": nucleocentric_chammi_sc_fs,
        "agg_well_output_path": nucleocentric_chammi_agg_well_output_path,
        "consensus_output_path": nucleocentric_chammi_consensus_output_path,
    },
}


# ### Aggregate the profiles
# We will aggregated with a few different stratifications:
# 1. Well
# 2. Treatment - i.e. the consensus profile for each treatment

# In[6]:


aggregate_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"]
consensus_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment"]


# In[7]:


for profile_name in run_dict.keys():
    print(f"Processing {profile_name} profiles...")
    df = run_dict[profile_name]["df"]
    agg_well_output_path = run_dict[profile_name]["agg_well_output_path"]
    consensus_output_path = run_dict[profile_name]["consensus_output_path"]

    metadata_columns = [x for x in df.columns if "Metadata" in x]
    features_columns = [col for col in df.columns if col not in metadata_columns]

    # aggregate by well
    agg_well_df = aggregate(
        population_df=df,
        strata=aggregate_strata,
        features=features_columns,
        operation="median",
    )
    agg_well_df.to_parquet(agg_well_output_path)

    # aggregate by treatment
    consensus_df = aggregate(
        population_df=df,
        strata=consensus_strata,
        features=features_columns,
        operation="median",
    )
    consensus_df.to_parquet(consensus_output_path)


# In[ ]:
