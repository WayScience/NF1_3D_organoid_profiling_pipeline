#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import aggregate, feature_select

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir


# In[2]:


patient_ids_path = pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(patient_ids_path, header=None, names=["patient_id"], dtype=str)[
    "patient_id"
].to_list()

all_patients_output_path = pathlib.Path(
    f"{profile_base_dir}/data/all_patient_profiles"
).resolve()
all_patients_output_path.mkdir(parents=True, exist_ok=True)


# In[3]:


levels_to_merge_dict = {
    "sc_norm": [],
    "organoid_norm": [],
    "sammed_sc_norm": [],
    "sammed_organoid_norm": [],
    "sammed_nucleocentric_norm": [],
    "chammi_nucleocentric_norm": [],
}


# In[4]:


for patient in patients:
    norm_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles"
    )
    for file in norm_path.glob("*.parquet"):
        for level in levels_to_merge_dict.keys():
            if level == file.stem:
                levels_to_merge_dict[level].append(file)
levels_to_merge_dict


# In[5]:


feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "variance_threshold",  # comment out to remove variance thresholding
    "correlation_threshold",  # comment out to remove correlation thresholding
]
na_cutoff = 0.05
corr_threshold = 0.9
freq_cut = 0.01
unique_cut = 0.01


# In[6]:


aggregate_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"]
consensus_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment"]


# In[7]:


for profile_type, files in levels_to_merge_dict.items():
    print(f"Found {len(files)} files for {profile_type} level.")
    list_of_dfs = []
    for file in files:
        df = pd.read_parquet(file)
        list_of_dfs.append(df)
    df = pd.concat(list_of_dfs, ignore_index=True)

    print(f"Concatenated DataFrame for {profile_type} has the shape: {df.shape}")
    df.to_parquet(
        f"{all_patients_output_path}/{profile_type}_norm_profile.parquet",
        index=False,
    )
    ###############################################
    # Feature selection
    ###############################################
    metadata_cols = [x for x in df.columns if "Metadata" in x]
    # only perform feature selection on DMSO and staurosporine treatments and apply to rest of profiles
    all_trt_df = df.copy()
    df = df.loc[
        df["Metadata_Experiment_Treatment"].isin(["DMSO 1%", "Staurosporine 10 nM"])
    ]
    # feature selection
    feature_columns = [col for col in df.columns if col not in metadata_cols]
    fs_profiles = feature_select(
        df,
        operation=feature_select_ops,
        features=feature_columns,
        na_cutoff=na_cutoff,
        corr_threshold=corr_threshold,  # comment out to use default value
        freq_cut=freq_cut,  # comment out to use default value
        unique_cut=unique_cut,  # comment out to use default value
    )
    # apply feature selection to all profiles
    fs_profiles = all_trt_df[
        [col for col in all_trt_df.columns if col in fs_profiles.columns]
    ]
    fs_profiles.to_parquet(
        f"{all_patients_output_path}/{profile_type}_fs_profiles.parquet",
        index=False,
    )
    ###############################################
    # Aggregation
    ###############################################
    feature_columns = [col for col in fs_profiles.columns if col not in metadata_cols]
    # aggregate the profiles
    sc_agg_df = aggregate(
        population_df=fs_profiles,
        strata=aggregate_strata,
        features=feature_columns,
        operation="median",
    )
    sc_agg_df.to_parquet(
        f"{all_patients_output_path}/{profile_type}_sc_agg_profiles.parquet",
        index=False,
    )
    ###############################################
    # Consensus profiles
    ###############################################
    # consensus profiles
    sc_consensus_df = aggregate(
        population_df=fs_profiles,
        strata=consensus_strata,
        features=feature_columns,
        operation="median",
    )
    sc_consensus_df.to_parquet(
        f"{all_patients_output_path}/{profile_type}_sc_consensus_profiles.parquet",
        index=False,
    )
    print("The number features before feature selection:", df.shape[1])
    print("The number features after feature selection:", fs_profiles.shape[1])
    print("The number of profiles after aggregation:", sc_agg_df.shape[0])
    print(
        "The number of profiles after consensus profile generation:",
        sc_consensus_df.shape[0],
    )
