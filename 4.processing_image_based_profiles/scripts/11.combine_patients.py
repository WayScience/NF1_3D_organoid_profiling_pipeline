#!/usr/bin/env python
# coding: utf-8

# # 11. Combine Patients
#
# ## Purpose
# Combine per-patient normalized profiles across all patients into a single
# cross-patient dataset, then run feature selection and aggregation at the
# population level.
#
# This is **step 11 of Stage 4 (image-based profiling)** and the only notebook
# that runs **once globally** (not per-patient). It must follow `10.aggregation.ipynb`
# for all patients.
#
# ## Inputs
# - `data/patient_IDs.txt` — list of all patient IDs (one per line)
# - Per-patient `5.normalized_profiles/*.parquet` for each of 6 profile types
#
# ## Outputs
#
# All outputs go to `data/all_patient_profiles/`. For each of 6 profile types,
# four files are produced:
#
# | Suffix | Content |
# |---|---|
# | `*_norm_profile.parquet` | All-patient concatenated normalized profiles |
# | `*_fs_profiles.parquet` | Feature-selected (cross-patient FS) |
# | `*_sc_agg_profiles.parquet` | Well-level aggregated (median by PatientTumor × Well) |
# | `*_sc_consensus_profiles.parquet` | Consensus (median by PatientTumor × Treatment) |

# In[1]:


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
    "nucleocentric_morphem_norm": [],
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


# In[6]:


# Feature selection operations applied in order:
#   drop_na_columns      — remove features with >na_cutoff fraction of NaN values
#   blocklist            — remove features on the pycytominer blocklist (known noisy/artifactual)
#   variance_threshold   — remove near-constant features (low frequency or unique value ratio)
#   correlation_threshold — remove one feature from each pair with Pearson r > corr_threshold
feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "variance_threshold",  # comment out to remove variance thresholding
    "correlation_threshold",  # comment out to remove correlation thresholding
]
na_cutoff = 0.05  # drop features with >5% NaN
corr_threshold = 0.90  # drop one of any pair with Pearson r >= 0.95
freq_cut = 0.05  # variance threshold: most-common / second-most-common value ratio
unique_cut = 0.05  # variance threshold: minimum fraction of unique values


# In[7]:


# Well-level strata: one row per (patient, well) combination
aggregate_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"]
# Consensus strata: one row per (patient, treatment) combination
consensus_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment"]


# In[8]:


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
    metadata_cols = [x for x in df.columns if x.startswith("Metadata_")]
    # Phase 1: fit feature selection on reference treatments only.
    # all_trt_df retains the full dataset; df is narrowed to the reference subset.
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
    # Phase 2: apply retained feature set back to the full dataset.
    fs_profiles = all_trt_df[
        [col for col in all_trt_df.columns if col in fs_profiles.columns]
    ]
    fs_profiles.to_parquet(
        f"{all_patients_output_path}/{profile_type}_fs_profiles.parquet",
        index=False,
    )
    ###############################################
    # Aggregation — produces well-level and consensus parquets
    ###############################################
    # Recompute feature columns from fs_profiles after feature selection.
    feature_columns = [
        col for col in fs_profiles.columns if not col.startswith("Metadata_")
    ]
    # aggregate the profiles
    agg_df = aggregate(
        population_df=fs_profiles,
        strata=aggregate_strata,
        features=feature_columns,
        operation="median",
    )
    agg_df.to_parquet(
        f"{all_patients_output_path}/{profile_type}_sc_agg_profiles.parquet",
        index=False,
    )
    ###############################################
    # Consensus profiles
    ###############################################
    consensus_df = aggregate(
        population_df=fs_profiles,
        strata=consensus_strata,
        features=feature_columns,
        operation="median",
    )
    consensus_df.to_parquet(
        f"{all_patients_output_path}/{profile_type}_sc_consensus_profiles.parquet",
        index=False,
    )
    print("The number features before feature selection:", df.shape[1])
    print("The number features after feature selection:", fs_profiles.shape[1])
    print("The number of profiles after aggregation:", agg_df.shape[0])
    print(
        "The number of profiles after consensus profile generation:",
        consensus_df.shape[0],
    )
