#!/usr/bin/env python
# coding: utf-8

# # 9. Feature Selection
#
# ## Purpose
# Remove low-information features from each normalized profile using pycytominer's
# `feature_select`. Feature selection is **fit on a reference subset** (DMSO and
# Staurosporine wells) but **applied to all treatments**, so the retained feature set
# is determined by the reference population and then used to subset the full dataset.
#
# We fit on DMSO and Staurosporine wells because we expect that these two treatments will have the most distinct profiles, so features that are uninformative in this context are likely to be uninformative across the full treatment set.
# By fitting on this reference subset, we can identify and retain features that capture meaningful variation while removing those that do not contribute to distinguishing between treatments.
#
# This is **step 9 of Stage 4 (image-based profiling)**. It runs once per patient
# and must follow `8.normalization.ipynb`.
#
# ## Inputs
#
# Six normalized parquets from `5.normalized_profiles/`:
#
# | File | Profile type |
# |---|---|
# | `sc_norm.parquet` | Hand-crafted SC |
# | `organoid_norm.parquet` | Hand-crafted organoid |
# | `sammed_sc_norm.parquet` | Deep-learning SC (SAMMed3D) |
# | `sammed_organoid_norm.parquet` | Deep-learning organoid (SAMMed3D) |
# | `sammed_nucleocentric_norm.parquet` | Deep-learning nucleocentric (SAMMed3D) |
# | `nucleocentric_morphem_norm.parquet` | Deep-learning nucleocentric (morphem) |
#
# ## Outputs
#
# Six feature-selected parquets in `6.feature_selected_profiles/`:
#
# | File | Profile type |
# |---|---|
# | `sc_fs.parquet` | Hand-crafted SC |
# | `organoid_fs.parquet` | Hand-crafted organoid |
# | `sammed_sc_fs.parquet` | Deep-learning SC (SAMMed3D) |
# | `sammed_organoid_fs.parquet` | Deep-learning organoid (SAMMed3D) |
# | `sammed_nucleocentric_fs.parquet` | Deep-learning nucleocentric (SAMMed3D) |
# | `nucleocentric_morphem_fs.parquet` | Deep-learning nucleocentric (morphem) |
#
# ## Notes
# - Feature selection is fit on DMSO and Staurosporine rows only, then the retained
#   feature set is applied back to the full (all-treatment) dataset. This ensures
#   feature selection is not biased by the full treatment distribution.
# - QC-flagged rows are not filtered here; that is left to downstream analysis.

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import feature_select

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


## Pathing
sc_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sc_norm.parquet"
).resolve(strict=True)
organoid_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/organoid_norm.parquet"
).resolve(strict=True)
sc_sammed_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sammed_sc_norm.parquet"
).resolve(strict=True)
organoid_sc_sammed_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sammed_organoid_norm.parquet"
).resolve(strict=True)
nucleocentric_sammed_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sammed_nucleocentric_norm.parquet"
).resolve(strict=True)
nucleocentric_morphem_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/nucleocentric_morphem_norm.parquet"
).resolve(strict=True)


# output path
sc_fs_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sc_fs.parquet"
).resolve()
organoid_fs_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/organoid_fs.parquet"
).resolve()
sc_sammed_feature_selected_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sammed_sc_fs.parquet"
).resolve()
organoid_sc_sammed_feature_selected_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sammed_organoid_fs.parquet"
).resolve()
nucleocentric_sammed_feature_selected_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/sammed_nucleocentric_fs.parquet"
).resolve()
nucleocentric_morphem_feature_selected_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/nucleocentric_morphem_fs.parquet"
).resolve()

organoid_fs_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_normalized = pd.read_parquet(sc_normalized_path)
organoid_normalized = pd.read_parquet(organoid_normalized_path)
sc_sammed_normalized = pd.read_parquet(sc_sammed_normalized_path)
organoid_sc_sammed_normalized = pd.read_parquet(organoid_sc_sammed_normalized_path)
nucleocentric_sammed_normalized = pd.read_parquet(nucleocentric_sammed_normalized_path)
nucleocentric_morphem_normalized = pd.read_parquet(
    nucleocentric_morphem_normalized_path
)

print(f"SC normalized loaded. Shape: {sc_normalized.shape}")
print(f"Organoid normalized loaded. Shape: {organoid_normalized.shape}")
print(f"SAMMed3D SC normalized loaded. Shape: {sc_sammed_normalized.shape}")
print(
    f"SAMMed3D organoid normalized loaded. Shape: {organoid_sc_sammed_normalized.shape}"
)
print(
    f"SAMMed3D nucleocentric normalized loaded. Shape: {nucleocentric_sammed_normalized.shape}"
)
print(
    f"morphem nucleocentric normalized loaded. Shape: {nucleocentric_morphem_normalized.shape}"
)


# In[5]:


run_dict = {
    "sc_normalized": {
        "df": sc_normalized,
        "output_path": sc_fs_output_path,
    },
    "organoid_normalized": {
        "df": organoid_normalized,
        "output_path": organoid_fs_output_path,
    },
    "sc_sammed": {
        "df": sc_sammed_normalized,
        "output_path": sc_sammed_feature_selected_output_path,
    },
    "organoid_sc_sammed": {
        "df": organoid_sc_sammed_normalized,
        "output_path": organoid_sc_sammed_feature_selected_output_path,
    },
    "nucleocentric_sammed": {
        "df": nucleocentric_sammed_normalized,
        "output_path": nucleocentric_sammed_feature_selected_output_path,
    },
    "nucleocentric_chammi": {
        "df": nucleocentric_morphem_normalized,
        "output_path": nucleocentric_morphem_feature_selected_output_path,
    },
}


# In[6]:


# Feature selection operations applied in order:
#   drop_na_columns      — remove features with >na_cutoff fraction of NaN values
#   blocklist            — remove features on the pycytominer blocklist (known noisy/artifactual)
#   correlation_threshold — remove one feature from each pair with Pearson r > corr_threshold
#   variance_threshold   — remove near-constant features (low frequency or unique value ratio)
feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "correlation_threshold",  # comment out to remove correlation thresholding
    "variance_threshold",  # comment out to remove variance thresholding
]
na_cutoff = 0.05  # drop features with >5% NaN
corr_threshold = 0.90  # drop one of any pair with Pearson r >= 0.95
freq_cut = 0.05  # variance threshold: most-common / second-most-common value ratio
unique_cut = 0.05  # variance threshold: minimum fraction of unique values


# ## Feature select the profiles
#
# For each profile type:
# 1. Feature selection is **fit** on DMSO and Staurosporine rows only — a controlled
#    reference that avoids biasing feature selection on the full treatment distribution.
# 2. The retained feature set is applied back to the **full dataset** (all treatments),
#    so no treatment rows are dropped from the output.

# In[ ]:


for profile_name in run_dict.keys():
    print(f"Running feature selection for {profile_name} profiles...")
    df = run_dict[profile_name]["df"]
    output_path = run_dict[profile_name]["output_path"]
    # prep profiles for feature selection
    # grab metadata columns
    metadata_columns = [x for x in df.columns if x.startswith("Metadata_")]
    # grab feature columns
    features_columns = [col for col in df.columns if col not in metadata_columns]
    # Phase 1: fit feature selection on reference treatments only.
    # all_trt_df retains the full dataset; df is narrowed to the reference subset.
    all_trt_df = df.copy()
    df = df.loc[df["Metadata_Experiment_Treatment"].isin(["DMSO", "Staurosporine"])]

    # run feature selection
    fs_profiles = feature_select(
        df,
        operation=feature_select_ops,
        features=features_columns,
        na_cutoff=na_cutoff,
        corr_threshold=corr_threshold,
        freq_cut=freq_cut,
        unique_cut=unique_cut,
    )
    # Phase 2: apply the retained feature set back to the full dataset.
    fs_profiles = all_trt_df[
        [col for col in all_trt_df.columns if col in fs_profiles.columns]
    ]

    original_data_shape = df.shape
    print("The number features before feature selection:", original_data_shape[1])
    print("The number features after feature selection:", fs_profiles.shape[1])
    fs_profiles.to_parquet(output_path, index=False)
