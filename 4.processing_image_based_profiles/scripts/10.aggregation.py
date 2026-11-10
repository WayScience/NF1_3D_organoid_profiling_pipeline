#!/usr/bin/env python
# coding: utf-8

# # 10. Aggregation
#
# ## Purpose
# Aggregate each profile type into well-level and consensus (treatment-level)
# summaries, using median aggregation via pycytominer's `aggregate`.
#
# For each profile type, aggregation and consensus are each run from **two**
# inputs, producing 4 outputs total:
# 1. Normalized profiles (`5.normalized_profiles/*_norm.parquet`) — "from norm"
# 2. Feature-selected profiles (`6.feature_selected_profiles/*_fs.parquet`) — "from fs"
#
# Each input is aggregated at two levels:
# 1. **Well-level** — median across all objects in a well (`PatientTumor × Well`)
# 2. **Consensus** — median across all objects sharing a treatment (`PatientTumor × Treatment`)
#
# The aggregation/consensus strata below are kept identical (by hand) to the
# ones in `11.combine_patients.ipynb`'s cross-patient step, so both stages
# compute aggregation/consensus the same way.
#
# This is **step 10 of Stage 4 (image-based profiling)**. It runs once per patient
# and must follow `9.feature_selection.ipynb`.
#
# ## Inputs
#
# Per profile type, two parquets:
#
# | Directory | File pattern |
# |---|---|
# | `5.normalized_profiles/` | `{profile}_norm.parquet` |
# | `6.feature_selected_profiles/` | `{profile}_fs.parquet` |
#
# Profile types: `sc`, `organoid` (always present), plus the optional
# deep-learning types `sammed_sc`, `sammed_organoid`, `sammed_nucleocentric`,
# `nucleocentric_morphem` (only produced for datasets with deep-learning
# features; skipped here if the feature-selected file doesn't exist).
#
# ## Outputs
#
# Four parquets per profile type, across two stage directories:
#
# | Directory | File pattern |
# |---|---|
# | `7.aggregated_profiles/` | `{profile}_agg_from_norm.parquet`, `{profile}_agg_from_fs.parquet` |
# | `8.consensus_profiles/` | `{profile}_consensus_from_norm.parquet`, `{profile}_consensus_from_fs.parquet` |
#
# ## Notes
# - QC-flagged rows are not filtered before aggregation; downstream analysis decides
#   whether to exclude them.
# - Consensus profiles collapse treatment replicates to a single row per treatment,
#   making them the primary input for treatment-level comparisons.
# - "From norm" outputs aggregate every normalized feature column; "from fs" outputs
#   aggregate only the feature-selected subset. Comparing the two lets downstream
#   analysis see whether feature selection changed the aggregate/consensus signal.

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
# NOTE: previously this line unconditionally overrode bandicoot_check()
# with root_dir, meaning bandicoot was never actually used even when
# mounted. Removed so bandicoot_check()'s own bandicoot-first behavior
# takes effect.


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0037_T1_CQ1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


## Pathing
profile_dir = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}"
)
normalized_dir = profile_dir / "5.normalized_profiles"
feature_selected_dir = profile_dir / "6.feature_selected_profiles"
agg_output_dir = profile_dir / "7.aggregated_profiles"
consensus_output_dir = profile_dir / "8.consensus_profiles"
agg_output_dir.mkdir(parents=True, exist_ok=True)
consensus_output_dir.mkdir(parents=True, exist_ok=True)

# "sc" and "organoid" (hand-crafted) are always present. The 4 deep-learning
# profile types are optional: a dataset with no deep-learning features (e.g.
# ZEDProfiler-only) never has 9.feature_selection.py produce them, so each is
# only included below if its feature-selected file actually exists.
required_profile_names = ["sc", "organoid"]
optional_profile_names = [
    "sammed_sc",
    "sammed_organoid",
    "sammed_nucleocentric",
    "nucleocentric_morphem",
]

profile_names = list(required_profile_names)
for profile_name in optional_profile_names:
    fs_path = feature_selected_dir / f"{profile_name}_fs.parquet"
    if fs_path.exists():
        profile_names.append(profile_name)
    else:
        print(f"Skipping {profile_name}: {fs_path} does not exist.")


# ## Aggregate the profiles
#
# For each profile type, both the normalized and feature-selected input are
# aggregated at two levels:
# 1. **Well-level** (`aggregate_strata`): median across all objects in a well,
#    grouped by `PatientTumor x Well`. Preserves within-patient well-to-well variation.
# 2. **Consensus** (`consensus_strata`): median across all objects sharing a treatment,
#    grouped by `PatientTumor x Treatment`. Collapses replicates for treatment-level
#    comparisons.

# In[4]:


# Well-level strata: one row per (patient, well) combination. Kept identical
# (by hand) to 11.combine_patients.ipynb's cross-patient strata.
aggregate_strata = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Well",
    "Metadata_Experiment_Class",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Target",
    "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Unit",
    "Metadata_Experiment_ViabilityPercentage",
]
# Consensus strata: one row per (patient, treatment) combination.
consensus_strata = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Class",
    "Metadata_Experiment_Target",
    "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Experiment_Unit",
    "Metadata_Experiment_ViabilityPercentage",
]


# In[6]:


for profile_name in profile_names:
    print(f"Processing {profile_name} profiles...")
    norm_path = (normalized_dir / f"{profile_name}_norm.parquet").resolve(strict=True)
    fs_path = (feature_selected_dir / f"{profile_name}_fs.parquet").resolve(strict=True)
    norm_df = pd.read_parquet(norm_path)
    fs_df = pd.read_parquet(fs_path)
    print(f"  Normalized loaded. Shape: {norm_df.shape}")
    print(f"  Feature-selected loaded. Shape: {fs_df.shape}")

    # Produce the 4 standard outputs for this profile type: well-level
    # aggregation and treatment-level consensus, from both the normalized
    # and the feature-selected input.
    for source_name, df in (("norm", norm_df), ("fs", fs_df)):
        features_columns = [
            col for col in df.columns if not col.startswith("Metadata_")
        ]
        agg_path = agg_output_dir / f"{profile_name}_agg_from_{source_name}.parquet"
        consensus_path = (
            consensus_output_dir
            / f"{profile_name}_consensus_from_{source_name}.parquet"
        )

        agg_df = aggregate(
            population_df=df,
            strata=aggregate_strata,
            features=features_columns,
            operation="median",
            output_file=agg_path,
            output_type="parquet",
        )
        consensus_df = aggregate(
            population_df=df,
            strata=consensus_strata,
            features=features_columns,
            operation="median",
            output_file=consensus_path,
            output_type="parquet",
        )
        print(
            f"  agg_from_{source_name} aggregated. Shape: {pd.read_parquet(agg_path).shape}"
        )
        print(
            f"  consensus_from_{source_name} aggregated. Shape: {pd.read_parquet(consensus_path).shape}"
        )


# In[ ]:
