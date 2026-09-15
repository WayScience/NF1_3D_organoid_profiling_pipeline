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


import argparse
import os
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import aggregate, feature_select
from pycytominer.cyto_utils import infer_cp_features

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir


# In[2]:


# Unlike the other per-patient steps, this script never accepted a
# --image_based_profiles_subparent_name argument at all -- it hardcoded
# "image_based_profiles", so it could never see per-patient output written
# under a different subparent name (e.g. this project's own
# "image_based_profiles_production_zedprofiler"). Defaults preserve the
# previous hardcoded behavior for existing callers.
if not in_notebook:
    _arg_parser = argparse.ArgumentParser()
    _arg_parser.add_argument(
        "--image_based_profiles_subparent_name", default="image_based_profiles"
    )
    _arg_parser.add_argument("--output_subdir", default="all_patient_profiles")
    _args = _arg_parser.parse_args()
    image_based_profiles_subparent_name = _args.image_based_profiles_subparent_name
    output_subdir = _args.output_subdir
else:
    image_based_profiles_subparent_name = "image_based_profiles"
    output_subdir = "all_patient_profiles"

patient_ids_path = pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(patient_ids_path, header=None, names=["patient_id"], dtype=str)[
    "patient_id"
].to_list()

all_patients_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{output_subdir}"
).resolve()
all_patients_output_path.mkdir(parents=True, exist_ok=True)
# 0.normalized_profiles/ gets its own mkdir at write time below; the other
# three were missing theirs entirely -- confirmed a real, pre-existing bug:
# output() -> to_parquet() fails outright the first time a fresh
# all_patients_output_path is used, since pandas requires the parent
# directory to already exist.
for _subdir in (
    "1.feature_selected_profiles",
    "2.aggregated_profiles",
    "3.consensus_profiles",
):
    (all_patients_output_path / _subdir).mkdir(parents=True, exist_ok=True)


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
        f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles"
    )
    for file in norm_path.glob("*.parquet"):
        for level in levels_to_merge_dict.keys():
            if level == file.stem:
                levels_to_merge_dict[level].append(file)
levels_to_merge_dict


# In[ ]:


# Feature selection operations applied in order:
#   drop_na_columns      — remove features with >na_cutoff fraction of NaN values
#   blocklist            — remove features on the pycytominer blocklist (known noisy/artifactual)
#   variance_threshold   — remove near-constant features (low frequency or unique value ratio)
#   correlation_threshold — remove one feature from each pair with Pearson r > corr_threshold
feature_select_ops = [
    "drop_na_columns",
    "drop_outliers",
    "blocklist",
    "variance_threshold",
    "frequency_threshold",
    "correlation_threshold",  # comment out to remove correlation thresholding
]
na_cutoff = 0.05  # drop features with >5% NaN
outlier_cutoff = 100  # drop features whose min/max absolute value exceeds this
corr_threshold = 0.90  # drop one of any pair with Pearson r >= 0.90
freq_cut = 0.05  # frequency threshold: most-common / second-most-common value ratio
# unique_cut (frequency_threshold) and min_variance (variance_threshold) are
# intentionally left at pycytominer's own defaults (0.01 and 1e-6) -- not overridden.


# In[6]:


# Well-level strata: one row per (patient, well) combination
aggregate_strata = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Well",
    "Metadata_Experiment_Class",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Target",
    "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Unit",
]
# Consensus strata: one row per (patient, treatment) combination
consensus_strata = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Class",
    "Metadata_Experiment_Target",
    "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Experiment_Unit",
]
potential_compartments = ["Cell", "Cytoplasm", "Nuclei", "Organoid", "Nucleocentric"]


# In[ ]:


for profile_type, files in levels_to_merge_dict.items():
    print(f"Found {len(files)} files for {profile_type} level.")
    if not files:
        # No patient produced this profile type -- expected for the 4
        # deep-learning types when the whole cohort has no deep-learning
        # features (e.g. ZEDProfiler-only); pd.concat([]) would otherwise
        # raise ValueError("No objects to concatenate").
        print(f"  Skipping {profile_type}: no per-patient files found.")
        continue
    list_of_dfs = []
    for file in files:
        df = pd.read_parquet(file)
        list_of_dfs.append(df)
    df = pd.concat(list_of_dfs, ignore_index=True)

    print(f"Concatenated DataFrame for {profile_type} has the shape: {df.shape}")
    normalized_profiles_path = pathlib.Path(
        f"{all_patients_output_path}/0.normalized_profiles/{profile_type}_norm_profile.parquet"
    )
    normalized_profiles_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        normalized_profiles_path,
        index=False,
    )
    ###############################################
    # Feature selection
    ###############################################
    feature_columns = infer_cp_features(
        df, metadata=False, compartments=potential_compartments
    )
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    # Confirmed real, pre-existing bugs unrelated to ZEDProfiler compatibility,
    # both of which would crash this call for any dataset: `method` isn't a
    # feature_select() parameter at all (that belongs to normalize()), and
    # `unique_cut` was referenced here without ever being defined anywhere in
    # this file. Dropped rather than given a value -- this file's own comment
    # a few lines up already states unique_cut/min_variance are "intentionally
    # left at pycytominer's own defaults", matching 9.feature_selection.py's
    # identical comment and identical (omit-the-kwarg) behavior for the same
    # operation set. outlier_cutoff was defined above but never actually
    # passed through, so "drop_outliers" was silently using pycytominer's own
    # default (500.0) instead of this file's own outlier_cutoff = 100 -- now
    # passed explicitly, matching 9.feature_selection.py's own call.
    fs_profile = feature_select(
        df,
        operation=feature_select_ops,
        features=infer_cp_features(
            df, metadata=False, compartments=potential_compartments
        ),
        na_cutoff=na_cutoff,
        outlier_cutoff=outlier_cutoff,
        corr_threshold=corr_threshold,  # comment out to use default value
        freq_cut=freq_cut,  # comment out to use default value
        samples="(Metadata_Experiment_Treatment == 'DMSO' and Metadata_Experiment_Dose == 1) or (Metadata_Experiment_Treatment == 'Staurosporine' and Metadata_Experiment_Dose == 10)",
        output_file=f"{all_patients_output_path}/1.feature_selected_profiles/{profile_type}_fs_profiles.parquet",
        output_type="parquet",
    )

    ###############################################
    # Aggregation — produces well-level and consensus parquets
    ###############################################
    fs_df = pd.read_parquet(fs_profile)
    # aggregate the profiles
    agg_profile = aggregate(
        population_df=fs_df,
        strata=aggregate_strata,
        features=infer_cp_features(
            fs_df, metadata=False, compartments=potential_compartments
        ),
        operation="median",
        output_file=f"{all_patients_output_path}/2.aggregated_profiles/{profile_type}_sc_agg_profiles.parquet",
        output_type="parquet",
    )

    ###############################################
    # Consensus profiles
    ###############################################
    consensus_profile = aggregate(
        population_df=fs_df,
        strata=consensus_strata,
        features=infer_cp_features(
            fs_df, metadata=False, compartments=potential_compartments
        ),
        operation="median",
        output_file=f"{all_patients_output_path}/3.consensus_profiles/{profile_type}_sc_consensus_profiles.parquet",
        output_type="parquet",
    )

    ###############################################
    # print shapes as a sanity check
    ###############################################
    fs_pq_file, agg_pq_file, consensus_pq_file = (
        pq.ParquetFile(fs_profile),
        pq.ParquetFile(agg_profile),
        pq.ParquetFile(consensus_profile),
    )
    fs_shape = (fs_pq_file.metadata.num_rows, len(fs_pq_file.schema.names))
    agg_shape = (agg_pq_file.metadata.num_rows, len(agg_pq_file.schema.names))
    consensus_shape = (
        consensus_pq_file.metadata.num_rows,
        len(consensus_pq_file.schema.names),
    )
    print(f"  Feature-selected profile shape: {fs_shape}")
    print(f"  Well-level aggregated profile shape: {agg_shape}")
    print(f"  Consensus aggregated profile shape: {consensus_shape}")
