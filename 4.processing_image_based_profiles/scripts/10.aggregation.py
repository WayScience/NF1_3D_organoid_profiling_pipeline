#!/usr/bin/env python
# coding: utf-8

# # 10. Aggregation
#
# ## Purpose
# Aggregate each feature-selected profile across single cells or organoids into
# two levels of summary:
# 1. **Well-level** — median across all objects in a well (`PatientTumor × Well`)
# 2. **Consensus** — median across all objects sharing a treatment (`PatientTumor × Treatment`)
#
# Both aggregations use pycytominer's `aggregate` with median as the operation.
#
# This is **step 10 of Stage 4 (image-based profiling)**. It runs once per patient
# and must follow `9.feature_selection.ipynb`.
#
# ## Inputs
#
# Six feature-selected parquets from `6.feature_selected_profiles/`:
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
# ## Outputs
#
# Twelve parquets across two stage directories:
#
# | Directory | Files |
# |---|---|
# | `7.aggregated_profiles/` | `sc_agg_well_level.parquet`, `organoid_agg_well_level.parquet`, `sammed_sc_agg_well_level.parquet`, `sammed_organoid_agg_well_level.parquet`, `sammed_nucleocentric_agg_well_level.parquet`, `nucleocentric_morphem_agg_well_level.parquet` |
# | `8.consensus_profiles/` | `sc_consensus.parquet`, `organoid_consensus.parquet`, `sammed_sc_consensus.parquet`, `sammed_organoid_consensus.parquet`, `sammed_nucleocentric_consensus.parquet`, `nucleocentric_morphem_consensus.parquet` |
#
# ## Notes
# - QC-flagged rows are not filtered before aggregation; downstream analysis decides
#   whether to exclude them.
# - Consensus profiles collapse treatment replicates to a single row per treatment,
#   making them the primary input for treatment-level comparisons.

# In[19]:


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
profile_base_dir = root_dir


# In[20]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[21]:


## Pathing
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
nucleocentric_morphem_sc_fs_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/nucleocentric_morphem_fs.parquet"
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

nucleocentric_morphem_agg_well_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/7.aggregated_profiles/nucleocentric_morphem_agg_well_level.parquet"
).resolve()
nucleocentric_morphem_consensus_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/8.consensus_profiles/nucleocentric_morphem_consensus.parquet"
).resolve()

nucleocentric_morphem_agg_well_output_path.parent.mkdir(parents=True, exist_ok=True)
nucleocentric_morphem_consensus_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[22]:


# read in the data
sc_fs = pd.read_parquet(sc_fs_path)
organoid_fs = pd.read_parquet(organoid_fs_path)
sc_sammed_fs = pd.read_parquet(sc_sammed_fs_path)
organoid_sammed_fs = pd.read_parquet(organoid_sammed_fs_path)
nucleocentric_sammed_sc_fs = pd.read_parquet(nucleocentric_sammed_sc_fs_path)
nucleocentric_morphem_sc_fs = pd.read_parquet(nucleocentric_morphem_sc_fs_path)

print(f"SC feature-selected loaded. Shape: {sc_fs.shape}")
print(f"Organoid feature-selected loaded. Shape: {organoid_fs.shape}")
print(f"SAMMed3D SC feature-selected loaded. Shape: {sc_sammed_fs.shape}")
print(f"SAMMed3D organoid feature-selected loaded. Shape: {organoid_sammed_fs.shape}")
print(
    f"SAMMed3D nucleocentric feature-selected loaded. Shape: {nucleocentric_sammed_sc_fs.shape}"
)
print(
    f"morphem nucleocentric feature-selected loaded. Shape: {nucleocentric_morphem_sc_fs.shape}"
)


# In[23]:


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
    "nucleocentric_morphem_sc": {
        "df": nucleocentric_morphem_sc_fs,
        "agg_well_output_path": nucleocentric_morphem_agg_well_output_path,
        "consensus_output_path": nucleocentric_morphem_consensus_output_path,
    },
}


# ## Aggregate the profiles
#
# Each profile type is aggregated at two levels:
# 1. **Well-level** (`aggregate_strata`): median across all objects in a well,
#    grouped by `PatientTumor x Well`. Preserves within-patient well-to-well variation.
# 2. **Consensus** (`consensus_strata`): median across all objects sharing a treatment,
#    grouped by `PatientTumor x Treatment`. Collapses replicates for treatment-level
#    comparisons.

# In[27]:


# Well-level strata: one row per (patient, well) combination
aggregate_strata = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"]
# Consensus strata: one row per (patient, treatment) combination
consensus_strata = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
]


# In[28]:


for profile_name in run_dict.keys():
    print(f"Processing {profile_name} profiles...")
    df = run_dict[profile_name]["df"]
    agg_well_output_path = run_dict[profile_name]["agg_well_output_path"]
    consensus_output_path = run_dict[profile_name]["consensus_output_path"]

    metadata_columns = [x for x in df.columns if x.startswith("Metadata_")]
    features_columns = [col for col in df.columns if col not in metadata_columns]

    # aggregate by well
    agg_well_df = aggregate(
        population_df=df,
        strata=aggregate_strata,
        features=features_columns,
        operation="median",
    )
    agg_well_df.to_parquet(agg_well_output_path, index=False)
    print(f"  Well-level aggregated. Shape: {agg_well_df.shape}")

    # aggregate by treatment
    consensus_df = aggregate(
        population_df=df,
        strata=consensus_strata,
        features=features_columns,
        operation="median",
    )
    consensus_df.to_parquet(consensus_output_path, index=False)
    print(f"  Consensus aggregated. Shape: {consensus_df.shape}")
