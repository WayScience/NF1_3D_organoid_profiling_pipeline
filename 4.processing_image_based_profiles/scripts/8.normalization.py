#!/usr/bin/env python
# coding: utf-8

# # 8. Normalization
#
# ## Purpose
# Normalize per-patient profiles using **MAD_robustize** (Median Absolute Deviation
# robust z-score), fitting on DMSO-treated samples that passed QC as the reference
# population. Normalization is applied independently to each of the six profile types.
#
# This is **step 8 of Stage 4 (image-based profiling)**. It runs once per patient
# and must follow `7b.single_cell_qc.ipynb`.
#
# ## Inputs
#
# | File | Stage |
# |---|---|
# | `4.qc_profiles/sc_flagged_outliers.parquet` | post-QC hand-crafted SC |
# | `4.qc_profiles/organoid_flagged_outliers.parquet` | post-QC hand-crafted organoid |
# | `3.annotated_profiles/sammed_sc_anno.parquet` | deep-learning SC (SAMMed3D) |
# | `3.annotated_profiles/sammed_organoid_anno.parquet` | deep-learning organoid (SAMMed3D) |
# | `3.annotated_profiles/nucleocentric_sammed_anno.parquet` | deep-learning nucleocentric (SAMMed3D) |
# | `3.annotated_profiles/nucleocentric_morphem_anno.parquet` | deep-learning nucleocentric (morphem) |
#
# ## Outputs
#
# Six normalized parquets in `data/{patient}/image_based_profiles/5.normalized_profiles/`:
#
# | File | Content |
# |---|---|
# | `sc_norm.parquet` | Normalized hand-crafted SC profiles |
# | `organoid_norm.parquet` | Normalized hand-crafted organoid profiles |
# | `sammed_sc_norm.parquet` | Normalized SAMMed3D SC profiles |
# | `sammed_organoid_norm.parquet` | Normalized SAMMed3D organoid profiles |
# | `sammed_nucleocentric_norm.parquet` | Normalized SAMMed3D nucleocentric profiles |
# | `nucleocentric_morphem_norm.parquet` | Normalized morphem nucleocentric profiles |
#
# ## Notes
# - **MAD_robustize**: subtracts the median and divides by the MAD of the reference
#   population, producing a robust z-score that is less sensitive to outliers than
#   standard z-score normalization.
# - **Reference population**: DMSO-treated samples that passed all QC criteria
#   (no `Metadata_cqc_*` flag set to True). QC-flagged rows remain in the output
#   but are excluded from fitting the normalization parameters.
# - Deep-learning profiles (`sammed_*`, `morphem_*`) do not have corresponding QC
#   outputs and are normalized using all DMSO samples as the reference.

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from pycytominer import normalize

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
sc_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/sc_flagged_outliers.parquet"
).resolve(strict=True)
organoid_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/organoid_flagged_outliers.parquet"
).resolve(strict=True)
sc_sammed_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/sammed_sc_flagged_outliers.parquet"
).resolve(strict=True)
organoid_sc_sammed_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/sammed_organoid_flagged_outliers.parquet"
).resolve(strict=True)
nucleocentric_sammed_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/nucleocentric_sammed_flagged_outliers.parquet"
).resolve(strict=True)
nucleocentric_morphem_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/nucleocentric_morphem_flagged_outliers.parquet"
).resolve(strict=True)


# output path
sc_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sc_norm.parquet"
).resolve()
organoid_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/organoid_norm.parquet"
).resolve()
sc_sammed_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sammed_sc_norm.parquet"
).resolve()
organoid_sc_sammed_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sammed_organoid_norm.parquet"
).resolve()
nucleocentric_sammed_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/sammed_nucleocentric_norm.parquet"
).resolve()
nucleocentric_morphem_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/nucleocentric_morphem_norm.parquet"
).resolve()
sc_normalized_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_annotated_profiles = pd.read_parquet(sc_annotated_path)
organoid_annotated_profiles = pd.read_parquet(organoid_annotated_path)
sc_sammed_annotated_profiles = pd.read_parquet(sc_sammed_annotated_path)
organoid_sc_sammed_annotated_profiles = pd.read_parquet(
    organoid_sc_sammed_annotated_path
)
nucleocentric_sammed_annotated_profiles = pd.read_parquet(
    nucleocentric_sammed_annotated_path
)
nucleocentric_morphem_annotated_profiles = pd.read_parquet(
    nucleocentric_morphem_annotated_path
)

print(f"SC profiles loaded. Shape: {sc_annotated_profiles.shape}")
print(f"Organoid profiles loaded. Shape: {organoid_annotated_profiles.shape}")
print(f"SAMMed3D SC profiles loaded. Shape: {sc_sammed_annotated_profiles.shape}")
print(
    f"SAMMed3D organoid profiles loaded. Shape: {organoid_sc_sammed_annotated_profiles.shape}"
)
print(
    f"SAMMed3D nucleocentric profiles loaded. Shape: {nucleocentric_sammed_annotated_profiles.shape}"
)
print(
    f"morphem nucleocentric profiles loaded. Shape: {nucleocentric_morphem_annotated_profiles.shape}"
)


# ## Normalize the profiles
#
# For each profile type, metadata columns (those starting with `Metadata_`) are
# identified and separated from feature columns. Features are then normalized using
# `MAD_robustize` from pycytominer.
#
# For hand-crafted profiles (SC, organoid), the reference population is DMSO-treated
# samples that passed all QC checks — i.e., all `Metadata_cqc_*` flags are `False`.
# This prevents outlier cells from skewing the normalization fit while keeping them
# in the output for downstream filtering decisions.
#
# For deep-learning profiles, the reference is all DMSO-treated samples (no QC
# filter exists for these profiles).

# In[5]:


# Metadata columns start with "Metadata_". Use startswith for precision
# to avoid matching feature columns that happen to contain the word "Metadata".
sc_metadata_cols = [
    col for col in sc_annotated_profiles.columns if col.startswith("Metadata_")
]
sc_sammed_metadata_cols = [
    col for col in sc_sammed_annotated_profiles.columns if "Metadata" in col
]
organoid_metadata_cols = [
    col for col in organoid_annotated_profiles.columns if "Metadata" in col
]
organoid_sc_sammed_metadata_cols = [
    col for col in organoid_sc_sammed_annotated_profiles.columns if "Metadata" in col
]
nucleocentric_sammed_metadata_cols = [
    col for col in nucleocentric_sammed_annotated_profiles.columns if "Metadata" in col
]
nucleocentric_morphem_metadata_cols = [
    col for col in nucleocentric_morphem_annotated_profiles.columns if "Metadata" in col
]

# get the feature columns by excluding the metadata columns
sc_feature_cols = [
    col for col in sc_annotated_profiles.columns if col not in sc_metadata_cols
]
sc_sammed_feature_cols = [
    col
    for col in sc_sammed_annotated_profiles.columns
    if col not in sc_sammed_metadata_cols
]
organoid_feature_cols = [
    col
    for col in organoid_annotated_profiles.columns
    if col not in organoid_metadata_cols
]
organoid_sc_sammed_feature_cols = [
    col
    for col in organoid_sc_sammed_annotated_profiles.columns
    if col not in organoid_sc_sammed_metadata_cols
]
nucleocentric_sammed_feature_cols = [
    col
    for col in nucleocentric_sammed_annotated_profiles.columns
    if col not in nucleocentric_sammed_metadata_cols
]
nucleocentric_morphem_feature_cols = [
    col
    for col in nucleocentric_morphem_annotated_profiles.columns
    if col not in nucleocentric_morphem_metadata_cols
]


def _dmso_qc_samples_query(df):
    """Build a pandas query string for DMSO rows that passed all QC checks."""
    cqc_cols = [col for col in df.columns if col.startswith("Metadata_cqc_")]
    qc_filter = " and ".join(f"`{col}` == False" for col in cqc_cols)
    base = "Metadata_Experiment_Treatment == 'DMSO'"
    return f"{base} and {qc_filter}" if qc_filter else base


# In[6]:


# sc_normalized_df = normalize(
#     profiles=sc_annotated_profiles,
#     features=sc_feature_cols,
#     meta_features=sc_metadata_cols,
#     method="MAD_robustize",
#     samples=_dmso_qc_samples_query(sc_annotated_profiles),
#     output_file=sc_normalized_output_path,
#     output_type="parquet",
# )
# sc_sammed_normalized_df = normalize(
#     profiles=sc_sammed_annotated_profiles,
#     features=sc_sammed_feature_cols,
#     meta_features=sc_sammed_metadata_cols,
#     method="MAD_robustize",
#     samples="Metadata_Experiment_Treatment == 'DMSO'",
#     output_file=sc_sammed_normalized_output_path,
#     output_type="parquet",
# )
# nucleocentric_sammed_normalized_df = normalize(
#     profiles=nucleocentric_sammed_annotated_profiles,
#     features=nucleocentric_sammed_feature_cols,
#     meta_features=nucleocentric_sammed_metadata_cols,
#     method="MAD_robustize",
#     samples="Metadata_Experiment_Treatment == 'DMSO'",
#     output_file=nucleocentric_sammed_normalized_output_path,
#     output_type="parquet",
# )
# nucleocentric_morphem_normalized_df = normalize(
#     profiles=nucleocentric_morphem_annotated_profiles,
#     features=nucleocentric_morphem_feature_cols,
#     meta_features=nucleocentric_morphem_metadata_cols,
#     method="MAD_robustize",
#     samples="Metadata_Experiment_Treatment == 'DMSO'",
#     output_file=nucleocentric_morphem_normalized_output_path,
#     output_type="parquet",
# )

# for organoid normalization
# we will normalize to the whole plate instead of just the DMSO samples,
# since there are fewer organoid samples and thus
# fewer DMSO samples to use for normalization.


# type cast to all float64 so numpy and scipy can properly handle the data during normalization
organoid_annotated_profiles[organoid_feature_cols] = organoid_annotated_profiles[
    organoid_feature_cols
].astype("float64")
organoid_sc_sammed_annotated_profiles[organoid_sc_sammed_feature_cols] = (
    organoid_sc_sammed_annotated_profiles[organoid_sc_sammed_feature_cols].astype(
        "float64"
    )
)
organoid_normalized_df = normalize(
    profiles=organoid_annotated_profiles,
    features=organoid_feature_cols,
    meta_features=organoid_metadata_cols,
    method="MAD_robustize",
    output_file=organoid_normalized_output_path,
    output_type="parquet",
)
organoid_sc_sammed_normalized_df = normalize(
    profiles=organoid_sc_sammed_annotated_profiles,
    features=organoid_sc_sammed_feature_cols,
    meta_features=organoid_sc_sammed_metadata_cols,
    method="MAD_robustize",
    output_file=organoid_sc_sammed_normalized_output_path,
    output_type="parquet",
)


output_df_paths = [
    sc_normalized_output_path,
    organoid_normalized_output_path,
    sc_sammed_normalized_output_path,
    organoid_sc_sammed_normalized_output_path,
    nucleocentric_sammed_normalized_output_path,
    nucleocentric_morphem_normalized_output_path,
]
for output_path in output_df_paths:
    if not output_path.exists():
        print(f"Error: Normalized output file {output_path} was not created.")
