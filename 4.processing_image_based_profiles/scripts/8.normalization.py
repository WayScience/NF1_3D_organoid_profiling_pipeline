#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile normalization.
# All profiles are normalized to the DMSO control treated profiles.

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
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
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


# pathing
sc_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/sc_flagged_outliers.parquet"
).resolve(strict=True)
organoid_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.qc_profiles/organoid_flagged_outliers.parquet"
).resolve(strict=True)
sc_sammed_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/sammed_sc_anno.parquet"
).resolve(strict=True)
organoid_sc_sammed_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/sammed_organoid_anno.parquet"
).resolve(strict=True)
nucleocentric_sammed_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/nucleocentric_sammed_anno.parquet"
).resolve(strict=True)
nucleocentric_chammi_annotated_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/nucleocentric_chammi_anno.parquet"
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
nucleocentric_chammi_normalized_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/chammi_nucleocentric_norm.parquet"
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
nucleocentric_chammi_annotated_profiles = pd.read_parquet(
    nucleocentric_chammi_annotated_path
)


# ### Normalize the profiles

# In[5]:


# get the metadata columns (those that start with "Metadata_")
sc_metadata_cols = [col for col in sc_annotated_profiles.columns if "Metadata" in col]
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
nucleocentric_chammi_metadata_cols = [
    col for col in nucleocentric_chammi_annotated_profiles.columns if "Metadata" in col
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
nucleocentric_chammi_feature_cols = [
    col
    for col in nucleocentric_chammi_annotated_profiles.columns
    if col not in nucleocentric_chammi_metadata_cols
]


# In[6]:


normalization_dict = {
    "sc": {
        "annotated_profiles": sc_annotated_profiles,
        "metadata_cols": sc_metadata_cols,
        "feature_cols": sc_feature_cols,
        "normalized_output_path": sc_normalized_output_path,
    },
    "organoid": {
        "annotated_profiles": organoid_annotated_profiles,
        "metadata_cols": organoid_metadata_cols,
        "feature_cols": organoid_feature_cols,
        "normalized_output_path": organoid_normalized_output_path,
    },
    "sc_sammed": {
        "annotated_profiles": sc_sammed_annotated_profiles,
        "metadata_cols": sc_sammed_metadata_cols,
        "feature_cols": sc_sammed_feature_cols,
        "normalized_output_path": sc_sammed_normalized_output_path,
    },
    "organoid_sc_sammed": {
        "annotated_profiles": organoid_sc_sammed_annotated_profiles,
        "metadata_cols": organoid_sc_sammed_metadata_cols,
        "feature_cols": organoid_sc_sammed_feature_cols,
        "normalized_output_path": organoid_sc_sammed_normalized_output_path,
    },
    "nucleocentric_sammed": {
        "annotated_profiles": nucleocentric_sammed_annotated_profiles,
        "metadata_cols": nucleocentric_sammed_metadata_cols,
        "feature_cols": nucleocentric_sammed_feature_cols,
        "normalized_output_path": nucleocentric_sammed_normalized_output_path,
    },
    "nucleocentric_chammi": {
        "annotated_profiles": nucleocentric_chammi_annotated_profiles,
        "metadata_cols": nucleocentric_chammi_metadata_cols,
        "feature_cols": nucleocentric_chammi_feature_cols,
        "normalized_output_path": nucleocentric_chammi_normalized_output_path,
    },
}


# In[7]:


for profile_type in normalization_dict.keys():
    print(f"Normalizing {profile_type} profiles...")
    try:
        _output_path = normalize(
            profiles=normalization_dict[profile_type]["annotated_profiles"],
            features=normalization_dict[profile_type]["feature_cols"],
            meta_features=normalization_dict[profile_type]["metadata_cols"],
            method="MAD_robustize",
            samples="Metadata_Experiment_Treatment == 'DMSO 1%'",
            output_file=normalization_dict[profile_type]["normalized_output_path"],
            output_type="parquet",
        )
    except Exception as e:
        print(f"Error normalizing {profile_type} profiles: {e}")
        normalization_dict[profile_type]["annotated_profiles"].dropna(inplace=True)
        _output_path = normalize(
            profiles=normalization_dict[profile_type]["annotated_profiles"],
            features=normalization_dict[profile_type]["feature_cols"],
            meta_features=normalization_dict[profile_type]["metadata_cols"],
            method="MAD_robustize",
            samples="Metadata_Experiment_Treatment == 'DMSO 1%'",
            output_file=normalization_dict[profile_type]["normalized_output_path"],
            output_type="parquet",
        )
    if not normalization_dict[profile_type]["normalized_output_path"].exists():
        print(f"Error: Normalized {profile_type} profiles were not saved to disk.")
