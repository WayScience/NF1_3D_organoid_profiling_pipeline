#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile feature selection.

# In[ ]:


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


# pathing
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
nucleocentric_chammi_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/5.normalized_profiles/chammi_nucleocentric_norm.parquet"
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
nucleocentric_chammi_feature_selected_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/6.feature_selected_profiles/chammi_nucleocentric_fs.parquet"
).resolve()

organoid_fs_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_normalized = pd.read_parquet(sc_normalized_path)
organoid_normalized = pd.read_parquet(organoid_normalized_path)
sc_sammed_normalized = pd.read_parquet(sc_sammed_normalized_path)
organoid_sc_sammed_normalized = pd.read_parquet(organoid_sc_sammed_normalized_path)
nucleocentric_sammed_normalized = pd.read_parquet(nucleocentric_sammed_normalized_path)
nucleocentric_chammi_normalized = pd.read_parquet(nucleocentric_chammi_normalized_path)


# In[5]:


run_dict = {
    "normalized": {
        "df": sc_normalized,
        "output_path": sc_fs_output_path,
    },
    "feature_selected_output_path": {
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
        "df": nucleocentric_chammi_normalized,
        "output_path": nucleocentric_chammi_feature_selected_output_path,
    },
}


# In[6]:


feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    "correlation_threshold",  # comment out to remove correlation thresholding
    "variance_threshold",  # comment out to remove variance thresholding
]
na_cutoff = 0.05
corr_threshold = 0.95
freq_cut = 0.01
unique_cut = 0.01


# ### Feature select the profiles

# In[7]:


for profile_name in run_dict.keys():
    print(f"Running feature selection for {profile_name} profiles...")
    df = run_dict[profile_name]["df"]
    output_path = run_dict[profile_name]["output_path"]
    ###################################################
    # prep profiles for feature selection
    ###################################################
    # grab metadata columns
    metadata_columns = [x for x in df.columns if "Metadata" in x]
    # grab feature columns
    features_columns = [col for col in df.columns if col not in metadata_columns]
    # retain all treatments
    all_trt_df = df.copy()
    # get treatments to process - for now, just DMSO and Staurosporine
    df = df.loc[
        df["Metadata_Experiment_Treatment"].isin(["DMSO 1%", "Staurosporine 10 nM"])
    ]

    ###################################################
    # run feature selection
    ###################################################
    fs_profiles = feature_select(
        df,
        operation=feature_select_ops,
        features=features_columns,
        na_cutoff=na_cutoff,
        corr_threshold=corr_threshold,
        freq_cut=freq_cut,
        unique_cut=unique_cut,
    )
    ###################################################
    # subset the original profiles to the features that were retained after feature selection
    ###################################################
    fs_profiles = all_trt_df[
        [col for col in all_trt_df.columns if col in fs_profiles.columns]
    ]

    original_data_shape = df.shape
    print("The number features before feature selection:", original_data_shape[1])
    print("The number features after feature selection:", fs_profiles.shape[1])
    fs_profiles.to_parquet(output_path, index=False)


# In[8]:


fs_profiles.head()
