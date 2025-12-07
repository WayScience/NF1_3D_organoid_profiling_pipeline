#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile feature selection.

# In[1]:


import os
import pathlib

import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import feature_select

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[ ]:


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
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.normalized_profiles/sc_norm.parquet"
).resolve(strict=True)
organoid_normalized_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.normalized_profiles/organoid_norm.parquet"
).resolve(strict=True)


# output path
sc_fs_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.feature_selected_profiles/sc_fs.parquet"
).resolve()
organoid_fs_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/4.feature_selected_profiles/organoid_fs.parquet"
).resolve()

organoid_fs_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# read in the data
sc_normalized = pd.read_parquet(sc_normalized_path)
organoid_normalized = pd.read_parquet(organoid_normalized_path)


# In[5]:


feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    # "correlation_threshold", # comment out to remove correlation thresholding
    # "variance_threshold", # comment out to remove variance thresholding
]


# In[6]:


na_cutoff = 0.05
corr_threshold = 0.95
freq_cut = 0.01
unique_cut = 0.01


# ### Feature select the single-cell profiles

# In[7]:


sc_normalized.head()


# In[8]:


sc_blocklist = [
    x
    for x in sc_normalized.columns
    if "Area" in x and ("MAX" in x or "MIN" in x or "BBOX" in x or "CENTER" in x)
]
sc_blocklist += [
    x
    for x in sc_normalized.columns
    if "Intensity" in x
    and (
        "MIN.X" in x
        or "MAX.X" in x
        or "MIN.Y" in x
        or "MAX.Y" in x
        or "MIN.Z" in x
        or "MAX.Z" in x
    )
]
# write the blocklist to a file
# add "blocklist" the beginning of the list
sc_blocklist = ["blocklist"] + sc_blocklist
sc_blocklist_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/data/blocklist/sc_blocklist.txt"
).resolve()
sc_blocklist_path.parent.mkdir(parents=True, exist_ok=True)
with open(sc_blocklist_path, "w") as f:
    for item in sc_blocklist:
        f.write(f"{item}\n")


# In[9]:


sc_metadata_columns = [x for x in sc_normalized.columns if "Metadata" in x]

sc_metadata_columns += [
    "Area.Size.Shape_Cell_CENTER.X",
    "Area.Size.Shape_Cell_CENTER.Y",
    "Area.Size.Shape_Cell_CENTER.Z",
]
sc_features_columns = [
    col for col in sc_normalized.columns if col not in sc_metadata_columns
]
all_trt_df = sc_normalized.copy()

sc_normalized = sc_normalized.loc[
    sc_normalized["Metadata_treatment"].isin(["DMSO", "Staurosporine"])
]

sc_features_columns = [
    col for col in sc_normalized.columns if col not in sc_metadata_columns
]
sc_features_df = sc_normalized.drop(columns=sc_metadata_columns, errors="ignore")


# In[10]:


# fs the data
sc_fs_profiles = feature_select(
    sc_features_df,
    operation=feature_select_ops,
    features=sc_features_columns,
    blocklist_file=sc_blocklist_path,
    na_cutoff=na_cutoff,
    corr_threshold=corr_threshold,
    freq_cut=freq_cut,
    unique_cut=unique_cut,
)
fs_profiles = all_trt_df[
    [col for col in all_trt_df.columns if col in sc_fs_profiles.columns]
]

original_data_shape = sc_normalized.shape
sc_fs_profiles = pd.concat(
    [
        all_trt_df[sc_metadata_columns].reset_index(drop=True),
        sc_fs_profiles.reset_index(drop=True),
    ],
    axis=1,
)
print("The number features before feature selection:", original_data_shape[1])
print("The number features after feature selection:", sc_fs_profiles.shape[1])
sc_fs_profiles.to_parquet(sc_fs_output_path, index=False)
sc_fs_profiles.head()


# ### Normalize the organoid profiles

# In[11]:


organoid_normalized.head()


# In[12]:


organoid_blocklist = [
    x
    for x in organoid_normalized.columns
    if "Area" in x and ("MAX" in x or "MIN" in x or "BBOX" in x or "CENTER" in x)
]
organoid_blocklist += [
    x
    for x in organoid_normalized.columns
    if "Intensity" in x
    and (
        "MIN.X" in x
        or "MAX.X" in x
        or "MIN.Y" in x
        or "MAX.Y" in x
        or "MIN.Z" in x
        or "MAX.Z" in x
    )
]
# write the blocklist to a file
# add "blocklist" the beginning of the list
organoid_blocklist = ["blocklist"] + organoid_blocklist
organoid_blocklist_path = pathlib.Path(
    "../data/blocklist/organoid_blocklist.txt"
).resolve()
organoid_blocklist_path.parent.mkdir(parents=True, exist_ok=True)
with open(organoid_blocklist_path, "w") as f:
    for item in organoid_blocklist:
        f.write(f"{item}\n")


# In[13]:


organoid_metadata_columns = [x for x in organoid_normalized.columns if "Metadata" in x]
organoid_metadata_columns += [
    "Area.Size.Shape_Organoid_CENTER.X",
    "Area.Size.Shape_Organoid_CENTER.Y",
    "Area.Size.Shape_Organoid_CENTER.Z",
]
organoid_features_columns = [
    col for col in organoid_normalized.columns if col not in organoid_metadata_columns
]
all_trt_df = organoid_normalized.copy()
organoid_normalized = organoid_normalized.loc[
    organoid_normalized["Metadata_treatment"].isin(["DMSO", "Staurosporine"])
]
organoid_features_columns = [
    col for col in organoid_normalized.columns if col not in organoid_metadata_columns
]
organoid_features_df = organoid_normalized.drop(
    columns=organoid_metadata_columns, errors="ignore"
)


# In[14]:


# normalize the data
organoid_fs_profiles = feature_select(
    organoid_features_df,
    operation=feature_select_ops,
    features=organoid_features_columns,
    blocklist_file=organoid_blocklist_path,
    na_cutoff=na_cutoff,
    corr_threshold=corr_threshold,
    freq_cut=freq_cut,
    unique_cut=unique_cut,
)
# apply feature selection to all profiles
organoid_fs_profiles = all_trt_df[
    [col for col in all_trt_df.columns if col in organoid_fs_profiles.columns]
]
# concatenate the metadata and the feature selected profiles
original_data_shape = organoid_normalized.shape
organoid_fs_profiles = pd.concat(
    [
        all_trt_df[organoid_metadata_columns].reset_index(drop=True),
        organoid_fs_profiles.reset_index(drop=True),
    ],
    axis=1,
)

print("The number features before feature selection:", original_data_shape[1])
print("The number features after feature selection:", organoid_fs_profiles.shape[1])
organoid_fs_profiles.to_parquet(organoid_fs_output_path, index=False)
organoid_fs_profiles.head()
