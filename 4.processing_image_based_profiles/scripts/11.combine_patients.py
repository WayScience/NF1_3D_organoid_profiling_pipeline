#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import duckdb
import pandas as pd
from pycytominer import aggregate, feature_select

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


patient_ids_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(patient_ids_path, header=None, names=["patient_id"], dtype=str)[
    "patient_id"
].to_list()

all_patients_output_path = pathlib.Path(
    f"{root_dir}/data/all_patient_profiles"
).resolve()
all_patients_output_path.mkdir(parents=True, exist_ok=True)


# In[3]:


levels_to_merge_dict = {
    "sc": [],
    "organoid": [],
}


# In[4]:


for patient in patients:
    norm_path = pathlib.Path(
        f"{root_dir}/data/{patient}/image_based_profiles/3.normalized_profiles"
    )
    for file in norm_path.glob("*.parquet"):
        if "sc" in file.name:
            levels_to_merge_dict["sc"].append(file)
        elif "organoid" in file.name:
            levels_to_merge_dict["organoid"].append(file)


# In[ ]:


feature_select_ops = [
    "drop_na_columns",
    "blocklist",
    # "variance_threshold", # comment out to remove variance thresholding
    # "correlation_threshold", # comment out to remove correlation thresholding
]
metadata_cols = [
    "patient",
    "object_id",
    "unit",
    "dose",
    "treatment",
    "Target",
    "Class",
    "Therapeutic Categories",
    "image_set",
    "Well",
    "parent_organoid",
]
na_cutoff = 0.05
corr_threshold = 0.9
freq_cut = 0.01
unique_cut = 0.01


# In[ ]:


for compartment, files in levels_to_merge_dict.items():
    print(f"Found {len(files)} files for {compartment} level.")
    list_of_dfs = []
    for file in files:
        patient_id = str(file.parent).split("/")[-3]
        df = pd.read_parquet(file)
        df["patient"] = patient_id
        list_of_dfs.append(df)
    df = pd.concat(list_of_dfs, ignore_index=True)

    print(f"Concatenated DataFrame for {compartment} has the shape: {df.shape}")
    df.to_parquet(
        f"{all_patients_output_path}/{compartment}_profiles.parquet",
        index=False,
    )
    if compartment == "sc":
        blocklist_path = pathlib.Path(
            f"{root_dir}/4.processing_image_based_profiles/data/blocklist/sc_blocklist.txt"
        )
        metadata_cols = [
            "patient",
            "object_id",
            "unit",
            "dose",
            "treatment",
            "Target",
            "Class",
            "Therapeutic Categories",
            "image_set",
            "Well",
            "parent_organoid",
            "Area.Size.Shape_Cell_CENTER.X",
            "Area.Size.Shape_Cell_CENTER.Y",
            "Area.Size.Shape_Cell_CENTER.Z",
        ]
        # only perform feature selection on DMSO and staurosporine treatments and apply to rest of profiles
        all_trt_df = df.copy()
        df = df.loc[df["treatment"].isin(["DMSO", "Staurosporine"])]
        # feature selection
        feature_columns = [col for col in df.columns if col not in metadata_cols]
        features_df = df[feature_columns]
        fs_profiles = feature_select(
            features_df,
            operation=feature_select_ops,
            features=feature_columns,
            blocklist_file=blocklist_path,
            na_cutoff=na_cutoff,
            # corr_threshold=corr_threshold, # comment out to use default value
            # freq_cut=freq_cut, # comment out to use default value
            # unique_cut=unique_cut, # comment out to use default value
        )
        original_data_shape = features_df.shape
        # apply feature selection to all profiles
        fs_profiles = all_trt_df[
            [col for col in all_trt_df.columns if col in fs_profiles.columns]
        ]
        fs_profiles = pd.concat(
            [
                all_trt_df[metadata_cols].reset_index(drop=True),
                fs_profiles.reset_index(drop=True),
            ],
            axis=1,
        )
        fs_profiles.to_parquet(
            f"{all_patients_output_path}/sc_fs_profiles.parquet",
            index=False,
        )
        feature_columns = [
            col for col in fs_profiles.columns if col not in metadata_cols
        ]
        features_df = fs_profiles[feature_columns]
        # aggregate the profiles
        sc_agg_df = aggregate(
            population_df=fs_profiles,
            strata=[
                "patient",
                "Well",
                "treatment",
                "dose",
                "unit",
                "Target",
                "Class",
                "Therapeutic Categories",
            ],
            features=feature_columns,
            operation="median",
        )
        sc_agg_df.to_parquet(
            f"{all_patients_output_path}/sc_agg_profiles.parquet",
            index=False,
        )
        # consensus profiles
        sc_consensus_df = aggregate(
            population_df=fs_profiles,
            strata=[
                "patient",
                "treatment",
                "dose",
                "unit",
                "Target",
                "Class",
                "Therapeutic Categories",
            ],
            features=feature_columns,
            operation="median",
        )
        sc_consensus_df.to_parquet(
            f"{all_patients_output_path}/sc_consensus_profiles.parquet",
            index=False,
        )
        print("The number features before feature selection:", original_data_shape[1])
        print("The number features after feature selection:", fs_profiles.shape[1])

    elif compartment == "organoid":
        blocklist_path = pathlib.Path(
            f"{root_dir}/4.processing_image_based_profiles/data/blocklist/organoid_blocklist.txt"
        )
        metadata_cols = [
            "patient",
            "object_id",
            "unit",
            "dose",
            "treatment",
            "Target",
            "Class",
            "Therapeutic Categories",
            "image_set",
            "Well",
            "single_cell_count",
            "Area.Size.Shape_Organoid_CENTER.X",
            "Area.Size.Shape_Organoid_CENTER.Y",
            "Area.Size.Shape_Organoid_CENTER.Z",
        ]
        all_trt_df = df.copy()
        df = df.loc[df["treatment"].isin(["DMSO", "Staurosporine"])]
        feature_columns = [col for col in df.columns if col not in metadata_cols]
        features_df = df[feature_columns]
        fs_profiles = feature_select(
            features_df,
            operation=feature_select_ops,
            features=feature_columns,
            blocklist_file=blocklist_path,
            na_cutoff=na_cutoff,
            corr_threshold=corr_threshold,
            freq_cut=freq_cut,
            unique_cut=unique_cut,
        )
        fs_profiles = all_trt_df[
            [col for col in all_trt_df.columns if col in fs_profiles.columns]
        ]
        original_data_shape = features_df.shape
        fs_profiles = pd.concat(
            [
                all_trt_df[metadata_cols].reset_index(drop=True),
                fs_profiles.reset_index(drop=True),
            ],
            axis=1,
        )
        fs_profiles.to_parquet(
            f"{all_patients_output_path}/organoid_fs_profiles.parquet",
            index=False,
        )
        feature_columns = [
            col for col in fs_profiles.columns if col not in metadata_cols
        ]
        features_df = fs_profiles[feature_columns]
        # aggregate the profiles
        agg_df = aggregate(
            population_df=fs_profiles,
            strata=[
                "patient",
                "Well",
                "treatment",
                "dose",
                "unit",
                "Target",
                "Class",
                "Therapeutic Categories",
            ],
            features=feature_columns,
            operation="median",
        )
        agg_df.to_parquet(
            f"{all_patients_output_path}/organoid_agg_profiles.parquet",
            index=False,
        )
        # consensus profiles
        consensus_df = aggregate(
            population_df=fs_profiles,
            strata=[
                "patient",
                "treatment",
                "dose",
                "unit",
                "Target",
                "Class",
                "Therapeutic Categories",
            ],
            features=feature_columns,
            operation="median",
        )
        consensus_df.to_parquet(
            f"{all_patients_output_path}/organoid_consensus_profiles.parquet",
            index=False,
        )

        print("The number features before feature selection:", original_data_shape[1])
        print("The number features after feature selection:", fs_profiles.shape[1])
