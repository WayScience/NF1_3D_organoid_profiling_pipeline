#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib

import duckdb
import pandas as pd
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import aggregate, feature_select

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


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
    "sc": [],
    "organoid": [],
}


# In[4]:


for patient in patients:
    norm_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/3.normalized_profiles"
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
    "Metadata_patient_tumor",
    "Metadata_patient",
    "Metadata_tumor",
    "Metadata_object_id",
    "Metadata_unit",
    "Metadata_dose",
    "Metadata_treatment",
    "Metadata_Target",
    "Metadata_Class",
    "Metadata_Therapeutic_Categories",
    "Metadata_image_set",
    "Metadata_Well",
    "Metadata_parent_organoid",
]
na_cutoff = 0.05
corr_threshold = 0.9
freq_cut = 0.01
unique_cut = 0.01


# In[6]:


for compartment, files in levels_to_merge_dict.items():
    print(f"Found {len(files)} files for {compartment} level.")
    list_of_dfs = []
    for file in files:
        patient_id = str(file.parent).split("/")[-3]
        df = pd.read_parquet(file)
        df["Metadata_patient_tumor"] = patient_id
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
            "Metadata_patient_tumor",
            "Metadata_patient",
            "Metadata_tumor",
            "Metadata_object_id",
            "Metadata_unit",
            "Metadata_dose",
            "Metadata_treatment",
            "Metadata_Target",
            "Metadata_Class",
            "Metadata_Therapeutic_Categories",
            "Metadata_image_set",
            "Metadata_Well",
            "Metadata_parent_organoid",
            "Area.Size.Shape_Cell_CENTER.X",
            "Area.Size.Shape_Cell_CENTER.Y",
            "Area.Size.Shape_Cell_CENTER.Z",
        ]
        # only perform feature selection on DMSO and staurosporine treatments and apply to rest of profiles
        all_trt_df = df.copy()
        df = df.loc[df["Metadata_treatment"].isin(["DMSO", "Staurosporine"])]
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
                "Metadata_patient_tumor",
                "Metadata_patient",
                "Metadata_tumor",
                "Metadata_Well",
                "Metadata_treatment",
                "Metadata_dose",
                "Metadata_unit",
                "Metadata_Target",
                "Metadata_Class",
                "Metadata_Therapeutic_Categories",
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
                "Metadata_patient_tumor",
                "Metadata_patient",
                "Metadata_tumor",
                "Metadata_treatment",
                "Metadata_dose",
                "Metadata_unit",
                "Metadata_Target",
                "Metadata_Class",
                "Metadata_Therapeutic_Categories",
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
            "Metadata_patient_tumor",
            "Metadata_patient",
            "Metadata_tumor",
            "Metadata_object_id",
            "Metadata_unit",
            "Metadata_dose",
            "Metadata_treatment",
            "Metadata_Target",
            "Metadata_Class",
            "Metadata_Therapeutic_Categories",
            "Metadata_image_set",
            "Metadata_Well",
            "Metadata_single_cell_count",
            "Area.Size.Shape_Organoid_CENTER.X",
            "Area.Size.Shape_Organoid_CENTER.Y",
            "Area.Size.Shape_Organoid_CENTER.Z",
        ]
        all_trt_df = df.copy()
        df = df.loc[df["Metadata_treatment"].isin(["DMSO", "Staurosporine"])]
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
                "Metadata_patient_tumor",
                "Metadata_patient",
                "Metadata_tumor",
                "Metadata_Well",
                "Metadata_treatment",
                "Metadata_dose",
                "Metadata_unit",
                "Metadata_Target",
                "Metadata_Class",
                "Metadata_Therapeutic_Categories",
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
                "Metadata_patient_tumor",
                "Metadata_patient",
                "Metadata_tumor",
                "Metadata_Well",
                "Metadata_treatment",
                "Metadata_dose",
                "Metadata_unit",
                "Metadata_Target",
                "Metadata_Class",
                "Metadata_Therapeutic_Categories",
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
