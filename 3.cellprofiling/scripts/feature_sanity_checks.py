#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import itertools
import os
import pathlib
import sys
from functools import reduce

import duckdb
import pandas as pd
import tomli
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir  # default to root_dir instead of NAS


# In[2]:


patient_id_file = pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()


# In[3]:


out_dict = {
    "file_path": [],
    "patient_id": [],
    "well_fov": [],
    "feature_type": [],
    "compartment": [],
    # "df_shape": [],
}

# get all well_fovs for a patient
for patient in tqdm.tqdm(patients, desc="Processing patients", leave=True):
    patient_dir = profile_base_dir / "data" / patient / "extracted_features"
    well_fovs = patient_dir.glob("*")  # get all well_fovs for a patient
    # print(f"Found well_fovs: {well_fovs}")
    for well_fov in tqdm.tqdm(well_fovs, desc="Processing well_fovs", leave=False):
        if "stats" in well_fov.stem:
            continue
        features = pathlib.Path(well_fov).glob("*.parquet")
        for feature in features:
            feature_type = feature.stem.split("_")[2]
            compartment = feature.stem.split("_")[0]
            out_dict["file_path"].append(feature)
            out_dict["patient_id"].append(patient)
            out_dict["well_fov"].append(feature.parent.stem)
            out_dict["feature_type"].append(feature_type)
            out_dict["compartment"].append(compartment)
            # out_dict["df_shape"].append(pd.read_parquet(feature).shape)
df = pd.DataFrame(out_dict)
# df = df.loc[df["patient_id"] == "NF0014_T1"]
df = df.loc[(df["feature_type"] == "Granularity") & (df["compartment"] != "Organoid")]
df = df.loc[(df["compartment"] != "Organoid")]

df.head()


# In[4]:


from tqdm import tqdm

tqdm.pandas()


def safe_read_shape(x):
    try:
        df = pd.read_parquet(x)
        return df.shape, df.isna().sum().sum()
    except Exception as e:
        print(f"Error reading {x}: {e}")
        return None, None


# if not pathlib.Path("../logs/feature_file_info.parquet").exists():
df[["df_shape", "missing_values"]] = df["file_path"].progress_apply(
    lambda x: pd.Series(safe_read_shape(x))
)
df["file_path"] = df["file_path"].astype(str)
df.to_parquet("../logs/feature_file_info.parquet", index=False)
# else:
#     df = pd.read_parquet("../logs/feature_file_info.parquet")


# In[5]:


df.sort_values(["patient_id", "well_fov"], inplace=True)
df.reset_index(drop=True, inplace=True)
df


# In[6]:


# merge the cells, cytoplasm, and whole cell features for a given well_fov and patient_id
# check for missing values and shape of the dataframes
out_dict = {
    "patient_id": [],
    "well_fov": [],
    "path": [],
    "type": [],
    "feature_type": [],
}
for row in tqdm(
    df.itertuples(), total=df.shape[0], desc="Merging features", leave=True
):
    out_dict["patient_id"].append(row.patient_id)
    out_dict["well_fov"].append(row.well_fov)
    out_dict["path"].append(row.file_path)
    out_dict["type"].append(f"{row.compartment}")
    out_dict["feature_type"].append(row.feature_type)
out_df = pd.DataFrame(out_dict)
out_df.drop_duplicates(subset=["patient_id", "well_fov", "type"], inplace=True)
# pivot such that each type has its own column
out_df = out_df.pivot(
    index=[
        "patient_id",
        "well_fov",
    ],
    columns="type",
    values="path",
).reset_index()


# In[7]:


out_df


# In[8]:


labels_dict = {
    "Cell_labels": [],
    "Cytoplasm_labels": [],
    "Nuclei_labels": [],
    "patient_id": [],
    "well_fov": [],
}

# merge the dataframes and check for missing values and shape
for row in tqdm(
    out_df.itertuples(),
    total=out_df.shape[0],
    desc="Checking merged features",
    leave=True,
):
    try:
        cell_df = pd.read_parquet(row.Cell)
        cytoplasm_df = pd.read_parquet(row.Cytoplasm)
        nuclei_df = pd.read_parquet(row.Nuclei)
        labels_dict["Cell_labels"].append(cell_df["object_id"].tolist())
        labels_dict["Cytoplasm_labels"].append(cytoplasm_df["object_id"].tolist())
        labels_dict["Nuclei_labels"].append(nuclei_df["object_id"].tolist())
        labels_dict["patient_id"].append(row.patient_id)
        labels_dict["well_fov"].append(row.well_fov)
    except Exception as e:
        print(f"Error reading files for {row.patient_id} {row.well_fov}: {e}")
labels_df = pd.DataFrame(labels_dict)


# In[9]:


labels_df["labels_match"] = labels_df.apply(
    lambda row: row["Cell_labels"] == row["Nuclei_labels"],
    axis=1,
)
labels_df["same_number_of_labels"] = labels_df.apply(
    lambda row: len(row["Cell_labels"]) == len(row["Nuclei_labels"]),
    axis=1,
)
labels_df["unique_labels_across_compartments"] = labels_df.apply(
    lambda row: set(row["Nuclei_labels"]) - set(row["Cytoplasm_labels"]), axis=1
)
labels_df.loc[labels_df["labels_match"] == False]


# In[10]:


# get the patient_id and well_fov for the rows where the labels do not match and check the corresponding feature files for those rows
mismatched_labels = labels_df.loc[
    labels_df["labels_match"] == False, ["patient_id", "well_fov"]
]
mismatched_labels
for row in tqdm(
    mismatched_labels.itertuples(),
    total=mismatched_labels.shape[0],
    desc="Checking mismatched labels",
    leave=True,
):
    patient_id = row.patient_id
    well_fov = row.well_fov
    print(f"cd ../../{patient_id}/extracted_features/ ; rm -r {well_fov}")


# In[11]:


for patient_id in patients:
    print(
        f'cd ../../{patient_id}/extracted_features/ ; find . -type f -name "*Intensity*" -delete'
    )
    print(
        f'cd ../../{patient_id}/extracted_features/ ; find . -type f -name "*Colocalization*" -delete'
    )
    print(
        f'cd ../../{patient_id}/extracted_features/ ; find . -type f -name "*Granularity*" -delete'
    )
    print(
        f'cd ../../{patient_id}/extracted_features ; find . -type f -name "*Texture*" -delete'
    )
    print(
        f'cd ../../{patient_id}/extracted_features ; find . -type f -name "*Neighbors*" -delete'
    )


# In[12]:


labels_df.loc[labels_df["same_number_of_labels"] == False].value_counts("patient_id")


# In[13]:


# # show the well fov and the patient id
# # print the unique well fovs that have mismatched labels
# for row in labels_df.loc[labels_df["labels_match"] == False][
#     ["patient_id", "well_fov"]
# ].itertuples():
#     print(f"cd ../../{row.patient_id}/extracted_features/ ; rm -r {row.well_fov}")


# In[14]:


tmp_df = pd.merge(
    left=pd.merge(
        left=cell_df,
        right=cytoplasm_df,
        on=["object_id", "image_set"],
    ),
    right=nuclei_df,
    on=["object_id", "image_set"],
)
tmp_df.head()


# In[ ]:
