#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import os
import pathlib

import numpy as np
import pandas as pd
from arg_parsing_utils import check_for_missing_args, parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)

from file_checking import check_number_of_files

# In[2]:


patients_file_path = pathlib.Path("../../data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(patients_file_path, header=None)[0].tolist()
patients


# In[3]:


features_to_check_for = {
    "patient": [],
    "well_fov": [],
    "file_name": [],
    "exists": [],
    "file_count": [],
    "file_path": [],
}
for patient in patients:
    extracted_features_dir = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/extracted_features/"
    ).resolve(strict=True)
    # get all of the well_fov directories
    well_fov_dirs = [d for d in extracted_features_dir.iterdir() if d.is_dir()]
    well_fovs = [d.name for d in well_fov_dirs if "run_stats" not in d.name]
    print(f"Patient {patient} has {len(well_fovs)} well_fovs to check.")
    for well_fov in well_fovs:
        converted_profile_dir = pathlib.Path(
            f"{profile_base_dir}/data/{patient}/image_based_profiles/0.converted_profiles/{well_fov}/{well_fov}.duckdb"
        ).resolve()
        features_to_check_for["patient"].append(patient)
        features_to_check_for["well_fov"].append(well_fov)
        features_to_check_for["file_path"].append(str(converted_profile_dir))
        features_to_check_for["file_name"].append(converted_profile_dir.name)
        features_to_check_for["exists"].append(converted_profile_dir.exists())
        features_to_check_for["file_count"].append(1)


features_to_check_for_df = pd.DataFrame(features_to_check_for)
# print the number of total, present, and missing files
total_files = len(features_to_check_for_df)
present_files = features_to_check_for_df["exists"].sum()
missing_files = total_files - present_files
print(f"Total files to check: {total_files}")
print(f"Present files: {present_files}")
print(f"Missing files: {missing_files}")
features_to_check_for_df.head()


# In[4]:


load_file_path = pathlib.Path("../load_data/load_file.txt").resolve()
load_file_path.parent.mkdir(parents=True, exist_ok=True)
with open(load_file_path, "w") as f:
    for idx, row in features_to_check_for_df.iterrows():
        if not row["exists"]:
            f.write(f"{row['patient']}\t{row['well_fov']}\n")


# In[5]:


grouped_df = (
    features_to_check_for_df.groupby(["patient"])[["exists", "file_count"]]
    .sum()
    .reset_index()
)
grouped_df["total_missing"] = grouped_df["file_count"] - grouped_df["exists"]
grouped_df
