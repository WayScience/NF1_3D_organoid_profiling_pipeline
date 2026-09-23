#!/usr/bin/env python
# coding: utf-8

# # 7a. Organoid QC
#
# ## Purpose
# Flag low-quality organoids per patient using two criteria applied in sequence:
# 1. **NaN detection** — organoids missing key metadata or feature values
# 2. **Size outliers** — abnormally small or large organoids by volume (z-score)
#
# This is **step 7a of Stage 4 (image-based profiling)**. It runs once per patient
# and must complete before `7b.single_cell_qc.ipynb`, which inherits organoid flags.
#
# ## Inputs
# - `data/{patient}/image_based_profiles/3.annotated_profiles/organoid_anno.parquet`
#
# ## Outputs
# - `data/{patient}/image_based_profiles/4.qc_profiles/organoid_flagged_outliers.parquet`
#   — original organoid profile with three added `Metadata_cqc_*` flag columns
#
# ## Notes
# - QC flags are **additive**: an organoid can be flagged by multiple criteria simultaneously.
# - Outlier detection only runs on the subset of organoids that passed the NaN check,
#   so NaN rows are never evaluated for size outliers.

# In[1]:


import json
import os
import pathlib

import pandas as pd
from cosmicqc import find_outliers
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
# profile_base_dir = root_dir
print(profile_base_dir)


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    image_based_profiles_subparent_name = (
        "image_based_profiles_production_zedprofiler_v20260915_f5a6f16"
    )
    patient = "SARCO361_T1"


# In[3]:


# Per-patient small-organoid outlier z-score thresholds. Patients are tuned individually
# by visually inspecting flagged organoids and adjusting their entry in this file.
small_outlier_thresholds_path = (
    root_dir
    / "4.processing_image_based_profiles"
    / "data"
    / "qc_thresholds"
    / "organoid_small_outlier_thresholds.json"
).resolve(strict=True)
with open(small_outlier_thresholds_path) as f:
    small_outlier_thresholds = json.load(f)

if patient not in small_outlier_thresholds:
    raise ValueError(
        f"No small organoid outlier threshold configured for patient '{patient}' in "
        f"{small_outlier_thresholds_path}. Add an entry for this patient before running QC."
    )

small_outlier_threshold = small_outlier_thresholds[patient]
print(
    f"Using small organoid outlier threshold for {patient}: {small_outlier_threshold}"
)


# ## Load in all the organoid profiles and concat together

# In[4]:


organoid_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles"
    / "organoid_anno.parquet"
).resolve(strict=True)

sammed_annotated_organoid_profiles_path = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles"
    / "sammed_organoid_anno.parquet"
).resolve()

qc_output_dir = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles"
)
qc_output_dir.mkdir(parents=True, exist_ok=True)

organoid_qc_output_path = f"{qc_output_dir}/organoid_flagged_outliers.parquet"
sammed_organoid_qc_output_path = (
    f"{qc_output_dir}/sammed_organoid_flagged_outliers.parquet"
)

orig_organoid_profiles_df = pd.read_parquet(organoid_file)

# Print the shape and head of the combined organoid profiles DataFrame
print(orig_organoid_profiles_df.shape)
orig_organoid_profiles_df.head()


# ## (Sanity check) Round 1 QC: flag rows with NaN in key columns
#
# `Metadata_cqc_*` columns are boolean flags added by this notebook. A value of `True`
# means the organoid failed that criterion. Multiple flags can be True simultaneously.
#
# We flag organoids where `ObjectID`, `SingleCellCount`, or `Volume` is NaN because:
# - An organoid with no cells (`SingleCellCount` NaN) cannot be a valid profile row.
# - A NaN `ObjectID` means the object does not exist and all features will be NaN.
# - A NaN `Volume` means the core morphology feature is missing.

# In[5]:


organoid_profiles_df = orig_organoid_profiles_df.copy()
organoid_profiles_df["Metadata_cqc_nan_detected"] = (
    organoid_profiles_df[
        [
            "Metadata_Object_ObjectID",
            "Metadata_Object_OrganoidSingleCellCount",
            "Organoid_NoChannel_VolumeSizeShape_Volume",
        ]
    ]
    .isna()
    .any(axis=1)
)
# Print the number of organoids flagged
flagged_count = organoid_profiles_df["Metadata_cqc_nan_detected"].sum()
print(f"Number of organoids flagged: {flagged_count}")


# ## Process non-NaN rows to detect abnormally small and large organoids and flag them

# In[6]:


# Set the metadata columns to be used in the QC process
metadata_columns = [x for x in organoid_profiles_df.columns if "Metadata" in x]


# In[7]:


## Round 2 QC: size-based outlier detection

# `find_outliers` uses z-score thresholds: negative values flag objects below the mean,
# positive values flag objects above. Threshold magnitude is the number of standard
# deviations from the mean. Only non-NaN rows (from Round 1) are evaluated.

# Only process the rows that are not flagged
filtered_profile_df = organoid_profiles_df[
    ~organoid_profiles_df["Metadata_cqc_nan_detected"]
]

# Find outlier organoids based on the 'Area.Size.Shape_Organoid_VOLUME' column
print("Finding small organoid outliers...")
small_size_outliers = find_outliers(
    df=filtered_profile_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Organoid_NoChannel_VolumeSizeShape_Volume": small_outlier_threshold,
    },
)
# Ensure the column exists before assignment
organoid_profiles_df["Metadata_cqc_small_organoid_outlier"] = False
organoid_profiles_df.loc[
    small_size_outliers.index, "Metadata_cqc_small_organoid_outlier"
] = True

# Print number of outliers (only in filtered rows)
small_count = filtered_profile_df.index.intersection(small_size_outliers.index).shape[0]
print(f"Small organoid outliers found: {small_count}")

display(
    small_size_outliers[
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Well",
            "Metadata_Imaging_FieldID",
            "Organoid_NoChannel_VolumeSizeShape_Volume",
            "Metadata_Object_ObjectID",
        ]
    ]
    .sort_values("Organoid_NoChannel_VolumeSizeShape_Volume", ascending=False)
    .head()
)

organoid_profiles_df.to_parquet(organoid_qc_output_path, index=False)


# In[8]:


# Print example output of the flagged organoid profiles
print(organoid_profiles_df.shape)
organoid_profiles_df.head()


# ## Merge the qc flags to the deep learning-based profiles and save the output
# Merge the QC flags back to the original organoid profiles, which will be used in downstream analyses and single cell QC.
# We need to do this beacuase we do not run qc on black-box features.
# Merge on the Metadata_Biology_PatientTumor, Metadata_Experiment_WellFOV
# and the Metadata_Object_ObjectID columns, which together uniquely identify each organoid profile row.

# In[9]:


sammed_organoid_df = pd.read_parquet(sammed_annotated_organoid_profiles_path)
original_sammed_shape = sammed_organoid_df.shape
# set the merge keys to int for both dataframes to ensure they match
merge_keys = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_WellFOV",
    "Metadata_Object_ObjectID",
]
qc_keys = [col for col in organoid_profiles_df.columns if "Metadata_cqc" in col]


# merge the flagged organoid profiles with the sammed annotated organoid profiles
qc_annotated_sammed_organoid_df = sammed_organoid_df.merge(
    organoid_profiles_df[qc_keys + merge_keys],
    on=merge_keys,
    how="left",
)
if qc_annotated_sammed_organoid_df.shape[1] == original_sammed_shape[1]:
    raise ValueError(
        f"No new columns were added during the merge. Check that the merge keys {merge_keys} are correct and that the qc keys {qc_keys} are present in the organoid_profiles_df."
    )
qc_annotated_sammed_organoid_df.to_parquet(sammed_organoid_qc_output_path, index=False)
qc_annotated_sammed_organoid_df.head()
