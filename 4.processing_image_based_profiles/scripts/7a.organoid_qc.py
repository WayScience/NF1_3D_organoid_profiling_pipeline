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

# In[ ]:


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
profile_base_dir = root_dir


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    image_based_profiles_subparent_name = "image_based_profiles"
    patient = "NF0014_T1"


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

output_dir = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles"
)
output_dir.mkdir(parents=True, exist_ok=True)

orig_organoid_profiles_df = pd.read_parquet(organoid_file)

# Print the shape and head of the combined organoid profiles DataFrame
print(orig_organoid_profiles_df.shape)
orig_organoid_profiles_df.head()


# ## Round 1 QC: flag rows with NaN in key columns
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
            "Organoid_NoChannel_AreaSizeShape_Volume",
        ]
    ]
    .isna()
    .any(axis=1)
)
# Print the number of organoids flagged
flagged_count = organoid_profiles_df["Metadata_cqc_nan_detected"].sum()
print(f"Number of organoids flagged: {flagged_count}")

organoid_profiles_df.head()


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
        "Organoid_NoChannel_AreaSizeShape_Volume": -1,  # Detect very small organoids
    },
)

# Ensure the column exists before assignment
organoid_profiles_df["Metadata_cqc_small_organoid_outlier"] = False
organoid_profiles_df.loc[
    small_size_outliers.index, "Metadata_cqc_small_organoid_outlier"
] = True

print("Finding large organoid outliers...")
large_size_outliers = find_outliers(
    df=filtered_profile_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Organoid_NoChannel_AreaSizeShape_Volume": 3,  # Detect very large organoids
    },
)

# Ensure the column exists before assignment
organoid_profiles_df["Metadata_cqc_large_organoid_outlier"] = False
organoid_profiles_df.loc[
    large_size_outliers.index, "Metadata_cqc_large_organoid_outlier"
] = True

# Print number of outliers (only in filtered rows)
small_count = filtered_profile_df.index.intersection(small_size_outliers.index).shape[0]
large_count = filtered_profile_df.index.intersection(large_size_outliers.index).shape[0]
print(f"Small organoid outliers found: {small_count}")
print(f"Large organoid outliers found: {large_count}")

# Save updated plate_df with flag columns included
output_file_path = pathlib.Path(
    f"{output_dir}/organoid_flagged_outliers.parquet"
).resolve()
organoid_profiles_df.to_parquet(output_file_path, index=False)


# In[8]:


# Print example output of the flagged organoid profiles
print(organoid_profiles_df.shape)
organoid_profiles_df.head()

