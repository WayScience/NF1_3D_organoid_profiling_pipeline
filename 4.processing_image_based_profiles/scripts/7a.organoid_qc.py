#!/usr/bin/env python
# coding: utf-8

# # Perform organoid-level quality control

# In[1]:


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


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    image_based_profiles_subparent_name = "image_based_profiles"
    patient = "NF0014_T1"


# ## Load in all the organoid profiles and concat together

# In[3]:


organoid_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles/organoid_anno.parquet"
)

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


# ## Perform a first round of QC by flagging any row with NaNs in metadata
#
# We check for NaNs in the `object_id` and/or the `single_cell_count` column and flag them because:
#    - An organoid can not exist if there aren't any cells.
#    - NaN in object_id would be incorrect as that means the object/organoid does not exist (will have all NaNs in the feature space).

# In[4]:


organoid_profiles_df = orig_organoid_profiles_df.copy()
organoid_profiles_df["Metadata_cqc_nan_detected"] = (
    organoid_profiles_df[
        [
            "Metadata_Object_ObjectID",
            "Metadata_Object_SingleCellCount",
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

# In[5]:


# Set the metadata columns to be used in the QC process
metadata_columns = [x for x in organoid_profiles_df.columns if "Metadata" in x]


# In[6]:


# Process each plate (patient_id) independently in the combined dataframe

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

# Update original dataframe so flags persist
organoid_profiles_df.loc[
    small_size_outliers.index, "Metadata_cqc_small_organoid_outlier"
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


# In[7]:


# Print example output of the flagged organoid profiles
print(organoid_profiles_df.shape)
organoid_profiles_df.head()
