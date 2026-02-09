#!/usr/bin/env python
# coding: utf-8

# # Perform organoid-level quality control

# In[1]:


import os
import pathlib
import sys

import pandas as pd
from arg_parsing_utils import parse_args
from cosmicqc import find_outliers
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[2]:


if not in_notebook:
    args = parse_args()
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    image_based_profiles_subparent_name = "image_based_profiles"


# ## Load in all the organoid profiles and concat together

# In[ ]:


# Path to patient folders
path_to_patients = pathlib.Path(f"{profile_base_dir}/data/")

# Get all organoid profiles per patient folder and concatenate them
dfs = []
for patient_folder in path_to_patients.iterdir():
    organoid_file = pathlib.Path(
        patient_folder
        / f"{image_based_profiles_subparent_name}"
        / "2.annotated_profiles/organoid_anno.parquet"
    )
    if organoid_file.exists():
        print(f"Processing {organoid_file}")
        df = pd.read_parquet(organoid_file)
        dfs.append(df)
    else:
        print(f"Organoid profiles file not found for patient folder: {patient_folder}")
orig_organoid_profiles_df = pd.concat(dfs, ignore_index=True)

# Print the shape and head of the combined organoid profiles DataFrame
print(orig_organoid_profiles_df.shape)
orig_organoid_profiles_df.head()


# ## Perform a first round of QC by flagging any row with NaNs in metadata
#
# We check for NaNs in the `object_id` and/or the `single_cell_count` column and flag them because:
#    - An organoid can not exist if there aren't any cells.
#    - NaN in object_id would be incorrect as that means the object/organoid does not exist (will have all NaNs in the feature space).

# In[ ]:


organoid_profiles_df = orig_organoid_profiles_df.copy()
organoid_profiles_df["Metadata_cqc_nan_detected"] = (
    organoid_profiles_df[
        [
            "Metadata_object_id",
            "Metadata_single_cell_count",
            "Area.Size.Shape_Organoid_VOLUME",
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

# In[ ]:


# Set the metadata columns to be used in the QC process
metadata_columns = [
    "Metadata_patient_tumor",
    "Metadata_image_set",
    "Metadata_single_cell_count",
    "Metadata_object_id",
    "Metadata_cqc_nan_detected",
]


# In[ ]:


# Process each plate (patient_id) independently in the combined dataframe
for plate_name, plate_df in organoid_profiles_df.groupby("Metadata_patient_tumor"):
    print(f"Processing plate: {plate_name}")

    # Only process the rows that are not flagged
    filtered_plate_df = plate_df[~plate_df["Metadata_cqc_nan_detected"]]

    # Find outlier organoids based on the 'Area.Size.Shape_Organoid_VOLUME' column
    print("Finding small organoid outliers...")
    small_size_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Area.Size.Shape_Organoid_VOLUME": -1,  # Detect very small organoids
        },
    )

    # Ensure the column exists before assignment
    plate_df["Metadata_cqc_small_organoid_outlier"] = False
    plate_df.loc[small_size_outliers.index, "Metadata_cqc_small_organoid_outlier"] = (
        True
    )

    print("Finding large organoid outliers...")
    large_size_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Area.Size.Shape_Organoid_VOLUME": 3,  # Detect very large organoids
        },
    )

    # Ensure the column exists before assignment
    plate_df["Metadata_cqc_large_organoid_outlier"] = False
    plate_df.loc[large_size_outliers.index, "Metadata_cqc_large_organoid_outlier"] = (
        True
    )

    # Update original dataframe so flags persist
    organoid_profiles_df.loc[plate_df.index, :] = plate_df

    # Print number of outliers (only in filtered rows)
    small_count = filtered_plate_df.index.intersection(small_size_outliers.index).shape[
        0
    ]
    large_count = filtered_plate_df.index.intersection(large_size_outliers.index).shape[
        0
    ]
    print(f"Small organoid outliers found: {small_count}")
    print(f"Large organoid outliers found: {large_count}")

    # Save updated plate_df with flag columns included
    output_folder = path_to_patients / plate_name / "image_based_profiles/3.qc_profiles"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "organoid_flagged_outliers.parquet"
    plate_df.to_parquet(output_file, index=False)
    print(f"Saved organoid profiles with outlier flags to {output_file}\n")


# In[11]:


# Print example output of the flagged organoid profiles
print(f"Example flagged organoid profiles: {plate_name}")
print(plate_df.shape)
plate_df.head()
