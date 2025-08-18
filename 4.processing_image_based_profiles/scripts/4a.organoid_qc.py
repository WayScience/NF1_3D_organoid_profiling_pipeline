#!/usr/bin/env python
# coding: utf-8

# # Perform organoid-level quality control

# In[1]:


import pandas as pd
import pathlib
from cosmicqc import find_outliers


# ## Load in all the organoid profiles and concat together

# In[2]:


# Path to patient folders
path_to_patients = pathlib.Path(
    "/home/jenna/mnt/bandicoot/NF1_organoid_processed_patients/"
)

# Get all organoid profiles per patient folder and concatenate them
dfs = []
for patient_folder in path_to_patients.iterdir():
    organoid_file = (
        patient_folder / "image_based_profiles/1.combined_profiles" / "organoid.parquet"
    )
    if organoid_file.exists():
        df = pd.read_parquet(organoid_file)
        df["patient_id"] = patient_folder.name
        # Group by image_set and count organoids
        organoid_counts = (
            df.groupby("image_set")["object_id"].count().rename("organoid_count")
        )
        df = df.merge(organoid_counts, on="image_set", how="left")
        dfs.append(df)
orig_organoid_profiles_df = pd.concat(dfs, ignore_index=True)

# Print the shape and head of the combined organoid profiles DataFrame
print(orig_organoid_profiles_df.shape)
orig_organoid_profiles_df.head()


# ## Perform a first round of QC by flagging any row with NaNs in metadata
# 
# We check for NaNs in the `object_id` and/or the `single_cell_count` column and flag them because:
#    - An organoid can not exist if there aren't any cells.
#    - NaN in object_id would be incorrect as that means the object/organoid does not exist (will have all NaNs in the feature space).

# In[3]:


organoid_profiles_df = orig_organoid_profiles_df.copy()
organoid_profiles_df["cqc.nan_detected"] = (
    organoid_profiles_df[["object_id", "single_cell_count"]].isna().any(axis=1)
)

# Print the number of organoids flagged
flagged_count = organoid_profiles_df["cqc.nan_detected"].sum()
print(f"Number of organoids flagged: {flagged_count}")

organoid_profiles_df.head()


# ## Process non-NaN rows to detect abnormally small and large organoids and flag them

# In[4]:


# Set the metadata columns to be used in the QC process
metadata_columns = [
    "patient_id",
    "image_set",
    "object_id",
    "single_cell_count",
    "organoid_count",
    "cqc.nan_detected",
]


# In[5]:


# Process each plate (patient_id) independently in the combined dataframe
for plate_name, plate_df in organoid_profiles_df.groupby("patient_id"):
    print(f"Processing plate: {plate_name}")

    # Only process the rows that are not flagged
    filtered_plate_df = plate_df[~plate_df["cqc.nan_detected"]]

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
    plate_df["cqc.small_organoid_outlier"] = False
    plate_df.loc[small_size_outliers.index, "cqc.small_organoid_outlier"] = True

    print("Finding large organoid outliers...")
    large_size_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Area.Size.Shape_Organoid_VOLUME": 3,  # Detect very large organoids
        },
    )

    # Ensure the column exists before assignment
    plate_df["cqc.large_organoid_outlier"] = False
    plate_df.loc[large_size_outliers.index, "cqc.large_organoid_outlier"] = True

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
    output_folder = (
        path_to_patients / plate_name / "image_based_profiles/1a.qc_profiles"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "organoid_flagged_outliers.parquet"
    plate_df.to_parquet(output_file, index=False)
    print(f"Saved organoid profiles with outlier flags to {output_file}\n")


# In[6]:


# Print example output of the flagged organoid profiles
print(f"Example flagged organoid profiles: {plate_name}")
print(plate_df.shape)
plate_df.head()

