#!/usr/bin/env python
# coding: utf-8

# # Perform single-cell level quality control

# In[1]:


import pandas as pd
import pathlib
from cosmicqc import find_outliers


# ## Load in each single-cell level profile per patient and process
# 
# 1. Load in the single-cell data (add `patient_id` column).
# 2. Load in respective organoid qc data (only metadata and cqc columns) to already flag cells that come from a flagged organoid.
#    - Also add a flag for if single-cells do not have an organoid segmentation (`parent_organoid` == -1).
#    - Also add flag for if the `object_id` for a single-cell is NaN.
# 3. Concat single-cell data together.

# In[2]:


# Path to patient folders
path_to_patients = pathlib.Path(
    "/home/jenna/mnt/bandicoot/NF1_organoid_processed_patients/"
)

dfs = []
for patient_folder in path_to_patients.iterdir():
    single_cell_file = (
        patient_folder / "image_based_profiles/1.combined_profiles" / "sc.parquet"
    )
    organoid_flags_file = (
        patient_folder
        / "image_based_profiles/1a.qc_profiles"
        / "organoid_flagged_outliers.parquet"
    )

    if single_cell_file.exists():
        sc_df = pd.read_parquet(single_cell_file)
        sc_df["patient_id"] = patient_folder.name

        # Default QC flags
        sc_df["cqc.organoid_flagged"] = False
        sc_df["cqc.nan_detected"] = sc_df["object_id"].isna()
        sc_df["cqc.missing_organoid"] = sc_df["parent_organoid"] == -1

        if organoid_flags_file.exists():
            organoid_flags_df = pd.read_parquet(organoid_flags_file)[
                ["object_id", "image_set"]
                + [
                    col
                    for col in pd.read_parquet(organoid_flags_file).columns
                    if col.startswith("cqc")
                ]
            ]

            # Get flagged (object_id, image_set) pairs
            flagged_pairs = set(
                organoid_flags_df.loc[
                    organoid_flags_df.filter(like="cqc").any(axis=1),
                    ["object_id", "image_set"],
                ].itertuples(index=False, name=None)
            )

            # Flag SC rows where both parent_organoid & image_set match a flagged organoid
            sc_df["cqc.organoid_flagged"] = sc_df.apply(
                lambda row: (row["parent_organoid"], row["image_set"]) in flagged_pairs,
                axis=1,
            )

        dfs.append(sc_df)

orig_single_cell_profiles_df = pd.concat(dfs, ignore_index=True)

print(orig_single_cell_profiles_df.shape)
orig_single_cell_profiles_df.head()


# ## Detect outlier single-cells using the non-flagged data
# 
# We will attempt to detect instances of poor quality segmentations using the nuclei compartment as the base. The conditions we are using are as follows:
# 
# 1. Abnormally small or large nuclei using `Volume`
# 2. Abnormally high `mass displacement` in the nuclei for instances of mis-segmentation of background/no longer in-focus

# In[3]:


# Set the metadata columns to be used in the QC process
metadata_columns = [
    "patient_id",
    "image_set",
    "object_id",
    "parent_organoid",
    "Area.Size.Shape_Nuclei_CENTER.X",
    "Area.Size.Shape_Nuclei_CENTER.Y",
    "cqc.nan_detected",
    "cqc.organoid_flagged",
    "cqc.missing_organoid",
]


# In[4]:


# Process each plate (patient_id) independently in the combined dataframe
for plate_name, plate_df in orig_single_cell_profiles_df.groupby("patient_id"):
    print(f"Processing plate: {plate_name}")

    # Make a contiguous copy to prevent DataFrame fragmentation
    plate_df = plate_df.copy()

    # Only process the rows that are not flagged
    filtered_plate_df = plate_df[
        ~(
            plate_df["cqc.nan_detected"]
            | plate_df["cqc.organoid_flagged"]
            | plate_df["cqc.missing_organoid"]
        )
    ]

    # --- Find size based nuclei outliers ---
    print("Finding small nuclei outliers...")
    small_nuclei_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Area.Size.Shape_Nuclei_VOLUME": -1,  # Detect very small nuclei
        },
    )

    # Ensure the column exists before assignment
    plate_df["cqc.small_nuclei_outlier"] = False
    plate_df.loc[small_nuclei_outliers.index, "cqc.small_nuclei_outlier"] = True

    print("Finding large nuclei outliers...")
    large_nuclei_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Area.Size.Shape_Nuclei_VOLUME": 2,  # Detect very large nuclei
        },
    )

    # Ensure the column exists before assignment
    plate_df["cqc.large_nuclei_outlier"] = False
    plate_df.loc[large_nuclei_outliers.index, "cqc.large_nuclei_outlier"] = True

    # --- Find mass displacement based nuclei outliers ---
    print("Finding high mass displacement outliers...")
    high_mass_displacement_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Intensity_Nuclei_DNA_MASS.DISPLACEMENT": 2,  # Detect high mass displacement
        },
    )

    # Ensure the column exists before assignment
    plate_df["cqc.mass_displacement_outlier"] = False
    plate_df.loc[
        high_mass_displacement_outliers.index, "cqc.mass_displacement_outlier"
    ] = True

    # Print number of outliers (only in filtered rows)
    small_count = filtered_plate_df.index.intersection(
        small_nuclei_outliers.index
    ).shape[0]
    large_count = filtered_plate_df.index.intersection(
        large_nuclei_outliers.index
    ).shape[0]
    high_mass_count = filtered_plate_df.index.intersection(
        high_mass_displacement_outliers.index
    ).shape[0]

    print(f"Small nuclei outliers found: {small_count}")
    print(f"Large nuclei outliers found: {large_count}")
    print(f"High mass displacement outliers found: {high_mass_count}")

    # Save updated plate_df with flag columns included
    output_folder = (
        path_to_patients / plate_name / "image_based_profiles/1a.qc_profiles"
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "sc_flagged_outliers.parquet"
    plate_df.to_parquet(output_file, index=False)
    print(f"Saved single-cell profiles with outlier flags to {output_file}\n")


# In[5]:


# Print example output of the flagged single-cell profiles
print(f"Example flagged single-cell profiles: {plate_name}")
print(plate_df.shape)
plate_df.head()

