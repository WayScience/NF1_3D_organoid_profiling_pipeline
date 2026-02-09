#!/usr/bin/env python
# coding: utf-8

# # Perform single-cell level quality control

# In[ ]:


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


# In[ ]:


if not in_notebook:
    args = parse_args()
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    image_based_profiles_subparent_name = "image_based_profiles"


# ## Load in each single-cell level profile per patient and process
#
# 1. Load in the single-cell data (add `patient_id` column).
# 2. Load in respective organoid qc data (only metadata and cqc columns) to already flag cells that come from a flagged organoid.
#    - Also add a flag for if single-cells do not have an organoid segmentation (`parent_organoid` == -1).
#    - Also add flag for if the `object_id` for a single-cell is NaN.
# 3. Concat single-cell data together.

# In[ ]:


# Path to patient folders
path_to_patients = pathlib.Path(f"{profile_base_dir}/data/")

dfs = []
for patient_folder in path_to_patients.iterdir():
    single_cell_file = (
        patient_folder
        / f"{image_based_profiles_subparent_name}/2.annotated_profiles"
        / "sc_anno.parquet"
    )
    organoid_flags_file = (
        patient_folder
        / f"{image_based_profiles_subparent_name}/3.qc_profiles"
        / "organoid_flagged_outliers.parquet"
    )

    if single_cell_file.exists():
        sc_df = pd.read_parquet(single_cell_file)

        # Default QC flags
        sc_df["Metadata_cqc_organoid_flagged"] = False
        sc_df["Metadata_cqc_nan_detected"] = (
            sc_df[["Metadata_object_id", "Area.Size.Shape_Nuclei_VOLUME"]]
            .isna()
            .any(axis=1)
        )
        sc_df["Metadata_cqc_missing_parent_organoid"] = (
            sc_df["Metadata_parent_organoid"] == -1
        )

        if organoid_flags_file.exists():
            organoid_flags_df = pd.read_parquet(organoid_flags_file)[
                ["Metadata_object_id", "Metadata_image_set"]
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
                    ["Metadata_object_id", "Metadata_image_set"],
                ].itertuples(index=False, name=None)
            )

            # Flag SC rows where both parent_organoid & image_set match a flagged organoid
            sc_df["Metadata_cqc_organoid_flagged"] = sc_df.apply(
                lambda row: (row["Metadata_parent_organoid"], row["Metadata_image_set"])
                in flagged_pairs,
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

# In[ ]:


orig_single_cell_profiles_df


# In[ ]:


# Set the metadata columns to be used in the QC process
metadata_columns = [
    "Metadata_patient_tumor",
    "Metadata_image_set",
    "Metadata_object_id",
    "Metadata_parent_organoid",
    "Area.Size.Shape_Nuclei_CENTER.X",
    "Area.Size.Shape_Nuclei_CENTER.Y",
    "Metadata_cqc_nan_detected",
    "Metadata_cqc_organoid_flagged",
    "Metadata_cqc_missing_parent_organoid",
]


# In[ ]:


# Process each plate (patient_id) independently in the combined dataframe
for plate_name, plate_df in orig_single_cell_profiles_df.groupby(
    "Metadata_patient_tumor"
):
    print(f"Processing plate: {plate_name}")

    # Make a contiguous copy to prevent DataFrame fragmentation
    plate_df = plate_df.copy()

    # Only process the rows that are not flagged
    filtered_plate_df = plate_df[
        ~(
            plate_df["Metadata_cqc_nan_detected"]
            | plate_df["Metadata_cqc_organoid_flagged"]
            | plate_df["Metadata_cqc_missing_parent_organoid"]
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
    plate_df["Metadata_cqc_small_nuclei_outlier"] = False
    plate_df.loc[small_nuclei_outliers.index, "Metadata_cqc_small_nuclei_outlier"] = (
        True
    )

    print("Finding large nuclei outliers...")
    large_nuclei_outliers = find_outliers(
        df=filtered_plate_df,
        metadata_columns=metadata_columns,
        feature_thresholds={
            "Area.Size.Shape_Nuclei_VOLUME": 2,  # Detect very large nuclei
        },
    )

    # Ensure the column exists before assignment
    plate_df["Metadata_cqc_large_nuclei_outlier"] = False
    plate_df.loc[large_nuclei_outliers.index, "Metadata_cqc_large_nuclei_outlier"] = (
        True
    )

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
    plate_df["Metadata_cqc_mass_displacement_outlier"] = False
    plate_df.loc[
        high_mass_displacement_outliers.index, "Metadata_cqc_mass_displacement_outlier"
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
    output_folder = path_to_patients / plate_name / "image_based_profiles/3.qc_profiles"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / "sc_flagged_outliers.parquet"
    plate_df.to_parquet(output_file, index=False)
    print(f"Saved single-cell profiles with outlier flags to {output_file}\n")


# In[ ]:


# Print example output of the flagged single-cell profiles
print(f"Example flagged single-cell profiles: {plate_name}")
print(plate_df.shape)
plate_df.head()
