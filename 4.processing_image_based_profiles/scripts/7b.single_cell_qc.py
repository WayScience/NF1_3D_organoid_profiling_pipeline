#!/usr/bin/env python
# coding: utf-8

# # Perform single-cell level quality control

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
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# ## Load in each single-cell level profile per patient and process
#
# 1. Load in the single-cell data (add `patient_id` column).
# 2. Load in respective organoid qc data (only metadata and cqc columns) to already flag cells that come from a flagged organoid.
#    - Also add a flag for if single-cells do not have an organoid segmentation (`parent_organoid` == -1).
#    - Also add flag for if the `object_id` for a single-cell is NaN.
# 3. Concat single-cell data together.

# In[3]:


sc_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles/sc_anno.parquet"
)
organoid_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles/organoid_flagged_outliers.parquet"
)

output_dir = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles"
)
output_dir.mkdir(parents=True, exist_ok=True)

orig_sc_profiles_df = pd.read_parquet(sc_file)
organoid_qc_profiles_df = pd.read_parquet(organoid_file)
# Print the shape and head of the combined organoid profiles DataFrame
print(orig_sc_profiles_df.shape)
orig_sc_profiles_df


# In[4]:


sc_profiles_df = orig_sc_profiles_df.copy()
sc_profiles_df["Metadata_cqc_nan_detected"] = (
    sc_profiles_df[
        [
            "Metadata_Object_ObjectID",
            "Metadata_Object_ParentOrganoid",
            "Cell_NoChannel_AreaSizeShape_Volume",
        ]
    ]
    .isna()
    .any(axis=1)
)
# Print the number of organoids flagged
flagged_count = sc_profiles_df["Metadata_cqc_nan_detected"].sum()
print(f"Number of organoids flagged: {flagged_count}")

sc_profiles_df.head()


# In[5]:


# Path to patient folders


# Default QC flags
sc_profiles_df["Metadata_cqc_organoid_flagged"] = False
sc_profiles_df["Metadata_cqc_nan_detected"] = (
    sc_profiles_df[
        ["Metadata_Object_ObjectID", "Nuclei_NoChannel_AreaSizeShape_Volume"]
    ]
    .isna()
    .any(axis=1)
)
sc_profiles_df["Metadata_cqc_missing_parent_organoid"] = (
    sc_profiles_df["Metadata_Object_ParentOrganoid"] == -1
)


organoid_flags_df = organoid_qc_profiles_df[
    ["Metadata_Object_ObjectID", "Metadata_Experiment_WellFOV"]
    + [col for col in organoid_qc_profiles_df.columns if col.startswith("Metadata_cqc")]
]

# Get flagged (object_id, image_set) pairs
flagged_pairs = set(
    organoid_flags_df.loc[
        organoid_flags_df.filter(like="cqc").any(axis=1),
        ["Metadata_Object_ObjectID", "Metadata_Experiment_WellFOV"],
    ].itertuples(index=False, name=None)
)

# Flag SC rows where both parent_organoid & image_set match a flagged organoid
sc_profiles_df["Metadata_cqc_organoid_flagged"] = sc_profiles_df.apply(
    lambda row: (
        (row["Metadata_Object_ParentOrganoid"], row["Metadata_Experiment_WellFOV"])
        in flagged_pairs
    ),
    axis=1,
)

print(sc_profiles_df.shape)
sc_profiles_df.head()


# In[6]:


sc_profiles_df["Nuclei_NoChannel_AreaSizeShape_Volume"].describe()


# ## Detect outlier single-cells using the non-flagged data
#
# We will attempt to detect instances of poor quality segmentations using the nuclei compartment as the base. The conditions we are using are as follows:
#
# 1. Abnormally small or large nuclei using `Volume`
# 2. Abnormally high `mass displacement` in the nuclei for instances of mis-segmentation of background/no longer in-focus

# In[7]:


# Set the metadata columns to be used in the QC process
metadata_columns = [x for x in sc_profiles_df.columns if "Metadata" in x]


# In[8]:


# Only process the rows that are not flagged
filtered_plate_df = sc_profiles_df[
    ~(
        sc_profiles_df["Metadata_cqc_nan_detected"]
        | sc_profiles_df["Metadata_cqc_organoid_flagged"]
        | sc_profiles_df["Metadata_cqc_missing_parent_organoid"]
    )
]

# --- Find size based nuclei outliers ---
print("Finding small nuclei outliers...")
small_nuclei_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_NoChannel_AreaSizeShape_Volume": -1,  # Detect very small nuclei
    },
)

# Ensure the column exists before assignment
sc_profiles_df["Metadata_cqc_small_nuclei_outlier"] = False
sc_profiles_df.loc[small_nuclei_outliers.index, "Metadata_cqc_small_nuclei_outlier"] = (
    True
)

print("Finding large nuclei outliers...")
large_nuclei_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_NoChannel_AreaSizeShape_Volume": 2,  # Detect very large nuclei
    },
)

# Ensure the column exists before assignment
sc_profiles_df["Metadata_cqc_large_nuclei_outlier"] = False
sc_profiles_df.loc[large_nuclei_outliers.index, "Metadata_cqc_large_nuclei_outlier"] = (
    True
)

# --- Find mass displacement based nuclei outliers ---
print("Finding high mass displacement outliers...")
high_mass_displacement_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_DNA_Intensity_MassDisplacement": 2,  # Detect high mass displacement
    },
)

# Ensure the column exists before assignment
sc_profiles_df["Metadata_cqc_mass_displacement_outlier"] = False
sc_profiles_df.loc[
    high_mass_displacement_outliers.index, "Metadata_cqc_mass_displacement_outlier"
] = True

# Print number of outliers (only in filtered rows)
small_count = filtered_plate_df.index.intersection(small_nuclei_outliers.index).shape[0]
large_count = filtered_plate_df.index.intersection(large_nuclei_outliers.index).shape[0]
high_mass_count = filtered_plate_df.index.intersection(
    high_mass_displacement_outliers.index
).shape[0]

print(f"Small nuclei outliers found: {small_count}")
print(f"Large nuclei outliers found: {large_count}")
print(f"High mass displacement outliers found: {high_mass_count}")

# Save updated plate_df with flag columns included
output_file_path = pathlib.Path(f"{output_dir}/sc_flagged_outliers.parquet").resolve()
sc_profiles_df.to_parquet(output_file_path, index=False)


# In[9]:


sc_profiles_df.head()
