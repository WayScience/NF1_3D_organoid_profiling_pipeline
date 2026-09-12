#!/usr/bin/env python
# coding: utf-8

# # 6. Annotation
#
# ## Purpose
# Annotate the three combined profiles (SC, organoid, nucleocentric) for a single patient
# with treatment metadata, drug information, and microscope metadata. Standardize column
# naming under a `Metadata_*` prefix scheme and split each profile into hand-crafted and
# deep-learning feature subsets, producing 6 output parquets.
#
# This is **step 6 of Stage 4 (image-based profiling)**. It runs once per patient.
#
# ## Inputs
# - `data/{patient}/image_based_profiles/2.combined_profiles/sc.parquet`
# - `data/{patient}/image_based_profiles/2.combined_profiles/organoid.parquet`
# - `data/{patient}/image_based_profiles/2.combined_profiles/nucleocentric.parquet`
# - `config/platemaps/platemap.csv` — well-level treatment assignments
# - `config/drug_information/drug_information.csv` — drug target, class, therapeutic category
#
# ## Outputs
# Six annotated parquets in `data/{patient}/image_based_profiles/3.annotated_profiles/`:
#
# | File | Profile type | Feature set |
# |---|---|---|
# | `sc_anno.parquet` | Single-cell | Hand-crafted |
# | `organoid_anno.parquet` | Organoid | Hand-crafted |
# | `sammed_sc_anno.parquet` | Single-cell | SAMMed3D |
# | `sammed_organoid_anno.parquet` | Organoid | SAMMed3D |
# | `nucleocentric_sammed_anno.parquet` | Nucleocentric | SAMMed3D |
# | `nucleocentric_morphem_anno.parquet` | Nucleocentric | morphem |
#
# ## Notes
# - Metadata columns are sub-categorized as `Metadata_Biology_*`, `Metadata_Experiment_*`,
#   `Metadata_Object_*`, `Metadata_Location_*`, `Metadata_Neighbors_*`, `Metadata_Microscopy_*`.
# - Location and neighbor features are promoted to `Metadata_*` so they are excluded from
#   normalization and feature selection in downstream steps.

# In[1]:


import os
import pathlib

import pandas as pd
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
    patient = "NF0037_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# ## Combine the metadata into a single annotation file

# In[3]:


main_annotation_file_output = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/annotation_data/annotation_file_{patient}.csv"
).resolve()

if not main_annotation_file_output.exists():
    main_annotation_file_output.parent.mkdir(parents=True, exist_ok=True)

    platemap_path = pathlib.Path(
        f"{root_dir}/config/platemaps/barcode_platemap.csv"
    ).resolve(strict=True)

    drug_information = pd.read_csv(
        pathlib.Path(f"{root_dir}/config/drug_information/drug_information.csv")
    )
    patient_tumor_type = pd.read_csv(
        pathlib.Path(
            f"{root_dir}/config/patient_tumor_information/patient_tumor_information.csv"
        ),
    )
    patient_viabilities = pathlib.Path(f"{root_dir}/config/viabilities")
    patient_viabilities_paths = list(patient_viabilities.glob("*"))
    list_of_viabilities_dfs = []
    for path in patient_viabilities_paths:
        patient_tumor = path.stem.strip("_Viabilities")
        df = pd.read_csv(path)
        df["Metadata_Biology_PatientTumor"] = patient_tumor
        # replace the dose of DMSO from 0 to 1
        df.loc[df["Drug"] == "DMSO", "Concentration_uM"] = 1
        list_of_viabilities_dfs.append(df)
    viabilities_df = pd.concat(list_of_viabilities_dfs, ignore_index=True)
    # read platemap
    platemap = pd.read_csv(platemap_path)
    if patient == "NF0037_T1_CQ1":
        platemap = platemap[platemap["patient_tumor_barcode"] == "NF0037_T1"][
            "platemap_number"
        ].values[0]
    else:
        platemap = platemap[platemap["patient_tumor_barcode"] == patient][
            "platemap_number"
        ].values[0]
    platemap = pd.read_csv(pathlib.Path(f"{root_dir}/config/platemaps/{platemap}.csv"))
    # if % is in Treatment then delete the space leading to %
    platemap["Treatment"] = platemap["Treatment"].str.replace(r"\s+%", "%", regex=True)

    platemap.head()
    # Work on a copy to avoid mutating the caller's platemap across repeated calls.
    platemap_df = platemap.copy()
    # Merge strategy:
    #   1. Join platemap with drug_information on the first word of Treatment
    #      (e.g. "ARV-825 1 uM" → join key "ARV-825") to get Target, Class, etc.
    #   2. Join the resulting table onto the profile on Well == WellPosition.
    drug_information_platemap_merged = pd.merge(
        platemap_df,
        drug_information,
        left_on="Treatment",
        right_on="Treatment",
    )
    drug_information_platemap_viabilities_merged = pd.merge(
        left=drug_information_platemap_merged,
        right=viabilities_df,
        how="left",
        left_on=["Treatment", "Dose"],
        right_on=["Drug", "Concentration_uM"],
    )

    drug_information_platemap_viabilities_tumor_type_merged = pd.merge(
        left=drug_information_platemap_viabilities_merged,
        right=patient_tumor_type,
        how="left",
        left_on=["Metadata_Biology_PatientTumor"],
        right_on=["Metadata_Biology_PatientTumor"],
    )
    drug_information_platemap_viabilities_tumor_type_merged.drop(
        columns=[
            "WellRow",
            "WellCol",
            # "WellPosition",
            # "Treatment",
            # "Dose",
            # "Unit",
            # "Target",
            "Class",
            # "TherapeuticCategories",
            "Drug",
            "Concentration_uM",
            # "Viability_percentage",
            # "Metadata_Biology_PatientTumor",
            # "Metadata_Biology_TumorType"
        ],
        inplace=True,
    )
    annotation_df = drug_information_platemap_viabilities_tumor_type_merged.rename(
        columns={
            "WellPosition": "Metadata_Experiment_Well",
            "Treatment": "Metadata_Experiment_Treatment",
            "Dose": "Metadata_Experiment_Dose",
            "Unit": "Metadata_Experiment_Unit",
            "Target": "Metadata_Experiment_Target",
            "TherapeuticCategories": "Metadata_Experiment_TherapeuticCategories",
            "Viability_percentage": "Metadata_Experiment_ViabilityPercentage",
        }
    )
    annotation_df.to_csv(main_annotation_file_output, index=False)
else:
    annotation_df = pd.read_csv(main_annotation_file_output)

# subset the annotation_df to only include the patient of interest
# if NF0037_T1_CQ1, then subset to NF0037_T1 metadata
if patient == "NF0037_T1_CQ1":
    annotation_df = annotation_df.loc[
        annotation_df["Metadata_Biology_PatientTumor"] == "NF0037_T1"
    ]
else:
    annotation_df = annotation_df.loc[
        annotation_df["Metadata_Biology_PatientTumor"] == patient
    ]


# ## Pathing

# In[4]:


sc_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/sc.parquet"
).resolve(strict=True)
organoid_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/organoid.parquet"
).resolve(strict=True)
nucleocentric_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/nucleocentric.parquet"
).resolve(strict=True)

# output path
sc_annotated_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/sc_anno.parquet"
).resolve()
organoid_annotated_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/organoid_anno.parquet"
).resolve()
nucleocentric_annotated_sammed_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/nucleocentric_sammed_anno.parquet"
).resolve()
nucleocentric_annotated_morphem_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/nucleocentric_morphem_anno.parquet"
).resolve()
sammed_annotated_sc_profiles_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/sammed_sc_anno.parquet"
).resolve()
sammed_annotated_organoid_profiles_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/sammed_organoid_anno.parquet"
).resolve()

organoid_annotated_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[5]:


# read data
sc_merged = pd.read_parquet(sc_merged_path)
organoid_merged = pd.read_parquet(organoid_merged_path)
nucleocentric_merged = pd.read_parquet(nucleocentric_merged_path)

sc_merged["Well"] = sc_merged["image_set"].str.split("-").str[0]
organoid_merged["Well"] = organoid_merged["image_set"].str.split("-").str[0]
nucleocentric_merged["Well"] = nucleocentric_merged["image_set"].str.split("-").str[0]


# In[6]:


sc_merged = pd.merge(
    left=sc_merged,
    right=annotation_df,
    how="left",
    left_on=["Well"],
    right_on=["Metadata_Experiment_Well"],
)
organoid_merged = pd.merge(
    left=organoid_merged,
    right=annotation_df,
    how="left",
    left_on=["Well"],
    right_on=["Metadata_Experiment_Well"],
)
nucleocentric_merged = pd.merge(
    left=nucleocentric_merged,
    right=annotation_df,
    how="left",
    left_on=["Well"],
    right_on=["Metadata_Experiment_Well"],
)
# remove redundant columns
columns_to_drop = [
    "image_set_1",
    "image_set_2",
    "WellRow",
    "WellCol",
]
sc_merged.drop(
    columns=[x for x in columns_to_drop if x in sc_merged.columns], inplace=True
)
organoid_merged.drop(
    columns=[x for x in columns_to_drop if x in organoid_merged.columns], inplace=True
)
nucleocentric_merged.drop(
    columns=[x for x in columns_to_drop if x in nucleocentric_merged.columns],
    inplace=True,
)


# ### Get single cell counts per well and organoid counts per well

# In[7]:


sc_merged["Metadata_WellSingleCellCount"] = sc_merged.groupby("Well")[
    "image_set"
].transform("count")
organoid_merged["Metadata_WellOrganoidCount"] = organoid_merged.groupby("Well")[
    "image_set"
].transform("count")
nucleocentric_merged["Metadata_WellNucleocentricCount"] = nucleocentric_merged.groupby(
    "Well"
)["image_set"].transform("count")


# In[8]:


column_rename_mapping = {
    "patient": "PatientTumor",
    "image_set": "WellFOV",
    "object_id": "ObjectID",
}

# rename columns for consistency across profiles
sc_merged.rename(columns=column_rename_mapping, inplace=True)
organoid_merged.rename(columns=column_rename_mapping, inplace=True)
nucleocentric_merged.rename(columns=column_rename_mapping, inplace=True)


# In[9]:


# Promote spatial coordinate columns to Metadata_Location_* so they are excluded
# from normalization and feature selection in downstream steps.
# Intensity-based location columns (MinX/MaxX etc. from intensity measurements)
# are dropped entirely — only AreaSizeShape-derived coordinates are kept.

organoid_location_features = [
    x
    for x in organoid_merged.columns
    if (
        ("area" in (xl := x.lower()) and any(k in xl for k in ("max", "min", "center")))
        or (
            "intensity" in xl
            and any(k in xl for k in ("maxx", "minx", "maxy", "miny", "maxz", "minz"))
        )
    )
]

sc_location_features = [
    x
    for x in sc_merged.columns
    if (
        ("area" in (xl := x.lower()) and any(k in xl for k in ("max", "min", "center")))
        or (
            "intensity" in xl
            and any(k in xl for k in ("maxx", "minx", "maxy", "miny", "maxz", "minz"))
        )
    )
]
# drop the intensity location features
sc_merged.drop(
    columns=[col for col in sc_location_features if "intensity" in col.lower()],
    inplace=True,
)
organoid_merged.drop(
    columns=[col for col in organoid_location_features if "intensity" in col.lower()],
    inplace=True,
)
# remove the intensity location features from the list of features
sc_location_features = [
    col for col in sc_location_features if "intensity" not in col.lower()
]
organoid_location_features = [
    col for col in organoid_location_features if "intensity" not in col.lower()
]
_ = [
    organoid_merged.rename(
        columns={
            feature: f"Metadata_Location_{feature.split('_')[0]}_{feature.split('_')[-1]}"
        },
        inplace=True,
    )
    for feature in organoid_location_features
]
_ = [
    sc_merged.rename(
        columns={
            feature: f"Metadata_Location_{feature.split('_')[0]}_{feature.split('_')[-1]}"
        },
        inplace=True,
    )
    for feature in sc_location_features
]


# In[10]:


sc_neighbors_features = [col for col in sc_merged.columns if "neighbors" in col.lower()]
# replace "Object_Channel with Metadata_"
_ = [
    sc_merged.rename(
        columns={feature: f"Metadata_Neighbors_{feature.split('_')[-1]}"},
        inplace=True,
    )
    for feature in sc_neighbors_features
]


# In[11]:


metadata_features_list = [
    "PatientTumor",
    "Tumor",
    "ObjectID",
    "Well",
    "Treatment",
    "Dose",
    "Unit",
    "WellFOV",
    "ParentOrganoid",
    "OrganoidSingleCellCount",
    "Target",
    "Class",
    "TherapeuticCategories",
]
# prepend "Metadata_" to metadata features
sc_merged = sc_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)
organoid_merged = organoid_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)
nucleocentric_merged = nucleocentric_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)
# add microscope metadata
(
    sc_merged["Metadata_MicroscopeType"],
    organoid_merged["Metadata_MicroscopeType"],
    nucleocentric_merged["Metadata_MicroscopeType"],
) = ("spinning disk confocal", "spinning disk confocal", "spinning disk confocal")
(
    sc_merged["Metadata_MicroscopeName"],
    organoid_merged["Metadata_MicroscopeName"],
    nucleocentric_merged["Metadata_MicroscopeName"],
) = (
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
)
(
    sc_merged["Metadata_Magnification"],
    organoid_merged["Metadata_Magnification"],
    nucleocentric_merged["Metadata_Magnification"],
) = ("60x", "60x", "60x")

(
    sc_merged["Metadata_XResolutionUm"],
    organoid_merged["Metadata_XResolutionUm"],
    nucleocentric_merged["Metadata_XResolutionUm"],
) = (0.101, 0.101, 0.101)
(
    sc_merged["Metadata_YResolutionUm"],
    organoid_merged["Metadata_YResolutionUm"],
    nucleocentric_merged["Metadata_YResolutionUm"],
) = (0.101, 0.101, 0.101)
(
    sc_merged["Metadata_ZResolutionUm"],
    organoid_merged["Metadata_ZResolutionUm"],
    nucleocentric_merged["Metadata_ZResolutionUm"],
) = (1.0, 1.0, 1.0)


# In[12]:


# Sub-categorize all Metadata_* columns into four namespaces:
#   Biology_    — patient/tumor identity (who the sample came from)
#   Experiment_ — treatment, well, and drug annotation (what was done)
#   Object_     — per-object identifiers and counts (what object this row represents)
#   Microscopy_ — instrument and acquisition parameters (how it was imaged)
# Metadata_Location_* and Metadata_Neighbors_* were already renamed in earlier cells.
# After renaming, all Metadata_* columns are moved to the front and rows are sorted.
biology_features = [
    "Metadata_PatientTumor",
    "Metadata_Patient",
    "Metadata_Tumor",
]
experiment_features = [
    "Metadata_Treatment",
    "Metadata_Dose",
    "Metadata_Unit",
    "Metadata_Well",
    "Metadata_WellFOV",
    "Metadata_Target",
    "Metadata_Class",
    "Metadata_TherapeuticCategories",
]
object_features = [
    "Metadata_ObjectID",
    "Metadata_ParentOrganoid",
    "Metadata_SingleCellCount",
    "Metadata_WellSingleCellCount",
    "Metadata_OrganoidSingleCellCount",
]
microscopy_features = [
    "Metadata_MicroscopeType",
    "Metadata_MicroscopeName",
    "Metadata_Magnification",
    "Metadata_XResolutionUm",
    "Metadata_YResolutionUm",
    "Metadata_ZResolutionUm",
]

# Build rename mapping once
rename_map = {}
for col in biology_features:
    rename_map[col] = col.replace("Metadata_", "Metadata_Biology_")
for col in experiment_features:
    rename_map[col] = col.replace("Metadata_", "Metadata_Experiment_")
for col in object_features:
    rename_map[col] = col.replace("Metadata_", "Metadata_Object_")
for col in microscopy_features:
    rename_map[col] = col.replace("Metadata_", "Metadata_Microscopy_")

# Apply once to each dataframe
sc_merged.rename(columns=rename_map, inplace=True)
organoid_merged.rename(columns=rename_map, inplace=True)
nucleocentric_merged.rename(columns=rename_map, inplace=True)

# move all metadata columns to the front by sorting columns based on the prefix "Metadata_"
sc_merged = sc_merged[
    sorted(sc_merged.columns, key=lambda x: (not x.startswith("Metadata_"), x))
]
organoid_merged = organoid_merged[
    sorted(organoid_merged.columns, key=lambda x: (not x.startswith("Metadata_"), x))
]
nucleocentric_merged = nucleocentric_merged[
    sorted(
        nucleocentric_merged.columns, key=lambda x: (not x.startswith("Metadata_"), x)
    )
]

# sort the dfs by patient, WellFov, and ObjectID (for single-cell)
sc_merged = sc_merged.sort_values(
    by=[
        "Metadata_Biology_PatientTumor",
        "Metadata_Experiment_WellFOV",
        "Metadata_Object_ObjectID",
    ]
).reset_index(drop=True)
organoid_merged = organoid_merged.sort_values(
    by=["Metadata_Biology_PatientTumor", "Metadata_Experiment_WellFOV"]
).reset_index(drop=True)
nucleocentric_merged = nucleocentric_merged.sort_values(
    by=[
        "Metadata_Biology_PatientTumor",
        "Metadata_Experiment_WellFOV",
        "Metadata_Object_ObjectID",
    ]
).reset_index(drop=True)

# find duplicate columns and keep one of the duplicates
sc_merged = sc_merged.loc[:, ~sc_merged.columns.duplicated()]
organoid_merged = organoid_merged.loc[:, ~organoid_merged.columns.duplicated()]
nucleocentric_merged = nucleocentric_merged.loc[
    :, ~nucleocentric_merged.columns.duplicated()
]


# In[13]:


# Split each profile into feature subsets by column name pattern:
#   - Hand-crafted: columns with no 'sammed' or 'morphem' in name (AreaSizeShape, Intensity, etc.)
#   - SAMMed3D: columns containing 'sammed' (3D volumetric deep learning embeddings)
#   - morphem: columns containing 'morphem' (2D nucleocentric projection embeddings)
#
# This produces 6 output dataframes (3 profile types × 2 feature sets for SC/organoid,
# and SAMMed3D + morphem for nucleocentric) saved separately in cell 16.
sc_metadata_columns = [x for x in sc_merged.columns if "Metadata" in x]
sc_handcrafted_columns = [
    x for x in sc_merged.columns if "Metadata" not in x and "sammed" not in x.lower()
]
sc_sammed_columns = [x for x in sc_merged.columns if "sammed" in x.lower()]

organoid_metadata_columns = [x for x in organoid_merged.columns if "Metadata" in x]
organoid_handcrafted_columns = [
    x
    for x in organoid_merged.columns
    if "Metadata" not in x and "sammed" not in x.lower()
]
organoid_sammed_columns = [x for x in organoid_merged.columns if "sammed" in x.lower()]

nucleocentric_metadata_columns = [
    x for x in nucleocentric_merged.columns if "Metadata" in x
]
nucleocentric_sammed_columns = [
    x for x in nucleocentric_merged.columns if "sammed" in x.lower()
]
nucleocentric_morphem_columns = [
    x for x in nucleocentric_merged.columns if "chammi" in x.lower()
]

# split the profiles
sc_annotated = sc_merged[sc_metadata_columns + sc_handcrafted_columns]
sc_annotated_sammed = sc_merged[sc_metadata_columns + sc_sammed_columns]
organoid_annotated = organoid_merged[
    organoid_metadata_columns + organoid_handcrafted_columns
]
organoid_annotated_sammed = organoid_merged[
    organoid_metadata_columns + organoid_sammed_columns
]
nucleocentric_sammed_annotated = nucleocentric_merged[
    nucleocentric_metadata_columns + nucleocentric_sammed_columns
]
nucleocentric_morphem_annotated = nucleocentric_merged[
    nucleocentric_metadata_columns + nucleocentric_morphem_columns
]


# In[ ]:


# save annotated profiles
sc_annotated.to_parquet(sc_annotated_output_path, index=False)
organoid_annotated.to_parquet(organoid_annotated_output_path, index=False)
sc_annotated_sammed.to_parquet(sammed_annotated_sc_profiles_path, index=False)
organoid_annotated_sammed.to_parquet(
    sammed_annotated_organoid_profiles_path, index=False
)
nucleocentric_sammed_annotated.to_parquet(
    nucleocentric_annotated_sammed_output_path, index=False
)
nucleocentric_morphem_annotated.to_parquet(
    nucleocentric_annotated_morphem_output_path, index=False
)


# In[ ]:


sc_annotated.head()


# In[ ]:


organoid_annotated.head()
