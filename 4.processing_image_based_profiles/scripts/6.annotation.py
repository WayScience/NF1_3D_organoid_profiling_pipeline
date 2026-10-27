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


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0037_T1_CQ1"
    image_based_profiles_subparent_name = "image_based_profiles"


# ## Combine the metadata into a single annotation file

# In[3]:


barcode_platemap = pd.read_csv(
    pathlib.Path(f"{root_dir}/config/platemaps/barcode_platemap.csv").resolve(
        strict=True
    )
)
# NF0037_T1_CQ1 shares NF0037_T1's own plate layout/barcode row -- look up by
# that name, but keep `patient` itself as "NF0037_T1_CQ1" everywhere else.
# Previously this branch assigned the whole filtered DataFrame to `platemap`
# instead of the platemap_number string like the else branch does -- an
# f-string of a DataFrame produces garbage, so building this file for
# NF0037_T1_CQ1 first would have failed outright.
_platemap_lookup_patient = "NF0037_T1" if patient == "NF0037_T1_CQ1" else patient
_platemap_matches = barcode_platemap[
    barcode_platemap["patient_tumor_barcode"] == _platemap_lookup_patient
]["platemap_number"]
if len(_platemap_matches) != 1:
    raise ValueError(
        f"Expected exactly one platemap_number for patient {patient} "
        f"(looked up as {_platemap_lookup_patient}) in "
        f"config/platemaps/barcode_platemap.csv, found {len(_platemap_matches)}"
    )
platemap = _platemap_matches.iloc[0]

# Cache keyed by platemap number, not a single shared filename: multiple
# patients can use different platemaps (confirmed directly against
# config/platemaps/barcode_platemap.csv -- NF0037_T1/NF0040_T1/NF0055_T1 use
# platemap2, everyone else here uses platemap1). A single shared cache file
# would lock in whichever platemap the first patient to run happened to use,
# silently leaving every other-platemap patient's annotation_df with zero
# matching rows.
#
# This cache holds ONLY the patient-agnostic plate layout (well -> treatment/
# dose/target/class/therapeutic category) -- identical for every patient
# sharing this platemap. Viability and tumor-type were previously baked into
# this same cached table via a left-merge keyed on (Treatment, Dose), which
# is patient-specific data disguised as if it were part of the shared plate
# layout. Confirmed directly this caused silent data loss: viability is
# fanned out one row per patient who has a measurement for that exact
# (Treatment, Dose), so any well+treatment where a DIFFERENT patient (not
# this one) happened to be the one with a recorded viability produced a row
# whose PatientTumor was that other patient -- excluded entirely once this
# patient's own subset was filtered down, even though the well/treatment
# itself applies to this patient's plate too (real measured impact: up to
# 87% of NF0040_T1's single-cell rows lost their entire annotation this
# way, not just DMSO). Viability/tumor-type are now looked up fresh per
# patient below, outside the cache, as optional enrichment that never gates
# which plate-layout rows survive.
main_annotation_file_output = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/annotation_data/external_platemap_metadata_{platemap}.csv"
).resolve()

if not main_annotation_file_output.exists():
    main_annotation_file_output.parent.mkdir(parents=True, exist_ok=True)

    drug_information = pd.read_csv(
        pathlib.Path(f"{root_dir}/config/drug_information/drug_information.csv")
    )
    platemap_df = pd.read_csv(
        pathlib.Path(f"{root_dir}/config/platemaps/{platemap}.csv")
    )
    # if % is in Treatment then delete the space leading to %
    platemap_df["Treatment"] = platemap_df["Treatment"].str.replace(
        r"\s+%", "%", regex=True
    )
    drug_information_platemap_merged = pd.merge(
        platemap_df,
        drug_information,
        left_on="Treatment",
        right_on="Treatment",
    )
    drug_information_platemap_merged.drop(
        columns=["WellRow", "WellCol"],
        inplace=True,
    )
    annotation_df = drug_information_platemap_merged.rename(
        columns={
            "WellPosition": "Metadata_Experiment_Well",
            "Treatment": "Metadata_Experiment_Treatment",
            "Dose": "Metadata_Experiment_Dose",
            "Unit": "Metadata_Experiment_Unit",
            "Target": "Metadata_Experiment_Target",
            "Class": "Metadata_Experiment_Class",
            "TherapeuticCategories": "Metadata_Experiment_TherapeuticCategories",
        }
    )
    annotation_df.to_csv(main_annotation_file_output, index=False)
else:
    annotation_df = pd.read_csv(main_annotation_file_output)

# Enrich with this specific patient's own identity, tumor type, and (where
# available) measured viability -- all patient-specific, so applied fresh
# every run rather than cached. Every plate-layout row from above is kept
# regardless of whether a viability measurement exists for it: viability is
# optional metadata, not a gate on which wells belong to this patient.
annotation_df["Metadata_Biology_PatientTumor"] = _platemap_lookup_patient

patient_tumor_type = pd.read_csv(
    pathlib.Path(
        f"{root_dir}/config/patient_tumor_information/patient_tumor_information.csv"
    ),
)
annotation_df = annotation_df.merge(
    patient_tumor_type,
    how="left",
    on="Metadata_Biology_PatientTumor",
)

patient_viabilities_df = pd.read_csv(
    pathlib.Path(f"{root_dir}/config/viabilities/raw_viabilities_combined.csv").resolve(
        strict=True
    )
)
patient_viabilities_df = patient_viabilities_df.loc[
    patient_viabilities_df["Metadata_Biology_PatientTumor"] == _platemap_lookup_patient,
    ["Drug", "Concentration_uM", "Viability_percentage"],
]
annotation_df = annotation_df.merge(
    patient_viabilities_df,
    how="left",
    left_on=["Metadata_Experiment_Treatment", "Metadata_Experiment_Dose"],
    right_on=["Drug", "Concentration_uM"],
).drop(columns=["Drug", "Concentration_uM"])
annotation_df = annotation_df.rename(
    columns={"Viability_percentage": "Metadata_Experiment_ViabilityPercentage"}
)
if patient == "NF0037_T1_CQ1":
    # NF0037_T1_CQ1 shares NF0037_T1's own plate layout/barcode row -- look up by
    # that name, but keep `patient` itself as "NF0037_T1_CQ1" everywhere else.
    annotation_df = annotation_df.loc[
        annotation_df["Metadata_Biology_PatientTumor"] == "NF0037_T1"
    ]
    annotation_df["Metadata_Biology_PatientTumor"] = patient
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
sc_sammed_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/sc_sammed.parquet"
).resolve(strict=True)
organoid_sammed_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/organoid_sammed.parquet"
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
sc_sammed_merged = pd.read_parquet(sc_sammed_merged_path)
organoid_sammed_merged = pd.read_parquet(organoid_sammed_merged_path)
nucleocentric_merged = pd.read_parquet(nucleocentric_merged_path)


# In[6]:


sc_merged["image_set"] = (
    sc_merged["Metadata_Experiment_WellID"].astype(str)
    + "_"
    + sc_merged["Metadata_Imaging_FieldID"].astype(str)
)
organoid_merged["image_set"] = (
    organoid_merged["Metadata_Experiment_WellID"].astype(str)
    + "_"
    + organoid_merged["Metadata_Imaging_FieldID"].astype(str)
)

sc_merged["Well"] = sc_merged["image_set"].str.split("-").str[0]
organoid_merged["Well"] = organoid_merged["image_set"].str.split("-").str[0]
sc_sammed_merged["Well"] = sc_sammed_merged["image_set"].str.split("-").str[0]
organoid_sammed_merged["Well"] = (
    organoid_sammed_merged["image_set"].str.split("-").str[0]
)
nucleocentric_merged["Well"] = nucleocentric_merged["image_set"].str.split("-").str[0]


# In[7]:


sc_merged = pd.merge(
    left=sc_merged,
    right=annotation_df,
    how="left",
    left_on=["Metadata_Experiment_WellID", "Metadata_Biology_PatientTumor"],
    right_on=["Metadata_Experiment_Well", "Metadata_Biology_PatientTumor"],
)
organoid_merged = pd.merge(
    left=organoid_merged,
    right=annotation_df,
    how="left",
    left_on=["Metadata_Experiment_WellID", "Metadata_Biology_PatientTumor"],
    right_on=["Metadata_Experiment_Well", "Metadata_Biology_PatientTumor"],
)
sc_sammed_merged = pd.merge(
    left=sc_sammed_merged,
    right=annotation_df,
    how="left",
    left_on=["Well"],
    right_on=["Metadata_Experiment_Well"],
)
organoid_sammed_merged = pd.merge(
    left=organoid_sammed_merged,
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
sc_sammed_merged.drop(
    columns=[x for x in columns_to_drop if x in sc_sammed_merged.columns], inplace=True
)
organoid_sammed_merged.drop(
    columns=[x for x in columns_to_drop if x in organoid_sammed_merged.columns],
    inplace=True,
)
nucleocentric_merged.drop(
    columns=[x for x in columns_to_drop if x in nucleocentric_merged.columns],
    inplace=True,
)


# ### Get single cell counts per well and organoid counts per well

# In[8]:


sc_merged["Metadata_WellSingleCellCount"] = sc_merged.groupby("Well")[
    "image_set"
].transform("count")
organoid_merged["Metadata_WellOrganoidCount"] = organoid_merged.groupby("Well")[
    "image_set"
].transform("count")
sc_sammed_merged["Metadata_WellSingleCellCount"] = sc_sammed_merged.groupby("Well")[
    "image_set"
].transform("count")
organoid_sammed_merged["Metadata_WellOrganoidCount"] = organoid_sammed_merged.groupby(
    "Well"
)["image_set"].transform("count")
nucleocentric_merged["Metadata_WellNucleocentricCount"] = nucleocentric_merged.groupby(
    "Well"
)["image_set"].transform("count")


# In[9]:


column_rename_mapping = {
    "patient": "PatientTumor",
    "image_set": "WellFOV",
    "object_id": "ObjectID",
}

# rename columns for consistency across profiles
sc_merged.rename(columns=column_rename_mapping, inplace=True)
organoid_merged.rename(columns=column_rename_mapping, inplace=True)
sc_sammed_merged.rename(columns=column_rename_mapping, inplace=True)
organoid_sammed_merged.rename(columns=column_rename_mapping, inplace=True)
nucleocentric_merged.rename(columns=column_rename_mapping, inplace=True)


# In[10]:


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
            and any(
                k in xl for k in ("maxx", "minx", "maxy", "miny", "maxz", "minz", "cmi")
            )
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
            and any(
                k in xl for k in ("maxx", "minx", "maxy", "miny", "maxz", "minz", "cmi")
            )
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


# In[11]:


sc_neighbors_features = [col for col in sc_merged.columns if "neighbors" in col.lower()]
# replace "Object_Channel with Metadata_"
_ = [
    sc_merged.rename(
        columns={feature: f"Metadata_Neighbors_{feature.replace('_', '')}"},
        inplace=True,
    )
    for feature in sc_neighbors_features
]


# In[12]:


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
    columns={
        col: f"Metadata_{col}"
        for col in metadata_features_list
        if not f"Metadata_{col}" in sc_merged.columns
    }
)
organoid_merged = organoid_merged.rename(
    columns={
        col: f"Metadata_{col}"
        for col in metadata_features_list
        if not f"Metadata_{col}" in organoid_merged.columns
    }
)
sc_sammed_merged = sc_sammed_merged.rename(
    columns={
        col: f"Metadata_{col}"
        for col in metadata_features_list
        if not f"Metadata_{col}" in sc_sammed_merged.columns
    }
)
organoid_sammed_merged = organoid_sammed_merged.rename(
    columns={
        col: f"Metadata_{col}"
        for col in metadata_features_list
        if not f"Metadata_{col}" in organoid_sammed_merged.columns
    }
)
nucleocentric_merged = nucleocentric_merged.rename(
    columns={
        col: f"Metadata_{col}"
        for col in metadata_features_list
        if not f"Metadata_{col}" in nucleocentric_merged.columns
    }
)
# add microscope metadata
(
    sc_merged["Metadata_MicroscopeType"],
    organoid_merged["Metadata_MicroscopeType"],
    sc_sammed_merged["Metadata_MicroscopeType"],
    organoid_sammed_merged["Metadata_MicroscopeType"],
    nucleocentric_merged["Metadata_MicroscopeType"],
) = (
    "spinning disk confocal",
    "spinning disk confocal",
    "spinning disk confocal",
    "spinning disk confocal",
    "spinning disk confocal",
)
(
    sc_merged["Metadata_MicroscopeName"],
    organoid_merged["Metadata_MicroscopeName"],
    sc_sammed_merged["Metadata_MicroscopeName"],
    organoid_sammed_merged["Metadata_MicroscopeName"],
    nucleocentric_merged["Metadata_MicroscopeName"],
) = (
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
    "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1",
)
(
    sc_merged["Metadata_Magnification"],
    organoid_merged["Metadata_Magnification"],
    sc_sammed_merged["Metadata_Magnification"],
    organoid_sammed_merged["Metadata_Magnification"],
    nucleocentric_merged["Metadata_Magnification"],
) = ("60x", "60x", "60x", "60x", "60x")

(
    sc_merged["Metadata_XResolutionUm"],
    organoid_merged["Metadata_XResolutionUm"],
    sc_sammed_merged["Metadata_XResolutionUm"],
    organoid_sammed_merged["Metadata_XResolutionUm"],
    nucleocentric_merged["Metadata_XResolutionUm"],
) = (0.101, 0.101, 0.101, 0.101, 0.101)
(
    sc_merged["Metadata_YResolutionUm"],
    organoid_merged["Metadata_YResolutionUm"],
    sc_sammed_merged["Metadata_YResolutionUm"],
    organoid_sammed_merged["Metadata_YResolutionUm"],
    nucleocentric_merged["Metadata_YResolutionUm"],
) = (0.101, 0.101, 0.101, 0.101, 0.101)
(
    sc_merged["Metadata_ZResolutionUm"],
    organoid_merged["Metadata_ZResolutionUm"],
    sc_sammed_merged["Metadata_ZResolutionUm"],
    organoid_sammed_merged["Metadata_ZResolutionUm"],
    nucleocentric_merged["Metadata_ZResolutionUm"],
) = (1.0, 1.0, 1.0, 1.0, 1.0)


# In[13]:


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


# Apply once to each dataframe, keeping only renames whose source column exists
# in that dataframe and whose target name is not already present (e.g. don't
# rename Metadata_Well into an existing Metadata_Experiment_Well)
for _df in (
    sc_merged,
    organoid_merged,
    nucleocentric_merged,
    organoid_sammed_merged,
    sc_sammed_merged,
):
    _df.rename(
        columns={
            k: v
            for k, v in rename_map.items()
            if k in _df.columns and v not in _df.columns
        },
        inplace=True,
    )

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
organoid_sammed_merged = organoid_sammed_merged[
    sorted(
        organoid_sammed_merged.columns, key=lambda x: (not x.startswith("Metadata_"), x)
    )
]
sc_sammed_merged = sc_sammed_merged[
    sorted(sc_sammed_merged.columns, key=lambda x: (not x.startswith("Metadata_"), x))
]
# if old prefix columns exist, drop them
old_prefixes = [
    "Metadata_PatientTumor",
    "Metadata_Treatment",
    "Metadata_Dose",
    "Metadata_Unit",
    "Metadata_Well",
    "Metadata_WellFOV",
    "Metadata_Target",
    "Metadata_Class",
    "Metadata_TherapeuticCategories",
    "Metadata_ObjectID",
    "Metadata_ParentOrganoid",
    "Metadata_SingleCellCount",
    "Metadata_WellSingleCellCount",
    "Metadata_OrganoidSingleCellCount",
    "Metadata_MicroscopeType",
    "Metadata_MicroscopeName",
    "Metadata_Magnification",
    "Metadata_XResolutionUm",
    "Metadata_YResolutionUm",
    "Metadata_ZResolutionUm",
]
sc_merged.drop(
    columns=[col for col in old_prefixes if col in sc_merged.columns], inplace=True
)
organoid_merged.drop(
    columns=[col for col in old_prefixes if col in organoid_merged.columns],
    inplace=True,
)
nucleocentric_merged.drop(
    columns=[col for col in old_prefixes if col in nucleocentric_merged.columns],
    inplace=True,
)
organoid_sammed_merged.drop(
    columns=[col for col in old_prefixes if col in organoid_sammed_merged.columns],
    inplace=True,
)
sc_sammed_merged.drop(
    columns=[col for col in old_prefixes if col in sc_sammed_merged.columns],
    inplace=True,
)
# drop duplicate columns if they exist
sc_merged = sc_merged.loc[:, ~sc_merged.columns.duplicated()]
organoid_merged = organoid_merged.loc[:, ~organoid_merged.columns.duplicated()]
nucleocentric_merged = nucleocentric_merged.loc[
    :, ~nucleocentric_merged.columns.duplicated()
]
organoid_sammed_merged = organoid_sammed_merged.loc[
    :, ~organoid_sammed_merged.columns.duplicated()
]
sc_sammed_merged = sc_sammed_merged.loc[:, ~sc_sammed_merged.columns.duplicated()]


# In[14]:


# Split each profile into feature subsets by column name pattern:
#   - Hand-crafted: columns with no 'sammed' or 'morphem' in name (AreaSizeShape, Intensity, etc.)
#   - SAMMed3D: columns containing 'sammed' (3D volumetric deep learning embeddings)
#   - morphem: columns containing 'morphem' (2D nucleocentric projection embeddings)
#
# This produces 6 output dataframes (3 profile types × 2 feature sets for SC/organoid,
# and SAMMed3D + morphem for nucleocentric) saved separately in cell 16.

nucleocentric_metadata_columns = [
    x for x in nucleocentric_merged.columns if "Metadata" in x
]
nucleocentric_sammed_columns = [
    x for x in nucleocentric_merged.columns if "sammed" in x.lower()
]
nucleocentric_morphem_columns = [
    x for x in nucleocentric_merged.columns if "chammi" in x.lower()
]

nucleocentric_sammed_annotated = nucleocentric_merged[
    nucleocentric_metadata_columns + nucleocentric_sammed_columns
]
nucleocentric_morphem_annotated = nucleocentric_merged[
    nucleocentric_metadata_columns + nucleocentric_morphem_columns
]


# In[15]:


# save annotated profiles
sc_merged.to_parquet(sc_annotated_output_path, index=False)
organoid_merged.to_parquet(organoid_annotated_output_path, index=False)
sc_sammed_merged.to_parquet(sammed_annotated_sc_profiles_path, index=False)
organoid_sammed_merged.to_parquet(sammed_annotated_organoid_profiles_path, index=False)
nucleocentric_sammed_annotated.to_parquet(
    nucleocentric_annotated_sammed_output_path, index=False
)
nucleocentric_morphem_annotated.to_parquet(
    nucleocentric_annotated_morphem_output_path, index=False
)


# In[16]:


shapes_dict = {
    "sc_merged": sc_merged.shape,
    "organoid_merged": organoid_merged.shape,
    "sc_sammed_merged": sc_sammed_merged.shape,
    "organoid_sammed_merged": organoid_sammed_merged.shape,
    "nucleocentric_merged": nucleocentric_merged.shape,
    "nucleocentric_sammed_annotated": nucleocentric_sammed_annotated.shape,
    "nucleocentric_morphem_annotated": nucleocentric_morphem_annotated.shape,
}
for key, value in shapes_dict.items():
    print(f"{key}: {value}")
