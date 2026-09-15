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
    f"{root_dir}/4.processing_image_based_profiles/annotation_data/external_platemap_metadata.csv"
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
    patient_viabilities = pathlib.Path(
        f"{root_dir}/config/viabilities/raw_viabilities_combined.csv"
    ).resolve(strict=True)
    patient_viabilities_df = pd.read_csv(patient_viabilities)
    # read platemap
    barcode_platemap = pd.read_csv(platemap_path)
    if patient == "NF0037_T1_CQ1":
        platemap = barcode_platemap[
            barcode_platemap["patient_tumor_barcode"] == "NF0037_T1"
        ]
    else:
        platemap = barcode_platemap[
            barcode_platemap["patient_tumor_barcode"] == patient
        ]["platemap_number"].values[0]
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
    drug_information_platemap_viabilities_merged = pd.merge(
        left=drug_information_platemap_merged,
        right=patient_viabilities_df,
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
    # "Class" (drug_information.csv's own field, e.g. "Small Molecule",
    # "PROTAC") was previously dropped here rather than kept and renamed
    # like its sibling Target/TherapeuticCategories fields -- confirmed a
    # real bug: 10.aggregation.py and 11.combine_patients.py's own
    # aggregate_strata/consensus_strata both group by
    # "Metadata_Experiment_Class", which as a result never existed.
    drug_information_platemap_viabilities_tumor_type_merged.drop(
        columns=[
            "WellRow",
            "WellCol",
            "Drug",
            "Concentration_uM",
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
            "Class": "Metadata_Experiment_Class",
            "TherapeuticCategories": "Metadata_Experiment_TherapeuticCategories",
            "Viability_percentage": "Metadata_Experiment_ViabilityPercentage",
        }
    )
    annotation_df.to_csv(main_annotation_file_output, index=False)
else:
    annotation_df = pd.read_csv(main_annotation_file_output)

# subset the annotation_df to only include the patient of interest
# if NF0037_T1_CQ1, then subset to NF0037_T1 metadata
#
# Also keep rows with a NaN Metadata_Biology_PatientTumor: that column is
# populated from the viabilities left-merge above, keyed on
# (Treatment, Dose) -- a treatment nobody has a recorded viability
# measurement for (verified directly: DMSO, the plate's own vehicle
# control, has zero rows in config/viabilities/raw_viabilities_combined.csv
# for any patient) ends up NaN there regardless of which patient's plate it
# came from. Since platemap_df was already scoped to this patient's own
# barcode/platemap above, a NaN-patient row still belongs on this patient's
# plate -- dropping it here silently removed every DMSO well from every
# patient's annotation data, which in turn left 8.normalization.py's
# DMSO-based reference-population query matching zero rows. The one
# accepted side effect: such rows still carry a NaN
# Metadata_Biology_TumorType (that column comes from a separate merge keyed
# on the same PatientTumor value), since there's no per-patient value to
# recover for them after the fact.
if patient == "NF0037_T1_CQ1":
    annotation_df = annotation_df.loc[
        (annotation_df["Metadata_Biology_PatientTumor"] == "NF0037_T1")
        | annotation_df["Metadata_Biology_PatientTumor"].isna()
    ]
else:
    annotation_df = annotation_df.loc[
        (annotation_df["Metadata_Biology_PatientTumor"] == patient)
        | annotation_df["Metadata_Biology_PatientTumor"].isna()
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
).resolve()
# Not required: datasets with no deep-learning features (e.g. ZEDProfiler) never
# have nucleocentric_*_related.parquet inputs, so step 5 doesn't write this file
# for those patients at all -- treat its absence as "no nucleocentric data" for
# this patient rather than a hard failure.
has_nucleocentric = nucleocentric_merged_path.exists()

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
nucleocentric_merged = (
    pd.read_parquet(nucleocentric_merged_path) if has_nucleocentric else None
)

sc_merged["Well"] = sc_merged["image_set"].str.split("-").str[0]
organoid_merged["Well"] = organoid_merged["image_set"].str.split("-").str[0]
if has_nucleocentric:
    nucleocentric_merged["Well"] = (
        nucleocentric_merged["image_set"].str.split("-").str[0]
    )


# In[6]:


# ZEDProfiler-derived input already carries its own Metadata_Biology_*/
# Metadata_Experiment_* columns, passed straight through from steps 3/5 --
# confirmed directly against real data: Metadata_Biology_PatientTumor
# already exists in sc_merged/organoid_merged here. Left un-dropped, that
# collides with annotation_df's own same-named column below (neither side
# is the join key, so pandas would otherwise silently suffix both copies
# _x/_y instead of erroring). Drop the input's copy first so annotation_df's
# copy is the sole, authoritative one post-merge -- matching this script's
# original design (written for the older CellProfiler-era pipeline, whose
# own input never carried any of these columns before this merge).
_annotation_overlap_cols = [
    c for c in annotation_df.columns if c != "Metadata_Experiment_Well"
]
sc_merged = sc_merged.drop(
    columns=[c for c in _annotation_overlap_cols if c in sc_merged.columns]
)
organoid_merged = organoid_merged.drop(
    columns=[c for c in _annotation_overlap_cols if c in organoid_merged.columns]
)
if has_nucleocentric:
    nucleocentric_merged = nucleocentric_merged.drop(
        columns=[
            c for c in _annotation_overlap_cols if c in nucleocentric_merged.columns
        ]
    )

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
if has_nucleocentric:
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
if has_nucleocentric:
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
if has_nucleocentric:
    nucleocentric_merged["Metadata_WellNucleocentricCount"] = (
        nucleocentric_merged.groupby("Well")["image_set"].transform("count")
    )


# In[8]:


# Rename straight to the fully category-qualified final names (per
# docs/RFC-2119-Feature-Naming-Convention.md section 2.2) rather than a bare
# name later blanket-prefixed with "Metadata_" below: 7a/7b/7c all expect
# Metadata_Object_ObjectID/ParentOrganoid/OrganoidSingleCellCount and
# Metadata_Experiment_WellFOV specifically, not the flat Metadata_ObjectID/
# Metadata_WellFOV a plain prefix would produce. object_id/ParentOrganoid/
# OrganoidSingleCellCount come from step 3's own output; ParentOrganoid is
# sc/nucleocentric-only and OrganoidSingleCellCount is organoid-only, so
# .rename() is a no-op wherever a given key isn't present in that profile's
# columns. "patient" is the older CellProfiler-era pipeline's own bare
# column name (never present in ZEDProfiler-derived data, which already
# carries Metadata_Biology_PatientTumor natively) -- kept here as a no-op
# for that older input shape rather than removed.
column_rename_mapping = {
    "patient": "Metadata_Biology_PatientTumor",
    "image_set": "Metadata_Experiment_WellFOV",
    "object_id": "Metadata_Object_ObjectID",
    "ParentOrganoid": "Metadata_Object_ParentOrganoid",
    "OrganoidSingleCellCount": "Metadata_Object_OrganoidSingleCellCount",
}

# rename columns for consistency across profiles
sc_merged.rename(columns=column_rename_mapping, inplace=True)
organoid_merged.rename(columns=column_rename_mapping, inplace=True)
if has_nucleocentric:
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


# PatientTumor/ObjectID/WellFOV/ParentOrganoid/OrganoidSingleCellCount/Class
# are no longer listed here -- they already arrive fully category-qualified,
# either via column_rename_mapping above or (Class/Treatment/Dose/Unit/
# Target/TherapeuticCategories) via the annotation_df merge.
metadata_features_list = [
    "Tumor",
    "Well",
]
# prepend "Metadata_" to metadata features
sc_merged = sc_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)
organoid_merged = organoid_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)
if has_nucleocentric:
    nucleocentric_merged = nucleocentric_merged.rename(
        columns={col: f"Metadata_{col}" for col in metadata_features_list}
    )

# add microscope metadata
_microscope_name = "Discover Echo" if "CQ1" not in patient else "Yokogawa CQ1"
_dfs_for_microscope_metadata = [sc_merged, organoid_merged] + (
    [nucleocentric_merged] if has_nucleocentric else []
)
for _df in _dfs_for_microscope_metadata:
    _df["Metadata_MicroscopeType"] = "spinning disk confocal"
    _df["Metadata_MicroscopeName"] = _microscope_name
    _df["Metadata_Magnification"] = "60x"
    _df["Metadata_XResolutionUm"] = 0.101
    _df["Metadata_YResolutionUm"] = 0.101
    _df["Metadata_ZResolutionUm"] = 1.0


# In[12]:


# find duplicate columns and keep one of the duplicates
sc_merged = sc_merged.loc[:, ~sc_merged.columns.duplicated()]
organoid_merged = organoid_merged.loc[:, ~organoid_merged.columns.duplicated()]
if has_nucleocentric:
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

# split the profiles
sc_annotated = sc_merged[sc_metadata_columns + sc_handcrafted_columns]
organoid_annotated = organoid_merged[
    organoid_metadata_columns + organoid_handcrafted_columns
]

# A dataset with no deep-learning features (e.g. ZEDProfiler) has zero columns
# containing "sammed"/"chammi" in every profile type -- not just nucleocentric.
# Each DL-derived output is only built/saved when its own column list is
# non-empty, so `4.qc_profiles`/`5.normalized_profiles`/etc. simply have no
# file for a profile type this dataset never produced (see 8.normalization.py,
# 9.feature_selection.py, 10.aggregation.py, 11.combine_patients.py,
# 12.validate_profiles.py, all updated to skip files that don't exist).
has_sc_sammed = bool(sc_sammed_columns)
has_organoid_sammed = bool(organoid_sammed_columns)
if has_sc_sammed:
    sc_annotated_sammed = sc_merged[sc_metadata_columns + sc_sammed_columns]
if has_organoid_sammed:
    organoid_annotated_sammed = organoid_merged[
        organoid_metadata_columns + organoid_sammed_columns
    ]

has_nucleocentric_sammed = False
has_nucleocentric_morphem = False
if has_nucleocentric:
    nucleocentric_metadata_columns = [
        x for x in nucleocentric_merged.columns if "Metadata" in x
    ]
    nucleocentric_sammed_columns = [
        x for x in nucleocentric_merged.columns if "sammed" in x.lower()
    ]
    nucleocentric_morphem_columns = [
        x for x in nucleocentric_merged.columns if "chammi" in x.lower()
    ]
    has_nucleocentric_sammed = bool(nucleocentric_sammed_columns)
    has_nucleocentric_morphem = bool(nucleocentric_morphem_columns)
    if has_nucleocentric_sammed:
        nucleocentric_sammed_annotated = nucleocentric_merged[
            nucleocentric_metadata_columns + nucleocentric_sammed_columns
        ]
    if has_nucleocentric_morphem:
        nucleocentric_morphem_annotated = nucleocentric_merged[
            nucleocentric_metadata_columns + nucleocentric_morphem_columns
        ]


# In[14]:


# save annotated profiles -- hand-crafted SC/organoid are always produced;
# each deep-learning profile is only written when this dataset actually has
# that feature type (see the has_* flags set above).
sc_annotated.to_parquet(sc_annotated_output_path, index=False)
organoid_annotated.to_parquet(organoid_annotated_output_path, index=False)
if has_sc_sammed:
    sc_annotated_sammed.to_parquet(sammed_annotated_sc_profiles_path, index=False)
else:
    print("No SAMMed3D SC columns found -- skipping sammed_sc_anno.parquet output.")
if has_organoid_sammed:
    organoid_annotated_sammed.to_parquet(
        sammed_annotated_organoid_profiles_path, index=False
    )
else:
    print(
        "No SAMMed3D organoid columns found -- skipping sammed_organoid_anno.parquet output."
    )
if has_nucleocentric_sammed:
    nucleocentric_sammed_annotated.to_parquet(
        nucleocentric_annotated_sammed_output_path, index=False
    )
else:
    print(
        "No nucleocentric data (or no SAMMed3D nucleocentric columns) -- "
        "skipping nucleocentric_sammed_anno.parquet output."
    )
if has_nucleocentric_morphem:
    nucleocentric_morphem_annotated.to_parquet(
        nucleocentric_annotated_morphem_output_path, index=False
    )
else:
    print(
        "No nucleocentric data (or no morphem nucleocentric columns) -- "
        "skipping nucleocentric_morphem_anno.parquet output."
    )


# In[15]:


sc_annotated.head()


# In[16]:


organoid_annotated.head()
