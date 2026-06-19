#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile annotation.
# The platemap is mapped back to the profile to retain the sample metadata.
#

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
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


def annotate_profiles(
    profile_df: pd.DataFrame,
    platemap_df: pd.DataFrame,
    drug_information_df: pd.DataFrame,
    patient: str,
) -> pd.DataFrame:
    """
    Annotate profiles with treatment, dose, and unit information from the platemap.

        Parameters
        ----------
        profile_df : pd.DataFrame
            Profile DataFrame containing image_set information.
            Could be either single-cell or organoid profiles.
        platemap_df : pd.DataFrame
            Platmap DataFrame containing well_position, treatment, dose, and unit.
        drug_information_df : pd.DataFrame
            Drug DataFrame containing drug information.
        patient : str
            Patient ID to annotate the profiles with.

        Returns
        -------
        pd.DataFrame
            Annotated profile DataFrame with additional columns for treatment, dose, and unit.
    """
    platemap_df["Treatment_platemap"] = platemap_df["Treatment"]
    platemap_df["merging_string_treatment"] = (
        platemap_df["Treatment_platemap"].str.split().str[0]
    )
    drug_information_platemap_merged = (
        pd.merge(
            platemap_df[
                ["WellPosition", "Treatment_platemap", "merging_string_treatment"]
            ],
            drug_information_df,
            left_on="merging_string_treatment",
            right_on="Treatment",
        )
        .drop(columns=["merging_string_treatment", "Treatment"])
        .rename(columns={"Treatment_platemap": "Treatment"})
    )

    profile_df["Well"] = profile_df["image_set"].str.split("-").str[0]
    profile_df.insert(2, "Well", profile_df.pop("Well"))

    profile_df = profile_df.merge(
        drug_information_platemap_merged,
        how="left",
        left_on="Well",
        right_on="WellPosition",
    )
    profile_df.drop(columns=["WellPosition"], inplace=True)
    for col in ["Treatment"]:
        profile_df.insert(1, col, profile_df.pop(col))
    profile_df.insert(0, "patient", patient)
    return profile_df


# ## pathing

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
platemap_path = pathlib.Path(
    f"{root_dir}/config/platemaps/{patient}_platemap.csv"
).resolve(strict=True)
drug_information = pd.read_csv(
    pathlib.Path(f"{root_dir}/config/drug_information/drug_information.csv")
)
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
nucleocentric_annotated_chammi_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/3.annotated_profiles/nucleocentric_chammi_anno.parquet"
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
# read platemap
platemap = pd.read_csv(platemap_path)
platemap["Treatment"] = (
    platemap["Treatment"]
    + " "
    + platemap["Dose"].astype(str)
    + " "
    + platemap["Unit"].astype(str)
)
# if % is in Treatment then delete the space leading to %
platemap["Treatment"] = platemap["Treatment"].str.replace(r"\s+%", "%", regex=True)
platemap.drop(columns=["Unit", "Dose"], inplace=True)
platemap.head()


# In[6]:


sc_merged = annotate_profiles(
    profile_df=sc_merged,
    platemap_df=platemap,
    drug_information_df=drug_information,
    patient=patient,
)
organoid_merged = annotate_profiles(
    profile_df=organoid_merged,
    platemap_df=platemap,
    drug_information_df=drug_information,
    patient=patient,
)
nucleocentric_merged = annotate_profiles(
    profile_df=nucleocentric_merged,
    platemap_df=platemap,
    drug_information_df=drug_information,
    patient=patient,
)
# remove redundant columns
columns_to_drop = [
    col for col in sc_merged.columns if "image_set_1" in col or "image_set_2" in col
]
sc_merged.drop(columns=columns_to_drop, inplace=True)


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
) = (0.106, 0.106, 0.106)
(
    sc_merged["Metadata_YResolutionUm"],
    organoid_merged["Metadata_YResolutionUm"],
    nucleocentric_merged["Metadata_YResolutionUm"],
) = (0.106, 0.106, 0.106)
(
    sc_merged["Metadata_ZResolutionUm"],
    organoid_merged["Metadata_ZResolutionUm"],
    nucleocentric_merged["Metadata_ZResolutionUm"],
) = (1.0, 1.0, 1.0)


# In[12]:


# Categorize the metadata features
# Biology, Experiment, Image, Object, Microscopy,
biology_features = [
    "Metadata_PatientTumor",
    "Metadata_Patient",
    "Metadata_Tumor",
]
experiment_features = [
    "Metadata_Treatment",
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
    "Metadata_WellOrganoidCount",
    "Metadata_OrganoidSingleCellCount",
    "Metadata_WellNucleocentricCount",
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


# In[13]:


# split the sc_merged_data into hand crafted and sammed features
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
nucleocentric_chammi75_columns = [
    x for x in nucleocentric_merged.columns if "chammi75" in x.lower()
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
nucleocentric_chammi_annotated = nucleocentric_merged[
    nucleocentric_metadata_columns + nucleocentric_chammi75_columns
]


# In[14]:


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
nucleocentric_chammi_annotated.to_parquet(
    nucleocentric_annotated_chammi_output_path, index=False
)


# In[15]:


sc_annotated.head()


# In[16]:


organoid_annotated.head()
