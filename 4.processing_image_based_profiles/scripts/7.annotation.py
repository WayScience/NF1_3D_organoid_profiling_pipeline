#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile annotation.
# The platemap is mapped back to the profile to retain the sample metadata.
#

# In[1]:


import os
import pathlib

import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[ ]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


def annotate_profiles(
    profile_df: pd.DataFrame, platemap_df: pd.DataFrame, patient: str
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
        patient : str
            Patient ID to annotate the profiles with.

        Returns
        -------
        pd.DataFrame
            Annotated profile DataFrame with additional columns for treatment, dose, and unit.
    """
    drug_information = pd.read_csv(
        pathlib.Path(
            f"{root_dir}/4.processing_image_based_profiles/data/drugs/drug_information.csv"
        )
    )
    profile_df["Well"] = profile_df["image_set"].str.split("-").str[0]
    profile_df.insert(2, "Well", profile_df.pop("Well"))
    profile_df = pd.merge(
        profile_df,
        platemap_df[["well_position", "treatment", "dose", "unit"]],
        left_on="Well",
        right_on="well_position",
        how="left",
    ).drop(columns=["well_position"])
    profile_df = profile_df.merge(
        drug_information, how="left", left_on="treatment", right_on="Treatment"
    )
    profile_df.drop(columns=["Treatment"], inplace=True)
    for col in ["treatment", "dose", "unit"]:
        profile_df.insert(1, col, profile_df.pop(col))
    profile_df.insert(0, "patient", patient)
    return profile_df


# ## pathing

# In[4]:


sc_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/1.combined_profiles/sc.parquet"
).resolve(strict=True)
organoid_merged_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/1.combined_profiles/organoid.parquet"
).resolve(strict=True)

platemap_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/platemap/platemap.csv"
).resolve(strict=True)

# output path
sc_annotated_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.annotated_profiles/sc_anno.parquet"
).resolve()
organoid_annotated_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.annotated_profiles/organoid_anno.parquet"
).resolve()

organoid_annotated_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[5]:


# read data
sc_merged = pd.read_parquet(sc_merged_path)
organoid_merged = pd.read_parquet(organoid_merged_path)
# read platemap
platemap = pd.read_csv(platemap_path)
platemap.head()


# In[6]:


sc_merged = annotate_profiles(sc_merged, platemap, patient)
organoid_merged = annotate_profiles(organoid_merged, platemap, patient)


# In[7]:


sc_merged.rename(columns={"patient": "patient_tumor"}, inplace=True)
organoid_merged.rename(columns={"patient": "patient_tumor"}, inplace=True)
sc_merged[["patient", "tumor"]] = sc_merged["patient_tumor"].str.split("_", expand=True)
organoid_merged[["patient", "tumor"]] = organoid_merged["patient_tumor"].str.split(
    "_", expand=True
)


# In[8]:


metadata_features_list = [
    "patient_tumor",
    "patient",
    "tumor",
    "object_id",
    "unit",
    "dose",
    "Well",
    "treatment",
    "image_set",
    "parent_organoid",
    "single_cell_count",
    "Target",
    "Class",
    "Therapeutic_Categories",
]
# prepend "Metadata_" to metadata features
sc_merged = sc_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)
organoid_merged = organoid_merged.rename(
    columns={col: f"Metadata_{col}" for col in metadata_features_list}
)


# In[9]:


sc_merged.head()


# In[10]:


organoid_merged.head()


# In[11]:


# save annotated profiles
sc_merged.to_parquet(sc_annotated_output_path, index=False)
organoid_merged.to_parquet(organoid_annotated_output_path, index=False)
