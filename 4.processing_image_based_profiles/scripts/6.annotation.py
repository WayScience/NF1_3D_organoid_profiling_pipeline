#!/usr/bin/env python
# coding: utf-8

# This notebook performs profile annotation.
# The platemap is mapped back to the profile to retain the sample metadata.
#

# In[1]:


import argparse
import pathlib

import pandas as pd

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    patient = args.patient

else:
    patient = "NF0014"


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


# In[4]:


# pathing

sc_merged_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/1.combined_profiles/sc.parquet"
).resolve(strict=True)
organoid_merged_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/1.combined_profiles/organoid.parquet"
).resolve(strict=True)

platemap_path = pathlib.Path(
    f"{root_dir}/data/{patient}/platemap/platemap.csv"
).resolve(strict=True)

# output path
sc_annotated_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/2.annotated_profiles/sc_anno.parquet"
).resolve()
organoid_annotated_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/2.annotated_profiles/organoid_anno.parquet"
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


sc_merged.head()


# In[8]:


organoid_merged.head()


# In[9]:


# save annotated profiles
sc_merged.to_parquet(sc_annotated_output_path, index=False)
organoid_merged.to_parquet(organoid_annotated_output_path, index=False)
