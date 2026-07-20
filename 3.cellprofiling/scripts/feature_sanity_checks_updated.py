#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import itertools
import os
import pathlib
import sys
from functools import reduce

import duckdb
import pandas as pd
import tomli
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir  # default to root_dir instead of NAS


# In[2]:


patient_id_file = pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()


# In[3]:


out_dict = {
    "file_path": [],
    "patient_id": [],
    "well_fov": [],
    "feature_type": [],
    "compartment": [],
    # "df_shape": [],
}

# get all well_fovs for a patient
for patient in tqdm.tqdm(patients, desc="Processing patients", leave=True):
    patient_dir = profile_base_dir / "data" / patient / "extracted_features"
    well_fovs = patient_dir.glob("*")  # get all well_fovs for a patient
    # print(f"Found well_fovs: {well_fovs}")
    for well_fov in tqdm.tqdm(well_fovs, desc="Processing well_fovs", leave=False):
        if "stats" in well_fov.stem:
            continue
        features = pathlib.Path(well_fov).glob("*.parquet")
        for feature in features:
            feature_type = feature.stem.split("_")[2]
            compartment = feature.stem.split("_")[0]
            out_dict["file_path"].append(feature)
            out_dict["patient_id"].append(patient)
            out_dict["well_fov"].append(feature.parent.stem)
            out_dict["feature_type"].append(feature_type)
            out_dict["compartment"].append(compartment)
            # out_dict["df_shape"].append(pd.read_parquet(feature).shape)
df = pd.DataFrame(out_dict)
# df = df.loc[df["patient_id"] == "NF0014_T1"]

# NOTE: previously this also filtered to feature_type == "Granularity", which
# silently dropped every other feature type from `df`. Removed so that `df`
# contains every feature-type file per compartment/well_fov -- needed for the
# "alignment across feature types within a compartment" check below.
df = df.loc[(df["compartment"] != "Organoid")]

df.head()


# In[4]:


from tqdm import tqdm

tqdm.pandas()


def safe_read_shape(x):
    try:
        df = pd.read_parquet(x)
        return df.shape, df.isna().sum().sum()
    except Exception as e:
        print(f"Error reading {x}: {e}")
        return None, None


# if not pathlib.Path("../logs/feature_file_info.parquet").exists():
df[["df_shape", "missing_values"]] = df["file_path"].progress_apply(
    lambda x: pd.Series(safe_read_shape(x))
)
df["file_path"] = df["file_path"].astype(str)
df.to_parquet("../logs/feature_file_info.parquet", index=False)
# else:
#     df = pd.read_parquet("../logs/feature_file_info.parquet")


# In[5]:


df.sort_values(["patient_id", "well_fov"], inplace=True)
df.reset_index(drop=True, inplace=True)
df


# In[6]:


# merge the cells, cytoplasm, and whole cell features for a given well_fov and patient_id
# check for missing values and shape of the dataframes
out_dict = {
    "patient_id": [],
    "well_fov": [],
    "path": [],
    "type": [],
    "feature_type": [],
}
for row in tqdm(
    df.itertuples(), total=df.shape[0], desc="Merging features", leave=True
):
    out_dict["patient_id"].append(row.patient_id)
    out_dict["well_fov"].append(row.well_fov)
    out_dict["path"].append(row.file_path)
    out_dict["type"].append(f"{row.compartment}")
    out_dict["feature_type"].append(row.feature_type)
out_df = pd.DataFrame(out_dict)
out_df.drop_duplicates(subset=["patient_id", "well_fov", "type"], inplace=True)
# pivot such that each type has its own column
out_df = out_df.pivot(
    index=[
        "patient_id",
        "well_fov",
    ],
    columns="type",
    values="path",
).reset_index()


# In[7]:


out_df


# In[8]:


labels_dict = {
    "Cell_labels": [],
    "Cytoplasm_labels": [],
    "Nuclei_labels": [],
    "patient_id": [],
    "well_fov": [],
}

# merge the dataframes and check for missing values and shape
for row in tqdm(
    out_df.itertuples(),
    total=out_df.shape[0],
    desc="Checking merged features",
    leave=True,
):
    try:
        cell_df = pd.read_parquet(row.Cell)
        cytoplasm_df = pd.read_parquet(row.Cytoplasm)
        nuclei_df = pd.read_parquet(row.Nuclei)
        labels_dict["Cell_labels"].append(cell_df["object_id"].tolist())
        labels_dict["Cytoplasm_labels"].append(cytoplasm_df["object_id"].tolist())
        labels_dict["Nuclei_labels"].append(nuclei_df["object_id"].tolist())
        labels_dict["patient_id"].append(row.patient_id)
        labels_dict["well_fov"].append(row.well_fov)
    except Exception as e:
        print(f"Error reading files for {row.patient_id} {row.well_fov}: {e}")
labels_df = pd.DataFrame(labels_dict)


# In[12]:


labels_df["labels_match"] = labels_df.apply(
    lambda row: set(row["Cell_labels"]) == set(row["Nuclei_labels"]),
    axis=1,
)
labels_df["same_number_of_labels"] = labels_df.apply(
    lambda row: len(row["Cell_labels"]) == len(row["Nuclei_labels"]),
    axis=1,
)

# Full pairwise, set-based (order no longer matters) object_id differences
# across all three compartments -- previously only three of the six possible
# one-directional differences were computed, and two of those were duplicates
# of each other, so a mismatch could exist without ever showing up.
labels_df["cell_only"] = labels_df.apply(
    lambda row: (
        set(row["Cell_labels"])
        - set(row["Cytoplasm_labels"])
        - set(row["Nuclei_labels"])
    ),
    axis=1,
)
labels_df["cytoplasm_only"] = labels_df.apply(
    lambda row: (
        set(row["Cytoplasm_labels"])
        - set(row["Cell_labels"])
        - set(row["Nuclei_labels"])
    ),
    axis=1,
)
labels_df["nuclei_only"] = labels_df.apply(
    lambda row: (
        set(row["Nuclei_labels"])
        - set(row["Cell_labels"])
        - set(row["Cytoplasm_labels"])
    ),
    axis=1,
)

# Single well_fov-level flag: do Cell, Cytoplasm, and Nuclei all agree on the
# exact same set of object_ids? This is the authoritative "aligned across
# compartments" answer -- use this instead of `labels_match`, which only ever
# compared Cell to Nuclei and ignored Cytoplasm entirely.
labels_df["compartments_object_ids_aligned"] = labels_df.apply(
    lambda row: (
        set(row["Cell_labels"])
        == set(row["Cytoplasm_labels"])
        == set(row["Nuclei_labels"])
    ),
    axis=1,
)
print(labels_df.loc[labels_df["compartments_object_ids_aligned"] == False].shape)
labels_df.loc[labels_df["compartments_object_ids_aligned"] == False].head(20)


# In[ ]:


# In[10]:


# get the patient_id and well_fov for the rows where the compartments'
# object_ids do not align, and print the corresponding cleanup commands
mismatched_labels = labels_df.loc[
    labels_df["compartments_object_ids_aligned"] == False, ["patient_id", "well_fov"]
]
mismatched_labels
for row in tqdm(
    mismatched_labels.itertuples(),
    total=mismatched_labels.shape[0],
    desc="Checking mismatched labels",
    leave=True,
):
    patient_id = row.patient_id
    well_fov = row.well_fov
    print(f"cd ../../{patient_id}/segmentation_masks/ ; rm -r {well_fov}")
    print(f"cd ../../{patient_id}/extracted_features/ ; rm -r {well_fov}")

    continue


# In[11]:


labels_df.loc[labels_df["same_number_of_labels"] == False].value_counts("patient_id")


# ## Object-id alignment across feature-type files within the same compartment
#
# The checks above only ever look at **one** feature file per compartment (whichever one the earlier pivot happened to keep). A compartment can have several feature-type files (`Cell_AreaShape.parquet`, `Cell_Texture.parquet`, `Cell_Granularity.parquet`, ...) that are all supposed to describe the exact same set of objects for a given well_fov. If one of those feature-extraction steps drops or gains objects relative to the others, none of the checks above would ever catch it. This section checks that directly.

# In[ ]:


# --- Object-level alignment WITHIN a compartment, across its different
# feature-type files (e.g. Cell_AreaShape vs Cell_Texture vs Cell_Granularity
# for the same patient_id/well_fov should all report the same object_ids) ---
feature_type_alignment_records = []
import tqdm.notebook as tqdm

for (patient_id, well_fov, compartment), group in tqdm.tqdm(
    df.groupby(["patient_id", "well_fov", "compartment"]),
    desc="Checking object_id alignment across feature types",
):
    object_id_sets = {}
    for row in group.itertuples():
        try:
            obj_ids = set(
                pd.read_parquet(row.file_path, columns=["object_id"])["object_id"]
            )
            object_id_sets[row.feature_type] = obj_ids
        except Exception as e:
            print(f"Error reading {row.file_path}: {e}")
            object_id_sets[row.feature_type] = None

    valid = {ft: s for ft, s in object_id_sets.items() if s is not None}
    unique_sets = {frozenset(s) for s in valid.values()}
    aligned = len(unique_sets) <= 1

    mismatched_feature_types = []
    if not aligned:
        # treat the most common object_id set as the reference; anything
        # that differs from it is flagged as the odd one out
        counts = pd.Series([frozenset(s) for s in valid.values()]).value_counts()
        reference = counts.index[0]
        mismatched_feature_types = [
            ft for ft, s in valid.items() if frozenset(s) != reference
        ]

    feature_type_alignment_records.append(
        {
            "patient_id": patient_id,
            "well_fov": well_fov,
            "compartment": compartment,
            "n_feature_types_checked": len(valid),
            "object_ids_aligned": aligned,
            "mismatched_feature_types": mismatched_feature_types,
        }
    )

feature_type_alignment_df = pd.DataFrame(feature_type_alignment_records)
feature_type_alignment_df.loc[feature_type_alignment_df["object_ids_aligned"] == False]


# In[ ]:


# Roll up to one row per well_fov (per your request, well_fov level is
# enough -- no need to list individual file paths): flagged if ANY
# compartment's feature-type files disagree with each other for that well_fov
def _summarize(group):
    misaligned = group.loc[~group["object_ids_aligned"], "compartment"].tolist()
    return pd.Series(
        {
            "any_feature_type_misaligned": len(misaligned) > 0,
            "misaligned_compartments": misaligned,
        }
    )


feature_type_alignment_by_well_fov = (
    feature_type_alignment_df.groupby(["patient_id", "well_fov"])
    .apply(_summarize)
    .reset_index()
)
feature_type_alignment_by_well_fov.loc[
    feature_type_alignment_by_well_fov["any_feature_type_misaligned"]
]


# ## Combined well_fov-level summary
#
# One table, one row per well_fov, flagged `True` if it fails **either** check: the three compartments disagree with each other (`compartments_object_ids_aligned`), or a compartment's own feature-type files disagree with each other (`any_feature_type_misaligned`).

# In[ ]:


combined_alignment = labels_df[
    ["patient_id", "well_fov", "compartments_object_ids_aligned"]
].merge(
    feature_type_alignment_by_well_fov[
        ["patient_id", "well_fov", "any_feature_type_misaligned"]
    ],
    on=["patient_id", "well_fov"],
    how="outer",
)
combined_alignment["needs_review"] = (
    combined_alignment["compartments_object_ids_aligned"] == False
) | (combined_alignment["any_feature_type_misaligned"] == True)
combined_alignment.loc[combined_alignment["needs_review"]]


# In[ ]:


# parse through all feature files and find any files that have nas for all of a column


# In[ ]:


patient_ids = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    names=["patient_id"],
).patient_id.tolist()


# In[ ]:


import tqdm.notebook as tqdm

all_nans = []
for patient in tqdm.tqdm(
    patient_ids, desc="Checking for missing values in features", leave=True
):
    patient_dir = profile_base_dir / "data" / patient / "extracted_features"
    well_fovs = patient_dir.glob("*")  # get all well_fovs for a patient
    for well_fov in tqdm.tqdm(
        well_fovs, desc=f"Checking {patient} for missing values", leave=False
    ):
        if "stats" in well_fov.stem:
            continue
        features = pathlib.Path(well_fov).glob("*.parquet")
        for feature in features:
            if "sam" not in feature.stem.lower():
                continue
            try:
                df = pd.read_parquet(feature)
                if df.isna().all().any():
                    all_nans.append(feature)
            except Exception as e:
                print(f"Error reading {feature}: {e}")


# In[ ]:


all_nans


# In[ ]:


all_nans[0]
