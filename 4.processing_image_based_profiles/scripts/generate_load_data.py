#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from tqdm.auto import tqdm

root_dir, in_notebook = init_notebook()
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[2]:


patients_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patients_file_path,
    header=None,
    names=["patient_id"],
).patient_id.tolist()

load_file_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/load_data/load_file.txt"
).resolve()
load_file_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


rows = []

for patient in tqdm(patients, desc="Patients"):
    extracted_features_dir = profile_base_dir / "data" / patient / "extracted_features"
    if not extracted_features_dir.exists():
        print(f"No extracted_features directory for patient {patient}; skipping.")
        continue

    well_fovs = sorted(
        d.name
        for d in extracted_features_dir.iterdir()
        if d.is_dir() and "run_stats" not in d.name
    )
    for well_fov in well_fovs:
        rows.append({"patient": patient, "well_fov": well_fov})

df = pd.DataFrame(rows)
print(f"Total patient/well_fov combinations: {df.shape[0]}")


# In[4]:


# Build expected .duckdb paths with vectorized string ops
df["file_path"] = (
    profile_base_dir.as_posix()
    + "/data/"
    + df["patient"]
    + "/image_based_profiles/0.converted_profiles/"
    + df["well_fov"]
    + "/"
    + df["well_fov"]
    + ".duckdb"
)

# Scan each converted_profiles directory once — faster than per-row .exists()
existing_duckdbs: set[str] = set()
candidate_dir_df = df[["patient", "well_fov"]].drop_duplicates()

for patient, well_fov in tqdm(
    candidate_dir_df.itertuples(index=False, name=None),
    total=len(candidate_dir_df),
    desc="Scanning converted_profiles directories",
):
    profile_dir = (
        profile_base_dir
        / "data"
        / patient
        / "image_based_profiles"
        / "0.converted_profiles"
        / well_fov
    )
    if profile_dir.exists():
        existing_duckdbs.update(
            p.as_posix() for p in profile_dir.glob("*.duckdb") if p.is_file()
        )

df["exists"] = df["file_path"].isin(existing_duckdbs)

total = len(df)
present = df["exists"].sum()
print(f"Total files to check : {total}")
print(f"Present              : {present}")
print(f"Missing              : {total - present}")


# In[5]:


# Write missing combinations to load_file.txt
df_missing = df.loc[~df["exists"], ["patient", "well_fov"]].reset_index(drop=True)
df_missing.to_csv(load_file_path, sep="\t", index=False, header=False)
print(f"Wrote {len(df_missing)} missing combinations to: {load_file_path}")


# In[6]:


# Summary by patient
summary_df = (
    df.groupby("patient")[["exists"]]
    .agg(total=("exists", "count"), present=("exists", "sum"))
    .assign(missing=lambda x: x["total"] - x["present"])
)
summary_df.reset_index()
