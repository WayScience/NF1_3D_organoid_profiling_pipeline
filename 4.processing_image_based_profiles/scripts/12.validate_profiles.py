#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import numpy as np
import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
profile_base_dir = root_dir


# In[2]:


patient_ids_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patient_ids = pd.read_csv(patient_ids_file_path, header=None).iloc[:, 0].tolist()


# In[6]:


patient_ids = ["NF0014_T1"]


# In[7]:


for patient in patient_ids:
    print(
        f"==================================================\nProcessing patient: {patient}\n=================================================="
    )
    # output path
    sc_normalized_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sc_norm.parquet"
    ).resolve()
    organoid_normalized_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/organoid_norm.parquet"
    ).resolve()
    sc_sammed_normalized_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sammed_sc_norm.parquet"
    ).resolve()
    organoid_sc_sammed_normalized_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sammed_organoid_norm.parquet"
    ).resolve()
    nucleocentric_sammed_normalized_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sammed_nucleocentric_norm.parquet"
    ).resolve()
    nucleocentric_morphem_normalized_path = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/nucleocentric_morphem_norm.parquet"
    ).resolve()

    sc_annotated_df = pd.read_parquet(sc_normalized_path)
    organoid_annotated_df = pd.read_parquet(organoid_normalized_path)
    sc_sammed_annotated_df = pd.read_parquet(sc_sammed_normalized_path)
    organoid_sc_sammed_annotated_df = pd.read_parquet(
        organoid_sc_sammed_normalized_path
    )
    nucleocentric_sammed_annotated_df = pd.read_parquet(
        nucleocentric_sammed_normalized_path
    )
    nucleocentric_morphem_annotated_df = pd.read_parquet(
        nucleocentric_morphem_normalized_path
    )

    # check for duplicates, NAs, infs, and get counts for each annotated df
    for df, name in zip(
        [
            sc_annotated_df,
            organoid_annotated_df,
            sc_sammed_annotated_df,
            organoid_sc_sammed_annotated_df,
            nucleocentric_sammed_annotated_df,
            nucleocentric_morphem_annotated_df,
        ],
        [
            "sc_annotated_df",
            "organoid_annotated_df",
            "sc_sammed_annotated_df",
            "organoid_sc_sammed_annotated_df",
            "nucleocentric_sammed_annotated_df",
            "nucleocentric_morphem_annotated_df",
        ],
    ):
        print(f"{name} - shape: {df.shape}")
        if df.duplicated().sum() > 0:
            print(f"{name} - duplicates: {df.duplicated().sum()}")
        if df.isna().sum().sum() > 0:
            print(f"{name} - NAs: {df.isna().sum().sum()}")
        if np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0:
            print(
                f"{name} - infs: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}"
            )
