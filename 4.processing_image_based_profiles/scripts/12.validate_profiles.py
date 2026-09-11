#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
from datetime import datetime

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


threshold = 1e5


# In[3]:


patient_ids_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patient_ids = pd.read_csv(patient_ids_file_path, header=None).iloc[:, 0].tolist()


# In[ ]:


# set up log file
log_dir = pathlib.Path("../logs/profile_validation").resolve()
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"profile_check_{datetime.now():%Y%m%d_%H}.log"


def log(msg, log_file):
    """Print to stdout and write to the log file."""
    print(msg)
    log_file.write(msg + "\n")


with open(log_path, "w") as log_file:
    for patient in patient_ids:
        # construct the profile path dict for this patient
        profile_path_dict = {
            "sc_normalized": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sc_norm.parquet"
            ).resolve(strict=True),
            "organoid_normalized": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/organoid_norm.parquet"
            ).resolve(strict=True),
            "sc_sammed_normalized": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sammed_sc_norm.parquet"
            ).resolve(strict=True),
            "organoid_sammed_normalized": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sammed_organoid_norm.parquet"
            ).resolve(strict=True),
            "nucleocentric_sammed_normalized": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/sammed_nucleocentric_norm.parquet"
            ).resolve(strict=True),
            "nucleocentric_morphem_normalized": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/5.normalized_profiles/nucleocentric_morphem_norm.parquet"
            ).resolve(strict=True),
            "sc_fs": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/6.feature_selected_profiles/sc_fs.parquet"
            ).resolve(strict=True),
            "organoid_fs": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/6.feature_selected_profiles/organoid_fs.parquet"
            ).resolve(strict=True),
            "sc_sammed_fs": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/6.feature_selected_profiles/sammed_sc_fs.parquet"
            ).resolve(strict=True),
            "organoid_sammed_fs": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/6.feature_selected_profiles/sammed_organoid_fs.parquet"
            ).resolve(strict=True),
            "nucleocentric_sammed_fs": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/6.feature_selected_profiles/sammed_nucleocentric_fs.parquet"
            ).resolve(strict=True),
            "nucleocentric_morphem_fs": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/6.feature_selected_profiles/nucleocentric_morphem_fs.parquet"
            ).resolve(strict=True),
            "sc_agg_well": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/7.aggregated_profiles/sc_agg_well_level.parquet"
            ).resolve(strict=True),
            "organoid_agg_well": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/7.aggregated_profiles/organoid_agg_well_level.parquet"
            ).resolve(strict=True),
            "organoid_sammed_agg_well": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/7.aggregated_profiles/sammed_organoid_agg_well_level.parquet"
            ).resolve(strict=True),
            "sc_sammed_agg_well": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/7.aggregated_profiles/sammed_sc_agg_well_level.parquet"
            ).resolve(strict=True),
            "nucleocentric_sammed_agg_well": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/7.aggregated_profiles/sammed_nucleocentric_agg_well_level.parquet"
            ).resolve(strict=True),
            "nucleocentric_morphem_agg_well": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/7.aggregated_profiles/nucleocentric_morphem_agg_well_level.parquet"
            ).resolve(strict=True),
            "sc_agg_treatment": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/8.consensus_profiles/sc_consensus.parquet"
            ).resolve(strict=True),
            "organoid_agg_treatment": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/8.consensus_profiles/organoid_consensus.parquet"
            ).resolve(strict=True),
            "organoid_sammed_agg_treatment": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/8.consensus_profiles/sammed_organoid_consensus.parquet"
            ).resolve(strict=True),
            "sc_sammed_agg_treatment": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/8.consensus_profiles/sammed_sc_consensus.parquet"
            ).resolve(strict=True),
            "nucleocentric_sammed_agg_treatment": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/8.consensus_profiles/sammed_nucleocentric_consensus.parquet"
            ).resolve(strict=True),
            "nucleocentric_morphem_agg_treatment": pathlib.Path(
                f"{profile_base_dir}/data/{patient}/image_based_profiles/8.consensus_profiles/nucleocentric_morphem_consensus.parquet"
            ).resolve(strict=True),
        }

        log(f"\n{'=' * 60}", log_file)
        log(f"Processing patient: {patient}", log_file)
        log(f"{'=' * 60}", log_file)
        log(
            f"{'name':40} | {'shape':>15} | {'nans':>8} | {'infs':>8} | {'dupes':>8} | {'values_above_threshold':>8}",
            log_file,
        )
        log(
            f"{'-' * 40} | {'-' * 15} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8}",
            log_file,
        )

        for name, path in profile_path_dict.items():
            df = pd.read_parquet(path)
            nas = df.isna().sum().sum()
            infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            duplicates = df.duplicated().sum()
            values_above_threshold = (
                (df.select_dtypes(include=[np.number]) > threshold).sum().sum()
            )
            shape = df.shape

            log(
                f"{name:40} | {str(shape):>15} | {nas:>8} | {infs:>8} | {duplicates:>8} | {values_above_threshold:>8}",
                log_file,
            )

print(f"\nLog written to: {log_path}")


# In[ ]:
