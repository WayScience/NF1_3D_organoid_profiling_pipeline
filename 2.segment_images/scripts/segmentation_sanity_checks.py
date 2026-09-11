#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import multiprocessing
import os
import pathlib

import numpy as np
import pandas as pd
import tifffile
import tomli
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.auto as tqdm
else:
    import tqdm

from tqdm import tqdm as tqdm_main

bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot"))
bandicoot_mount_path = bandicoot_check(bandicoot_mount_path, root_dir)


# In[2]:


def normalize_mask_labels(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, np.ndarray):
        return val.flatten().tolist()  # flatten handles 0-d arrays
    if isinstance(val, list):
        return val
    # scalar int (or anything else) → wrap it
    return [val]


def safe_image_read(path: pathlib.Path):
    try:
        return tifffile.imread(path)
    except Exception as e:
        return None


# In[3]:


patient_id_file = pathlib.Path(f"{bandicoot_mount_path}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()

load_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/load_combinations.txt"
)
load_combinations_path.parent.mkdir(parents=True, exist_ok=True)

sanity_df_save_path = pathlib.Path(
    f"{root_dir}/2.segment_images/results/segmentation_sanity_checks_df.parquet"
).resolve()
sanity_df_save_path_id_checks = pathlib.Path(
    f"{root_dir}/2.segment_images/results/segmentation_sanity_checks_df_id_checks.parquet"
).resolve()
sanity_df_save_path.parent.mkdir(parents=True, exist_ok=True)

channel_mapping_file_path = pathlib.Path(
    f"{root_dir}/config/channel_mapping.toml"
).resolve(strict=True)
with open(channel_mapping_file_path, "rb") as f:
    channel_mapping_dict = tomli.load(f)
channel_n_compartment_mapping = channel_mapping_dict["channel_mapping"]

channels = ["DNA", "ER", "Mito", "AGP"]
compartments = ["Organoid", "Nuclei", "Cytoplasm", "Cell"]


# In[4]:


final_dict = {
    "patient": [],
    "well_fov": [],
    "nuclei_mask_path": [],
    "cytoplasm_mask_path": [],
    "cell_mask_path": [],
    "organoid_mask_path": [],
}
for patient in tqdm.tqdm(patients, desc="Patients", unit="patient", leave=True):
    patient_well_fovs = sorted(
        [
            path.name
            for path in (
                bandicoot_mount_path / "data" / patient / "zstack_images"
            ).glob("*")
            if path.is_dir()
        ]
    )
    for well_fov in tqdm.tqdm(
        patient_well_fovs, desc="Well/FOV", unit="well_fov", leave=False
    ):
        mask_path = (
            bandicoot_mount_path / "data" / patient / "segmentation_masks" / well_fov
        )
        final_dict["patient"].append(patient)
        final_dict["well_fov"].append(well_fov)
        final_dict["nuclei_mask_path"].append(mask_path / "nuclei_mask.tiff")
        final_dict["cytoplasm_mask_path"].append(mask_path / "cytoplasm_mask.tiff")
        final_dict["cell_mask_path"].append(mask_path / "cell_mask.tiff")
        final_dict["organoid_mask_path"].append(mask_path / "organoid_mask.tiff")

df = pd.DataFrame.from_dict(final_dict)
df["nuclei_mask_path"] = df["nuclei_mask_path"].astype(str)
df["cytoplasm_mask_path"] = df["cytoplasm_mask_path"].astype(str)
df["cell_mask_path"] = df["cell_mask_path"].astype(str)
df["organoid_mask_path"] = df["organoid_mask_path"].astype(str)
df.to_parquet(sanity_df_save_path, index=False)


# In[5]:


OVERWRITE = True  # Set to True to overwrite the sanity check dataframe for ID checks
if sanity_df_save_path_id_checks.exists() and not OVERWRITE:
    print(
        f"Sanity check dataframe for ID checks already exists at {sanity_df_save_path_id_checks}. Set OVERWRITE = True to overwrite."
    )
    df = pd.read_parquet(sanity_df_save_path_id_checks)
else:
    from tqdm.auto import tqdm

    tqdm.pandas()

    df["nuclei_labels"] = df["nuclei_mask_path"].progress_apply(
        lambda x: np.unique(safe_image_read(x))
    )
    df["cell_labels"] = df["cell_mask_path"].progress_apply(
        lambda x: np.unique(safe_image_read(x))
    )
    df["cytoplasm_labels"] = df["cytoplasm_mask_path"].progress_apply(
        lambda x: np.unique(safe_image_read(x))
    )
    df["organoid_labels"] = df["organoid_mask_path"].progress_apply(
        lambda x: np.unique(safe_image_read(x))
    )
    # idientify well fovs where the number of unique labels in nuclei, cytoplasm, cell do not match.
    # This is a sanity check to ensure that the segmentation masks are consistent across compartments.
    df["nuclei_cell_mismatch"] = df.apply(
        lambda row: (set(row["nuclei_labels"]) - set(row["cell_labels"])) != set(),
        axis=1,
    )
    df["nuclei_cytoplasm_mismatch"] = df.apply(
        lambda row: (set(row["nuclei_labels"]) - set(row["cytoplasm_labels"])) != set(),
        axis=1,
    )
    df["cell_cytoplasm_mismatch"] = df.apply(
        lambda row: (set(row["cell_labels"]) - set(row["cytoplasm_labels"])) != set(),
        axis=1,
    )
    df["mistmatch_present_across_compartments"] = df.apply(
        lambda row: (
            row["nuclei_cell_mismatch"]
            or row["nuclei_cytoplasm_mismatch"]
            or row["cell_cytoplasm_mismatch"]
        ),
        axis=1,
    )
    mismatch_present = df.pop("mistmatch_present_across_compartments")
    df.insert(0, "mistmatch_present_across_compartments", mismatch_present)
    for col in ["nuclei_labels", "cell_labels", "cytoplasm_labels", "organoid_labels"]:
        df[col] = df[col].apply(lambda arr: np.asarray(arr).astype(np.int32))

    df.to_parquet(sanity_df_save_path_id_checks, index=False)
    df.head()


# In[10]:


for col in ["nuclei_labels", "cell_labels", "cytoplasm_labels", "organoid_labels"]:

    def has_none_element(arr):
        if arr is None:
            return False
        arr = np.asarray(arr, dtype=object)
        return any(v is None for v in arr.ravel())

    bad = df[df[col].apply(has_none_element)]
    if len(bad):
        print(f"{col}: {len(bad)} rows with None *inside* the array")
        print(bad[col].head())


# In[12]:


def safe_cast(arr):
    if arr is None:
        return np.array([0, 0, 0, 0], dtype=np.int32)
    arr = np.asarray(arr, dtype=object)
    # Replace any None elements with 0 (or filter them out — see note below)
    cleaned = np.array([0 if v is None else v for v in arr.ravel()], dtype=np.int32)
    return cleaned


for col in ["nuclei_labels", "cell_labels", "cytoplasm_labels", "organoid_labels"]:
    df[col] = df[col].apply(safe_cast)

df.to_parquet(sanity_df_save_path_id_checks, index=False)


# In[13]:


df.groupby("mistmatch_present_across_compartments").size().reset_index(name="count")


# In[14]:


df.loc[df["mistmatch_present_across_compartments"]]


# In[15]:


rerun_df = df.loc[df["mistmatch_present_across_compartments"]]
for row in rerun_df.itertuples(index=False):
    patient = row.patient
    well_fov = row.well_fov
    print(f"cd ../../{patient}/segmentation_masks/ ; rm -r {well_fov}")


# In[ ]:
