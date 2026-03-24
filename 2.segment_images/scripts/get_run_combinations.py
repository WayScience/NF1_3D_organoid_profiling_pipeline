#!/usr/bin/env python
# coding: utf-8

# # Get run combinations — segmentation (refactor)
#
# Generates `input_combinations.txt` (all jobs) and `rerun_combinations.txt` (jobs missing masks).
# Uses an `add_row` helper to avoid repetitive appends, and scans output directories once for fast existence checks.

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
bandicoot_path = pathlib.Path(
    os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")
).resolve()
image_base_path = bandicoot_check(
    bandicoot_mount_path=bandicoot_path, root_dir=root_dir
)


# In[ ]:


patient_id_file = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()

rerun_combinations_path = pathlib.Path(
    f"{root_dir}/2.segment_images/load_data/load_combinations.txt"
)
rerun_combinations_path.parent.mkdir(parents=True, exist_ok=True)

# Patients that have extra z-spacing variants
z_stack_testing_patients = [
    "NF0037_T1-Z-1",
    "NF0037_T1-Z-0.5",
    "NF0037_T1-Z-0.2",
    "NF0037_T1-Z-0.1",
    "NF0055_T1-Z-0.1",
]
patients += z_stack_testing_patients


# Convolution iteration values used for NF0014_T1 / C4-2
convolution_iters = list(range(1, 26)) + [50, 75, 100]


# In[3]:


rows = []


def add_row(
    patient: str,
    well_fov: str,
    input_subparent_name: str = "zstack_images",
    mask_subparent_name: str = "segmentation_masks",
):
    """Append one segmentation job row.

    Parameters
    ----------
    patient : str
        Patient ID.
    well_fov : str
        Well + field-of-view identifier.
    input_subparent_name : str
        Sub-directory containing input images, by default 'zstack_images'.
    mask_subparent_name : str
        Sub-directory where segmentation masks are written, by default 'segmentation_masks'.
    """
    rows.append(
        {
            "patient": patient,
            "well_fov": well_fov,
            "input_subparent_name": input_subparent_name,
            "mask_subparent_name": mask_subparent_name,
        }
    )


# In[ ]:


for patient in tqdm(patients, desc="Patients"):
    patient_image_dir = image_base_path / "data" / patient / "zstack_images"
    if not patient_image_dir.exists():
        print(f"No zstack_images directory for patient {patient}; skipping.")
        continue

    patient_well_fovs = sorted(
        path.name for path in patient_image_dir.glob("*") if path.is_dir()
    )

    for well_fov in patient_well_fovs:
        # --- standard entry ---
        add_row(patient, well_fov)

        # --- NF0014_T1 / C4-2: convolution iterations + deconvolved images ---
        if patient == "NF0014_T1" and well_fov == "C4-2":
            for conv_iter in convolution_iters:
                add_row(
                    patient,
                    well_fov,
                    input_subparent_name=f"convolution_{conv_iter}",
                    mask_subparent_name=f"convolution_{conv_iter}_segmentation_masks",
                )
            add_row(
                patient,
                well_fov,
                input_subparent_name="deconvolved_images",
                mask_subparent_name="deconvolved_segmentation_masks",
            )

        # --- z-stack spacing test patients: extra mask variant ---
        elif patient in z_stack_testing_patients:
            add_row(
                patient,
                well_fov,
                input_subparent_name="zstack_images",
                mask_subparent_name="segmentation_masks_from_0_1um",
            )

df = pd.DataFrame(rows)
print(f"Total combinations: {df.shape[0]}")
df.head()


# ## Rerun list
#
# A job is considered complete when its mask directory contains at least 4 `.tiff` files.
# Scan each unique mask directory once to build counts, then filter.

# In[5]:


# Scan each unique mask directory once — count existing .tiff files
mask_tiff_counts: dict[str, int] = {}

candidate_dirs = (
    df[["patient", "mask_subparent_name", "well_fov"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)
)

for patient, mask_subparent_name, well_fov in tqdm(
    candidate_dirs,
    total=df[["patient", "mask_subparent_name", "well_fov"]].drop_duplicates().shape[0],
    desc="Scanning mask directories",
):
    mask_dir = image_base_path / "data" / patient / mask_subparent_name / well_fov
    key = str(mask_dir)
    if mask_dir.exists():
        mask_tiff_counts[key] = len(list(mask_dir.glob("*.tiff")))
    else:
        mask_tiff_counts[key] = 0


def _mask_dir_key(row) -> str:
    return str(
        image_base_path
        / "data"
        / row["patient"]
        / row["mask_subparent_name"]
        / row["well_fov"]
    )


df["num_of_masks"] = df.apply(
    lambda row: mask_tiff_counts.get(_mask_dir_key(row), 0), axis=1
)

df_rerun = df.loc[df["num_of_masks"] < 1].copy()
df_rerun = df_rerun.drop(columns=["num_of_masks"]).reset_index(drop=True)

print(f"{df.shape[0]} total segmentation jobs")
print(f"{df.shape[0] - df_rerun.shape[0]} complete (≥4 masks)")
print(f"{df_rerun.shape[0]} jobs to rerun")
df_rerun.head()


# In[6]:


df_rerun.to_csv(rerun_combinations_path, sep="\t", index=False)


# In[7]:


df_rerun.groupby("patient").size().to_frame(name="reruns").reset_index()
