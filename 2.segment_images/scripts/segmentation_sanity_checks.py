#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import multiprocessing
import os
import pathlib

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


OVERWRITE = True


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


if sanity_df_save_path.exists() and not OVERWRITE:
    print(
        f"Sanity check dataframe already exists at {sanity_df_save_path}. Set OVERWRITE = True to overwrite."
    )
    df = pd.read_parquet(sanity_df_save_path)

else:
    list_of_dicts = []

    for patient in tqdm.tqdm(patients, desc="Patients", unit="patient", leave=True):
        final_dict = {
            "patient": [],
            "well_fov": [],
            "image_path": [],
            # "image_shape": [],
        }
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
            images = sorted(
                (
                    bandicoot_mount_path / "data" / patient / "zstack_images" / well_fov
                ).glob("*.tif*")
            )
            masks = sorted(
                (
                    bandicoot_mount_path
                    / "data"
                    / patient
                    / "segmentation_masks"
                    / well_fov
                ).glob("*.tif*")
            )
            for image in images:
                final_dict["patient"].append(patient)
                final_dict["well_fov"].append(well_fov)
                final_dict["image_path"].append(image)
            for mask in masks:
                final_dict["patient"].append(patient)
                final_dict["well_fov"].append(well_fov)
                final_dict["image_path"].append(mask)
        list_of_dicts.append(final_dict)

    df = pd.DataFrame(
        {
            "patient": [],
            "well_fov": [],
            "image_path": [],
            # "image_shape": [],
        }
    )

    # concatenate all the dictionaries into a single dataframe
    for d in list_of_dicts:
        df = pd.concat([df, pd.DataFrame(d)], ignore_index=True)
    # add another column to capture the image shape
    shapes = []
    for image_path in tqdm.tqdm(
        df["image_path"], desc="Loading image shapes", unit="image"
    ):
        try:
            with tifffile.TiffFile(image_path) as tif:
                shape = tif.series[0].shape
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            shape = None
        shapes.append(shape)
    df["image_shape"] = shapes
    # convert the posix path to string for parquet compatibility
    df["image_path"] = df["image_path"].astype(str)

    df.to_parquet(sanity_df_save_path, index=False)

print(f"Sanity check df shape: {df.shape}")
df["z_shape"] = df["image_shape"].apply(lambda x: x[0] if x is not None else None)
df["y_shape"] = df["image_shape"].apply(lambda x: x[1] if x is not None else None)
df["x_shape"] = df["image_shape"].apply(lambda x: x[2] if x is not None else None)
df["unique_shape_string"] = (
    f"{df['z_shape'].astype(str)}_{df['y_shape'].astype(str)}_{df['x_shape'].astype(str)}"
)
df.head()


# In[5]:


df["unique_shape_string"] = (
    df["z_shape"].astype(str)
    + "_"
    + df["y_shape"].astype(str)
    + "_"
    + df["x_shape"].astype(str)
)
df


# In[6]:


mismatched_shapes = 0
mismatched_shapes_list = []

# verify shapes for each patient/well_fov combination are the same
for (patient, well_fov), group in df.groupby(["patient", "well_fov"]):
    image_shapes = group["unique_shape_string"].unique()
    if len(image_shapes) > 1:
        mismatched_shapes += 1
        mismatched_shapes_list.append((patient, well_fov))
print(
    f"Number of patient/well_fov combinations with mismatched shapes: {mismatched_shapes}"
)
mismatched_shapes_list


# In[ ]:
