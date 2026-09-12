#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import tifffile
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.auto as tqdm
else:
    import tqdm

bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot"))
bandicoot_mount_path = bandicoot_check(bandicoot_mount_path, root_dir)


# In[2]:


def change_mito_shape_to_match_nuclei(row):
    """
    If the mito image has a different shape than the nuclei image, copy the first z-slice and insert it at the beginning of the stack to make the shapes match.
    """
    if row.mito_image_shape != row.nuclei_image_shape:
        mito_image = tifffile.imread(row.mito_image_path)

        if row.mito_image_shape < row.nuclei_image_shape:
            # copy the first z-slice and insert it at the beginning of the stack
            first_slice = mito_image[0:1, :, :]
            new_mito_image = np.concatenate([first_slice, mito_image], axis=0)
            print(mito_image.shape, new_mito_image.shape)
        elif row.mito_image_shape > row.nuclei_image_shape:
            # remove the first z-slice to make the shapes match
            new_mito_image = mito_image[1:, :, :]
            print(mito_image.shape, new_mito_image.shape)

        # save the new mito image to a new path
        tifffile.imwrite(row.mito_image_path, new_mito_image)


def retrieve_image_shape(image_path: pathlib.Path) -> Tuple[int, int, int]:
    """
    Retrieve the shape of the image from path

    Parameters
    ----------
    image_path : pathlib.Path
        Path to the image file

    Returns
    -------
    tuple
        Shape of the image as (z, y, x) or None if there was an error loading the image
    """
    try:
        with tifffile.TiffFile(image_path) as tif:
            shape = tif.series[0].shape
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        shape = None
    return shape


# In[3]:


patients = ["NF0037_T1", "NF0037_T1_CQ1"]


# In[4]:


final_dict = {
    "patient": [],
    "well_fov": [],
    "dna_image_path": [],
    "er_image_path": [],
    "agp_image_path": [],
    "mito_image_path": [],
    "nuclei_mask_path": [],
    "cell_mask_path": [],
    "cyto_mask_path": [],
    "organoid_mask_path": [],
}
for patient in tqdm.tqdm(patients, desc="Patient", unit="patient"):
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
            (bandicoot_mount_path / "data" / patient / "zstack_images" / well_fov).glob(
                "*.tif*"
            )
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
            if "405" in image.name:
                dna_image_path = image
            elif "488" in image.name:
                er_image_path = image
            elif "555" in image.name:
                agp_image_path = image
            elif "640" in image.name:
                mito_image_path = image
        for mask in masks:
            if "nuclei" in mask.name:
                nuclei_mask_path = mask
            elif "cell" in mask.name:
                cell_mask_path = mask
            elif "cyto" in mask.name:
                cyto_mask_path = mask
            elif "organoid" in mask.name:
                organoid_mask_path = mask

        final_dict["patient"].append(patient)
        final_dict["well_fov"].append(well_fov)
        final_dict["mito_image_path"].append(mito_image_path)
        final_dict["dna_image_path"].append(dna_image_path)
        final_dict["er_image_path"].append(er_image_path)
        final_dict["agp_image_path"].append(agp_image_path)
        final_dict["nuclei_mask_path"].append(nuclei_mask_path)
        final_dict["cell_mask_path"].append(cell_mask_path)
        final_dict["cyto_mask_path"].append(cyto_mask_path)
        final_dict["organoid_mask_path"].append(organoid_mask_path)
df = pd.DataFrame(final_dict)


# In[5]:


columns_to_iterate = {
    # path column name : shape column name (output)
    "dna_image_path": "dna_shape",
    "er_image_path": "er_shape",
    "agp_image_path": "agp_shape",
    "mito_image_path": "mito_shape",
    "nuclei_mask_path": "nuclei_shape",
    "cell_mask_path": "cell_shape",
    "cyto_mask_path": "cyto_shape",
    "organoid_mask_path": "organoid_shape",
}


# In[6]:


# create the new columns for the data to be inserted into
for shape_column_name in columns_to_iterate.values():
    df[shape_column_name] = None
for row in tqdm.tqdm(
    df.itertuples(), total=df.shape[0], desc="Rows", unit="row", leave=True
):
    for path_column_name, shape_column_name in columns_to_iterate.items():
        shape = retrieve_image_shape(row._asdict()[path_column_name])
        df.at[row.Index, shape_column_name] = shape
# convert the pathlib.Path objects to strings for saving to parquet
for path_column_name in columns_to_iterate.keys():
    df[path_column_name] = df[path_column_name].astype(str)

df.head()


# In[7]:


df["match_bool"] = df.apply(
    lambda row: (
        row.dna_shape
        == row.er_shape
        == row.agp_shape
        == row.mito_shape
        == row.nuclei_shape
        == row.cell_shape
        == row.cyto_shape
        == row.organoid_shape
    ),
    axis=1,
)
# show all the rows where the match_bool is False
mismatched_rows = df[df["match_bool"] == False]
# extract the unique patient/well_fov combinations where the mismatch occurs
mismatched_patient_well_fovs = mismatched_rows[
    ["patient", "well_fov"]
].drop_duplicates()
for x in mismatched_patient_well_fovs.itertuples():
    print(x.patient, x.well_fov)
mismatched_rows


# In[8]:


print(len(mismatched_rows))


# In[10]:


# for row in tqdm.tqdm(
#     df.itertuples(), total=len(df), desc="Checking shapes", unit="row"
# ):
#     change_mito_shape_to_match_nuclei(row)
