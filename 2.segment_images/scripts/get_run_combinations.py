#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import os
import pathlib
import sys
from itertools import product

import numpy as np
import pandas as pd

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


from notebook_init_utils import bandicoot_check, init_notebook

# In[2]:


root_dir, in_notebook = init_notebook()
bandicoot_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve()
image_base_path = bandicoot_check(
    bandicoot_mount_path=bandicoot_path, root_dir=root_dir
)
patient_id_file = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()

input_combinations_path = pathlib.Path(
    f"{root_dir}/2.segment_images/load_data/input_combinations.txt"
)
rerun_combinations_path = pathlib.Path(
    f"{root_dir}/2.segment_images/load_data/rerun_combinations.txt"
)
input_combinations_path.parent.mkdir(parents=True, exist_ok=True)
rerun_combinations_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


channel_mapping = {
    "DNA": "405",
    "AGP": "488",
    "ER": "555",
    "Mito": "640",
    "BF": "TRANS",
    "Nuclei": "nuclei_",
    "Cell": "cell_",
    "Cytoplasm": "cytoplasm_",
    "Organoid": "organoid_",
}


# In[4]:


output_dict = {
    "patient": [],
    "well_fov": [],
    "input_subparent_name": [],
    "mask_subparent_name": [],
}


# In[5]:


convolution_iters = [x for x in range(1, 26)]
convolution_iters = convolution_iters + [50, 75, 100]


# In[6]:


for patient in patients:
    # get the well_fov for each patient
    patient_well_fovs = pathlib.Path(
        f"{image_base_path}/data/{patient}/zstack_images/"
    ).glob("*")
    for well_fov in patient_well_fovs:
        well_fov = well_fov.name
        output_dict["patient"].append(patient)
        output_dict["well_fov"].append(well_fov)
        output_dict["input_subparent_name"].append("zstack_images")
        output_dict["mask_subparent_name"].append("segmentation_masks")

        # this is specific to the segmentation of the convolutions performed on the NF0014_T1 C4-2 well_fov
        if patient == "NF0014_T1" and well_fov == "C4-2":
            for convolution_iter in convolution_iters:
                output_dict["patient"].append(patient)
                output_dict["well_fov"].append(well_fov)
                output_dict["input_subparent_name"].append(
                    f"convolution_{convolution_iter}"
                )
                output_dict["mask_subparent_name"].append(
                    f"convolution_{convolution_iter}_segmentation_masks"
                )
            output_dict["patient"].append(patient)
            output_dict["well_fov"].append(well_fov)
            output_dict["input_subparent_name"].append("deconvolved_images")
            output_dict["mask_subparent_name"].append("deconvolved_segmentation_masks")


# In[7]:


df = pd.DataFrame(output_dict)
print(f"Total combinations: {df.shape[0]}")
df.head()


# In[8]:


# write to a txt file with each row as a combination
# each column is a feature of the combination
df.to_csv(input_combinations_path, sep="\t", index=False)


# ## Rerun list

# In[9]:


# check which to rerun by checking if file exists
df["file_path"] = df.apply(
    lambda df: pathlib.Path(
        f"{image_base_path}/data/{df['patient']}/{df['mask_subparent_name']}/{df['well_fov']}/organoid_mask.tiff"
    ),
    axis=1,
)
df["exists"] = df["file_path"].apply(lambda x: x.exists())
print(f"{df.shape[0]} segmentations needed")
df_rerun = df[~df["exists"]]
print(f"{df.shape[0] - df_rerun.shape[0]} segmentations exist")
print(f"{df_rerun.shape[0]} combinations to rerun")
df_rerun


# In[10]:


df_rerun = df_rerun.drop(columns=["file_path", "exists"])
# write to a txt file with each row as a combination
# each column is a feature of the combination
df_rerun.to_csv(rerun_combinations_path, sep="\t", index=False)
