#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import time
import urllib

import numpy as np
import pandas as pd
import psutil
from arg_parsing_utils import check_for_missing_args, parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling
from sammed3d_featurizer import call_SAMMed3D_pipeline

# In[2]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    compartment = arguments_dict["compartment"]
    channel = arguments_dict["channel"]

else:
    well_fov = "C4-2"
    patient = "NF0014_T1"
    compartment = "Organoid"
    channel = "DNA"

image_set_path = pathlib.Path(f"{root_dir}/data/{patient}/zstack_images/{well_fov}/")

output_parent_path = pathlib.Path(
    f"{root_dir}/data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[3]:


sam3dmed_checkpoint_url = (
    "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
)
sam3dmed_checkpoint_path = pathlib.Path("../models/sam-med3d-turbo.pth").resolve()
if not sam3dmed_checkpoint_path.exists():
    sam3dmed_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(sam3dmed_checkpoint_url, str(sam3dmed_checkpoint_path))


# In[4]:


channel_n_compartment_mapping = {
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


# In[5]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[6]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)


# In[7]:


# load the objects for the compartment and channel of interest
object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)

feature_dict = call_SAMMed3D_pipeline(
    object_loader=object_loader,
    SAMMed3D_model_path=str(sam3dmed_checkpoint_path),
    feature_type="cls",
)


# In[8]:


final_df = pd.DataFrame(feature_dict)

final_df["feature_name"] = (
    final_df["feature_name"] + "_" + final_df["compartment"] + "_" + final_df["channel"]
)
final_df["feature_name"] = final_df["feature_name"].str.replace("_feature_", ".")
final_df = final_df.drop(columns=["compartment", "channel"])
final_df.head()


# In[9]:


final_df = final_df.pivot(
    index="object_id", columns="feature_name", values="value"
).reset_index()
# drop the multiindexing from pivot
final_df.columns.name = None
final_df


# In[10]:


# prepend compartment and channel to column names
for col in final_df.columns:
    if col not in ["object_id"]:
        final_df[col] = final_df[col].astype(np.float32)


# In[11]:


# de-fragment
final_df = final_df.copy()
final_df.insert(1, "image_set", image_set_loader.image_set_name)

output_file = pathlib.Path(
    output_parent_path / f"SAMMed3D_{compartment}_{channel}_features.parquet"
)
final_df.to_parquet(output_file, index=False)
final_df.head()


# In[12]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="AreaSizeShape",
    well_fov=well_fov,
    patient_id=patient,
    channel="DNA",
    compartment=compartment,
    CPU_GPU="GPU",
    output_file_dir=pathlib.Path(
        f"{root_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_AreaSizeShape_DNA_{compartment}_GPU.parquet"
    ),
)
