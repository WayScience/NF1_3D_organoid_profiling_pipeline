#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import logging
import os
import pathlib
import time
import urllib

import numpy as np
import pandas as pd
import psutil
from arg_parsing_utils import check_for_missing_args, parse_args
from loading_classes import ImageSetLoader, ObjectLoader
from notebook_init_utils import bandicoot_check, init_notebook
from resource_profiling_util import get_mem_and_time_profiling
from sammed3d_featurizer import call_SAMMed3D_pipeline

root_dir, in_notebook = init_notebook()
from notebook_init_utils import bandicoot_check, init_notebook

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


# set up logging
logging.basicConfig(level=logging.INFO)


# In[3]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    compartment = arguments_dict["compartment"]
    channel = arguments_dict["channel"]
    input_subparent_name = arguments_dict["input_subparent_name"]
    mask_subparent_name = arguments_dict["mask_subparent_name"]
    output_features_subparent_name = arguments_dict["output_features_subparent_name"]

else:
    well_fov = "D11-2"
    patient = "NF0016_T1"
    compartment = "Nuclei"
    channel = "Mito"
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"
    output_features_subparent_name = "extracted_features"

image_set_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}/"
)
mask_set_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/{well_fov}/"
)

output_parent_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{output_features_subparent_name}/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[4]:


sam3dmed_checkpoint_url = (
    "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
)
sam3dmed_checkpoint_path = pathlib.Path("../models/sam-med3d-turbo.pth").resolve()
if not sam3dmed_checkpoint_path.exists():
    sam3dmed_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(sam3dmed_checkpoint_url, str(sam3dmed_checkpoint_path))


# In[5]:


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


# In[6]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[7]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    mask_set_path=mask_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)
image_set_loader.image_set_dict.keys()


# In[8]:


# if channel and compartment are "all", create all combinations
if channel == "all" and compartment == "all":
    channels = ["DNA", "AGP", "ER", "Mito", "BF"]
    compartments = ["Nuclei", "Cell", "Cytoplasm", "Organoid"]
    all_channel_compartment_combinations = list(
        itertools.product(channels, compartments)
    )
# if not then pass through the single combination
else:
    all_channel_compartment_combinations = [(channel, compartment)]


# In[9]:


for channel, compartment in all_channel_compartment_combinations:
    # load the objects for the compartment and channel of interest
    object_loader = ObjectLoader(
        image_set_loader.image_set_dict[channel],
        image_set_loader.image_set_dict[compartment],
        channel,
        compartment,
    )
    #  redirect stdout to logging
    logging.info("Starting SAM-Med3D feature extraction")
    feature_dict = call_SAMMed3D_pipeline(
        object_loader=object_loader,
        SAMMed3D_model_path=str(sam3dmed_checkpoint_path),
        feature_type="cls",
    )
    # write out the features to parquet
    # convert to dataframe
    final_df = pd.DataFrame(feature_dict)
    try:
        final_df["feature_name"] = (
            final_df["feature_name"]
            + "_"
            + final_df["compartment"]
            + "_"
            + final_df["channel"]
        )
        final_df["feature_name"] = final_df["feature_name"].str.replace(
            "_feature_", "."
        )
        final_df = final_df.drop(columns=["compartment", "channel"])
    except Exception as e:
        logging.error(f"Probably a zero object error: {e}")
    # reshape dataframe such that features are columns and the object_ids are rows
    final_df = final_df.pivot(
        index="object_id", columns="feature_name", values="value"
    ).reset_index()
    # drop the multiindexing from pivot
    final_df.columns.name = None
    # prepend compartment and channel to column names
    for col in final_df.columns:
        if col not in ["object_id"]:
            final_df[col] = final_df[col].astype(np.float32)

    # de-fragment
    final_df = final_df.copy()
    # add the image_set_name column
    final_df.insert(1, "image_set", image_set_loader.image_set_name)
    # set the save path
    output_file = pathlib.Path(
        output_parent_path / f"SAMMed3D_{compartment}_{channel}_GPU_features.parquet"
    )
    final_df.to_parquet(output_file, index=False)
    final_df.head()

    end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    end_time = time.time()
    get_mem_and_time_profiling(
        start_mem=start_mem,
        end_mem=end_mem,
        start_time=start_time,
        end_time=end_time,
        feature_type="SAMMed3D",
        well_fov=well_fov,
        patient_id=patient,
        channel="DNA",
        compartment=compartment,
        CPU_GPU="GPU",
        output_file_dir=pathlib.Path(
            f"{root_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_SAMMed3D_{channel}_{compartment}_GPU.parquet"
        ),
    )
