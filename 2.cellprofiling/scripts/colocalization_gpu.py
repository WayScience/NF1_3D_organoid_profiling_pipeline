#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import pathlib
import sys
import time

import psutil

sys.path.append("../featurization_utils")
import itertools

import cucim.skimage.measure
import cupy
import cupyx
import cupyx.scipy.ndimage
import numpy as np
import pandas as pd
import scipy
import skimage
from colocalization_utils_gpu import (
    measure_3D_colocalization_gpu,
    prepare_two_images_for_colocalization_gpu,
)
from loading_classes import ImageSetLoader, ObjectLoader, TwoObjectLoader
from resource_profiling_util import get_mem_and_time_profiling

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# In[ ]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default="None",
        help="Well and field of view to process, e.g. 'A01_1'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    if well_fov == "None":
        raise ValueError(
            "Please provide a well and field of view to process, e.g. 'A01_1'"
        )

    image_set_path = pathlib.Path(f"../../data/NF0014/cellprofiler/{well_fov}/")
else:
    well_fov = "C4-2"
    image_set_path = pathlib.Path(f"../../data/NF0014/cellprofiler/{well_fov}/")


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


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[ ]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[ ]:


for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel1, channel2 in tqdm(
        channel_combinations,
        desc="Processing channel combinations",
        leave=False,
        position=1,
    ):
        output_dir = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/Colocalization_{compartment}_{channel1}.{channel2}_features"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        coloc_loader = TwoObjectLoader(
            image_set_loader=image_set_loader,
            compartment=compartment,
            channel1=channel1,
            channel2=channel2,
        )
        list_of_dfs = []
        for object_id in tqdm(
            coloc_loader.object_ids,
            desc="Processing object IDs",
            leave=False,
            position=2,
        ):
            cropped_image1, cropped_image2 = prepare_two_images_for_colocalization_gpu(
                label_object1=coloc_loader.label_image,
                label_object2=coloc_loader.label_image,
                image_object1=coloc_loader.image1,
                image_object2=coloc_loader.image2,
                object_id1=object_id,
                object_id2=object_id,
            )
            colocalization_features = measure_3D_colocalization_gpu(
                cropped_image_1=cropped_image1,
                cropped_image_2=cropped_image2,
                thr=15,
                fast_costes="Accurate",
            )
            coloc_df = pd.DataFrame(colocalization_features, index=[0])
            coloc_df.columns = [
                f"Colocalization_{compartment}_{channel1}.{channel2}_{col}"
                for col in coloc_df.columns
            ]
            coloc_df.insert(0, "object_id", object_id)
            coloc_df.insert(1, "image_set", image_set_loader.image_set_name)
            coloc_df.to_parquet(output_dir / f"object_{object_id}.parquet")


# In[ ]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    process_name="Colocalization",
    well_fov=well_fov,
    CPU_GPU="GPU",
)
