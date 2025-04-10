#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys
import time

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


# In[2]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


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
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[5]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[6]:


start_time = time.time()


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
        coloc_loader = TwoObjectLoader(
            image_set_loader=image_set_loader,
            compartment=compartment,
            channel1=channel1,
            channel2=channel2,
        )
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
                f"{compartment}_{channel1}.{channel2}_{col}" for col in coloc_df.columns
            ]
            coloc_df["object_id"] = object_id
            coloc_df["channel1"] = channel1
            coloc_df["channel2"] = channel2
            coloc_df["compartment"] = compartment
            coloc_df["image_set"] = image_set_loader.image_set_name
        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/Colocalization_{compartment}_{channel1}.{channel2}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        coloc_df.to_parquet(output_file)


# In[8]:


print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
