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
import os

import cucim
import cucim.skimage.morphology
import cupy as cp
import cupyx.scipy.ndimage
import numpy
import numpy as np
import pandas as pd
import psutil
import scipy
import skimage
import tqdm
from granularity_utils import measure_3D_granularity_gpu
from loading_classes import ImageSetLoader, ObjectLoader
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


# In[ ]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default="None",
        help="Well and field of view to process, e.g. 'A01_1'",
    )

    argparser.add_argument(
        "--patient",
        type=str,
        help="Patient ID, e.g. 'NF0014'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
    if well_fov == "None":
        raise ValueError(
            "Please provide a well and field of view to process, e.g. 'A01_1'"
        )

else:
    well_fov = "C4-2"
    patient = "NF0014"

image_set_path = pathlib.Path(f"../../data/{patient}/cellprofiler/{well_fov}/")
output_parent_path = pathlib.Path(
    f"../../data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[ ]:


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


# In[ ]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[ ]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[ ]:


for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel in tqdm(
        image_set_loader.image_names,
        desc="Processing channels",
        leave=False,
        position=1,
    ):
        channel = "DNA"
        compartment = "Nuclei"

        object_loader = ObjectLoader(
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )
        object_measurements = measure_3D_granularity_gpu(
            object_loader=object_loader,
            image_set_loader=image_set_loader,
            radius=10,
            granular_spectrum_length=16,
            subsample_size=0.25,
            image_name=object_loader.channel,
        )
        final_df = pd.DataFrame(object_measurements)
        # get the mean of each value in the array
        # melt the dataframe to wide format
        final_df = final_df.pivot_table(
            index=["object_id"], columns=["feature"], values=["value"]
        )
        final_df.columns = final_df.columns.droplevel()
        final_df = final_df.reset_index()
        # prepend compartment and channel to column names
        for col in final_df.columns:
            if col == "object_id":
                continue
            else:
                final_df.rename(
                    columns={col: f"Granularity_{compartment}_{channel}_{col}"},
                    inplace=True,
                )
        final_df.insert(0, "image_set", image_set_loader.image_set_name)

        output_file = pathlib.Path(
            output_parent_path / f"Granularity_{compartment}_{channel}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)
        final_df.head()


# In[ ]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Granularity",
    well_fov=well_fov,
    patient_id=patient,
    CPU_GPU="GPU",
    output_file_dir=pathlib.Path(
        f"../../data/{patient}/extracted_features/run_stats/{well_fov}_Granularity_GPU.parquet"
    ),
)
