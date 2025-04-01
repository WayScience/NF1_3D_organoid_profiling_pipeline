#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys
import time

sys.path.append("../featurization_utils")
import numpy as np
import pandas as pd
import skimage
from area_size_shape_utils import measure_3D_area_size_shape
from loading_classes import ImageSetLoader, ObjectLoader

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import gc

# In[2]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


# In[ ]:


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


# In[ ]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)


# In[ ]:


start_time = time.time()


# In[5]:


for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel in tqdm(
        image_set_loader.image_names,
        desc="Processing channels",
        leave=False,
        position=1,
    ):
        object_loader = ObjectLoader(
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )

        # area, size, shape
        size_shape_dict = measure_3D_area_size_shape(
            image_set_loader=image_set_loader,
            object_loader=object_loader,
        )
        final_df = pd.DataFrame(size_shape_dict)
        # prepend compartment and channel to column names
        final_df.columns = [
            f"{compartment}_{channel}_{col}" for col in final_df.columns
        ]
        final_df["image_set"] = image_set_loader.image_set_name

        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/AreaSize_Shape_{compartment}_{channel}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)

        # remove the objects initialized in the beginning of the loop
        del object_loader
        del size_shape_dict
        del final_df
        gc.collect()


# In[ ]:


print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
