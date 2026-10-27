#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


raw_viabilities_path = pathlib.Path(
    f"{root_dir}/config/viabilities/raw_viability_files"
).resolve()
raw_viabilities_combined_output_file_path = pathlib.Path(
    f"{root_dir}/config/viabilities/raw_viabilities_combined.csv"
).resolve()
raw_viabilities = list(raw_viabilities_path.glob("*"))
combined_viabilities_df_list = []
for raw_viability in raw_viabilities:
    df = pd.read_csv(raw_viability)
    df["patient_id"] = raw_viability.stem.strip("_Viabilities")

    combined_viabilities_df_list.append(df)
combined_viabilities_df = pd.concat(combined_viabilities_df_list, axis=0)
combined_viabilities_df.to_csv(raw_viabilities_combined_output_file_path, index=False)
