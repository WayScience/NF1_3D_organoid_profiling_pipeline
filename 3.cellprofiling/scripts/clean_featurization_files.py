#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import skimage
import tifffile
import tomli
import torch
from image_analysis_3D.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
    save_features_as_parquet,
)
from image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)
from image_analysis_3D.featurization_utils.morphem_featurization import (
    PerImageNormalize,
    SaturationNoiseInjector,
    call_morphem_featurization_pipeline,
    featurize_2D_image_w_morphem,
    get_morphem_model,
)
from image_analysis_3D.featurization_utils.resource_profiling_util import (
    start_profiling,
    stop_profiling,
)
from image_analysis_3D.featurization_utils.sammed3d_featurizer import (
    MicroscopySAMMed3DPipeline,
    call_whole_image_sammed3d_pipeline,
)
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from image_analysis_3D.image_utils.image_utils import (
    check_for_xy_squareness,
    crop_3D_image,
    expand_box,
    new_crop_border,
    select_objects_from_label,
    single_3D_image_expand_bbox,
    square_off_xy_crop_bbox,
)

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[ ]:


foreign_column = 0
# get a list of all files in all extracted_features dirs
extracted_features = list(pathlib.Path(f"{image_base_dir}/data").glob("*"))
extracted_features = [x for x in extracted_features if x.is_dir()]
extracted_features = [
    pathlib.Path(f"{x}/extracted_features")
    for x in extracted_features
    if (x / "extracted_features").exists()
]
for patient_dir in tqdm.tqdm(extracted_features, desc="patients"):
    if not patient_dir.exists():
        print(f"no extracted features dir for {patient_dir}")
        continue
    well_dirs = list(patient_dir.glob("*"))
    for well_dir in tqdm.tqdm(well_dirs, desc="wells"):
        feature_files = list(pathlib.Path(f"{well_dir}/").glob("*.parquet"))
        feature_files = [x for x in feature_files if x.is_file()]
        feature_files = [
            x
            for x in feature_files
            if "morphem" in x.stem.lower() or "sammed3d" in x.stem.lower()
        ]
        for feature_file in feature_files:
            if "chammi" in feature_files[0].stem.lower():
                feature_type = "morphem"
                opposite_feature_type = "sammed3d"
            elif "sammed3d" in feature_files[0].stem.lower():
                feature_type = "sammed3d"
                opposite_feature_type = "morphem"
            else:
                raise ValueError(f"unknown feature type for {feature_files[0]}")
            df = pd.read_parquet(feature_files[0])
            num_of_wrong_columns = len(
                [x for x in df.columns if opposite_feature_type in x.lower()]
            )
            if num_of_wrong_columns > 0:
                print(
                    f"{feature_files[0]} has {num_of_wrong_columns} columns with the wrong feature type in the name"
                )
                foreign_column += 1


# In[ ]:


foreign_column
