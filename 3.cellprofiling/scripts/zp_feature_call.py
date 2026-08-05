#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import os
import pathlib
import sys
import time

import bioio
import numpy as np
import pandas as pd
import psutil
import tifffile
import tomli
import tqdm.notebook as tqdm
import zedprofiler
from image_analysis_3D.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
    save_features_as_parquet,
)

# bug in the cucim module but we are using CPU so it does not matter for now
# from image_analysis_3D.featurization_utils.area_size_shape_utils_gpu import measure_3D_area_size_shape_gpu
from image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)
from image_analysis_3D.featurization_utils.resource_profiling_util import (
    start_profiling,
    stop_profiling,
)
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from zedprofiler.IO import loading_classes

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


start_time, start_mem = start_profiling()


# In[3]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    input_subparent_name = arguments_dict["input_subparent_name"]
    mask_subparent_name = arguments_dict["mask_subparent_name"]
    output_features_subparent_name = arguments_dict["output_features_subparent_name"]

else:
    well_fov = "C4-2"
    patient = "NF0014_T1"
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
channel_mapping_file_path = pathlib.Path(
    f"{root_dir}/config/channel_mapping.toml"
).resolve(strict=True)


# In[4]:


channel_mapping = tomli.loads(channel_mapping_file_path.read_text())


# In[5]:


# load in the images once
nuclei_image_path = mask_set_path / "nuclei_mask.tiff"
cell_image_path = mask_set_path / "cell_mask.tiff"
cytoplasm_image_path = mask_set_path / "cytoplasm_mask.tiff"
organoid_image_path = mask_set_path / "organoid_mask.tiff"
agp_image_path = image_set_path / f"{well_fov}_555.tif"
dna_image_path = image_set_path / f"{well_fov}_405.tif"
er_image_path = image_set_path / f"{well_fov}_488.tif"
mito_image_path = image_set_path / f"{well_fov}_640.tif"
nuclei_mask = tifffile.imread(nuclei_image_path)
cell_mask = tifffile.imread(cell_image_path)
cytoplasm_mask = tifffile.imread(cytoplasm_image_path)
organoid_mask = tifffile.imread(organoid_image_path)
agp_image = tifffile.imread(agp_image_path)
dna_image = tifffile.imread(dna_image_path)
er_image = tifffile.imread(er_image_path)
mito_image = tifffile.imread(mito_image_path)


# In[11]:


# for testing purposes, slice the images to a smaller size
# get the first 6 slices of the images
nuclei_mask = nuclei_mask[:6, :, :]
cell_mask = cell_mask[:6, :, :]
cytoplasm_mask = cytoplasm_mask[:6, :, :]
organoid_mask = organoid_mask[:6, :, :]
agp_image = agp_image[:6, :, :]
dna_image = dna_image[:6, :, :]
er_image = er_image[:6, :, :]
mito_image = mito_image[:6, :, :]


# In[24]:


# set channel combinatorics
channel_combinations = list(itertools.combinations(["AGP", "DNA", "ER", "Mito"], 2))
channel_combinations_with_compartment_combinations = list(
    itertools.product(channel_combinations, ["Organoid", "Cell", "Cytoplasm", "Nuclei"])
)
channel_combinations_with_compartment_combinations = [
    (a, b, c) for (a, b), c in channel_combinations_with_compartment_combinations
]
print(channel_combinations_with_compartment_combinations)

channel_and_compartment_combinations = list(
    itertools.product(
        ["AGP", "DNA", "ER", "Mito"], ["Organoid", "Cell", "Cytoplasm", "Nuclei"]
    )
)
channel_and_compartment_combinations += [
    ("NoChannel", "Organoid"),
    ("NoChannel", "Cell"),
    ("NoChannel", "Cytoplasm"),
    ("NoChannel", "Nuclei"),
]


# In[13]:


dict_of_images = {
    "AGP": agp_image,
    "DNA": dna_image,
    "ER": er_image,
    "Mito": mito_image,
    "Organoid": organoid_mask,
    "Cell": cell_mask,
    "Cytoplasm": cytoplasm_mask,
    "Nuclei": nuclei_mask,
    "NoChannel": np.zeros_like(organoid_mask),
}


# In[14]:


isc = loading_classes.ImageSetConfig(
    image_set_name=f"{patient}_{well_fov}",
    raw_image_key_name=["AGP", "DNA", "ER", "Mito", "NoChannel"],
    label_key_name=["Nuclei", "Cell", "Cytoplasm", "Organoid"],
)


# In[34]:


list_of_sc_dfs = []
list_of_organoid_dfs = []
for channel, compartment in tqdm.tqdm(channel_and_compartment_combinations):
    image_set_loader = loading_classes.ImageSetLoader(
        anisotropy_spacing=[1, 0.1, 0.1],
        channel_mapping=channel_mapping,
        image_set_array=dict_of_images[channel],
        label_set_array=dict_of_images[compartment],
        image_set_path=None,
        label_set_path=None,
        # image_set_path=image_set_path,
        # label_set_path=mask_set_path,
        config=isc,
    )
    ol = loading_classes.ObjectLoader(
        image_set_loader=image_set_loader,
        channel_name=channel,
        compartment_name=compartment,
    )
    if channel != "NoChannel":
        # compute features
        granularity_features = zedprofiler.granularity.compute_granularity(
            object_loader=ol,
            radius=10,  # radius of the structuring element for background
            # removal (CellProfiler default)
            granular_spectrum_length=16,  # range of the granular spectrum
            subsample_size=0.25,  # subsample the image for faster processing
            image_sample_size=0.25,  # further subsample for background removal
            mask_threshold=0.9,  # threshold for determining mask after interpolation
            verbose=False,  # print out intermediate steps and values for debugging
        )
        intensity_features = zedprofiler.intensity.compute_intensity(
            object_loader=ol,
        )

        # texture_features = zedprofiler.texture.compute_texture(
        #     object_loader=ol,
        #     distance=3,
        #     grayscale=256,
        # )
        if compartment == "Organoid":
            list_of_organoid_dfs.append(granularity_features)
            list_of_organoid_dfs.append(intensity_features)
            # list_of_organoid_dfs.append(texture_features)
        else:
            list_of_sc_dfs.append(granularity_features)
            list_of_sc_dfs.append(intensity_features)
            # list_of_feature_dfs.append(texture_features)
    elif channel == "NoChannel":
        volume_size_shape_features = (
            zedprofiler.volumesizeshape.compute_volume_size_shape(
                image_set_loader=image_set_loader,
                object_loader=ol,
            )
        )
        if compartment == "Organoid":
            list_of_organoid_dfs.append(volume_size_shape_features)

        else:
            list_of_sc_dfs.append(volume_size_shape_features)

        if compartment == "Nuclei":
            neighbors_features = zedprofiler.neighbors.compute_neighbors(
                object_loader=ol,
                distance_threshold=10,
                anisotropy_factor=image_set_loader.anisotropy_spacing[0]
                / image_set_loader.anisotropy_spacing[
                    1
                ],  # z to xy spacing ratio for distance calculation
            )
            if compartment == "Organoid":
                list_of_organoid_dfs.append(neighbors_features)
            else:
                list_of_sc_dfs.append(neighbors_features)


# In[35]:


for channel1, channel2, compartment in tqdm.tqdm(
    channel_combinations_with_compartment_combinations
):
    image_set_loader = loading_classes.ImageSetLoader(
        anisotropy_spacing=[1, 0.1, 0.1],
        channel_mapping=channel_mapping,
        image_set_array=dict_of_images[channel1],
        label_set_array=dict_of_images[compartment],
        image_set_path=None,
        label_set_path=None,
        # image_set_path=image_set_path,
        # label_set_path=mask_set_path,
        config=isc,
    )
    coloc_loader = loading_classes.TwoObjectLoader(
        image_set_loader=image_set_loader,
        compartment=compartment,
        channel1=channel1,
        channel2=channel2,
    )
    if channel1 != "NoChannel" and channel2 != "NoChannel":
        colocalization_features = zedprofiler.colocalization.compute_colocalization(
            two_object_loader=coloc_loader,
            thr=15,
            fast_costes="Faster",
            channel1=channel1,
            channel2=channel2,
        )
        if compartment == "Organoid":
            list_of_organoid_dfs.append(colocalization_features)

        else:
            list_of_sc_dfs.append(colocalization_features)


# In[ ]:


stop_profiling(
    start_time=start_time,
    start_mem=start_mem,
    feature_type="AreaSizeShape",
    well_fov=well_fov,
    patient_id=patient,
    channel="NoChannel",
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"{image_base_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_AreaSizeShape_NoChannel_{compartment}_{processor_type}.parquet"
    ),
)


# In[ ]:


# In[36]:


sc_df = pd.DataFrame()
for df in list_of_sc_dfs:
    if sc_df.empty:
        sc_df = df
    sc_df = pd.merge(
        left=sc_df,
        right=df,
        how="left",
        on=["Metadata_Object_ObjectID", "Metadata_Experiment_ImageSet"],
    )
print(sc_df.shape)
sc_df.head()


# In[37]:


organoid_df = pd.DataFrame()
for df in list_of_sc_dfs:
    if organoid_df.empty:
        organoid_df = df
    organoid_df = pd.merge(
        left=organoid_df,
        right=df,
        how="left",
        on=["Metadata_Object_ObjectID", "Metadata_Experiment_ImageSet"],
    )
print(organoid_df.shape)
organoid_df.head()
