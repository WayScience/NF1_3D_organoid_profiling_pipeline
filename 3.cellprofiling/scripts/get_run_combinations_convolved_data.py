#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import os
import pathlib
from itertools import product

import numpy as np
import pandas as pd
from loading_classes import ImageSetLoader
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()


# In[2]:


bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot"))
bandicoot_mount_path = bandicoot_check(bandicoot_mount_path, root_dir)


# In[3]:


patients = ["NF0014_T1"]
input_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/input_combinations.txt"
)
rerun_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/rerun_combinations.txt"
)
input_combinations_path.parent.mkdir(parents=True, exist_ok=True)
rerun_combinations_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


features = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "SAMMed3D",
    "Texture",
]


# In[5]:


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


# In[6]:


# example image set path to get the image set loader working
image_set_path = pathlib.Path(
    f"{bandicoot_mount_path}/data/NF0014_T1/zstack_images/C2-1/"
)
mask_set_path = pathlib.Path(
    f"{bandicoot_mount_path}/data/NF0014_T1/segmentation_masks/C2-1/"
)
image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    mask_set_path=mask_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[7]:


output_dict = {
    "patient": [],
    "well_fov": [],
    "feature": [],
    "compartment": [],
    "channel": [],
    "processor_type": [],
    "subdir_input": [],
    "subdir_mask": [],
    "subdir_output": [],
}
processor_types = [
    "CPU",
    # "GPU"
]


# In[8]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[9]:


list_of_subdirs_to_scan = []
iterations = list(range(1, 26)) + [50, 75, 100]
for i in iterations:
    list_of_subdirs_to_scan.append(f"convolution_{i}")
list_of_subdirs_to_scan.append("deconvolved_images")


# In[10]:


for patient in patients:
    for subdir in list_of_subdirs_to_scan:
        # get the well_fov for each patient
        patient_well_fovs = pathlib.Path(
            f"{bandicoot_mount_path}/data/{patient}/{subdir}/"
        ).glob("*")
        patient_well_fovs = [
            pathlib.Path(f"{bandicoot_mount_path}/data/{patient}/zstack_images/C4-2")
        ]  # for deconvolution testing only
        for well_fov in patient_well_fovs:
            well_fov = well_fov.name

            for feature in features:
                if feature == "Neighbors":
                    output_dict["patient"].append(patient)
                    output_dict["well_fov"].append(well_fov)
                    output_dict["feature"].append("Neighbors")
                    output_dict["compartment"].append("Nuclei")
                    output_dict["channel"].append("DNA")
                    output_dict["processor_type"].append("CPU")
                    output_dict["subdir_input"].append(subdir)
                    output_dict["subdir_mask"].append("segmentation_masks")
                    output_dict["subdir_output"].append(f"{subdir}_extracted_features")
                for compartment in image_set_loader.compartments:
                    if feature == "AreaSizeShape":
                        for processor_type in processor_types:
                            output_dict["patient"].append(patient)
                            output_dict["well_fov"].append(well_fov)
                            output_dict["feature"].append("AreaSizeShape")
                            output_dict["compartment"].append(compartment)
                            output_dict["channel"].append("DNA")
                            output_dict["processor_type"].append(processor_type)
                            output_dict["subdir_input"].append(subdir)
                            output_dict["subdir_mask"].append("segmentation_masks")
                            output_dict["subdir_output"].append(
                                f"{subdir}_extracted_features"
                            )
                    elif feature == "Colocalization":
                        for channel in channel_combinations:
                            for processor_type in processor_types:
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append("Colocalization")
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(
                                    channel[0] + "." + channel[1]
                                )
                                output_dict["processor_type"].append(processor_type)
                                output_dict["subdir_input"].append(subdir)
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    f"{subdir}_extracted_features"
                                )
                    for channel in image_set_loader.image_names:
                        if (
                            feature != "Neighbors"
                            and feature != "AreaSizeShape"
                            and feature != "Colocalization"
                        ):
                            if feature == "Granularity":
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append("CPU")
                                output_dict["subdir_input"].append(subdir)
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    f"{subdir}_extracted_features"
                                )
                            elif feature == "Intensity":
                                for processor_type in processor_types:
                                    output_dict["patient"].append(patient)
                                    output_dict["well_fov"].append(well_fov)
                                    output_dict["feature"].append(feature)
                                    output_dict["compartment"].append(compartment)
                                    output_dict["channel"].append(channel)
                                    output_dict["processor_type"].append(processor_type)
                                    output_dict["subdir_input"].append(subdir)
                                    output_dict["subdir_mask"].append(
                                        "segmentation_masks"
                                    )
                                    output_dict["subdir_output"].append(
                                        f"{subdir}_extracted_features"
                                    )
                            elif feature == "Texture":
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append("CPU")
                                output_dict["subdir_input"].append(subdir)
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    f"{subdir}_extracted_features"
                                )
                            elif feature == "SAMMed3D":
                                for processor_type in processor_types:
                                    output_dict["patient"].append(patient)
                                    output_dict["well_fov"].append(well_fov)
                                    output_dict["feature"].append(feature)
                                    output_dict["compartment"].append(compartment)
                                    output_dict["channel"].append(channel)
                                    output_dict["processor_type"].append(processor_type)
                                    output_dict["subdir_input"].append(subdir)
                                    output_dict["subdir_mask"].append(
                                        "segmentation_masks"
                                    )
                                    output_dict["subdir_output"].append(
                                        f"{subdir}_extracted_features"
                                    )
                            else:
                                raise ValueError(f"Unknown feature: {feature}")


# In[11]:


df = pd.DataFrame(output_dict)
print(f"Total combinations: {df.shape[0]}")
df.head()


# In[12]:


# number of combinations we should have
subdir_inputs = np.unique(df["subdir_input"])
# per well_fov
area_combos = len(image_set_loader.compartments) * len(processor_types)
coloc_combos = (
    len(channel_combinations)
    * len(image_set_loader.compartments)
    * len(processor_types)
)
intensity_combos = (
    len(image_set_loader.image_names)
    * len(image_set_loader.compartments)
    * len(processor_types)
)
granularity_combos = len(image_set_loader.image_names) * len(
    image_set_loader.compartments
)
SAMMed3D_combos = (
    len(image_set_loader.image_names)
    * len(image_set_loader.compartments)
    * len(processor_types)
)
neighbors_combos = 1  # Neighbors is always DNA and Nuclei
texture_combos = len(image_set_loader.image_names) * len(image_set_loader.compartments)
total_well_fov_combos = (
    area_combos
    + coloc_combos
    + intensity_combos
    + granularity_combos
    + neighbors_combos
    + texture_combos
    + SAMMed3D_combos
)
total_patient_well_fov_combos = len(np.unique(df["patient"] + "_" + df["well_fov"]))
total_combos = total_well_fov_combos * total_patient_well_fov_combos
total_convolution_subdirs = len(subdir_inputs)
total_combos = total_combos * total_convolution_subdirs
# print the total number of combinations
print(
    f"For {total_patient_well_fov_combos} patient-well_fov combinations, we have {total_combos} total combinations across all features."
)


# In[13]:


# write to a txt file with each row as a combination
# each column is a feature of the combination
df.to_csv(input_combinations_path, sep="\t", index=False)
