#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import pathlib
from itertools import product

import numpy as np
import pandas as pd

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

from loading_classes import ImageSetLoader

# In[2]:


patient_id_file = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()

input_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/input_combinations.txt"
)
input_combinations_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


features = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "Texture",
]


# In[4]:


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


# In[5]:


# example image set path to get the image set loader working
image_set_path = pathlib.Path(f"{root_dir}/data/NF0014_T1/profiling_input_images/C2-1/")
image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[6]:


output_dict = {
    "patient": [],
    "well_fov": [],
    "feature": [],
    "compartment": [],
    "channel": [],
    "processor_type": [],
}
processor_types = [
    "CPU",
    # "GPU"
]


# In[7]:


# get all channel combinations
channel_combinations = list(itertools.combinations(image_set_loader.image_names, 2))


# In[8]:


for patient in patients:
    # get the well_fov for each patient
    patient_well_fovs = pathlib.Path(
        f"{root_dir}/data/{patient}/profiling_input_images/"
    ).glob("*")
    for well_fov in patient_well_fovs:
        well_fov = well_fov.name
        print(f"Processing patient: {patient}, well_fov: {well_fov}")

        for feature in features:
            if feature == "Neighbors":
                output_dict["patient"].append(patient)
                output_dict["well_fov"].append(well_fov)
                output_dict["feature"].append("Neighbors")
                output_dict["compartment"].append("Nuclei")
                output_dict["channel"].append("DNA")
                output_dict["processor_type"].append("CPU")
            for compartment in image_set_loader.compartments:
                if feature == "AreaSizeShape":
                    for processor_type in processor_types:
                        output_dict["patient"].append(patient)
                        output_dict["well_fov"].append(well_fov)
                        output_dict["feature"].append("AreaSizeShape")
                        output_dict["compartment"].append(compartment)
                        output_dict["channel"].append("DNA")
                        output_dict["processor_type"].append(processor_type)
                elif feature == "Colocalization":
                    for channel in channel_combinations:
                        for processor_type in processor_types:
                            output_dict["patient"].append(patient)
                            output_dict["well_fov"].append(well_fov)
                            output_dict["feature"].append("Colocalization")
                            output_dict["compartment"].append(compartment)
                            output_dict["channel"].append(channel[0] + "." + channel[1])
                            output_dict["processor_type"].append(processor_type)
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
                        elif feature == "Intensity":
                            for processor_type in processor_types:
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append(processor_type)
                        elif feature == "Texture":
                            output_dict["patient"].append(patient)
                            output_dict["well_fov"].append(well_fov)
                            output_dict["feature"].append(feature)
                            output_dict["compartment"].append(compartment)
                            output_dict["channel"].append(channel)
                            output_dict["processor_type"].append("CPU")
                        else:
                            raise ValueError(f"Unknown feature: {feature}")


# In[9]:


df = pd.DataFrame(output_dict)
print(f"Total combinations: {df.shape[0]}")
df.head()


# In[10]:


# number of combinations we should have
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
neighbors_combos = 1  # Neighbors is always DNA and Nuclei
texture_combos = len(image_set_loader.image_names) * len(image_set_loader.compartments)
total_well_fov_combos = (
    area_combos
    + coloc_combos
    + intensity_combos
    + granularity_combos
    + neighbors_combos
    + texture_combos
)
total_patient_well_fov_combos = len(np.unique(df["patient"] + "_" + df["well_fov"]))
total_combos = total_well_fov_combos * total_patient_well_fov_combos
# print the total number of combinations
print(
    f"For {total_patient_well_fov_combos} patient-well_fov combinations, we have {total_combos} total combinations across all features."
)


# In[11]:


# write to a txt file with each row as a combination
# each column is a feature of the combination
df.to_csv(input_combinations_path, sep="\t", index=False)
