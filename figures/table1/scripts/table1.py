#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook
from IPython.display import Markdown, display

root_dir, in_notebook = init_notebook()


# In[2]:


sc_profiles_path = pathlib.Path(
    root_dir,
    "data/all_patient_profiles/0.normalized_profiles/sc_norm_norm_profile.parquet",
).resolve(strict=True)
organoid_profiles_path = pathlib.Path(
    root_dir,
    "data/all_patient_profiles/0.normalized_profiles/organoid_norm_norm_profile.parquet",
).resolve(strict=True)

patient_extra_metadata_path = pathlib.Path(
    root_dir,
    "config/patient_extra_metadata/patient_drug_screen_theoretical_counts_and_tumor_type"
    ".tsv",
).resolve(strict=True)
table2_file_info_path = pathlib.Path(
    root_dir,
    "figures/table2/results/table2/file_info_df.parquet",
).resolve(strict=True)
table1_results_path = pathlib.Path(
    root_dir,
    "figures/table1/results/table1_results.tsv",
).resolve()
table1_results_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


sc_df = pd.read_parquet(sc_profiles_path)
organoid_df = pd.read_parquet(organoid_profiles_path)

patient_extra_metadata_df = pd.read_csv(patient_extra_metadata_path, sep="\t")
file_info_df = pd.read_parquet(table2_file_info_path)


# ## Compound, treatment, well, well-FOV, organoid, and single-cell counts per patient

# In[4]:


# get the unique combinations of Metadata_Biology_PatientTumor and Metadata_Experiment_Treatment
compounds_counts = (
    organoid_df.groupby(
        ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Treatment"]
    )
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "count"})
    .drop(columns="count")
    .groupby(["Metadata_Biology_PatientTumor"])
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "number_of_compounds"})
)


# In[5]:


# get the unique combinations of Metadata_Biology_PatientTumor, Metadata_Experiment_Treatment, and Metadata_Experiment_Dose
treatments_counts = (
    organoid_df.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Dose",
        ]
    )
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "count"})
    .drop(columns="count")
    .groupby(["Metadata_Biology_PatientTumor"])
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "number_of_treatments"})
)


# In[6]:


# get the unique combinations of Metadata_Biology_PatientTumor and Metadata_Experiment_Well
well_counts = (
    sc_df.groupby(["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"])
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "count"})
    .drop(columns="count")
    .groupby(["Metadata_Biology_PatientTumor"])
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "number_of_wells"})
)


# In[7]:


well_fov_counts = (
    sc_df.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Well",
            "Metadata_Experiment_WellFOV",
        ]
    )
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "count"})
    .drop(columns="count")
    .groupby(["Metadata_Biology_PatientTumor"])
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "number_of_well_fovs"})
)


# In[8]:


organoid_counts = (
    sc_df.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Well",
            "Metadata_Experiment_WellFOV",
            "Metadata_Object_ParentOrganoid",
        ]
    )
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "count"})
    .drop(columns="count")
    .loc[sc_df["Metadata_Object_ParentOrganoid"] != -1]
    .groupby(["Metadata_Biology_PatientTumor"])
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "number_of_organoids"})
)


# In[9]:


single_cell_counts = (
    sc_df.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Well",
            "Metadata_Experiment_WellFOV",
        ]
    )
    .size()
    .to_frame()
    .reset_index()
    .rename(columns={0: "number_of_single_cells"})
    .groupby(["Metadata_Biology_PatientTumor"])
    .sum()
    .drop(
        columns=[
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Well",
            "Metadata_Experiment_WellFOV",
        ]
    )
    .reset_index()
)


# In[10]:


table1 = pd.merge(
    pd.merge(
        pd.merge(
            pd.merge(
                pd.merge(
                    compounds_counts,
                    treatments_counts,
                    on="Metadata_Biology_PatientTumor",
                ),
                well_counts,
                on="Metadata_Biology_PatientTumor",
            ),
            well_fov_counts,
            on="Metadata_Biology_PatientTumor",
        ),
        organoid_counts,
        on="Metadata_Biology_PatientTumor",
    ),
    single_cell_counts,
    on="Metadata_Biology_PatientTumor",
)

table1 = pd.merge(
    table1,
    patient_extra_metadata_df,
    left_on="Metadata_Biology_PatientTumor",
    right_on="patient",
    how="left",
).drop(columns=["patient"])

# remove the NF0037CQ1 patient from the table
# this is a test patient and we don't want to include it in the analysis
# different microscope was used
table1 = table1.loc[
    table1["Metadata_Biology_PatientTumor"] != "NF0037_T1_CQ1"
].reset_index(drop=True)


# ## Aggregate the raw image file info (from figures/table2) to the patient level

# In[11]:


file_info_counts = (
    file_info_df.groupby("patient")
    .agg(
        TotalImages=("z_dimension_size", "sum"),
        total_size_bytes=("file_size_bytes", "sum"),
    )
    .reset_index()
)
file_info_counts["TotalSize(TB)"] = (
    file_info_counts["total_size_bytes"] / (1024**4)
).round(2)
file_info_counts = file_info_counts.drop(columns=["total_size_bytes"])


# ## Combine patient/tumor level counts with image file counts and sizes

# In[12]:


table1 = pd.merge(
    table1,
    file_info_counts,
    left_on="Metadata_Biology_PatientTumor",
    right_on="patient",
    how="left",
).drop(columns=["patient"])


# In[13]:


tumor_type = table1.pop("Tumor_type")
table1.insert(1, "Tumor_type", tumor_type)
table1.rename(
    columns={
        "Metadata_Biology_PatientTumor": "Patient Tumor ",
        "Tumor_type": "Tumor type ",
        "number_of_compounds": "Compound Count",
        "number_of_treatments": "Treatment Count",
        "number_of_wells": "Well Count",
        "number_of_well_fovs": "Well FOV Count",
        "number_of_organoids": "Organoid Count",
        "number_of_single_cells": "Single Cell Count",
        "theoretical_number_of_compounds": "Theoretical Compound Count",
        "theoretical_number_of_treatments": "Theoretical Treatment Count",
        "theoretical_number_of_well_fovs": "Theoretical Well FOV Count",
        "TotalImages": "Total Image Count",
        "TotalSize(TB)": "Total Size (TB)",
    },
    inplace=True,
)


# In[14]:


table1["Total size (TB)"] = (
    table1["Total Size (TB)"] + (table1["Total Size (TB)"] / 5) * 4
).round(2)

table1 = table1.drop(
    columns=[
        "Compound Count",
        "Well Count",
        "Theoretical Compound Count",
        "Theoretical Treatment Count",
        "Theoretical Well FOV Count",
        "Total Size (TB)",
    ]
)

# add a total row to the table
total_row = pd.DataFrame(
    {
        "Patient Tumor ": ["Total"],
        "Tumor type ": ["-"],
        "Treatment Count": [table1["Treatment Count"].sum()],
        "Well FOV Count": [table1["Well FOV Count"].sum()],
        "Organoid Count": [table1["Organoid Count"].sum()],
        "Single Cell Count": [table1["Single Cell Count"].sum()],
        "Total Image Count": [table1["Total Image Count"].sum()],
        "Total size (TB)": [table1["Total size (TB)"].sum().round(2)],
    }
)
table1 = pd.concat([table1, total_row], ignore_index=True)
table1.to_csv(table1_results_path, index=False, sep="\t")
table1


# In[15]:


# convert the table to a markdown table
table1_md = table1.to_markdown(index=False, tablefmt="pipe")


# In[16]:


# Display as formatted markdown
print("Rendered Table:")
display(Markdown(table1_md))
