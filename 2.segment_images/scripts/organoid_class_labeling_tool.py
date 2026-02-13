#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import json
import os
import pathlib
import sys
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
import tifffile
import torch
from arg_parsing_utils import check_for_missing_args, parse_args
from cellpose import models
from file_reading import *
from file_reading import read_zstack_image
from general_segmentation_utils import *
from notebook_init_utils import bandicoot_check, init_notebook
from organoid_segmentation import *
from segmentation_decoupling import *
from skimage.filters import sobel

# In[ ]:


def save_labels(dictionary: dict, outfile: str):
    """
    Description
    ----------
    Save labels to a parquet file.
    Parameters
    ----------
    dictionary : dict
        Dictionary containing labels to save.
    outfile : str
        Path to the output parquet file.
    Returns
    -------
    None

    """

    try:
        pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(dictionary)
        df.to_parquet(outfile, index=False)
    except Exception as e:
        print(f"Error saving labels to {outfile}: {e}")
        return False
    return True


def read_labels(infile: str) -> dict:
    """
    Description
    ----------
    Read labels df from a parquet file.
    Parameters
    ----------
    infile : str
        Path to the input parquet file.
    Returns
    -------
    dict
        Dictionary containing the labels.
    """
    data = pd.read_parquet(infile).to_dict(orient="list")
    return data


def clean_labels(labels: dict) -> dict:
    """
    This function removes any and all records that contain NaN values
    This ensures that the image can be re-annotated.

    Parameters
    ----------
    labels : dict
        Dictionary with the following structure
        {
            "patient": list of patient identifiers,
            "well_fov": list of well FOV identifiers,
            "label": list of labels assigned to each image,
                where labels are the morohology class annotated
                for a given well fov (e.g. "globular", "small/dissociated", etc.),
            "annotator": list of annotator names
        }

    Returns
    -------
    dict
        Updated dictionary with records containing NaN values removed
    """
    # convert to dataframe for easier cleaning
    labels_df = pd.DataFrame(labels)
    # if NaN in any rows, drop those rows
    labels_df = labels_df.dropna()
    # convert back to dictionary
    labels = labels_df.to_dict(orient="list")
    return labels


def check_for_image_labels(
    dictionary: dict,
    patient: str,
    well_fov: str,
    annotator: str,
) -> bool:
    """
    Description
    ----------
    Check if an image has already been labeled.
    Parameters
    ----------
    dictionary : dict
        Dictionary containing existing labels.
    patient : str
        Patient identifier.
    well_fov : str
        Well FOV identifier.
    annotator : str
        Annotator name.
    Returns
    -------
    bool
        True if the image has been labeled, False otherwise.
    """
    for i in range(len(dictionary["patient"])):
        if (
            dictionary["patient"][i] == patient
            and dictionary["well_fov"][i] == well_fov
        ):
            return True
    return False


# In[ ]:


def label_images_keypress(
    image_dict: dict,
    label_map: dict,
    labels_save_file: pathlib.Path,
    batch_size: int = 100,
    in_notebook: bool = True,
    annotator: str | None = None,
    input_mode: str = "auto",  # "auto", "keypress", "prompt"
    keypress_timeout: float = 15,  # seconds; 0 = wait forever
) -> dict:
    """
    Label images using keyboard input (with notebook-safe fallback).

    Parameters
    ----------
    in_notebook : bool
        If True, tries non-blocking display behavior for Jupyter.
    annotator : str | None
        Optional annotator name. If None, prompt user.
    input_mode : str
        "auto" -> use keypress if backend supports it, else prompt.
        "keypress" -> force keypress.
        "prompt" -> force text input prompt.
    keypress_timeout : float
        Seconds to wait for keypress before fallback to prompt (0 = no timeout).
    """
    if annotator is None:
        annotator = input("Enter annotator name: ")

    # Convert input to DataFrame for filtering
    image_df = pd.DataFrame(image_dict)

    labels = {"patient": [], "well_fov": [], "label": [], "annotator": []}
    if labels_save_file.exists():
        labels = read_labels(labels_save_file)

    labeled_pairs = set(zip(labels.get("patient", []), labels.get("well_fov", [])))

    if input_mode == "auto":
        input_mode = "prompt" if in_notebook else "keypress"
    elif in_notebook and input_mode == "keypress":
        input_mode = "prompt"

    for patient_id in tqdm.tqdm(
        np.unique(image_df["patient"]), desc="Processing patients", leave=True
    ):
        subset_patient_df = image_df[image_df["patient"] == patient_id].reset_index(
            drop=True
        )

        is_labeled = subset_patient_df.apply(
            lambda row: (row["patient"], row["well_fov"]) in labeled_pairs, axis=1
        )
        unlabeled_df = subset_patient_df[~is_labeled].reset_index(drop=True)

        if unlabeled_df.empty:
            continue

        total_images = len(unlabeled_df)
        counter = 0

        for batch_start in range(0, len(unlabeled_df), batch_size):
            batch_df = unlabeled_df.iloc[batch_start : batch_start + batch_size].copy()

            batch_df["image"] = [
                (lambda img: img[img.shape[0] // 2])(read_zstack_image(image_path))
                for image_path in tqdm.tqdm(
                    batch_df["image_path"],
                    desc="Loading images",
                    leave=False,
                )
            ]

            for i, image in enumerate(batch_df["image"]):
                counter += 1
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(batch_df["image"].iloc[i], cmap="inferno")
                ax.axis("off")
                key_pressed = {"key": None}

                def on_key(event):
                    key_pressed["key"] = event.key
                    plt.close(fig)

                if in_notebook:
                    plt.show()
                    key = input(
                        "Enter label for image: \n1=Globular\n2=Small/Dissociated\n3=Elongated\n"
                    ).strip()
                    plt.close(fig)
                else:
                    # connect key press handler
                    cid = fig.canvas.mpl_connect("key_press_event", on_key)

                    plt.show(block=False)

                    # wait for key press (timeout in seconds)
                    pressed = plt.waitforbuttonpress(timeout=10)

                    if pressed and key_pressed["key"] is not None:
                        key = key_pressed["key"]
                    else:
                        # fallback to input
                        key = input(
                            "Enter label for image: \n1=Globular\n2=Small/Dissociated\n3=Elongated\n"
                        ).strip()

                    fig.canvas.mpl_disconnect(cid)
                    plt.close(fig)

                labels["annotator"].append(annotator)
                labels["patient"].append(batch_df["patient"].iloc[i])
                labels["well_fov"].append(batch_df["well_fov"].iloc[i])
                labels["label"].append(label_map.get(key, None))
                save_labels(labels, labels_save_file)
                print(f"{counter / total_images:.2%} labeled")

            labeled_pairs = set(
                zip(labels.get("patient", []), labels.get("well_fov", []))
            )

    return labels


# In[4]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[5]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
patient_list_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)


# In[6]:


labels_save_file = pathlib.Path(
    "../image_labels/organoid_image_labels.parquet"
).resolve()
labels_save_file_bandicoot = pathlib.Path(
    f"{image_base_dir}/data/organoid_image_labels/organoid_image_labels.parquet"
).resolve()
labels_save_file.parent.mkdir(exist_ok=True, parents=True)


# In[7]:


save_labels(  # step 3: saves the cleaned labels back to the parquet file
    clean_labels(  # step 2: cleans the labels (removes records with NaN values)
        read_labels(labels_save_file)  # step 1: reads in the labels
    ),
    labels_save_file,  # path to save the cleaned labels back to the parquet file
)


# In[8]:


patients = pd.read_csv(patient_list_file_path, header=None)[0].tolist()
images_to_process = {"patient": [], "well_fov": [], "image_path": []}

for patient_id in tqdm.tqdm(patients):
    input_subparent_name = "zstack_images"
    image_dir = pathlib.Path(
        f"{image_base_dir}/data/{patient_id}/{input_subparent_name}"
    )
    well_fovs = sorted([d for d in image_dir.iterdir() if d.is_dir()])
    for well_fov_path in well_fovs:
        well_fov = well_fov_path.name

        images_to_process["patient"].append(patient_id)
        images_to_process["well_fov"].append(well_fov)
        images_to_process["image_path"].append(
            pathlib.Path(f"{image_dir}/{well_fov}/{well_fov}_555.tif")
        )

df = pd.DataFrame(images_to_process)
print(f"Images to process: {len(df)}")
df.head()


# ## run the rest via script (code cells won't be executed in notebook)

# In[ ]:


label_map = {"1": "globular", "2": "small/dissociated", "3": "elongated"}


# In[ ]:


labels = label_images_keypress(
    images_to_process,
    label_map,
    labels_save_file,
    in_notebook=in_notebook,
    batch_size=250,
)
# loads in all images for a single patient - might take a few minutes to load sinlge slices into memory
# this is to speed up the labeling process since we don't have to read in the image for each individual FOV
save_labels(labels, labels_save_file_bandicoot)


# In[ ]:


labels = read_labels(labels_save_file)
# show stats for the labeling
df = pd.DataFrame(labels)
print("Label counts:")
print(df["label"].value_counts())
# print a list of well fovs for each label
for label in label_map.values():
    well_fovs = df[df["label"] == label][["patient", "well_fov"]]
    print(f"\nWell FOVs for label '{label}':")
    counter = 0
    for index, row in well_fovs.iterrows():
        if counter >= 10:
            break
        print(f"{row['well_fov']}")
        counter += 1
