#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import tomli
import tqdm
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from microfilm.microplot import Micropanel, microshow

# In[2]:


OVERWRITE_COMPOSITES = False
WHOLE_MONTAGE = True
PIXEL_SIZE_MICRONS = 0.106
SCALE_BAR_LENGTH_MICRONS = 10


# In[3]:


PERCENTILE_CLIP = 99.9

# DNA -> cyan, Mito -> magenta, AGP -> green, ER -> red
CMAP_NAMES = {
    "DNA": "pure_cyan",
    "Mito": "pure_magenta",
    "AGP": "pure_green",
    "ER": "pure_red",
}
CHANNEL_ORDER = ["DNA", "ER", "AGP", "Mito"]


def _channel_limits(image: np.ndarray, contrast_max: float) -> list:
    """
    Convert a (min, max) contrast-dict entry, expressed on an output 0-255
    display scale, into the raw-intensity "limits" microfilm's "limits"
    rescale expects (which maps [0, limit] -> [0, 1] directly).
    """
    p999 = np.percentile(image, PERCENTILE_CLIP)
    return (
        [0.0, float(p999 * 255 / contrast_max)] if contrast_max else [0.0, float(p999)]
    )


def generate_montage(
    pixel_size_microns: float = PIXEL_SIZE_MICRONS,
    bar_length_microns: float = SCALE_BAR_LENGTH_MICRONS,
    full_montage: bool = True,
    composite_only: bool = False,
    patient_label: str = None,
    treatment_label: str = None,
    **kwargs,
):
    """Generate a montage of the 4 channels with a scale bar added to the composite RGB image.

    Parameters
    ----------
    pixel_size_microns: float
        Size of each pixel in microns
    bar_length_microns: float
        Length of the scale bar in microns
    patient_label: str
        If provided, burned into the upper-left corner of the composite image
        (e.g. the patient name)
    treatment_label: str
        If provided, burned into the lower-left corner of the composite image
        (e.g. the drug and dose)
    kwargs: dictionary containing the following keys:
        agp_image_path: pathlib.Path
            file path to the AGP channel image
        dna_image_path: pathlib.Path
            file path to the DNA channel image
        er_image_path: pathlib.Path
            file path to the ER channel image
        mito_image_path: pathlib.Path
            file path to the Mito channel image
        contrast_dict: dictionary containing contrast limits for each channel (optional, if not provided the full 0-255 output range is used)


    Returns
    -------
    microfilm.microplot.Micropanel or microfilm.microplot.Microimage
        Micropanel (full_montage) with the 4 channels plus composite, or a
        single Microimage (composite_only). Both expose a .savefig(path) method.
    """

    if composite_only and full_montage:
        raise ValueError(
            "Cannot set both composite_only and full_montage to True. Please choose one or the other."
        )

    channel_images = {
        "AGP": tifffile.imread(kwargs["agp_image_path"]).astype(np.float32),
        "DNA": tifffile.imread(kwargs["dna_image_path"]).astype(np.float32),
        "ER": tifffile.imread(kwargs["er_image_path"]).astype(np.float32),
        "Mito": tifffile.imread(kwargs["mito_image_path"]).astype(np.float32),
    }
    contrast_dict = kwargs.get("contrast_dict", None)

    channel_limits = {
        key: _channel_limits(
            image, contrast_dict.get(key, (0, 255))[1] if contrast_dict else 255
        )
        for key, image in channel_images.items()
    }

    scalebar_kwargs = dict(
        unit="um",
        scalebar_unit_per_pix=pixel_size_microns,
        scalebar_size_in_units=bar_length_microns,
        scalebar_color="white",
    )

    composite_microim = microshow(
        images=[channel_images[key] for key in CHANNEL_ORDER],
        cmaps=[CMAP_NAMES[key] for key in CHANNEL_ORDER],
        rescale_type="limits",
        limits=[channel_limits[key] for key in CHANNEL_ORDER],
        proj_type="sum",
        **scalebar_kwargs,
    )

    # burn the patient name / drug+dose directly into the composite so each
    # saved tile is self-labeled, independent of any grid it later sits in
    if patient_label is not None:
        composite_microim.add_label(
            patient_label,
            label_name="patient",
            label_location="upper left",
            label_color="white",
            label_font_size=14,
            label_kwargs={"weight": "bold"},
        )
    if treatment_label is not None:
        composite_microim.add_label(
            treatment_label,
            label_name="treatment",
            label_location="lower left",
            label_color="white",
            label_font_size=14,
            label_kwargs={"weight": "bold"},
        )

    if composite_only and not full_montage:
        return composite_microim

    # full_montage: the 4 individual channels plus the composite, as equal-sized
    # sub-montages in a single-row Micropanel
    panel = Micropanel(
        rows=1, cols=len(CHANNEL_ORDER) + 1, figsize=(4 * (len(CHANNEL_ORDER) + 1), 4)
    )
    for col, key in enumerate(CHANNEL_ORDER):
        microim = microshow(
            images=channel_images[key],
            cmaps=CMAP_NAMES[key],
            rescale_type="limits",
            limits=channel_limits[key],
            **scalebar_kwargs,
        )
        panel.add_element([0, col], microim)
    panel.add_element([0, len(CHANNEL_ORDER)], composite_microim)
    return panel


# ## Get the paths

# In[4]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
image_base_dir = image_base_dir / "data"

drugs_plate_map_path = pathlib.Path(f"{root_dir}/config/platemaps/platemap2.csv")
drugs_df = pd.read_csv(drugs_plate_map_path)


# In[5]:


output_dict = {
    "patient": [],
    "well_fov": [],
    "AGP_file_path": [],
    "DNA_file_path": [],
    "ER_file_path": [],
    "Mito_file_path": [],
    "Trans_file_path": [],
}
patients = image_base_dir.glob("*")
patients = [
    pathlib.Path(f"{p}/2D_analysis/0a.zmax_proj") for p in patients if p.is_dir()
]
# remove if any path does not exist
patients = [p for p in patients if p.exists()]
# remove patient NF0037_T1_CQ1
patients = [p for p in patients if p.name != "NF0037_T1_CQ1"]
for patient_path in tqdm.tqdm(
    patients, desc="Processing patients", total=len(patients), leave=False
):
    well_fovs = patient_path.glob("*")
    well_fovs = [f for f in well_fovs if f.is_dir()]
    for well_fov in tqdm.tqdm(
        well_fovs, desc="Processing well/fov", total=len(well_fovs), leave=False
    ):
        if well_fov.name in ["run_stats"]:
            continue
        files = [x for x in well_fov.glob("*") if x.is_file()]
        for f in tqdm.tqdm(
            files, desc="Processing files", total=len(files), leave=False
        ):
            if "555" in f.name:
                output_dict["AGP_file_path"].append(str(f))
            elif "405" in f.name:
                output_dict["DNA_file_path"].append(str(f))
            elif "488" in f.name:
                output_dict["ER_file_path"].append(str(f))
            elif "640" in f.name:
                output_dict["Mito_file_path"].append(str(f))
            elif "TRANS" in f.name:
                output_dict["Trans_file_path"].append(str(f))

        output_dict["patient"].append(patient_path.parent.parent.name)
        output_dict["well_fov"].append(well_fov.name)
df = pd.DataFrame(output_dict)
df.insert(2, "well", df["well_fov"].apply(lambda x: x.split("-")[0]))
df = df.merge(drugs_df, left_on=["well"], right_on=["WellPosition"], how="left")
df.dropna(inplace=True)
df["Dose"] = df["Dose"].astype(int)
df["full_treatment_name"] = df["Treatment"] + " " + df["Dose"].astype(str) + df["Unit"]
df["full_treatment_name_machine_readable"] = (
    df["Treatment"] + "_" + df["Dose"].astype(str) + df["Unit"]
)
df.head()


# ## merge the parameters and the file paths

# In[6]:


config_file_path = pathlib.Path(
    f"{root_dir}/figures/montage/config/montage_config.toml"
).resolve(strict=True)
# read in the config file
# each treatment group is an array of tables ([[group]]); tomli parses each
# group into a list of self-contained record dicts, one per example, so
# entries can be added/removed/reordered independently of one another
montage_config = tomli.load(config_file_path.open("rb"))

REQUIRED_FIELDS = ["patient", "well_fov"]
CONTRAST_FIELDS = {"agp_max": 255, "dna_max": 255, "er_max": 255, "mito_max": 255}

list_of_dfs = []
for treatment_group, entries in montage_config.items():
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"[[{treatment_group}]] entry {entry} is missing required field(s): {missing}"
            )
    group_df = pd.DataFrame(entries)
    group_df.insert(0, "treatment_group", treatment_group)
    # default any omitted contrast field to the full display range (255)
    for field, default_value in CONTRAST_FIELDS.items():
        if field not in group_df.columns:
            group_df[field] = default_value
        else:
            group_df[field] = group_df[field].fillna(default_value)
    list_of_dfs.append(group_df)
all_examples_df = pd.concat(list_of_dfs, ignore_index=True)
sampled_df = df.merge(all_examples_df, on=["patient", "well_fov"], how="inner")
print(sampled_df.shape)
# sort by patient for easier visualization
sampled_df = sampled_df.sort_values(by="patient")


# In[7]:


# generate montages for the example well_fovs listed above with individual contrast adjustments for each channel
for idx, row in tqdm.tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
    composite_image_path = pathlib.Path(
        f"../montages/individual_composites/{row['patient']}_{row['well_fov']}_{row['full_treatment_name_machine_readable']}_composite.png"
    ).resolve()
    composite_image_path.parent.mkdir(parents=True, exist_ok=True)
    if composite_image_path.exists() and not OVERWRITE_COMPOSITES:
        continue
    elif not composite_image_path.exists() or OVERWRITE_COMPOSITES:
        fig = generate_montage(
            agp_image_path=row["AGP_file_path"],
            dna_image_path=row["DNA_file_path"],
            er_image_path=row["ER_file_path"],
            mito_image_path=row["Mito_file_path"],
            composite_only=True,
            full_montage=False,
            contrast_dict={
                "AGP": (0, row["agp_max"]),
                "DNA": (0, row["dna_max"]),
                "ER": (0, row["er_max"]),
                "Mito": (0, row["mito_max"]),
            },
            patient_label=row["patient"],
            treatment_label=row["full_treatment_name"],
        )
        fig.savefig(
            composite_image_path,
            dpi=600,
        )


# ## Combine the individual composites into a single grid montage

# In[8]:


if WHOLE_MONTAGE:
    composite_dir = pathlib.Path("../montages/individual_composites").resolve()

    # keep the whole montage readable: a handful of patient/well_fovs crossed with a
    # few treatments, rather than every composite that was generated above
    WHOLE_MONTAGE_N_PATIENT_WELL_FOVS = 10  # 6-10
    WHOLE_MONTAGE_TREATMENT_GROUPS = [
        "dmso_examples",
        "selumetnib_1um_examples",
        "trametinib_1um_examples",
        "staurosporine_10nm_examples",
    ]

    montage_df = sampled_df[
        sampled_df["treatment_group"].isin(WHOLE_MONTAGE_TREATMENT_GROUPS)
    ]

    # each treatment group uses its own well_fov per patient (not a shared one),
    # so a column is keyed on patient alone; only keep patients that have an
    # example for every selected treatment group so no grid cell is ever blank
    coverage = montage_df.groupby("patient")["treatment_group"].nunique()
    full_coverage_patients = coverage[
        coverage == len(WHOLE_MONTAGE_TREATMENT_GROUPS)
    ].index
    selected_patients = sorted(full_coverage_patients)[
        :WHOLE_MONTAGE_N_PATIENT_WELL_FOVS
    ]

    # human-readable drug/dose name per treatment group (e.g. "Selumetinib 1uM"),
    # for the row labels
    drug_labels = {
        treatment_group: montage_df.loc[
            montage_df["treatment_group"] == treatment_group, "full_treatment_name"
        ].iloc[0]
        for treatment_group in WHOLE_MONTAGE_TREATMENT_GROUPS
    }

    # horizontal layout: drug/treatment as rows (few) and patient as columns
    # (many), so the figure is wide rather than tall
    n_rows = len(WHOLE_MONTAGE_TREATMENT_GROUPS)
    n_cols = len(selected_patients)

    # each pre-rendered composite PNG is already a finished sub-montage, so
    # Micropanel is used purely to arrange them into an evenly spaced,
    # full-bleed grid (it plots directly on panel.ax, bypassing microshow
    # since these are already-colored RGB images, not raw channel data)
    panel = Micropanel(rows=n_rows, cols=n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for row_idx, treatment_group in enumerate(WHOLE_MONTAGE_TREATMENT_GROUPS):
        for col_idx, patient in enumerate(selected_patients):
            ax = panel.ax[row_idx, col_idx]
            match = montage_df[
                (montage_df["patient"] == patient)
                & (montage_df["treatment_group"] == treatment_group)
            ]
            composite_image_path = None
            if not match.empty:
                row = match.iloc[0]
                composite_image_path = (
                    composite_dir
                    / f"{row['patient']}_{row['well_fov']}_{row['full_treatment_name_machine_readable']}_composite.png"
                )
            if composite_image_path is not None and composite_image_path.exists():
                image = plt.imread(composite_image_path)
                ax.imshow(image, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(patient, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(drug_labels[treatment_group], fontsize=9)

    whole_montage_path = pathlib.Path("../montages/Whole_montage.png").resolve()
    panel.savefig(whole_montage_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(panel.fig)
