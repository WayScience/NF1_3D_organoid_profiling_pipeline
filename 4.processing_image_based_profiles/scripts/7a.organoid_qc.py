#!/usr/bin/env python
# coding: utf-8

# # 7a. Organoid QC
#
# ## Purpose
# Flag low-quality organoids per patient using two criteria applied in sequence:
# 1. **NaN detection** — organoids missing key metadata or feature values
# 2. **Size outliers** — abnormally small or large organoids by volume (z-score)
#
# This is **step 7a of Stage 4 (image-based profiling)**. It runs once per patient
# and must complete before `7b.single_cell_qc.ipynb`, which inherits organoid flags.
#
# ## Inputs
# - `data/{patient}/image_based_profiles/3.annotated_profiles/organoid_anno.parquet`
#
# ## Outputs
# - `data/{patient}/image_based_profiles/4.qc_profiles/organoid_flagged_outliers.parquet`
#   — original organoid profile with three added `Metadata_cqc_*` flag columns
#
# ## Notes
# - QC flags are **additive**: an organoid can be flagged by multiple criteria simultaneously.
# - Outlier detection only runs on the subset of organoids that passed the NaN check,
#   so NaN rows are never evaluated for size outliers.

# In[1]:


import json
import os
import pathlib
import re
import tempfile

import pandas as pd
import pyvista as pv
from cosmicqc import find_outliers
from cytodataframe import CytoDataFrame
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
# NOTE: previously this line unconditionally overrode bandicoot_check()
# with root_dir, meaning bandicoot was never actually used even when
# mounted. Removed so bandicoot_check()'s own bandicoot-first behavior
# takes effect.


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    image_based_profiles_subparent_name = "image_based_profiles"
    patient = "SARCO361_T1"


# In[3]:


# Serve the interactive trame views on a fixed port so VS Code Remote-SSH can
# forward it to the same local port (see .vscode/settings.json). This must be
# set before the first view renders; restart the kernel if it was already started.
pv.global_theme.trame.jupyter_server_port = 8686

# The TIFFs carry no voxel-size metadata, so the 3D views take voxel size (um) from
# the profiles' Metadata_Microscopy_*ResolutionUm columns (set in 6.annotation), which
# keeps the scale bar consistent with the um-based features.
RESOLUTION_COLUMNS = [f"Metadata_Microscopy_{axis}ResolutionUm" for axis in "XYZ"]
SCALE_BAR_LENGTHS_UM = (1, 2, 5, 10, 20, 50, 100)


CHANNEL = "AGP"
CHANNEL_CODE = "555"
COMPARTMENT = "Organoid"
# CytoDataFrame's `scale_bar` display option only draws on 2D crops, so wrap the hook
# that adds the mask overlay to each 3D plotter (interactive view and static snapshot)
# to also draw a scale bar. Keep the original on the class so re-running this cell
# does not stack wrappers.
if not hasattr(CytoDataFrame, "_orig_add_label_overlay_to_plotter"):
    CytoDataFrame._orig_add_label_overlay_to_plotter = (
        CytoDataFrame._add_label_overlay_to_plotter
    )


def add_label_overlay_and_scale_bar(self, plotter, volume, spacing, **kwargs):
    """Add the mask overlay, then an XY scale bar in um below the crop."""
    overlay_actors = CytoDataFrame._orig_add_label_overlay_to_plotter(
        self, plotter=plotter, volume=volume, spacing=spacing, **kwargs
    )
    # volume is (z, y, x) and the plotter's world units are um via volume_spacing,
    # so the bar stays true to scale when the view is rotated or zoomed
    n_z, n_y, n_x = volume.shape
    width_um = (n_x - 1) * spacing[0]
    length_um = max(
        (length for length in SCALE_BAR_LENGTHS_UM if length <= width_um / 2),
        default=SCALE_BAR_LENGTHS_UM[0],
    )
    # draw on the top z-plane, just outside the crop, so the volume does not hide it
    bar_y_um = -0.1 * (n_y - 1) * spacing[1]
    bar_z_um = (n_z - 1) * spacing[2]
    plotter.add_mesh(
        pv.Line((0.0, bar_y_um, bar_z_um), (length_um, bar_y_um, bar_z_um)),
        color="white",
        line_width=6,
        render_lines_as_tubes=True,
    )
    plotter.add_point_labels(
        [(length_um / 2, 2 * bar_y_um, bar_z_um)],
        [f"{length_um} µm"],
        show_points=False,
        shape=None,
        text_color="white",
        font_size=12,
        always_visible=True,
    )
    return overlay_actors


CytoDataFrame._add_label_overlay_to_plotter = add_label_overlay_and_scale_bar


bbox_column_map = {
    "x_min": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MinX",
    "x_max": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MaxX",
    "y_min": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MinY",
    "y_max": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MaxY",
    "z_min": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MinZ",
    "z_max": f"{COMPARTMENT}_NoChannel_VolumeSizeShape_MaxZ",
}
center_columns = [
    f"{COMPARTMENT}_NoChannel_VolumeSizeShape_Center{axis}" for axis in "XYZ"
]

mask_name = f"{COMPARTMENT.lower()}_mask.tiff"
# scratch directory (not part of the repo) holding per-well-FOV mask symlinks
mask_link_dir = (
    pathlib.Path(tempfile.gettempdir()) / "cytodataframe_nf1_3d_mask_links" / patient
)
mask_link_dir.mkdir(parents=True, exist_ok=True)


def stage_mask(well_fov: str) -> pathlib.Path:
    """
    Description
    -----------
    Symlink a well-FOV's mask into mask_link_dir under a unique name.

    Parameters
    ----------
    well_fov : str
        The well-FOV for which to stage the mask.

    Returns
    -------
    pathlib.Path
        The path to the staged mask.
    """
    channel_path = pathlib.Path(
        profile_base_dir
        / "data"
        / f"{patient}"
        / "zstack_images"
        / f"{well_fov}"
        / f"{well_fov}_{CHANNEL_CODE}.tif"
    )
    mask_path = pathlib.Path(
        profile_base_dir
        / "data"
        / f"{patient}"
        / "segmentation_masks"
        / f"{well_fov}"
        / mask_name
    ).resolve(strict=True)
    link = mask_link_dir / f"{channel_path.stem}__{mask_name}"
    if not link.exists():
        link.symlink_to(mask_path)
    return link


def make_voxel_view(
    profiles_df: pd.DataFrame, list_of_columns_to_include: list
) -> CytoDataFrame:
    """
    Description
    -----------
    Build a CytoDataFrame 3D voxel view (with mask overlay) of organoid rows.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        The DataFrame containing the organoid profiles.
    list_of_columns_to_include : list
        The list of columns to include in the CytoDataFrame.

    Returns
    -------
    CytoDataFrame
        The CytoDataFrame with the 3D voxel view.
    """
    profiles_df = profiles_df.copy()
    # metadata stores well-FOV as e.g. C11_4, directories on disk use C11-4
    well_fovs = profiles_df["Metadata_Experiment_WellFOV"].str.replace("_", "-")
    profiles_df[f"Image_FileName_{CHANNEL}"] = [
        str(
            profile_base_dir
            / "data"
            / f"{patient}"
            / "zstack_images"
            / f"{well_fov}"
            / f"{well_fov}_{CHANNEL_CODE}.tif"
        )
        for well_fov in well_fovs
    ]
    for well_fov in well_fovs.unique():
        stage_mask(well_fov)

    resolutions = profiles_df[RESOLUTION_COLUMNS].drop_duplicates()
    if len(resolutions) != 1:
        raise ValueError(
            f"Expected one voxel size across rows, found:\n{resolutions.to_string()}"
        )
    voxel_spacing = tuple(float(value) for value in resolutions.iloc[0])

    return CytoDataFrame(
        data=profiles_df[list_of_columns_to_include],
        data_bounding_box=profiles_df[list(bbox_column_map.values())],
        compartment_center_xy=profiles_df[center_columns],
        data_mask_context_dir=str(mask_link_dir),
        segmentation_file_regex={rf"__{re.escape(mask_name)}$": r"_\d+\.tif$"},
        display_options={
            "width": 260,
            "height": 260,
            "table_max_height": "580px",
            "label_overlay_mode": "filled",
            # Voxel size (x, y, z) in um; also sets the scale bar's units.
            "volume_spacing": voxel_spacing,
            "volume_bbox_column_map": bbox_column_map,
            "label_overlay_color": (128, 128, 128),  # grey
            "label_overlay_opacity": 0.2,
            "label_overlay_toggle": True,
            "label_overlay_toggle_position": "top-right",
            "label_overlay_toggle_vertical_offset": 10,
            "label_overlay_toggle_label": "Mask",
            "label_overlay_toggle_font_size": 9,
            "label_overlay_toggle_label_gap": 24,
            "label_overlay_toggle_label_shift_left": 212,
        },
    )


# In[4]:


# Per-patient small-organoid outlier z-score thresholds. Patients are tuned individually
# by visually inspecting flagged organoids and adjusting their entry in this file.
small_outlier_thresholds_path = (
    root_dir
    / "4.processing_image_based_profiles"
    / "data"
    / "qc_thresholds"
    / "organoid_small_outlier_thresholds.json"
).resolve(strict=True)
with open(small_outlier_thresholds_path) as f:
    small_outlier_thresholds = json.load(f)

if patient not in small_outlier_thresholds:
    raise ValueError(
        f"No small organoid outlier threshold configured for patient '{patient}' in "
        f"{small_outlier_thresholds_path}. Add an entry for this patient before running QC."
    )

small_outlier_threshold = small_outlier_thresholds[patient]
print(
    f"Using small organoid outlier threshold for {patient}: {small_outlier_threshold}"
)


# ## Load in all the organoid profiles and concat together

# In[5]:


organoid_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles"
    / "organoid_anno.parquet"
).resolve(strict=True)

sammed_annotated_organoid_profiles_path = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles"
    / "sammed_organoid_anno.parquet"
).resolve()

qc_output_dir = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles"
)
qc_output_dir.mkdir(parents=True, exist_ok=True)

organoid_qc_output_path = f"{qc_output_dir}/organoid_flagged_outliers.parquet"
sammed_organoid_qc_output_path = (
    f"{qc_output_dir}/sammed_organoid_flagged_outliers.parquet"
)

orig_organoid_profiles_df = pd.read_parquet(organoid_file)

# Print the shape and head of the combined organoid profiles DataFrame
print(orig_organoid_profiles_df.shape)
orig_organoid_profiles_df.head()


# ## (Sanity check) Round 1 QC: flag rows with NaN in key columns
#
# `Metadata_cqc_*` columns are boolean flags added by this notebook. A value of `True`
# means the organoid failed that criterion. Multiple flags can be True simultaneously.
#
# We flag organoids where `ObjectID`, `SingleCellCount`, or `Volume` is NaN because:
# - An organoid with no cells (`SingleCellCount` NaN) cannot be a valid profile row.
# - A NaN `ObjectID` means the object does not exist and all features will be NaN.
# - A NaN `Volume` means the core morphology feature is missing.

# In[6]:


organoid_profiles_df = orig_organoid_profiles_df.copy()
organoid_profiles_df["Metadata_cqc_nan_detected"] = (
    organoid_profiles_df[
        [
            "Metadata_Object_ObjectID",
            "Metadata_Object_OrganoidSingleCellCount",
            "Organoid_NoChannel_VolumeSizeShape_Volume",
        ]
    ]
    .isna()
    .any(axis=1)
)
# Print the number of organoids flagged
flagged_count = organoid_profiles_df["Metadata_cqc_nan_detected"].sum()
print(f"Number of organoids flagged: {flagged_count}")


# ## Process non-NaN rows to detect abnormally small and large organoids and flag them

# In[7]:


# Set the metadata columns to be used in the QC process
metadata_columns = [x for x in organoid_profiles_df.columns if "Metadata" in x]


# In[8]:


## Round 2 QC: size-based outlier detection

# `find_outliers` uses z-score thresholds: negative values flag objects below the mean,
# positive values flag objects above. Threshold magnitude is the number of standard
# deviations from the mean. Only non-NaN rows (from Round 1) are evaluated.

# Only process the rows that are not flagged
filtered_profile_df = organoid_profiles_df[
    ~organoid_profiles_df["Metadata_cqc_nan_detected"]
]

# Find outlier organoids based on the 'Volume.Size.Shape_Organoid_VOLUME' column
print("Finding small organoid outliers...")
small_size_outliers = find_outliers(
    df=filtered_profile_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Organoid_NoChannel_VolumeSizeShape_Volume": small_outlier_threshold,
    },
)
# Ensure the column exists before assignment
organoid_profiles_df["Metadata_cqc_small_organoid_outlier"] = False
organoid_profiles_df.loc[
    small_size_outliers.index, "Metadata_cqc_small_organoid_outlier"
] = True

# Print number of outliers (only in filtered rows)
small_count = filtered_profile_df.index.intersection(small_size_outliers.index).shape[0]
print(f"Small organoid outliers found: {small_count}")
small_organoid_profiles = organoid_profiles_df.loc[
    organoid_profiles_df["Metadata_cqc_small_organoid_outlier"]
]
organoid_profiles_df.to_parquet(organoid_qc_output_path, index=False)


# ## Merge the qc flags to the deep learning-based profiles and save the output
# Merge the QC flags back to the original organoid profiles, which will be used in downstream analyses and single cell QC.
# We need to do this beacuase we do not run qc on black-box features.
# Merge on the Metadata_Biology_PatientTumor, Metadata_Experiment_WellFOV
# and the Metadata_Object_ObjectID columns, which together uniquely identify each organoid profile row.

# In[9]:


sammed_organoid_df = pd.read_parquet(sammed_annotated_organoid_profiles_path)
original_sammed_shape = sammed_organoid_df.shape
# set the merge keys to int for both dataframes to ensure they match
merge_keys = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_WellFOV",
    "Metadata_Object_ObjectID",
]
qc_keys = [col for col in organoid_profiles_df.columns if "Metadata_cqc" in col]


# merge the flagged organoid profiles with the sammed annotated organoid profiles
qc_annotated_sammed_organoid_df = sammed_organoid_df.merge(
    organoid_profiles_df[qc_keys + merge_keys],
    on=merge_keys,
    how="left",
)
if qc_annotated_sammed_organoid_df.shape[1] == original_sammed_shape[1]:
    raise ValueError(
        f"No new columns were added during the merge. Check that the merge keys {merge_keys} are correct and that the qc keys {qc_keys} are present in the organoid_profiles_df."
    )
qc_annotated_sammed_organoid_df.to_parquet(sammed_organoid_qc_output_path, index=False)


# ## Set up 3D voxel views of organoids with `CytoDataFrame`
#
# Adapted from the CytoDataFrame NF1 3D pilot verification example, using this notebook's pathing.
#
# - There are no `Image_FileName_*` columns, so each row's raw image path is built from
#   the patient and well-FOV metadata.
# - Every well-FOV's mask file has the same generic name (`organoid_mask.tiff`), but
#   `data_mask_context_dir` only matches masks by filename pattern within one directory.
#   `stage_mask` fills a scratch directory with per-well-FOV symlinks to the real masks,
#   renamed to embed each well-FOV's identifier, so matching works.

# ### Visualize the small organoids

# In[10]:


# backend="server" renders server-side and streams images (no client-side geometry
# sync, which fails with many views per table), so the in-view "Mask" checkbox
# toggles the overlay on/off; a red dot marks each object's center. Views stay blank
# unless port 8686 is forwarded to the same local port.
if in_notebook:
    display(
        make_voxel_view(
            small_organoid_profiles,
            [
                "Metadata_Experiment_WellFOV",
                "Metadata_cqc_small_organoid_outlier",
                f"Image_FileName_{CHANNEL}",
            ],
        ).show_widget_table(column=f"Image_FileName_{CHANNEL}", backend="server")
    )


# ### Visualize a random selection of organoids

# In[11]:


# backend="server" renders server-side and streams images (no client-side geometry
# sync, which fails with many views per table), so the in-view "Mask" checkbox
# toggles the overlay on/off; a red dot marks each object's center. Views stay blank
# unless port 8686 is forwarded to the same local port.
if in_notebook:
    display(
        make_voxel_view(
            organoid_profiles_df,
            [
                "Metadata_Experiment_WellFOV",
                "Metadata_cqc_small_organoid_outlier",
                f"Image_FileName_{CHANNEL}",
            ],
        ).show_widget_table(column=f"Image_FileName_{CHANNEL}", backend="server")
    )
