#!/usr/bin/env python
# coding: utf-8

# # 7b. Single-Cell QC
#
# ## Purpose
# Flag low-quality single cells per patient using three criteria applied in cascade:
# 1. **NaN detection** — cells missing key metadata or feature values
# 2. **Inherited organoid flags** — cells whose parent organoid was flagged in `7a`
# 3. **Nucleus outliers** — abnormally small/large nuclei or high mass displacement
#
# Outlier detection (step 3) only runs on cells that passed steps 1 and 2.
#
# This is **step 7b of Stage 4 (image-based profiling)**. It runs once per patient
# and depends on `7a.organoid_qc.ipynb` having run first.
#
# ## Inputs
# - `data/{patient}/image_based_profiles/3.annotated_profiles/sc_anno.parquet`
# - `data/{patient}/image_based_profiles/4.qc_profiles/organoid_flagged_outliers.parquet`
#
# ## Outputs
# - `data/{patient}/image_based_profiles/4.qc_profiles/sc_flagged_outliers.parquet`
#   — SC profile with added `Metadata_cqc_*` flag columns
#
# ## Notes
# - QC flags are additive: a cell can be flagged by multiple criteria simultaneously.
# - The `Metadata_cqc_organoid_flagged` column propagates organoid-level flags down
#   to all cells belonging to that organoid, linking 7a and 7b outputs.

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
# profile_base_dir = root_dir
print(profile_base_dir)


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "SARCO361_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


# Per-patient single-cell outlier z-score thresholds. Patients are tuned individually
# by visually inspecting flagged nuclei and adjusting their entry in this file.
sc_outlier_thresholds_path = (
    root_dir
    / "4.processing_image_based_profiles"
    / "data"
    / "qc_thresholds"
    / "single_cell_outlier_thresholds.json"
).resolve(strict=True)
with open(sc_outlier_thresholds_path) as f:
    sc_outlier_thresholds = json.load(f)

if patient not in sc_outlier_thresholds:
    raise ValueError(
        f"No single-cell outlier thresholds configured for patient '{patient}' in "
        f"{sc_outlier_thresholds_path}. Add an entry for this patient before running QC."
    )

patient_sc_thresholds = sc_outlier_thresholds[patient]
small_nuclei_threshold = patient_sc_thresholds["small_nuclei_volume"]
large_nuclei_threshold = patient_sc_thresholds["large_nuclei_volume"]
high_mass_displacement_threshold = patient_sc_thresholds["high_mass_displacement"]
print(f"Using single-cell outlier thresholds for {patient}: {patient_sc_thresholds}")


# ## Set up 3D voxel views of nuclei with `CytoDataFrame`
#
# Same setup as `7a.organoid_qc.ipynb`, using the nuclei masks and the DNA channel.
#
# - There are no `Image_FileName_*` columns, so each row's raw image path is built from
#   the patient and well-FOV metadata.
# - Every well-FOV's mask file has the same generic name (`nuclei_mask.tiff`), but
#   `data_mask_context_dir` only matches masks by filename pattern within one directory.
#   `stage_mask` fills a scratch directory with per-well-FOV symlinks to the real masks,
#   renamed to embed each well-FOV's identifier, so matching works.
# - `backend="server"` renders server-side and streams images (the hybrid `"trame"`
#   backend's client-side geometry sync drops arrays when a table has many views), so
#   the in-view "Mask" checkbox toggles the overlay on/off; a red dot marks each object's center and a scale bar gives size in um.
#   Views stay blank unless port 8687 is forwarded to the same local port.

# In[4]:


# Serve the interactive trame views on a fixed port so VS Code Remote-SSH can
# forward it to the same local port (see .vscode/settings.json). This must be
# set before the first view renders; restart the kernel if it was already started.
# 7a uses 8686, so a different port lets both notebooks' kernels run at once.
pv.global_theme.trame.jupyter_server_port = 8687

# The TIFFs carry no voxel-size metadata, so the 3D views take voxel size (um) from
# the profiles' Metadata_Microscopy_*ResolutionUm columns (set in 6.annotation), which
# keeps the scale bar consistent with the um-based features.
RESOLUTION_COLUMNS = [f"Metadata_Microscopy_{axis}ResolutionUm" for axis in "XYZ"]
SCALE_BAR_LENGTHS_UM = (1, 2, 5, 10, 20, 50, 100)


# Nucleus QC is driven by DNA-channel features, so view nuclei in the DNA channel
CHANNEL = "DNA"
CHANNEL_CODE = "405"
COMPARTMENT = "Nuclei"
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
    pathlib.Path(tempfile.gettempdir())
    / "cytodataframe_nf1_3d_mask_links"
    / patient
    / COMPARTMENT
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
    Build a CytoDataFrame 3D voxel view (with mask overlay) of nucleus rows.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        The DataFrame containing the single-cell profiles.
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


# ## Load profiles and initialize QC flags
#
# QC is applied in three rounds:
# 1. **NaN detection** (`Metadata_cqc_nan_detected`) — missing ObjectID, volume, or parent
# 2. **Inherited organoid flags** (`Metadata_cqc_organoid_flagged`, `Metadata_cqc_missing_parent_organoid`)
#    — cells whose parent organoid failed QC in 7a, or have no parent organoid at all
# 3. **Nucleus outliers** — applied only to cells that passed rounds 1 and 2

# In[5]:


sc_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles/sc_anno.parquet"
)
organoid_file = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles/organoid_flagged_outliers.parquet"
)

nucleocentric_annotated_sammed_path = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles/nucleocentric_sammed_anno.parquet"
).resolve()
nucleocentric_annotated_morphem_output_path = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles/nucleocentric_morphem_anno.parquet"
).resolve()
sammed_annotated_sc_profiles_path = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "3.annotated_profiles/sammed_sc_anno.parquet"
).resolve()


output_dir = pathlib.Path(
    profile_base_dir
    / "data"
    / f"{patient}"
    / f"{image_based_profiles_subparent_name}"
    / "4.qc_profiles"
)
output_dir.mkdir(parents=True, exist_ok=True)

sc_qc_output_path = pathlib.Path(f"{output_dir}/sc_flagged_outliers.parquet").resolve()
sammed_sc_qc_output_path = pathlib.Path(
    f"{output_dir}/sammed_sc_flagged_outliers.parquet"
).resolve()
nucleocentric_sammed_qc_output_path = pathlib.Path(
    f"{output_dir}/nucleocentric_sammed_flagged_outliers.parquet"
).resolve()
nucleocentric_morphem_qc_output_path = pathlib.Path(
    f"{output_dir}/nucleocentric_morphem_flagged_outliers.parquet"
).resolve()

orig_sc_profiles_df = pd.read_parquet(sc_file)
organoid_qc_profiles_df = pd.read_parquet(organoid_file)
# Print the shape and head of the combined organoid profiles DataFrame
print(orig_sc_profiles_df.shape)
orig_sc_profiles_df


# In[6]:


sc_profiles_df = orig_sc_profiles_df.copy()
sc_profiles_df["Metadata_cqc_nan_detected"] = (
    sc_profiles_df[
        [
            "Metadata_Object_ObjectID",
            "Metadata_Object_ParentOrganoid",
            "Cell_NoChannel_VolumeSizeShape_Volume",
        ]
    ]
    .isna()
    .any(axis=1)
)
# Print the number of organoids flagged
flagged_count = sc_profiles_df["Metadata_cqc_nan_detected"].sum()
print(f"Number of organoids flagged: {flagged_count}")

sc_profiles_df.head()


# In[7]:


# Round 2: propagate organoid-level QC flags to single cells.
# A cell is flagged if its parent organoid was flagged in 7a.
# We match on (ParentOrganoid, WellFOV) rather than ParentOrganoid alone because
# object IDs are reassigned per-FOV and are not globally unique across the patient.

# Default QC flags
sc_profiles_df["Metadata_cqc_organoid_flagged"] = False
sc_profiles_df["Metadata_cqc_nan_detected"] = (
    sc_profiles_df[
        ["Metadata_Object_ObjectID", "Nuclei_NoChannel_VolumeSizeShape_Volume"]
    ]
    .isna()
    .any(axis=1)
)
sc_profiles_df["Metadata_cqc_missing_parent_organoid"] = (
    sc_profiles_df["Metadata_Object_ParentOrganoid"] == -1
)


organoid_flags_df = organoid_qc_profiles_df[
    ["Metadata_Object_ObjectID", "Metadata_Experiment_WellFOV"]
    + [col for col in organoid_qc_profiles_df.columns if col.startswith("Metadata_cqc")]
]

# Get flagged (object_id, image_set) pairs
flagged_pairs = set(
    organoid_flags_df.loc[
        organoid_flags_df.filter(like="cqc").any(axis=1),
        ["Metadata_Object_ObjectID", "Metadata_Experiment_WellFOV"],
    ].itertuples(index=False, name=None)
)

# Flag SC rows where both parent_organoid & image_set match a flagged organoid
sc_profiles_df["Metadata_cqc_organoid_flagged"] = sc_profiles_df.apply(
    lambda row: (
        (row["Metadata_Object_ParentOrganoid"], row["Metadata_Experiment_WellFOV"])
        in flagged_pairs
    ),
    axis=1,
)

print(sc_profiles_df.shape)
sc_profiles_df.head()


# In[8]:


sc_profiles_df["Nuclei_NoChannel_VolumeSizeShape_Volume"].describe()


# ## Detect outlier single-cells using the non-flagged data
#
# We will attempt to detect instances of poor quality segmentations using the nuclei compartment as the base. The conditions we are using are as follows:
#
# 1. Abnormally small or large nuclei using `Volume`
# 2. Abnormally high `mass displacement` in the nuclei for instances of mis-segmentation of background/no longer in-focus

# In[9]:


# Set the metadata columns to be used in the QC process
metadata_columns = [x for x in sc_profiles_df.columns if "Metadata" in x]


# In[10]:


# Round 3: nucleus-based outlier detection using z-score thresholds.
# Threshold sign: negative = flag below mean, positive = flag above mean.
# Threshold magnitude: number of standard deviations from the mean.
# Only cells that passed rounds 1 and 2 are evaluated here.
# Only process the rows that are not flagged
filtered_plate_df = sc_profiles_df[
    ~(
        sc_profiles_df["Metadata_cqc_nan_detected"]
        | sc_profiles_df["Metadata_cqc_organoid_flagged"]
        | sc_profiles_df["Metadata_cqc_missing_parent_organoid"]
    )
]

# --- Find size based nuclei outliers ---
print("Finding small nuclei outliers...")
small_nuclei_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_NoChannel_VolumeSizeShape_Volume": small_nuclei_threshold,  # Detect very small nuclei
    },
)

# Ensure the column exists before assignment
sc_profiles_df["Metadata_cqc_small_nuclei_outlier"] = False
sc_profiles_df.loc[small_nuclei_outliers.index, "Metadata_cqc_small_nuclei_outlier"] = (
    True
)

# Print number of outliers (only in filtered rows)
small_count = filtered_plate_df.index.intersection(small_nuclei_outliers.index).shape[0]
print(f"Small nuclei outliers found: {small_count}")

print("Finding large nuclei outliers...")
large_nuclei_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_NoChannel_VolumeSizeShape_Volume": large_nuclei_threshold,  # Detect very large nuclei
    },
)

# Ensure the column exists before assignment
sc_profiles_df["Metadata_cqc_large_nuclei_outlier"] = False
sc_profiles_df.loc[large_nuclei_outliers.index, "Metadata_cqc_large_nuclei_outlier"] = (
    True
)

# Print number of outliers (only in filtered rows)
large_count = filtered_plate_df.index.intersection(large_nuclei_outliers.index).shape[0]
print(f"Large nuclei outliers found: {large_count}")

# --- Find mass displacement based nuclei outliers ---
print("Finding high mass displacement outliers...")
high_mass_displacement_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_DNA_Intensity_MassDisplacement": high_mass_displacement_threshold,  # Detect high mass displacement
    },
)

# Ensure the column exists before assignment
sc_profiles_df["Metadata_cqc_mass_displacement_outlier"] = False
sc_profiles_df.loc[
    high_mass_displacement_outliers.index, "Metadata_cqc_mass_displacement_outlier"
] = True

# Print number of outliers (only in filtered rows)
high_mass_count = filtered_plate_df.index.intersection(
    high_mass_displacement_outliers.index
).shape[0]
print(f"High mass displacement outliers found: {high_mass_count}")

# Save updated plate_df with flag columns included
sc_profiles_df.to_parquet(sc_qc_output_path, index=False)


# In[11]:


sc_profiles_df.head()


# ## Visualize nuclei in 3D to tune this patient's thresholds

# ### Visualize the small nuclei outliers

# In[12]:


# Each view is a live server-side render, so show a random sample of each group
N_NUCLEI_TO_VIEW = 10

if in_notebook:
    display(
        make_voxel_view(
            sc_profiles_df.loc[
                sc_profiles_df["Metadata_cqc_small_nuclei_outlier"]
            ].sample(
                n=min(
                    N_NUCLEI_TO_VIEW,
                    int(sc_profiles_df["Metadata_cqc_small_nuclei_outlier"].sum()),
                ),
                random_state=0,
            ),
            [
                "Metadata_Experiment_WellFOV",
                "Metadata_Object_ObjectID",
                "Nuclei_NoChannel_VolumeSizeShape_Volume",
                f"Image_FileName_{CHANNEL}",
            ],
        ).show_widget_table(column=f"Image_FileName_{CHANNEL}", backend="server")
    )


# ### Visualize the large nuclei outliers

# In[13]:


if in_notebook:
    display(
        make_voxel_view(
            sc_profiles_df.loc[
                sc_profiles_df["Metadata_cqc_large_nuclei_outlier"]
            ].sample(
                n=min(
                    N_NUCLEI_TO_VIEW,
                    int(sc_profiles_df["Metadata_cqc_large_nuclei_outlier"].sum()),
                ),
                random_state=0,
            ),
            [
                "Metadata_Experiment_WellFOV",
                "Metadata_Object_ObjectID",
                "Nuclei_NoChannel_VolumeSizeShape_Volume",
                f"Image_FileName_{CHANNEL}",
            ],
        ).show_widget_table(column=f"Image_FileName_{CHANNEL}", backend="server")
    )


# ### Visualize the high mass displacement outliers

# In[14]:


if in_notebook:
    display(
        make_voxel_view(
            sc_profiles_df.loc[
                sc_profiles_df["Metadata_cqc_mass_displacement_outlier"]
            ].sample(
                n=min(
                    N_NUCLEI_TO_VIEW,
                    int(sc_profiles_df["Metadata_cqc_mass_displacement_outlier"].sum()),
                ),
                random_state=0,
            ),
            [
                "Metadata_Experiment_WellFOV",
                "Metadata_Object_ObjectID",
                "Nuclei_DNA_Intensity_MassDisplacement",
                f"Image_FileName_{CHANNEL}",
            ],
        ).show_widget_table(column=f"Image_FileName_{CHANNEL}", backend="server")
    )


# ### Visualize a random selection of nuclei

# In[15]:


if in_notebook:
    display(
        make_voxel_view(
            sc_profiles_df.sample(
                n=min(N_NUCLEI_TO_VIEW, len(sc_profiles_df)), random_state=0
            ),
            [
                "Metadata_Experiment_WellFOV",
                "Metadata_Object_ObjectID",
                "Metadata_cqc_small_nuclei_outlier",
                "Metadata_cqc_large_nuclei_outlier",
                "Metadata_cqc_mass_displacement_outlier",
                "Nuclei_NoChannel_VolumeSizeShape_Volume",
                f"Image_FileName_{CHANNEL}",
            ],
        ).show_widget_table(column=f"Image_FileName_{CHANNEL}", backend="server")
    )


# ### Merge the qc flags to the deep learning-based profiles and save the output
# Merge the QC flags back to the original single cell profiles, which will be used in downstream analyses and single cell QC.
# We need to do this beacuase we do not run qc on black-box features.
# Merge on the Metadata_Biology_PatientTumor, Metadata_Experiment_WellFOV
# and the Metadata_Object_ObjectID columns, which together uniquely identify each organoid profile row.

# In[16]:


nucleocentric_annotated_sammed_df = pd.read_parquet(nucleocentric_annotated_sammed_path)
nucleocentric_annotated_morphem_df = pd.read_parquet(
    nucleocentric_annotated_morphem_output_path
)
sammed_annotated_sc_profiles_df = pd.read_parquet(sammed_annotated_sc_profiles_path)
df_dict = {
    "nulceocentric_sammed": {
        "df": nucleocentric_annotated_sammed_df,
        "qc_output_path": nucleocentric_sammed_qc_output_path,
    },
    "nucleocentric_chammi": {
        "df": nucleocentric_annotated_morphem_df,
        "qc_output_path": nucleocentric_morphem_qc_output_path,
    },
    "sammed_sc_profiles": {
        "df": sammed_annotated_sc_profiles_df,
        "qc_output_path": sammed_sc_qc_output_path,
    },
}


# In[17]:


# set the merge keys to int for both dataframes to ensure they match
merge_keys = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_WellFOV",
    "Metadata_Object_ObjectID",
]
qc_keys = [col for col in sc_profiles_df.columns if "Metadata_cqc" in col]

for profile_name in df_dict:
    df = df_dict[profile_name]["df"]
    for key in merge_keys:
        if key not in df.columns:
            raise ValueError(f"Merge key {key} not found in dataframe columns.")
    qc_annotated_df = df.merge(
        sc_profiles_df[qc_keys + merge_keys],
        on=merge_keys,
        how="left",
    )
    if qc_annotated_df.shape[1] == df.shape[1]:
        raise ValueError(
            f"No new columns were added during the merge. Check that the merge keys {merge_keys} are correct and that the qc keys {qc_keys} are present in the sc_profiles_df."
        )
    qc_annotated_df.to_parquet(df_dict[profile_name]["qc_output_path"], index=False)
