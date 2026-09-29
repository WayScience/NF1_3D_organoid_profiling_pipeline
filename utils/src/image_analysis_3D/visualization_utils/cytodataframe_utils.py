import pathlib
import re
from dataclasses import dataclass

import pandas as pd
import pyvista as pv
from cytodataframe import CytoDataFrame


@dataclass
class CdfDataClass:
    """
    Description
    -----------
    A class to hold constants and init data

    Parameters
    ----------
    profile_base_dir : pathlib.Path
        The base directory for the profile.
    patient : str
        The patient ID.
    well_fov : str
        The well-FOV ID.
    channel_code : str
        The channel code.
    channel : str
        The channel name.
    mask_link_dir : pathlib.Path
        The directory to link the mask to.
    scale_bar_lengths_um : list
        The lengths of the scale bars in um.
    mask_name : str
        The name of the mask.
    resolution_columns : list
        The columns containing the resolution information.
    bbox_column_map : dict
        A map of bounding box columns.
    center_columns : list
        The columns containing the center coordinates.
    """

    profile_base_dir: pathlib.Path
    patient: str
    well_fov: str
    channel_code: str
    channel: str
    mask_link_dir: pathlib.Path
    scale_bar_lengths_um: list
    mask_name: str
    resolution_columns: list
    bbox_column_map: dict
    center_columns: list


def add_label_overlay_and_scale_bar(
    self, plotter, volume, spacing, cdf_params: CdfDataClass, **kwargs
):
    """
    Description
    -----------
    Add the mask overlay, then an XY scale bar in um below the crop.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The plotter to add the overlay to.
    volume : numpy.ndarray
        The 3D volume to display.
    spacing : tuple
        The spacing of the volume in um.
    cdf_params : CdfDataClass
        The parameters for the CytoDataFrame.

    Returns
    -------
    list
        The list of overlay actors.
    """
    overlay_actors = CytoDataFrame._orig_add_label_overlay_to_plotter(
        self, plotter=plotter, volume=volume, spacing=spacing, **kwargs
    )
    # volume is (z, y, x) and the plotter's world units are um via volume_spacing,
    # so the bar stays true to scale when the view is rotated or zoomed
    n_z, n_y, n_x = volume.shape
    width_um = (n_x - 1) * spacing[0]
    length_um = max(
        (
            length
            for length in cdf_params.scale_bar_lengths_um
            if length <= width_um / 2
        ),
        default=cdf_params.scale_bar_lengths_um[0],
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


def stage_mask(mask_name: str, cdf_params: CdfDataClass) -> pathlib.Path:
    """
    Description
    -----------
    Symlink a well-FOV's mask into mask_link_dir under a unique name.

    Parameters
    ----------
    well_fov : str
        The well-FOV for which to stage the mask.
    cdf_params : CdfDataClass
        The parameters for the CytoDataFrame.

    Returns
    -------
    pathlib.Path
        The path to the staged mask.
    """
    channel_path = pathlib.Path(
        cdf_params.profile_base_dir
        / "data"
        / f"{cdf_params.patient}"
        / "zstack_images"
        / f"{cdf_params.well_fov}"
        / f"{cdf_params.well_fov}_{cdf_params.channel_code}.tif"
    )
    mask_path = pathlib.Path(
        cdf_params.profile_base_dir
        / "data"
        / f"{cdf_params.patient}"
        / "segmentation_masks"
        / f"{cdf_params.well_fov}"
        / mask_name
    ).resolve(strict=True)
    link = cdf_params.mask_link_dir / f"{channel_path.stem}__{mask_name}"
    if not link.exists():
        link.symlink_to(mask_path)
    return link


def make_voxel_view(
    profiles_df: pd.DataFrame,
    list_of_columns_to_include: list,
    cdf_params: CdfDataClass,
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
    cdf_params : CdfDataClass
        The parameters for the CytoDataFrame.

    Returns
    -------
    CytoDataFrame
        The CytoDataFrame with the 3D voxel view.
    """
    profiles_df = profiles_df.copy()
    # metadata stores well-FOV as e.g. C11_4, directories on disk use C11-4
    well_fovs = profiles_df["Metadata_Experiment_WellFOV"].str.replace("_", "-")
    profiles_df[f"Image_FileName_{cdf_params.channel}"] = [
        str(
            cdf_params.profile_base_dir
            / "data"
            / f"{cdf_params.patient}"
            / "zstack_images"
            / f"{cdf_params.well_fov}"
            / f"{cdf_params.well_fov}_{cdf_params.channel_code}.tif"
        )
        for well_fov in well_fovs
    ]
    for well_fov in well_fovs.unique():
        cdf_params.well_fov = well_fov
        stage_mask(mask_name=cdf_params.mask_name, cdf_params=cdf_params)
    resolutions = profiles_df[cdf_params.resolution_columns].drop_duplicates()
    if len(resolutions) != 1:
        raise ValueError(
            f"Expected one voxel size across rows, found:\n{resolutions.to_string()}"
        )
    voxel_spacing = tuple(float(value) for value in resolutions.iloc[0])

    return CytoDataFrame(
        data=profiles_df[list_of_columns_to_include],
        data_bounding_box=profiles_df[list(cdf_params.bbox_column_map.values())],
        data_center_xy=profiles_df[cdf_params.center_columns],
        data_mask_context_dir=str(cdf_params.mask_link_dir),
        segmentation_file_regex={
            rf"__{re.escape(cdf_params.mask_name)}$": r"_\d+\.tif$"
        },
        display_options={
            "width": 260,
            "height": 260,
            "table_max_height": "580px",
            "label_overlay_mode": "filled",
            # Voxel size (x, y, z) in um; also sets the scale bar's units.
            "volume_spacing": voxel_spacing,
            "volume_bbox_column_map": cdf_params.bbox_column_map,
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
