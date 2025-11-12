import pathlib
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skimage
import torch
import tqdm
from cellpose import core, models
from file_reading import read_zstack_image
from segmentation_decoupling import (
    euclidian_2D_distance,
    extract_unique_masks,
    get_combinations_of_indices,
    get_number_of_unique_labels,
    merge_sets_df,
    reassemble_each_mask,
)
from skimage.filters import sobel


# ----------------------------------------------------------------------
# extensions and reads
# ----------------------------------------------------------------------
def find_files_available(
    input_dir: pathlib.Path,
    image_extensions: set = {".tif", ".tiff"},
) -> List[pathlib.Path]:
    files = sorted(input_dir.glob("*"))
    files = [str(x) for x in files if x.suffix in image_extensions]
    return files


def read_in_channels(
    files,
    channel_dict: dict = {
        "nuclei": "405",
        "cyto1": "488",
        "cyto2": "555",
        "cyto3": "640",
        "brightfield": "TRANS",
    },
):
    for f in files:
        if channel_dict["nuclei"] in f:
            nuclei = read_zstack_image(f)
        elif channel_dict["cyto1"] in f:
            cyto1 = read_zstack_image(f)
        elif channel_dict["cyto2"] in f:
            cyto2 = read_zstack_image(f)
        elif channel_dict["cyto3"] in f:
            cyto3 = read_zstack_image(f)
        elif channel_dict["brightfield"] in f:
            brightfield = read_zstack_image(f)
        else:
            print(f"Unknown channel: {f}")

    return {
        "nuclei": np.array(nuclei),
        "cyto1": np.array(cyto1),
        "cyto2": np.array(cyto2),
        "cyto3": np.array(cyto3),
        "brightfield": np.array(brightfield),
    }


# ----------------------------------------------------------------------
# convert to 2.5 D image stack
# ----------------------------------------------------------------------
def sliding_window_two_point_five_D(image_stack, window_size):
    image_stack_2_5D = np.empty(
        (0, image_stack.shape[1], image_stack.shape[2]), dtype=image_stack.dtype
    )
    for image_index in range(image_stack.shape[0]):
        image_stack_window = image_stack[image_index : image_index + window_size]
        if not image_stack_window.shape[0] == window_size:
            break
        # max project the image stack
        image_stack_2_5D = np.array(
            np.append(
                image_stack_2_5D,
                np.max(image_stack_window, axis=0)[np.newaxis, :, :],
                axis=0,
            )
        )
    return image_stack_2_5D


def reverse_sliding_window_max_projection(
    output_dict, window_size, original_z_slice_count
):
    # reverse the sliding window
    # reverse sliding window max projection
    full_mask_z_stack = []
    reconstruction_dict = {index: [] for index in range(original_z_slice_count)}
    # loop through the sliding window max projected masks and decouple them
    for z_stack_mask_index in range(len(output_dict["labels"])):
        z_stack_decouple = []
        # make n copies of the mask for sliding window decoupling
        # where n is the size of the sliding window
        [
            z_stack_decouple.append(output_dict["labels"][z_stack_mask_index])
            for _ in range(window_size)
        ]
        for z_window_index, z_stack_mask in enumerate(z_stack_decouple):
            # append the masks to the reconstruction_dict
            if not (z_stack_mask_index + z_window_index) >= original_z_slice_count:
                reconstruction_dict[z_stack_mask_index + z_window_index].append(
                    z_stack_mask
                )
    return reconstruction_dict


# ----------------------------------------------------------------------
# Organoid segmentation with dynamic diameter search
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Butterworth filtering function
# ----------------------------------------------------------------------
def butterworth_grid_optimization(
    img,
    return_plot: bool = False,
):
    # get the median most image from the cyto image stack
    # this is the image that will be used for the butterworth filter optimization
    # get the median image from the cyto image stack
    middle_index = int(img.shape[0] / 2)
    img_to_optimize = img[middle_index]
    optimization_steps = 5
    # optimize the butterworth filter for the cyto image
    search_space_cutoff_freq = np.linspace(0.01, 0.5, optimization_steps)
    search_space_order = np.linspace(1, 10, optimization_steps)
    # create a list of optimzation parameter pairs
    optimization_parameter_pairs = []
    for cutoff_freq_option in search_space_cutoff_freq:
        for order_option in search_space_order:
            optimization_parameter_pairs.append((cutoff_freq_option, order_option))

    optimized_images = []
    # loop through the optimization pairs to find the best pararmeters
    for cutoff_freq_option, order_option in tqdm.tqdm(optimization_parameter_pairs):
        optimized_images.append(
            skimage.filters.butterworth(
                img_to_optimize,
                cutoff_frequency_ratio=cutoff_freq_option,
                high_pass=False,
                order=order_option,
                squared_butterworth=True,
            )
        )
    if return_plot:
        # visualize the optimized images in a grid
        fig, ax = plt.subplots(optimization_steps, optimization_steps, figsize=(20, 20))
        for i in range(optimization_steps):
            for j in range(optimization_steps):
                ax[i, j].imshow(optimized_images[i * optimization_steps + j])
                ax[i, j].axis("off")
                # add the cutoff frequency and order to the plot
                ax[i, j].set_title(
                    f"Freq: {search_space_cutoff_freq[i]:.2f}, Order: {search_space_order[j]:.2f}"
                )
        plt.show()


def apply_butterworth_filter(
    img: np.ndarray,
    cutoff_frequency_ratio: float = 0.05,
    order: int = 1,
    high_pass: bool = False,
    squared_butterworth: bool = True,
):
    # Use butterworth FFT filter to remove high frequency noise :)
    for i in range(img.shape[0]):
        img[i, :, :] = skimage.filters.butterworth(
            img[i, :, :],
            cutoff_frequency_ratio=cutoff_frequency_ratio,
            high_pass=high_pass,
            order=order,
            squared_butterworth=squared_butterworth,
        )

    # add a guassian blur to the image
    img = skimage.filters.gaussian(img, sigma=1)
    return img


# ----------------------------------------------------------------------
# decoupling segmented masks
# ----------------------------------------------------------------------
def decouple_masks(
    reconstruction_dict: dict,
    original_img_shape: np.ndarray,
    distance_threshold: int,
    verbose: bool = False,
):
    masks_dict = {}
    for zslice, arrays in tqdm.tqdm(enumerate(reconstruction_dict)):
        df = extract_unique_masks(reconstruction_dict[zslice])
        merged_df = get_combinations_of_indices(
            df, distance_threshold=distance_threshold
        )
        # combine dfs for each window index
        merged_df = merge_sets_df(merged_df)
        if not merged_df.empty:
            merged_df.loc[:, "slice"] = zslice
            reassembled_masks = reassemble_each_mask(
                merged_df, original_img_shape=original_img_shape
            )
            masks_dict[zslice] = reassembled_masks
        else:
            if verbose:
                print(f"Warning: merged_df is empty for zslice {zslice}")
            masks_dict[zslice] = reconstruction_dict[zslice][0]
    return masks_dict


# ------------------------------------------------------
# reconstruct full 3D masks from decoupled masks
# ------------------------------------------------------
def generate_coordinates_for_reconstruction(image: np.ndarray) -> pd.DataFrame:
    cordinates = {
        "original_label": [],
        "slice": [],
        "centroid-0": [],
        "centroid-1": [],
        "bbox-0": [],
        "bbox-1": [],
        "bbox-2": [],
        "bbox-3": [],
    }

    for slice in range(image.shape[0]):
        props = skimage.measure.regionprops_table(
            image[slice, :, :], properties=["label", "centroid", "bbox"]
        )

        label, centroid1, centroid2, bbox0, bbox1, bbox2, bbox3 = (
            props["label"],
            props["centroid-0"],
            props["centroid-1"],
            props["bbox-0"],
            props["bbox-1"],
            props["bbox-2"],
            props["bbox-3"],
        )
        if len(label) > 0:
            for i in range(len(label)):
                cordinates["original_label"].append(label[i])
                cordinates["slice"].append(slice)
                cordinates["centroid-0"].append(centroid1[i])
                cordinates["centroid-1"].append(centroid2[i])
                cordinates["bbox-0"].append(bbox0[i])
                cordinates["bbox-1"].append(bbox1[i])
                cordinates["bbox-2"].append(bbox2[i])
                cordinates["bbox-3"].append(bbox3[i])

    coordinates_df = pd.DataFrame(cordinates)
    coordinates_df["unique_id"] = coordinates_df.index
    return coordinates_df


def generate_distance_pairs(
    coordinates_df: pd.DataFrame, x_y_vector_radius_max_constraint: int
):
    # generate distance pairs for each slice
    distance_pairs = {
        "slice1": [],
        "slice2": [],
        "index1": [],
        "index2": [],
        "distance": [],
        "coordinates1": [],
        "coordinates2": [],
        "pass": [],
        "original_label1": [],
        "original_label2": [],
    }

    distance_pairs_list = [
        {
            "slice1": coordinates_df.loc[i, "slice"],
            "slice2": coordinates_df.loc[j, "slice"],
            "index1": i,
            "index2": j,
            "distance": euclidian_2D_distance(
                coordinates_df.loc[i, ["centroid-0", "centroid-1"]].values,
                coordinates_df.loc[j, ["centroid-0", "centroid-1"]].values,
            ),
            "coordinates1": tuple(
                coordinates_df.loc[i, ["centroid-0", "centroid-1"]].values
            ),
            "coordinates2": tuple(
                coordinates_df.loc[j, ["centroid-0", "centroid-1"]].values
            ),
            "pass": True,
            "original_label1": coordinates_df.loc[i, "original_label"],
            "original_label2": coordinates_df.loc[j, "original_label"],
        }
        for i in range(coordinates_df.shape[0])
        for j in range(coordinates_df.shape[0])
        if i != j
        and euclidian_2D_distance(
            coordinates_df.loc[i, ["centroid-0", "centroid-1"]].values,
            coordinates_df.loc[j, ["centroid-0", "centroid-1"]].values,
        )
        < x_y_vector_radius_max_constraint
    ]

    # Convert to DataFrame (if needed)
    df = pd.DataFrame(distance_pairs_list)
    if not df.empty:
        df["indexes"] = df["index1"].astype(str) + "-" + df["index2"].astype(str)
        df = df[df["pass"] == True]
        df["index_comparison"] = (
            df["index1"].astype(str) + "," + df["index2"].astype(str)
        )
        df.head()
    return df


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a bounding box.

    Parameters
    ----------
    bbox : Tuple[int, int, int, int]
        The bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns
    -------
    int
        The area of the bounding box.
    """
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def calculate_overlap(
    bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
) -> float:
    """
    Calculate the percentage overlap between two bounding boxes.

    Parameters
    ----------
    bbox1 : Tuple[int, int, int, int]
        The first bounding box (x_min, y_min, x_max, y_max).
    bbox2 : Tuple[int, int, int, int]
        The second bounding box (x_min, y_min, x_max, y_max).

    Returns
    -------
    float
        The percentage overlap of the smaller bounding box with the larger one.
    """
    # Calculate intersection coordinates
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    # Calculate intersection area
    overlap_width = max(0, x_max - x_min)
    overlap_height = max(0, y_max - y_min)
    overlap_area = overlap_width * overlap_height

    # Calculate areas of both bounding boxes
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)

    # Return the percentage overlap relative to the smaller bounding box
    smaller_area = min(area1, area2)
    return overlap_area / smaller_area if smaller_area > 0 else 0.0


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """
    Calculate the Intersection over Union (IoU) between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        The first binary mask.
    mask2 : np.ndarray
        The second binary mask.

    Returns
    -------
    bool
        True if the IoU is greater than 0.5, False otherwise.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    if np.sum(union) == 0:
        return False

    iou = np.sum(intersection) / np.sum(union)

    return iou


def graph_creation(df):
    # create a graph where each node is a unique centroid and each edge is a distance between centroids
    # edges between nodes with the same slice are not allowed
    # edge weight is the distance between the nodes (euclidian distance)
    G = nx.Graph()
    for row in df.iterrows():
        G.add_node(
            row[1]["index1"], slice=row[1]["slice1"], coordinates=row[1]["coordinates1"]
        )
        G.add_node(
            row[1]["index2"], slice=row[1]["slice2"], coordinates=row[1]["coordinates2"]
        )
        G.add_edge(
            row[1]["index1"],
            row[1]["index2"],
            weight=row[1]["distance"],
            original_label1=row[1]["original_label1"],
            original_label2=row[1]["original_label2"],
        )

    # plot the graph with each slice being on a different row
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, "weight")
    return G


def solve_graph(G):
    # solve the the shortest path problem
    # find the longest paths in the graph with the smallest edge weights
    # this will find the longest paths between centroids closest to each other
    # the longest path is the path with the most edges
    longest_paths = []
    for path in nx.all_pairs_shortest_path(G, cutoff=10):
        longest_path = []
        for key in path[1].keys():
            if len(path[1][key]) > len(longest_path):
                longest_path = path[1][key]
        longest_paths.append(longest_path)
    return longest_paths


def merge_sets(list_of_sets: list) -> list:
    for i, set1 in enumerate(list_of_sets):
        for j, set2 in enumerate(list_of_sets):
            if i != j and len(set1.intersection(set2)) > 0:
                set1.update(set2)
    return list_of_sets


def collapse_labels(df, longest_paths):
    list_of_sets = [set(x) for x in longest_paths]
    merged_sets = merge_sets(list_of_sets)
    merged_sets_dict = {}
    for i in range(len(list_of_sets)):
        merged_sets_dict[i] = list_of_sets[i]
    for row in df.iterrows():
        for num_set in merged_sets_dict:
            if int(row[1]["unique_id"]) in merged_sets_dict[num_set]:
                df.at[row[0], "label"] = num_set
    # drop nan
    df = df.dropna()
    return df


def reassign_labels(
    image: np.ndarray,
    df: pd.DataFrame,
):
    new_mask_image = np.zeros_like(image)
    # mask label reassignment
    for slice in range(image.shape[0]):
        mask = image[slice, :, :]
        tmp_df = df[df["slice"] == slice]
        if tmp_df.empty:
            continue
        # check if label is present or if reassignment is needed
        if "label" not in tmp_df.columns:
            continue
        for i in range(tmp_df.shape[0]):
            mask[mask == tmp_df.iloc[i]["original_label"]] = tmp_df.iloc[i]["label"]

        new_mask_image[slice, :, :] = mask
    return new_mask_image


# ----------------------------------------------------------------------
# post hoc refinements
# ----------------------------------------------------------------------
def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a bounding box.

    Parameters
    ----------
    bbox : Tuple[int, int, int, int]
        The bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns
    -------
    int
        The area of the bounding box.
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def calculate_overlap(
    bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
) -> float:
    # calculate the % overlap of the second bbox with the first bbox
    if calculate_bbox_area(bbox1) == 0 or calculate_bbox_area(bbox2) == 0:
        return 0.0
    if calculate_bbox_area(bbox1) >= calculate_bbox_area(bbox2):
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x_max - x_min)
        overlap_height = max(0, y_max - y_min)
        overlap_area = overlap_width * overlap_height
        bbox1_area = calculate_bbox_area(bbox1)
        bbox2_area = calculate_bbox_area(bbox2)
        overlap_percentage = overlap_area / bbox2_area if bbox2_area > 0 else 0
        return overlap_percentage
    elif calculate_bbox_area(bbox1) < calculate_bbox_area(bbox2):
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        x_max = min(bbox1[2], bbox2[2])
        y_max = min(bbox1[3], bbox2[3])
        overlap_width = max(0, x_max - x_min)
        overlap_height = max(0, y_max - y_min)
        overlap_area = overlap_width * overlap_height
        bbox1_area = calculate_bbox_area(bbox1)
        bbox2_area = calculate_bbox_area(bbox2)
        overlap_percentage = overlap_area / bbox1_area if bbox1_area > 0 else 0
        return overlap_percentage
    else:
        print("Error: Bboxes are the same size")


def check_for_all_same_labels(
    object_information_df: pd.DataFrame,
) -> bool:
    """
    Check if all labels in the object information DataFrame are the same.

    Parameters
    ----------
    object_information_df : pd.DataFrame
        The DataFrame containing object information with 'label' column.

    Returns
    -------
    bool
        True if all labels are the same, False otherwise.
    """
    return object_information_df["label"].nunique() == 1


def missing_slice_check(
    object_information_df: pd.DataFrame,
    window_min: int = 0,
    window_max: int = 2,
    interpolated_rows_to_add: List[int] = [],
) -> List[pd.DataFrame]:
    """
    Check for missing slices in the object information DataFrame and add interpolated rows if necessary.

    Parameters
    ----------
    object_information_df : pd.DataFrame
        The DataFrame containing object information with 'z' and 'label' columns.
    window_min : int, optional
        The minimum window size for checking missing slices, by default 0
    window_max : int, optional
        The maximum window size for checking missing slices, by default 2
    interpolated_rows_to_add : List[int], optional
        A list to store rows to be added for interpolation, by default []

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames containing rows to be added for interpolation.
    """
    max_z = object_information_df["z"].max()
    min_z = object_information_df["z"].min()
    if max_z - min_z > 1:
        if len(object_information_df) < 3:
            # get the first row
            row = object_information_df.iloc[0]
            new_row = {
                "added_z": row["z"],
                "added_new_label": row["label"],
                "zslice_to_copy": row["z"],
            }

            # interpolate the labels to the middle most slice
            # get the middle slice
            middle_slice = int((max_z + min_z) / 2)
            # insert one slice
            z_zlice_to_copy = row["z"]

            new_row = {
                # 'index': object_information_df['index'].values[0],
                # 'index': object_max_slice_label,
                "added_z": middle_slice,
                "added_new_label": row["label"],
                "zslice_to_copy": z_zlice_to_copy,
            }
            interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
    return interpolated_rows_to_add


def add_min_max_boundry_slices(
    object_information_df: pd.DataFrame,
    global_min_z: int,
    global_max_z: int,
    interpolated_rows_to_add: List[pd.DataFrame] = [],
) -> List[pd.DataFrame]:
    """
    Add slices to the object information DataFrame that are one slice away from the global min and max z slices.

    Parameters
    ----------
    object_information_df : pd.DataFrame
        The DataFrame containing object information with 'z' and 'label' columns.
    global_min_z : int
        The global minimum z slice.
    global_max_z : int
        The global maximum z slice.
    interpolated_rows_to_add : List[pd.DataFrame], optional
        A list to store rows to be added for interpolation, by default []

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames containing rows to be added for interpolation at the min and max z slices.
    """
    # find labels that are 1 slice away from the min or max and extend the label
    for i, row in object_information_df.iterrows():
        # check if the z slice is one away from the min or max (global min and max)
        if row["z"] == global_max_z - 1:
            new_row = {
                "added_z": global_max_z,
                "added_new_label": row["label"],
                "zslice_to_copy": row["z"],
            }
            interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
        elif row["z"] == global_min_z + 1:
            new_row = {
                "added_z": global_min_z,
                "added_new_label": row["label"],
                "zslice_to_copy": row["z"],
            }
            interpolated_rows_to_add.append(pd.DataFrame(new_row, index=[0]))
    return interpolated_rows_to_add


def add_masks_where_missing(
    new_mask_image: np.ndarray,
    interpolated_rows_to_add_df: pd.DataFrame,
) -> np.ndarray:
    """
    Add masks to the new mask image where the slices are missing based on the interpolated rows.

    Parameters
    ----------
    new_mask_image : np.ndarray
        The new mask image to which the slices will be added.
    interpolated_rows_to_add_df : pd.DataFrame
        The DataFrame containing the rows to be added for interpolation, with columns 'added_z', 'added_new_label', and 'zslice_to_copy'.

    Returns
    -------
    np.ndarray
        The new mask image with the added slices.
    """
    for slice in interpolated_rows_to_add_df["added_z"].unique():
        # get the rows that correspond to the slice
        tmp_df = interpolated_rows_to_add_df[
            interpolated_rows_to_add_df["added_z"] == slice
        ]
        if tmp_df.shape[0] == 0:
            continue
        for i, row in tmp_df.iterrows():
            # get the z slice to copy mask
            new_slice = new_mask_image[row["zslice_to_copy"].astype(int), :, :].copy()
            new_slice[new_slice != row["added_new_label"]] = 0

            old_slice = new_mask_image[row["added_z"].astype(int), :, :].copy()
            max_projected_slice = np.maximum(old_slice, new_slice)
            new_mask_image[row["added_z"].astype(int), :, :] = max_projected_slice
    return new_mask_image


def reorder_organoid_labels(
    label_image: np.ndarray,
) -> np.ndarray:
    """
    Reorder the labels in the label image to ensure they are sequential starting from 1.

    Parameters
    ----------
    label_image : np.ndarray
        The label image where labels need to be reordered.

    Returns
    -------
    np.ndarray
        The label image with reordered labels.
    """
    unique_labels = np.unique(label_image)
    # remove the background label (0)
    unique_labels = unique_labels[unique_labels != 0]
    # exit early if there are no labels (only background)
    if len(unique_labels) == 0:
        return label_image
    # create a mapping from old label to new label
    label_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(unique_labels, start=1)
    }
    label_image_corrected = np.copy(label_image)
    for old_label, new_label in label_mapping.items():
        label_image_corrected[label_image == old_label] = new_label
    return label_image_corrected


def run_post_hoc_refinement(
    mask_image: List[int],
    sliding_window_context: int,
) -> np.ndarray:
    new_mask_image = mask_image.copy()
    global_max_z = mask_image.shape[0]  # number of z slices
    global_min_z = 0
    # expand the z slices into a list of slices between the min and max z slices
    z_slices = [x for x in range(global_min_z, global_max_z)]
    for z in z_slices[: -(sliding_window_context - 1)]:
        interpolated_rows_to_add = []

        final_dict = {
            "index1": [],
            "index2": [],
            "z1": [],
            "z2": [],
            "distance": [],
            "label1": [],
            "label2": [],
        }
        list_of_cell_masks = []
        for z_slice in range(0, new_mask_image.shape[0] - 1):
            compartment_df = pd.DataFrame.from_dict(
                skimage.measure.regionprops_table(
                    new_mask_image[z, :, :],
                    properties=["centroid", "bbox"],
                )
            )
            compartment_df["z"] = z_slice

            list_of_cell_masks.append(compartment_df)
        compartment_df = pd.concat(list_of_cell_masks)

        # get the pixel value of the organoid mask at each x,y,z coordinate
        compartment_df["label"] = new_mask_image[
            compartment_df["z"].astype(int),
            compartment_df["centroid-0"].astype(int),
            compartment_df["centroid-1"].astype(int),
        ]
        compartment_df.reset_index(drop=True, inplace=True)
        compartment_df["new_label"] = compartment_df["label"]
        # drop all labels that are 0
        compartment_df = compartment_df[compartment_df["label"] != 0]

        # Get the temporary sliding window
        tmp_window_df = compartment_df[
            (compartment_df["z"] >= z)
            & (compartment_df["z"] < z + sliding_window_context)
        ]

        if tmp_window_df["z"].nunique() < sliding_window_context:
            continue
        for i, row1 in tmp_window_df.iterrows():
            for j, row2 in tmp_window_df.iterrows():
                if i != j:  # Ensure you're not comparing the same row
                    if row1["z"] != row2["z"]:
                        # get the first bbox

                        distance = euclidian_2D_distance(
                            (row1["centroid-0"], row1["centroid-1"]),
                            (row2["centroid-0"], row2["centroid-1"]),
                        )

                        if distance < 20:
                            final_dict["index1"].append(i)
                            final_dict["index2"].append(j)
                            final_dict["z1"].append(row1["z"])
                            final_dict["z2"].append(row2["z"])
                            final_dict["distance"].append(distance)
                            final_dict["label1"].append(row1["label"])
                            final_dict["label2"].append(row2["label"])
        final_df = pd.DataFrame.from_dict(final_dict)
        final_df["index_set"] = final_df.apply(
            lambda row: frozenset([row["index1"], row["index2"]]), axis=1
        )
        final_df["index_set"] = final_df["index_set"].apply(lambda x: tuple(sorted(x)))

        list_of_sets = final_df["index_set"].tolist()
        list_of_sets = [set(s) for s in list_of_sets]
        merged_sets = merge_sets(list_of_sets)
        # drop the duplicates
        merged_sets = list({frozenset(s): s for s in merged_sets}.values())

        # from final_df generate the z-ordered cases
        for object_set in merged_sets:
            # find rows that contain integers that are in the object_set
            rows_that_contain_object_set = final_df[
                final_df["index_set"].apply(lambda x: set(x).issubset(object_set))
            ]
            # get the index, label, and z pair
            dict_of_object_information = {"index": [], "label": [], "z": []}
            for i, row in rows_that_contain_object_set.iterrows():
                dict_of_object_information["index"].append(row["index1"])
                dict_of_object_information["label"].append(row["label1"])
                dict_of_object_information["z"].append(row["z1"])
                dict_of_object_information["index"].append(row["index2"])
                dict_of_object_information["label"].append(row["label2"])
                dict_of_object_information["z"].append(row["z2"])
            object_information_df = pd.DataFrame.from_dict(dict_of_object_information)
            object_information_df.drop_duplicates(
                subset=["index", "label", "z"], inplace=True
            )
            object_information_df.sort_values(by=["index", "z"], inplace=True)
            if check_for_all_same_labels(object_information_df):
                # if all labels are the same, skip this object
                continue
            interpolated_rows_to_add = missing_slice_check(
                object_information_df, interpolated_rows_to_add=interpolated_rows_to_add
            )
            interpolated_rows_to_add = add_min_max_boundry_slices(
                object_information_df,
                global_min_z=global_min_z,
                global_max_z=global_max_z,
                interpolated_rows_to_add=interpolated_rows_to_add,
            )

        if len(interpolated_rows_to_add) == 0:
            if z == z_slices[-1]:
                # tifffile.imwrite(mask_output_path, new_mask_image)
                return new_mask_image
            else:
                continue
        interpolated_rows_to_add_df = pd.concat(interpolated_rows_to_add, axis=0)
        new_mask_image = new_mask_image.copy()
        new_mask_image = add_masks_where_missing(
            new_mask_image=new_mask_image,
            interpolated_rows_to_add_df=interpolated_rows_to_add_df,
        )
    return new_mask_image


# ----------------------------------------------------------------------
# Segment the cells with 3D watershed
# ----------------------------------------------------------------------


def segment_cells_with_3D_watershed(
    cyto_signal: np.ndarray,
    nuclei_mask: np.ndarray,
) -> np.ndarray:
    # gaussian filter to smooth the image
    cell_signal_image = skimage.filters.gaussian(cyto_signal, sigma=1.0)
    # scale the pixels to max 255
    nuclei_mask = (nuclei_mask / nuclei_mask.max() * 255).astype(np.uint8)
    # generate the elevation map using the Sobel filter
    elevation_map = sobel(cell_signal_image)

    # set up seeded watersheding where the nuclei masks are used as seeds
    # note: the cytoplasm is used as the signal for this.

    labels = skimage.segmentation.watershed(
        image=elevation_map,
        markers=nuclei_mask,
    )

    # change the largest label (by area) to 0
    # cleans up the output and sets the background properly
    unique, counts = np.unique(labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    labels[labels == largest_label] = 0
    cell_mask = labels.copy()
    cell_mask = run_post_hoc_refinement(
        mask_image=cell_mask,
        sliding_window_context=3,
    )
    return labels


# ----------------------------------------------------------------------
# post hoc reassignments
# ----------------------------------------------------------------------
def remove_edge_cases(
    mask: np.ndarray,
    border: int = 10,
) -> np.ndarray:
    """
    Remove masks that are image edge cases
    In this case - the edge literally means the edge of the image
    This is useful to remove masks that are not fully contained within the image

    Parameters
    ----------
    mask : np.ndarray
        The mask to process, should be a 3D numpy array
    border : int, optional
        The number of pixels in width to create border to scan for edge cased, by default 10

    Returns
    -------
    np.ndarray
        The mask with edge cases removed
    """

    edge_pixels = np.concatenate(
        [
            # all of z, last n rows (y), all columns (x) - bottom edge
            mask[:, -border:, :].flatten(),
            # all of z, first n rows (y), all columns (x) - top edge
            mask[:, 0:border, :].flatten(),
            # all of z, all rows (y), first n columns (x) - left edge
            mask[:, :, 0:border:].flatten(),
            # all of z, all rows (y), last n columns (x) - right edge
            mask[:, :, -border:].flatten(),
            # each are the edges stacked for the whole volume -> no need to specify every z slice or 3D edge
        ]
    )
    # get unique edge pixel values
    edge_pixels = np.unique(edge_pixels[edge_pixels > 0])

    for edge_pixel_case in edge_pixels:
        # make the edge cases equal to zero
        mask[mask == edge_pixel_case] = 0

    # return the mask with edge cases removed
    return mask


def centroid_within_bbox_detection(
    centroid: tuple,
    bbox: tuple,
) -> bool:
    """
    Check if the centroid is within the bbox

    Parameters
    ----------
    centroid : tuple
        Centroid of the object in the order of (z, y, x)
        Order of the centroid is important
    bbox : tuple
        Where the bbox is in the order of (z_min, y_min, x_min, z_max, y_max, x_max)
        Order of the bbox is important

    Returns
    -------
    bool
        True if the centroid is within the bbox, False otherwise
    """
    z_min, y_min, x_min, z_max, y_max, x_max = bbox
    z, y, x = centroid
    # check if the centroid is within the bbox
    if (
        z >= z_min
        and z <= z_max
        and y >= y_min
        and y <= y_max
        and x >= x_min
        and x <= x_max
    ):
        return True
    else:
        return False


def check_if_centroid_within_mask(
    centroid: tuple, mask: np.ndarray, label: int
) -> bool:
    """
    Check if the centroid is within the mask

    Parameters
    ----------
    centroid : tuple
        Centroid of the object in the order of (z, y, x)
        Order of the centroid is important
    mask : np.ndarray
        The mask to check against

    Returns
    -------
    bool
        True if the centroid is within the mask, False otherwise
    """
    z, y, x = centroid
    z = np.round(z).astype(int)
    y = np.round(y).astype(int)
    x = np.round(x).astype(int)
    # check if the centroid is within the segmentation mask
    cell_label = mask[z, y, x]
    if cell_label > 0 and cell_label == label:
        return True
    else:
        return False


def get_labels_for_post_hoc_reassignment(
    compartment_mask: np.ndarray,
    compartment_name: str,
) -> pd.DataFrame:
    # get the centroid and bbox of the cell mask
    compartment_df = pd.DataFrame.from_dict(
        skimage.measure.regionprops_table(
            compartment_mask,
            properties=["centroid", "bbox"],
        )
    )
    compartment_df["compartment"] = compartment_name
    compartment_df["label"] = compartment_mask[
        compartment_df["centroid-0"].astype(int),
        compartment_df["centroid-1"].astype(int),
        compartment_df["centroid-2"].astype(int),
    ]
    # remove all 0 labels
    compartment_df = compartment_df[compartment_df["label"] > 0].reset_index(drop=True)
    return compartment_df


def mask_label_reassignment(
    mask_df: pd.DataFrame,
    mask_input: np.ndarray,
) -> np.ndarray:
    """
    Reassign the labels of the mask based on the mask_df

    Parameters
    ----------
    mask_df : pd.DataFrame
        DataFrame containing the labels and centroids of the mask
    mask_input : np.ndarray
        The input mask to reassign the labels to

    Returns
    -------
    np.ndarray
        The mask with reassigned labels
    """
    for i, row in mask_df.iterrows():
        if row["label"] == row["new_label"]:
            # if the label is already the new label, skip
            continue
        mask_input[mask_input == row["label"]] = row["new_label"]
    return mask_input


def run_post_hoc_mask_reassignment(
    nuclei_mask: np.ndarray,
    cell_mask: np.ndarray,
    nuclei_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    return_dataframe=False,
):
    # if a centroid of the nuclei is inside the cell mask,
    # then make the cell retain the label of the nuclei
    nuclei_df["new_label"] = nuclei_df["label"].copy()
    for i, row in nuclei_df.iterrows():
        for j, row2 in cell_df.iterrows():
            nuc_contained_in_cell_bool = check_if_centroid_within_mask(
                centroid=(
                    row["centroid-0"],
                    row["centroid-1"],
                    row["centroid-2"],
                ),
                mask=cell_mask,
                label=row2["label"],
            )
            if nuc_contained_in_cell_bool:
                # if the centroid of the nuclei is within the cell mask,
                # then make the cell retain the label of the nuclei
                nuclei_df.at[i, "new_label"] = row2["label"]
                break
            else:
                pass

    # merge the dataframes
    nuclei_and_cell_df = pd.merge(
        nuclei_df,
        cell_df,
        left_on="new_label",
        right_on="label",
        suffixes=("_nuclei", "_cell"),
    )
    # remove the edge cases
    cell_mask = remove_edge_cases(
        mask=cell_mask,
        border=10,
    )
    nuclei_mask = remove_edge_cases(
        mask=nuclei_mask,
        border=10,
    )

    # reassign the labels of the cell mask
    nuclei_mask = mask_label_reassignment(
        mask_df=nuclei_df,
        mask_input=nuclei_mask,
    )
    if return_dataframe:
        return nuclei_mask, nuclei_and_cell_df
    else:
        return nuclei_mask, None


# ----------------------------------------------------------------------
# cytoplasm mask creation
# ----------------------------------------------------------------------
def create_cytoplasm_masks(
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray,
) -> np.ndarray:
    cytoplasm_masks = np.zeros_like(cell_masks)
    # filter masks that are not the background
    for z_slice_index in range(nuclei_masks.shape[0]):
        nuclei_slice_mask = nuclei_masks[z_slice_index]
        cell_slice_mask = cell_masks[z_slice_index]
        cytoplasm_mask = cell_slice_mask.copy()
        cytoplasm_mask[nuclei_slice_mask > 0] = 0  # subtraction happens here
        cytoplasm_masks[z_slice_index] = cytoplasm_mask

    return cytoplasm_masks
