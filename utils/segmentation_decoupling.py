import numpy as np
import pandas as pd
import skimage


def euclidian_2D_distance(coord_set_1: tuple, coord_set_2: tuple) -> float:
    """
    This function calculates the euclidian distance between two sets of coordinates (2D)

    sqrt((x1 - x2)^2 + (y1 - y2)^2)

    Parameters
    ----------
    coord_set_1 : tuple
        The first set of coordinates (x, y)
    coord_set_2 : tuple
        The second set of coordinates (x, y)

    Returns
    -------
    float
        The euclidian distance between the two sets of coordinates
    """
    return np.sqrt(
        (coord_set_1[0] - coord_set_2[0]) ** 2 + (coord_set_1[1] - coord_set_2[1]) ** 2
    )


def check_coordinate_inside_box(
    coord: tuple,
    box: tuple,
) -> bool:
    """
    This function checks if a coordinate is inside a box

    Parameters
    ----------
    coord : tuple
        The coordinate to check (y, x)
    box : tuple
        The box to check against [y_min, x_min, y_max, x_max]

    Returns
    -------
    bool
        True if the coordinate is inside the box, False otherwise
    """
    # check if coords and box are valid
    if not isinstance(coord, tuple):
        raise TypeError("coord must be a tuple")
    if not isinstance(box, tuple):
        raise TypeError("box must be a list")
    if not len(box) == 4:
        raise ValueError("box must be a list of length 4")
    if not len(coord) == 2:
        raise ValueError("coord must be a tuple of length 2")

    y_coord = coord[0]
    x_coord = coord[1]

    y_min = box[0]
    x_min = box[1]
    y_max = box[2]
    x_max = box[3]

    if x_coord >= x_min and x_coord <= x_max and y_coord >= y_min and y_coord <= y_max:
        return True
    else:
        return False


def get_larger_bbox(bbox1: tuple, bbox2: tuple) -> tuple:
    """
    This function returns the larger of two bounding boxes

    Parameters
    ----------
    bbox1 : tuple
        The first bounding box [y_min, x_min, y_max, x_max]
    bbox2 : tuple
        The second bounding box [y_min, x_min, y_max, x_max]

    Returns
    -------
    tuple
        A tuple of the larger bounding box [y_min, x_min, y_max, x_max]
    """
    # check if boxes are valid
    if not isinstance(bbox1, tuple):
        raise TypeError("bbox1 must be a tuple")
    if not isinstance(bbox2, tuple):
        raise TypeError("bbox2 must be a tuple")
    if not len(bbox1) == 4:
        raise ValueError("bbox1 must be a list of length 4")
    if not len(bbox2) == 4:
        raise ValueError("bbox2 must be a list of length 4")

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if bbox1_area >= bbox2_area:
        return bbox1
    elif bbox2_area >= bbox1_area:
        return bbox2


def extract_unique_masks(image_stack) -> pd.DataFrame:
    """
    This function extracts unique masks from an image stack

    Parameters
    ----------
    image_stack : np.ndarray
        The image stack to extract unique masks from

    Returns
    -------
    pd.DataFrame
        The dataframe containing the unique masks
    """
    mask_indices_dict = {
        "unique_mask": [],
        "mask_indices": [],
        "pseudo_slice": [],
        "y_x_coords": [],
        "bbox": [],
        "merged/lone": [],
    }
    # find each unqiue mask identity via pixel value
    unique_masks = np.unique(image_stack)
    # loop through each unique mask identity
    # drop the backgound as a mask identity
    unique_masks = unique_masks[unique_masks != 0]
    for unique_mask in unique_masks:
        # loop through each mask image
        for pseudo_slice in range(len(image_stack)):
            # find where the unique mask identity is in the mask image
            tmp_image = image_stack[pseudo_slice]
            # get only the unique mask
            image_for_coordinates = np.array(tmp_image == unique_mask, dtype=np.uint8)
            mask_indices = np.where(tmp_image == unique_mask)
            # if the mask identity is in the mask image
            mask_indices_dict["unique_mask"].append(unique_mask)
            mask_indices_dict["mask_indices"].append(mask_indices)
            mask_indices_dict["pseudo_slice"].append(pseudo_slice)
            c = skimage.measure.regionprops(image_for_coordinates)
            for property in c:
                centroid = property.centroid
                bbox = property.bbox
                mask_indices_dict["y_x_coords"].append(centroid)
                mask_indices_dict["bbox"].append(bbox)
            if len(mask_indices_dict["y_x_coords"]) < len(
                mask_indices_dict["mask_indices"]
            ):
                mask_indices_dict["y_x_coords"].append(None)
                mask_indices_dict["bbox"].append(None)
            # check if the mask is merged or lone
            mask_indices_dict["merged/lone"].append(None)

    unique_masks_df = pd.DataFrame(mask_indices_dict)
    return unique_masks_df


def compare_masks_for_merged(
    df: pd.DataFrame, index1: int, index2: int, distance_threshold: int = 10
) -> pd.DataFrame:
    """
    This function compares masks for merging

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the masks
    index1 : int
        Index 1
    index2 : int
        Index 2
    distance_threshold : int, optional
        The distance threshold, by default 10

    Returns
    -------
    pd.DataFrame
        The dataframe containing the masks for merging
    """
    output_dict = {
        "unique_mask": [],
        "mask_indices": [],
        "pseudo_slice": [],
        "y_x_coords": [],
        "bbox": [],
        "merged/lone": [],
        "area": [],
        "index_comparison": [],
    }
    number_of_pseudo_slices = len(df["pseudo_slice"].unique())
    # get the x_y coordinates and bbox of the other index
    current_xy = df["y_x_coords"][index1]
    current_bbox = df["bbox"][index1]
    other_xy = df["y_x_coords"][index2]
    other_bbox = df["bbox"][index2]
    if not number_of_pseudo_slices < 2:
        # if the x_y coordinates are not None
        if other_xy is not None and current_xy is not None:
            distance = euclidian_2D_distance(current_xy, other_xy)
            if distance < distance_threshold:
                # check which bounding box has the larger area
                if get_larger_bbox(current_bbox, other_bbox) == current_bbox:
                    # check if the x_y coordinates are within the bbox of the other coordinates
                    coordinate_inside_box = check_coordinate_inside_box(
                        other_xy, current_bbox
                    )
                    if coordinate_inside_box:
                        output_dict["unique_mask"].append(index1)
                        output_dict["mask_indices"].append(df["mask_indices"][index1])
                elif get_larger_bbox(current_bbox, other_bbox) == other_bbox:
                    coordinate_inside_box = check_coordinate_inside_box(
                        current_xy, other_bbox
                    )
                    if coordinate_inside_box:
                        output_dict["unique_mask"].append(index2)
                        output_dict["mask_indices"].append(df["mask_indices"][index2])
                else:
                    coordinate_inside_box = False
                # check if the x_y coordinates are within the bbox of the other coordinates
                if coordinate_inside_box:
                    # add the indices to the output dict
                    output_dict["pseudo_slice"].append(df["pseudo_slice"][index1])
                    output_dict["y_x_coords"].append(other_xy)
                    output_dict["bbox"].append(other_bbox)
                    mask_array = df["mask_indices"][index2]

                    # get the area of the mask
                    mask_array = np.array(mask_array)
                    area = mask_array.shape[0] * mask_array.shape[1]
                    output_dict["area"].append(area)
                    output_dict["index_comparison"].append(f"{index1},{index2}")
                    output_dict["merged/lone"].append("merged")
    else:
        output_dict["unique_mask"].append(index1)
        output_dict["mask_indices"].append(df["mask_indices"][index1])
        output_dict["pseudo_slice"].append(df["pseudo_slice"][index1])
        output_dict["y_x_coords"].append(current_xy)
        output_dict["bbox"].append(current_bbox)
        output_dict["area"].append(1)
        output_dict["index_comparison"].append(f"{index1},{index2}")
        output_dict["merged/lone"].append("lone")

    new_df = pd.DataFrame(output_dict)
    return new_df


# get each combination of indices
def get_combinations_of_indices(
    df: pd.DataFrame, distance_threshold: int = 10
) -> pd.DataFrame:
    """
    This function gets the combinations of indices

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the masks
    distance_threshold : int, optional
        The distance threshold, by default 10

    Returns
    -------
    pd.DataFrame
        The dataframe containing the combinations of indices
    """
    original_shape = df.shape[0]
    if original_shape < 2:
        return df
    list_of_dfs = []
    for index1 in range(len(df)):
        for index2 in range(1 + index1, len(df)):
            new_df = compare_masks_for_merged(df, index1, index2, distance_threshold=10)
            list_of_dfs.append(new_df)
    dfs = pd.concat(list_of_dfs).reset_index(drop=True)

    # get only the merged masks
    if dfs["pseudo_slice"].nunique() < 2:
        merged_df = dfs
    elif not dfs.shape[0] < 2:
        merged_df = dfs[dfs["merged/lone"] == "merged"]
    else:
        merged_df = dfs
    if merged_df.shape[0] < 2:
        return merged_df
    merged_df = merged_df.assign(
        index1=merged_df["index_comparison"].str.split(",", expand=True)[0].astype(int),
        index2=merged_df["index_comparison"].str.split(",", expand=True)[1].astype(int),
    )
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


# if 0 merges with 1 and 0 merges with 2, then 1 and 2 are merged
def merge_sets(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function merges the sets of masks

    Parameters
    ----------
    merged_df : pd.DataFrame
        The dataframe containing the masks

    Returns
    -------
    pd.DataFrame
        The dataframe containing the merged masks
    """

    if merged_df.shape[0] < 2:
        merged_df["label"] = 1
        return merged_df
    index_sets = merged_df["index_comparison"]
    # convert to list of sets
    list_of_sets = [set(map(int, x.split(","))) for x in index_sets]
    for i in list_of_sets:
        for j in list_of_sets:
            if i != j and len(i.intersection(j)) > 0:
                i.update(j)
                list_of_sets.remove(j)
    merged_sets_dict = {}
    for i in range(len(list_of_sets)):
        merged_sets_dict[i] = list_of_sets[i]

    for row in merged_df.iterrows():
        for num_set in merged_sets_dict:
            if int(row[1]["index1"]) in merged_sets_dict[num_set]:
                merged_df.at[row[0], "label"] = num_set
    list_of_dfs = []
    for unique_label in merged_df["label"].unique():
        tmp_df = merged_df[merged_df["label"] == unique_label]
        # keep one row that has the largest area
        largest_area_index = tmp_df["area"].idxmax()
        tmp_df = tmp_df.loc[largest_area_index]
        list_of_dfs.append(tmp_df)
    merged_df = pd.DataFrame(list_of_dfs)

    return merged_df


def reassemble_each_mask(df, original_img_shape) -> np.ndarray:
    """
    This function reassembles the masks from the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the masks
    original_img_shape : tuple
        The shape of the original image

    Returns
    -------
    np.ndarray
        The reassembled masks
    """
    # get the dimensions of the original image
    # make an empty array to hold the reassembled masks

    dict_of_masks = {}
    # get all unique group indices
    label = np.unique(df["label"])
    for index in label:
        tmp_df = df.loc[df["label"] == index]
        dict_of_masks[index] = tmp_df["mask_indices"].values
    reassembled_masks = np.zeros(
        (original_img_shape[1], original_img_shape[2]), dtype=np.uint8
    )
    for index, mask in dict_of_masks.items():
        # set the pixels in the reassembled masks to the index value
        for m in mask:
            reassembled_masks[m] = index + 1
    return reassembled_masks


def get_dimensionality(image_array: np.ndarray) -> int:
    """
    This function returns the dimensionality of an image array while checking if the input is a numpy array

    Parameters
    ----------
    image_array : np.ndarray
        The image array to check the dimensionality of

    Returns
    -------
    int
        The dimensionality of the image array

    Raises
    ------
    TypeError
        If the input is not a numpy array
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy array")
    return len(image_array.shape)


def get_number_of_unique_labels(image_array: np.ndarray) -> int:
    """
    This function returns the number of unique labels in an image array

    Parameters
    ----------
    image_array : np.ndarray
        The image array to check the number of unique labels

    Returns
    -------
    int
        The number of unique labels in the image array
    """
    get_dimensionality(image_array)
    return len(np.unique(image_array))
